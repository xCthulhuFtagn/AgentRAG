"""Vector search tools — only interface to LanceDB.

LangChain @tool wrappers for use with bind_tools() in the Search Fanout agent.
"""

from langchain_core.tools import tool

from src.vectordb.config import vdb_settings
from src.vectordb.embeddings import embed, _get_embedding_model_name, _get_embedding_dim
from src.vectordb.client import get_async_db
from src.vectordb.descriptions import load_descriptions, validate_embedding_model


def _rrf_merge(
    vec_results: list[dict], fts_results: list[dict], top_k: int, k: int = 60
) -> list[dict]:
    """Reciprocal Rank Fusion of a vector-KNN and a full-text ranked list.

    Vector similarity misses exact terms, names and abbreviations — precisely
    what a rephrased query is trying to recover; BM25-style full-text search
    misses paraphrases and synonyms. RRF combines the two RANKINGS (score =
    sum of 1/(k + rank)) rather than comparing their raw scores, which live on
    incomparable scales (L2 distance vs. a BM25-like relevance score). k=60 is
    the standard damping constant so no single top rank dominates the fusion.

    Dedup key: `seq` when present (both lists come from the same table, so seq
    is a stable row identity), else the chunk's text (legacy tables without
    seq). When a row appears in both lists, the vector-side row is kept (it
    carries `_distance`, used for display/removal bookkeeping downstream).
    """
    def key(r):
        seq = r.get("seq")
        return ("seq", seq) if seq is not None else ("text", r.get("text", ""))

    scores: dict = {}
    rows: dict = {}
    for source in (vec_results, fts_results):
        for rank, r in enumerate(source):
            k_ = key(r)
            scores[k_] = scores.get(k_, 0.0) + 1.0 / (k + rank + 1)
            rows.setdefault(k_, r)

    ordered = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    return [rows[k_] for k_ in ordered[:top_k]]


@tool
async def vector_search(
    query: str,
    collection: str,
    top_k: int = vdb_settings.search_top_k,
    db_path: str | None = None,
    hybrid: bool = False,
) -> dict:
    """Search a LanceDB collection by vector similarity (optionally hybrid).

    Use this tool to find relevant text chunks from a specific document collection.
    Choose the right collection based on the Planner's route.

    Args:
        query: The search query text (will be embedded, and used verbatim for
            full-text search when hybrid=True).
        collection: Name of the LanceDB table/collection to search.
        top_k: Number of top results to return (default 5).
        db_path: Optional LanceDB path to scope the search (per-project isolation).
            None → global LANCE_DB_PATH.
        hybrid: When True, fuse the vector KNN results with a full-text search
            over the same query (Reciprocal Rank Fusion) — recovers exact-term
            matches (names, abbreviations) that pure embedding similarity can
            miss. Silently falls back to vector-only if the table has no
            full-text index (legacy tables indexed before this feature, or a
            failed index build).

    Returns:
        Dict with keys: collection, query, chunks (list of text), scores (list of distances).
    """
    # Catch embedding model mismatches before producing garbage distances.
    validate_embedding_model(
        db_path,
        expected_model=_get_embedding_model_name(),
        expected_dim=_get_embedding_dim(),
    )
    db = await get_async_db(db_path)
    try:
        table = await db.open_table(collection)
    except Exception:
        return {
            "collection": collection,
            "query": query,
            "chunks": [],
            "scores": [],
            "error": f"Collection '{collection}' not found",
        }

    query_embedding = await embed(query)
    # Async LanceDB: search() is a coroutine — await it before chaining.
    search_query = await table.search(query_embedding)
    results = await search_query.limit(top_k).to_list()

    if hybrid:
        try:
            fts_results = (
                await table.query()
                .nearest_to_text(query)
                .select(["text", "seq", "_score"])
                .limit(top_k)
                .to_list()
            )
        except Exception:
            # No FTS index on this table (legacy, or the index build failed
            # at index time) — fall back to vector-only, same as hybrid=False.
            fts_results = []
        if fts_results:
            results = _rrf_merge(results, fts_results, top_k)

    return {
        "collection": collection,
        "query": query,
        "chunks": [r.get("text", "") for r in results],
        "scores": [r.get("_distance", 0.0) for r in results],
        # seq = chunk position in the document; None for legacy tables indexed
        # before the seq column existed (neighbor stitching then no-ops).
        "seqs": [r.get("seq") for r in results],
    }


def _merge_windows(
    seqs: list[int], padding: int, bridge_gap: int
) -> list[tuple[int, int]]:
    """Turn hit seqs into merged, document-ordered [lo, hi] ranges.

    Each seq → window [seq-padding, seq+padding] (lower-clamped to 0). Two
    windows merge when the uncovered gap between them is <= bridge_gap, so a
    single weak chunk that fell out of top-k gets bridged, while far-apart hits
    stay as separate neighborhoods. Effective merge distance = 2*P + gap + 1.
    """
    uniq = sorted({s for s in seqs if s is not None})
    if not uniq:
        return []
    windows = [(max(0, s - padding), s + padding) for s in uniq]
    merged: list[list[int]] = [list(windows[0])]
    for lo, hi in windows[1:]:
        if lo - merged[-1][1] - 1 <= bridge_gap:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _strip_overlap(prev: str, cur: str, max_overlap: int) -> str:
    """Drop cur's leading text that duplicates prev's tail (chunk overlap).

    Consecutive chunks share ~CHUNK_OVERLAP chars by construction, but the real
    overlap drifts (strip/clean_text) and isn't a fixed length — so we find the
    longest suffix of prev that is a prefix of cur and cut it, rather than
    blindly removing CHUNK_OVERLAP chars. Search is bounded to max_overlap for
    speed and to avoid over-matching coincidental repetition.
    """
    window = min(len(prev), len(cur), max_overlap)
    for k in range(window, 0, -1):
        if prev[-k:] == cur[:k]:
            return cur[k:].lstrip()
    return cur


async def gather_neighbors(
    collection: str,
    hit_seqs: list[int],
    db_path: str | None = None,
    padding: int = vdb_settings.expand_padding,
    bridge_gap: int = vdb_settings.bridge_gap,
    cap: int = vdb_settings.max_expanded,
) -> list[dict]:
    """Expand vector hits into their contiguous seq-neighborhoods (one collection).

    Fetches every chunk in the merged ranges by seq filter-scan (no vector,
    vectors not selected), returns [{seq, text}] in document order, capped at
    `cap`. Empty if the table has no seq column (legacy index) → caller falls
    back to the raw KNN chunks.
    """
    ranges = _merge_windows(hit_seqs, padding, bridge_gap)
    if not ranges:
        return []

    db = await get_async_db(db_path)
    table = await db.open_table(collection)

    by_seq: dict[int, str] = {}
    for lo, hi in ranges:
        rows = (
            await table.query()
            .where(f"seq >= {lo} AND seq <= {hi}")
            .select(["seq", "text"])
            .limit(hi - lo + 1)
            .to_list()
        )
        for r in rows:
            by_seq[r["seq"]] = r.get("text", "")

    result = [{"seq": s, "text": by_seq[s]} for s in sorted(by_seq)][:cap]

    # De-overlap consecutive chunks: a stitched run shares the chunk overlap at
    # each boundary, so strip the duplicated prefix. Only between truly adjacent
    # seqs (within a merged range) — never across a gap. prev's tail is intact
    # even if prev was itself front-stripped, so comparing against it is correct.
    max_ol = 2 * vdb_settings.chunk_overlap
    for i in range(1, len(result)):
        if result[i]["seq"] == result[i - 1]["seq"] + 1:
            result[i]["text"] = _strip_overlap(
                result[i - 1]["text"], result[i]["text"], max_ol
            )
    return result


async def fetch_all_chunks(collection: str, db_path: str | None = None) -> list[dict]:
    """Every stored chunk of one collection in document order — [{seq, text}].

    Powers the web UI's parsed-text preview: reads the table content directly,
    so what is shown is exactly what search sees (extracted → cleaned → chunked
    at index time), not a re-parse of the source file. Legacy tables without a
    `seq` column return rows in storage order with seq=None. Raises if the
    collection does not exist (the caller decides how to report that).
    """
    db = await get_async_db(db_path)
    table = await db.open_table(collection)
    total = await table.count_rows()
    if not total:
        return []
    try:
        raw = await table.query().select(["seq", "text"]).limit(total).to_list()
    except Exception:  # legacy table without the seq column
        raw = await table.query().select(["text"]).limit(total).to_list()
    rows = [{"seq": r.get("seq"), "text": r.get("text", "")} for r in raw]
    if all(r["seq"] is not None for r in rows):
        rows.sort(key=lambda r: r["seq"])
    return rows


def merge_chunk_texts(
    texts: list[str], chunk_overlap: int = vdb_settings.chunk_overlap
) -> str:
    """Reassemble consecutive stored chunks into one continuous text.

    Chunks share ~chunk_overlap chars at each boundary by construction; strip
    the duplicated prefix (same drift-aware suffix/prefix match as the
    stitching path) so a full-document read renders like the document, not
    like overlapping windows. The search window is the overlap the chunks
    were cut with, EXACTLY — not the 2x margin the stitching path uses:
    stored chunks are stripped, so the true duplicated prefix never exceeds
    chunk_overlap, while a wider window lets a repeated-character run at the
    boundary (form blanks '____', rules '----', dot leaders) match past the
    real overlap and silently swallow genuine content. Chunks were stripped
    at index time, so boundaries re-join with a newline (the original
    boundary whitespace is not recoverable).
    """
    if not texts:
        return ""
    parts = [texts[0]]
    for prev, cur in zip(texts, texts[1:]):
        parts.append(_strip_overlap(prev, cur, chunk_overlap))
    return "\n".join(p for p in parts if p)


async def count_chunks(collection: str, db_path: str | None = None) -> int | None:
    """Total number of chunks (rows) in one collection, or None if unreadable.

    Mechanical input for the Planner's coverage statistic («извлечено K/N
    чанков») — computed by code, never reconstructed by the model. None (table
    missing/corrupt) simply omits the coverage figure rather than failing the
    node.
    """
    db = await get_async_db(db_path)
    try:
        table = await db.open_table(collection)
        return await table.count_rows()
    except Exception:
        return None


async def _list_table_names(db_path: str | None = None) -> list[str]:
    """Walk LanceDB's paginated table listing into a flat list of names."""
    db = await get_async_db(db_path)
    # Async LanceDB: list_tables() (table_names() is deprecated) returns a
    # paginated ListTablesResponse — walk page_token to collect every name.
    names: list[str] = []
    page_token = None
    while True:
        resp = await db.list_tables(page_token=page_token)
        names.extend(resp.tables)
        page_token = resp.page_token
        if not page_token:
            break
    return names


@tool
async def list_collections(db_path: str | None = None) -> list[str]:
    """List all available LanceDB collections (tables).

    Use this to discover what document sources are available before searching.
    The Planner should call this first to decide where to route queries.

    Args:
        db_path: Optional LanceDB path to scope the listing (per-project isolation).
            None → global LANCE_DB_PATH.

    Returns:
        List of collection/table names.
    """
    return await _list_table_names(db_path)


async def list_collections_described(db_path: str | None = None) -> list[dict]:
    """List collections paired with their per-file content description and language.

    Returns [{collection, description, language}] — the description (from the
    sidecar written at index time) helps the Planner pick a source from a summary,
    not just the table name; `language` is the ISO 639-1 code (default "ru" for
    legacy sidecars). Empty description for tables indexed before the feature.
    """
    validate_embedding_model(
        db_path,
        expected_model=_get_embedding_model_name(),
        expected_dim=_get_embedding_dim(),
    )
    names = await _list_table_names(db_path)
    descs = load_descriptions(db_path)
    return [
        {
            "collection": n,
            "description": descs.get(n, {}).get("description", ""),
            "language": descs.get(n, {}).get("language", "ru") or "ru",
        }
        for n in names
    ]
