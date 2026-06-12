"""Vector search tools — only interface to LanceDB.

LangChain @tool wrappers for use with bind_tools() in the Search Fanout agent.
"""

from langchain_core.tools import tool

from src.vectordb.config import vdb_settings
from src.vectordb.embeddings import embed
from src.vectordb.client import get_async_db
from src.vectordb.descriptions import load_descriptions


@tool
async def vector_search(
    query: str,
    collection: str,
    top_k: int = vdb_settings.search_top_k,
    db_path: str | None = None,
) -> dict:
    """Search a LanceDB collection by vector similarity.

    Use this tool to find relevant text chunks from a specific document collection.
    Choose the right collection based on the Planner's route.

    Args:
        query: The search query text (will be embedded).
        collection: Name of the LanceDB table/collection to search.
        top_k: Number of top results to return (default 5).
        db_path: Optional LanceDB path to scope the search (per-project isolation).
            None → global LANCE_DB_PATH.

    Returns:
        Dict with keys: collection, query, chunks (list of text), scores (list of distances).
    """
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
    """List collections paired with their per-file content description.

    Returns [{collection, description}] — the description (from the sidecar
    written at index time) helps the Planner pick a source from a summary, not
    just the table name. Empty description for tables indexed before the feature.
    """
    names = await _list_table_names(db_path)
    descs = load_descriptions(db_path)
    return [
        {"collection": n, "description": descs.get(n, {}).get("description", "")}
        for n in names
    ]
