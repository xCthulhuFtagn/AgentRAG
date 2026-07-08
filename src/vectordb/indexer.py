"""Document indexing — loads documents into LanceDB with FastEmbed embeddings.

Text extraction is hybrid:
- Rich documents (PDF/DOCX/PPTX) → LiteParse (Rust, in-process, OCR-capable).
  OCR uses built-in Tesseract by default; set OCR_SERVER_URL to delegate to a
  local EasyOCR/PaddleOCR sidecar (better Cyrillic, no Tesseract noise).
- Plain text (TXT/MD) → direct read (no parser needed).

Usage:
    python -m src.vectordb.indexer --dir docs/sample_docs
    python -m src.vectordb.indexer --dir docs/sample_docs --db ./data/lancedb/_cli
"""

import argparse
import asyncio
import hashlib
import re
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from liteparse import LiteParse

from src.vectordb.config import vdb_settings
from src.vectordb.embeddings import (
    embed_batch,
    _get_embedding_model_name,
    _get_embedding_dim,
)
from src.vectordb.client import get_sync_db
from src.vectordb.describe import describe_document
from src.vectordb.descriptions import (
    load_descriptions,
    save_descriptions,
    language_for_collection,
)

# ISO 639-1 → LanceDB FTS language name. Anything not in this map falls back
# to "Russian" (the corpus default — most of our documents are Russian).
_ISO_LANG_TO_FTS: dict[str, str] = {
    "ru": "Russian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "cs": "Czech",
    "sk": "Slovak",
    "sl": "Slovenian",
    "hr": "Croatian",
    "sr": "Serbian",
    "mk": "Macedonian",
    "da": "Danish",
    "sv": "Swedish",
    "no": "Norwegian",
    "fi": "Finnish",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "el": "Greek",
    "tr": "Turkish",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
    "hi": "Hindi",
    "bn": "Bengali",
    "th": "Thai",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ro": "Romanian",
    "hu": "Hungarian",
    "ca": "Catalan",
    "eu": "Basque",
    "gl": "Galician",
    "cy": "Welsh",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "mt": "Maltese",
    "is": "Icelandic",
    "sq": "Albanian",
    "hy": "Armenian",
    "ka": "Georgian",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "tg": "Tajik",
    "tk": "Turkmen",
    "uz": "Uzbek",
    "mn": "Mongolian",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Tagalog",
    "sw": "Swahili",
    "af": "Afrikaans",
    "am": "Amharic",
    "ur": "Urdu",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "si": "Sinhala",
    "km": "Khmer",
    "lo": "Lao",
    "my": "Burmese",
    "ne": "Nepali",
    "ps": "Pashto",
}


def _fts_language(iso_lang: str) -> str:
    """Map an ISO 639-1 code to a LanceDB FTS language name; fallback "Russian"."""
    return _ISO_LANG_TO_FTS.get(iso_lang, "Russian")


# Rich document formats parsed by LiteParse.
DOC_SUFFIXES = {".pdf", ".docx", ".pptx"}
# Plain-text formats read directly.
TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_SUFFIXES = DOC_SUFFIXES | TEXT_SUFFIXES

# Cyrillic → Latin (RU/UK) so non-ASCII filenames stay readable as table names.
_CYRILLIC_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "і": "i", "ї": "yi", "є": "ye", "ґ": "g",
}


def _transliterate(s: str) -> str:
    """Map Cyrillic letters to Latin; leave everything else untouched."""
    out = []
    for ch in s:
        mapped = _CYRILLIC_MAP.get(ch.lower())
        if mapped is None:
            out.append(ch)
        elif ch.isupper() and mapped:
            out.append(mapped.capitalize())
        else:
            out.append(mapped)
    return "".join(out)


def safe_table_name(stem: str) -> str:
    """Turn a file stem into a LanceDB-valid table name.

    LanceDB allows only [A-Za-z0-9._-]. We transliterate Cyrillic (to keep the
    name readable for the Planner), replace any remaining disallowed character
    with '_', and fall back to a hashed name if nothing usable remains.
    """
    name = _transliterate(stem)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    if not name or not any(c.isalnum() for c in name):
        digest = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
        name = f"doc_{digest}"
    return name


@lru_cache(maxsize=1)
def _get_parser() -> LiteParse:
    """Cached LiteParse instance (spawns OCR workers lazily).

    When `OCR_SERVER_URL` is set, OCR is delegated to a local HTTP OCR sidecar
    (EasyOCR/PaddleOCR) instead of the built-in Tesseract — better Cyrillic and
    no 'Image too small to scale!!' native noise. Unset → built-in Tesseract.
    Passing None for either arg is a no-op (LiteParse keeps its defaults).
    """
    return LiteParse(
        quiet=True,
        ocr_server_url=vdb_settings.ocr_server_url,
        ocr_language=vdb_settings.ocr_language,
    )


def extract_text(file_path: Path) -> str:
    """Extract text from a supported file.

    PDF/DOCX/PPTX via LiteParse; TXT/MD read directly.
    """
    ext = file_path.suffix.lower()
    if ext in TEXT_SUFFIXES:
        return file_path.read_text(encoding="utf-8")
    if ext in DOC_SUFFIXES:
        return _get_parser().parse(str(file_path)).text
    raise ValueError(f"Unsupported file type: {ext}")


def clean_text(text: str) -> str:
    """Normalize extracted text before chunking.

    Parsers (esp. PDF) emit ragged whitespace: per-line indentation, runs of
    blank lines, double spaces. Collapse them so chunk boundaries land on real
    paragraph/sentence breaks instead of inside the noise.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)   # 3+ newlines → one paragraph break
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse runs of spaces/tabs
    return text.strip()


def split_text(
    text: str,
    chunk_size: int = vdb_settings.chunk_size,
    overlap: int = vdb_settings.chunk_overlap,
) -> list[str]:
    """Boundary-aware chunking — never cuts mid-word/sentence.

    Cleans the text, then splits recursively on paragraph → line → sentence →
    word boundaries (RecursiveCharacterTextSplitter), keeping ~chunk_size chars
    with `overlap` carried between chunks for context continuity.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
        keep_separator=True,
    )
    return [c.strip() for c in splitter.split_text(clean_text(text)) if c.strip()]


def _sync_table_names(db) -> list[str]:
    """Walk sync LanceDB's paginated table listing into a flat list of names.

    Mirrors `_list_table_names` in tools.py (async side): list_tables() —
    table_names() is deprecated — returns a paginated ListTablesResponse.
    """
    names: list[str] = []
    page_token = None
    while True:
        resp = db.list_tables(page_token=page_token)
        names.extend(resp.tables)
        page_token = resp.page_token
        if not page_token:
            break
    return names


# The subset of project settings that shapes the stored vectors — changing any
# of these invalidates existing chunks, so applying them requires a full
# reindex. The stitching keys below are search-time and apply on the next query.
INDEX_TIME_KEYS = (
    "chunk_size", "chunk_overlap", "descriptions_enabled", "describe_max_chars"
)


def resolve_index_settings(overrides: dict | None = None) -> dict:
    """Per-corpus vector-DB hyperparameters with overrides applied.

    The knobs a caller may tune per corpus (the web UI stores them per
    project): four index-time ones (INDEX_TIME_KEYS — consumed by
    `index_files`) plus the search-time ones — `search_top_k` (nearest chunks
    fetched per search before stitching), the two neighbor-stitching ones
    (`expand_padding`, `bridge_gap` — threaded into `gather_neighbors` via
    graph state), `reranking_enabled` / `reranking_remove_irrelevant` (LLM
    per-chunk relevance assessment in search_fanout), `hybrid_search_enabled`
    (BM25 fusion in vector_search — the underlying FTS index is built at index
    time regardless, see index_files) and `max_iterations` (iteration budget
    for the planner→judge loop). Any key absent or None falls back to the
    global `vdb_settings` (.env) value — so the .env values are the defaults.
    """
    resolved = {
        "chunk_size": vdb_settings.chunk_size,
        "chunk_overlap": vdb_settings.chunk_overlap,
        "descriptions_enabled": vdb_settings.descriptions_enabled,
        "describe_max_chars": vdb_settings.describe_max_chars,
        "search_top_k": vdb_settings.search_top_k,
        "expand_padding": vdb_settings.expand_padding,
        "bridge_gap": vdb_settings.bridge_gap,
        "reranking_enabled": vdb_settings.reranking_enabled,
        "reranking_remove_irrelevant": vdb_settings.reranking_remove_irrelevant,
        "hybrid_search_enabled": vdb_settings.hybrid_search_enabled,
        "max_iterations": vdb_settings.max_iterations,
    }
    for key, value in (overrides or {}).items():
        if key in resolved and value is not None:
            resolved[key] = value
    return resolved


async def index_documents(
    docs_dir: str,
    db_path: str = vdb_settings.lance_db_path,
    progress_cb: Callable[[str, bool], None] | None = None,
    settings: dict | None = None,
):
    """Index all documents from a directory into LanceDB (a from-scratch run).

    Each file becomes a separate LanceDB collection (table). `progress_cb`, if
    given, is called `(filename, ok)` once each file finishes — ok=False when it
    was skipped (extraction error / no text) — letting a UI show per-file
    progress and flag failures.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Error: directory '{docs_dir}' does not exist")
        sys.exit(1)

    files = [f for f in docs_path.glob("*.*") if f.suffix.lower() in SUPPORTED_SUFFIXES]

    if not files:
        print(f"No supported files found in {docs_dir}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        return

    print(f"Found {len(files)} file(s) to index\n")
    await index_files(files, db_path, progress_cb, fresh=True, settings=settings)


async def index_files(
    files: list[Path],
    db_path: str = vdb_settings.lance_db_path,
    progress_cb: Callable[[str, bool], None] | None = None,
    *,
    fresh: bool = False,
    settings: dict | None = None,
):
    """Index the given files into LanceDB, one table per file.

    fresh=True — a from-scratch run: starts an empty descriptions sidecar and
    assumes the caller owns the whole DB (full reindex / CLI). fresh=False —
    incremental: tables of files NOT in `files` stay untouched; each given
    file's previous table (located via the sidecar's file→table mapping, with
    a `safe_table_name` fallback for sidecar-less legacy tables) is dropped
    up front so the index always reflects the file's current content, and the
    sidecar is merged, not overwritten.
    """
    cfg = resolve_index_settings(settings)
    db = get_sync_db(db_path)

    if fresh:
        used: set[str] = set()
        descriptions: dict[str, dict] = {}  # table_name -> {file, description}
    else:
        descriptions = load_descriptions(db_path)
        used = set(await asyncio.to_thread(_sync_table_names, db))
        for file_path in files:
            old = table_for_file(file_path.name, descriptions=descriptions)
            if old is not None and (old in used or old in descriptions):
                await asyncio.to_thread(db.drop_table, old, ignore_missing=True)
                used.discard(old)
                descriptions.pop(old, None)

    # Table names are assigned sequentially first: the collision-disambiguation
    # check depends on the `used` set built up incrementally over this run, so
    # it can't be parallelized. Only the actually expensive per-file work below
    # (extraction, embedding, description, table write) runs concurrently.
    assignments: list[tuple[Path, str]] = []
    for file_path in files:
        base = safe_table_name(file_path.stem)
        table_name = base
        if table_name in used:
            # Two different files sanitized to the same name — disambiguate.
            digest = hashlib.md5(file_path.name.encode("utf-8")).hexdigest()[:6]
            table_name = f"{base}_{digest}"
        used.add(table_name)
        assignments.append((file_path, table_name))

    semaphore = asyncio.Semaphore(vdb_settings.index_concurrency)

    async def _bounded(file_path: Path, table_name: str):
        async with semaphore:
            return await _index_one_file(file_path, table_name, db_path, cfg, progress_cb)

    results = await asyncio.gather(*[_bounded(fp, tn) for fp, tn in assignments])
    for table_name, entry in results:
        if entry is not None:
            descriptions[table_name] = entry

    # Persist per-file descriptions next to the DB for the Planner to read.
    # Stamp the embedding model metadata so a later provider change is caught
    # at search time (mismatched vectors → clear error, not garbage distances).
    save_descriptions(
        db_path,
        descriptions,
        embedding_model=_get_embedding_model_name(),
        embedding_dim=_get_embedding_dim(),
    )
    print(f"\nIndexing complete. DB at: {db_path}")


async def _index_one_file(
    file_path: Path,
    table_name: str,
    db_path: str,
    cfg: dict,
    progress_cb: Callable[[str, bool], None] | None,
) -> tuple[str, dict | None]:
    """Extract, chunk, embed and store one file into its own table.

    Runs concurrently with other files (bounded by index_concurrency, see
    index_files) — opens its OWN sync LanceDB connection rather than sharing
    the caller's, since get_sync_db is a cheap handle (not cached) and several
    of these connections may be doing disk writes to DIFFERENT tables at once
    from different threads (asyncio.to_thread).

    Returns (table_name, description_entry); description_entry is None when
    the file was skipped (extraction failed / no text extracted) — the caller
    then records no sidecar entry for it.
    """
    print(f"  Indexing: {file_path.name} → table '{table_name}'")

    # Text extraction (LiteParse/OCR) and chunking are CPU-heavy and fully
    # synchronous — run them off the event loop so the caller's UI stays
    # responsive (the web reindex awaits this on the same loop as NiceGUI).
    try:
        text = await asyncio.to_thread(extract_text, file_path)
    except Exception as e:
        print(f"    Error extracting text ({file_path.name}): {e}")
        if progress_cb:
            progress_cb(file_path.name, False)  # failed → flagged in UI
        return table_name, None

    if not text.strip():
        print(f"    Warning: no text extracted ({file_path.name})")
        if progress_cb:
            progress_cb(file_path.name, False)
        return table_name, None

    chunks = await asyncio.to_thread(
        split_text, text, cfg["chunk_size"], cfg["chunk_overlap"]
    )
    print(f"    {file_path.name}: {len(chunks)} chunks, embedding...")

    # Embedding and the (optional) LLM description are independent given the
    # text — run them concurrently so the description adds little wall-clock.
    if cfg["descriptions_enabled"]:
        embeddings, (description, language) = await asyncio.gather(
            embed_batch(chunks),
            describe_document(text, cfg["describe_max_chars"]),
        )
    else:
        embeddings = await embed_batch(chunks)
        description = ""
        language = "ru"

    # seq = chunk's position in the document — lets the retriever stitch
    # back contiguous neighborhoods (see gather_neighbors in tools.py).
    records = [
        {"text": chunk, "vector": emb, "seq": i}
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    db = get_sync_db(db_path)

    # Sync LanceDB disk writes — also off-loop.
    try:
        await asyncio.to_thread(db.drop_table, table_name, ignore_missing=True)
    except Exception:
        pass

    table = await asyncio.to_thread(db.create_table, table_name, data=records)

    # Build a native full-text index alongside the vector one, so hybrid
    # search (BM25 + vector, see tools.py) has something to query — cheap, no
    # LLM involved. Built unconditionally regardless of the project's current
    # hybrid_search_enabled toggle, since the toggle only gates whether search
    # USES it; a later "turn hybrid on" then needs no reindex. The stemmer
    # language now comes from the document's detected ISO 639-1 code (not
    # hardcoded "Russian" — that was breaking non-Russian documents). A failure
    # here must not break indexing (legacy behavior: plain vector search still
    # works without an FTS index).
    fts_lang = _fts_language(language)
    try:
        await asyncio.to_thread(
            table.create_fts_index,
            "text",
            use_tantivy=False,
            language=fts_lang,
            stem=True,
            remove_stop_words=True,
            replace=True,
        )
    except Exception as e:
        print(f"    Warning: could not build full-text index ({file_path.name}): {e}")

    print(f"    Done — {len(records)} vectors stored ({file_path.name})")
    if progress_cb:
        progress_cb(file_path.name, True)
    # chunk_overlap = the overlap these chunks were ACTUALLY cut with — read
    # back by the parsed-text preview so merging strips the historical value
    # even if the project's settings drift later (absent in legacy sidecars →
    # consumers fall back to the currently resolved setting).
    return table_name, {
        "file": file_path.name,
        "description": description,
        "language": language,
        "chunk_overlap": cfg["chunk_overlap"],
    }


def table_for_file(
    file_name: str,
    db_path: str | None = None,
    descriptions: dict | None = None,
    siblings: list[str] | None = None,
) -> str | None:
    """Table name backing one uploaded file, or None if unresolvable.

    Single resolution point for file → table (the viewer, `index_files` and
    `remove_files_from_index` all go through it). Ownership comes from the
    descriptions sidecar; a sidecar-less file falls back to
    `safe_table_name(stem)` — but only when the sidecar doesn't claim that
    table for a DIFFERENT file. Two files whose stems sanitize to the same
    base name (e.g. `file_copy.pdf` and `file (copy).pdf`) share a fallback
    name while only one of them owns the table — treating the fallback as
    "this file's table" made the incremental indexer silently drop the other
    file's data, and the viewer render the wrong document. `descriptions`
    lets a caller that already loaded (and is mutating) the sidecar pass it
    in instead of re-reading it from `db_path`. `siblings` (the corpus's
    other file names, when the caller knows them) extends the guard to
    pure-legacy tables the sidecar has never seen: if ANOTHER sidecar-less
    sibling sanitizes to the same fallback name, the table's owner is
    genuinely unknowable → None (honest refusal beats rendering the wrong
    document; a reindex writes the sidecar and resolves it).
    """
    if descriptions is None:
        descriptions = load_descriptions(db_path)
    for tbl, info in descriptions.items():
        if info.get("file") == file_name:
            return tbl
    fallback = safe_table_name(Path(file_name).stem)
    owner = descriptions.get(fallback, {}).get("file")
    if owner not in (None, file_name):
        return None
    mapped = {info.get("file") for info in descriptions.values()}
    for other in siblings or ():
        if (
            other != file_name
            and other not in mapped
            and safe_table_name(Path(other).stem) == fallback
        ):
            return None
    return fallback


async def remove_files_from_index(
    file_names: list[str],
    db_path: str = vdb_settings.lance_db_path,
) -> None:
    """Drop the tables (and sidecar entries) backing the given uploaded files.

    Table ownership comes from the descriptions sidecar (file → table); a
    sidecar-less legacy table falls back to `safe_table_name(stem)`. Missing
    tables are ignored — removing an unindexed file is a no-op.
    """
    if not file_names:
        return
    db = get_sync_db(db_path)
    descriptions = load_descriptions(db_path)
    for name in file_names:
        table = table_for_file(name, descriptions=descriptions)
        if table is None:  # fallback name owned by another file — nothing ours
            continue
        await asyncio.to_thread(db.drop_table, table, ignore_missing=True)
        descriptions.pop(table, None)
    save_descriptions(db_path, descriptions)


def main():
    parser = argparse.ArgumentParser(
        description="Index documents into LanceDB for Agentic RAG"
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing documents to index",
    )
    parser.add_argument(
        "--db",
        default=vdb_settings.lance_db_path,
        help=f"Path to LanceDB database (default: {vdb_settings.lance_db_path})",
    )
    args = parser.parse_args()

    asyncio.run(index_documents(args.dir, args.db))


if __name__ == "__main__":
    main()
