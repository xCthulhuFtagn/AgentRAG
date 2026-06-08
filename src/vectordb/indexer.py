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
from src.vectordb.embeddings import embed_batch
from src.vectordb.client import get_sync_db
from src.vectordb.describe import describe_document
from src.vectordb.descriptions import save_descriptions

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


async def index_documents(
    docs_dir: str,
    db_path: str = vdb_settings.lance_db_path,
    progress_cb: Callable[[str, bool], None] | None = None,
):
    """Index all documents from a directory into LanceDB.

    Each file becomes a separate LanceDB collection (table). `progress_cb`, if
    given, is called `(filename, ok)` once each file finishes — ok=False when it
    was skipped (extraction error / no text) — letting a UI show per-file
    progress and flag failures.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Error: directory '{docs_dir}' does not exist")
        sys.exit(1)

    db = get_sync_db(db_path)
    files = [f for f in docs_path.glob("*.*") if f.suffix.lower() in SUPPORTED_SUFFIXES]

    if not files:
        print(f"No supported files found in {docs_dir}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        return

    print(f"Found {len(files)} file(s) to index\n")

    used: set[str] = set()
    descriptions: dict[str, dict] = {}  # table_name -> {file, description}
    for file_path in files:
        base = safe_table_name(file_path.stem)
        table_name = base
        if table_name in used:
            # Two different files sanitized to the same name — disambiguate.
            digest = hashlib.md5(file_path.name.encode("utf-8")).hexdigest()[:6]
            table_name = f"{base}_{digest}"
        used.add(table_name)
        print(f"  Indexing: {file_path.name} → table '{table_name}'")

        # Text extraction (LiteParse/OCR) and chunking are CPU-heavy and fully
        # synchronous — run them off the event loop so the caller's UI stays
        # responsive (the web reindex awaits this on the same loop as NiceGUI).
        try:
            text = await asyncio.to_thread(extract_text, file_path)
        except Exception as e:
            print(f"    Error extracting text: {e}")
            if progress_cb:
                progress_cb(file_path.name, False)  # failed → flagged in UI
            continue

        if not text.strip():
            print(f"    Warning: no text extracted")
            if progress_cb:
                progress_cb(file_path.name, False)
            continue

        chunks = await asyncio.to_thread(split_text, text)
        print(f"    {len(chunks)} chunks, embedding...")

        # Embedding and the (optional) LLM description are independent given the
        # text — run them concurrently so the description adds little wall-clock.
        if vdb_settings.descriptions_enabled:
            embeddings, description = await asyncio.gather(
                embed_batch(chunks),
                describe_document(text),
            )
        else:
            embeddings = await embed_batch(chunks)
            description = ""

        # seq = chunk's position in the document — lets the retriever stitch
        # back contiguous neighborhoods (see gather_neighbors in tools.py).
        records = [
            {"text": chunk, "vector": emb, "seq": i}
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        # Sync LanceDB disk writes — also off-loop.
        try:
            await asyncio.to_thread(db.drop_table, table_name, ignore_missing=True)
        except Exception:
            pass

        await asyncio.to_thread(db.create_table, table_name, data=records)
        descriptions[table_name] = {"file": file_path.name, "description": description}
        print(f"    Done — {len(records)} vectors stored")
        if progress_cb:
            progress_cb(file_path.name, True)

    # Persist per-file descriptions next to the DB for the Planner to read.
    save_descriptions(db_path, descriptions)
    print(f"\nIndexing complete. DB at: {db_path}")


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
