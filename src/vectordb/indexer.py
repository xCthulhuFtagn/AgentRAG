"""Document indexing — loads documents into LanceDB with FastEmbed embeddings.

Usage:
    python -m src.vectordb.indexer --dir docs/sample_docs
    python -m src.vectordb.indexer --dir docs/sample_docs --db ./lancedb_data
"""

import argparse
import asyncio
import sys
from pathlib import Path

from pypdf import PdfReader

from src.config import LANCE_DB_PATH
from src.vectordb.embeddings import embed_batch
from src.vectordb.client import get_sync_db


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(str(file_path))
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def extract_text(file_path: Path) -> str:
    """Extract text from various file formats."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"):
        return file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Simple overlapping chunk splitter."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap

    return chunks


async def index_documents(docs_dir: str, db_path: str = LANCE_DB_PATH):
    """Index all documents from a directory into LanceDB.

    Each file becomes a separate LanceDB collection (table).
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Error: directory '{docs_dir}' does not exist")
        sys.exit(1)

    # Use sync DB for indexing (simpler for bulk operations)
    db = get_sync_db(db_path)
    files = list(docs_path.glob("*.*"))

    if not files:
        print(f"No files found in {docs_dir}")
        return

    print(f"Found {len(files)} file(s) to index\n")

    for file_path in files:
        if file_path.suffix.lower() not in (".pdf", ".txt", ".md"):
            print(f"  Skipping unsupported: {file_path.name}")
            continue

        table_name = file_path.stem.replace(" ", "_").replace("-", "_").replace(".", "_")
        print(f"  Indexing: {file_path.name} → table '{table_name}'")

        try:
            text = extract_text(file_path)
        except Exception as e:
            print(f"    Error extracting text: {e}")
            continue

        if not text.strip():
            print(f"    Warning: no text extracted")
            continue

        chunks = split_text(text, chunk_size=500, overlap=50)
        print(f"    {len(chunks)} chunks, embedding...")

        # Embed all chunks in batches
        embeddings = await embed_batch(chunks)

        # Create or replace table
        records = [
            {"text": chunk, "vector": emb}
            for chunk, emb in zip(chunks, embeddings)
        ]

        try:
            db.drop_table(table_name, ignore_missing=True)
        except Exception:
            pass

        db.create_table(table_name, data=records)
        print(f"    Done — {len(records)} vectors stored")

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
        default=LANCE_DB_PATH,
        help=f"Path to LanceDB database (default: {LANCE_DB_PATH})",
    )
    args = parser.parse_args()

    asyncio.run(index_documents(args.dir, args.db))


if __name__ == "__main__":
    main()
