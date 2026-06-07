"""Async LanceDB client — singleton connection."""

from pathlib import Path

import lancedb

from src.config import LANCE_DB_PATH


def _ensure_dir(path: str) -> str:
    """Create the LanceDB directory (and parents) if absent."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


async def get_async_db(db_path: str | None = None):
    """Get or create async LanceDB connection (dir created if missing)."""
    path = _ensure_dir(db_path or LANCE_DB_PATH)
    return await lancedb.connect_async(path)


def get_sync_db(db_path: str | None = None):
    """Get sync LanceDB connection for indexing (dir created if missing)."""
    path = _ensure_dir(db_path or LANCE_DB_PATH)
    return lancedb.connect(path)
