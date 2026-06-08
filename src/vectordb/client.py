"""Async LanceDB client — singleton connection."""

from pathlib import Path

import lancedb

from src.vectordb.config import vdb_settings


def _ensure_dir(path: str) -> str:
    """Create the LanceDB directory (and parents) if absent."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


async def get_async_db(db_path: str | None = None):
    """Get or create async LanceDB connection (dir created if missing)."""
    path = _ensure_dir(db_path or vdb_settings.lance_db_path)
    return await lancedb.connect_async(path)


def get_sync_db(db_path: str | None = None):
    """Get sync LanceDB connection for indexing (dir created if missing)."""
    path = _ensure_dir(db_path or vdb_settings.lance_db_path)
    return lancedb.connect(path)
