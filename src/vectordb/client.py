"""Async LanceDB client — singleton connection."""

import lancedb

from src.config import LANCE_DB_PATH


async def get_async_db(db_path: str | None = None):
    """Get or create async LanceDB connection."""
    path = db_path or LANCE_DB_PATH
    return await lancedb.connect_async(path)


def get_sync_db(db_path: str | None = None):
    """Get sync LanceDB connection (for indexing)."""
    path = db_path or LANCE_DB_PATH
    return lancedb.connect(path)
