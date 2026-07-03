"""Async LanceDB client — connections cached per (event loop, path)."""

import asyncio
from pathlib import Path

import lancedb

from src.vectordb.config import vdb_settings

# Async connections are bound to the event loop they were opened on, so the
# cache key includes the running loop's id (distinct test runs / asyncio.run
# calls get their own loop and must not reuse another loop's connection).
# Without this cache, every vector_search / gather_neighbors / count_chunks
# call opened a brand-new connection — cheap individually, but a single
# judge/planner turn opens several per search.
_async_db_cache: dict[tuple[int, str], "lancedb.AsyncConnection"] = {}


def _ensure_dir(path: str) -> str:
    """Create the LanceDB directory (and parents) if absent."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


async def get_async_db(db_path: str | None = None):
    """Get or create a cached async LanceDB connection (dir created if missing)."""
    path = _ensure_dir(db_path or vdb_settings.lance_db_path)
    key = (id(asyncio.get_running_loop()), path)
    db = _async_db_cache.get(key)
    if db is None:
        db = await lancedb.connect_async(path)
        _async_db_cache[key] = db
    return db


def get_sync_db(db_path: str | None = None):
    """Get sync LanceDB connection for indexing (dir created if missing).

    Not cached: indexing runs are one-shot and a sync connection is a cheap
    handle, unlike the async client which is worth reusing across the many
    per-turn searches above.
    """
    path = _ensure_dir(db_path or vdb_settings.lance_db_path)
    return lancedb.connect(path)


def invalidate_db_cache(db_path: str) -> None:
    """Drop any cached async connection(s) for this path.

    Call after the directory is wiped or rebuilt out from under an open
    connection (project delete, full reindex) — otherwise a cached handle can
    keep pointing at now-gone or now-different on-disk state.
    """
    for key in [k for k in _async_db_cache if k[1] == db_path]:
        db = _async_db_cache.pop(key)
        try:
            db.close()
        except Exception:
            pass
