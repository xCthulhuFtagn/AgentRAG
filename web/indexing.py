"""Per-project (re)indexing — wraps src.vectordb.indexer.

Two entry points, both serialized per-project via a lock and freezing the chat
("reindexing" status) for the duration:
- reindex_project: wipe the project's LanceDB and rebuild from all files on
  disk (used when indexing settings change — old chunks are invalid).
- update_project_index: incremental — drop the tables of removed files and
  index only the added/changed ones; untouched files keep their tables.

Both use the project's own indexing settings (ProjectStore.get_index_settings).
"""

import shutil
from pathlib import Path

from src.vectordb.client import invalidate_db_cache
from src.vectordb.indexer import (
    SUPPORTED_SUFFIXES,
    index_documents,
    index_files,
    remove_files_from_index,
)
from web import runtime


async def reindex_project(pid: str) -> None:
    """Rebuild the project's vector DB from scratch from its uploaded files."""
    store = runtime.STORE
    lock = runtime.get_lock(pid)

    async with lock:
        runtime.set_status(pid, "reindexing")
        runtime.start_progress(pid)  # all files pending → UI spins each
        try:
            db_path = store.db_path(pid)
            files_dir = store.files_dir(pid)

            # Wipe existing tables so deleted/renamed files don't linger. Drop
            # any cached connection first — otherwise it would keep pointing at
            # the now-deleted directory once index_documents recreates it.
            invalidate_db_cache(db_path)
            db_dir = Path(db_path)
            if db_dir.exists():
                shutil.rmtree(db_dir, ignore_errors=True)

            # Only index if the project has at least one supported file.
            has_files = files_dir.exists() and any(
                f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
                for f in files_dir.iterdir()
            )
            if has_files:
                await index_documents(
                    str(files_dir),
                    db_path,
                    progress_cb=lambda fn, ok: runtime.mark_file(pid, fn, ok),
                    settings=store.get_index_settings(pid),
                )
        finally:
            # Keep the progress map (failed files stay flagged until the next
            # reindex); only the frozen status clears.
            runtime.set_status(pid, "idle")


async def update_project_index(
    pid: str, added: list[str], removed: list[str]
) -> None:
    """Apply a file-edit delta to the project's vector DB incrementally.

    `added` — file names to (re)index (new uploads, replacements, rename
    targets); `removed` — file names whose tables must be dropped (deletions,
    rename sources). Files not in either list keep their existing tables.
    """
    store = runtime.STORE
    lock = runtime.get_lock(pid)

    async with lock:
        runtime.set_status(pid, "reindexing")
        try:
            db_path = store.db_path(pid)
            files_dir = store.files_dir(pid)

            to_index = [files_dir / name for name in added if (files_dir / name).is_file()]
            present = {f["name"] for f in store.list_files(pid)}
            runtime.init_partial_progress(
                pid, pending={p.name for p in to_index}, present=present
            )

            if removed:
                await remove_files_from_index(removed, db_path)
            if to_index:
                await index_files(
                    to_index,
                    db_path,
                    progress_cb=lambda fn, ok: runtime.mark_file(pid, fn, ok),
                    settings=store.get_index_settings(pid),
                )
        finally:
            runtime.set_status(pid, "idle")
