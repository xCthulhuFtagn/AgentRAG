"""Per-project reindexing — wraps src.vectordb.indexer.index_documents.

Wipes the project's LanceDB and rebuilds it from the current files on disk.
Sets the project status to "reindexing" for the duration (freezes the chat).
"""

import shutil
from pathlib import Path

from src.vectordb.indexer import index_documents, SUPPORTED_SUFFIXES
from web import runtime


async def reindex_project(pid: str) -> None:
    """Rebuild the project's vector DB from its uploaded files.

    Safe to call after any file mutation (add/rename/delete). Serialized
    per-project via a lock so concurrent uploads don't race.
    """
    store = runtime.STORE
    lock = runtime.get_lock(pid)

    async with lock:
        runtime.set_status(pid, "reindexing")
        runtime.start_progress(pid)  # all files pending → UI spins each
        try:
            db_path = store.db_path(pid)
            files_dir = store.files_dir(pid)

            # Wipe existing tables so deleted/renamed files don't linger.
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
                )
        finally:
            # Keep the progress map (failed files stay flagged until the next
            # reindex); only the frozen status clears.
            runtime.set_status(pid, "idle")
