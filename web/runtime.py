"""App-global runtime state — built once, shared across all clients.

- GRAPH: the compiled Agentic RAG graph (MemorySaver checkpointer).
- STORE: the ProjectStore.
- per-project status (idle | reindexing) and asyncio locks for reindex.
"""

import asyncio

from src.graph import build_graph
from web.projects import ProjectStore

# Built once at import — reused for every chat request.
GRAPH = build_graph()
STORE = ProjectStore()

# project_id -> "idle" | "reindexing"
_status: dict[str, str] = {}
# project_id -> asyncio.Lock (serialize reindex per project)
_locks: dict[str, asyncio.Lock] = {}
# project_id -> {filename: ok}  (True = indexed, False = failed); a filename
# absent from the map is still pending. Persists after the reindex so failed
# files stay flagged until the next one (reset by start_progress).
_progress: dict[str, dict[str, bool]] = {}


def get_status(pid: str) -> str:
    return _status.get(pid, "idle")


def set_status(pid: str, status: str) -> None:
    _status[pid] = status


def is_frozen(pid: str) -> bool:
    return get_status(pid) == "reindexing"


def get_lock(pid: str) -> asyncio.Lock:
    if pid not in _locks:
        _locks[pid] = asyncio.Lock()
    return _locks[pid]


def start_progress(pid: str) -> None:
    """Reset progress at the start of a full reindex (all files pending)."""
    _progress[pid] = {}


def init_partial_progress(pid: str, pending: set[str], present: set[str]) -> None:
    """Progress at the start of an incremental update.

    Only `pending` files show as in-progress; the untouched rest of `present`
    show as already indexed (keeping any earlier failure flags); entries for
    files no longer on disk are dropped.
    """
    prog = {n: ok for n, ok in _progress.get(pid, {}).items() if n in present}
    for name in present - pending:
        prog.setdefault(name, True)
    for name in pending:
        prog.pop(name, None)
    _progress[pid] = prog


def mark_file(pid: str, filename: str, ok: bool) -> None:
    """Record a file as indexed (ok=True) or failed (ok=False)."""
    _progress.setdefault(pid, {})[filename] = ok


def get_progress(pid: str) -> dict[str, bool]:
    """{filename: ok} — absent filename = pending."""
    return _progress.get(pid, {})


def clear_progress(pid: str) -> None:
    _progress.pop(pid, None)
