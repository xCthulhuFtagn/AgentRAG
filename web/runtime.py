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
