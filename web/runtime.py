"""App-global runtime state — built once, shared across all clients.

- GRAPH: the compiled Agentic RAG graph (no checkpointer — see src/graph.py).
- STORE: the ProjectStore.
- per-project status (idle | reindexing) and asyncio locks for reindex.
- OCR sidecar: auto-launched in a daemon thread when GigaChat credentials are
  configured (GigaChat Vision OCR bypasses Tesseract for better multilingual
  accuracy on scanned documents).
"""

import asyncio
import logging

from src.graph import build_graph
from src.config import general_settings
from src.vectordb.config import vdb_settings
from web.projects import ProjectStore

log = logging.getLogger("agentrag.web")

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


# ── OCR GigaChat Vision sidecar — auto-launch ─────────────────────────────

_OCR_DEFAULT_PORT = 8830
_OCR_STARTUP_TIMEOUT = 10.0  # seconds to wait for the sidecar to become ready


def _maybe_start_ocr_sidecar() -> None:
    """Start the OCR GigaChat sidecar in a daemon thread if credentials exist.

    When GigaChat is configured (credentials are present), the OCR sidecar is
    launched automatically on http://127.0.0.1:{_OCR_DEFAULT_PORT}/ocr — no
    separate process to manage. If OCR_SERVER_URL is already set to a DIFFERENT
    URL (e.g. an EasyOCR/PaddleOCR sidecar), we don't override it.

    When we auto-start, we also set ``vdb_settings.ocr_server_url`` so the
    indexer's ``_get_parser()`` picks it up — LiteParse will then delegate OCR
    to our sidecar instead of its built-in Tesseract.

    The thread has auto-restart (if uvicorn dies, it restarts after a backoff),
    and we poll the health endpoint at startup so the first OCR request never
    hits an unbound port.
    """
    import time
    import urllib.request

    if not general_settings.gigachat_credentials:
        log.info("OCR sidecar: GigaChat credentials not set — skipping auto-start")
        return

    existing = vdb_settings.ocr_server_url
    if existing:
        log.info(
            "OCR sidecar: OCR_SERVER_URL already set to %s — skipping auto-start",
            existing,
        )
        return

    sidecar_url = f"http://127.0.0.1:{_OCR_DEFAULT_PORT}/ocr"
    health_url = f"http://127.0.0.1:{_OCR_DEFAULT_PORT}/health"
    from src.vectordb.ocr_gigachat_server import run_in_thread

    run_in_thread(_OCR_DEFAULT_PORT)
    vdb_settings.ocr_server_url = sidecar_url

    # Block until the sidecar is actually listening, so the first OCR call
    # (which may come within milliseconds) doesn't get a connection refused.
    # Use a custom opener without proxy — urllib honours http_proxy by default
    # and localhost through a proxy returns 502.
    no_proxy_handler = urllib.request.ProxyHandler({})
    no_proxy_opener = urllib.request.build_opener(no_proxy_handler)
    deadline = time.monotonic() + _OCR_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            resp = no_proxy_opener.open(health_url, timeout=2)
            if resp.status == 200:
                log.info(
                    "OCR sidecar: ready on %s (set OCR_SERVER_URL to override)",
                    sidecar_url,
                )
                return
        except Exception:
            pass
        time.sleep(0.2)

    log.warning(
        "OCR sidecar: did not become ready within %.1fs — "
        "OCR requests may fail until it starts",
        _OCR_STARTUP_TIMEOUT,
    )


_maybe_start_ocr_sidecar()
