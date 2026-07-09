"""OCR sidecar — delegates image text recognition to GigaChat Vision API.

A minimal HTTP server that implements the LiteParse OCR sidecar contract:
POST /ocr, multipart image → {"results": [{"text", "bbox", "confidence"}]}
(the EasyOCR-server shape — LiteParse's HTTP OCR client rejects any other
response as unparseable and the page counts as OCR-failed). Uses
GigaChat-2-Pro (the flagship model with vision support) via the GigaChat
Python SDK.

Run standalone:
    python -m src.vectordb.ocr_gigachat_server --port 8830

Then set OCR_SERVER_URL=http://localhost:8830/ocr in .env — the indexer's
_get_parser() already passes ocr_server_url to LiteParse, zero code changes
needed in the main pipeline.

Also importable: `run_in_thread(port)` starts the server in a daemon thread,
used by the web app to auto-launch the sidecar alongside the main service.
"""

import argparse
import asyncio
import logging
import random
import threading
import time
from functools import lru_cache
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.config import general_settings

log = logging.getLogger("agentrag.ocr_sidecar")

OCR_PROMPT = (
    "Распознай весь текст на этом изображении. "
    "Верни только текст, без комментариев. "
    "Сохраняй структуру: абзацы, списки, заголовки — как видишь."
)

# GigaChat free tier (PERS scope) is heavily rate-limited.  Retry transient
# failures (429 / 5xx) with exponential backoff + retry_after honouring.
_OCR_MAX_RETRIES = 5
_OCR_BACKOFF_BASE = 3.0   # seconds: 3 → 6 → 12 → 24 → 48 (capped)
_OCR_BACKOFF_CAP = 60.0

# Magic bytes → extension so GigaChat can infer the MIME type from the filename
# even when the uploader doesn't send a recognised name.
_MAGIC_TO_EXT: dict[bytes, str] = {
    b"\x89PNG\r\n\x1a\n": ".png",
    b"\xff\xd8\xff": ".jpg",
    b"GIF87a": ".gif",
    b"GIF89a": ".gif",
    b"RIFF": ".webp",   # RIFF....WEBP — check further below
    b"BM": ".bmp",
}


def _guess_extension(image_bytes: bytes) -> str:
    """Best-effort file extension from magic bytes; falls back to .png."""
    for magic, ext in _MAGIC_TO_EXT.items():
        if image_bytes.startswith(magic):
            if ext == ".webp" and len(image_bytes) >= 12:
                # RIFF container: could be AVI or WebP — check the FOURCC.
                if image_bytes[8:12] == b"WEBP":
                    return ".webp"
                continue  # not WebP — keep scanning
            return ext
    return ".png"  # safe fallback — GigaChat Vision handles PNG


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    """Best-effort (width, height) of the uploaded image; (0, 0) if unreadable."""
    import io

    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as im:
            return im.width, im.height
    except Exception:
        return 0, 0


@lru_cache(maxsize=1)
def _build_gigachat_client():
    """Create a GigaChat client for vision OCR, reading creds from settings.

    Cached: creating a new GigaChat() instance triggers an OAuth token exchange,
    and the free tier rate-limits that too.  One client, one token exchange,
    reused for the lifetime of the sidecar process.

    httpx (used internally by the GigaChat SDK) honors ``http_proxy`` /
    ``https_proxy`` env vars.  A local proxy can't forward HTTPS to GigaChat
    (SSL EOF), so we temporarily clear those vars during client creation AND
    force-create the internal httpx clients while they're cleared (they are
    lazily instantiated on first API call).
    """
    import os
    from gigachat import GigaChat

    proxy_keys = [
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
        "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY",
    ]
    saved = {k: os.environ.pop(k, None) for k in proxy_keys}
    try:
        client = GigaChat(
            credentials=general_settings.gigachat_credentials,
            scope=general_settings.gigachat_scope,
            base_url=general_settings.gigachat_base_url,
            verify_ssl_certs=general_settings.gigachat_verify_ssl_certs,
            model="GigaChat-2-Pro",  # flagship with vision
            # SDK default is 30s — a dense scanned page times out mid-reply
            # ("The read operation timed out"); see gigachat_ocr_timeout.
            timeout=general_settings.gigachat_ocr_timeout,
        )
        # Force-create internal httpx clients NOW, while proxy vars are cleared.
        # These are @property getters that lazily build httpx.Client/AsyncClient.
        # Once created they keep their settings (including no-proxy) for life.
        _ = client._client       # triggers httpx.Client(**kwargs) — no proxy
        _ = client._auth_client  # triggers httpx.Client(**kwargs) — no proxy
        return client
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ── Endpoint ──────────────────────────────────────────────────────────────────

async def ocr_endpoint(request):
    """POST /ocr — LiteParse-compatible OCR endpoint.

    Accepts multipart/form-data with an image file, sends it to GigaChat Vision
    for recognition, returns {"text": "..."}.
    """
    form = await request.form()
    # LiteParse sends the image as the first file field (name varies).
    image_field = next(
        (f for f in form.values() if hasattr(f, "file")), None
    )
    if image_field is None:
        return JSONResponse(
            {"error": "no image file in request"}, status_code=400
        )

    image_bytes = await image_field.read()
    if not image_bytes:
        return JSONResponse(
            {"error": "empty image file"}, status_code=400
        )

    # Preserve the original filename so GigaChat can infer the MIME type from
    # the extension.  Raw bytes default to application/octet-stream → 400.
    original_name = getattr(image_field, "filename", None) or ""

    # The GigaChat SDK's upload_file + chat are synchronous — run off the event
    # loop so the sidecar stays responsive to health checks. The semaphore caps
    # in-flight GigaChat calls (GigaChat limits concurrent requests per
    # account; excess fan-out = 429 storms) — waiters queue on the event loop,
    # holding no threads. The CLIENT keeps the queue short: LiteParse's OCR
    # HTTP client hard-times-out at 60s per request, queue time included, so
    # the indexer bounds its page fan-out via vdb_settings.ocr_workers.
    try:
        async with request.app.state.gigachat_semaphore:
            text = await asyncio.to_thread(
                _recognize_sync, image_bytes, original_name
            )
    except Exception as e:
        log.warning("GigaChat Vision OCR failed: %s", e)
        return JSONResponse(
            {"error": f"OCR failed: {e}"}, status_code=500
        )

    if not text:
        return JSONResponse({"results": []})

    # GigaChat Vision reads the whole image at once (no per-line coordinates),
    # so the reply is a single full-image block. bbox is in the uploaded
    # image's pixel space, like EasyOCR's.
    width, height = _image_size(image_bytes)
    return JSONResponse(
        {"results": [{"text": text, "bbox": [0, 0, width, height], "confidence": 1.0}]}
    )


# ── Sync recognition (runs off the event loop) ────────────────────────────────

def _recognize_sync(image_bytes: bytes, original_name: str = "") -> str:
    """Upload image → GigaChat Vision → recognized text (sync, off-loop).

    Retries transient failures (429 / 5xx) with exponential backoff +
    retry_after honouring.  Two-step GigaChat Vision flow:
    1. upload_file — sends the image with a filename whose extension tells
       GigaChat the MIME type (raw bytes → application/octet-stream → 400)
    2. chat with Messages(attachments=[file_id]) — vision model reads the image

    *original_name* is the filename from the multipart upload; if it lacks a
    recognised extension, we guess from magic bytes (fallback: .png).
    """
    # Build a filename whose extension GigaChat can map to a MIME type.
    stem = Path(original_name).stem if original_name else "image"
    ext = Path(original_name).suffix if original_name else ""
    if not ext or ext.lower() not in {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
    }:
        ext = _guess_extension(image_bytes)
    filename = f"{stem}{ext}"

    return _recognize_with_retry(filename, image_bytes)


def _recognize_with_retry(filename: str, image_bytes: bytes) -> str:
    """Call upload_file → chat with retry on transient GigaChat errors.

    Honors the ``retry_after`` field on ``RateLimitError`` when present;
    otherwise uses exponential backoff with jitter.
    """
    from gigachat.exceptions import RateLimitError, ServerError
    from gigachat.models import Messages, Chat

    client = _build_gigachat_client()
    last_exc = None

    for attempt in range(_OCR_MAX_RETRIES + 1):
        try:
            # Step 1: upload the image — pass (filename, bytes) tuple so
            # GigaChat sees the extension and sets the correct Content-Type.
            uploaded = client.upload_file((filename, image_bytes))
            file_id = uploaded.id_  # Pydantic: id → id_ (Python builtin)

            # Step 2: chat with the image attached.
            messages = [
                Messages(
                    role="user",
                    content=OCR_PROMPT,
                    attachments=[file_id],
                )
            ]
            response = client.chat(Chat(messages=messages, max_tokens=4096))
            content = response.choices[0].message.content

            # Best-effort cleanup: every page uploads an image into the
            # account's quota-limited GigaChat file storage — a book-sized
            # corpus is hundreds of files. Never fail the OCR over it.
            try:
                client.delete_file(file_id)
            except Exception:
                pass

            return (content or "").strip()

        except (RateLimitError, ServerError) as e:
            last_exc = e
            if attempt >= _OCR_MAX_RETRIES:
                break

            # Honor the server's Retry-After if given; otherwise exponential
            # backoff. LiteParse OCRs pages concurrently (× several files
            # concurrently during indexing), so many callers hit this at once
            # with the same attempt number — a deterministic delay has them
            # all retry together, in lockstep, producing repeated 429 waves
            # instead of spreading out. Jitter (equal-jitter: half fixed, half
            # random) decorrelates them while keeping the backoff's magnitude.
            ra = getattr(e, "retry_after", None)
            if ra and ra > 0:
                base = float(ra) + 1.0  # pad by 1s to be safe
            else:
                base = min(_OCR_BACKOFF_BASE * (2 ** attempt), _OCR_BACKOFF_CAP)
            delay = base / 2 + random.uniform(0, base / 2)

            log.info(
                "OCR attempt %d/%d: %s — retrying in %.1fs",
                attempt + 1, _OCR_MAX_RETRIES + 1, e, delay,
            )
            time.sleep(delay)

    raise last_exc  # exhausted retries


# ── Server ────────────────────────────────────────────────────────────────────

async def health_endpoint(request):
    """GET /health — liveness probe so the web app can detect a dead sidecar."""
    return JSONResponse({"status": "ok"})


def make_app() -> Starlette:
    """Build the OCR sidecar ASGI app (importable for programmatic use)."""
    app = Starlette(
        routes=[
            Route("/ocr", ocr_endpoint, methods=["POST"]),
            Route("/health", health_endpoint, methods=["GET"]),
        ]
    )
    # Created per app (not module level): run_in_thread restarts a crashed
    # server on a NEW event loop, and an asyncio primitive must not outlive
    # the loop it first awaited on.
    app.state.gigachat_semaphore = asyncio.Semaphore(
        general_settings.gigachat_ocr_concurrency
    )
    return app


def run_server(port: int = 8830):
    """Run the OCR sidecar (blocking — called via `python -m` or a thread)."""
    app = make_app()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def run_in_thread(port: int = 8830) -> None:
    """Start the OCR sidecar in a daemon thread with auto-restart.

    If the uvicorn server crashes (e.g. port conflict, runtime error), the
    thread restarts it after a short delay.  The thread is a daemon so it dies
    with the main process; no cleanup needed.
    """
    def _run_with_restart():
        restart_delay = 2.0
        while True:
            try:
                log.info(
                    "OCR GigaChat sidecar starting on http://127.0.0.1:%d/ocr",
                    port,
                )
                run_server(port)
            except Exception as e:
                log.error("OCR sidecar crashed: %s — restarting in %.1fs", e, restart_delay)
                time.sleep(restart_delay)
                restart_delay = min(restart_delay * 2, 30.0)

    t = threading.Thread(
        target=_run_with_restart,
        name="ocr-gigachat-sidecar",
        daemon=True,
    )
    t.start()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OCR GigaChat Vision sidecar for LiteParse"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8830,
        help="Port to listen on (default: 8830)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    log.info(
        "Starting OCR GigaChat Vision sidecar on http://127.0.0.1:%d/ocr",
        args.port,
    )
    run_server(args.port)


if __name__ == "__main__":
    main()
