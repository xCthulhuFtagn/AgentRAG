"""Embedding provider — FastEmbed (local ONNX) or GigaChat (API), one switch.

Mirrors the LLM_PROVIDER pattern: both stacks are configured side-by-side in
.env, `EMBEDDING_PROVIDER` picks which one runs at import time. Changing the
provider (or the GigaChat model) invalidates existing vectors → detected at
search time via the sidecar's `embedding_model` / `embedding_dim` fields and
raises a clear error ("full reindex required").

Signatures stay unchanged: embed(text) → list[float], embed_batch(texts) →
list[list[float]]. FastEmbed runs sync ONNX off the loop via asyncio.to_thread;
GigaChatEmbeddings is async-native.
"""

import asyncio
from functools import lru_cache

from src.config import general_settings
from src.vectordb.config import vdb_settings


def _get_embedding_dim() -> int:
    """Dimension of the active embedding model (read without loading the model)."""
    if general_settings.embedding_provider == "gigachat":
        return _GIGACHAT_EMBEDDING_DIMS.get(
            general_settings.gigachat_embedding_model, 2560
        )
    return 384  # paraphrase-multilingual-MiniLM-L12-v2


def _get_embedding_model_name() -> str:
    """Human-readable name of the active model for sidecar validation."""
    if general_settings.embedding_provider == "gigachat":
        return f"gigachat:{general_settings.gigachat_embedding_model}"
    return vdb_settings.embedding_model


# Known GigaChat embedding model dimensions.
_GIGACHAT_EMBEDDING_DIMS = {
    "EmbeddingsGigaR": 2560,
    "Embeddings-2": 1024,
    "GigaEmbeddings-3B-2025-09": 2048,
}


# ── FastEmbed path ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_fastembed_model():
    """Load FastEmbed model (cached): vdb_settings.embedding_model, 384 dims.

    cache_dir must be explicit: FastEmbed's default is {tempdir}/fastembed_cache,
    and /tmp is tmpfs on many distros — the ~252MB model would re-download after
    every reboot (and air-gapped runs would break).
    """
    from fastembed import TextEmbedding

    return TextEmbedding(
        model_name=vdb_settings.embedding_model,
        cache_dir=vdb_settings.embedding_cache_dir,
    )


async def _embed_fastembed(text: str) -> list[float]:
    model = _get_fastembed_model()
    embeddings = await asyncio.to_thread(
        lambda: list(model.embed([text]))[0].tolist()
    )
    return embeddings


async def _embed_batch_fastembed(texts: list[str]) -> list[list[float]]:
    model = _get_fastembed_model()
    embeddings = await asyncio.to_thread(
        lambda: [e.tolist() for e in model.embed(texts)]
    )
    return embeddings


# ── GigaChat path ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_gigachat_embeddings():
    """Cached GigaChatEmbeddings instance for the configured model."""
    from langchain_gigachat.embeddings import GigaChatEmbeddings

    return GigaChatEmbeddings(
        model=general_settings.gigachat_embedding_model,
        credentials=general_settings.gigachat_credentials,
        scope=general_settings.gigachat_scope,
        base_url=general_settings.gigachat_base_url,
        verify_ssl_certs=general_settings.gigachat_verify_ssl_certs,
    )


async def _embed_gigachat(text: str) -> list[float]:
    """Single-text embedding via GigaChat — embed_query with asymmetric prefix."""
    emb = _get_gigachat_embeddings()
    # embed_query applies the query prefix (use_prefix_query=True) — asymmetric
    # embeddings give better retrieval when query and passage use different prefixes.
    vec = await emb.aembed_query(text)
    return vec


# GigaChat's gateway rejects large embedding payloads outright (413 Request
# Entity Too Large) — a whole book's chunks in one aembed_documents call is
# megabytes of JSON. Cap each request by count AND cumulative text size; both
# limits are far under the observed rejection threshold, so 413 splitting
# below is a fallback, not the normal path.
_EMBED_BATCH_MAX_TEXTS = 64
_EMBED_BATCH_MAX_CHARS = 100_000


async def _embed_batch_gigachat(texts: list[str]) -> list[list[float]]:
    """Batch embedding via GigaChat — embed_documents WITHOUT query prefix.

    Requests go out sequentially (GigaChat caps concurrent requests per
    account) in slices of ≤ _EMBED_BATCH_MAX_TEXTS / _EMBED_BATCH_MAX_CHARS.
    A slice the gateway still 413s is halved recursively; a single text that
    413s is genuinely unembeddable — re-raised.
    """
    from gigachat.exceptions import RequestEntityTooLargeError

    emb = _get_gigachat_embeddings()

    async def _embed_slice(slice_: list[str]) -> list[list[float]]:
        try:
            # embed_documents does NOT use the query prefix — correct for
            # passage/index embeddings. The query/document asymmetry is
            # handled by the two methods.
            return await emb.aembed_documents(slice_)
        except RequestEntityTooLargeError:
            if len(slice_) == 1:
                raise
            mid = len(slice_) // 2
            return await _embed_slice(slice_[:mid]) + await _embed_slice(slice_[mid:])

    vecs: list[list[float]] = []
    batch: list[str] = []
    batch_chars = 0
    for text in texts:
        if batch and (
            len(batch) >= _EMBED_BATCH_MAX_TEXTS
            or batch_chars + len(text) > _EMBED_BATCH_MAX_CHARS
        ):
            vecs += await _embed_slice(batch)
            batch, batch_chars = [], 0
        batch.append(text)
        batch_chars += len(text)
    if batch:
        vecs += await _embed_slice(batch)
    return vecs


# ── Public API (dispatched by provider) ───────────────────────────────────────

# Set at import time based on EMBEDDING_PROVIDER — the provider is fixed for the
# process lifetime, so this is a one-time branch, not a per-call if/else.
if general_settings.embedding_provider == "gigachat":
    embed = _embed_gigachat
    embed_batch = _embed_batch_gigachat
else:
    embed = _embed_fastembed
    embed_batch = _embed_batch_fastembed
