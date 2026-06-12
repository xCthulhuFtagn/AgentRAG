"""FastEmbed wrapper — async via asyncio.to_thread.

Uses ONNX runtime, no PyTorch needed. Works in air-gapped environments.
"""

import asyncio
from functools import lru_cache

from fastembed import TextEmbedding

from src.vectordb.config import vdb_settings


@lru_cache(maxsize=1)
def _get_embedding_model() -> TextEmbedding:
    """Load FastEmbed model (cached): vdb_settings.embedding_model, 384 dims.

    cache_dir must be explicit: FastEmbed's default is {tempdir}/fastembed_cache,
    and /tmp is tmpfs on many distros — the ~252MB model would re-download after
    every reboot (and air-gapped runs would break).
    """
    return TextEmbedding(
        model_name=vdb_settings.embedding_model,
        cache_dir=vdb_settings.embedding_cache_dir,
    )


async def embed(text: str) -> list[float]:
    """Embed a single text asynchronously.

    FastEmbed is sync (CPU-bound ONNX), so we use asyncio.to_thread.
    """
    model = _get_embedding_model()
    embeddings = await asyncio.to_thread(
        lambda: list(model.embed([text]))[0].tolist()
    )
    return embeddings


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts asynchronously."""
    model = _get_embedding_model()
    embeddings = await asyncio.to_thread(
        lambda: [e.tolist() for e in model.embed(texts)]
    )
    return embeddings
