"""FastEmbed wrapper — async via asyncio.to_thread.

Uses ONNX runtime, no PyTorch needed. Works in air-gapped environments.
"""

import asyncio
from functools import lru_cache

from fastembed import TextEmbedding

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_embedding_model() -> TextEmbedding:
    """Load FastEmbed model (cached). BAAI/bge-small-en-v1.5 — 384 dims."""
    return TextEmbedding(model_name=EMBEDDING_MODEL)


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
