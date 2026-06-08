"""Vector DB configuration — typed settings loaded from the environment (.env).

Self-contained config for the vectordb package (embeddings, LanceDB client,
indexer, search tools). Uses pydantic-settings so values are parsed, validated,
and bounds-checked instead of raw os.getenv. The global src/config.py keeps only
DeepSeek/agent-loop concerns.

Access values via the `vdb_settings` instance (e.g. vdb_settings.search_top_k).
Env var names are the UPPERCASE field names (matching is case-insensitive),
e.g. SEARCH_TOP_K, EXPAND_PADDING.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorDBSettings(BaseSettings):
    """All vectordb knobs, read from .env / process env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── Storage ──────────────────────────────────────────────────────────────
    # CLI default lives under data/ alongside the web's per-project DBs. "_cli"
    # can't collide with project ids (uuid4 hex) and isn't listed as a project.
    lance_db_path: str = "./data/lancedb/_cli"

    # ── Embeddings (FastEmbed) ───────────────────────────────────────────────
    # BAAI/bge-small-en-v1.5 → 384 dims. FOOTGUN: changing the model changes the
    # vector dimension, so existing tables become incompatible → full reindex
    # required. The dimension is derived from the model, never hardcoded.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ── Indexing / chunking (chars) ──────────────────────────────────────────
    # Only affects newly indexed documents; reindex to apply to existing ones.
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)

    # ── Per-file descriptions ────────────────────────────────────────────────
    # When True, indexing generates an LLM summary per file and the Planner
    # routes using those summaries. When False, no summary is generated and the
    # Planner sees only table names (no LLM cost at index time).
    descriptions_enabled: bool = True

    # ── Search ───────────────────────────────────────────────────────────────
    # Nearest chunks per (collection, query) before stitching.
    search_top_k: int = Field(default=5, ge=1)

    # ── Neighbor stitching (deterministic context expansion) ─────────────────
    # Each hit's seq → window [seq-expand_padding, seq+expand_padding]; two
    # windows merge when the uncovered gap between them is <= bridge_gap, then
    # every chunk in the merged ranges is fetched by seq (capped at max_expanded
    # per result). Effective hit-merge distance = 2*expand_padding + bridge_gap + 1.
    # Defaults tuned (P=1, gap=2 → merge distance 5) for ~25% less pulled content
    # than P=2/gap=1 while still recovering split blocks; do NOT lower top_k —
    # a block's head may be anchored by a low-ranked hit that top_k=5 still catches.
    expand_padding: int = Field(default=1, ge=0)
    bridge_gap: int = Field(default=2, ge=0)
    max_expanded: int = Field(default=16, ge=1)


vdb_settings = VectorDBSettings()
