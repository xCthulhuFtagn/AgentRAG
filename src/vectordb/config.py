"""Vector DB configuration — typed settings loaded from the environment (.env).

Self-contained config for the vectordb package (embeddings, LanceDB client,
indexer, search tools). Uses pydantic-settings so values are parsed, validated,
and bounds-checked instead of raw os.getenv. The global src/config.py keeps only
DeepSeek/agent-loop concerns.

Access values via the `vdb_settings` instance (e.g. vdb_settings.search_top_k).
Env var names are the UPPERCASE field names (matching is case-insensitive),
e.g. SEARCH_TOP_K, EXPAND_PADDING.
"""

from typing import Optional

from pydantic import Field, field_validator
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
    # paraphrase-multilingual-MiniLM-L12-v2 → 384 dims, multilingual (incl.
    # Russian) — an English-only model (e.g. bge-small-en) blinds retrieval on a
    # non-English corpus. FOOTGUN: changing the model (even at the same dim)
    # makes existing vectors incompatible → full reindex required. The dimension
    # is derived from the model, never hardcoded.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Where FastEmbed stores the downloaded ONNX model (~252MB). Without an
    # explicit dir FastEmbed defaults to {tempdir}/fastembed_cache, and /tmp is
    # tmpfs on many distros — wiped on reboot → the model re-downloads every
    # boot (and breaks air-gapped use). Kept under data/ with the rest of the
    # persistent state.
    embedding_cache_dir: str = "./data/fastembed_cache"

    # ── Indexing / chunking (chars) ──────────────────────────────────────────
    # Only affects newly indexed documents; reindex to apply to existing ones.
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)

    # ── Per-file descriptions ────────────────────────────────────────────────
    # When True, indexing generates an LLM summary per file and the Planner
    # routes using those summaries. When False, no summary is generated and the
    # Planner sees only table names (no LLM cost at index time).
    descriptions_enabled: bool = True
    # How many leading chars of a file are sent to the LLM for its description.
    # Title/abstract/intro carry most of the routing signal; this bounds cost.
    describe_max_chars: int = Field(default=6000, ge=1)

    # ── OCR (LiteParse) ──────────────────────────────────────────────────────
    # By default LiteParse OCRs scanned/image content with its built-in Tesseract,
    # which is noisy on tiny embedded images ("Image too small to scale!!") and
    # weaker on Cyrillic. Point ocr_server_url at a LOCAL OCR sidecar to delegate
    # OCR there — better multilingual accuracy, no native warnings, fully offline
    # (heavy torch/paddle deps live in the sidecar's own venv, not this project).
    # Run one from the liteparse repo's ocr/ dir via `uv run server.py`:
    #   EasyOCR   → http://localhost:8828/ocr
    #   PaddleOCR → http://localhost:8829/ocr
    # ocr_language forwards the language code (e.g. "ru") to that server.
    # Both unset → LiteParse's built-in Tesseract (current behavior).
    ocr_server_url: Optional[str] = None
    ocr_language: Optional[str] = None

    @field_validator("ocr_server_url", "ocr_language", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        # Unspecified address (unset, or set-but-empty `OCR_SERVER_URL=` in .env)
        # → None → LiteParse uses its built-in Tesseract. So OCR works by default
        # without requiring an external sidecar.
        if isinstance(v, str) and not v.strip():
            return None
        return v

    # ── Search ───────────────────────────────────────────────────────────────
    # Nearest chunks per (collection, query) before stitching.
    search_top_k: int = Field(default=5, ge=1)

    # ── Reranking (LLM per-chunk relevance assessment) ──────────────────────
    # When True, search_fanout calls the LLM for every retrieved chunk to
    # assess relevance to the original query.  The per-search topic-hit trend
    # («прирост по теме») shown to the judge is then powered by these LLM
    # scores instead of keyword matching.  When False, no relevance assessment
    # happens — the judge sees only the raw +N novelty delta.  Search-time
    # (no reindex on change), threaded via stitch_settings.
    reranking_enabled: bool = True

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
