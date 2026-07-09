"""Configuration — LLM provider (DeepSeek/GigaChat) and agent-loop settings.

Vector DB settings live in src/vectordb/config.py. Access values via the
`general_settings` instance (e.g. general_settings.deepseek_model).
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeneralSettings(BaseSettings):
    """DeepSeek/GigaChat API + agent-loop knobs, read from .env / process env.

    Env var names are the UPPERCASE field names (case-insensitive),
    e.g. DEEPSEEK_API_KEY, MAX_ITERATIONS.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Which provider get_llm() builds (src/agents/common.py, src/vectordb/describe.py).
    # Both stacks stay preconfigured in .env at all times — this is the only
    # switch, no separate branch/deployment per provider.
    llm_provider: Literal["deepseek", "gigachat"] = "deepseek"

    # Which embedding provider to use for chunk/index embeddings and search queries.
    # "fastembed" — local ONNX model (paraphrase-multilingual-MiniLM-L12-v2, 384d),
    #   air-gapped, no API cost. "gigachat" — GigaChat Embeddings API (2560d by
    #   default), better multilingual alignment, needs network + credentials.
    # Changing this requires a full reindex (vectors from different models are
    # incompatible — a mismatch is detected at search time and raises an error).
    embedding_provider: Literal["fastembed", "gigachat"] = "fastembed"
    # GigaChat embedding model: EmbeddingsGigaR (2560d), Embeddings-2 (1024d),
    # or GigaEmbeddings-3B-2025-09 (2048d). Only used when embedding_provider=gigachat.
    gigachat_embedding_model: str = "EmbeddingsGigaR"

    # Which OCR engine handles scanned/image pages at index time — independent
    # of LLM_PROVIDER (agents on DeepSeek + OCR on GigaChat is a valid split:
    # GigaChat's per-account concurrency cap hurts the agents' parallel calls
    # but is fine for the serialized OCR sidecar).
    # "gigachat" — the web app auto-starts the built-in GigaChat Vision OCR
    #   sidecar (src/vectordb/ocr_gigachat_server.py, :8830); needs
    #   GIGACHAT_CREDENTIALS. "standard" — no auto-start: an explicitly set
    #   OCR_SERVER_URL (external EasyOCR/PaddleOCR sidecar) is used as-is,
    #   otherwise LiteParse's built-in Tesseract. A set OCR_SERVER_URL always
    #   wins over the auto-start in either mode.
    ocr_provider: Literal["gigachat", "standard"] = "standard"

    # DeepSeek API (OpenAI-compatible)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    # Transient-error retries (tenacity policy in src/llm_retry.py): 429, 5xx
    # and connection drops, with exponential backoff + jitter (Retry-After
    # honored). 0 disables — a single 429 becomes give_up.
    deepseek_connection_retries: int = Field(default=3, ge=0)
    # Initial backoff delay in seconds (doubles each retry, capped at 60s).
    deepseek_retry_backoff_factor: float = Field(default=1.0, ge=0)

    # GigaChat API (Sber). `credentials` is the base64 authorization key from
    # the developer portal; the SDK exchanges it for a 30-min access token and
    # refreshes automatically.
    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"  # PERS / B2B / CORP
    gigachat_model: str = "GigaChat-2-Pro"
    gigachat_base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    # The API is served with certs from the Russian Ministry of Digital
    # Development CA, absent from standard trust stores — verification is off
    # by default; install the CA and set true to enable.
    gigachat_verify_ssl_certs: bool = False
    # Same tenacity policy as DeepSeek, own knobs (the free PERS scope is
    # heavily rate-limited).
    gigachat_connection_retries: int = Field(default=3, ge=0)
    gigachat_retry_backoff_factor: float = Field(default=1.0, ge=0)
    # Max in-flight GigaChat Vision calls in the OCR sidecar
    # (src/vectordb/ocr_gigachat_server.py). GigaChat caps concurrent requests
    # per account — paid tiers lift token quotas, not this cap — and an
    # unbounded fan-out of page-OCR calls turns into 429 storms (including on
    # the OAuth endpoint). Raise it only if the account's plan allows more
    # simultaneous requests.
    gigachat_ocr_concurrency: int = Field(default=2, ge=1)
    # HTTP timeout (seconds) for the OCR sidecar's GigaChat client. The SDK
    # default is 30s — too short to transcribe a dense scanned book page
    # (thousands of output tokens). Ceiling is LiteParse's own hard 60s per
    # OCR request: anything above ~55s just moves where the failure fires.
    gigachat_ocr_timeout: float = Field(default=55.0, gt=0)

    # Agent loop
    max_iterations: int = Field(default=3, ge=1)

    # Structured-output generation: extra re-prompts when the provider's
    # function-calling returns a result that fails the schema's Pydantic
    # validation (missing/blank fields, cross-field rules). 0 disables retries.
    structured_max_retries: int = Field(default=1, ge=0)


general_settings = GeneralSettings()
