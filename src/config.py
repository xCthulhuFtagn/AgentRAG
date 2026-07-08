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

    # Agent loop
    max_iterations: int = Field(default=3, ge=1)

    # Structured-output generation: extra re-prompts when the provider's
    # function-calling returns a result that fails the schema's Pydantic
    # validation (missing/blank fields, cross-field rules). 0 disables retries.
    structured_max_retries: int = Field(default=1, ge=0)


general_settings = GeneralSettings()
