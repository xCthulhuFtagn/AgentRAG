"""Configuration — GigaChat API and agent-loop settings.

Vector DB settings live in src/vectordb/config.py. Access values via the
`general_settings` instance (e.g. general_settings.gigachat_model).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeneralSettings(BaseSettings):
    """GigaChat API + agent-loop knobs, read from .env / process env.

    Env var names are the UPPERCASE field names (case-insensitive),
    e.g. GIGACHAT_CREDENTIALS, MAX_ITERATIONS.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # GigaChat API (Sber). `credentials` is the base64 authorization key from
    # the developer portal; the SDK exchanges it for a 30-min access token and
    # refreshes automatically.
    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"  # PERS / B2B / CORP
    gigachat_model: str = "GigaChat-2-Max"
    gigachat_base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    # The API is served with certs from the Russian Ministry of Digital
    # Development CA, absent from standard trust stores — verification is off
    # by default; install the CA and set true to enable.
    gigachat_verify_ssl_certs: bool = False
    # Transient-error retries (tenacity policy in src/llm_retry.py): 429 (the
    # free PERS scope is heavily rate-limited and the graph fires calls
    # back-to-back), 5xx and connection drops, with exponential backoff +
    # jitter (Retry-After honored). 0 disables — a single 429 becomes give_up.
    gigachat_max_retries: int = Field(default=3, ge=0)
    # Initial backoff delay in seconds (doubles each retry, capped at 60s).
    gigachat_retry_backoff_factor: float = Field(default=1.0, ge=0)

    # Agent loop
    max_iterations: int = Field(default=3, ge=1)

    # Structured-output generation: extra re-prompts when GigaChat's
    # function-calling returns a result that fails the schema's Pydantic
    # validation (missing/blank fields, cross-field rules). 0 disables retries.
    structured_max_retries: int = Field(default=1, ge=0)


general_settings = GeneralSettings()
