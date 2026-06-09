"""Configuration — DeepSeek API and agent-loop settings.

Vector DB settings live in src/vectordb/config.py. Access values via the
`general_settings` instance (e.g. general_settings.deepseek_model).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeneralSettings(BaseSettings):
    """DeepSeek API + agent-loop knobs, read from .env / process env.

    Env var names are the UPPERCASE field names (case-insensitive),
    e.g. DEEPSEEK_API_KEY, MAX_ITERATIONS.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

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

    # Agent loop
    max_iterations: int = Field(default=3, ge=1)

    # Structured-output generation: extra re-prompts when DeepSeek's
    # function-calling returns a result that fails the schema's Pydantic
    # validation (missing/blank fields, cross-field rules). 0 disables retries.
    structured_max_retries: int = Field(default=1, ge=0)


general_settings = GeneralSettings()
