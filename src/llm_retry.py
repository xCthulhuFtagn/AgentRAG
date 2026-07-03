"""Tenacity retry layer for LLM calls — transient failures only, either provider.

A neutral module (no `agents` import) so both the agent graph and the
self-contained vectordb package (describe.py) share one retry policy.

What gets retried: 429 (our graph fires calls back-to-back — query_rewriter's
asyncio.gather can burst past the account's rate limit), 5xx, and
connection-level drops — from whichever provider is active (both exception
sets are checked unconditionally; `general_settings.llm_provider` only picks
which retry-count/backoff knobs apply, see `_retry_settings`). Anything else —
auth failures, other 4xx, schema validation — fails fast; retrying cannot fix
it. A 429's Retry-After header is honored, otherwise exponential backoff with
jitter. Exhausted retries reraise the original exception, which `llm_failsafe`
then turns into an honest give_up refusal.
"""

import logging

import httpx
from gigachat.exceptions import RateLimitError as GigaChatRateLimitError
from gigachat.exceptions import ServerError as GigaChatServerError
from openai import APIConnectionError, InternalServerError
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import general_settings

log = logging.getLogger("agentrag.llm")

TRANSIENT_LLM_ERRORS = (
    OpenAIRateLimitError,  # DeepSeek 429
    InternalServerError,  # DeepSeek 5xx
    APIConnectionError,  # DeepSeek connection drops, timeouts (wraps httpx errors)
    GigaChatRateLimitError,  # GigaChat 429
    GigaChatServerError,  # GigaChat 5xx
    httpx.TransportError,
)


def _retry_after_seconds(exc) -> float:
    """Seconds from a 429's Retry-After, or 0 if absent/unusable. Either provider."""
    if isinstance(exc, GigaChatRateLimitError):
        return max(0.0, exc.retry_after or 0)
    response = getattr(exc, "response", None)
    if response is None:
        return 0.0
    try:
        return max(0.0, float(response.headers.get("retry-after", 0)))
    except (TypeError, ValueError):  # HTTP-date form or garbage — ignore
        return 0.0


class _wait_rate_limit_aware(wait_exponential_jitter):
    """Exponential backoff with jitter, but honor a 429's Retry-After."""

    def __call__(self, retry_state) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, (OpenAIRateLimitError, GigaChatRateLimitError)):
            retry_after = _retry_after_seconds(exc)
            if retry_after > 0:
                return retry_after
        return super().__call__(retry_state)


def _retry_settings() -> tuple[int, float]:
    """(connection_retries, backoff_factor) for the active provider."""
    if general_settings.llm_provider == "gigachat":
        return (
            general_settings.gigachat_connection_retries,
            general_settings.gigachat_retry_backoff_factor,
        )
    return (
        general_settings.deepseek_connection_retries,
        general_settings.deepseek_retry_backoff_factor,
    )


async def ainvoke_with_retry(runnable, input_, **kwargs):
    """`runnable.ainvoke(input_)` under the transient-error retry policy.

    Built per call (not a decorator) so the settings are read at invocation
    time and the policy wraps any runnable — plain chat model or structured
    (with_structured_output) chain alike.
    """
    connection_retries, backoff_factor = _retry_settings()
    retrying = AsyncRetrying(
        retry=retry_if_exception_type(TRANSIENT_LLM_ERRORS),
        wait=_wait_rate_limit_aware(initial=backoff_factor, max=60),
        stop=stop_after_attempt(connection_retries + 1),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    result = None
    async for attempt in retrying:
        with attempt:
            result = await runnable.ainvoke(input_, **kwargs)
    return result
