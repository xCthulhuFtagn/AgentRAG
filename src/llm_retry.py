"""Tenacity retry layer for GigaChat calls — transient failures only.

A neutral module (no `agents` import) so both the agent graph and the
self-contained vectordb package (describe.py) share one retry policy.

What gets retried: 429 (the free PERS scope allows one concurrent request and
our graph fires calls back-to-back), 5xx, and connection-level drops. Anything
else — auth failures, other 4xx, schema validation — fails fast; retrying
cannot fix it. A 429's Retry-After header is honored, otherwise exponential
backoff with jitter. Exhausted retries reraise the original exception, which
`llm_failsafe` then turns into an honest give_up refusal.
"""

import logging

import httpx
from gigachat.exceptions import RateLimitError, ServerError
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import general_settings

log = logging.getLogger("agentrag.llm")

TRANSIENT_LLM_ERRORS = (RateLimitError, ServerError, httpx.TransportError)


class _wait_rate_limit_aware(wait_exponential_jitter):
    """Exponential backoff with jitter, but honor a 429's Retry-After."""

    def __call__(self, retry_state) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, RateLimitError) and exc.retry_after > 0:
            return exc.retry_after
        return super().__call__(retry_state)


async def ainvoke_with_retry(runnable, input_, **kwargs):
    """`runnable.ainvoke(input_)` under the transient-error retry policy.

    Built per call (not a decorator) so the settings are read at invocation
    time and the policy wraps any runnable — plain chat model or structured
    (with_structured_output) chain alike.
    """
    retrying = AsyncRetrying(
        retry=retry_if_exception_type(TRANSIENT_LLM_ERRORS),
        wait=_wait_rate_limit_aware(
            initial=general_settings.gigachat_retry_backoff_factor, max=60
        ),
        stop=stop_after_attempt(general_settings.gigachat_connection_retries + 1),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    result = None
    async for attempt in retrying:
        with attempt:
            result = await runnable.ainvoke(input_, **kwargs)
    return result
