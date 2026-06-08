"""Common utilities shared across all agents."""

import functools
import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.config import general_settings

log = logging.getLogger("agentrag.node")


def logged_node(fn):
    """Wrap a graph node so its trace entries are emitted as log records.

    Every node already builds `trace` entries via `make_trace_entry` and returns
    them in its `Command.update` (or plain dict). This decorator is the single
    point that turns those entries into logs — node bodies stay untouched, and
    the same logs appear under the CLI and the web app.
    """

    @functools.wraps(fn)
    async def wrapper(state, **kwargs):
        result = await fn(state, **kwargs)
        update = result.update if isinstance(result, Command) else result
        if isinstance(update, dict):
            for entry in update.get("trace", []):
                detail = entry.get("detail", "")
                suffix = f" — {detail[:200]}" if detail else ""
                log.info("[%s] %s%s", entry["agent"], entry["decision"], suffix)
        return result

    return wrapper


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0, model: str | None = None) -> ChatOpenAI:
    """Get a configured DeepSeek LLM instance (cached).

    Uses OpenAI-compatible endpoint at api.deepseek.com/v1.
    """
    return ChatOpenAI(
        model=model or general_settings.deepseek_model,
        api_key=general_settings.deepseek_api_key,
        base_url=general_settings.deepseek_base_url,
        temperature=temperature,
    )


def get_structured_llm(schema, temperature: float = 0.0):
    """LLM that returns a validated Pydantic object.

    Uses method="function_calling" — DeepSeek's API does not support the
    json_schema `response_format` that with_structured_output() picks by
    default (returns 'This response_format type is unavailable now').
    Function calling is the OpenAI-compatible path DeepSeek does support.
    """
    return get_llm(temperature).with_structured_output(
        schema, method="function_calling"
    )
