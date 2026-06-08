"""Common utilities shared across all agents."""

import functools
import logging
from contextvars import ContextVar
from functools import lru_cache

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.config import general_settings
from src.vectordb.tools import list_collections_described

log = logging.getLogger("agentrag.node")


# ── Per-node token accounting ────────────────────────────────────────────────
# logged_node sets a fresh sink ({"input", "output"}) before each node runs; the
# callback below adds every LLM call's usage into whatever sink is current. A
# ContextVar (not a global) so concurrent graph runs accumulate independently,
# and so asyncio.gather subtasks — which copy the context at creation — land
# their tokens in the same sink the parent node set before the gather.
_token_sink: ContextVar[dict | None] = ContextVar("agentrag_token_sink", default=None)


class _TokenUsageHandler(AsyncCallbackHandler):
    """Adds each LLM call's token usage to the current node's sink.

    Attached to every LLM instance so it fires for plain AND structured
    (function-calling) calls — structured calls return a parsed Pydantic object
    with no usage_metadata, so reading the message wouldn't catch them; the
    callback hooks the underlying LLM and does.
    """

    async def on_llm_end(self, response, **kwargs) -> None:
        sink = _token_sink.get()
        if sink is None:
            return
        inp = out = 0
        usage = (response.llm_output or {}).get("token_usage") if response.llm_output else None
        if usage:
            inp = usage.get("prompt_tokens", 0) or 0
            out = usage.get("completion_tokens", 0) or 0
        else:
            # Fallback: sum usage_metadata across generations (e.g. streaming).
            for gens in response.generations:
                for gen in gens:
                    um = getattr(getattr(gen, "message", None), "usage_metadata", None)
                    if um:
                        inp += um.get("input_tokens", 0) or 0
                        out += um.get("output_tokens", 0) or 0
        sink["input"] += inp
        sink["output"] += out


_token_handler = _TokenUsageHandler()


async def get_inventory_str(db_path: str | None) -> str:
    """The full corpus inventory — every collection plus its description.

    Ground truth for "what's in the knowledge base": the complete list of
    collections that exist (with their index-time summaries). Given to the
    Sufficient Context judge so it can confirm completeness on "describe all
    files"-type queries — vector search returns similar chunks but never proves
    it has seen every document, so without this the judge can never satisfy an
    "all/every/complete" request and the loop always ends in give_up. Also given
    to Synthesis so it can describe each file from its summary.
    """
    described = await list_collections_described(db_path)
    if not described:
        return "(knowledge base is empty — no collections indexed)"
    return "\n".join(
        f"- {c['collection']} — {c['description'] or '(no description)'}"
        for c in described
    )


def logged_node(fn):
    """Wrap a graph node so its trace entries are emitted as log records.

    Every node already builds `trace` entries via `make_trace_entry` and returns
    them in its `Command.update` (or plain dict). This decorator is the single
    point that turns those entries into logs — node bodies stay untouched, and
    the same logs appear under the CLI and the web app.

    It also meters token usage: a fresh sink is installed for the node's run, the
    LLM callback fills it, and the totals are stamped onto each trace entry
    (`input_tokens`/`output_tokens`) so the web UI can show per-step cost.
    """

    @functools.wraps(fn)
    async def wrapper(state, **kwargs):
        sink = {"input": 0, "output": 0}
        tok = _token_sink.set(sink)
        try:
            result = await fn(state, **kwargs)
        finally:
            _token_sink.reset(tok)

        update = result.update if isinstance(result, Command) else result
        if isinstance(update, dict):
            for entry in update.get("trace", []):
                entry["input_tokens"] = sink["input"]
                entry["output_tokens"] = sink["output"]
                detail = entry.get("detail", "")
                suffix = f" — {detail[:200]}" if detail else ""
                toks = (
                    f" [in={sink['input']} out={sink['output']}]"
                    if (sink["input"] or sink["output"])
                    else ""
                )
                log.info(
                    "[%s] %s%s%s", entry["agent"], entry["decision"], toks, suffix
                )
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
        callbacks=[_token_handler],
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
