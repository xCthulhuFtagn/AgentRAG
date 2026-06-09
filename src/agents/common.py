"""Common utilities shared across all agents."""

import functools
import logging
from contextvars import ContextVar
from functools import lru_cache

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.config import general_settings
from src.llm_retry import ainvoke_with_retry
from src.state import make_trace_entry
from src.vectordb.tools import list_collections_described

log = logging.getLogger("agentrag.node")


# ── LLM failure → honest refusal ─────────────────────────────────────────────
# When the model can't produce a usable result (validation keeps failing, the
# API errors out, or no tool call comes back), we route to give_up rather than
# crash the graph — matching the system's "honest refusal" design.

class StructuredGenerationError(RuntimeError):
    """The LLM could not produce a usable structured result after all retries."""


# Transport/API errors from the OpenAI-compatible client count as LLM failures
# too (rate limits, 5xx, connection drops). Imported defensively so a missing
# openai package never breaks startup.
try:  # pragma: no cover - import guard
    from openai import APIError as _OpenAIAPIError
    _LLM_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (_OpenAIAPIError,)
except Exception:  # pragma: no cover
    _LLM_TRANSPORT_ERRORS = ()

# What llm_failsafe treats as "the model failed" → give_up (not a code bug).
LLM_FAILURE_ERRORS: tuple[type[BaseException], ...] = (
    StructuredGenerationError,
) + _LLM_TRANSPORT_ERRORS


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
        return "(база знаний пуста — нет ни одной проиндексированной коллекции)"
    return "\n".join(
        f"- {c['collection']} — {c['description'] or '(без описания)'}"
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


def llm_failsafe(node_name: str):
    """Wrap a node so an unrecoverable LLM failure routes to give_up, not a crash.

    Catches `LLM_FAILURE_ERRORS` — StructuredGenerationError (the model couldn't
    produce a usable structured result after retries) and OpenAI API/transport
    errors (raised by the plain-LLM nodes). Anything else propagates: a code bug
    must not be silently masked as "the model failed". On catch, redirects to
    give_up with `llm_error` set so the refusal honestly cites the model problem.

    NOT applied to give_up itself — it uses no LLM, and redirecting it to itself
    would loop.
    """

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(state, **kwargs):
            try:
                return await fn(state, **kwargs)
            except LLM_FAILURE_ERRORS as e:
                log.warning("[%s] LLM failure → give_up: %s", node_name, e)
                trace_entry = make_trace_entry(
                    agent=node_name,
                    decision="llm_failure → give_up",
                    detail=str(e),
                )
                return Command(
                    goto="give_up",
                    update={"llm_error": str(e), "trace": [trace_entry]},
                )

        return wrapper

    return deco


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


async def generate_structured(schema, prompt: str, *, temperature: float = 0.0):
    """Invoke a structured LLM, retrying with clarification, else fail loudly.

    Every requirement on the result is a Pydantic constraint — required fields,
    non-empty strings, cross-field rules (model_validator). A violation raises
    ValidationError, exactly like a transport error or a missing tool call. We
    re-prompt with the specific deficiency up to
    `general_settings.structured_max_retries` times; if the model still can't
    satisfy the schema, raise StructuredGenerationError, which `llm_failsafe`
    turns into an honest give_up refusal instead of letting the graph crash.

    Two retry layers with distinct jobs: ainvoke_with_retry (tenacity) absorbs
    transient transport errors (429/5xx/drops) inside each attempt; this loop
    re-prompts only on semantic failures the model can actually correct.
    """
    retries = general_settings.structured_max_retries
    llm = get_structured_llm(schema, temperature)
    current = prompt
    for attempt in range(retries + 1):
        last = attempt >= retries
        try:
            result = await ainvoke_with_retry(llm, current)
            if result is None:  # no tool call came back
                raise ValueError("the model returned no structured output")
            return result
        except Exception as e:  # ValidationError, API/transport errors, no tool call
            if last:
                raise StructuredGenerationError(
                    f"{schema.__name__}: the model failed to produce a valid "
                    f"response after {retries + 1} attempt(s) — {type(e).__name__}: {e}"
                ) from e
            current = (
                f"{prompt}\n\nТвой предыдущий ответ не прошёл проверку:\n{e}\n\n"
                "Верни исправленный ответ — вызови функцию так, чтобы каждое "
                "обязательное поле присутствовало, имело правильный тип и "
                "удовлетворяло правилам, описанным в схеме."
            )
