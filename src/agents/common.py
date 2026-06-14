"""Common utilities shared across all agents."""

import functools
import logging
from contextvars import ContextVar
from functools import lru_cache

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_gigachat.chat_models import GigaChat
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


# Transport/API errors from the GigaChat client count as LLM failures too
# (auth/token refresh failures, rate limits, 5xx, connection drops — the SDK
# raises GigaChatException, lower-level drops surface as httpx.HTTPError).
# Imported defensively so a missing package never breaks startup.
try:  # pragma: no cover - import guard
    import httpx
    from gigachat.exceptions import GigaChatException as _GigaChatException
    _LLM_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
        _GigaChatException,
        httpx.HTTPError,
    )
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


def format_inventory(described: list[dict], *, backtick_names: bool = False) -> str:
    """Render [{collection, description}] as the inventory block for prompts.

    backtick_names wraps each collection name in `…` — for Synthesis, whose
    output the web UI renders as markdown: a bare table name like
    07_Rodnaya_literatura gets its underscores eaten as italics. The model
    mirrors the formatting it sees, so the prompt shows names pre-backticked.
    The judge/planner prompts keep bare names (the planner must emit the exact
    table name in RouteStep.collection — backticks there would corrupt it).
    """
    if not described:
        return "(база знаний пуста — нет ни одной проиндексированной коллекции)"
    lines = []
    for c in described:
        name = f"`{c['collection']}`" if backtick_names else c["collection"]
        lines.append(f"- {name} — {c['description'] or '(без описания)'}")
    return "\n".join(lines)


async def get_inventory_str(db_path: str | None, *, backtick_names: bool = False) -> str:
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
    return format_inventory(described, backtick_names=backtick_names)


# ── Mechanical search statistics ─────────────────────────────────────────────
# Everything below is computed by code from the accumulated search_results —
# the model only reads it. A weak model cannot reliably reconstruct "what has
# been searched" from collection tags scattered across ~tens of kB of chunks
# (it hallucinates "not searched yet" for a collection searched on iteration 1),
# so the searched set, the last-search novelty delta and the coverage fraction
# are handed to it as ground truth. Split along the judge/planner contract:
# the judge gets searched× + last-search delta (an exhaustion detector — +0 new
# chunks = diminishing returns); the planner gets coverage K/N (a routing
# signal, never a verdict criterion — a low percentage must not read as
# "insufficient", vector search retrieves what's relevant, not everything).

def _topic_hits_in_chunks(
    chunks: list[str],
    seqs: list[int | None],
    new_seq_set: set[int],
    user_query: str,
) -> int | None:
    """How many NEW chunks mention the original user query.

    Walks aligned chunks/seqs; for each chunk whose seq is in *new_seq_set*,
    checks whether any content word (≥4 chars) from *user_query* appears as a
    substring.  Words ≥6 chars also contribute a stem prefix (``word[:-2]``) so
    Russian morphological variants are caught (e.g. «одноклеточные» stem
    «одноклеточн» matches «одноклеточных»).

    Returns the hit count, or ``None`` when *user_query* has no content words
    (too short / all stopwords) — the signal is then unavailable, and the
    display falls back to the raw ``+N`` novelty delta.
    """
    if not user_query or not chunks:
        return None
    query_words = [w.lower() for w in user_query.split() if len(w) >= 4]
    if not query_words:
        return None
    terms: set[str] = set()
    for w in query_words:
        terms.add(w)
        if len(w) >= 6:
            terms.add(w[:-2])  # stem prefix for morphological variants
    hits = 0
    for i, s in enumerate(seqs):
        if s is not None and s in new_seq_set and i < len(chunks):
            chunk_lower = chunks[i].lower()
            if any(term in chunk_lower for term in terms):
                hits += 1
    return hits


def collection_search_stats(
    search_results: list[dict], *, user_query: str | None = None
) -> dict[str, dict]:
    """Per-collection statistics over every search actually executed.

    Walks the accumulated entries in execution order (entries that errored are
    not searches and are skipped; empty results count — an empty search is the
    strongest exhaustion signal). Returns an insertion-ordered
    {collection: {searches, queries, retrieved, last_new, seqs_known,
    new_topic_hits, new_counts}} where
    `queries` lists the executed search queries in order (deduped — repeats
    are the angle-starvation signal the judge/rewriter must see), `retrieved`
    is the set of distinct chunk seqs seen so far, `last_new` is how many
    chunks of the LAST search were new vs all earlier searches of the same
    collection, `seqs_known=False` marks legacy tables without a seq
    column (novelty can't be deduplicated → last_new=None, coverage omitted),
    and `new_topic_hits`/`new_counts` are parallel per-search lists tracking
    how many NEW chunks per search mention *user_query* — a declining trend
    signals off-topic exhaustion even when the raw ``+N`` stays positive.
    """
    stats: dict[str, dict] = {}
    for r in search_results:
        collection = r.get("collection")
        if not collection or r.get("error"):
            continue
        st = stats.setdefault(
            collection,
            {
                "searches": 0,
                "queries": [],
                "retrieved": set(),
                "last_new": None,
                "seqs_known": True,
                "new_topic_hits": [],
                "new_counts": [],
            },
        )
        st["searches"] += 1
        query = (r.get("subquery") or "").strip()
        if query and query not in st["queries"]:
            st["queries"].append(query)
        chunks = r.get("chunks") or []
        seqs = {s for s in (r.get("seqs") or []) if s is not None}
        if chunks and not seqs:
            st["seqs_known"] = False
            st["last_new"] = None
            continue
        new = seqs - st["retrieved"]
        st["last_new"] = len(new)
        st["retrieved"] |= new

        # Per-search topic-hit trend: of the chunks NEW in this search, how
        # many mention the original user query?  A falling trend means the
        # collection is scraping increasingly off-topic parts of the document
        # — exhaustion the raw +N cannot see.
        if user_query:
            seq_list = [s for s in (r.get("seqs") or []) if s is not None]
            th = _topic_hits_in_chunks(chunks, seq_list, new, user_query)
            if th is not None:
                st["new_topic_hits"].append(th)
                st["new_counts"].append(len(new))
    return stats


def _ru_times(n: int) -> str:
    """«1 раз», «2 раза», «5 раз» — searches counter."""
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} раза"
    return f"{n} раз"


def _ru_new_chunks(n: int) -> str:
    """«+1 новый чанк», «+3 новых чанка», «+0 новых чанков» — novelty delta."""
    if n % 10 == 1 and n % 100 != 11:
        return f"+{n} новый чанк"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"+{n} новых чанка"
    return f"+{n} новых чанков"


def format_search_stats_for_judge(stats: dict[str, dict]) -> str:
    """The judge's view: searched collections, topic-hit trend, executed queries.

    When ``user_query`` was passed to `collection_search_stats`, a per-search
    topic-hit trend is shown («прирост по теме: 12/15 → 3/13 → 1/3») so the
    judge can see the *dynamics* — a declining trend signals off-topic exhaustion
    even when every ``+N`` is non-zero.  Without a query, falls back to the raw
    ``+N`` novelty delta.
    """
    if not stats:
        return "(поисков ещё не было)"
    lines = []
    for collection, st in stats.items():
        line = f"- {collection}: обыскана {_ru_times(st['searches'])}"
        th = st.get("new_topic_hits", [])
        nc = st.get("new_counts", [])
        if th and nc and len(th) == len(nc):
            if len(th) >= 2:
                # Multi-search trend — the arrow chain shows dynamics.
                parts = [f"{th[i]}/{nc[i]}" for i in range(len(th))]
                line += f"\n  прирост по теме: {' → '.join(parts)}"
            else:
                # Single search — no trend to read, just the count.
                line += (
                    f", +{nc[0]} новых чанков"
                    f" (по теме: {th[0]})"
                )
        elif st["last_new"] is not None:
            # Fallback: no query given or query had no content words.
            line += f", последний поиск дал {_ru_new_chunks(st['last_new'])}"
        if st["queries"]:
            line += "\n  выполненные запросы: " + "; ".join(
                f"«{q}»" for q in st["queries"]
            )
        lines.append(line)
    return "\n".join(lines)


def format_search_stats_for_planner(
    stats: dict[str, dict], totals: dict[str, int | None]
) -> str:
    """The planner's view: searches, coverage «извлечено K/N (P%)», novelty delta.

    The delta is the planner's stop signal — «+0 новых» says a repeat visit
    with similar wording is wasted; coverage alone misleads (a weak model reads
    low coverage as "barely explored → dig the same spot again").
    """
    if not stats:
        return "(пока нигде)"
    lines = []
    for collection, st in stats.items():
        line = f"- {collection}: обыскана {_ru_times(st['searches'])}"
        total = totals.get(collection)
        if total and st["seqs_known"]:
            k = len(st["retrieved"])
            line += f", извлечено {k}/{total} чанков ({round(100 * k / total)}%)"
        if st["last_new"] is not None:
            line += f", последний поиск дал {_ru_new_chunks(st['last_new'])}"
        lines.append(line)
    return "\n".join(lines)


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
    produce a usable structured result after retries) and GigaChat API/transport
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
def get_llm(temperature: float = 0.0, model: str | None = None) -> GigaChat:
    """Get a configured GigaChat LLM instance (cached).

    GigaChat's API rejects temperature=0 (the allowed range is (0, 2]); the
    documented way to get deterministic output is top_p=0, so a non-positive
    `temperature` is translated to that instead of being sent.
    """
    sampling = {"temperature": temperature} if temperature > 0 else {"top_p": 0.0}
    return GigaChat(
        model=model or general_settings.gigachat_model,
        credentials=general_settings.gigachat_credentials,
        scope=general_settings.gigachat_scope,
        base_url=general_settings.gigachat_base_url,
        verify_ssl_certs=general_settings.gigachat_verify_ssl_certs,
        callbacks=[_token_handler],
        **sampling,
    )


def get_structured_llm(schema, temperature: float = 0.0):
    """LLM that returns a validated Pydantic object.

    Uses method="function_calling" — GigaChat's native function-calling path,
    supported by langchain-gigachat. Kept explicit so every structured node
    goes through the same mechanism regardless of library defaults.
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
