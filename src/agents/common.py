"""Common utilities shared across all agents."""

import asyncio
import functools
import logging
from contextvars import ContextVar
from functools import lru_cache

from langchain_core.callbacks import AsyncCallbackHandler
from pydantic import BaseModel, Field
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


def render_search_context(
    search_results: list[dict], *, backtick_names: bool = False
) -> str:
    """Render accumulated search_results as prompt context — deduped, grouped.

    Both the judge and Synthesis used to render one block per EXECUTED SEARCH,
    so a chunk re-fetched by a later iteration's rewritten query (a near-repeat
    of an earlier one, or one that overlaps via neighbor stitching) appeared in
    the prompt once per search that returned it — wasted tokens that grow with
    every iteration. Here chunks are grouped by collection and deduplicated by
    seq (legacy tables without seq dedupe by exact text), sorted into document
    order, with the collection's executed queries listed once above its chunks.

    backtick_names wraps each collection name in `…` for Synthesis, whose
    output the web UI renders as markdown (see format_inventory).
    """
    chunked = [r for r in search_results if r.get("chunks")]
    if not chunked:
        return ""

    groups: dict[str, dict] = {}
    for r in chunked:
        collection = r.get("collection") or "?"
        g = groups.setdefault(
            collection,
            {"queries": [], "seen_seq": set(), "seen_text": set(), "items": []},
        )
        query = (r.get("subquery") or "").strip()
        if query and query not in g["queries"]:
            g["queries"].append(query)
        chunks = r.get("chunks", [])
        seqs = r.get("seqs") or []
        for i, chunk in enumerate(chunks):
            seq = seqs[i] if i < len(seqs) else None
            if seq is not None:
                if seq in g["seen_seq"]:
                    continue
                g["seen_seq"].add(seq)
            else:
                if chunk in g["seen_text"]:
                    continue
                g["seen_text"].add(chunk)
            g["items"].append((seq, chunk))

    blocks = []
    for collection, g in groups.items():
        name = f"`{collection}`" if backtick_names else collection
        # seq-known chunks in document order; seq-unknown (legacy) chunks
        # follow in first-seen order.
        known = sorted((it for it in g["items"] if it[0] is not None), key=lambda it: it[0])
        unknown = [it for it in g["items"] if it[0] is None]
        chunks_str = "\n---\n".join(
            f"[seq={seq}] {text}" if seq is not None else text
            for seq, text in known + unknown
        )
        queries_str = "; ".join(f"«{q}»" for q in g["queries"]) or "(без запроса)"
        blocks.append(
            f"\n### Коллекция: {name}\nЗапросы: {queries_str}\n{chunks_str}\n"
        )
    return "\n".join(blocks)


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

def _topic_hits_from_relevant(new_seq_set, seqs, chunks, relevant_flags):
    """Count NEW chunks the LLM marked relevant, out of NEW chunks it could assess.

    *relevant_flags* is a list of bool|None aligned with *chunks*/*seqs* by
    index, produced by `assess_chunks_relevance` (None = the assessment call
    itself failed — unknown, not "irrelevant"). Returns (hits, evaluated):
    hits = new chunks flagged True; evaluated = new chunks whose flag is not
    None. A chunk the LLM couldn't assess is excluded from both, so a
    transient API error doesn't dilute the topic-hit trend with a false miss.
    Returns (None, None) when *relevant_flags* is absent entirely (reranking
    was disabled for that search) — the trend display then falls back to the
    raw ``+N``.
    """
    if not relevant_flags:
        return None, None
    hits = 0
    evaluated = 0
    for i, s in enumerate(seqs):
        if s is not None and s in new_seq_set and i < len(relevant_flags):
            flag = relevant_flags[i]
            if flag is None:
                continue
            evaluated += 1
            if flag:
                hits += 1
    return hits, evaluated


# ── LLM per-chunk relevance assessment ─────────────────────────────────────
# When reranking is enabled (project setting), search_fanout calls the LLM for
# every retrieved chunk to judge relevance to the original query.  The results
# ride in search_results as a "relevant" list (bools, aligned with chunks).
# The prompt is deliberately minimal — a one-sentence yes/no with a low
# temperature for speed and consistency.

class ChunkRelevanceResult(BaseModel):
    """Оценка релевантности фрагмента вопросу — содержит ли полезную информацию."""
    relevant: bool = Field(
        description=(
            "True если фрагмент содержит информацию, полезную для ответа "
            "на вопрос (пусть даже косвенно или частично). "
            "False если фрагмент на другую тему или не помогает ответить."
        )
    )


_CHUNK_RELEVANCE_PROMPT = """Оцени, содержит ли фрагмент информацию, полезную для ответа на вопрос.
Отвечай «да» (relevant=true), только если фрагмент действительно по теме вопроса.

Вопрос: {query}

Фрагмент:
{chunk}"""


async def _assess_one(chunk: str, query: str, llm) -> bool | None:
    """Assess a single chunk → bool, or None if the assessment itself failed.

    None (not False) on failure: a transport error or empty tool call means we
    never actually judged the chunk, so it must not be treated as "found
    irrelevant" — that would let a transient API hiccup silently drop good
    context when reranking_remove_irrelevant is on.
    """
    try:
        prompt = _CHUNK_RELEVANCE_PROMPT.format(query=query, chunk=chunk)
        result = await ainvoke_with_retry(llm, prompt)
        return bool(result.relevant) if result else None
    except Exception:
        return None


async def assess_chunks_relevance(
    chunks: list[str], query: str
) -> list[bool | None]:
    """Assess each chunk's relevance to *query* via parallel LLM calls.

    One structured-LLM instance is shared across chunks (cached by schema type).
    Each call uses `ainvoke_with_retry` (tenacity) — transient transport errors
    retry; on any final failure the chunk's assessment is None (unknown), never
    False (which downstream code would read as "confirmed irrelevant").
    All chunks are assessed concurrently via `asyncio.gather`.
    """
    if not chunks:
        return []
    llm = get_structured_llm(ChunkRelevanceResult, temperature=0.0)
    tasks = [_assess_one(c, query, llm) for c in chunks]
    return list(await asyncio.gather(*tasks))


def collection_search_stats(
    search_results: list[dict],
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
    how many NEW chunks per search were LLM-assessed as relevant — a declining
    trend signals off-topic exhaustion even when the raw ``+N`` stays positive.
    When a search result has no ``relevant`` field (reranking disabled), the
    lists stay empty and the display falls back to raw ``+N``.
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

        # Per-search topic-hit trend: of the NEW chunks the LLM could actually
        # assess, how many were relevant?  A falling trend means the collection
        # is scraping increasingly off-topic parts of the document — exhaustion
        # the raw +N cannot see.  No "relevant" field → reranking was disabled
        # → skip (the display falls back to raw +N).
        seq_list = [s for s in (r.get("seqs") or []) if s is not None]
        th, evaluated = _topic_hits_from_relevant(
            new, seq_list, chunks, r.get("relevant")
        )
        if th is not None:
            st["new_topic_hits"].append(th)
            st["new_counts"].append(evaluated)
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
