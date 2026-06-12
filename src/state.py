"""AgentRAGState — shared state across all LangGraph nodes.

TypedDict with Annotated reducers for accumulation across iterations.
"""

import json
import operator
import re
from typing import Annotated, Any, Optional, Sequence, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Structured output schemas (Pydantic — used by LLM.with_structured_output) ──
#
# Every constraint a structured result must satisfy is expressed as Pydantic
# validation: a violation raises ValidationError, which generate_structured()
# turns into a clarification re-prompt and, if the model keeps failing, an honest
# give_up. There is one uniform mechanism — no separate per-schema retry path.

def _coerce_list(v):
    """Coerce a list field that arrived as a JSON-encoded string back to a list.

    A weak model's function-calling sometimes serializes an array argument as a
    string (e.g. missing_parts='["a","b"]' instead of ["a","b"]), which fails
    Pydantic's list validation and crashes the whole graph run. Parse the string
    as JSON when possible; a non-JSON string becomes a single-item list.
    """
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else [s]
        except (ValueError, TypeError):
            return [s]
    return v

class RouteStep(BaseModel):
    """A single search route from the Planner.

    Schema-Guided Reasoning: the rationale (why this source) is generated BEFORE
    the collection/subquery it justifies, so the choice follows the reasoning
    instead of being rationalized after the fact.

    All three fields are strictly required and non-empty: a step the model emits
    with a missing or blank collection/subquery (e.g. only 'rationale') is not a
    usable route, so it raises ValidationError. generate_structured then re-prompts
    with the error and, if the model keeps failing, routes to give_up — the same
    strict path used for the scalar verdicts, rather than silently dropping steps.
    """
    rationale: str = Field(description="СНАЧАЛА реши: почему именно эта коллекция — правильное место для поиска нужного фрагмента.")
    collection: str = Field(description="Имя коллекции/таблицы для поиска (должно совпадать с одной из доступных коллекций), выбранное согласно rationale.")
    subquery: str = Field(description="Что конкретно искать в этой коллекции (сфокусированный подзапрос).")

    @field_validator("collection", "subquery")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        # `required` alone accepts "" / whitespace (a valid str); reject those too
        # so a semantically-empty route fails validation instead of searching a
        # nonexistent "" collection.
        if not v or not v.strip():
            raise ValueError("поле должно быть непустой строкой")
        return v


class PlanResult(BaseModel):
    """Planner output: breakdown of the query into search routes."""
    is_multi_step: bool = Field(description="Требует ли вопрос нескольких шагов поиска")
    steps: list[RouteStep] = Field(description="Поисковые маршруты для выполнения")

    _coerce_steps = field_validator("steps", mode="before")(_coerce_list)


class SufficientContextResult(BaseModel):
    """Sufficient Context Agent output.

    Schema-Guided Reasoning: fields follow the order a person would reason in,
    because structured output is generated field-by-field in declaration order.
    First `question_verbatim` — a literal COPY of the user's question (copying is
    reliable even for weak models, unlike generating sub-questions), re-anchoring
    the judge on the question as asked after ~tens of kB of chunks inflated its
    idea of what was asked. Then the analysis (reason), the attempted answer
    (draft_answer), the gaps (missing_parts); then the `sufficient` verdict —
    grounded in all of the above instead of committed up front and rationalized
    afterward (which let "not found" drafts pass as True). `feedback` comes LAST
    of all: describing the information gap is a consequence of having concluded
    insufficiency, so it is written after the verdict, not before it.

    Separation of concerns: the judge speaks the language of INFORMATION (what
    was asked, what was found, which fact is missing). Routing — which collection
    to search next — belongs to the Planner, so collection names are banned from
    `feedback`/`missing_parts` (enforced by make_sufficient_context_schema, which
    knows the inventory).

    Verdict semantics: `sufficient` asks "would one more search of THIS corpus
    materially improve the answer?" — a retrieval-state call, not a grade
    against an ideal answer. An exhausted corpus whose findings are thin (only
    passing mentions) is sufficient: the honest answer is "the sources contain
    only …". Zero findings are never sufficient — that path stays a refusal.
    """
    question_verbatim: str = Field(
        description="САМОЕ ПЕРВОЕ: скопируй вопрос пользователя ДОСЛОВНО, символ в символ — ничего не добавляя, не сокращая и не перефразируя."
    )
    reason: str = Field(
        description="Анализ из двух частей: (1) что по вопросу из question_verbatim найденные фрагменты содержат, а чего в них нет — относительно вопроса, КАК ОН ЗАДАН, без добавления подвопросов, которых пользователь не задавал; (2) по статистике поисков — исчерпана ли база по теме вопроса, или есть конкретная причина ожидать от неё большего (необысканная коллекция, неиспробованные формулировки)."
    )
    draft_answer: str = Field(
        default="",
        description="Лучший ответ на вопрос из question_verbatim, построенный ТОЛЬКО из найденного контекста. Если по теме есть лишь отдельные упоминания — приведи их («в источниках об этом только …»): для исчерпанной базы это и есть ответ. Если не найдено вообще ничего — прямо так и напиши; не заполняй поле из общих знаний.",
    )
    missing_parts: list[str] = Field(
        default_factory=list,
        description="Конкретные недостающие ФАКТЫ — каждый элемент короткая именная группа (например «расшифровка аббревиатуры ЭВМ»). НЕ маршруты и НЕ имена коллекций («поиск в коллекции …» — неправильно), НЕ размытое («более подробная информация»). Пусто, если черновик уже отвечает на вопрос.",
    )

    _coerce_missing = field_validator("missing_parts", mode="before")(_coerce_list)
    sufficient: bool = Field(
        description="ВЕРДИКТ, выносится ПОСЛЕ полей выше; вопрос вердикта — даст ли ещё один поиск по базе что-то, заметно улучшающее ответ. True в двух случаях: (1) draft_answer отвечает на вопрос из question_verbatim, КАК ОН ЗАДАН, — даже если можно было бы найти «больше деталей»; (2) база ИСЧЕРПАНА по теме (все правдоподобно-релевантные коллекции обысканы, новые поиски не приносят нового по теме) и draft_answer собирает всё найденное, пусть и скудное. False — ТОЛЬКО при конкретной причине ожидать от базы большего: необысканная правдоподобная коллекция или неиспробованные формулировки в коллекции, ещё дающей новое. И всегда False, если по теме не найдено ВООБЩЕ ничего: пустота — не ответ."
    )
    feedback: str = Field(
        default="",
        description="В САМОМ КОНЦЕ: только когда sufficient=False. Опиши ИНФОРМАЦИОННЫЙ пробел; удобная форма: «Не хватает: …. Найдено вместо этого: …. Альтернативные формулировки: ….» НЕ называй коллекции и не указывай, где искать — выбор источника не твоя задача.",
    )

    @model_validator(mode="after")
    def _insufficient_verdict_is_actionable(self):
        # Validate only what CODE consumes; what an LLM consumes is guided by
        # the prompt, not policed. feedback's only consumer is the Planner
        # reading prose — but its PRESENCE is a code contract: the Planner's
        # iteration mode keys off bool(feedback), so an empty one at False
        # would silently demote it to initial-plan mode. The template in the
        # field description is guidance, deliberately NOT enforced: a label
        # check rejects semantically-good feedback worded differently, and a
        # re-prompt burned on formatting can escalate a correct verdict into
        # give_up. Checks needing node-time context (collection names, the
        # literal query) live in make_sufficient_context_schema.
        if self.sufficient:
            return self
        if not self.feedback.strip():
            raise ValueError(
                "Когда sufficient=false, ОБЯЗАТЕЛЬНО заполни feedback — опиши "
                "информационный пробел: какая информация отсутствует, что "
                "поиски дали вместо неё, какими ещё формулировками она может "
                "называться. Иначе поставь sufficient=true, если найденный "
                "контекст действительно отвечает на вопрос."
            )
        if any(not part or not part.strip() for part in self.missing_parts):
            raise ValueError(
                "каждый элемент missing_parts должен быть непустой строкой — "
                "короткой именной группой, называющей конкретный отсутствующий факт."
            )
        return self


def _normalize_question(s: str) -> str:
    """Normalize a question for the verbatim-copy comparison.

    Tolerates the trivial drift a weak model introduces when copying (case,
    ё/е, whitespace runs, wrapping quotes, trailing punctuation) while still
    rejecting paraphrase — the failure the check exists to catch.
    """
    s = (s or "").casefold().replace("ё", "е")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip("«»\"'“”‘’ ").rstrip("?.!…")
    return s.strip()


def make_sufficient_context_schema(
    collection_names: Sequence[str], query: str
) -> type[SufficientContextResult]:
    """SufficientContextResult bound to node-time context (inventory + query).

    Two constraints can only be checked with information the node has — the
    actual collection names and the literal user question — so they are baked
    into a dynamic subclass as validators. That keeps every requirement a
    Pydantic constraint flowing through the one uniform generate_structured
    re-prompt path: no separate node-level retry loop.

    - question_verbatim must be a (normalized-)literal copy of the user's query;
    - feedback/missing_parts of an insufficient verdict must not name a
      collection or prescribe where to search — the judge states the information
      gap, the Planner owns routing. Weak models violate this regularly, hence a
      mechanical check rather than a prompt-only rule.
    """
    # Tiny names (1–2 chars) would false-positive on ordinary words; routing
    # leakage that matters always quotes a recognizable table name.
    banned = sorted(
        {n.strip().casefold() for n in collection_names if n and len(n.strip()) >= 3}
    )
    expected_question = _normalize_question(query)

    class BoundSufficientContextResult(SufficientContextResult):
        @model_validator(mode="after")
        def _question_copied_verbatim(self):
            if _normalize_question(self.question_verbatim) != expected_question:
                raise ValueError(
                    "question_verbatim должен быть ДОСЛОВНОЙ копией вопроса "
                    f"пользователя: «{query}» — скопируй его без изменений, "
                    "ничего не добавляя и не перефразируя."
                )
            return self

        @model_validator(mode="after")
        def _no_routes_in_information_fields(self):
            if self.sufficient:
                return self
            fields = [("feedback", self.feedback)]
            fields += [("missing_parts", part) for part in self.missing_parts]
            for field_name, text in fields:
                lowered = (text or "").casefold()
                for name in banned:
                    if name in lowered:
                        raise ValueError(
                            f"в поле {field_name} указан маршрут вместо "
                            f"информационного пробела: «{name}» — это имя "
                            "коллекции. НЕ называй коллекции и не указывай, где "
                            "искать (это решает планировщик); опиши, КАКОЙ ФАКТ "
                            "отсутствует."
                        )
            return self

    # The class name is the function-calling tool name the model sees (and the
    # name quoted in StructuredGenerationError) — keep it stable across binds.
    BoundSufficientContextResult.__name__ = SufficientContextResult.__name__
    BoundSufficientContextResult.__qualname__ = SufficientContextResult.__qualname__
    return BoundSufficientContextResult


# ── Graph State (TypedDict with Annotated reducers) ──

class AgentRAGState(TypedDict):
    """Full state of the Agentic RAG graph.

    Fields with Annotated[list, operator.add] use concatenation reducer —
    updates are accumulated, not overwritten. Essential for iteration loops.
    """

    # Input
    query: str

    # Vector DB scope — which LanceDB to search (per-project isolation).
    # None → global LANCE_DB_PATH (CLI default).
    db_path: Optional[str]

    # Per-project neighbor-stitching overrides for search_fanout:
    # {"expand_padding": int, "bridge_gap": int}. None / missing keys →
    # the global vdb_settings values (CLI default).
    stitch_settings: Optional[dict]

    # Planner
    plan_steps: list[dict]
    # Current route being processed (set by Planner)
    current_route: Optional[dict]

    # Query Rewriter
    rewritten_queries: Annotated[list[str], operator.add]
    # Current-turn search tasks: [{"collection": str, "query": str}], one per
    # planner route. Overwritten each turn (no reducer).
    search_tasks: list[dict]

    # Search Fanout
    search_results: Annotated[list[dict], operator.add]

    # Sufficient Context
    sufficient: Optional[bool]
    sufficient_reason: str
    feedback: str
    missing_parts: list[str]
    draft_answer: str

    # Iteration control
    iteration_count: int
    max_iterations: int

    # Final answer
    final_answer: str

    # Set by llm_failsafe when a node's LLM call fails unrecoverably — give_up
    # reads it to render an honest "model problem" refusal. "" = no LLM failure.
    llm_error: str

    # Audit trail
    trace: Annotated[list[dict], operator.add]


def make_initial_state(
    query: str,
    max_iterations: int = 3,
    db_path: Optional[str] = None,
    stitch_settings: Optional[dict] = None,
) -> AgentRAGState:
    """Create a clean initial state for the graph.

    db_path scopes vector search to one LanceDB (per-project isolation).
    None → global LANCE_DB_PATH (CLI default, backward-compatible).
    stitch_settings carries per-project neighbor-stitching overrides
    (expand_padding/bridge_gap) into search_fanout; None → vdb_settings.
    """
    return AgentRAGState(
        query=query,
        db_path=db_path,
        stitch_settings=stitch_settings,
        plan_steps=[],
        current_route=None,
        rewritten_queries=[],
        search_tasks=[],
        search_results=[],
        sufficient=None,
        sufficient_reason="",
        feedback="",
        missing_parts=[],
        draft_answer="",
        iteration_count=0,
        max_iterations=max_iterations,
        final_answer="",
        llm_error="",
        trace=[],
    )


def make_trace_entry(agent: str, decision: str, detail: str = "", info: str = "") -> dict:
    """Create a trace entry dict (appended to trace via operator.add reducer).

    `info` is optional human-facing context shown as the middle line of a step in
    the web UI (Planner rationale, searched source+query, judge reason/feedback);
    `detail` stays the compact log-line string. Newlines in `info` render as
    separate lines in the UI.
    """
    return {"agent": agent, "decision": decision, "detail": detail, "info": info}
