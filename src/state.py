"""AgentRAGState — shared state across all LangGraph nodes.

TypedDict with Annotated reducers for accumulation across iterations.
"""

import json
import operator
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Structured output schemas (Pydantic — used by LLM.with_structured_output) ──
#
# Every constraint a structured result must satisfy is expressed as Pydantic
# validation: a violation raises ValidationError, which generate_structured()
# turns into a clarification re-prompt and, if the model keeps failing, an honest
# give_up. There is one uniform mechanism — no separate per-schema retry path.

def _coerce_list(v):
    """Coerce a list field that arrived as a JSON-encoded string back to a list.

    DeepSeek's function-calling sometimes serializes an array argument as a
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

    All three fields are strictly required and non-empty: a step DeepSeek emits
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
    First the analysis (reason), then the attempted answer (draft_answer), then
    the gaps (missing_parts); then the `sufficient` verdict — grounded in all of
    the above instead of committed up front and rationalized afterward (which let
    "not found" drafts pass as True). `feedback` comes LAST of all: deciding what
    to search next is a consequence of having concluded insufficiency, so it is
    chosen after the verdict, not before it.
    """
    reason: str = Field(
        description="СНАЧАЛА анализ: содержат ли найденные фрагменты позитивный содержательный ответ на каждую часть вопроса? Укажи, что присутствует и что отсутствует."
    )
    draft_answer: str = Field(
        default="",
        description="Лучший ответ, построенный ТОЛЬКО из найденного контекста. Если в контексте ответа нет — прямо так и напиши; не заполняй это поле из общих знаний.",
    )
    missing_parts: list[str] = Field(
        default_factory=list,
        description="Конкретные всё ещё недостающие фрагменты. Пусто, если черновик уже отвечает на вопрос.",
    )

    _coerce_missing = field_validator("missing_parts", mode="before")(_coerce_list)
    sufficient: bool = Field(
        description="ВЕРДИКТ, выносится ПОСЛЕ reason/draft/missing выше. True ТОЛЬКО если draft_answer — позитивный содержательный ответ, опирающийся на найденные фрагменты. Черновик вида «не найдено / отсутствует / не упоминается» НЕ достаточен — ставь False, пока остаётся необысканная правдоподобно-релевантная коллекция."
    )
    feedback: str = Field(
        default="",
        description="В САМОМ КОНЦЕ: только когда sufficient=False — конкретные указания для следующего поиска, закрывающие пробел (что искать и в какой коллекции).",
    )

    @model_validator(mode="after")
    def _verdict_must_be_actionable(self):
        # An "insufficient" verdict is only usable if it says what to do next:
        # the Planner re-routes on `feedback`, and give_up reports `missing_parts`.
        # If the judge said False but gave neither, reject it as a validation
        # error — generate_structured re-prompts with this message and, if the
        # model keeps failing, routes to give_up. Same uniform path as any other
        # schema violation; no separate retry mechanism.
        if not self.sufficient and not self.feedback.strip() and not self.missing_parts:
            raise ValueError(
                "Когда sufficient=false, ОБЯЗАТЕЛЬНО укажи missing_parts "
                "(конкретные недостающие фрагменты) и/или feedback (в какой "
                "коллекции искать дальше и с каким запросом). Иначе поставь "
                "sufficient=true, если найденный контекст действительно "
                "отвечает на вопрос."
            )
        return self


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
) -> AgentRAGState:
    """Create a clean initial state for the graph.

    db_path scopes vector search to one LanceDB (per-project isolation).
    None → global LANCE_DB_PATH (CLI default, backward-compatible).
    """
    return AgentRAGState(
        query=query,
        db_path=db_path,
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
