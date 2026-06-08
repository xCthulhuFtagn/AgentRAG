"""AgentRAGState — shared state across all LangGraph nodes.

TypedDict with Annotated reducers for accumulation across iterations.
"""

import json
import operator
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


# ── Structured output schemas (Pydantic — used by LLM.with_structured_output) ──

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
    """
    rationale: str = Field(description="Decide FIRST: why this collection is the right place to look for the needed piece.")
    collection: str = Field(description="The collection/table name to search (must match an available collection), chosen per the rationale.")
    subquery: str = Field(description="The focused thing to search for in that collection.")


class PlanResult(BaseModel):
    """Planner output: breakdown of the query into search routes."""
    is_multi_step: bool = Field(description="Whether this requires multiple search steps")
    steps: list[RouteStep] = Field(description="Search routes to execute")

    _coerce_steps = field_validator("steps", mode="before")(_coerce_list)


class OrchestratorResult(BaseModel):
    """Orchestrator output: complexity assessment.

    Schema-Guided Reasoning: the reasoning is generated BEFORE the is_complex
    verdict, so the boolean follows the analysis instead of being decided up
    front and rationalized afterward.
    """
    reasoning: str = Field(description="Analysis FIRST: does the query need multi-agent decomposition (multi-hop, multiple sources, planning) or can it be answered directly from one search?")
    is_complex: bool = Field(description="VERDICT, after the reasoning above: True if the query needs the multi-agent pipeline, False if a direct answer suffices.")


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
        description="Analysis FIRST: do the retrieved chunks contain a positive, substantive answer to every part of the question? State what is present and what is absent."
    )
    draft_answer: str = Field(
        default="",
        description="The best answer built ONLY from the retrieved context. If the context does not contain it, say so plainly — do not fill this in from general knowledge.",
    )
    missing_parts: list[str] = Field(
        default_factory=list,
        description="Concrete pieces still missing. Empty if the draft already answers the question.",
    )

    _coerce_missing = field_validator("missing_parts", mode="before")(_coerce_list)
    sufficient: bool = Field(
        description="The VERDICT, decided after reason/draft/missing above. True ONLY if draft_answer is a positive, substantive answer grounded in the retrieved chunks. A 'not found / absent / not mentioned' draft is NOT sufficient — set False while any plausibly-relevant collection is still unsearched."
    )
    feedback: str = Field(
        default="",
        description="LAST: only when sufficient is False — specific next-search instructions chosen to fill the gap (what to look for and which collection).",
    )


# ── Graph State (TypedDict with Annotated reducers) ──

class AgentRAGState(TypedDict):
    """Full state of the Agentic RAG graph.

    Fields with Annotated[list, operator.add] use concatenation reducer —
    updates are accumulated, not overwritten. Essential for iteration loops.
    """

    # Input
    query: str

    # Orchestrator
    is_complex: Optional[bool]
    orchestrator_reasoning: str

    # Vector DB scope — which LanceDB to search (per-project isolation).
    # None → global LANCE_DB_PATH (CLI default).
    db_path: Optional[str]

    # Planner
    plan_steps: list[dict]
    # Current route being processed (set by Planner)
    current_route: Optional[dict]

    # Query Rewriter
    rewritten_queries: Annotated[list[str], operator.add]
    # Current-turn search tasks: [{"collection": str|None, "query": str}].
    # Overwritten each turn (no reducer) — collection=None means "search all".
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
        is_complex=None,
        orchestrator_reasoning="",
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
