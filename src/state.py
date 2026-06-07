"""AgentRAGState — shared state across all LangGraph nodes.

TypedDict with Annotated reducers for accumulation across iterations.
"""

import operator
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field


# ── Structured output schemas (Pydantic — used by LLM.with_structured_output) ──

class RouteStep(BaseModel):
    """A single search route from Planner."""
    collection: str = Field(description="LanceDB collection/table name to search")
    subquery: str = Field(description="What to search for in this collection")
    rationale: str = Field(description="Why this collection is relevant")


class PlanResult(BaseModel):
    """Planner output: breakdown of the query into search routes."""
    is_multi_step: bool = Field(description="Whether this requires multiple search steps")
    steps: list[RouteStep] = Field(description="Search routes to execute")


class OrchestratorResult(BaseModel):
    """Orchestrator output: complexity assessment."""
    is_complex: bool = Field(description="Whether the query needs multi-agent decomposition")
    reasoning: str = Field(description="Brief explanation of the decision")


class SufficientContextResult(BaseModel):
    """Sufficient Context Agent output."""
    sufficient: bool = Field(description="Is there enough context to answer?")
    reason: str = Field(description="Why context is sufficient or not")
    feedback: str = Field(
        default="",
        description="Specific search instructions if insufficient: what to look for and where",
    )
    missing_parts: list[str] = Field(
        default_factory=list,
        description="Concrete pieces of information still missing",
    )
    draft_answer: str = Field(
        default="",
        description="Draft answer based on current context",
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


def make_trace_entry(agent: str, decision: str, detail: str = "") -> dict:
    """Create a trace entry dict (appended to trace via operator.add reducer)."""
    return {"agent": agent, "decision": decision, "detail": detail}
