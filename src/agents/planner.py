"""Planner Agent — breaks down query into search routes.

Always returns Command(goto="query_rewriter") or Command(goto="synthesis").
No Send — all routes processed together in query_rewriter.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, PlanResult, make_trace_entry
from src.agents.common import get_structured_llm
from src.vectordb.config import vdb_settings
from src.vectordb.tools import list_collections, list_collections_described

PLANNER_PROMPT = """You are the Planner Agent of an Agentic RAG system.

Your job: break down a complex query into specific search routes, each targeting
a specific document collection.

First, review the available collections (with a short description when available) to decide where to search.

Available collections:
{collections}

Then, for each piece of information needed, create a RouteStep with:
- collection: the exact table name to search (must match available collections)
- subquery: a focused search query for that specific piece
- rationale: why this collection is relevant

User query: {query}

If the query needs information from multiple collections, create multiple steps.
If the query can be answered from a single collection, create one step.
If no relevant collection exists, return an empty steps list."""

PLANNER_ITERATION_PROMPT = """You are the Planner Agent of an Agentic RAG system.

This is an ITERATION — earlier searches did not find everything needed. Re-route
the search to the collection(s) most likely to hold the MISSING information.

Original user question: {query}
Still missing: {missing_parts}
Feedback from the context checker: {feedback}

Available collections:
{collections}

Already searched (did NOT yield the answer): {searched}

For each collection that could plausibly contain a missing piece, create a RouteStep:
- collection: the exact table name (must match available collections)
- subquery: a focused query for the missing piece — use ALTERNATIVE keywords or a
  different angle than what was already tried
- rationale: why this collection might hold the missing piece

Strongly PREFER collections that have NOT been searched yet — re-routing to an
already-searched collection only makes sense with a genuinely different subquery.
Prefer the 1-3 most relevant collections. If no collection looks relevant,
return an empty steps list — the system will then broaden the search to all."""


async def planner_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Planner: create search routes, then command query_rewriter."""
    db_path = state.get("db_path")
    if vdb_settings.descriptions_enabled:
        described = await list_collections_described(db_path)
        lines = [
            f"- {c['collection']} — {c['description'] or '(no description)'}"
            for c in described
        ]
    else:
        names = await list_collections.ainvoke({"db_path": db_path})
        lines = [f"- {n}" for n in names]
    collections_str = (
        "\n".join(lines)
        if lines
        else "(no collections yet — index some documents first)"
    )

    # Iteration mode: the Sufficient Context Agent sent us back to RE-ROUTE for
    # the missing pieces (Google's loop re-enters before Search Plan). Plan a
    # narrow route to the collection(s) most likely to hold what's missing,
    # instead of blindly searching every collection.
    iteration = state.get("iteration_count", 0)
    feedback = state.get("feedback", "")
    is_iteration = iteration > 0 and bool(feedback)

    if is_iteration:
        searched_cols = sorted(
            {
                r.get("collection")
                for r in state.get("search_results", [])
                if r.get("collection")
            }
        )
        searched_str = ", ".join(searched_cols) if searched_cols else "(none yet)"
        prompt = PLANNER_ITERATION_PROMPT.format(
            query=state["query"],
            missing_parts=", ".join(state.get("missing_parts", [])) or "(unspecified)",
            feedback=feedback,
            collections=collections_str,
            searched=searched_str,
        )
    else:
        prompt = PLANNER_PROMPT.format(
            query=state["query"],
            collections=collections_str,
        )
    plan: PlanResult = await get_structured_llm(PlanResult).ainvoke(prompt)

    steps_dicts = [s.model_dump() for s in plan.steps]

    mode = "iteration" if is_iteration else "initial"
    trace_entry = make_trace_entry(
        agent="planner",
        decision=f"{mode}: {len(plan.steps)} route(s)",
        detail=str(steps_dicts),
    )

    if not plan.steps:
        # No relevant route. On iteration, hand off to query_rewriter so it can
        # broaden the search to all collections (the safety net). On the initial
        # turn, there's nothing to search → synthesize from what we have.
        return Command(
            goto="query_rewriter" if is_iteration else "synthesis",
            update={
                "plan_steps": [],
                "trace": [trace_entry],
            },
        )

    # All routes go to query_rewriter together (first route set as current_route
    # for search_fanout to know which collection to target initially)
    return Command(
        goto="query_rewriter",
        update={
            "plan_steps": steps_dicts,
            "current_route": steps_dicts[0],
            "trace": [trace_entry],
        },
    )
