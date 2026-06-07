"""Planner Agent — breaks down query into search routes.

Always returns Command(goto="query_rewriter") or Command(goto="synthesis").
No Send — all routes processed together in query_rewriter.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, PlanResult, make_trace_entry
from src.agents.common import get_structured_llm
from src.vectordb.tools import list_collections

PLANNER_PROMPT = """You are the Planner Agent of an Agentic RAG system.

Your job: break down a complex query into specific search routes, each targeting
a specific document collection.

First, review the available collections to decide where to search.

Available collections: {collections}

Then, for each piece of information needed, create a RouteStep with:
- collection: the exact table name to search (must match available collections)
- subquery: a focused search query for that specific piece
- rationale: why this collection is relevant

User query: {query}

If the query needs information from multiple collections, create multiple steps.
If the query can be answered from a single collection, create one step.
If no relevant collection exists, return an empty steps list."""


async def planner_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Planner: create search routes, then command query_rewriter."""
    collections = await list_collections.ainvoke({"db_path": state.get("db_path")})
    collections_str = (
        ", ".join(collections)
        if collections
        else "(no collections yet — index some documents first)"
    )

    prompt = PLANNER_PROMPT.format(
        query=state["query"],
        collections=collections_str,
    )
    plan: PlanResult = await get_structured_llm(PlanResult).ainvoke(prompt)

    steps_dicts = [s.model_dump() for s in plan.steps]

    trace_entry = make_trace_entry(
        agent="planner",
        decision=f"{len(plan.steps)} route(s)",
        detail=str(steps_dicts),
    )

    if not plan.steps:
        return Command(
            goto="synthesis",
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
