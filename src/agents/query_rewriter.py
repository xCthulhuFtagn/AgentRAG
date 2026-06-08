"""Query Rewriter Agent — rewrites queries for precise vector search.

Returns Command(goto="search_fanout") — edgeless routing.
Processes all plan_steps at once, or handles iteration feedback.
"""

import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, make_trace_entry
from src.agents.common import get_llm

REWRITER_PROMPT = """You are the Query Rewriter Agent of an Agentic RAG system.

Your job: convert a search route into a precise, search-optimized query for vector search.

Original user question: {original_query}
Current search target: collection '{collection}'
What we're looking for: {subquery}

Guidelines:
1. Be specific and use keywords likely to appear in documents
2. Remove question words (who, what, why) — use declarative form
3. Include synonyms and related terms
4. Keep it concise (1-2 sentences max)

Return ONLY the rewritten search query text, nothing else."""


async def _rewrite_route(llm, original_query: str, step: dict) -> tuple[str, str]:
    """Rewrite one plan route into a search-optimized query.

    Returns (collection, query). Module-level so it isn't redefined per node
    call; routes are rewritten concurrently via asyncio.gather.
    """
    collection = step.get("collection", "unknown")
    prompt = REWRITER_PROMPT.format(
        original_query=original_query,
        collection=collection,
        subquery=step.get("subquery", original_query),
    )
    result: str = (await llm.ainvoke(prompt)).content.strip().strip('"')
    return collection, result


async def query_rewriter_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Query Rewriter: produce search-optimized queries, command search_fanout.

    The Planner guarantees ≥1 route (no relevant collection → give_up), so we
    always rewrite its plan — one search task per route, no fallback modes.
    Routes are independent → rewrite them concurrently (asyncio.gather), which
    preserves order so rewritten[i]/search_tasks[i] align with plan_steps[i].
    """
    llm = get_llm(temperature=0.3)
    plan_steps = state.get("plan_steps", [])

    pairs = await asyncio.gather(
        *[_rewrite_route(llm, state["query"], s) for s in plan_steps]
    )
    rewritten = [q for _, q in pairs]
    search_tasks = [{"collection": c, "query": q} for c, q in pairs]

    trace_entry = make_trace_entry(
        agent="query_rewriter",
        decision=f"{len(rewritten)} queries",
        detail=str(search_tasks),
    )

    return Command(
        goto="search_fanout",
        update={
            "rewritten_queries": rewritten,
            "search_tasks": search_tasks,
            "trace": [trace_entry],
        },
    )
