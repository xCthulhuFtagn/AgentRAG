"""Query Rewriter Agent — rewrites queries for precise vector search.

Returns Command(goto="search_fanout") — edgeless routing.
Processes all plan_steps at once, or handles iteration feedback.
"""

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

REWRITER_ITERATION_PROMPT = """You are the Query Rewriter Agent of an Agentic RAG system.

This is an ITERATION — previous searches did not find everything needed.

Original user question: {original_query}
Previous search missed: {missing_parts}
Feedback from context checker: {feedback}
Previous queries tried: {previous_queries}

Your job: create a NEW search query that specifically targets the MISSING information.
Be more specific, use alternative keywords, try a different angle.
Focus ONLY on what was missed.

Return ONLY the rewritten search query text, nothing else."""


async def query_rewriter_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Query Rewriter: produce search-optimized queries, command search_fanout."""
    llm = get_llm(temperature=0.3)

    feedback = state.get("feedback", "")
    rewritten: list[str] = []
    # search_tasks pairs each query with its target collection.
    # collection=None means "search across all collections".
    search_tasks: list[dict] = []

    plan_steps = state.get("plan_steps", [])
    if feedback and state.get("iteration_count", 0) > 0 and not plan_steps:
        # ── Iteration fallback: planner found no relevant route to re-target ──
        # On iteration the Planner normally re-routes to the collection holding
        # the missing piece. Only if it couldn't (empty plan_steps) do we fall
        # back to a single targeted query across ALL collections (collection=None).
        prompt = REWRITER_ITERATION_PROMPT.format(
            original_query=state["query"],
            missing_parts=", ".join(state.get("missing_parts", [])),
            feedback=feedback,
            previous_queries=", ".join(state.get("rewritten_queries", [])[-5:]),
        )
        result: str = (await llm.ainvoke(prompt)).content.strip().strip('"')
        rewritten = [result]
        search_tasks = [{"collection": None, "query": result}]
        mode = "iteration"

        trace_entry = make_trace_entry(
            agent="query_rewriter",
            decision=mode,
            detail=f"query='{result}' (all collections)",
        )
    else:
        # ── Rewrite ALL plan routes, one task per route ──
        # Reached on the initial turn and on iterations where the Planner
        # re-routed (plan_steps populated) to target the missing piece.
        if not plan_steps:
            rewritten = [state["query"]]  # fallback: use original query
            search_tasks = [{"collection": None, "query": state["query"]}]
        else:
            for step in plan_steps:
                collection = step.get("collection", "unknown")
                prompt = REWRITER_PROMPT.format(
                    original_query=state["query"],
                    collection=collection,
                    subquery=step.get("subquery", state["query"]),
                )
                result: str = (await llm.ainvoke(prompt)).content.strip().strip('"')
                rewritten.append(result)
                search_tasks.append({"collection": collection, "query": result})
        mode = "initial"

        trace_entry = make_trace_entry(
            agent="query_rewriter",
            decision=f"{mode} ({len(rewritten)} queries)",
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
