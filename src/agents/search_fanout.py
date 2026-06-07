"""Search Fanout Agent — executes vector search for rewritten queries.

Returns Command(goto="sufficient_context") — edgeless routing.
Parallelism via asyncio.gather for tool calls.
"""

import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, make_trace_entry
from src.vectordb.tools import vector_search


async def search_fanout_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Search Fanout: execute vector searches, command sufficient_context."""
    route = state.get("current_route") or {}
    collection = route.get("collection", "unknown")

    rewritten = state.get("rewritten_queries", [])
    queries_to_search = rewritten[-5:] if rewritten else [state["query"]]

    async def search_one(query: str) -> dict:
        try:
            result = await vector_search.ainvoke({
                "query": query,
                "collection": collection,
                "top_k": 5,
            })
            return {
                "collection": result.get("collection", collection),
                "subquery": query,
                "chunks": result.get("chunks", []),
                "scores": result.get("scores", []),
            }
        except Exception as e:
            return {
                "collection": collection,
                "subquery": query,
                "chunks": [],
                "scores": [],
                "error": str(e),
            }

    results = await asyncio.gather(*[search_one(q) for q in queries_to_search])
    results = [r for r in results if r.get("chunks")]

    total_chunks = sum(len(r.get("chunks", [])) for r in results)

    trace_entry = make_trace_entry(
        agent="search_fanout",
        decision=f"searched {len(queries_to_search)} queries",
        detail=f"collection={collection}, found_chunks={total_chunks}",
    )

    return Command(
        goto="sufficient_context",
        update={
            "search_results": results,
            "trace": [trace_entry],
        },
    )
