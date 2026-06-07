"""Search Fanout Agent — executes vector search for the current-turn tasks.

Returns Command(goto="sufficient_context") — edgeless routing.
Parallelism via asyncio.gather for tool calls.

Reads state["search_tasks"] = [{"collection": str|None, "query": str}].
collection=None means search the query across ALL collections in the project DB
(used during iteration, when we don't know which file holds the missing piece).
"""

import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, make_trace_entry
from src.vectordb.tools import vector_search, list_collections


async def search_fanout_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Search Fanout: execute vector searches, command sufficient_context."""
    db_path = state.get("db_path")

    tasks = state.get("search_tasks") or []
    if not tasks:
        # Fallback: search the original query across all collections.
        tasks = [{"collection": None, "query": state["query"]}]

    # Resolve collection=None into concrete collections (search everywhere).
    all_collections: list[str] | None = None
    resolved: list[tuple[str, str]] = []  # (collection, query)
    for t in tasks:
        query = t.get("query", "")
        collection = t.get("collection")
        if collection is None:
            if all_collections is None:
                all_collections = await list_collections.ainvoke({"db_path": db_path})
            for col in all_collections:
                resolved.append((col, query))
        else:
            resolved.append((collection, query))

    async def search_one(collection: str, query: str) -> dict:
        try:
            result = await vector_search.ainvoke({
                "query": query,
                "collection": collection,
                "top_k": 5,
                "db_path": db_path,
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

    results = await asyncio.gather(*[search_one(c, q) for c, q in resolved])
    results = [r for r in results if r.get("chunks")]

    total_chunks = sum(len(r.get("chunks", [])) for r in results)
    collections_searched = sorted({c for c, _ in resolved})

    trace_entry = make_trace_entry(
        agent="search_fanout",
        decision=f"searched {len(resolved)} (collection, query) pairs",
        detail=f"collections={collections_searched}, found_chunks={total_chunks}",
    )

    return Command(
        goto="sufficient_context",
        update={
            "search_results": results,
            "trace": [trace_entry],
        },
    )
