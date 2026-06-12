"""Search Fanout Agent — executes vector search for the current-turn tasks.

Returns Command(goto="sufficient_context") — edgeless routing.
Parallelism via asyncio.gather for tool calls.

Reads state["search_tasks"] = [{"collection": str, "query": str}], one concrete
(collection, query) pair per planner route — no search-all mode.

Every executed search is appended to state["search_results"] — empty results
included. An empty search is data, not noise: the mechanical statistics
(searched set, last-search novelty, coverage) are computed from this record,
and a search that returned nothing is the strongest exhaustion signal.
"""

import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, make_trace_entry
from src.vectordb.config import vdb_settings
from src.vectordb.tools import vector_search, gather_neighbors


async def search_fanout_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Search Fanout: execute vector searches, command sufficient_context."""
    db_path = state.get("db_path")

    # Per-project stitching overrides (web threads them via make_initial_state);
    # missing keys fall back to the global vdb_settings defaults.
    stitch = state.get("stitch_settings") or {}
    padding = stitch.get("expand_padding")
    bridge_gap = stitch.get("bridge_gap")
    stitch_kwargs = {
        "padding": vdb_settings.expand_padding if padding is None else padding,
        "bridge_gap": vdb_settings.bridge_gap if bridge_gap is None else bridge_gap,
    }

    # Each task targets one concrete collection (query_rewriter built them from
    # the planner's routes); search every (collection, query) pair in parallel.
    resolved: list[tuple[str, str]] = [
        (t["collection"], t.get("query", ""))
        for t in state.get("search_tasks") or []
        if t.get("collection")
    ]

    async def search_one(collection: str, query: str) -> dict:
        try:
            result = await vector_search.ainvoke({
                "query": query,
                "collection": collection,
                "top_k": vdb_settings.search_top_k,
                "db_path": db_path,
            })
            chunks = result.get("chunks", [])
            seqs = result.get("seqs", [])

            # Deterministic context expansion: stitch each hit back to its
            # contiguous seq-neighborhood so truncated structural blocks (TOC,
            # reference lists) come back whole. Legacy tables (no seq) → no-op.
            if seqs and any(s is not None for s in seqs):
                expanded = await gather_neighbors(
                    result.get("collection", collection), seqs, db_path,
                    **stitch_kwargs,
                )
                if expanded:
                    chunks = [e["text"] for e in expanded]
                    seqs = [e["seq"] for e in expanded]

            return {
                "collection": result.get("collection", collection),
                "subquery": query,
                "chunks": chunks,
                "seqs": seqs,
                "scores": result.get("scores", []),
            }
        except Exception as e:
            return {
                "collection": collection,
                "subquery": query,
                "chunks": [],
                "seqs": [],
                "scores": [],
                "error": str(e),
            }

    results = await asyncio.gather(*[search_one(c, q) for c, q in resolved])
    # Keep EVERY executed search, including empty ones: search_results is the
    # record the mechanical statistics are computed from (searched set, «+0
    # новых чанков» exhaustion detector, coverage). Consumers that render
    # context (judge/synthesis/give_up) skip chunkless entries themselves.

    total_chunks = sum(len(r.get("chunks", [])) for r in results)
    collections_searched = sorted({c for c, _ in resolved})

    # Per-pair "source ← query (N chunks)" for the UI's middle line.
    found_by = {(r.get("collection"), r.get("subquery")): len(r.get("chunks", [])) for r in results}
    search_info = "\n".join(
        f"{c} ← «{q}»  ({found_by.get((c, q), 0)} chunks)" for c, q in resolved
    )

    trace_entry = make_trace_entry(
        agent="search_fanout",
        decision=f"searched {len(resolved)} (collection, query) pairs",
        detail=f"collections={collections_searched}, found_chunks={total_chunks}",
        info=search_info,
    )

    return Command(
        goto="sufficient_context",
        update={
            "search_results": results,
            "trace": [trace_entry],
        },
    )
