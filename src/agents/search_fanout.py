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

    # Per-project overrides (web threads them via make_initial_state); missing
    # keys fall back to the global vdb_settings defaults — this must hold for
    # EVERY key here (the CLI passes no stitch_settings at all, so it relies
    # entirely on these fallbacks to honor .env).
    stitch = state.get("stitch_settings") or {}
    padding = stitch.get("expand_padding")
    bridge_gap = stitch.get("bridge_gap")
    top_k = stitch.get("search_top_k")
    top_k = vdb_settings.search_top_k if top_k is None else top_k
    stitch_kwargs = {
        "padding": vdb_settings.expand_padding if padding is None else padding,
        "bridge_gap": vdb_settings.bridge_gap if bridge_gap is None else bridge_gap,
    }
    reranking_enabled = stitch.get("reranking_enabled")
    reranking_enabled = (
        vdb_settings.reranking_enabled if reranking_enabled is None else reranking_enabled
    )
    remove_irrelevant = stitch.get("reranking_remove_irrelevant")
    remove_irrelevant = (
        vdb_settings.reranking_remove_irrelevant
        if remove_irrelevant is None
        else remove_irrelevant
    )
    hybrid_enabled = stitch.get("hybrid_search_enabled")
    hybrid_enabled = (
        vdb_settings.hybrid_search_enabled if hybrid_enabled is None else hybrid_enabled
    )

    # Each task targets one concrete collection (query_rewriter built them from
    # the planner's routes); search every (collection, query) pair in parallel.
    resolved: list[tuple[str, str]] = [
        (t["collection"], t.get("query", ""))
        for t in state.get("search_tasks") or []
        if t.get("collection")
    ]

    async def search_one(collection: str, query: str) -> dict:
        """Raw KNN hits for one (collection, query) pair — no stitching yet.

        Stitching happens later, over whichever hits survive reranking, so a
        contiguous block pulled in by gather_neighbors is never independently
        judged or amputated by relevance filtering (see below).
        """
        try:
            result = await vector_search.ainvoke({
                "query": query,
                "collection": collection,
                "top_k": top_k,
                "db_path": db_path,
                "hybrid": hybrid_enabled,
            })
            return {
                "collection": result.get("collection", collection),
                "subquery": query,
                "chunks": result.get("chunks", []),
                "seqs": result.get("seqs", []),
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

    # ── LLM per-chunk relevance assessment (opt-in) — over KNN HITS only ────
    # Assessed BEFORE stitching: only the actual hits are judged, so a
    # coherent structural block pulled in afterward by gather_neighbors is
    # never independently judged (or amputated) chunk by chunk. Results are
    # stored as a "relevant" list (bool | None, aligned with chunks at this
    # point) — consumed by collection_search_stats for the topic-hit trend.
    if reranking_enabled:
        from src.agents.common import assess_chunks_relevance

        query = state["query"]
        # Collect all HIT chunks (flatten across non-errored results).
        all_chunks: list[str] = []
        chunk_map: list[tuple[int, int]] = []  # (result_idx, chunk_idx)
        for ri, r in enumerate(results):
            if r.get("error"):
                continue
            for ci in range(len(r.get("chunks", []))):
                all_chunks.append(r["chunks"][ci])
                chunk_map.append((ri, ci))

        if all_chunks:
            relevance = await assess_chunks_relevance(all_chunks, query)
            for ri, r in enumerate(results):
                if not r.get("error"):
                    r.setdefault("relevant", [None] * len(r.get("chunks", [])))
            for (ri, ci), rel in zip(chunk_map, relevance):
                results[ri]["relevant"][ci] = rel

        # ── Remove hits CONFIRMED irrelevant (opt-in) — before stitching ────
        # Only a strict False drops a hit. True keeps it, and None (the
        # assessment call itself failed — unknown, not "irrelevant") also
        # keeps it: a transient API error must never silently discard a chunk.
        if remove_irrelevant:
            for r in results:
                rel = r.get("relevant")
                if not rel:
                    continue
                keep = [i for i, flag in enumerate(rel) if flag is not False]
                if len(keep) == len(rel):
                    continue  # nothing confirmed irrelevant, nothing to drop
                r["chunks"] = [r["chunks"][i] for i in keep]
                if r.get("seqs"):
                    r["seqs"] = [r["seqs"][i] for i in keep if i < len(r["seqs"])]
                if r.get("scores"):
                    r["scores"] = [r["scores"][i] for i in keep if i < len(r["scores"])]
                r["relevant"] = [rel[i] for i in keep]

    # ── Neighbor stitching — over the surviving hits ────────────────────────
    # Deterministic context expansion: stitch each surviving hit back to its
    # contiguous seq-neighborhood so truncated structural blocks (TOC,
    # reference lists) come back whole. Legacy tables (no seq) → no-op.
    for r in results:
        if r.get("error"):
            continue
        seqs = r.get("seqs") or []
        if not (seqs and any(s is not None for s in seqs)):
            continue
        hit_relevant = r.get("relevant")  # aligned with current chunks/seqs
        hit_flag_by_seq = (
            {s: hit_relevant[i] for i, s in enumerate(seqs) if s is not None}
            if hit_relevant
            else None
        )
        expanded = await gather_neighbors(
            r["collection"], seqs, db_path, **stitch_kwargs,
        )
        if expanded:
            r["chunks"] = [e["text"] for e in expanded]
            r["seqs"] = [e["seq"] for e in expanded]
            if hit_flag_by_seq is not None:
                # A stitched-in chunk that wasn't itself a hit was never
                # individually assessed — None, not inherited from a hit
                # elsewhere in the same merged window.
                r["relevant"] = [hit_flag_by_seq.get(e["seq"]) for e in expanded]

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
