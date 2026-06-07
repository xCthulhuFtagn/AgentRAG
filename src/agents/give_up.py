"""Give Up Agent — system-generated refusal when context is exhausted.

No LLM involved — the system honestly reports what was found, what's missing,
and why the question cannot be fully answered.
"""

from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.config import MAX_ITERATIONS
from src.state import AgentRAGState, make_trace_entry


def _build_refusal_answer(state: AgentRAGState) -> str:
    """Build a system-generated refusal message.

    No LLM — the system itself states what it found and what's missing.
    """
    query = state["query"]
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", MAX_ITERATIONS)

    # Summarize what was found
    search_results = state.get("search_results", [])
    found_collections: set[str] = set()
    found_chunks = 0
    for r in search_results:
        chunks = r.get("chunks", [])
        if chunks:
            found_collections.add(r.get("collection", "unknown"))
            found_chunks += len(chunks)

    if found_collections:
        found_summary = (
            f"- Searched {len(found_collections)} collection(s): "
            f"{', '.join(sorted(found_collections))}\n"
            f"- Retrieved {found_chunks} text chunks total\n"
        )
    else:
        found_summary = "- No relevant documents were found in any collection\n"

    # What's missing
    missing = state.get("missing_parts", []) or [
        "specific information required to answer the question"
    ]
    missing_str = "\n".join(f"  • {m}" for m in missing)

    # Search attempts
    queries_tried = state.get("rewritten_queries", [])
    queries_str = (
        "\n".join(f"  • {q}" for q in queries_tried[-10:]) or "  (none)"
    )

    reason = state.get("sufficient_reason", "Insufficient context retrieved")

    return (
        f"## Unable to fully answer\n\n"
        f"**Question:** {query}\n\n"
        f"**What was found:**\n{found_summary}\n"
        f"**What is missing:**\n{missing_str}\n\n"
        f"**Why:** {reason}\n\n"
        f"**Search attempts ({iteration}/{max_iter} iterations):**\n{queries_str}\n\n"
        f"**Recommendation:** The requested information may not exist in the indexed "
        f"documents. Try rephrasing the query, indexing additional documents, "
        f"or breaking the question into smaller parts."
    )


async def give_up_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Give Up: build refusal answer, command END. No LLM call."""

    refusal = _build_refusal_answer(state)

    trace_entry = make_trace_entry(
        agent="give_up",
        decision="refusal",
        detail=(
            f"max_iterations={state.get('max_iterations', MAX_ITERATIONS)} "
            f"reached, context insufficient"
        ),
    )

    return Command(
        goto=END,
        update={
            "final_answer": refusal,
            "trace": [trace_entry],
        },
    )
