"""Sufficient Context Agent — the key innovation from Google Research.

Checks three things before allowing a response:
1. Retrieved snippets — do they contain needed information?
2. Intermediate draft — can we answer from what we have?
3. Missing pieces analysis — what EXACTLY is missing and where to look?

Routes:
- sufficient → Command(goto="synthesis")
- insufficient + iterations left → Command(goto="query_rewriter") with feedback
- insufficient + max iterations → Command(goto=END) with system-generated refusal
"""

from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.config import MAX_ITERATIONS
from src.state import AgentRAGState, SufficientContextResult, make_trace_entry
from src.agents.common import get_llm

SUFFICIENT_CONTEXT_PROMPT = """You are the Sufficient Context Agent — the quality-control inspector of an Agentic RAG system.

Your job: determine if the retrieved context is COMPLETE enough to answer the user's question.

User question: {query}

Retrieved context from searches:
{search_results}

Iteration: {iteration} of {max_iterations}
Previously identified gaps: {previous_gaps}

Analyze THREE things:

1. **Retrieved snippets**: Read all retrieved text chunks. Do they contain the FACTS needed to answer every part of the question?

2. **Draft answer**: Try to construct a draft answer. If you can fully answer the question, the context is sufficient.

3. **Missing pieces (CRITICAL)**: If anything is missing, be SPECIFIC:
   - What exact information is missing?
   - Which collection should we search in?
   - What alternative search terms should be tried?

Rules:
- If ALL parts of the question can be answered from the context → sufficient=True
- If ANY part is missing → sufficient=False, provide detailed feedback
- It's better to flag as insufficient and search again than to guess
- Be honest — do NOT set sufficient=True if information is missing"""


def _build_refusal_answer(state: AgentRAGState, result: SufficientContextResult) -> str:
    """Build a system-generated refusal message when context is exhausted.

    No LLM involved — the system itself states what it found and what's missing.
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

    found_summary = ""
    if found_collections:
        found_summary = (
            f"- Searched {len(found_collections)} collection(s): {', '.join(sorted(found_collections))}\n"
            f"- Retrieved {found_chunks} text chunks total\n"
        )
    else:
        found_summary = "- No relevant documents were found in any collection\n"

    # What's missing
    missing = result.missing_parts or ["specific information required to answer the question"]
    missing_str = "\n".join(f"  • {m}" for m in missing)

    # Search attempts
    queries_tried = state.get("rewritten_queries", [])
    queries_str = "\n".join(f"  • {q}" for q in queries_tried[-10:]) or "  (none)"

    return (
        f"## Unable to fully answer\n\n"
        f"**Question:** {query}\n\n"
        f"**What was found:**\n{found_summary}\n"
        f"**What is missing:**\n{missing_str}\n\n"
        f"**Why:** {result.reason}\n\n"
        f"**Search attempts ({iteration}/{max_iter} iterations):**\n{queries_str}\n\n"
        f"**Recommendation:** The requested information may not exist in the indexed "
        f"documents. Try rephrasing the query, indexing additional documents, "
        f"or breaking the question into smaller parts."
    )


async def sufficient_context_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Sufficient Context: check completeness, command next step.

    Three outcomes:
    1. sufficient=True  → Command(goto="synthesis")      — normal answer
    2. insufficient + iterations left → Command(goto="query_rewriter") — search more
    3. insufficient + max iterations  → Command(goto=END) — system refusal
    """
    llm = get_llm()

    max_iter = state.get("max_iterations", MAX_ITERATIONS)
    iteration = state.get("iteration_count", 0)

    # Format search results
    search_results = state.get("search_results", [])
    results_str = ""
    for i, r in enumerate(search_results[-10:]):
        chunks_str = "\n---\n".join(r.get("chunks", [])[:3])
        results_str += (
            f"\n[Result {i+1}] Collection: {r.get('collection')}, "
            f"Query: {r.get('subquery')}\n{chunks_str}\n"
        )

    if not results_str:
        results_str = "(No search results yet)"

    prompt = SUFFICIENT_CONTEXT_PROMPT.format(
        query=state["query"],
        search_results=results_str,
        iteration=iteration,
        max_iterations=max_iter,
        previous_gaps=", ".join(state.get("missing_parts", [])) or "(none)",
    )

    result: SufficientContextResult = await llm.with_structured_output(
        SufficientContextResult
    ).ainvoke(prompt)

    trace_entry = make_trace_entry(
        agent="sufficient_context",
        decision=f"sufficient={result.sufficient}",
        detail=(
            f"reason={result.reason[:100]}, "
            f"feedback={result.feedback[:100]}, "
            f"missing={result.missing_parts}"
        ),
    )

    # ── Outcome 1: context is sufficient → normal answer ──
    if result.sufficient:
        return Command(
            goto="synthesis",
            update={
                "sufficient": True,
                "sufficient_reason": result.reason,
                "draft_answer": result.draft_answer,
                "trace": [trace_entry],
            },
        )

    # ── Outcome 2: insufficient, but iterations left → search more ──
    if iteration < max_iter:
        return Command(
            goto="query_rewriter",
            update={
                "sufficient": False,
                "sufficient_reason": result.reason,
                "feedback": result.feedback,
                "missing_parts": result.missing_parts,
                "draft_answer": result.draft_answer,
                "iteration_count": iteration + 1,
                "trace": [trace_entry],
            },
        )

    # ── Outcome 3: insufficient + no iterations left → system refusal ──
    refusal = _build_refusal_answer(state, result)

    trace_entry2 = make_trace_entry(
        agent="sufficient_context",
        decision="give_up",
        detail=f"max_iterations={max_iter} reached, context insufficient",
    )

    return Command(
        goto=END,
        update={
            "sufficient": False,
            "sufficient_reason": result.reason,
            "final_answer": refusal,
            "trace": [trace_entry, trace_entry2],
        },
    )
