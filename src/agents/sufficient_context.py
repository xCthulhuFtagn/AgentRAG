"""Sufficient Context Agent — the key innovation from Google Research.

Checks three things before allowing a response:
1. Retrieved snippets — do they contain needed information?
2. Intermediate draft — can we answer from what we have?
3. Missing pieces analysis — what EXACTLY is missing and where to look?

Routes:
- sufficient → Command(goto="synthesis")
- insufficient + iterations left → Command(goto="query_rewriter") with feedback
- insufficient + max iterations → Command(goto="give_up")
"""

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.config import general_settings
from src.state import AgentRAGState, SufficientContextResult, make_trace_entry
from src.agents.common import get_structured_llm

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


async def sufficient_context_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Sufficient Context: check completeness, command next step.

    Three outcomes:
    1. sufficient=True  → Command(goto="synthesis")      — normal answer
    2. insufficient + iterations left → Command(goto="query_rewriter") — search more
    3. insufficient + max iterations  → Command(goto="give_up") — system refusal
    """
    max_iter = state.get("max_iterations", general_settings.max_iterations)
    iteration = state.get("iteration_count", 0)

    # Format search results
    search_results = state.get("search_results", [])
    results_str = ""
    for i, r in enumerate(search_results[-10:]):
        chunks = r.get("chunks", [])
        seqs = r.get("seqs", []) or []
        # Tag each chunk with its document position so the judge can see
        # contiguity and gaps (chunks arrive seq-ordered after stitching).
        lines = []
        for j, chunk in enumerate(chunks):
            seq = seqs[j] if j < len(seqs) and seqs[j] is not None else "?"
            lines.append(f"[seq={seq}] {chunk}")
        chunks_str = "\n---\n".join(lines)
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

    result: SufficientContextResult = await get_structured_llm(
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

    # ── Outcome 3: insufficient + no iterations left → give up ──
    return Command(
        goto="give_up",
        update={
            "sufficient": False,
            "sufficient_reason": result.reason,
            "missing_parts": result.missing_parts,
            "trace": [trace_entry],
        },
    )
