"""Sufficient Context Agent — the key innovation from Google Research.

Checks three things before allowing a response:
1. Retrieved snippets — do they contain needed information?
2. Intermediate draft — can we answer from what we have?
3. Missing pieces analysis — what EXACTLY is missing and where to look?

Routes:
- sufficient → Command(goto="synthesis")
- insufficient + iterations left → Command(goto="planner") with feedback (re-route)
- insufficient + max iterations → Command(goto="give_up")
"""

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.config import general_settings
from src.state import AgentRAGState, SufficientContextResult, make_trace_entry
from src.agents.common import get_structured_llm, get_inventory_str

SUFFICIENT_CONTEXT_PROMPT = """You are the Sufficient Context Agent — the quality-control inspector of an Agentic RAG system.

Your job: determine if the retrieved context is COMPLETE enough to answer the user's question.

User question: {query}

Complete knowledge base inventory (GROUND TRUTH — these are ALL the collections that exist, with a short description of each):
{inventory}

Retrieved context from searches (each block is tagged with the collection it came from):
{search_results}

Iteration: {iteration} of {max_iterations}
Previously identified gaps: {previous_gaps}

Analyze THREE things:

1. **Retrieved snippets**: Read all retrieved text chunks. Do they contain the FACTS needed to answer every part of the question?

2. **Draft answer**: Try to construct a draft answer from ALL the context accumulated so far (across every iteration). If it already answers the question, the context is SUFFICIENT — stop here. Do NOT keep searching for additional or confirmatory sources once the question is answerable.

3. **Missing pieces (CRITICAL)**: If anything is missing, be SPECIFIC:
   - What exact information is missing?
   - Which collection should we search in?
   - What alternative search terms should be tried?

Rules (evaluate sufficiency of the WHOLE accumulated context FIRST, before thinking about searching more):
- If the accumulated context answers the question → sufficient=True. This holds even if some collections were never searched and might contain related material. Do NOT mark insufficient just to be thorough, to double-check/confirm an answer you already have, or because another collection "might also" hold it (or "more" of it).
- Only when the answer is genuinely MISSING or INCOMPLETE → sufficient=False, with specific feedback (what's missing, which collection to search next)
- A "not found / not defined / not mentioned" answer is NOT sufficient while any collection that could plausibly hold the answer has NOT been searched yet — set sufficient=False and route there. (This applies only when the answer is actually absent — not to verify an answer already present.)
- It's better to flag insufficient and search again than to guess — but only when something is truly missing, not merely unconfirmed
- Be consistent: if your feedback says to search more, then sufficient MUST be False

How to use the inventory (it is the COMPLETE, authoritative list of every document — there are no others):
- "Describe/list ALL files"-type questions → full coverage means every collection has been searched OR is adequately summarized by its description. The inventory is exhaustive, so you CAN confirm completeness — do not demand proof of more documents.
- SPECIFIC questions (e.g. "what is X") → if the answer is NOT in the retrieved chunks, compare the inventory against the collections that actually appear in the retrieved context above. If any collection that is NOT yet among them has a description suggesting it could contain the answer, set sufficient=False and name that collection in feedback/missing_parts. Only accept a negative answer once every plausibly-relevant collection has actually been searched."""


async def sufficient_context_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Sufficient Context: check completeness, command next step.

    Three outcomes:
    1. sufficient=True  → Command(goto="synthesis")      — normal answer
    2. insufficient + iterations left → Command(goto="planner") — re-route & search more
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

    inventory = await get_inventory_str(state.get("db_path"))

    prompt = SUFFICIENT_CONTEXT_PROMPT.format(
        query=state["query"],
        inventory=inventory,
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

    # ── Outcome 2: insufficient, but iterations left → re-route & search more ──
    # Go back to the Planner (not query_rewriter): it re-routes to the
    # collection most likely to hold the missing piece, instead of blindly
    # searching every collection. Mirrors Google's loop that re-enters before
    # Search Plan.
    if iteration < max_iter:
        return Command(
            goto="planner",
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
