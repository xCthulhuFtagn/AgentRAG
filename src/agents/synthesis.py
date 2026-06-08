"""Synthesis Agent — generates the final answer from complete context.

Returns Command(goto=END) — edgeless termination.
"""

from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, make_trace_entry
from src.agents.common import get_llm, get_inventory_str

SYNTHESIS_PROMPT = """You are the Synthesis Agent of an Agentic RAG system.

Your job: produce a comprehensive, accurate, and well-structured final answer
based on ALL the retrieved context.

User question: {query}

Complete knowledge base inventory (every document that exists, with a short description of each):
{inventory}

Retrieved context from multiple searches:
{search_results}

Sufficient Context Agent assessment: {sufficient_reason}

Guidelines:
1. Answer ALL parts of the question completely
2. Base your answer ONLY on the retrieved context and the inventory above — do not make up facts, and do not guess/expand abbreviations from general knowledge
3. NEVER refuse. You were reached because the context was judged sufficient, so give a direct best-effort answer from what IS in the context. Extract whatever the chunks actually state (e.g. an abbreviation expanded inside a sentence) and lead with it. Refusal is a different node's job, not yours
4. State remaining uncertainty in ONE short closing line at most — do not turn the answer into a "what is missing / consult other documents" disclaimer, and do not tell the user to look elsewhere
5. Cite which collection/document each piece of information came from
6. Be clear, concise, and well-structured
7. For "describe/list ALL files"-type questions, the inventory is the authoritative
   list — describe every document in it, enriching each from the retrieved chunks
   where available

Context completeness note: {context_note}

Now, produce the final answer:"""


async def synthesis_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Synthesis: generate final answer, command END."""
    llm = get_llm(temperature=0.0)

    results_str = ""
    for i, r in enumerate(state.get("search_results", [])):
        chunks_str = "\n---\n".join(r.get("chunks", []))
        results_str += (
            f"\n### Source {i+1}: {r.get('collection', 'unknown')}\n"
            f"Query: {r.get('subquery', 'unknown')}\n"
            f"Content:\n{chunks_str}\n"
        )

    if not results_str:
        results_str = "(No context retrieved)"

    # Synthesis is only reached after the judge ruled the context sufficient.
    context_note = "Context is sufficient — answer fully from it."

    inventory = await get_inventory_str(state.get("db_path"))

    prompt = SYNTHESIS_PROMPT.format(
        query=state["query"],
        inventory=inventory,
        search_results=results_str,
        sufficient_reason=state.get("sufficient_reason", "Not assessed"),
        context_note=context_note,
    )

    answer: str = (await llm.ainvoke(prompt)).content.strip()

    trace_entry = make_trace_entry(
        agent="synthesis",
        decision="final_answer",
        detail=f"answer_length={len(answer)} chars",
    )

    return Command(
        goto=END,
        update={
            "final_answer": answer,
            "trace": [trace_entry],
        },
    )
