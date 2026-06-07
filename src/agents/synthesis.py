"""Synthesis Agent — generates the final answer from complete context.

Returns Command(goto=END) — edgeless termination.
"""

from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, make_trace_entry
from src.agents.common import get_llm

SYNTHESIS_PROMPT = """You are the Synthesis Agent of an Agentic RAG system.

Your job: produce a comprehensive, accurate, and well-structured final answer
based on ALL the retrieved context.

User question: {query}

Retrieved context from multiple searches:
{search_results}

Sufficient Context Agent assessment: {sufficient_reason}

Guidelines:
1. Answer ALL parts of the question completely
2. Base your answer ONLY on the retrieved context — do not make up facts
3. If the context is incomplete, clearly state what is known and what remains uncertain
4. Cite which collection/document each piece of information came from
5. Be clear, concise, and well-structured

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
        results_str = (
            "(No context retrieved — answer based on your knowledge, "
            "but clearly state this)"
        )

    context_note = (
        "Context is sufficient — answer fully."
        if state.get("sufficient")
        else "Context may be incomplete — answer what you can, note gaps."
    )

    prompt = SYNTHESIS_PROMPT.format(
        query=state["query"],
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
