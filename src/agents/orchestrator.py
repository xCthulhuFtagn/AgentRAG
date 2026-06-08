"""Orchestrator (Root Agent) — evaluates query complexity.

Returns Command(goto="synthesis") for simple queries,
Command(goto="planner") for complex multi-step queries.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, OrchestratorResult, make_trace_entry
from src.agents.common import generate_structured

ORCHESTRATOR_PROMPT = """You are the Orchestrator (Root Agent) of an Agentic RAG system.

Your job: decide whether a query needs to RETRIEVE from the indexed documents
(complex → full retrieval pipeline) or can be answered with NO document lookup
at all (simple → answer directly).

This is a RAG system over the user's private documents. Almost every real
question is about facts that live in those documents, so default to COMPLEX.

A query is COMPLEX (is_complex=True) — needs retrieval — if it asks about ANY
specific fact, entity, name, number, spec, detail, relationship, or content that
could plausibly be found in the documents. This includes single-fact questions.
When in doubt, choose COMPLEX.

A query is SIMPLE (is_complex=False) — NO retrieval — ONLY if it is clearly NOT
about the documents at all:
- Greetings / small talk ("hi", "thanks", "how are you")
- Meta questions about you, the assistant, or how to use the system
- General world knowledge with no connection to the user's documents

User query: {query}

Respond with a structured assessment."""


async def orchestrator_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command | dict:
    """Orchestrator: assess complexity and route accordingly."""
    prompt = ORCHESTRATOR_PROMPT.format(query=state["query"])
    result: OrchestratorResult = await generate_structured(
        OrchestratorResult, prompt
    )

    trace_entry = make_trace_entry(
        agent="orchestrator",
        decision=f"is_complex={result.is_complex}",
        detail=result.reasoning,
    )

    if not result.is_complex:
        return Command(
            goto="synthesis",
            update={
                "is_complex": False,
                "orchestrator_reasoning": result.reasoning,
                "trace": [trace_entry],
            },
        )
    else:
        return Command(
            goto="planner",
            update={
                "is_complex": True,
                "orchestrator_reasoning": result.reasoning,
                "trace": [trace_entry],
            },
        )
