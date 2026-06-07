"""Orchestrator (Root Agent) — evaluates query complexity.

Returns Command(goto="synthesis") for simple queries,
Command(goto="planner") for complex multi-step queries.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, OrchestratorResult, make_trace_entry
from src.agents.common import get_llm

ORCHESTRATOR_PROMPT = """You are the Orchestrator (Root Agent) of an Agentic RAG system.

Your job: assess whether a user query can be answered directly or requires multi-step retrieval.

A query is COMPLEX (is_complex=True) if:
- It requires information from multiple different sources/databases
- It requires multiple hops: find something, then use that to find something else
- It asks about multiple distinct entities or topics that need separate searches
- It has sub-questions that need to be answered independently
- The answer likely cannot be found in a single document

A query is SIMPLE (is_complex=False) if:
- It's a straightforward factual question
- The answer is likely in a single document/source
- It can be answered with a single search

User query: {query}

Respond with a structured assessment."""


async def orchestrator_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command | dict:
    """Orchestrator: assess complexity and route accordingly."""
    llm = get_llm()

    prompt = ORCHESTRATOR_PROMPT.format(query=state["query"])
    result: OrchestratorResult = await llm.with_structured_output(
        OrchestratorResult
    ).ainvoke(prompt)

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
