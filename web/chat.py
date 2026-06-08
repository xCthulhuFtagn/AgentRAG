"""Chat streaming — wraps the Agentic RAG graph for one question.

Each call is an independent RAG run (fresh thread_id) — the graph answers a
single query from the project's documents; it carries no conversation history,
so accumulating reducers (search_results, trace) must not bleed across messages.

Yields tuples:
    ("trace", {agent, decision, detail, input_tokens, output_tokens})
                                             — one per agent step, live
    ("answer", str)                          — the final answer
"""

import uuid
from typing import AsyncIterator

from src.state import make_initial_state
from web import runtime


async def run_chat(project_id: str, query: str) -> AsyncIterator[tuple[str, object]]:
    """Stream agent steps and the final answer for a query in a project."""
    db_path = runtime.STORE.db_path(project_id)
    initial = make_initial_state(query=query, db_path=db_path)

    # Fresh thread per message — independent run, no state bleed.
    thread_id = f"{project_id}-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    async for event in runtime.GRAPH.astream(
        initial, config=config, stream_mode="updates"
    ):
        for _node_name, node_output in event.items():
            if not isinstance(node_output, dict):
                continue

            for entry in node_output.get("trace", []):
                yield ("trace", entry)

            if "final_answer" in node_output and node_output["final_answer"]:
                yield ("answer", node_output["final_answer"])
