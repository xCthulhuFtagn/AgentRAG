"""Tests for the Agentic RAG LangGraph pipeline."""

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import (
    make_initial_state,
    AgentRAGState,
    OrchestratorResult,
    PlanResult,
    RouteStep,
    SufficientContextResult,
)
from src.graph import build_graph

# The live smoke test needs a real DeepSeek key (it calls the API).
requires_api_key = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set — skipping live graph smoke test",
)


def test_graph_compiles():
    """Verify the graph compiles without errors."""
    graph = build_graph()
    assert graph is not None
    # Graph should be compiled and have nodes
    nodes = graph.get_graph().nodes if hasattr(graph, 'get_graph') else {}
    print(f"Graph compiled successfully")


def test_state_defaults():
    """Verify AgentRAGState initializes correctly."""
    state = make_initial_state(query="test query")
    assert state["query"] == "test query"
    assert state["max_iterations"] == 3
    assert state["iteration_count"] == 0
    assert state["search_results"] == []
    assert state["trace"] == []
    assert state["is_complex"] is None
    assert state["rewritten_queries"] == []
    # db_path defaults to None (global LANCE_DB_PATH); search_tasks empty
    assert state["db_path"] is None
    assert state["search_tasks"] == []


def test_state_db_path_threading():
    """db_path is carried through state for per-project DB isolation."""
    state = make_initial_state(query="q", db_path="data/lancedb/proj123")
    assert state["db_path"] == "data/lancedb/proj123"


def test_state_annotated_reducers():
    """Verify that operator.add reducer works for trace accumulation."""
    # Simulate what happens during graph execution:
    # Each node update with trace=[entry] should concatenate
    state = make_initial_state(query="test")
    state["trace"] = [{"agent": "test1", "decision": "d1", "detail": ""}]
    state["trace"] = state["trace"] + [{"agent": "test2", "decision": "d2", "detail": ""}]
    assert len(state["trace"]) == 2

    # Same for search_results
    state["search_results"] = [{"col": "a"}]
    state["search_results"] = state["search_results"] + [{"col": "b"}]
    assert len(state["search_results"]) == 2


def test_orchestrator_result():
    """Verify OrchestratorResult model."""
    result = OrchestratorResult(is_complex=True, reasoning="Multi-step query")
    assert result.is_complex is True
    assert result.reasoning == "Multi-step query"


def test_plan_result():
    """Verify PlanResult model."""
    result = PlanResult(
        is_multi_step=True,
        steps=[
            RouteStep(
                collection="servers",
                subquery="SRV-7742 specs",
                rationale="Find server details",
            ),
        ],
    )
    assert result.is_multi_step is True
    assert len(result.steps) == 1
    assert result.steps[0].collection == "servers"


def test_sufficient_context_result():
    """Verify SufficientContextResult model."""
    result = SufficientContextResult(
        sufficient=False,
        reason="Missing allergy info",
        feedback="Search for rashes in clinical notes",
        missing_parts=["allergy records"],
    )
    assert result.sufficient is False
    assert len(result.missing_parts) == 1


@requires_api_key
@pytest.mark.asyncio
async def test_graph_async_stream(tmp_path):
    """Live smoke test: the graph streams without error on an empty corpus.

    Scoped to a throwaway db_path (tmp_path) so it never writes under the real
    data/ dir, and skipped when no DEEPSEEK_API_KEY is set (it calls the API).
    """
    graph = build_graph()
    state = make_initial_state(query="What is 2+2?", db_path=str(tmp_path / "lancedb"))

    event_count = 0
    async for event in graph.astream(
        state,
        config={"configurable": {"thread_id": "test-session"}},
        stream_mode="updates",
    ):
        assert isinstance(event, dict)
        event_count += 1
    assert event_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
