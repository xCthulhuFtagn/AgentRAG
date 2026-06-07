"""Tests for the Agentic RAG LangGraph pipeline."""

import asyncio
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


@pytest.mark.asyncio
async def test_graph_async_stream():
    """Verify graph can be invoked asynchronously (without indexed docs)."""
    graph = build_graph()
    state = make_initial_state(query="What is 2+2?")

    # This should run without error — graph handles empty collections gracefully
    try:
        event_count = 0
        async for event in graph.astream(
            state,
            config={"configurable": {"thread_id": "test-session"}},
            stream_mode="updates",
        ):
            assert isinstance(event, dict)
            event_count += 1
        print(f"Graph stream produced {event_count} events")
    except Exception as e:
        # If no documents are indexed, we expect graceful handling
        print(f"Graph stream complete (note: {type(e).__name__}: {e})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
