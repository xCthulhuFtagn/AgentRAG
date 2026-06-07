"""CLI entry point for Agentic RAG.

Usage:
    python -m src.main --query "What are the specs of the server used in Project X?"
    python -m src.main --query "Tell me about machine learning" --max-iterations 5
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import make_initial_state
from src.graph import build_graph


async def run_query(query: str, max_iterations: int = 3):
    """Run a query through the Agentic RAG pipeline."""
    graph = build_graph()
    initial_state = make_initial_state(query=query, max_iterations=max_iterations)

    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*70}\n")

    # Stream the graph execution — see each node's output
    async for event in graph.astream(
        initial_state,
        config={"configurable": {"thread_id": "cli-session"}},
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            if not isinstance(node_output, dict):
                continue

            # Print trace updates
            trace_entries = node_output.get("trace", [])
            for entry in trace_entries:
                agent = entry.get("agent", node_name)
                decision = entry.get("decision", "")
                detail = entry.get("detail", "")
                print(f"  [{agent}] {decision}")
                if detail and len(detail) < 200:
                    print(f"    └─ {detail}")
                elif detail:
                    print(f"    └─ {detail[:200]}...")

            # Print search results summary
            search_results = node_output.get("search_results", [])
            if search_results:
                chunk_count = sum(len(r.get("chunks", [])) for r in search_results)
                print(f"  [search] Found {chunk_count} chunks across {len(search_results)} result(s)")

            # Print sufficient context decision
            if "sufficient" in node_output:
                status = "✓ SUFFICIENT" if node_output["sufficient"] else "✗ INSUFFICIENT"
                print(f"  [sufficient_context] {status}")
                if not node_output.get("sufficient"):
                    fb = node_output.get("feedback", "")
                    print(f"    └─ feedback: {fb[:200]}")

            # Print final answer
            if "final_answer" in node_output:
                print(f"\n{'='*70}")
                print("FINAL ANSWER:")
                print(f"{'='*70}")
                print(node_output["final_answer"])
                print(f"{'='*70}")

    # Get final state for full trace
    final_state = await graph.aget_state(
        config={"configurable": {"thread_id": "cli-session"}}
    )
    if final_state and final_state.values:
        trace = final_state.values.get("trace", [])
        print(f"\n{'='*70}")
        print(f"AUDIT TRAIL ({len(trace)} steps):")
        print(f"{'='*70}")
        for i, entry in enumerate(trace, 1):
            print(f"  {i}. [{entry['agent']}] {entry['decision']}")
            if entry.get("detail"):
                print(f"     └─ {entry['detail'][:150]}")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG — Google Research multi-agent retrieval pipeline"
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="The query to answer",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum search iterations (default: 3)",
    )
    args = parser.parse_args()

    asyncio.run(run_query(args.query, args.max_iterations))


if __name__ == "__main__":
    main()
