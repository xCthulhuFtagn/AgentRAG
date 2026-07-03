"""CLI entry point for Agentic RAG.

Usage:
    python -m src.main --query "What are the specs of the server used in Project X?"
    python -m src.main --query "Tell me about machine learning" --max-iterations 5
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import general_settings
from src.state import make_initial_state
from src.graph import build_graph
from src.logging_setup import setup_logging

log = logging.getLogger("agentrag.cli")


async def run_query(query: str, max_iterations: int = 3):
    """Run a query through the Agentic RAG pipeline.

    Per-node decisions are emitted as logs by the graph itself (see
    `logged_node`). Here we only stream to surface the final answer — the
    program's actual output — which goes to stdout, separate from the logs.
    """
    graph = build_graph()
    initial_state = make_initial_state(query=query, max_iterations=max_iterations)

    log.info("query: %s (max_iterations=%d)", query, max_iterations)

    final_answer = None
    async for event in graph.astream(initial_state, stream_mode="updates"):
        for node_output in event.values():
            if isinstance(node_output, dict) and node_output.get("final_answer"):
                final_answer = node_output["final_answer"]

    print(f"\n{'='*70}")
    print("FINAL ANSWER:")
    print(f"{'='*70}")
    print(final_answer if final_answer else "(no answer produced)")
    print(f"{'='*70}")


def main():
    setup_logging()

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
        default=general_settings.max_iterations,
        help=f"Maximum search iterations (default: {general_settings.max_iterations}, from MAX_ITERATIONS)",
    )
    args = parser.parse_args()

    asyncio.run(run_query(args.query, args.max_iterations))


if __name__ == "__main__":
    main()
