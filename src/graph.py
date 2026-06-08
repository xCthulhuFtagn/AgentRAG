"""LangGraph StateGraph — fully edgeless Agentic RAG pipeline.

Zero add_edge calls. All routing via Command(goto=...).
No Send — fan-out via asyncio.gather inside search_fanout.
Entry/exit: set_entry_point + Command(goto=END).
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentRAGState
from src.agents.common import logged_node
from src.agents.orchestrator import orchestrator_node
from src.agents.planner import planner_node
from src.agents.query_rewriter import query_rewriter_node
from src.agents.search_fanout import search_fanout_node
from src.agents.sufficient_context import sufficient_context_node
from src.agents.synthesis import synthesis_node
from src.agents.give_up import give_up_node


def build_graph() -> StateGraph:
    """Build the edgeless Agentic RAG graph.

    Every node returns Command(goto=...). Zero edges. Pure Command-driven flow.

        orchestrator ◀── entry_point
          │
          ├⟶ Command(goto="synthesis")
          │
          └⟶ Command(goto="planner")
                │
                ▼
              planner ◄───────────────────────────┐
                │                                  │
                └⟶ Command(goto="query_rewriter")   │
                      │  (or "synthesis" if no route)│
                      ▼                             │
                  query_rewriter                    │
                      │                             │
                      └⟶ Command(goto="search")      │
                            │                        │
                            ▼                        │
                        search_fanout                │
                            │                        │
                            └⟶ Command(goto=         │
                                "sufficient")        │
                                  │                  │
                                  ▼                  │
                          sufficient_context ────────┘
                            │        insufficient + iters left:
                            │        Command(goto="planner") — re-route to the
                            │        collection holding the missing piece
                            │
                            ├⟶ sufficient:
                            │  Command(goto="synthesis")
                            │     │
                            │     ▼
                            │  synthesis → Command(goto=END)
                            │
                            └⟶ insufficient + max iters:
                               Command(goto="give_up")
                                  │
                                  ▼
                               give_up → Command(goto=END)
    """
    workflow = StateGraph(AgentRAGState)

    # logged_node wraps each node so its trace entries are emitted as logs —
    # one choke point, identical under CLI and web.
    nodes = {
        "orchestrator": orchestrator_node,
        "planner": planner_node,
        "query_rewriter": query_rewriter_node,
        "search_fanout": search_fanout_node,
        "sufficient_context": sufficient_context_node,
        "synthesis": synthesis_node,
        "give_up": give_up_node,
    }
    for name, node in nodes.items():
        workflow.add_node(name, logged_node(node))

    workflow.set_entry_point("orchestrator")

    return workflow.compile(checkpointer=MemorySaver())
