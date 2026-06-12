"""LangGraph StateGraph — fully edgeless Agentic RAG pipeline.

Zero add_edge calls. All routing via Command(goto=...).
No Send — fan-out via asyncio.gather inside search_fanout.
Entry/exit: set_entry_point + Command(goto=END).
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentRAGState
from src.agents.common import logged_node, llm_failsafe
from src.agents.planner import planner_node
from src.agents.query_rewriter import query_rewriter_node
from src.agents.search_fanout import search_fanout_node
from src.agents.sufficient_context import sufficient_context_node
from src.agents.synthesis import synthesis_node
from src.agents.give_up import give_up_node


def build_graph() -> StateGraph:
    """Build the edgeless Agentic RAG graph.

    Every node returns Command(goto=...). Zero edges. Pure Command-driven flow.
    Pure RAG: every query goes through retrieval — no orchestrator/complexity
    gate, no general-knowledge fallback. Refusals are evidence-based: the
    planner probes an implausible-looking corpus instead of refusing it
    unsearched, so give_up without a search needs an empty knowledge base.

        planner ◄── entry_point ◄───────────────┐
          │                                      │
          ├⟶ empty KB / iteration exhausted:     │
          │  Command(goto="give_up")             │
          │                                      │
          └⟶ Command(goto="query_rewriter")       │
                │                                 │
                ▼                                 │
            query_rewriter                        │
                │                                 │
                └⟶ Command(goto="search_fanout")   │
                      │                            │
                      ▼                            │
                  search_fanout                    │
                      │                            │
                      └⟶ Command(goto=             │
                          "sufficient_context")    │
                            │                      │
                            ▼                      │
                    sufficient_context ────────────┘
                      │        insufficient + iters left:
                      │        Command(goto="planner") — re-route to the
                      │        collection holding the missing piece
                      │
                      ├⟶ sufficient:
                      │  Command(goto="synthesis") → Command(goto=END)
                      │
                      └⟶ insufficient + max iters:
                         Command(goto="give_up") → Command(goto=END)
    """
    workflow = StateGraph(AgentRAGState)

    # logged_node wraps each node so its trace entries are emitted as logs —
    # one choke point, identical under CLI and web. llm_failsafe wraps every
    # node EXCEPT give_up so an unrecoverable model failure routes to give_up
    # (honest refusal) instead of crashing the run; give_up uses no LLM and must
    # not redirect to itself. Order: logged_node outside, so a node's tokens
    # spent before failing are still metered onto the give_up redirect entry.
    nodes = {
        "planner": planner_node,
        "query_rewriter": query_rewriter_node,
        "search_fanout": search_fanout_node,
        "sufficient_context": sufficient_context_node,
        "synthesis": synthesis_node,
        "give_up": give_up_node,
    }
    for name, node in nodes.items():
        guarded = node if name == "give_up" else llm_failsafe(name)(node)
        workflow.add_node(name, logged_node(guarded))

    workflow.set_entry_point("planner")

    return workflow.compile(checkpointer=MemorySaver())
