# CLAUDE.md

Agentic RAG — multi-agent retrieval pipeline on LangGraph + DeepSeek + LanceDB.

## Architecture

**Fully edgeless LangGraph graph.** All routing via `Command(goto=...)` returned from nodes. No `add_edge`, no `add_conditional_edges`, no `Send`.

Entry: `set_entry_point("orchestrator")`
Exit: `Command(goto=END)` in synthesis or give_up nodes.

## Graph flow

```
orchestrator ──simple──► synthesis ──► END
     │ complex
     ▼
  planner → query_rewriter → search_fanout → sufficient_context
                ▲                                  │
                │    insufficient + iters left      │
                └──────────────────────────────────┘
                                                   │
                         sufficient ──► synthesis ──► END
                         insufficient + max iters ──► give_up ──► END
```

## Nodes (7 total)

| # | Node | Returns |
|---|------|---------|
| 1 | orchestrator | `Command(goto="synthesis" \| "planner")` |
| 2 | planner | `Command(goto="query_rewriter" \| "synthesis")` |
| 3 | query_rewriter | `Command(goto="search_fanout")` |
| 4 | search_fanout | `Command(goto="sufficient_context")` |
| 5 | sufficient_context | `Command(goto="synthesis" \| "query_rewriter" \| "give_up")` |
| 6 | synthesis | `Command(goto=END)` |
| 7 | give_up | `Command(goto=END)` — system refusal, no LLM |

## Key design decisions

- **No edges** — every node returns `Command(goto=...)`, zero `add_edge` calls
- **No Send** — fan-out parallelism is `asyncio.gather` inside `search_fanout`
- **TypedDict state** — not Pydantic BaseModel; uses `Annotated[list, operator.add]` reducers so `search_results`/`trace`/`rewritten_queries` accumulate across iterations
- **All async** — nodes are `async def`, tools are `async def`, streaming via `graph.astream()`
- **Only vector search tools** — `vector_search` and `list_collections` via LanceDB; no web/Wikipedia APIs
- **Honest refusal** — Give Up node builds a system-generated message (no LLM) listing what was found, what's missing, and why

## State accumulation

`search_results`, `rewritten_queries`, `trace` use `operator.add` reducer — each Command.update appends, not overwrites. Critical for the iteration loop: each pass adds to previous results.

## vectordb module

Self-contained at `src/vectordb/`:
- `embeddings.py` — FastEmbed (ONNX, BAAI/bge-small-en-v1.5, 384d)
- `client.py` — LanceDB async/sync connections
- `tools.py` — `@tool` wrappers for LangChain bind_tools
- `indexer.py` — CLI: `python -m src.vectordb.indexer --dir docs/sample_docs`

## DeepSeek API

OpenAI-compatible endpoint at `https://api.deepseek.com/v1`. Model: `deepseek-chat`. Key from VSCode settings → `.env` file. LLM factory cached via `@lru_cache` in `src/agents/common.py`.

## Iteration loop

Sufficient Context Agent returns:
- Sufficient → `Command(goto="synthesis")`
- Insufficient + iters left → `Command(goto="query_rewriter")` with `feedback`, `missing_parts`
- Insufficient + max iters → `Command(goto="give_up")` (system refusal, no LLM)

Query Rewriter checks `state["feedback"]` — if set, generates targeted query for missing piece instead of rewriting all routes. Max iterations: 3.

## Running

```bash
pip install -r requirements.txt
python -m src.vectordb.indexer --dir docs/sample_docs
python -m src.main --query "What CPU does the Project Alpha server have?"
```
