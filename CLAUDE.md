# CLAUDE.md

Agentic RAG — multi-agent retrieval pipeline on LangGraph + DeepSeek + LanceDB.

## Architecture

**Fully edgeless LangGraph graph.** All routing via `Command(goto=...)` returned from nodes. No `add_edge`, no `add_conditional_edges`, no `Send`.

Entry: `set_entry_point("orchestrator")`
Exit: `Command(goto=END)` in synthesis node.

## Graph flow

```
orchestrator ──simple──► synthesis ──► END
     │ complex
     ▼
  planner → query_rewriter → search_fanout → sufficient_context
                ▲                                  │
                │    insufficient (feedback loop)   │
                └──────────────────────────────────┘
                         sufficient → synthesis → END
```

## Key design decisions

- **No edges** — every node returns `Command(goto=...)`, zero `add_edge` calls
- **No Send** — fan-out parallelism is `asyncio.gather` inside `search_fanout`
- **TypedDict state** — not Pydantic BaseModel; uses `Annotated[list, operator.add]` reducers so `search_results`/`trace`/`rewritten_queries` accumulate across iterations
- **All async** — nodes are `async def`, tools are `async def`, streaming via `graph.astream()`
- **Only vector search tools** — `vector_search` and `list_collections` via LanceDB; no web/Wikipedia APIs

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
- Insufficient → `Command(goto="query_rewriter")` with `feedback`, `missing_parts`

Query Rewriter checks `state["feedback"]` — if set, generates targeted query for missing piece instead of rewriting all routes. Max iterations: 3 (forced sufficient on last).

## Running

```bash
pip install -r requirements.txt
python -m src.vectordb.indexer --dir docs/sample_docs
python -m src.main --query "What CPU does the Project Alpha server have?"
```
