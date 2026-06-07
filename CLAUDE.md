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
- **Structured output via function calling** — `get_structured_llm()` in `common.py` uses `with_structured_output(schema, method="function_calling")`. DeepSeek's API rejects the default json_schema `response_format` ("This response_format type is unavailable now").

## State accumulation & DB scope

- `search_results`, `rewritten_queries`, `trace` use `operator.add` reducer — each Command.update appends. Critical for the iteration loop.
- `search_tasks` (no reducer, overwrite) carries `[{collection, query}]` for the current turn. `collection=None` → search ALL collections (used on iteration, when the missing piece's file is unknown).
- `db_path` (set by `make_initial_state(db_path=...)`, default None=global `LANCE_DB_PATH`) scopes every search to one LanceDB → per-project isolation. Threaded into `vector_search`/`list_collections`.

## Multi-file search

`planner` builds one route per relevant collection; `query_rewriter` emits one `search_task` per route; `search_fanout` searches every `(collection, query)` pair in parallel (`asyncio.gather`). This is what makes cross-file multi-hop work.

## vectordb module

Self-contained at `src/vectordb/`:
- `embeddings.py` — FastEmbed (ONNX, BAAI/bge-small-en-v1.5, 384d)
- `client.py` — LanceDB async/sync connections (`get_async_db(db_path)`)
- `tools.py` — `@tool` wrappers; **async LanceDB**: `await table.search(vec)` then `.limit().to_list()`
- `indexer.py` — hybrid text extraction: LiteParse for PDF/DOCX/PPTX, `read_text()` for TXT/MD. CLI: `python -m src.vectordb.indexer --dir docs/sample_docs`

## web module (NiceGUI)

`web/` is a sibling top-level package — absolute imports `from src... import ...`, one-directional (`web → src`). Run: `python -m web.app`.
- `app.py` — NiceGUI UI passed as `ui.run(root=index)` (script mode needs a root function, not `@ui.page`). Green theme; chat freezes (blue + tremble via inlined `static/style.css`) while a project reindexes.
- `projects.py` — `ProjectStore`: filesystem CRUD. `data/projects/{id}/{meta.json,files/}` + `data/lancedb/{id}/`. File list read from disk (no drift).
- `runtime.py` — `GRAPH` (built once), `STORE`, per-project status + `asyncio.Lock`.
- `indexing.py` — `reindex_project(id)`: wipe `data/lancedb/{id}`, re-run `index_documents`; status `reindexing`→`idle`.
- `chat.py` — `run_chat(project_id, query)`: fresh `thread_id` per message (no state bleed), streams `(trace|answer)` from `graph.astream`.

## DeepSeek API

OpenAI-compatible endpoint at `https://api.deepseek.com/v1`. Model: `deepseek-chat`. Key from VSCode settings → `.env`. LLM factory cached via `@lru_cache` in `src/agents/common.py`.

## Iteration loop

Sufficient Context Agent returns:
- Sufficient → `Command(goto="synthesis")`
- Insufficient + iters left → `Command(goto="query_rewriter")` with `feedback`, `missing_parts`
- Insufficient + max iters → `Command(goto="give_up")` (system refusal, no LLM)

Query Rewriter checks `state["feedback"]` — if set, generates targeted query for missing piece instead of rewriting all routes. Max iterations: 3.

## Running

```bash
pip install -r requirements.txt
python -m src.vectordb.indexer --dir docs/sample_docs            # CLI corpus
python -m src.main --query "What CPU does the Project Alpha server have?"
python -m web.app                                                # web UI → http://localhost:8080
```
