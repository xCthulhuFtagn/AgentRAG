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
- **Logging** — `logged_node` decorator (in `common.py`, applied centrally in `build_graph`) is the single point that emits each node's `trace` entries as `logging` records under the `agentrag` logger. `setup_logging()` (`src/logging_setup.py`) is called by both `src/main.py` and `web/app.py`, so node decisions appear identically under CLI and web. `main.py` prints only the final answer (program output, not a log); the web UI's live trace stream is a separate channel.

## State accumulation & DB scope

- `search_results`, `rewritten_queries`, `trace` use `operator.add` reducer — each Command.update appends. Critical for the iteration loop.
- `search_tasks` (no reducer, overwrite) carries `[{collection, query}]` for the current turn. `collection=None` → search ALL collections (used on iteration, when the missing piece's file is unknown).
- `db_path` (set by `make_initial_state(db_path=...)`, default None=global `LANCE_DB_PATH`) scopes every search to one LanceDB → per-project isolation. Threaded into `vector_search`/`list_collections`.

## Multi-file search

`planner` builds one route per relevant collection; `query_rewriter` emits one `search_task` per route; `search_fanout` searches every `(collection, query)` pair in parallel (`asyncio.gather`). This is what makes cross-file multi-hop work.

## Vector DB (LanceDB)

**LanceDB** — embedded/serverless (no DB process), stores Lance columnar files on disk, async, persists across restarts. Self-contained module at `src/vectordb/`:
- `embeddings.py` — FastEmbed (ONNX, `BAAI/bge-small-en-v1.5`, **384d**); `embed`/`embed_batch` run sync ONNX off the loop via `asyncio.to_thread`; model cached `@lru_cache`.
- `client.py` — `get_async_db(db_path)` / `get_sync_db(db_path)`; `db_path or LANCE_DB_PATH`.
- `config.py` — `VectorDBSettings` (pydantic-settings, `vdb_settings` instance): all vectordb knobs from `.env` (path, model, chunking, search, stitching). See [Configuration](#configuration).
- `tools.py` — `vector_search(query, collection, top_k, db_path)` and `list_collections(db_path)` as LangChain `@tool`s. **Async LanceDB gotcha**: `search()` is a coroutine — `q = await table.search(vec)` then `await q.limit(k).to_list()`. Returns chunk `text` + `_distance` (L2, default metric) + `seq` (chunk position). `gather_neighbors(collection, hit_seqs, …)` does the neighbor stitching (filter-scan by `seq`, no vector).
- `indexer.py` — `index_documents(dir, db_path)`. Hybrid extraction (LiteParse for PDF/DOCX/PPTX, `read_text` for TXT/MD) → `clean_text` (collapse ragged whitespace) → `split_text` (RecursiveCharacterTextSplitter, `CHUNK_SIZE`/`CHUNK_OVERLAP` chars, splits on para→line→sentence→word, never mid-word) → `embed_batch` → rows `{text, vector, seq}`. CLI: `python -m src.vectordb.indexer --dir docs/sample_docs`.

**Schema & layout:** one **file → one table** (collection); rows are `{text: str, vector: float[384], seq: int}` (`seq` = chunk index in the document, enables neighbor stitching). Table name = sanitized file stem via `safe_table_name()` (LanceDB allows only `[A-Za-z0-9._-]`; Cyrillic transliterated, hash fallback, collisions disambiguated). Per run each table is `drop_table` + `create_table` (no incremental upsert). No ANN index built → exhaustive search (fine at doc scale).

**Neighbor stitching (deterministic context expansion):** vector search returns the top-k *most similar* chunks, but a contiguous structural block (table of contents, reference list) splits across chunks where only the head ranks high — the tail falls below top-k and the answer truncates. After KNN, `search_fanout` calls `gather_neighbors`: each hit's `seq` becomes a window `[seq-EXPAND_PADDING, seq+EXPAND_PADDING]`; windows merge when the uncovered gap between them is `≤ BRIDGE_GAP` (effective merge distance `2*P + gap + 1`); every chunk in the merged ranges is fetched by `seq` filter-scan and the result is seq-ordered, capped at `MAX_EXPANDED`. Legacy tables without `seq` no-op (reindex to activate). `sufficient_context` shows each chunk seq-tagged so the judge sees contiguity and gaps.

**Storage & isolation:** everything under `data/` (auto-created by `get_async_db`/`get_sync_db` via `mkdir(parents=True)` and by `ProjectStore`). CLI uses global `LANCE_DB_PATH` = `./data/lancedb/_cli` (`_cli` can't collide with project UUIDs and isn't listed as a project). Web gives each project its own dir `data/lancedb/{project_id}/` and threads that `db_path` through state → `vector_search`/`list_collections`, so a project searches only its own files. Reindex = wipe `data/lancedb/{id}` + rebuild from current files. Data persists between runs.

## web module (NiceGUI)

`web/` is a sibling top-level package — absolute imports `from src... import ...`, one-directional (`web → src`). Run: `python -m web.app`.
- `app.py` — NiceGUI UI passed as `ui.run(root=index)` (script mode needs a root function, not `@ui.page`). Green theme; chat freezes (blue + tremble via inlined `static/style.css`) while a project reindexes.
- `projects.py` — `ProjectStore`: filesystem CRUD. `data/projects/{id}/{meta.json,files/}` + `data/lancedb/{id}/`. File list read from disk (no drift).
- `runtime.py` — `GRAPH` (built once), `STORE`, per-project status + `asyncio.Lock`.
- `indexing.py` — `reindex_project(id)`: wipe `data/lancedb/{id}`, re-run `index_documents`; status `reindexing`→`idle`.
- `chat.py` — `run_chat(project_id, query)`: fresh `thread_id` per message (no state bleed), streams `(trace|answer)` from `graph.astream`.

## DeepSeek API

OpenAI-compatible endpoint at `https://api.deepseek.com/v1`. Model: `deepseek-chat`. Key from VSCode settings → `.env`. LLM factory cached via `@lru_cache` in `src/agents/common.py`.

## Configuration

Settings are **pydantic-settings** `BaseSettings` classes — typed, validated, read from `.env` / process env (env var = UPPERCASE field name, case-insensitive). Two objects, two scopes:

- **`general_settings`** (`src/config.py`) — DeepSeek + agent loop: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `MAX_ITERATIONS`.
- **`vdb_settings`** (`src/vectordb/config.py`) — the vectordb package owns its own knobs:

| Env var | Default | Meaning |
| --- | --- | --- |
| `LANCE_DB_PATH` | `./data/lancedb/_cli` | CLI/global DB dir (web overrides per project) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model. **Changing it changes the vector dim → full reindex required** |
| `CHUNK_SIZE` | `1000` | chunk target (chars); new docs only |
| `CHUNK_OVERLAP` | `150` | chunk overlap (chars) |
| `SEARCH_TOP_K` | `5` | nearest chunks per (collection, query) before stitching |
| `EXPAND_PADDING` | `2` | neighbor stitching: window `[seq-P, seq+P]` per hit |
| `BRIDGE_GAP` | `1` | merge windows when uncovered gap ≤ this |
| `MAX_EXPANDED` | `20` | cap on stitched chunks per result |

All have defaults — only `DEEPSEEK_API_KEY` is required. Access values via the objects (`vdb_settings.search_top_k`), never module-level constants. Validation rejects bad values (e.g. `SEARCH_TOP_K=0` → `ge=1` error) at startup.

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
