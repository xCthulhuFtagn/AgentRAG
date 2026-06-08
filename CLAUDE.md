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
     ▲                                           │
     │       insufficient + iters left           │
     └───────────────────────────────────────────┘
     (re-route: planner re-plans for the missing piece)
                                                 │
                       sufficient ──► synthesis ──► END
                       insufficient + max iters ──► give_up ──► END
```

The loop re-enters at **planner**, not query_rewriter — on iteration the Planner
re-routes to the collection(s) most likely to hold the missing piece (mirrors
Google RAG Engine's loop that re-enters before its Search Plan agent). Only if
the Planner finds no relevant route does it fall through to query_rewriter's
broad fallback (search all collections, `collection=None`).

## Nodes (7 total)

| # | Node | Returns |
|---|------|---------|
| 1 | orchestrator | `Command(goto="synthesis" \| "planner")` |
| 2 | planner | `Command(goto="query_rewriter" \| "synthesis")` — `"query_rewriter"` also when iterating with no route (broad fallback); `"synthesis"` only on the initial turn with no route |
| 3 | query_rewriter | `Command(goto="search_fanout")` |
| 4 | search_fanout | `Command(goto="sufficient_context")` |
| 5 | sufficient_context | `Command(goto="synthesis" \| "planner" \| "give_up")` — `"planner"` re-routes for the missing piece |
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
- **Validation-driven retries + honest LLM-failure refusal** — every structured node calls `generate_structured(schema, prompt)` (`common.py`), not the raw LLM. Every requirement on a result is a Pydantic constraint (required fields; `RouteStep._non_empty` rejects blank `collection`/`subquery`; `SufficientContextResult._verdict_must_be_actionable` rejects a `sufficient=False` verdict with no `missing_parts`/`feedback`). A violation — or a transport error, or no tool call — raises, and `generate_structured` re-prompts with the failure text up to `STRUCTURED_MAX_RETRIES` times (one uniform path: no per-schema retry hooks). If the model still can't satisfy the schema it raises `StructuredGenerationError`; the `llm_failsafe` wrapper (applied to every node except `give_up` in `build_graph`) catches that **and** `openai.APIError` from the free-text nodes (`query_rewriter`/`synthesis`), routing to `give_up` with `llm_error` set so the refusal honestly cites the model failure instead of crashing. Non-LLM exceptions (code bugs) propagate.
- **Schema-Guided Reasoning (field order = reasoning order)** — structured output is generated field-by-field in declaration order, so a schema's fields must read like a person's reasoning chain with the verdict near the end, not the start. `SufficientContextResult` (`state.py`) is ordered `reason → draft_answer → missing_parts → sufficient → feedback`: the judge analyzes, drafts the answer, and lists gaps **before** emitting the `sufficient` boolean, and only **then** (if insufficient) the `feedback` on where to search next. The verdict is thus grounded in the reasoning it just generated. When `sufficient` was the *first* field, the judge committed to the boolean up front and rationalized it — letting "not found" drafts pass as `sufficient=True`. The field `description`s also carry the semantics (e.g. a "not found / absent" draft is never `sufficient`).
- **Logging** — `logged_node` decorator (in `common.py`, applied centrally in `build_graph`) is the single point that emits each node's `trace` entries as `logging` records under the `agentrag` logger. `setup_logging()` (`src/logging_setup.py`) is called by both `src/main.py` and `web/app.py`, so node decisions appear identically under CLI and web. `main.py` prints only the final answer (program output, not a log); the web UI's live trace stream is a separate channel.
- **Per-step token metering** — `logged_node` also meters token usage: it installs a fresh sink (a `_token_sink` `ContextVar`) before each node runs, and `_TokenUsageHandler` (an `AsyncCallbackHandler` attached to every LLM via `get_llm`'s `callbacks=`) adds each call's `prompt_tokens`/`completion_tokens` into the current sink. Works for **structured** calls too (function-calling returns a parsed object with no `usage_metadata`, so a callback is the only place to catch them), and sums across **concurrent** calls (a node's `asyncio.gather` subtasks copy the context, so they share the sink set before the gather). Totals are stamped onto each trace entry as `input_tokens`/`output_tokens` → logged inline (`[in=… out=…]`) and shown in the web UI as a second line under each step plus a per-message Σ total.

## State accumulation & DB scope

- `search_results`, `rewritten_queries`, `trace` use `operator.add` reducer — each Command.update appends. Critical for the iteration loop.
- `search_tasks` (no reducer, overwrite) carries `[{collection, query}]` for the current turn. `collection=None` → search ALL collections (the iteration **fallback**, only when the Planner couldn't re-route to a specific collection).
- `db_path` (set by `make_initial_state(db_path=...)`, default None=global `LANCE_DB_PATH`) scopes every search to one LanceDB → per-project isolation. Threaded into `vector_search`/`list_collections`.

## Multi-file search

`planner` builds one route per relevant collection; `query_rewriter` emits one `search_task` per route; `search_fanout` searches every `(collection, query)` pair in parallel (`asyncio.gather`). This is what makes cross-file multi-hop work.

## Vector DB (LanceDB)

**LanceDB** — embedded/serverless (no DB process), stores Lance columnar files on disk, async, persists across restarts. Self-contained module at `src/vectordb/`:
- `embeddings.py` — FastEmbed (ONNX, `paraphrase-multilingual-MiniLM-L12-v2`, **384d**, multilingual incl. Russian — an English-only model blinds retrieval on a non-English corpus); `embed`/`embed_batch` run sync ONNX off the loop via `asyncio.to_thread`; model cached `@lru_cache`.
- `client.py` — `get_async_db(db_path)` / `get_sync_db(db_path)`; `db_path or LANCE_DB_PATH`.
- `config.py` — `VectorDBSettings` (pydantic-settings, `vdb_settings` instance): all vectordb knobs from `.env` (path, model, chunking, search, stitching). See [Configuration](#configuration).
- `describe.py` — `describe_document(text)`: LLM reads an excerpt at index time → a 1–2 sentence content summary. Self-contained (builds its own DeepSeek client from `general_settings`, no `agents` import).
- `descriptions.py` — JSON sidecar (`{db_path}/_descriptions.json`, `table → {file, description}`) storage; `load_descriptions`/`save_descriptions`. Written at index time, read by the Planner.
- `tools.py` — `vector_search(query, collection, top_k, db_path)` and `list_collections(db_path)` as LangChain `@tool`s. **Async LanceDB gotcha**: `search()` is a coroutine — `q = await table.search(vec)` then `await q.limit(k).to_list()`. Returns chunk `text` + `_distance` (L2, default metric) + `seq` (chunk position). `gather_neighbors(collection, hit_seqs, …)` does the neighbor stitching (filter-scan by `seq`, no vector).
- `indexer.py` — `index_documents(dir, db_path)`. Hybrid extraction (LiteParse for PDF/DOCX/PPTX, `read_text` for TXT/MD) → `clean_text` (collapse ragged whitespace) → `split_text` (RecursiveCharacterTextSplitter, `CHUNK_SIZE`/`CHUNK_OVERLAP` chars, splits on para→line→sentence→word, never mid-word) → `embed_batch` → rows `{text, vector, seq}`. CLI: `python -m src.vectordb.indexer --dir docs/sample_docs`.

**Schema & layout:** one **file → one table** (collection); rows are `{text: str, vector: float[384], seq: int}` (`seq` = chunk index in the document, enables neighbor stitching). Table name = sanitized file stem via `safe_table_name()` (LanceDB allows only `[A-Za-z0-9._-]`; Cyrillic transliterated, hash fallback, collisions disambiguated). Per run each table is `drop_table` + `create_table` (no incremental upsert). No ANN index built → exhaustive search (fine at doc scale).

**Neighbor stitching (deterministic context expansion):** vector search returns the top-k *most similar* chunks, but a contiguous structural block (table of contents, reference list) splits across chunks where only the head ranks high — the tail falls below top-k and the answer truncates. After KNN, `search_fanout` calls `gather_neighbors`: each hit's `seq` becomes a window `[seq-EXPAND_PADDING, seq+EXPAND_PADDING]`; windows merge when the uncovered gap between them is `≤ BRIDGE_GAP` (effective merge distance `2*P + gap + 1`); every chunk in the merged ranges is fetched by `seq` filter-scan and the result is seq-ordered, capped at `MAX_EXPANDED`. Legacy tables without `seq` no-op (reindex to activate). `sufficient_context` shows each chunk seq-tagged so the judge sees contiguity and gaps.

**Per-file descriptions:** at index time `describe_document` (LLM) summarizes each file in 1–2 sentences; stored in `{db_path}/_descriptions.json`. The Planner reads them via `list_collections_described(db_path)` → `[{collection, description}]`, so routing sees a content summary, not just the table name. Legacy tables (indexed before the feature) → empty description; reindex to populate.

**Corpus inventory → judge & synthesis:** the same `list_collections_described` list is also injected (as `get_inventory_str` in `agents/common.py`) into the **Sufficient Context** and **Synthesis** prompts as the *complete, authoritative* inventory of the knowledge base. This closes an epistemic gap: vector search returns similar chunks but never proves it has seen every document, so for "describe/list ALL files"-type queries the judge could never confirm completeness and the loop always ended in `give_up`. With the ground-truth inventory the judge can confirm full coverage (every collection searched or summarized) and Synthesis can describe every file from its summary even where chunks are thin.

The judge uses the inventory **two ways**, and the prompt must keep them distinct (an early version conflated them and made the judge mark a "not found" answer as `sufficient=True`): (1) *"all files"* questions → coverage is complete once every collection is searched or summarized — don't demand proof of more documents. (2) *Specific* questions ("what is X") → if the answer isn't in the retrieved chunks, compare the inventory against the collections that actually appear in the retrieved context (each result block is tagged with its collection — no separate "searched" list is passed, it would be redundant) and, if any collection not yet among them could plausibly hold the answer, return `sufficient=False` naming it so the Planner re-routes there. A negative answer is only final once every plausibly-relevant collection has actually been searched. The **Planner's iteration prompt**, by contrast, *does* receive the already-searched set explicitly — it never sees `search_results`, so without it the Planner re-routes to a known-empty collection and burns iterations.

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

- **`general_settings`** (`src/config.py`) — DeepSeek + agent loop: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `MAX_ITERATIONS`, `STRUCTURED_MAX_RETRIES` (default `1`; extra clarification re-prompts on a schema-validation failure before `generate_structured` gives up → `give_up`; `0` disables).
- **`vdb_settings`** (`src/vectordb/config.py`) — the vectordb package owns its own knobs:

| Env var | Default | Meaning |
| --- | --- | --- |
| `LANCE_DB_PATH` | `./data/lancedb/_cli` | CLI/global DB dir (web overrides per project) |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | FastEmbed model (multilingual, 384d). **Changing the model → full reindex required** |
| `CHUNK_SIZE` | `1000` | chunk target (chars); new docs only |
| `CHUNK_OVERLAP` | `150` | chunk overlap (chars) |
| `DESCRIPTIONS_ENABLED` | `true` | generate LLM file summaries at index time + Planner routes with them |
| `DESCRIBE_MAX_CHARS` | `6000` | leading chars of each file sent to the LLM for its description |
| `SEARCH_TOP_K` | `5` | nearest chunks per (collection, query) before stitching |
| `EXPAND_PADDING` | `1` | neighbor stitching: window `[seq-P, seq+P]` per hit |
| `BRIDGE_GAP` | `2` | merge windows when uncovered gap ≤ this |
| `MAX_EXPANDED` | `16` | cap on stitched chunks per result |

All have defaults — only `DEEPSEEK_API_KEY` is required. Access values via the objects (`vdb_settings.search_top_k`), never module-level constants. Validation rejects bad values (e.g. `SEARCH_TOP_K=0` → `ge=1` error) at startup.

## Iteration loop

Sufficient Context Agent returns:
- Sufficient → `Command(goto="synthesis")`
- Insufficient + iters left → `Command(goto="planner")` with `feedback`, `missing_parts` (re-route)
- Insufficient + max iters → `Command(goto="give_up")` (system refusal, no LLM)

**Re-routing on iteration.** The loop re-enters at the **Planner**: with `feedback` + `iteration_count > 0` set, `planner` uses `PLANNER_ITERATION_PROMPT` to re-route to the collection(s) most likely to hold the missing piece (alternative keywords / different angle), then hands the new routes to `query_rewriter` as usual. If the Planner finds no relevant route, it goes to `query_rewriter`, which falls back to a single targeted query across ALL collections (`collection=None`). `query_rewriter` thus has two modes: rewrite the Planner's routes (initial turn **and** re-routed iterations), or the broad fallback (iteration with empty `plan_steps`). Max iterations: 3.

## Running

```bash
pip install -r requirements.txt
python -m src.vectordb.indexer --dir docs/sample_docs            # CLI corpus
python -m src.main --query "What CPU does the Project Alpha server have?"
python -m web.app                                                # web UI → http://localhost:8080
```
