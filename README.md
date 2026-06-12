# Agentic RAG — Multi-Agent Retrieval Pipeline

Implementation of Google Research's [Agentic RAG](https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/) (Gemini Enterprise Agent Platform) on LangGraph + DeepSeek + LanceDB.

**Key insight:** vanilla RAG gives up after one search. Agentic RAG iteratively searches, checks if context is sufficient, explicitly states *what's missing*, and searches again — up to 34% accuracy improvement on FramesQA.

## Architecture

Fully edgeless LangGraph graph — zero `add_edge` calls, all routing via `Command(goto=...)`.

**Pure RAG — single functionality.** Every query goes through retrieval: no
orchestrator / complexity gate (no answering without searching) and no fallbacks
(no broad search-all, no general-knowledge answer). If nothing in the corpus is
relevant, the system refuses honestly via Give Up.

```
      planner ◄── entry_point ◄────────────────────┐
        │                                          │
        ├─ no relevant collection:                 │
        │    Command(goto="give_up") → END          │
        │ Command(goto="query_rewriter")            │
        ▼                                          │
      query_rewriter                               │
        │ Command(goto="search_fanout")            │
        ▼                                          │
      search_fanout                                │
        │ Command(goto="sufficient_context")       │
        ▼                                          │
      sufficient_context ──────────────────────────┘
        │
        ├─ insufficient + iters left:
        │    Command(goto="planner")  ← re-route to the collection
        │                                holding the missing piece
        ├─ sufficient:
        │    Command(goto="synthesis") → END
        │
        └─ insufficient + max iters:
             Command(goto="give_up") → END
```

On iteration the loop re-enters at the **Planner**, which re-routes to the
collection(s) most likely to hold the missing piece (mirrors Google RAG
Engine's loop that re-enters before its Search Plan agent). If the Planner
finds no relevant route — initial turn or iteration — it goes straight to
**Give Up** (no broad fallback).

### 6 agents

| Agent | Role |
|-------|------|
| **Planner** | Breaks query into search routes: `[(collection, subquery), ...]`; no relevant collection → Give Up |
| **Query Rewriter** | Rewrites each route into a search-optimized query (routes rewritten concurrently via `asyncio.gather`); one search task per route |
| **Search Fanout** | Parallel vector search via `asyncio.gather` in LanceDB |
| **Sufficient Context** | Decides ONE thing: *would another search of the corpus materially improve the answer to the question as asked?* Answer found — sufficient; corpus exhausted on the topic — also sufficient (the answer states what the sources contain, even if thin); zero findings — never. Copies the question verbatim first (anti-inflation anchor), describes any gap in information terms (never names collections — routing is the Planner's job, enforced by validation); reads the inventory + code-computed search statistics |
| **Synthesis** | Generates final answer with source citations; can describe every document from the inventory + retrieved chunks |
| **Give Up** | System-generated refusal when context is exhausted; no LLM call |

### Stack

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | DeepSeek (`deepseek-chat`) | OpenAI-compatible API; structured output via function calling |
| Orchestration | LangGraph | Command-driven edgeless graph |
| Vector DB | LanceDB | Serverless, async, columnar files; per-project isolation |
| Embeddings | FastEmbed (`paraphrase-multilingual-MiniLM-L12-v2`) | ONNX, no PyTorch, multilingual (incl. Russian), air-gapped friendly |
| Parsing | LiteParse + read_text | PDF/DOCX/PPTX via LiteParse (Rust), TXT/MD direct |
| Web UI | NiceGUI | Python-only, WebSocket live trace, no JS toolchain |
| Runtime | Full async | `ainvoke`, `astream`, `asyncio.gather` |

## Quick start

```bash
# Install
pip install -r requirements.txt

# Configure (only DEEPSEEK_API_KEY is required; see Configuration)
cp .env.example .env && $EDITOR .env

# ── Web UI (projects + chat) ──
python -m web.app                       # → http://localhost:8080

# ── Or the CLI ──
python -m src.vectordb.indexer --dir docs/sample_docs
python -m src.main --query "What CPU and RAM does the server for Project Alpha have?"
```

### Optional: better OCR for scanned / image documents

By default LiteParse OCRs image content with its built-in **Tesseract**, which is
weak on Cyrillic and noisy on tiny images (`Image too small to scale!!`). For
better Russian OCR, run LiteParse's **EasyOCR sidecar** (from the upstream repo)
and point the app at it:

```bash
# clone upstream and run the ready-made EasyOCR server (its own deps: torch/opencv)
git clone https://github.com/run-llama/liteparse
cd liteparse/ocr/easyocr
uv run server.py                         # serves EasyOCR on http://localhost:8828
```

Then in the project's `.env` uncomment `OCR_SERVER_URL=http://localhost:8828/ocr`
and `OCR_LANGUAGE=ru` and (re)index — LiteParse routes OCR to the sidecar. It runs
in its own env (keeps torch out of the main app) and works offline once models are
cached. The sidecar must be running before you index. **By default** `OCR_SERVER_URL`
is unset, so OCR works out of the box with the built-in Tesseract — no sidecar
needed. PaddleOCR (`ocr/paddleocr`, `:8829`) works the same way — just change the URL.

## Web interface (NiceGUI)

A Python-only UI: **projects on the left, chat on the right**.

- Create / rename / delete / open projects (green theme).
- Each project holds uploaded files (`.pdf/.docx/.pptx/.txt/.md`). **"Edit files"** opens a staging session: add / rename / delete as many as you want — nothing touches disk yet. **Done & reindex** applies everything at once (one reindex); **Cancel** discards.
- That single reindex **freezes** the chat (turns blue, trembles, snows ❄) until it finishes — sending is blocked, but you can still open/return to the frozen chat to watch it. The freeze tracks the project's reindex status live, so switching chats and back keeps it correct.
- Chat streams the **live agent trace** (planner → rewrite → search → sufficient) then the final answer.
- Projects are **isolated** — each has its own LanceDB, so search never leaks across projects.

## Project structure

```
src/                      # RAG engine (graph + vectordb), unchanged by the UI
├── agents/               # 6 agents + common LLM factory (get_structured_llm)
│   ├── planner.py        #   Command(goto="query_rewriter" | "give_up") — re-routes on iteration
│   ├── query_rewriter.py #   Command(goto="search_fanout") — one search_task per route (gather)
│   ├── search_fanout.py  #   Command(goto="sufficient_context") — all (collection,query) pairs
│   ├── sufficient_context.py  # Command(goto="synthesis" | "planner" | "give_up")
│   ├── synthesis.py      #   Command(goto=END)
│   └── give_up.py        #   Command(goto=END) — system refusal, no LLM
├── vectordb/             # embeddings, LanceDB client, @tools, hybrid indexer
├── config.py · state.py · graph.py · main.py

web/                      # NiceGUI UI — imports from src/ (web → src, one-directional)
├── app.py                #   ui.run(root=index): projects + chat, green theme, frozen-chat CSS
├── projects.py           #   ProjectStore — filesystem CRUD (data/projects, data/lancedb)
├── runtime.py            #   GRAPH (built once), STORE, per-project status + locks
├── indexing.py           #   reindex_project() — wipe + rebuild project DB
├── chat.py               #   run_chat() — streams astream events, fresh thread per message
└── static/style.css      #   green theme + .frozen (blue + tremble)
```

## How the iteration loop works

1. Sufficient Context Agent decides **one question**: *would one more search of the corpus materially improve the answer to the question as asked?* It is a retrieval-state call, not a grade against an ideal answer:
   - **Answer found** → sufficient (don't keep searching for "more details" the user never asked for; the schema's first field is a verbatim copy of the question — a copy-not-generate anchor against question inflation)
   - **Corpus exhausted on the topic** (every plausible collection searched to diminishing returns) → also sufficient: *"the sources contain only …"* is the system's honest answer, even if the findings are thin
   - **Zero findings** → never sufficient; once no routes remain, the system refuses honestly
   - **Concrete reason to expect more** (an unsearched plausible collection, an untried search angle) → insufficient, with the **information gap** described: *what fact* is missing, *what was found instead*, *what alternative phrasings* might name it
2. If insufficient + iterations left → returns `Command(goto="planner")` with `missing_parts` and `feedback` following a strict template: *«Не хватает: …. Найдено вместо этого: …. Альтернативные формулировки: ….»* — pure information language. Naming a collection ("search in Y") is a **validation error**: the judge says *what* is missing, the Planner decides *where* to look (separation of concerns, enforced by schema + re-prompt)
3. Planner re-routes: it re-plans to the collection(s) most likely to hold the missing piece (alternative keywords / different angle), then hands the new routes to the Query Rewriter — which is shown the queries already executed against each collection, so iteration rewrites stop converging to the same bag of words. If no relevant route exists — or every plausible collection is already exhausted — the Planner returns empty steps and goes straight to Give Up (pure RAG — no broad fallback)
4. Search Fanout searches again → Sufficient Context checks again
5. If max iterations reached and still insufficient → `Command(goto="give_up")`
6. Give Up node builds an honest refusal: what was searched, what was found, what's missing, why

Both loop participants read **mechanical search statistics** computed by code from the record of executed searches (empty searches are recorded too — never reconstructed by the model from chunk tags, which weak models hallucinate about): the judge sees the searched set, the last-search novelty delta (*«обыскана 3 раза, последний поиск дал +0 новых чанков»* — a diminishing-returns exhaustion signal) and the executed queries (grounding its "untried angle" call); the Planner sees per-collection coverage (*«извлечено 28/210 чанков (13%)»*) plus the same delta as its stop signal — low coverage explicitly does **not** mean "barely explored": a similar query just re-returns the same top chunks.

The Sufficient Context Agent also receives the **complete corpus inventory** (every collection + its description) as ground truth. Without it, "describe all the files in the knowledge base"-type queries could never satisfy the judge — vector search returns similar chunks but never proves it has seen *every* document, so the loop always ran to `give_up`. With the inventory the judge can confirm full coverage, and for specific questions a *negative* answer only becomes final once every plausibly-relevant collection has actually been searched. Synthesis uses the same inventory to describe each document from its summary.

## Vector store (LanceDB)

**Why LanceDB:** embedded and serverless — no separate database process. It stores data as [Lance](https://lancedb.github.io/lance/) columnar files directly on disk, so it works offline / air-gapped and persists across restarts. All access is async.

### How documents become vectors

1. **Extract** text — hybrid: LiteParse (Rust) for `.pdf/.docx/.pptx`, direct `read_text()` for `.txt/.md` ([indexer.py](src/vectordb/indexer.py)).
2. **Chunk** — boundary-aware (`RecursiveCharacterTextSplitter`): text is cleaned (ragged PDF whitespace collapsed) then split on paragraph → line → sentence → word boundaries, `CHUNK_SIZE` chars (default `1000`) with `CHUNK_OVERLAP` overlap (default `150`). Never cuts mid-word/sentence.
3. **Embed** — FastEmbed `paraphrase-multilingual-MiniLM-L12-v2` (ONNX, **384 dims**, multilingual incl. Russian), batched, run off the event loop via `asyncio.to_thread` ([embeddings.py](src/vectordb/embeddings.py)).
4. **Store** — each chunk is a row `{text, vector, seq}` (`seq` = the chunk's position in the document, used for neighbor stitching). **One file → one table** (a "collection"). The table name is the file stem, sanitized to LanceDB's allowed charset (Cyrillic transliterated → `big_statya`, hash fallback otherwise).
5. **Describe** — an LLM reads an excerpt and writes a 1–2 sentence summary of the file, stored in `{db}/_descriptions.json` ([describe.py](src/vectordb/describe.py)). The **Planner** sees these descriptions (not just table names) so it routes queries to the right source.

### How it's stored on disk

```
data/
├── projects/{project_id}/          # web: meta.json + uploaded files/
├── lancedb/{project_id}/           # web: one isolated DB per project
│   └── {table}.lance/              # one table (Lance dataset) per file
│       ├── data/ … (columnar fragments: text + 384-d vector)
│       └── _versions, _transactions  # Lance manifest (versioned, ACID)
├── lancedb/_cli/                   # CLI default (LANCE_DB_PATH) — same data/ root
└── fastembed_cache/                # embedding model (ONNX, ~252MB), downloaded once
```
Everything lives under `data/` (created automatically). The CLI's global DB is just another dir under `data/lancedb/` (`_cli`), kept apart from per-project DBs whose names are project UUIDs.

### How it's queried

- `get_async_db(db_path)` opens a connection scoped to one directory ([client.py](src/vectordb/client.py)). The `db_path` is threaded from graph state, so each project searches **only its own DB** (isolation).
- `vector_search(query, collection, top_k, db_path)` embeds the query, then `await table.search(vec)` → `.limit(top_k).to_list()`, returning the chunk texts, `_distance` scores (L2, LanceDB default), and `seq` positions. `list_collections(db_path)` returns the table names so the Planner knows which files exist ([tools.py](src/vectordb/tools.py)).
- **Neighbor stitching** — vector search returns the most *similar* chunks, but a contiguous block (table of contents, reference list) splits across chunks where only the head ranks high; the tail falls below `top_k` and the answer truncates. After KNN, `gather_neighbors` expands each hit to its contiguous `seq`-neighborhood (`[seq-EXPAND_PADDING, seq+EXPAND_PADDING]`, merging windows whose gap ≤ `BRIDGE_GAP`, fetched by `seq` filter-scan, capped at `MAX_EXPANDED`) so whole blocks come back. Deterministic, no LLM. Tables indexed before the `seq` column no-op until reindexed.
- Search is exhaustive (no ANN index is built) — fine at document scale; add `table.create_index()` if a corpus grows large.

### Reindexing & persistence

- Editing a project's files commits a batch, then **reindexes**: the project's `data/lancedb/{id}` dir is wiped and rebuilt from the current files ([indexing.py](web/indexing.py)). This keeps the index consistent with deletes/renames. Each table is also `drop_table`-then-`create_table` on every run.
- Data persists between runs — restart the app/CLI and the tables are still there. Deleting a project removes both its files and its LanceDB dir.

## Configuration

All settings are **pydantic-settings** classes — typed, validated, loaded from `.env` (or the process environment). Env var names are the UPPERCASE field names. Every value has a default, so **only `DEEPSEEK_API_KEY` is strictly required**; bad values are rejected at startup (e.g. `SEARCH_TOP_K=0` fails the `≥1` check).

There are two scopes:

- **`general_settings`** ([src/config.py](src/config.py)) — DeepSeek API + the agent loop.
- **`vdb_settings`** ([src/vectordb/config.py](src/vectordb/config.py)) — the vectordb package owns its own knobs (path, embeddings, chunking, search, stitching).

```ini
# ── DeepSeek API (general_settings) ──
DEEPSEEK_API_KEY=sk-...                      # required
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_CONNECTION_RETRIES=3                # tenacity retries on 429/5xx/connection drops (0 disables)
DEEPSEEK_RETRY_BACKOFF_FACTOR=1.0            # initial backoff seconds, doubles per retry; 429 Retry-After honored

# ── Agent loop (general_settings) ──
MAX_ITERATIONS=3                             # max search→check retries before give_up

# ── Vector DB (vdb_settings) ──
LANCE_DB_PATH=./data/lancedb/_cli            # CLI/global DB dir (web overrides per project)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2   # ⚠ multilingual (RU+), 384d; changing model → full reindex
EMBEDDING_CACHE_DIR=./data/fastembed_cache   # where the ~252MB ONNX model lives (FastEmbed's default /tmp is wiped on reboot)
CHUNK_SIZE=1000                              # chunk target, chars (new docs only)
CHUNK_OVERLAP=150                            # chunk overlap, chars
DESCRIPTIONS_ENABLED=true                    # LLM file summary at index time; Planner routes with it
SEARCH_TOP_K=5                               # nearest chunks per (collection, query) before stitching

# ── Neighbor stitching (vdb_settings) ──
EXPAND_PADDING=1                             # window [seq-P, seq+P] around each hit
BRIDGE_GAP=2                                 # merge windows when the uncovered gap ≤ this
MAX_EXPANDED=16                              # cap on stitched chunks per result
```

### Tuning the vector DB

- **Retrieval recall vs. noise** — raise `SEARCH_TOP_K` to pull more candidate chunks per query (better recall, more context/noise), lower it to stay tight.
- **Whole-block retrieval** — if answers to "list the whole X" (table of contents, references) still truncate, raise `EXPAND_PADDING` (reach further from each hit) or `BRIDGE_GAP` (tolerate larger holes between relevant regions). The effective hit-merge distance is `2*EXPAND_PADDING + BRIDGE_GAP + 1`. `MAX_EXPANDED` caps how much a single result can grow.
- **Chunk granularity** — larger `CHUNK_SIZE` packs more context per chunk (fewer, coarser chunks → better for whole-section reads, worse precision); smaller is the opposite. `CHUNK_OVERLAP` carries context across boundaries. Changing either only affects **newly indexed** documents — reindex to apply.
- **Embedding model** — `EMBEDDING_MODEL` is a footgun: a different model means a different vector dimension, so existing tables become incompatible. **Reindex every project after changing it.**

> Changes to chunking/embedding settings require a **reindex** to take effect (web: *Edit files → Done & reindex*; CLI: re-run the indexer). Search/stitching settings (`SEARCH_TOP_K`, `EXPAND_PADDING`, `BRIDGE_GAP`, `MAX_EXPANDED`) apply immediately on the next query — no reindex needed.
