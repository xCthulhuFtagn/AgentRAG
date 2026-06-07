# Agentic RAG — Multi-Agent Retrieval Pipeline

Implementation of Google Research's [Agentic RAG](https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/) (Gemini Enterprise Agent Platform) on LangGraph + DeepSeek + LanceDB.

**Key insight:** vanilla RAG gives up after one search. Agentic RAG iteratively searches, checks if context is sufficient, explicitly states *what's missing*, and searches again — up to 34% accuracy improvement on FramesQA.

## Architecture

Fully edgeless LangGraph graph — zero `add_edge` calls, all routing via `Command(goto=...)`.

```
orchestrator ◀── entry_point
  │
  ├─ Command(goto="synthesis")          ← simple query
  └─ Command(goto="planner")            ← complex query
        │
        ▼
      planner → Command(goto="query_rewriter")
        │
        ▼
      query_rewriter ◄──────────────────────────┐
        │ Command(goto="search_fanout")          │
        ▼                                        │
      search_fanout                              │
        │ Command(goto="sufficient_context")     │
        ▼                                        │
      sufficient_context ────────────────────────┘
        │
        ├─ insufficient + iters left:
        │    Command(goto="query_rewriter")
        │
        ├─ sufficient:
        │    Command(goto="synthesis") → END
        │
        └─ insufficient + max iters:
             Command(goto="give_up") → END
```

### 7 agents

| Agent | Role |
|-------|------|
| **Orchestrator** | Assesses query complexity; simple → synthesis, complex → planner |
| **Planner** | Breaks query into search routes: `[(collection, subquery), ...]` |
| **Query Rewriter** | Rewrites routes into search-optimized queries; handles feedback from iteration |
| **Search Fanout** | Parallel vector search via `asyncio.gather` in LanceDB |
| **Sufficient Context** | Checks (1) snippets (2) draft answer (3) missing pieces → commands next step |
| **Synthesis** | Generates final answer with source citations |
| **Give Up** | System-generated refusal when context is exhausted; no LLM call |

### Stack

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | DeepSeek (`deepseek-chat`) | OpenAI-compatible API; structured output via function calling |
| Orchestration | LangGraph | Command-driven edgeless graph |
| Vector DB | LanceDB | Serverless, async, columnar files; per-project isolation |
| Embeddings | FastEmbed (`BAAI/bge-small-en-v1.5`) | ONNX, no PyTorch, air-gapped friendly |
| Parsing | LiteParse + read_text | PDF/DOCX/PPTX via LiteParse (Rust), TXT/MD direct |
| Web UI | NiceGUI | Python-only, WebSocket live trace, no JS toolchain |
| Runtime | Full async | `ainvoke`, `astream`, `asyncio.gather` |

## Quick start

```bash
# Install
pip install -r requirements.txt

# ── Web UI (projects + chat) ──
python -m web.app                       # → http://localhost:8080

# ── Or the CLI ──
python -m src.vectordb.indexer --dir docs/sample_docs
python -m src.main --query "What CPU and RAM does the server for Project Alpha have?"
```

## Web interface (NiceGUI)

A Python-only UI: **projects on the left, chat on the right**.

- Create / rename / delete / open projects (green theme).
- Each project holds uploaded files (`.pdf/.docx/.pptx/.txt/.md`) — add / rename / delete.
- Any file change **reindexes** the project's vector DB; the chat **freezes** (turns blue and trembles) until reindexing finishes. "Open in chat" is disabled for a reindexing project.
- Chat streams the **live agent trace** (orchestrator → planner → search → sufficient) then the final answer.
- Projects are **isolated** — each has its own LanceDB, so search never leaks across projects.

## Project structure

```
src/                      # RAG engine (graph + vectordb), unchanged by the UI
├── agents/               # 7 agents + common LLM factory (get_structured_llm)
│   ├── orchestrator.py   #   Command(goto="synthesis" | "planner")
│   ├── planner.py        #   Command(goto="query_rewriter")
│   ├── query_rewriter.py #   Command(goto="search_fanout") — one search_task per route
│   ├── search_fanout.py  #   Command(goto="sufficient_context") — all (collection,query) pairs
│   ├── sufficient_context.py  # Command(goto="synthesis" | "query_rewriter" | "give_up")
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

1. Sufficient Context Agent checks three things:
   - **Retrieved snippets** — do they contain the needed facts?
   - **Draft answer** — can we construct a complete answer?
   - **Missing pieces** — *what exactly* is missing and *where* to find it
2. If insufficient + iterations left → returns `Command(goto="query_rewriter")` with `feedback="search for X in Y"`
3. Query Rewriter sees feedback → generates a targeted query for the missing piece
4. Search Fanout searches again → Sufficient Context checks again
5. If max iterations reached and still insufficient → `Command(goto="give_up")`
6. Give Up node builds an honest refusal: what was found, what's missing, why

## Configuration

Copy settings from VSCode user settings or set in `.env`:

```
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
LANCE_DB_PATH=./lancedb_data
MAX_ITERATIONS=3
```
