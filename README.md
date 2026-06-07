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
| LLM | DeepSeek (`deepseek-chat`) | OpenAI-compatible API |
| Orchestration | LangGraph | Command-driven edgeless graph |
| Vector DB | LanceDB | Serverless, async, columnar files |
| Embeddings | FastEmbed (`BAAI/bge-small-en-v1.5`) | ONNX, no PyTorch, air-gapped friendly |
| Runtime | Full async | `ainvoke`, `astream`, `asyncio.gather` |

## Quick start

```bash
# Install
pip install -r requirements.txt

# Index sample documents
python -m src.vectordb.indexer --dir docs/sample_docs

# Run a query
python -m src.main --query "What CPU and RAM does the server for Project Alpha have?"

# Simple query (no multi-step needed)
python -m src.main --query "What is the capital of France?"
```

## Project structure

```
src/
├── agents/               # 7 agents + common LLM factory
│   ├── orchestrator.py   #   Command(goto="synthesis" | "planner")
│   ├── planner.py        #   Command(goto="query_rewriter")
│   ├── query_rewriter.py #   Command(goto="search_fanout")
│   ├── search_fanout.py  #   Command(goto="sufficient_context")
│   ├── sufficient_context.py  # Command(goto="synthesis" | "query_rewriter" | "give_up")
│   ├── synthesis.py      #   Command(goto=END)
│   └── give_up.py        #   Command(goto=END) — system refusal, no LLM
├── vectordb/             # Vector DB module
│   ├── embeddings.py     #   FastEmbed wrapper
│   ├── client.py         #   LanceDB connection
│   ├── tools.py          #   vector_search, list_collections (@tool)
│   └── indexer.py        #   Document indexing CLI
├── config.py             # DeepSeek + LanceDB + embedding settings
├── state.py              # AgentRAGState TypedDict
├── graph.py              # Edgeless StateGraph (set_entry_point only)
└── main.py               # CLI: python -m src.main --query "..."
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
