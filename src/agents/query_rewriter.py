"""Query Rewriter Agent — rewrites queries for precise vector search.

Returns Command(goto="search_fanout") — edgeless routing.
Processes all plan_steps at once, or handles iteration feedback.
"""

import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, make_trace_entry
from src.agents.common import get_llm
from src.llm_retry import ainvoke_with_retry

REWRITER_PROMPT = """Ты — Агент-Переписчик Запросов (Query Rewriter) в системе Agentic RAG.

Твоя задача: превратить поисковый маршрут в точный запрос, оптимизированный для векторного поиска.

Исходный вопрос пользователя: {original_query}
Текущая цель поиска: коллекция '{collection}'
Что ищем: {subquery}

Запросы, которые по этой коллекции УЖЕ выполнялись (они извлекли всё, что могли;
близкий к ним запрос вернёт те же фрагменты — НЕ повторяй их формулировки):
{tried}

Правила:
1. Будь конкретен, используй ключевые слова, которые вероятно встречаются в документах
2. Убери вопросительные слова (кто, что, почему) — используй утвердительную форму
3. Добавь синонимы и связанные термины
4. Если выше перечислены уже выполненные запросы — возьми ДРУГОЙ угол: другие
   термины, конкретные названия/имена из «что ищем», а не пересказ тех же слов
5. Будь краток (максимум 1-2 предложения)

Верни ТОЛЬКО текст переписанного поискового запроса, ничего больше."""


def _queries_already_tried(search_results: list[dict], collection: str) -> list[str]:
    """Queries already executed against this collection (run order, deduped).

    Mechanical input for the rewriter: without it, iteration rewrites converge
    to the same bag of words (the Пушкин trace ran four near-identical queries)
    and every repeat search returns the same chunks.
    """
    seen: set[str] = set()
    tried: list[str] = []
    for r in search_results or []:
        if r.get("collection") != collection or r.get("error"):
            continue
        query = (r.get("subquery") or "").strip()
        if query and query not in seen:
            seen.add(query)
            tried.append(query)
    return tried


async def _rewrite_route(
    llm, original_query: str, step: dict, tried: list[str]
) -> tuple[str, str]:
    """Rewrite one plan route into a search-optimized query.

    Returns (collection, query). Module-level so it isn't redefined per node
    call; routes are rewritten concurrently via asyncio.gather.
    """
    collection = step.get("collection", "unknown")
    prompt = REWRITER_PROMPT.format(
        original_query=original_query,
        collection=collection,
        subquery=step.get("subquery", original_query),
        tried="\n".join(f"- «{q}»" for q in tried) or "(пока никаких)",
    )
    result: str = (await ainvoke_with_retry(llm, prompt)).content.strip().strip('"')
    return collection, result


async def query_rewriter_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Query Rewriter: produce search-optimized queries, command search_fanout.

    The Planner guarantees ≥1 route (no relevant collection → give_up), so we
    always rewrite its plan — one search task per route, no fallback modes.
    Routes are independent → rewrite them concurrently (asyncio.gather), which
    preserves order so rewritten[i]/search_tasks[i] align with plan_steps[i].
    """
    llm = get_llm(temperature=0.3)
    plan_steps = state.get("plan_steps", [])
    search_results = state.get("search_results", [])

    pairs = await asyncio.gather(
        *[
            _rewrite_route(
                llm,
                state["query"],
                s,
                _queries_already_tried(search_results, s.get("collection", "unknown")),
            )
            for s in plan_steps
        ]
    )
    rewritten = [q for _, q in pairs]
    search_tasks = [{"collection": c, "query": q} for c, q in pairs]

    trace_entry = make_trace_entry(
        agent="query_rewriter",
        decision=f"{len(rewritten)} queries",
        detail=str(search_tasks),
    )

    return Command(
        goto="search_fanout",
        update={
            "rewritten_queries": rewritten,
            "search_tasks": search_tasks,
            "trace": [trace_entry],
        },
    )
