"""Planner Agent — breaks down query into search routes.

Returns Command(goto="query_rewriter") with routes, or Command(goto="give_up")
only when refusal is justified: the knowledge base is empty, or an iteration
found every plausible collection exhausted. A claim of absence needs retrieval
evidence — a description compresses a document to 1-2 sentences and can prove
relevance, never absence — so on the INITIAL turn an implausible-looking
corpus is probed (1-2 least-implausible collections, raw query), not refused;
the judge rules on what the search actually returns. Pure RAG otherwise stands:
answers come only from retrieved context, no general-knowledge fallback.
No Send — all routes processed together in query_rewriter.
"""

import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, PlanResult, make_trace_entry
from src.agents.common import (
    collection_search_stats,
    format_search_stats_for_planner,
    generate_structured,
)
from src.vectordb.config import vdb_settings
from src.vectordb.tools import count_chunks, list_collections, list_collections_described

# Probe backstop cap: when the model declines every route on the initial turn,
# this many collections are searched with the raw query before any refusal —
# vector search is LLM-free, so the probe costs no tokens by itself.
_PROBE_LIMIT = 3

PLANNER_PROMPT = """Ты — Агент-Планировщик (Planner) в системе Agentic RAG.

Твоя задача: разбить сложный вопрос на конкретные поисковые маршруты, каждый из
которых нацелен на определённую коллекцию документов.

Сначала изучи доступные коллекции (с кратким описанием, если оно есть), чтобы решить, где искать.

Доступные коллекции:
{collections}

Затем для каждого нужного фрагмента информации создай RouteStep:
- collection: точное имя таблицы для поиска (должно совпадать с одной из доступных коллекций)
- subquery: сфокусированный поисковый запрос для этого фрагмента
- rationale: почему эта коллекция релевантна

Вопрос пользователя: {query}

Если вопрос требует информации из нескольких коллекций — создай несколько шагов.
Если на вопрос можно ответить по одной коллекции — создай один шаг.
Если ни одна коллекция не выглядит релевантной — НЕ отказывайся: описание сжимает
документ до 1–2 предложений и не может доказать ОТСУТСТВИЕ чего-либо в тексте.
Выбери 1–2 наименее неправдоподобные коллекции (по любым зацепкам: сущности из
вопроса, смежные области) и создай зонд-маршруты — отсутствие подтверждается
только реальным поиском, и вердикт по его результатам вынесет судья."""

PLANNER_ITERATION_PROMPT = """Ты — Агент-Планировщик (Planner) в системе Agentic RAG.

Это ИТЕРАЦИЯ — предыдущие поиски нашли не всё. Перенаправь поиск в ту коллекцию
(или коллекции), где вероятнее всего находится НЕДОСТАЮЩАЯ информация.

Исходный вопрос пользователя: {query}
Всё ещё не хватает: {missing_parts}
Обратная связь от проверяющего контекст: {feedback}

Обратная связь описывает ИНФОРМАЦИОННЫЙ пробел: чего не хватает, что нашлось
вместо него и какими альтернативными формулировками искомое может называться
в документах. Она НЕ указывает коллекцию — выбор, ГДЕ искать, целиком твой:
сопоставь пробел с описаниями коллекций и статистикой ниже. Альтернативные
формулировки из обратной связи используй в subquery.

Доступные коллекции:
{collections}

Где уже искали — статистика вычислена системой; ответа эти поиски НЕ дали:
{searched}

Как читать статистику (она для маршрутизации, не для вердикта):
- «последний поиск дал +0 новых чанков» = коллекция при нынешних формулировках
  ИСЧЕРПАНА. Возвращаться в неё можно ТОЛЬКО с радикально другим углом
  (конкретные названия, имена, термины из «альтернативных формулировок»
  обратной связи) — НЕ с пересказом прежнего запроса другими словами.
- Низкое покрытие «извлечено K/N» само по себе НЕ означает «там ещё много
  неисследованного»: векторный поиск уже извлёк самое похожее на запрос, и
  похожий запрос вернёт то же самое. Новые чанки приносит только новый угол.
- Высокое покрытие маленькой коллекции = она прочитана почти целиком.

Для каждой коллекции, которая правдоподобно может содержать недостающий фрагмент, создай RouteStep:
- collection: точное имя таблицы (должно совпадать с одной из доступных)
- subquery: сфокусированный запрос по недостающему фрагменту — используй
  АЛЬТЕРНАТИВНЫЕ ключевые слова или другой угол, отличный от уже испробованного
- rationale: почему эта коллекция может содержать недостающее

Уверенно ПРЕДПОЧИТАЙ коллекции, в которых ещё НЕ искали. Выбирай 1-3 самые
релевантные. Если же каждая правдоподобно-релевантная коллекция уже исчерпана
и нового угла не видно — верни ПУСТОЙ список steps: система честно сообщит,
что не смогла найти недостающее. НЕ трать итерации на повтор исчерпанного."""


async def planner_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Planner: create search routes, then command query_rewriter."""
    db_path = state.get("db_path")
    if vdb_settings.descriptions_enabled:
        described = await list_collections_described(db_path)
        available = [c["collection"] for c in described]
        lines = [
            f"- {c['collection']} — {c['description'] or '(без описания)'}"
            for c in described
        ]
    else:
        names = await list_collections.ainvoke({"db_path": db_path})
        available = list(names)
        lines = [f"- {n}" for n in names]
    collections_str = (
        "\n".join(lines)
        if lines
        else "(коллекций пока нет — сначала проиндексируйте документы)"
    )

    # Iteration mode: the Sufficient Context Agent sent us back to RE-ROUTE for
    # the missing pieces (Google's loop re-enters before Search Plan). Plan a
    # narrow route to the collection(s) most likely to hold what's missing,
    # instead of blindly searching every collection.
    iteration = state.get("iteration_count", 0)
    feedback = state.get("feedback", "")
    is_iteration = iteration > 0 and bool(feedback)

    if is_iteration:
        # Mechanical coverage statistics (computed by code, the model only
        # reads them): which collections were actually searched, how many times,
        # and what fraction of their chunks is already retrieved — the routing
        # signal that lets the planner skip exhausted collections.
        stats = collection_search_stats(state.get("search_results", []))
        totals = dict(
            zip(
                stats.keys(),
                await asyncio.gather(*(count_chunks(c, db_path) for c in stats)),
            )
        )
        searched_str = format_search_stats_for_planner(stats, totals)
        prompt = PLANNER_ITERATION_PROMPT.format(
            query=state["query"],
            missing_parts=", ".join(state.get("missing_parts", [])) or "(не уточнено)",
            feedback=feedback,
            collections=collections_str,
            searched=searched_str,
        )
    else:
        prompt = PLANNER_PROMPT.format(
            query=state["query"],
            collections=collections_str,
        )
    # RouteStep requires a non-empty collection/subquery, so a step DeepSeek
    # under-fills (only rationale) fails validation; generate_structured re-prompts
    # with the error and, if the model keeps failing, routes to give_up.
    plan: PlanResult = await generate_structured(PlanResult, prompt)

    steps_dicts = [s.model_dump() for s in plan.steps]

    mode = "iteration" if is_iteration else "initial"
    rationale_info = "\n".join(
        f"• {s.get('collection', '?')}: {s.get('rationale', '')}" for s in steps_dicts
    )
    trace_entry = make_trace_entry(
        agent="planner",
        decision=f"{mode}: {len(plan.steps)} route(s)",
        detail=str(steps_dicts),
        info=rationale_info,
    )

    if not plan.steps:
        if not is_iteration and available:
            # Epistemic backstop: a claim of absence needs retrieval evidence.
            # Descriptions can justify routing priority, never absence — the
            # prompt already says "probe instead of refusing", so reaching this
            # branch means the model declined anyway. Probe a few collections
            # with the raw query and let the judge rule on the actual results.
            probe_steps = [
                {
                    "collection": collection,
                    "subquery": state["query"],
                    "rationale": (
                        "Зонд: ни одна коллекция не выглядела релевантной по "
                        "описанию, но описание не доказывает отсутствие — "
                        "проверяем реальным поиском."
                    ),
                }
                for collection in available[:_PROBE_LIMIT]
            ]
            trace_entry = make_trace_entry(
                agent="planner",
                decision=f"probe: {len(probe_steps)} route(s)",
                detail=str(probe_steps),
                info=(
                    "зонд-маршруты: отсутствие подтверждается только реальным "
                    "поиском, вердикт вынесет судья"
                ),
            )
            return Command(
                goto="query_rewriter",
                update={
                    "plan_steps": probe_steps,
                    "current_route": probe_steps[0],
                    "trace": [trace_entry],
                },
            )

        # Refusal is justified: either the knowledge base has no collections at
        # all, or an iteration concluded every plausible collection is already
        # searched to exhaustion. Pure RAG — no general-knowledge fallback.
        reason = (
            "Every plausibly-relevant collection has already been searched to "
            "exhaustion — no new route could close the remaining gap."
            if is_iteration
            else "The knowledge base contains no indexed collections."
        )
        return Command(
            goto="give_up",
            update={
                "plan_steps": [],
                "sufficient_reason": reason,
                "trace": [trace_entry],
            },
        )

    # All routes go to query_rewriter together (first route set as current_route
    # for search_fanout to know which collection to target initially)
    return Command(
        goto="query_rewriter",
        update={
            "plan_steps": steps_dicts,
            "current_route": steps_dicts[0],
            "trace": [trace_entry],
        },
    )
