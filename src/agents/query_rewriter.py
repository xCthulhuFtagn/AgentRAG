"""Query Rewriter Agent — rewrites queries for precise vector search.

Returns Command(goto="search_fanout") — edgeless routing.
Processes all plan_steps at once, or handles iteration feedback.

When a document's language differs from the query language, the rewriter
translates the search query to the document's language — otherwise a Russian
query against a German book hits neither embedding similarity nor BM25.
"""

import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, make_trace_entry
from src.agents.common import get_llm
from src.llm_retry import ainvoke_with_retry
from src.vectordb.descriptions import language_for_collection
from src.vectordb.tools import list_collections_described

REWRITER_PROMPT = """Ты — Агент-Переписчик Запросов (Query Rewriter) в системе Agentic RAG.

Твоя задача: превратить поисковый маршрут в точный запрос, оптимизированный для векторного поиска.

Исходный вопрос пользователя: {original_query}
Текущая цель поиска: коллекция '{collection}'
Что ищем: {subquery}
Язык документа: {doc_language}

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
{translation_rule}
Верни ТОЛЬКО текст переписанного поискового запроса, ничего больше."""


def _doc_language_label(iso: str) -> str:
    """Human-readable language label for the prompt."""
    names = {
        "ru": "русский", "en": "английский", "de": "немецкий",
        "fr": "французский", "es": "испанский", "it": "итальянский",
        "pt": "португальский", "nl": "нидерландский", "pl": "польский",
        "uk": "украинский", "be": "белорусский", "bg": "болгарский",
        "cs": "чешский", "sk": "словацкий", "sl": "словенский",
        "hr": "хорватский", "sr": "сербский", "mk": "македонский",
        "da": "датский", "sv": "шведский", "no": "норвежский",
        "fi": "финский", "et": "эстонский", "lv": "латышский",
        "lt": "литовский", "el": "греческий", "tr": "турецкий",
        "ar": "арабский", "he": "иврит", "fa": "персидский",
        "hi": "хинди", "bn": "бенгальский", "th": "тайский",
        "vi": "вьетнамский", "zh": "китайский", "ja": "японский",
        "ko": "корейский", "ro": "румынский", "hu": "венгерский",
        "ca": "каталанский", "eu": "баскский", "ka": "грузинский",
        "hy": "армянский", "az": "азербайджанский", "kk": "казахский",
        "uz": "узбекский", "id": "индонезийский", "ms": "малайский",
        "sw": "суахили", "af": "африкаанс", "ur": "урду",
        "ta": "тамильский", "te": "телугу", "mn": "монгольский",
    }
    return names.get(iso, iso)


def _language_for_route(
    collection: str, described: list[dict]
) -> str:
    """ISO 639-1 language for a route's target collection, or "ru"."""
    for c in described:
        if c["collection"] == collection:
            return c.get("language", "ru") or "ru"
    return "ru"


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
    llm, original_query: str, step: dict, tried: list[str], doc_language: str
) -> tuple[str, str]:
    """Rewrite one plan route into a search-optimized query.

    When *doc_language* is not Russian, adds a translation rule so the search
    query matches the document's actual language — a Russian question against a
    German book needs a German search query for both embedding similarity and
    BM25 to fire.

    Returns (collection, query). Module-level so it isn't redefined per node
    call; routes are rewritten concurrently via asyncio.gather.
    """
    collection = step.get("collection", "unknown")
    if doc_language != "ru":
        translation_rule = (
            f"6. ВАЖНО: язык документа — {_doc_language_label(doc_language)}. "
            f"ПЕРЕВЕДИ поисковый запрос на {_doc_language_label(doc_language)} "
            f"(в единственном числе, именительном падеже — так, как термины "
            f"записаны в документе). Исходный вопрос пользователя на русском — "
            f"переведи ключевые поисковые термины."
        )
    else:
        translation_rule = ""
    prompt = REWRITER_PROMPT.format(
        original_query=original_query,
        collection=collection,
        subquery=step.get("subquery", original_query),
        doc_language=_doc_language_label(doc_language),
        tried="\n".join(f"- «{q}»" for q in tried) or "(пока никаких)",
        translation_rule=translation_rule,
    )
    result: str = (await ainvoke_with_retry(llm, prompt)).content.strip().strip('"')
    return collection, result


async def query_rewriter_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Query Rewriter: produce search-optimized queries, command search_fanout.

    The Planner guarantees ≥1 route whenever it commands this node (it gives up
    directly on an empty KB / iteration exhaustion, and probes instead of
    refusing otherwise), so we always rewrite its plan — one search task per
    route, no fallback modes.
    Routes are independent → rewrite them concurrently (asyncio.gather), which
    preserves order so rewritten[i]/search_tasks[i] align with plan_steps[i].

    Each route now carries the target collection's language (ISO 639-1) from the
    descriptions sidecar; when the document language differs from Russian, the
    rewriter translates the search query so both vector similarity and BM25
    (hybrid search) can match across languages.
    """
    llm = get_llm(temperature=0.3)
    plan_steps = state.get("plan_steps", [])
    search_results = state.get("search_results", [])
    db_path = state.get("db_path")

    # Load described collections once to resolve each route's document language.
    described = await list_collections_described(db_path)

    pairs = await asyncio.gather(
        *[
            _rewrite_route(
                llm,
                state["query"],
                s,
                _queries_already_tried(search_results, s.get("collection", "unknown")),
                _language_for_route(s.get("collection", "unknown"), described),
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
