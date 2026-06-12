"""Sufficient Context Agent — the key innovation from Google Research.

The judge answers ONE question — a retrieval-state call, not a grade against an
ideal answer: would one more search of THIS corpus materially improve the
answer to the question as asked?
- No, because the answer is found → sufficient.
- No, because the corpus is exhausted on the topic (every plausible collection
  searched to diminishing returns) → sufficient too, even if the findings are
  thin: "the sources contain only …" is the system's honest answer.
- Zero findings are never sufficient — that path stays an honest refusal.
- Yes (an unsearched plausible collection, or an untried angle in a collection
  still yielding new on-topic chunks) → insufficient, with the information gap
  described in feedback.

Separation of concerns: the judge speaks the language of information — it never
names collections or prescribes where to search (that is the Planner's job; a
bound-schema validator enforces the ban). The searched set, per-collection
novelty and the executed queries come as code-computed statistics, never
reconstructed by the model from chunk tags.

Routes:
- sufficient → Command(goto="synthesis")
- insufficient + iterations left → Command(goto="planner") with feedback (re-route)
- insufficient + max iterations → Command(goto="give_up")
"""

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.config import general_settings
from src.state import AgentRAGState, make_sufficient_context_schema, make_trace_entry
from src.agents.common import (
    collection_search_stats,
    format_inventory,
    format_search_stats_for_judge,
    generate_structured,
)
from src.vectordb.tools import list_collections_described

SUFFICIENT_CONTEXT_PROMPT = """Ты — Агент Достаточности Контекста в системе Agentic RAG, которая отвечает СТРОГО по локальной базе документов.

ТВОЯ ЗАДАЧА — ровно одно решение: ДАСТ ЛИ ЕЩЁ ОДИН ПОИСК ПО БАЗЕ материал, который заметно улучшит ответ на вопрос, как он задан?
- Даст → sufficient=False (поиск продолжится).
- Не даст → sufficient=True (система отвечает тем, что нашла).
Ты оцениваешь состояние ПОИСКА относительно ЭТОЙ базы, а не качество ответа по абсолютной шкале: максимум возможного ответа системы — всё, что база содержит по теме вопроса.

Вопрос пользователя: {query}

Опись базы знаний (ПОЛНЫЙ список существующих коллекций — других документов не существует):
{inventory}

Статистика поисков (вычислена системой по фактически выполненным запросам — доверяй ей, а не своей памяти):
{search_stats}

Контекст, найденный поисками (каждый блок помечен коллекцией, из которой он пришёл):
{search_results}

Алгоритм решения:

1. Перечитай вопрос и скопируй его в question_verbatim ДОСЛОВНО. Оценивай только то, что спрошено. Краткий или общий вопрос («пушкин», «расскажи про X») означает «что в базе есть про это» — НЕ превращай его в анкету (биография, даты, список произведений…), которой пользователь не заказывал.

2. Построй draft_answer — лучший ответ из ВСЕГО накопленного контекста (за все итерации). Включи всё, что фрагменты реально говорят по теме вопроса, даже если это немного: отдельные упоминания и косвенные факты — тоже материал ответа.

3. Вынеси вердикт sufficient:

   True, если верно ЛЮБОЕ из двух:
   • draft_answer отвечает на вопрос, как он задан, — даже если можно было бы накопать «больше деталей»: не продолжай поиск ради дотошности или перепроверки;
   • база ИСЧЕРПАНА по теме вопроса: каждая правдоподобно-релевантная по описи коллекция уже обыскана, и последние поиски не приносят нового ПО ТЕМЕ (новые чанки не о том — тоже исчерпанность), а draft_answer собирает всё найденное, пусть и скудное. «В источниках об этом есть только …» — полноценный честный ответ системы.

   False — ТОЛЬКО если есть конкретная причина ожидать от базы большего:
   • в описи есть НЕобысканная коллекция (её нет в статистике), по описанию способная содержать ответ; или
   • релевантная коллекция ещё отдаёт новое по теме, и есть неиспробованный угол поиска — формулировки, которых нет среди выполненных запросов в статистике (назови их в feedback).
   И всегда False, если по теме не найдено ВООБЩЕ ничего, ни одного упоминания: пустоту нельзя выдать за ответ — система честно откажет, когда маршрутов не останется.

Как читать статистику:
- В ней перечислены ВСЕ выполненные поиски и их запросы. Коллекции, которых нет в статистике, ещё НЕ обыскивались — не утверждай обратное.
- «+0 новых чанков» = эти формулировки в этой коллекции исчерпаны; похожий запрос вернёт то же самое.
- Выполненные запросы показывают уже испробованные углы — не предлагай их повторно в «альтернативных формулировках».

Опись и вопросы «опиши/перечисли ВСЕ файлы»: полнота достигнута, когда каждая коллекция либо обыскана, либо адекватно покрыта своим описанием. Опись исчерпывающа — не требуй доказательств существования других документов.

Разделение ответственности (СТРОГО): ты говоришь на языке ИНФОРМАЦИИ — что спрошено, что найдено, какого ФАКТА не хватает. ГДЕ искать — решает планировщик, поэтому в missing_parts и feedback ЗАПРЕЩЕНЫ имена коллекций и любые указания «поищи в …».
- missing_parts: короткие именные группы, называющие отсутствующие факты (например «расшифровка аббревиатуры ЭВМ»). НЕ маршруты («поиск в коллекции …») и НЕ размытое («более подробная информация»).
- feedback (только при sufficient=False) — строго по шаблону: «Не хватает: …. Найдено вместо этого: …. Альтернативные формулировки: ….» (третья часть опциональна). Без мета-комментариев о процессе и пересказа этих инструкций.

Все текстовые поля заполняй ПО-РУССКИ."""


async def sufficient_context_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Sufficient Context: check completeness, command next step.

    Three outcomes:
    1. sufficient=True  → Command(goto="synthesis")      — normal answer
    2. insufficient + iterations left → Command(goto="planner") — re-route & search more
    3. insufficient + max iterations  → Command(goto="give_up") — system refusal
    """
    max_iter = state.get("max_iterations", general_settings.max_iterations)
    iteration = state.get("iteration_count", 0)

    # Format search results. search_results records every executed search,
    # including empty ones (the statistics need them); only entries that
    # actually brought chunks are worth showing as context.
    search_results = state.get("search_results", [])
    chunked = [r for r in search_results if r.get("chunks")]
    results_str = ""
    for i, r in enumerate(chunked[-10:]):
        chunks = r.get("chunks", [])
        seqs = r.get("seqs", []) or []
        # Tag each chunk with its document position so the judge can see
        # contiguity and gaps (chunks arrive seq-ordered after stitching).
        lines = []
        for j, chunk in enumerate(chunks):
            seq = seqs[j] if j < len(seqs) and seqs[j] is not None else "?"
            lines.append(f"[seq={seq}] {chunk}")
        chunks_str = "\n---\n".join(lines)
        results_str += (
            f"\n[Результат {i+1}] Коллекция: {r.get('collection')}, "
            f"Запрос: {r.get('subquery')}\n{chunks_str}\n"
        )

    if not results_str:
        results_str = "(результатов поиска пока нет)"

    described = await list_collections_described(state.get("db_path"))
    inventory = format_inventory(described)

    # Mechanical statistics: the searched set and the last-search novelty delta
    # are computed by code — a weak model can't reliably reconstruct them from
    # collection tags in the chunks (it hallucinates "not searched yet").
    stats = collection_search_stats(search_results)

    # Deliberately NOT in the prompt: the iteration counter (the budget is
    # code's job — outcome 3 below — and showing it invites "last iteration,
    # so accept" gaming) and the previous missing_parts (echoing them back
    # re-anchored the judge on its own earlier question inflation: the invented
    # «биография, творчество, даты» survived every iteration of the Пушкин
    # trace). Each call re-derives the gap fresh from question + context.
    prompt = SUFFICIENT_CONTEXT_PROMPT.format(
        query=state["query"],
        inventory=inventory,
        search_stats=format_search_stats_for_judge(stats),
        search_results=results_str,
    )

    # Schema bound to node-time context: question_verbatim is checked against
    # the literal query, and feedback/missing_parts must not name a collection
    # (the judge states the information gap; routing belongs to the Planner).
    # Violations re-prompt through the same uniform generate_structured path.
    schema = make_sufficient_context_schema(
        [c["collection"] for c in described], state["query"]
    )
    result = await generate_structured(schema, prompt)

    judge_info = f"reason: {result.reason}"
    if result.feedback and result.feedback.strip():
        judge_info += f"\nfeedback: {result.feedback}"

    trace_entry = make_trace_entry(
        agent="sufficient_context",
        decision=f"sufficient={result.sufficient}",
        detail=(
            f"reason={result.reason[:100]}, "
            f"feedback={result.feedback[:100]}, "
            f"missing={result.missing_parts}"
        ),
        info=judge_info,
    )

    # ── Outcome 1: context is sufficient → normal answer ──
    if result.sufficient:
        return Command(
            goto="synthesis",
            update={
                "sufficient": True,
                "sufficient_reason": result.reason,
                "draft_answer": result.draft_answer,
                "trace": [trace_entry],
            },
        )

    # ── Outcome 2: insufficient, but iterations left → re-route & search more ──
    # Go back to the Planner (not query_rewriter): it re-routes to the
    # collection most likely to hold the missing piece, instead of blindly
    # searching every collection. Mirrors Google's loop that re-enters before
    # Search Plan.
    if iteration < max_iter:
        return Command(
            goto="planner",
            update={
                "sufficient": False,
                "sufficient_reason": result.reason,
                "feedback": result.feedback,
                "missing_parts": result.missing_parts,
                "draft_answer": result.draft_answer,
                "iteration_count": iteration + 1,
                "trace": [trace_entry],
            },
        )

    # ── Outcome 3: insufficient + no iterations left → give up ──
    return Command(
        goto="give_up",
        update={
            "sufficient": False,
            "sufficient_reason": result.reason,
            "missing_parts": result.missing_parts,
            "trace": [trace_entry],
        },
    )
