"""Sufficient Context Agent — the key innovation from Google Research.

Checks three things before allowing a response:
1. Retrieved snippets — do they contain needed information?
2. Intermediate draft — can we answer the question AS ASKED from what we have?
3. Information-gap analysis — what FACT exactly is missing, what was found
   instead, what alternative phrasings might name it in the documents?

Separation of concerns: the judge speaks the language of information — it never
names collections or prescribes where to search (that is the Planner's job; a
bound-schema validator enforces the ban). Sufficiency is measured against the
question as asked, not against an ideally exhaustive answer. The searched set
and per-collection novelty come as code-computed statistics, never
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

SUFFICIENT_CONTEXT_PROMPT = """Ты — Агент Достаточности Контекста (Sufficient Context) — контролёр качества в системе Agentic RAG.

Твоя задача: определить, достаточен ли найденный контекст, чтобы ответить на вопрос пользователя ТАК, КАК ОН ЗАДАН.

Вопрос пользователя: {query}

Полная опись базы знаний (ИСТИНА В ПОСЛЕДНЕЙ ИНСТАНЦИИ — это ВСЕ существующие коллекции, с кратким описанием каждой):
{inventory}

Статистика поисков (вычислена системой по фактически выполненным запросам — доверяй ей, а не своей памяти):
{search_stats}

Контекст, найденный поисками (каждый блок помечен коллекцией, из которой он пришёл):
{search_results}

Итерация: {iteration} из {max_iterations}
Ранее выявленные пробелы: {previous_gaps}

Проанализируй ТРИ вещи:

1. **Найденные фрагменты**: прочитай все найденные куски текста. Содержат ли они ФАКТЫ, нужные для ответа на каждую часть вопроса?

2. **Черновик ответа**: попробуй построить черновой ответ из ВСЕГО накопленного контекста (за все итерации). Если он уже отвечает на вопрос — контекст ДОСТАТОЧЕН, остановись. НЕ продолжай искать дополнительные или подтверждающие источники, когда ответ уже есть.

3. **Информационный пробел**: если ответа нет или он неполон, сформулируй пробел в терминах ИНФОРМАЦИИ:
   - Какой именно ФАКТ отсутствует (в терминах вопроса)?
   - Что поиски нашли ВМЕСТО него (какой угол не сработал)?
   - Какими АЛЬТЕРНАТИВНЫМИ формулировками этот факт может называться в документах?

Правила вердикта (мерило — ВОПРОС, КАК ОН ЗАДАН):
- Оценивай достаточность относительно ВОПРОСА, КАК ОН ЗАДАН, а не относительно идеально исчерпывающего ответа.
- НЕ добавляй подвопросы и критерии, которых пользователь не задавал. Для общего вопроса общий ответ из источников — достаточен.
- Если draft_answer отвечает на заданный вопрос — sufficient=True, даже если можно найти «больше деталей». НЕ ставь «недостаточно» ради дотошности, перепроверки уже имеющегося ответа или потому что другая коллекция «тоже может» что-то содержать.
- Ответ вида «не найдено / не определено / не упоминается» НЕ достаточен, пока по статистике поисков остаётся необысканная коллекция, которая по описанию правдоподобно может содержать ответ — ставь sufficient=False: сам факт её существования означает, что негативный вердикт ещё не финален. Какую коллекцию обыскивать следующей — решает планировщик, НЕ называй её.
- Если все правдоподобно-релевантные коллекции уже обысканы, а ответа нет — это честный финальный вердикт: опиши пробел, система честно откажет.
- Будь последователен: если твой feedback говорит, что информации не хватает, то sufficient ОБЯЗАН быть False.

Как читать статистику поисков:
- В ней перечислены ВСЕ выполненные поиски. Коллекция, которой нет в статистике, ещё НЕ обыскивалась — не утверждай обратное.
- «+0 новых чанков» в последнем поиске = поиски в этой коллекции исчерпаны (убывающая отдача): новых фрагментов оттуда уже не приходит, не жди их от повторного поиска.

Как пользоваться описью (это ПОЛНЫЙ, авторитетный список всех документов — других не существует):
- Вопросы типа «опиши/перечисли ВСЕ файлы» → полное покрытие означает, что каждая коллекция либо обыскана, либо адекватно описана своим описанием. Опись исчерпывающа, поэтому ты МОЖЕШЬ подтвердить полноту — не требуй доказательств существования других документов.
- КОНКРЕТНЫЕ вопросы (например, «что такое X») → если ответа НЕТ в найденных фрагментах, сравни опись со статистикой поисков: есть ли необысканная коллекция, которая по описанию может содержать ответ? Если есть — sufficient=False (вердикт ещё не финален). Принимай отрицательный ответ только после того, как каждая правдоподобно-релевантная коллекция реально обыскана.

Разделение ответственности (СТРОГО):
- Ты говоришь на языке ИНФОРМАЦИИ: что спрошено, что найдено, какого факта не хватает. Маршруты — в какой коллекции искать — зона ответственности планировщика, поэтому в missing_parts и feedback ЗАПРЕЩЕНЫ имена коллекций и любые указания «где искать».
- missing_parts: каждый элемент — короткая именная группа, называющая отсутствующий ФАКТ (например «расшифровка аббревиатуры ЭВМ», «описание личности поэта, а не только упоминание имени»). НЕ маршрут («поиск в коллекции …») и НЕ размытое («более подробная информация»).
- feedback: строго по шаблону «Не хватает: …. Найдено вместо этого: …. Альтернативные формулировки: ….» (третья часть опциональна). Никаких мета-комментариев о процессе («необходимы дополнительные поиски», «будет определено после вердикта») и пересказа этих инструкций.

Все текстовые поля заполняй ПО-РУССКИ. Поле question_verbatim — точная копия вопроса пользователя, без изменений."""


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

    prompt = SUFFICIENT_CONTEXT_PROMPT.format(
        query=state["query"],
        inventory=inventory,
        search_stats=format_search_stats_for_judge(stats),
        search_results=results_str,
        iteration=iteration,
        max_iterations=max_iter,
        previous_gaps=", ".join(state.get("missing_parts", [])) or "(нет)",
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
