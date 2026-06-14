"""Sufficient Context Agent — the key innovation from Google Research.

The judge classifies the RETRIEVAL STATE (SGR Routing — a closed set of named
situations, not a quality grade against an ideal answer): which `verdict` are
we in?
- «ответ_найден» — the answer is in the chunks → synthesis.
- «исчерпано_есть_упоминания» — the corpus is exhausted on the topic (every
  plausible collection searched to diminishing returns) and the draft collects
  the thin findings → synthesis too: "the sources contain only …" is the
  system's honest answer.
- «есть_необысканная_коллекция» / «есть_неиспробованный_угол» — a route remains
  (an unsearched plausible collection, or an untried angle in a collection still
  yielding new on-topic chunks) → re-route, with the gap described in feedback.
- «ничего_не_найдено» — zero mentions AND nowhere left → honest refusal.

A bool muddled two axes (route availability vs. what was found) and let a weak
model rationalize a thin-but-complete answer into "insufficient"; the named
situations split the axes and remove the quality-grade framing.

Separation of concerns: the judge speaks the language of information — it never
names collections or prescribes where to search (that is the Planner's job; a
bound-schema validator enforces the ban). The searched set, per-collection
novelty and the executed queries come as code-computed statistics, never
reconstructed by the model from chunk tags.

Routes (verdict → outcome):
- SYNTHESIS_VERDICTS → Command(goto="synthesis")
- CONTINUE_VERDICTS + iterations left → Command(goto="planner") with feedback
- «ничего_не_найдено», or CONTINUE at max iterations → Command(goto="give_up")
"""

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.config import general_settings
from src.state import (
    CONTINUE_VERDICTS,
    SYNTHESIS_VERDICTS,
    AgentRAGState,
    make_sufficient_context_schema,
    make_trace_entry,
)
from src.agents.common import (
    collection_search_stats,
    format_inventory,
    format_search_stats_for_judge,
    generate_structured,
)
from src.vectordb.tools import list_collections_described

SUFFICIENT_CONTEXT_PROMPT = """Ты — Агент Достаточности Контекста в системе Agentic RAG, которая отвечает СТРОГО по локальной базе документов.

ТВОЯ ЗАДАЧА — определить, в каком СОСТОЯНИИ ПОИСКА мы находимся, и выбрать ровно одну ситуацию (поле verdict). Это НЕ оценка качества ответа по абсолютной шкале: максимум возможного ответа системы — всё, что база содержит по теме вопроса. Тощий, но честный ответ из исчерпанной базы — это полноценный ответ, а НЕ повод искать дальше.

Вопрос пользователя: {query}

Опись базы знаний (ПОЛНЫЙ список существующих коллекций — других документов не существует):
{inventory}

Статистика поисков (вычислена системой по фактически выполненным запросам — доверяй ей, а не своей памяти):
{search_stats}

Контекст, найденный поисками (каждый блок помечен коллекцией, из которой он пришёл):
{search_results}

Алгоритм решения:

1. Оценивай ТОЛЬКО то, что спрошено, как оно задано. Краткий или общий вопрос («пушкин», «расскажи про X») означает «что в базе есть про это» — НЕ превращай его в анкету (биография, даты, список произведений…), которой пользователь не заказывал.

2. Построй draft_answer — лучший ответ из ВСЕГО накопленного контекста (за все итерации). Включи всё, что фрагменты реально говорят по теме вопроса, даже если это немного: отдельные упоминания и косвенные факты — тоже материал ответа.

3. Заполни retrieval_state — проговори по статистике, остались ли вообще маршруты: (а) есть ли в описи правдоподобно-релевантная коллекция, которой НЕТ в статистике; (б) даёт ли хоть одна обысканная коллекция новое по теме, или последний поиск везде «+0 новых»/не по теме.

4. Выбери verdict — РОВНО ОДНУ ситуацию:

   Поиск можно ЗАВЕРШИТЬ (система отвечает тем, что нашла):
   • «ответ_найден» — draft_answer отвечает на вопрос, как он задан. Не продолжай ради дотошности или перепроверки.
   • «исчерпано_есть_упоминания» — каждая правдоподобно-релевантная коллекция обыскана, последние поиски не приносят нового ПО ТЕМЕ (новые чанки не о том — тоже исчерпанность; если «прирост по теме» упал до нуля или почти нуля — коллекция исчерпана, даже если +N ещё не ноль), а draft_answer собрал всё найденное, пусть и скудное. «В источниках об этом есть только …» — это и есть честный ответ.

   Поиск стоит ПРОДОЛЖИТЬ (есть конкретная причина ждать от базы большего):
   • «есть_необысканная_коллекция» — в описи есть коллекция, которой НЕТ в статистике, и по описанию она правдоподобно может содержать ответ.
   • «есть_неиспробованный_угол» — релевантная коллекция ещё отдаёт НОВЫЕ по теме чанки (её последний поиск НЕ «+0» И «прирост по теме» не на спаде — новые чанки действительно о том, что спрошено) и есть неиспробованная формулировка, которой нет среди выполненных запросов. Назови её в feedback.

   Поиск ИСЧЕРПАН и ответа нет:
   • «ничего_не_найдено» — по теме нет НИ ОДНОГО упоминания, и искать больше негде (все правдоподобные коллекции обысканы). Пустоту нельзя выдать за ответ — система честно откажет.

Как читать статистику:
- В ней перечислены ВСЕ выполненные поиски и их запросы. Коллекции, которых нет в статистике, ещё НЕ обыскивались — не утверждай обратное.
- «+0 новых чанков» = эти формулировки в этой коллекции ИСЧЕРПАНЫ; похожий запрос вернёт то же самое. Для такой коллекции выбирать «есть_неиспробованный_угол» НЕЛЬЗЯ — это и есть сигнал исчерпанности, а не повод искать там же ещё раз.
- «прирост по теме: X₁/N₁ → X₂/N₂ → …» — динамика того, сколько НОВЫХ чанков каждого поиска упоминают тему вопроса. Спад (например 12/15 → 3/13 → 1/3) означает: коллекция отдаёт всё менее релевантный материал. Это ИСЧЕРПАННОСТЬ — не путай с продуктивностью. Выбирать «есть_неиспробованный_угол» при спаде прироста НЕЛЬЗЯ, даже если формально +N > 0.
- Выполненные запросы показывают уже испробованные углы — не предлагай их повторно в «альтернативных формулировках».

Опись и вопросы «опиши/перечисли ВСЕ файлы»: полнота достигнута, когда каждая коллекция либо обыскана, либо адекватно покрыта своим описанием. Опись исчерпывающа — не требуй доказательств существования других документов.

Разделение ответственности (СТРОГО): ты говоришь на языке ИНФОРМАЦИИ — что спрошено, что найдено, какого ФАКТА не хватает. ГДЕ искать — решает планировщик, поэтому в missing_parts и feedback ЗАПРЕЩЕНЫ имена коллекций и любые указания «поищи в …».
- missing_parts: короткие именные группы, называющие отсутствующие факты (например «расшифровка аббревиатуры ЭВМ»). НЕ маршруты («поиск в коллекции …») и НЕ размытое («более подробная информация»). Пустой список, если ответ найден.
- feedback (только при «есть_необысканная_коллекция»/«есть_неиспробованный_угол») — опиши информационный пробел; удобная форма: «Не хватает: …. Найдено вместо этого: …. Альтернативные формулировки: ….» (третья часть опциональна). Без мета-комментариев о процессе и пересказа этих инструкций. Для остальных вердиктов — пустая строка.

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
    stats = collection_search_stats(search_results, user_query=state["query"])

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

    # Schema bound to node-time context: feedback/missing_parts must not name a
    # collection (the judge states the information gap; routing belongs to the
    # Planner). Violations re-prompt through the uniform generate_structured path.
    schema = make_sufficient_context_schema([c["collection"] for c in described])
    result = await generate_structured(schema, prompt)

    judge_info = f"verdict: {result.verdict}\nreason: {result.reason}"
    if result.feedback and result.feedback.strip():
        judge_info += f"\nfeedback: {result.feedback}"

    trace_entry = make_trace_entry(
        agent="sufficient_context",
        decision=f"verdict={result.verdict}",
        detail=(
            f"reason={result.reason[:100]}, "
            f"feedback={result.feedback[:100]}, "
            f"missing={result.missing_parts}"
        ),
        info=judge_info,
    )

    # The verdict is a closed set of retrieval situations (SGR Routing); each
    # maps to exactly one outcome. SYNTHESIS_VERDICTS → answer (found, or the
    # honest "only …" over an exhausted corpus); CONTINUE_VERDICTS → re-route
    # while a route remains; "ничего_не_найдено" → refuse (nothing found AND
    # nowhere left, so don't burn the remaining iterations on exhaustion).
    verdict = result.verdict

    # ── Outcome 1: the corpus has its answer → synthesis ──
    if verdict in SYNTHESIS_VERDICTS:
        return Command(
            goto="synthesis",
            update={
                "sufficient": True,
                "sufficient_reason": result.reason,
                "draft_answer": result.draft_answer,
                "trace": [trace_entry],
            },
        )

    # ── Outcome 2: a route remains and iterations are left → re-route ──
    # Go back to the Planner (not query_rewriter): it re-routes to the
    # collection most likely to hold the missing piece, instead of blindly
    # searching every collection. Mirrors Google's loop that re-enters before
    # Search Plan.
    if verdict in CONTINUE_VERDICTS and iteration < max_iter:
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

    # ── Outcome 3: nothing found, or routes remained but iterations are spent
    #    → honest refusal ──
    return Command(
        goto="give_up",
        update={
            "sufficient": False,
            "sufficient_reason": result.reason,
            "missing_parts": result.missing_parts,
            "trace": [trace_entry],
        },
    )
