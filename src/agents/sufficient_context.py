"""Sufficient Context Agent — the key innovation from Google Research.

Checks three things before allowing a response:
1. Retrieved snippets — do they contain needed information?
2. Intermediate draft — can we answer from what we have?
3. Missing pieces analysis — what EXACTLY is missing and where to look?

Routes:
- sufficient → Command(goto="synthesis")
- insufficient + iterations left → Command(goto="planner") with feedback (re-route)
- insufficient + max iterations → Command(goto="give_up")
"""

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.config import general_settings
from src.state import AgentRAGState, SufficientContextResult, make_trace_entry
from src.agents.common import generate_structured, get_inventory_str

SUFFICIENT_CONTEXT_PROMPT = """Ты — Агент Достаточности Контекста (Sufficient Context) — контролёр качества в системе Agentic RAG.

Твоя задача: определить, ПОЛОН ли найденный контекст настолько, чтобы ответить на вопрос пользователя.

Вопрос пользователя: {query}

Полная опись базы знаний (ИСТИНА В ПОСЛЕДНЕЙ ИНСТАНЦИИ — это ВСЕ существующие коллекции, с кратким описанием каждой):
{inventory}

Контекст, найденный поисками (каждый блок помечен коллекцией, из которой он пришёл):
{search_results}

Итерация: {iteration} из {max_iterations}
Ранее выявленные пробелы: {previous_gaps}

Проанализируй ТРИ вещи:

1. **Найденные фрагменты**: прочитай все найденные куски текста. Содержат ли они ФАКТЫ, нужные для ответа на каждую часть вопроса?

2. **Черновик ответа**: попробуй построить черновой ответ из ВСЕГО накопленного контекста (за все итерации). Если он уже отвечает на вопрос — контекст ДОСТАТОЧЕН, остановись. НЕ продолжай искать дополнительные или подтверждающие источники, когда ответ уже есть.

3. **Недостающие фрагменты (КРИТИЧНО)**: если чего-то не хватает, будь КОНКРЕТЕН:
   - Какая именно информация отсутствует?
   - В какой коллекции её искать?
   - Какие альтернативные поисковые формулировки попробовать?

Правила (СНАЧАЛА оцени достаточность ВСЕГО накопленного контекста, и только потом думай о новых поисках):
- Если накопленный контекст отвечает на вопрос → sufficient=True. Это верно, даже если какие-то коллекции не обысканы и могли бы содержать смежный материал. НЕ ставь «недостаточно» ради дотошности, ради перепроверки уже имеющегося ответа или потому что другая коллекция «тоже может» его содержать (или содержать «больше»).
- Только когда ответ действительно ОТСУТСТВУЕТ или НЕПОЛОН → sufficient=False с конкретной обратной связью (чего не хватает, в какой коллекции искать дальше)
- Ответ вида «не найдено / не определено / не упоминается» НЕ достаточен, пока остаётся необысканная коллекция, которая правдоподобно может содержать ответ — ставь sufficient=False и направляй туда. (Это касается только случая, когда ответ действительно отсутствует — не перепроверки уже имеющегося.)
- Лучше пометить «недостаточно» и поискать ещё, чем гадать — но только когда чего-то действительно не хватает, а не когда ответ просто «не подтверждён»
- Будь последователен: если твоя обратная связь говорит искать дальше, то sufficient ОБЯЗАН быть False

Как пользоваться описью (это ПОЛНЫЙ, авторитетный список всех документов — других не существует):
- Вопросы типа «опиши/перечисли ВСЕ файлы» → полное покрытие означает, что каждая коллекция либо обыскана, либо адекватно описана своим описанием. Опись исчерпывающа, поэтому ты МОЖЕШЬ подтвердить полноту — не требуй доказательств существования других документов.
- КОНКРЕТНЫЕ вопросы (например, «что такое X») → если ответа НЕТ в найденных фрагментах, сравни опись с коллекциями, которые реально встречаются в найденном контексте выше. Если какая-то коллекция, которой среди них ещё НЕТ, судя по описанию может содержать ответ — ставь sufficient=False и назови её в feedback/missing_parts. Принимай отрицательный ответ только после того, как каждая правдоподобно-релевантная коллекция реально обыскана."""


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

    # Format search results
    search_results = state.get("search_results", [])
    results_str = ""
    for i, r in enumerate(search_results[-10:]):
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

    inventory = await get_inventory_str(state.get("db_path"))

    prompt = SUFFICIENT_CONTEXT_PROMPT.format(
        query=state["query"],
        inventory=inventory,
        search_results=results_str,
        iteration=iteration,
        max_iterations=max_iter,
        previous_gaps=", ".join(state.get("missing_parts", [])) or "(нет)",
    )

    result: SufficientContextResult = await generate_structured(
        SufficientContextResult, prompt
    )

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
