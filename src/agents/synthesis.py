"""Synthesis Agent — generates the final answer from complete context.

Returns Command(goto=END) — edgeless termination.
"""

from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.state import AgentRAGState, make_trace_entry
from src.agents.common import get_llm, get_inventory_str
from src.llm_retry import ainvoke_with_retry

SYNTHESIS_PROMPT = """Ты — Агент-Синтезатор (Synthesis) в системе Agentic RAG.

Твоя задача: дать исчерпывающий, точный и хорошо структурированный финальный
ответ на основе ВСЕГО найденного контекста.

Вопрос пользователя: {query}

Полная опись базы знаний (все существующие документы, с кратким описанием каждого):
{inventory}

Контекст, найденный за несколько поисков:
{search_results}

Оценка Агента Достаточности Контекста: {sufficient_reason}

Правила:
1. Ответь на ВСЕ части вопроса полностью
2. Опирайся ТОЛЬКО на найденный контекст и опись выше — не выдумывай факты и не угадывай/не расшифровывай аббревиатуры из общих знаний
3. НИКОГДА не отказывайся отвечать. Ты вызван потому, что контекст признан достаточным, — дай прямой ответ по максимуму того, что В контексте ЕСТЬ. Извлеки то, что фрагменты реально утверждают (например, аббревиатуру, расшифрованную внутри предложения), и начни с этого. Отказ — работа другого узла, не твоя
4. Оставшуюся неопределённость умести максимум в ОДНУ короткую завершающую строку — не превращай ответ в дисклеймер «чего не хватает / посмотрите в других документах» и не отправляй пользователя искать в другом месте
5. Указывай, из какой коллекции/документа взят каждый фрагмент информации;
   имя коллекции ВСЕГДА оборачивай в обратные кавычки (`07_Imya_kollekcii`) —
   без них подчёркивания в имени ломают разметку ответа
6. Пиши ясно, лаконично и структурированно
7. Для вопросов типа «опиши/перечисли ВСЕ файлы» опись — авторитетный список:
   опиши каждый документ из неё, дополняя найденными фрагментами там, где они есть

Заметка о полноте контекста: {context_note}

Теперь дай финальный ответ:"""


async def synthesis_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Synthesis: generate final answer, command END."""
    llm = get_llm(temperature=0.0)

    # search_results records every executed search, empty ones included (the
    # statistics need them) — only entries that brought chunks are sources.
    chunked = [r for r in state.get("search_results", []) if r.get("chunks")]
    results_str = ""
    for i, r in enumerate(chunked):
        chunks_str = "\n---\n".join(r.get("chunks", []))
        # Collection names are backticked here and in the inventory: the answer
        # is rendered as markdown, where bare underscores turn into italics —
        # the model mirrors the formatting it sees in the prompt.
        results_str += (
            f"\n### Источник {i+1}: `{r.get('collection', 'неизвестно')}`\n"
            f"Запрос: {r.get('subquery', 'неизвестно')}\n"
            f"Содержимое:\n{chunks_str}\n"
        )

    if not results_str:
        results_str = "(контекст не получен)"

    # Synthesis is only reached after the judge ruled the context sufficient.
    context_note = "Контекст признан достаточным — отвечай полностью на его основе."

    inventory = await get_inventory_str(state.get("db_path"), backtick_names=True)

    prompt = SYNTHESIS_PROMPT.format(
        query=state["query"],
        inventory=inventory,
        search_results=results_str,
        sufficient_reason=state.get("sufficient_reason", "Not assessed"),
        context_note=context_note,
    )

    answer: str = (await ainvoke_with_retry(llm, prompt)).content.strip()

    trace_entry = make_trace_entry(
        agent="synthesis",
        decision="final_answer",
        detail=f"answer_length={len(answer)} chars",
    )

    return Command(
        goto=END,
        update={
            "final_answer": answer,
            "trace": [trace_entry],
        },
    )
