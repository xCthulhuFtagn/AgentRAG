"""Give Up Agent — system-generated refusal when context is exhausted.

No LLM involved — the system honestly reports what was found, what's missing,
and why the question cannot be fully answered.

The refusal is rendered as markdown in the web UI, where bare table names
(07_Rodnaya_…) lose their underscores to italics — so collection names are
wrapped in backticks, both the code-built lists and any bare occurrences
inside the judge-written reason text.
"""

import re

from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.config import general_settings
from src.state import AgentRAGState, make_trace_entry
from src.vectordb.tools import list_collections_described


def _backtick_names(text: str, names: list[str]) -> str:
    """Wrap bare occurrences of collection names in `…` (markdown-safe).

    Guards: skip occurrences already inside backticks or embedded in a longer
    name-like token ([A-Za-z0-9._-]), so `lit` is not wrapped inside lit_extra.
    """
    for name in sorted(names, key=len, reverse=True):
        if not name:
            continue
        pattern = rf"(?<![\w.`-]){re.escape(name)}(?![\w.`-])"
        text = re.sub(pattern, f"`{name}`", text)
    return text


def _build_llm_error_answer(state: AgentRAGState) -> str:
    """Refusal for an unrecoverable model failure (set by llm_failsafe).

    Distinct from the "insufficient context" refusal: here the retrieval pipeline
    didn't conclude — the language model itself failed to return a valid response,
    so the system reports that honestly instead of crashing.
    """
    return (
        f"## Не удалось ответить — ошибка языковой модели\n\n"
        f"**Вопрос:** {state['query']}\n\n"
        f"**Что произошло:** Языковая модель системы не смогла вернуть "
        f"корректный ответ после нескольких попыток, поэтому запрос не "
        f"удалось обработать.\n\n"
        f"**Подробности:** {state.get('llm_error', 'ошибка модели не уточнена')}\n\n"
        f"**Рекомендация:** Обычно это временная проблема (перегрузка модели "
        f"или сбой API). Попробуйте повторить запрос через некоторое время."
    )


def _build_refusal_answer(state: AgentRAGState, collection_names: list[str]) -> str:
    """Build a system-generated refusal message.

    No LLM — the system itself states what it found and what's missing.
    collection_names (full inventory) is used to backtick-escape bare table
    names inside the judge-written reason text for markdown rendering.
    """
    query = state["query"]
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", general_settings.max_iterations)

    # Summarize what was searched vs what was found. search_results records
    # every executed search (empty ones included), so the refusal can honestly
    # distinguish "searched N collections" from "retrieved M chunks".
    search_results = state.get("search_results", [])
    searched_collections: set[str] = set()
    found_chunks = 0
    for r in search_results:
        collection = r.get("collection")
        if collection and not r.get("error"):
            searched_collections.add(collection)
        found_chunks += len(r.get("chunks", []))

    if searched_collections:
        found_summary = (
            f"- Обыскано коллекций: {len(searched_collections)} — "
            f"{', '.join(f'`{c}`' for c in sorted(searched_collections))}\n"
        )
        found_summary += (
            f"- Всего извлечено фрагментов текста: {found_chunks}\n"
            if found_chunks
            else "- Ни одного релевантного фрагмента не найдено\n"
        )
    else:
        found_summary = "- Поиск не выполнялся (в базе знаний нет коллекций для поиска)\n"

    # What's missing
    missing = state.get("missing_parts", []) or [
        "конкретная информация, необходимая для ответа на вопрос"
    ]
    missing_str = "\n".join(f"  • {m}" for m in missing)

    # Search attempts
    queries_tried = state.get("rewritten_queries", [])
    queries_str = (
        "\n".join(f"  • {q}" for q in queries_tried[-10:]) or "  (нет)"
    )

    # The judge's reason legitimately mentions collections (its verdict
    # reasoning); wrap any bare names so markdown doesn't eat the underscores.
    reason = _backtick_names(
        state.get("sufficient_reason", "Найденного контекста недостаточно"),
        list(set(collection_names) | searched_collections),
    )

    return (
        f"## Не удалось полностью ответить\n\n"
        f"**Вопрос:** {query}\n\n"
        f"**Что найдено:**\n{found_summary}\n"
        f"**Чего не хватает:**\n{missing_str}\n\n"
        f"**Почему:** {reason}\n\n"
        f"**Попытки поиска ({iteration}/{max_iter} итераций):**\n{queries_str}\n\n"
        f"**Рекомендация:** Возможно, запрошенная информация отсутствует в "
        f"проиндексированных документах. Попробуйте переформулировать вопрос, "
        f"проиндексировать дополнительные документы или разбить вопрос на более "
        f"простые части."
    )


async def give_up_node(
    state: AgentRAGState, *, config: RunnableConfig
) -> Command:
    """Give Up: build refusal answer, command END. No LLM call."""

    llm_error = state.get("llm_error")
    if llm_error:
        refusal = _build_llm_error_answer(state)
        trace_entry = make_trace_entry(
            agent="give_up",
            decision="refusal (llm_error)",
            detail=llm_error,
        )
    else:
        # Inventory names are only needed to markdown-escape the reason text;
        # an unreadable/empty DB simply means nothing extra gets wrapped.
        described = await list_collections_described(state.get("db_path"))
        refusal = _build_refusal_answer(
            state, [c["collection"] for c in described]
        )
        trace_entry = make_trace_entry(
            agent="give_up",
            decision="refusal",
            detail=(
                f"max_iterations={state.get('max_iterations', general_settings.max_iterations)} "
                f"reached, context insufficient"
            ),
        )

    return Command(
        goto=END,
        update={
            "final_answer": refusal,
            "trace": [trace_entry],
        },
    )
