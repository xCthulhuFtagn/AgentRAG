"""Document description — a one-line content summary per file.

Part of the vectordb module: at index time an LLM reads an excerpt of the
document and produces a short description, stored alongside the table (see
descriptions.py) and surfaced to the Planner so it can pick a source from a
summary, not just the table name.

Self-contained: builds its own LLM client from general_settings (the shared
API config, provider picked by `general_settings.llm_provider`), so vectordb
stays independent of the agent graph.
"""

from functools import lru_cache

from langchain_gigachat.chat_models import GigaChat
from langchain_openai import ChatOpenAI

from src.config import general_settings
from src.llm_retry import ainvoke_with_retry
from src.vectordb.config import vdb_settings

DESCRIBE_PROMPT = """Ты составляешь краткое описание документа, чтобы планировщик поиска мог решить, искать ли в нём ответ на запрос.

В 1-2 предложениях укажи, что это за документ: его тип (например, научная статья, руководство, эссе, отчёт, набор данных) и основную тему/предмет. Будь конкретен (назови область, ключевые сущности). Не начинай со слов «Этот документ». Пиши на языке самого документа.

Текст документа (фрагмент):
{excerpt}

Описание:"""


@lru_cache(maxsize=1)
def _describe_llm() -> ChatOpenAI | GigaChat:
    """Plain LLM client for text descriptions (cached), for the active provider.

    GigaChat rejects temperature=0 (see src/agents/common.py:get_llm) — top_p=0
    is its documented deterministic-output equivalent.
    """
    if general_settings.llm_provider == "gigachat":
        return GigaChat(
            model=general_settings.gigachat_model,
            credentials=general_settings.gigachat_credentials,
            scope=general_settings.gigachat_scope,
            base_url=general_settings.gigachat_base_url,
            verify_ssl_certs=general_settings.gigachat_verify_ssl_certs,
            top_p=0.0,
        )
    return ChatOpenAI(
        model=general_settings.deepseek_model,
        api_key=general_settings.deepseek_api_key,
        base_url=general_settings.deepseek_base_url,
        temperature=0.0,
    )


async def describe_document(text: str, max_chars: int | None = None) -> str:
    """Return a short content description for routing. Empty string on failure.

    Only the first `max_chars` are sent — title/abstract/intro carry most of
    the routing signal and this bounds cost. Defaults to
    `vdb_settings.describe_max_chars` (env DESCRIBE_MAX_CHARS). A failure must
    never break indexing, so any error degrades to an empty description.
    """
    if max_chars is None:
        max_chars = vdb_settings.describe_max_chars
    excerpt = text[:max_chars].strip()
    if not excerpt:
        return ""
    try:
        resp = await ainvoke_with_retry(
            _describe_llm(), DESCRIBE_PROMPT.format(excerpt=excerpt)
        )
        return resp.content.strip()
    except Exception:
        return ""
