"""Document description — a one-line content summary per file.

Part of the vectordb module: at index time an LLM reads an excerpt of the
document and produces a short description, stored alongside the table (see
descriptions.py) and surfaced to the Planner so it can pick a source from a
summary, not just the table name.

Self-contained: builds its own DeepSeek client from general_settings (the
shared API config), so vectordb stays independent of the agent graph.
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import general_settings

DESCRIBE_PROMPT = """You summarize a document so a retrieval planner can decide whether to search it for a given query.

In 1-2 sentences, state what this document is: its type (e.g. research paper, manual, essay, report, dataset) and its main topic/subject. Be specific and concrete (name the domain, key entities). Do not begin with "This document". Write in the document's own language.

Document text (excerpt):
{excerpt}

Description:"""


@lru_cache(maxsize=1)
def _describe_llm() -> ChatOpenAI:
    """Plain DeepSeek client for text descriptions (cached)."""
    return ChatOpenAI(
        model=general_settings.deepseek_model,
        api_key=general_settings.deepseek_api_key,
        base_url=general_settings.deepseek_base_url,
        temperature=0.0,
    )


async def describe_document(text: str, max_chars: int = 6000) -> str:
    """Return a short content description for routing. Empty string on failure.

    Only the first `max_chars` are sent — title/abstract/intro carry most of
    the routing signal and this bounds cost. A failure must never break
    indexing, so any error degrades to an empty description.
    """
    excerpt = text[:max_chars].strip()
    if not excerpt:
        return ""
    try:
        resp = await _describe_llm().ainvoke(DESCRIBE_PROMPT.format(excerpt=excerpt))
        return resp.content.strip()
    except Exception:
        return ""
