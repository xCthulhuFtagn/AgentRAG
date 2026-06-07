"""Common utilities shared across all agents."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0, model: str | None = None) -> ChatOpenAI:
    """Get a configured DeepSeek LLM instance (cached).

    Uses OpenAI-compatible endpoint at api.deepseek.com/v1.
    """
    return ChatOpenAI(
        model=model or DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


def get_structured_llm(schema, temperature: float = 0.0):
    """LLM that returns a validated Pydantic object.

    Uses method="function_calling" — DeepSeek's API does not support the
    json_schema `response_format` that with_structured_output() picks by
    default (returns 'This response_format type is unavailable now').
    Function calling is the OpenAI-compatible path DeepSeek does support.
    """
    return get_llm(temperature).with_structured_output(
        schema, method="function_calling"
    )
