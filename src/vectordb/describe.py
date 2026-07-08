"""Document description — a one-line content summary per file, plus language detection.

Part of the vectordb module: at index time an LLM reads an excerpt of the
document and produces a short description + ISO 639-1 language code, stored
alongside the table (see descriptions.py) and surfaced to the Planner so it
can pick a source from a summary, not just the table name.

Self-contained: builds its own LLM client from general_settings (the shared
API config, provider picked by `general_settings.llm_provider`), so vectordb
stays independent of the agent graph.
"""

from functools import lru_cache
from typing import Literal

from langchain_gigachat.chat_models import GigaChat
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import general_settings
from src.llm_retry import ainvoke_with_retry
from src.vectordb.config import vdb_settings


# Language names the model can pick → mapped to ISO 639-1 for FTS/sidecar storage.
_MODEL_LANG_TO_ISO: dict[str, str] = {
    "russian": "ru",
    "english": "en",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "polish": "pl",
    "ukrainian": "uk",
    "belarusian": "be",
    "bulgarian": "bg",
    "czech": "cs",
    "slovak": "sk",
    "slovenian": "sl",
    "croatian": "hr",
    "serbian": "sr",
    "macedonian": "mk",
    "danish": "da",
    "swedish": "sv",
    "norwegian": "no",
    "finnish": "fi",
    "estonian": "et",
    "latvian": "lv",
    "lithuanian": "lt",
    "greek": "el",
    "turkish": "tr",
    "arabic": "ar",
    "hebrew": "he",
    "persian": "fa",
    "hindi": "hi",
    "bengali": "bn",
    "thai": "th",
    "vietnamese": "vi",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "romanian": "ro",
    "hungarian": "hu",
    "catalan": "ca",
    "basque": "eu",
    "galician": "gl",
    "welsh": "cy",
    "irish": "ga",
    "scottish_gaelic": "gd",
    "maltese": "mt",
    "icelandic": "is",
    "albanian": "sq",
    "armenian": "hy",
    "georgian": "ka",
    "azerbaijani": "az",
    "kazakh": "kk",
    "kyrgyz": "ky",
    "tajik": "tg",
    "turkmen": "tk",
    "uzbek": "uz",
    "mongolian": "mn",
    "indonesian": "id",
    "malay": "ms",
    "tagalog": "tl",
    "swahili": "sw",
    "afrikaans": "af",
    "amharic": "am",
    "urdu": "ur",
    "punjabi": "pa",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
    "sinhala": "si",
    "khmer": "km",
    "lao": "lo",
    "burmese": "my",
    "nepali": "ne",
    "pashto": "ps",
}

# Closed set the model picks from — more reliable than free-form ISO code generation.
_LANG_LITERAL = Literal[tuple(_MODEL_LANG_TO_ISO.keys()) + ("other",)]  # type: ignore[misc]


class DocumentDescription(BaseModel):
    """Per-document description + detected language for FTS indexing and routing."""

    description: str = Field(
        description=(
            "В 1-2 предложениях: тип документа (научная статья, руководство, "
            "эссе, отчёт, набор данных) и основная тема/предмет. Будь конкретен "
            "(назови область, ключевые сущности). Не начинай со слов «Этот "
            "документ». Пиши на языке самого документа."
        )
    )
    language: _LANG_LITERAL = Field(  # type: ignore[valid-type]
        description=(
            "Язык, на котором написан документ. Определи по тексту фрагмента. "
            "Выбери 'other' только если язык действительно не входит в список."
        )
    )


DESCRIBE_PROMPT = """Ты составляешь краткое описание документа и определяешь его язык.

Текст документа (фрагмент):
{excerpt}"""


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


@lru_cache(maxsize=1)
def _describe_structured_llm():
    """Structured LLM for description + language (cached by schema type)."""
    return _describe_llm().with_structured_output(
        DocumentDescription, method="function_calling"
    )


async def describe_document(
    text: str, max_chars: int | None = None
) -> tuple[str, str]:
    """Return (description, language) for routing and FTS indexing.

    Only the first `max_chars` are sent — title/abstract/intro carry most of
    the routing signal and this bounds cost. Defaults to
    `vdb_settings.describe_max_chars` (env DESCRIBE_MAX_CHARS). A failure must
    never break indexing, so any error degrades to an empty description with
    language="ru" (the corpus default).
    """
    if max_chars is None:
        max_chars = vdb_settings.describe_max_chars
    excerpt = text[:max_chars].strip()
    if not excerpt:
        return "", "ru"
    try:
        result = await ainvoke_with_retry(
            _describe_structured_llm(), DESCRIBE_PROMPT.format(excerpt=excerpt)
        )
        if result is None:
            return "", "ru"
        desc = (result.description or "").strip()
        model_lang = (result.language or "russian").strip().lower()
        iso_lang = _MODEL_LANG_TO_ISO.get(model_lang, "ru")
        return desc, iso_lang
    except Exception:
        return "", "ru"
