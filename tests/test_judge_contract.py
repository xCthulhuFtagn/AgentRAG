"""Contract tests: the Sufficient Context judge schema and mechanical search stats.

The judge speaks the language of INFORMATION — sufficiency is judged against the
question as asked (question_verbatim, copied not generated). Validation covers
only what CODE consumes: feedback presence at sufficient=False (the Planner's
iteration mode keys off it) and the collection-name ban (routing is the
Planner's job); feedback's FORM is prompt guidance, deliberately not enforced.
The searched set / novelty / coverage are computed by code
(collection_search_stats), never reconstructed by the model.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import (
    SufficientContextResult,
    make_initial_state,
    make_sufficient_context_schema,
)
from src.agents.common import (
    collection_search_stats,
    format_search_stats_for_judge,
    format_search_stats_for_planner,
)

QUERY = "Как описан Пушкин в источниках?"
COLLECTIONS = ["07_Rodnaya_literatura", "09-10_Obschaya_biologiya"]

TEMPLATE_FEEDBACK = (
    "Не хватает: характеристика Пушкина как личности и поэта. "
    "Найдено вместо этого: только упоминания имени в списках произведений. "
    "Альтернативные формулировки: «великий русский поэт», «биография»."
)


def judge_fields(**overrides) -> dict:
    """A valid sufficient=True result; override fields per test."""
    fields = dict(
        question_verbatim=QUERY,
        reason="Найдено общее описание Пушкина в учебнике литературы",
        draft_answer="Пушкин описан как выдающийся русский поэт",
        missing_parts=[],
        sufficient=True,
        feedback="",
    )
    fields.update(overrides)
    return fields


# ── Base schema: information-gap contract ────────────────────────────────────

def test_question_verbatim_is_required():
    fields = judge_fields()
    del fields["question_verbatim"]
    with pytest.raises(ValidationError):
        SufficientContextResult(**fields)


def test_sufficient_true_needs_no_feedback():
    result = SufficientContextResult(**judge_fields())
    assert result.sufficient is True


def test_insufficient_without_feedback_is_rejected():
    # missing_parts alone is not enough: the Planner routes on feedback, and
    # an empty feedback at False would silently demote it to initial-plan mode
    # (is_iteration keys off bool(feedback)) — presence is a CODE contract.
    with pytest.raises(ValidationError, match="ОБЯЗАТЕЛЬНО заполни feedback"):
        SufficientContextResult(
            **judge_fields(sufficient=False, feedback="", missing_parts=["описание Пушкина"])
        )


def test_insufficient_free_form_feedback_is_valid():
    # Form is guidance, not law: the only consumer of feedback is the
    # Planner-LLM reading prose, so semantically useful feedback worded
    # off-template must NOT be rejected — a re-prompt burned on formatting
    # can escalate a correct verdict into give_up.
    result = SufficientContextResult(
        **judge_fields(
            sufficient=False,
            feedback="Отсутствует характеристика Пушкина; поиски дали только списки произведений.",
        )
    )
    assert result.sufficient is False


def test_insufficient_with_template_feedback_passes():
    result = SufficientContextResult(
        **judge_fields(
            sufficient=False,
            feedback=TEMPLATE_FEEDBACK,
            missing_parts=["характеристика Пушкина как личности"],
        )
    )
    assert result.sufficient is False


def test_blank_missing_parts_item_is_rejected():
    with pytest.raises(ValidationError, match="непустой"):
        SufficientContextResult(
            **judge_fields(
                sufficient=False,
                feedback=TEMPLATE_FEEDBACK,
                missing_parts=["описание Пушкина", "  "],
            )
        )


# ── Bound schema: node-time context baked in as validators ──────────────────

def test_bound_schema_keeps_function_calling_name():
    schema = make_sufficient_context_schema(COLLECTIONS, QUERY)
    assert schema.__name__ == "SufficientContextResult"


def test_bound_rejects_paraphrased_question():
    schema = make_sufficient_context_schema(COLLECTIONS, QUERY)
    with pytest.raises(ValidationError, match="ДОСЛОВНОЙ"):
        schema(**judge_fields(question_verbatim="Что известно о Пушкине?"))


def test_bound_tolerates_trivial_copy_drift():
    # Case / trailing «?» / whitespace drift is not paraphrase — re-prompting a
    # weak model over it would burn retries for nothing.
    schema = make_sufficient_context_schema(COLLECTIONS, QUERY)
    result = schema(**judge_fields(question_verbatim="как описан пушкин  в источниках"))
    assert result.sufficient is True


def test_bound_rejects_collection_name_in_feedback():
    schema = make_sufficient_context_schema(COLLECTIONS, QUERY)
    with pytest.raises(ValidationError, match="имя\\s+коллекции|маршрут"):
        schema(
            **judge_fields(
                sufficient=False,
                feedback=TEMPLATE_FEEDBACK + " Поищи в 09-10_obschaya_biologiya.",
            )
        )


def test_bound_rejects_collection_name_in_missing_parts():
    schema = make_sufficient_context_schema(COLLECTIONS, QUERY)
    with pytest.raises(ValidationError, match="маршрут"):
        schema(
            **judge_fields(
                sufficient=False,
                feedback=TEMPLATE_FEEDBACK,
                missing_parts=["поиск в коллекции 07_Rodnaya_literatura"],
            )
        )


def test_bound_ignores_route_ban_when_sufficient():
    # feedback/missing_parts are unused on the True path — burning a re-prompt
    # over junk there would help nobody.
    schema = make_sufficient_context_schema(COLLECTIONS, QUERY)
    result = schema(**judge_fields(feedback="см. 07_Rodnaya_literatura"))
    assert result.sufficient is True


# ── Mechanical statistics ────────────────────────────────────────────────────

def test_stats_count_searches_and_novelty():
    results = [
        {"collection": "lit", "chunks": ["a", "b"], "seqs": [1, 2]},
        {"collection": "bio", "chunks": [], "seqs": []},  # empty search — recorded
        {"collection": "lit", "chunks": ["b", "c"], "seqs": [2, 3]},  # one new chunk
        {"collection": "lit", "chunks": ["a"], "seqs": [1]},  # nothing new
    ]
    stats = collection_search_stats(results)
    assert stats["lit"]["searches"] == 3
    assert stats["lit"]["retrieved"] == {1, 2, 3}
    assert stats["lit"]["last_new"] == 0  # the exhaustion detector
    assert stats["bio"]["searches"] == 1
    assert stats["bio"]["last_new"] == 0


def test_stats_skip_errored_searches():
    stats = collection_search_stats(
        [{"collection": "lit", "chunks": [], "seqs": [], "error": "not found"}]
    )
    assert stats == {}


def test_stats_legacy_table_without_seqs():
    stats = collection_search_stats([{"collection": "old", "chunks": ["x"], "seqs": []}])
    assert stats["old"]["seqs_known"] is False
    assert stats["old"]["last_new"] is None


def test_judge_stats_formatting():
    stats = collection_search_stats(
        [
            {"collection": "lit", "chunks": ["a"], "seqs": [1]},
            {"collection": "lit", "chunks": ["a"], "seqs": [1]},
            {"collection": "lit", "chunks": ["a"], "seqs": [1]},
        ]
    )
    line = format_search_stats_for_judge(stats)
    assert "обыскана 3 раза" in line
    assert "+0 новых чанков" in line
    assert format_search_stats_for_judge({}) == "(поисков ещё не было)"


def test_judge_stats_show_executed_queries():
    # The judge must see which angles were already tried — both to judge
    # "is there an untried angle?" and to avoid re-suggesting them.
    stats = collection_search_stats(
        [
            {"collection": "lit", "subquery": "Пушкин биография", "chunks": ["a"], "seqs": [1]},
            {"collection": "lit", "subquery": "Пушкин творчество", "chunks": ["a"], "seqs": [1]},
        ]
    )
    text = format_search_stats_for_judge(stats)
    assert "выполненные запросы" in text
    assert "«Пушкин биография»" in text
    assert "«Пушкин творчество»" in text


def test_judge_stats_singular_plural():
    stats = collection_search_stats([{"collection": "lit", "chunks": ["a"], "seqs": [7]}])
    line = format_search_stats_for_judge(stats)
    assert "обыскана 1 раз" in line
    assert "+1 новый чанк" in line


def test_planner_stats_coverage():
    stats = collection_search_stats(
        [{"collection": "lit", "chunks": ["a", "b", "c"], "seqs": [1, 2, 3]}]
    )
    text = format_search_stats_for_planner(stats, {"lit": 10})
    assert "извлечено 3/10 чанков (30%)" in text
    # The novelty delta is the planner's stop signal — shown alongside coverage
    # (coverage alone reads as "barely explored → dig the same spot again").
    assert "+3 новых чанка" in text
    # Unknown total (unreadable table) → coverage omitted, count still shown.
    text_no_total = format_search_stats_for_planner(stats, {"lit": None})
    assert "обыскана 1 раз" in text_no_total
    assert "извлечено" not in text_no_total
    assert format_search_stats_for_planner({}, {}) == "(пока нигде)"


def test_planner_stats_exhaustion_delta():
    stats = collection_search_stats(
        [
            {"collection": "lit", "chunks": ["a", "b"], "seqs": [1, 2]},
            {"collection": "lit", "chunks": ["a", "b"], "seqs": [1, 2]},
        ]
    )
    text = format_search_stats_for_planner(stats, {"lit": 100})
    assert "+0 новых чанков" in text


def test_queries_already_tried_per_collection():
    from src.agents.query_rewriter import _queries_already_tried

    results = [
        {"collection": "lit", "subquery": "Пушкин биография", "chunks": ["a"]},
        {"collection": "bio", "subquery": "клетка строение", "chunks": []},
        {"collection": "lit", "subquery": "Пушкин биография", "chunks": ["a"]},  # dup
        {"collection": "lit", "subquery": "Полтава Медный всадник", "chunks": []},
        {"collection": "lit", "subquery": "упала", "error": "boom"},  # not a search
    ]
    assert _queries_already_tried(results, "lit") == [
        "Пушкин биография",
        "Полтава Медный всадник",
    ]
    assert _queries_already_tried(results, "geo") == []


# ── Planner: claims of absence need retrieval evidence ──────────────────────

def _patch_planner(monkeypatch, collections: list[str]):
    """Stub the planner's LLM and DB lookups; the model declines every route."""
    from src.agents import planner as pl
    from src.state import PlanResult

    async def fake_generate(schema, prompt, **kwargs):
        return PlanResult(is_multi_step=False, steps=[])

    async def fake_described(db_path=None):
        return [{"collection": c, "description": "…"} for c in collections]

    async def fake_count(collection, db_path=None):
        return 100

    monkeypatch.setattr(pl, "generate_structured", fake_generate)
    monkeypatch.setattr(pl, "list_collections_described", fake_described)
    monkeypatch.setattr(pl, "count_chunks", fake_count)
    return pl


@pytest.mark.asyncio
async def test_planner_probes_instead_of_initial_refusal(monkeypatch):
    # The model declined every route on the initial turn — but a description
    # can't prove absence, so the planner must probe, not give up unsearched.
    pl = _patch_planner(monkeypatch, ["lit", "bio", "geo", "astro"])
    state = make_initial_state(query="свиные крылья")

    command = await pl.planner_node(state, config={})

    assert command.goto == "query_rewriter"
    probed = command.update["plan_steps"]
    assert [s["collection"] for s in probed] == ["lit", "bio", "geo"]  # capped at 3
    assert all(s["subquery"] == "свиные крылья" for s in probed)


@pytest.mark.asyncio
async def test_planner_iteration_exhaustion_gives_up(monkeypatch):
    # On iteration an empty plan is a legitimate, evidence-based exit.
    pl = _patch_planner(monkeypatch, ["lit"])
    state = make_initial_state(query="q")
    state["iteration_count"] = 1
    state["feedback"] = "Не хватает: факт. Найдено вместо этого: ничего."
    state["search_results"] = [
        {"collection": "lit", "subquery": "q", "chunks": [], "seqs": []}
    ]

    command = await pl.planner_node(state, config={})

    assert command.goto == "give_up"
    assert "exhaustion" in command.update["sufficient_reason"]


@pytest.mark.asyncio
async def test_planner_empty_corpus_gives_up(monkeypatch):
    # No collections at all — the only refusal allowed without a search.
    pl = _patch_planner(monkeypatch, [])
    state = make_initial_state(query="q")

    command = await pl.planner_node(state, config={})

    assert command.goto == "give_up"
    assert "no indexed collections" in command.update["sufficient_reason"]


# ── Give Up: collection names must survive markdown rendering ───────────────

def test_refusal_backticks_collection_names():
    # The web UI renders the refusal as markdown: bare 07_Lit_1991 loses its
    # underscores to italics, so every name — in the code-built searched list
    # and inside the judge-written reason — must be backtick-wrapped.
    from src.agents.give_up import _build_refusal_answer

    state = make_initial_state(query="пушкин")
    state["search_results"] = [
        {"collection": "07_Lit_1991", "subquery": "x", "chunks": ["c"], "seqs": [1]}
    ]
    state["sufficient_reason"] = (
        "Коллекция 07_Lit_1991 обыскана, 08_Geo_1933 нерелевантна."
    )

    text = _build_refusal_answer(state, ["07_Lit_1991", "08_Geo_1933"])

    assert "`07_Lit_1991`" in text
    assert "`08_Geo_1933`" in text
    # No bare occurrences left anywhere in the rendered refusal.
    assert text.count("07_Lit_1991") == text.count("`07_Lit_1991`")


def test_backtick_names_guards():
    from src.agents.give_up import _backtick_names

    # Already-wrapped and substring-of-longer-name occurrences stay untouched.
    text = _backtick_names(
        "уже `lit` обёрнут, lit отдельно, внутри lit_extra не трогать",
        ["lit", "lit_extra"],
    )
    assert "уже `lit` обёрнут" in text
    assert ", `lit` отдельно" in text
    assert "`lit_extra`" in text
    assert "`lit`_extra" not in text


# ── Fanout: every executed search is recorded, empty ones included ──────────

@pytest.mark.asyncio
async def test_search_fanout_records_empty_searches(monkeypatch):
    from src.agents import search_fanout as sf

    async def fake_search(args: dict) -> dict:
        return {
            "collection": args["collection"],
            "query": args["query"],
            "chunks": [],
            "scores": [],
            "seqs": [],
        }

    monkeypatch.setattr(sf, "vector_search", SimpleNamespace(ainvoke=fake_search))
    state = make_initial_state(query="кто такой Пушкин")
    state["search_tasks"] = [{"collection": "lit", "query": "Пушкин биография"}]

    command = await sf.search_fanout_node(state, config={})

    assert command.update["search_results"] == [
        {
            "collection": "lit",
            "subquery": "Пушкин биография",
            "chunks": [],
            "seqs": [],
            "scores": [],
        }
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
