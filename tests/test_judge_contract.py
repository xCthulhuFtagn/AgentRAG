"""Contract tests: the Sufficient Context judge schema and mechanical search stats.

The judge speaks the language of INFORMATION — sufficiency is judged against the
question as asked (question_verbatim, copied not generated), an insufficient
verdict must describe the gap via the «Не хватает / Найдено вместо этого»
template, and naming a collection (routing — the Planner's job) is a validation
error that re-prompts. The searched set / novelty / coverage are computed by
code (collection_search_stats), never reconstructed by the model.
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
    # missing_parts alone is no longer enough: the Planner routes on feedback,
    # and an empty feedback used to silently demote it to initial-plan mode.
    with pytest.raises(ValidationError, match="Не хватает"):
        SufficientContextResult(
            **judge_fields(sufficient=False, feedback="", missing_parts=["описание Пушкина"])
        )


def test_insufficient_feedback_must_follow_template():
    # Meta-commentary / free-form advice instead of the information-gap
    # template — the regular failure mode of a weak model.
    with pytest.raises(ValidationError, match="шаблону"):
        SufficientContextResult(
            **judge_fields(sufficient=False, feedback="Необходимы дополнительные поиски")
        )


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
    # Unknown total (unreadable table) → coverage omitted, count still shown.
    text_no_total = format_search_stats_for_planner(stats, {"lit": None})
    assert "обыскана 1 раз" in text_no_total
    assert "извлечено" not in text_no_total
    assert format_search_stats_for_planner({}, {}) == "(пока нигде)"


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
