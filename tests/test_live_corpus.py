"""Live end-to-end tests against the real indexed corpus + real GigaChat.

These run the UNCHANGED graph (build_graph → ainvoke) over the actual project
corpus on disk — four Soviet-era school textbooks indexed under a web project
("школа"). Each query is a SEPARATE parametrized case (own pytest id) so a
single behaviour can be run and its trace read in isolation, e.g.:

    pytest tests/test_live_corpus.py -s -k geo-karaganda

Every case prints a compact trace (per-node decision + verdict chain + token Σ +
the answer) via `_run`, so `-s` shows exactly what the graph did.

── Corpus reality (sampled from the indexed chunks, not the descriptions) ──
OCR quality differs sharply per document, which dictates which outcome each
collection can produce:
  • 08_Geografiya_Baranskiy_1933 (611 chunks) — CLEAN Cyrillic. Verifiable
    facts: Карагандинский угольный бассейн, цветная металлургия (медь/цинк/
    свинец), сахарная свёкла (СССР 1-е место, Украина), Березниковский
    химкомбинат, реки Сибири/Амур, структура народного хозяйства. → ANSWER_FOUND.
  • 10-astronomiya_vorontcov-velyaminov_1966 (207) — partly readable; the
    opening ("астрономия — наука о небесных телах") is clean. → ANSWER_FOUND
    for the subject, degraded deeper in.
  • 07_Rodnaya_literatura_Snezhnevskaya_1991 (698) — OCR garble (Cyrillic read
    as Latin); only a passing Пушкин mention. → EXHAUSTED-THIN / graceful noise.
  • 09-10-obschaya-biologiya_polyanskiy_1987 (469) — OCR garble. → graceful noise.

So the four case groups below are: clean facts → synthesis; an underspecified
thin topic → exhausted-but-synthesised; absent topics → honest refusal; and
OCR-degraded collections → clean termination either way (never a crash).

Doubly gated (skips unless both hold): GIGACHAT_CREDENTIALS present (read via
general_settings → .env) AND the textbook corpus discoverable under data/lancedb/.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import general_settings
from src.graph import build_graph
from src.state import make_initial_state

# Substrings identifying the textbook corpus among any LanceDB dirs on disk.
# A directory is the corpus if its table set covers at least three of these —
# enough to be unambiguous without pinning the volatile project UUID.
_CORPUS_MARKERS = ("literatura", "geografiya", "biologi", "astronomiya")


def _find_corpus_db() -> tuple[str, list[str]] | None:
    """Locate the indexed textbook corpus on disk, UUID-agnostically.

    LanceDB stores each table (collection) as a `{name}.lance` directory, so the
    collection names are just the `.lance` stems — no DB process or embedding
    model needed to enumerate them. Returns (db_path, collections) for the first
    dir matching ≥3 corpus markers, else None.
    """
    lancedb_root = Path(__file__).parent.parent / "data" / "lancedb"
    if not lancedb_root.is_dir():
        return None
    for db_dir in sorted(lancedb_root.iterdir()):
        if not db_dir.is_dir():
            continue
        collections = sorted(p.stem for p in db_dir.glob("*.lance"))
        lowered = " ".join(collections).lower()
        if sum(m in lowered for m in _CORPUS_MARKERS) >= 3:
            return str(db_dir), collections
    return None


_CORPUS = _find_corpus_db()

requires_live_corpus = pytest.mark.skipif(
    not general_settings.gigachat_credentials or _CORPUS is None,
    reason=(
        "live corpus test needs GIGACHAT_CREDENTIALS (.env) and the indexed "
        "textbook corpus under data/lancedb/"
    ),
)

# All live tests share ONE session-scoped event loop. get_llm() is lru_cached,
# so the GigaChat httpx async client is reused across tests; with the default
# per-test loop, connections pooled on a closed loop raise "Event loop is
# closed" during teardown. A single loop matches how the web app runs (one
# long-lived loop per process) — the graph itself is untouched.
pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture(scope="session")
def corpus_db_path() -> str:
    """db_path of the indexed textbook corpus (skips if absent)."""
    assert _CORPUS is not None  # guarded by requires_live_corpus
    return _CORPUS[0]


# ── trace reporting + run helper ─────────────────────────────────────────────

def _trace_agents(state: dict) -> list[str]:
    return [e.get("agent") for e in state.get("trace", [])]


def _judge_verdicts(state: dict) -> list[str]:
    """The verdict string from each sufficient_context trace entry, in order."""
    out = []
    for e in state.get("trace", []):
        if e.get("agent") == "sufficient_context":
            decision = e.get("decision", "")
            if "verdict=" in decision:
                out.append(decision.split("verdict=", 1)[1].strip())
    return out


def _reached_synthesis(state: dict) -> bool:
    return "synthesis" in _trace_agents(state)


def _reached_give_up(state: dict) -> bool:
    return "give_up" in _trace_agents(state)


def _format_trace(query: str, state: dict) -> str:
    """Compact per-node trace for `-s` viewing: decisions, verdicts, tokens, answer."""
    lines = [f"\n┌─ QUERY: {query!r}"]
    tin = tout = 0
    for e in state.get("trace", []):
        tin += e.get("input_tokens", 0) or 0
        tout += e.get("output_tokens", 0) or 0
        lines.append(f"│  {str(e.get('agent')):<18} {e.get('decision')}")
        info = " ⏎ ".join((e.get("info") or "").split("\n")).strip()
        if info:
            lines.append(f"│      {info[:200]}{'…' if len(info) > 200 else ''}")
    verdicts = _judge_verdicts(state)
    if verdicts:
        lines.append(f"│  verdict chain: {' → '.join(verdicts)}")
    lines.append(f"│  Σ tokens  in={tin}  out={tout}")
    ans = " ".join((state.get("final_answer") or "").split())
    lines.append(f"└─ ANSWER: {ans[:320]}{'…' if len(ans) > 320 else ''}")
    return "\n".join(lines)


async def _run(query: str, db_path: str, **kw) -> dict:
    """Run the unchanged graph to completion; print its trace; return final state."""
    graph = build_graph()
    state = make_initial_state(query=query, db_path=db_path, **kw)
    final = await graph.ainvoke(
        state, config={"configurable": {"thread_id": f"live-{abs(hash(query))}"}}
    )
    print(_format_trace(query, final))
    return final


def _contains_any(text: str, stems: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(s.lower() in low for s in stems)


def _skip_on_llm_failure(state: dict) -> None:
    """Skip when the run aborted on a transient GigaChat failure, not a verdict.

    A weak model occasionally returns no tool call (or a transport drop) for a
    structured node; after STRUCTURED_MAX_RETRIES the llm_failsafe wrapper routes
    to give_up with `llm_error` set. That is the model's reliability, not our
    routing logic — failing on it would be noise, and (worse) it would let a
    refusal case pass for the wrong reason. Skip such runs so the assertions only
    judge genuine retrieval outcomes (llm_error == "" on those)."""
    err = (state.get("llm_error") or "").strip()
    if err:
        pytest.skip(f"transient GigaChat failure, not a logic outcome: {err[:160]}")


# ── 1. ANSWER FOUND — clean collections answer a concrete question → SYNTHESIS ──
# (query, stems any of which the grounded answer should contain)
ANSWER_FOUND_CASES = [
    pytest.param(
        "Что говорится про Карагандинский угольный бассейн?",
        ("караганд", "уголь", "акмолинск"),
        id="geo-karaganda",
    ),
    pytest.param(
        "Где в СССР выращивают сахарную свёклу?",
        ("свёкл", "свекл", "украин", "сахар"),
        id="geo-sugar-beet",
    ),
    pytest.param(
        "Что сказано про цветную металлургию?",
        ("медь", "цинк", "свинец", "металл"),
        id="geo-nonferrous-metals",
    ),
    pytest.param(
        "Расскажи про структуру народного хозяйства довоенной России",
        ("хозяйств", "росси", "революц", "промышленн", "сельск"),
        id="geo-national-economy",
    ),
    pytest.param(
        "Что изучает астрономия?",
        ("небесн", "тел", "астроном", "строени"),
        id="astro-subject",
    ),
]


@requires_live_corpus
@pytest.mark.parametrize("query, stems", ANSWER_FOUND_CASES)
async def test_answer_found(corpus_db_path, query, stems):
    """A concrete question a clean collection covers → synthesis, grounded."""
    state = await _run(query, corpus_db_path)
    _skip_on_llm_failure(state)

    assert _reached_synthesis(state) and not _reached_give_up(state), (
        f"expected synthesis; agents={_trace_agents(state)}, "
        f"verdicts={_judge_verdicts(state)}"
    )
    answer = state.get("final_answer", "")
    assert answer.strip()
    assert _contains_any(answer, stems), (
        f"answer not grounded in expected content {stems}:\n{answer}"
    )


# ── 2. UNDERSPECIFIED THIN topic — exhausted but present → SYNTHESIS ─────────
# The corpus has only passing mentions, so it is exhausted (not empty): the
# judge must rule a synthesis verdict and answer honestly, NOT loop to give_up.
# This is the regression the Literal-verdict redesign fixed.
EXHAUSTED_THIN_CASES = [
    pytest.param("пушкин", ("пушкин",), id="thin-pushkin"),
]


@requires_live_corpus
@pytest.mark.parametrize("query, stems", EXHAUSTED_THIN_CASES)
async def test_underspecified_thin_synthesizes(corpus_db_path, query, stems):
    """An underspecified query over a thin topic synthesises the honest
    "только упоминания" answer rather than refusing."""
    state = await _run(query, corpus_db_path)
    _skip_on_llm_failure(state)

    assert _reached_synthesis(state) and not _reached_give_up(state), (
        f"expected synthesis (exhausted-thin); agents={_trace_agents(state)}, "
        f"verdicts={_judge_verdicts(state)}"
    )
    answer = state.get("final_answer", "")
    assert answer.strip()
    assert _contains_any(answer, stems), f"answer off-topic:\n{answer}"


# ── 3. ABSENT topic — nothing in any textbook → honest refusal (give_up) ─────
REFUSAL_CASES = [
    pytest.param(
        "Как настроить VPN-туннель на маршрутизаторе Cisco?", id="absent-cisco-vpn"
    ),
    pytest.param("Дай пошаговый рецепт борща с говядиной", id="absent-borscht"),
    pytest.param("Какая сейчас рыночная цена биткоина?", id="absent-bitcoin"),
]


@requires_live_corpus
@pytest.mark.parametrize("query", REFUSAL_CASES)
async def test_absent_topic_is_refused(corpus_db_path, query):
    """A topic no 1930s–1990s school textbook covers must end in an honest
    refusal (give_up), never a fabricated answer."""
    state = await _run(query, corpus_db_path)
    _skip_on_llm_failure(state)  # an llm_error give_up is not an honest refusal

    assert _reached_give_up(state) and not _reached_synthesis(state), (
        f"expected give_up; agents={_trace_agents(state)}, "
        f"verdicts={_judge_verdicts(state)}"
    )
    assert state.get("final_answer", "").strip(), "give_up produced no refusal text"


# ── 4. OCR-DEGRADED collections — must terminate cleanly either way ──────────
# Literature/biology indexed from scanned PDFs is OCR garble, so retrieval
# returns noise. The system must still TERMINATE honestly — synthesise the thin
# real signal or refuse — and never crash or hang. We assert clean termination,
# not a specific outcome (asserting a verdict over OCR noise would be flaky).
DEGRADED_CASES = [
    pytest.param("Тарас Бульба", id="degraded-taras-bulba"),
    pytest.param("стихи Лермонтова", id="degraded-lermontov"),
    pytest.param("что такое фотосинтез", id="degraded-photosynthesis"),
    pytest.param("кометы и их хвосты", id="degraded-comets"),
]


@requires_live_corpus
@pytest.mark.parametrize("query", DEGRADED_CASES)
async def test_degraded_collection_terminates_cleanly(corpus_db_path, query):
    """Over OCR-noisy collections the run must reach a terminal node with a
    non-empty answer (synthesis OR give_up) — graceful, never a crash."""
    state = await _run(query, corpus_db_path)
    _skip_on_llm_failure(state)

    terminal = _reached_synthesis(state) ^ _reached_give_up(state)
    assert terminal, (
        f"expected exactly one terminal node; agents={_trace_agents(state)}"
    )
    assert state.get("final_answer", "").strip(), "no final answer produced"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
