"""AgentRAGState — shared state across all LangGraph nodes.

TypedDict with Annotated reducers for accumulation across iterations.
"""

import json
import operator
from typing import Annotated, Any, Literal, Optional, Sequence, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Structured output schemas (Pydantic — used by LLM.with_structured_output) ──
#
# Every constraint a structured result must satisfy is expressed as Pydantic
# validation: a violation raises ValidationError, which generate_structured()
# turns into a clarification re-prompt and, if the model keeps failing, an honest
# give_up. There is one uniform mechanism — no separate per-schema retry path.

def _coerce_list(v):
    """Coerce a list field that arrived as a JSON-encoded string back to a list.

    A weak model's function-calling sometimes serializes an array argument as a
    string (e.g. missing_parts='["a","b"]' instead of ["a","b"]), which fails
    Pydantic's list validation and crashes the whole graph run. Parse the string
    as JSON when possible; a non-JSON string becomes a single-item list.
    """
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else [s]
        except (ValueError, TypeError):
            return [s]
    return v

class RouteStep(BaseModel):
    """A single search route from the Planner.

    Schema-Guided Reasoning: the rationale (why this source) is generated BEFORE
    the collection/subquery it justifies, so the choice follows the reasoning
    instead of being rationalized after the fact.

    All three fields are strictly required and non-empty: a step the model emits
    with a missing or blank collection/subquery (e.g. only 'rationale') is not a
    usable route, so it raises ValidationError. generate_structured then re-prompts
    with the error and, if the model keeps failing, routes to give_up — the same
    strict path used for the scalar verdicts, rather than silently dropping steps.
    """
    rationale: str = Field(description="СНАЧАЛА реши: почему именно эта коллекция — правильное место для поиска нужного фрагмента.")
    collection: str = Field(description="Имя коллекции/таблицы для поиска (должно совпадать с одной из доступных коллекций), выбранное согласно rationale.")
    subquery: str = Field(description="Что конкретно искать в этой коллекции (сфокусированный подзапрос).")

    @field_validator("collection", "subquery")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        # `required` alone accepts "" / whitespace (a valid str); reject those too
        # so a semantically-empty route fails validation instead of searching a
        # nonexistent "" collection.
        if not v or not v.strip():
            raise ValueError("поле должно быть непустой строкой")
        return v


class PlanResult(BaseModel):
    """Planner output: breakdown of the query into search routes."""
    is_multi_step: bool = Field(description="Требует ли вопрос нескольких шагов поиска")
    steps: list[RouteStep] = Field(description="Поисковые маршруты для выполнения")

    _coerce_steps = field_validator("steps", mode="before")(_coerce_list)


# ── Judge verdict: a closed set of retrieval situations (SGR Routing) ─────────
#
# Replaces the old `sufficient: bool`. A weak model rationalizes a free boolean
# it has to justify — it kept reading "is this thin answer good enough? (no)" as
# insufficient even when the corpus was exhausted (the «пушкин» trace: +0 new
# chunks, only one plausible collection, yet four iterations of inventing
# "missing biography/works"). Classifying into NAMED situations is what weak
# models do reliably, and each value's wording carries its own criterion, so the
# decision is "which retrieval state am I in?" — not a quality grade. The bool
# also muddled two orthogonal axes; the situations split them:
#   • route availability  → есть_необысканная_коллекция / есть_неиспробованный_угол
#   • what was found       → ответ_найден / исчерпано_есть_упоминания / ничего_не_найдено
# Each value maps to exactly one graph outcome (sufficient_context_node).
JudgeVerdict = Literal[
    "ответ_найден",
    "исчерпано_есть_упоминания",
    "есть_необысканная_коллекция",
    "есть_неиспробованный_угол",
    "ничего_не_найдено",
]
# Verdicts that route to Synthesis — the corpus has given its answer (the full
# answer, or the honest "the sources contain only …" over thin-but-real findings).
SYNTHESIS_VERDICTS: frozenset[str] = frozenset(
    {"ответ_найден", "исчерпано_есть_упоминания"}
)
# Verdicts that route back to the Planner — a concrete route remains. feedback
# is required for these (the Planner's iteration mode reads it).
CONTINUE_VERDICTS: frozenset[str] = frozenset(
    {"есть_необысканная_коллекция", "есть_неиспробованный_угол"}
)
# "ничего_не_найдено" is neither: nothing found AND nowhere left → give_up
# directly (no point burning the remaining iterations re-searching exhaustion).


class SufficientContextResult(BaseModel):
    """Sufficient Context Agent output.

    Schema-Guided Reasoning: fields follow the order a person would reason in,
    because structured output is generated field-by-field in declaration order.
    First the analysis (`reason`) — what the chunks say about the question AS
    ASKED, without inflating an underspecified query into sub-questions; then the
    attempted answer (`draft_answer`), the gaps (`missing_parts`), and
    `retrieval_state` — a cascade step forcing the model to read the
    route-availability signals (unsearched collection? a collection still
    yielding NEW on-topic chunks, or +0?) BEFORE it commits. Only then the
    `verdict` — a closed Literal of named retrieval situations (NOT a quality
    grade), grounded in everything above instead of committed up front and
    rationalized afterward (which let "not found" drafts pass as sufficient).
    `feedback` comes LAST: describing the information gap is a consequence of
    having concluded a CONTINUE verdict, written after it.

    A `question_verbatim` field used to lead, re-anchoring the judge on the
    question via a literal copy. It was dropped: GigaChat reliably copies short
    queries but rewords longer Russian sentences, and the verbatim-copy validator
    then rejected an otherwise-correct verdict, forcing a spurious give_up. The
    anti-inflation job it did is now carried by `reason`'s wording plus the
    verdict's situation framing (an underspecified query maps cleanly to
    `исчерпано_есть_упоминания`, so there is nothing to gain by inflating it).

    Separation of concerns: the judge speaks the language of INFORMATION (what
    was asked, what was found, which fact is missing). Routing — which collection
    to search next — belongs to the Planner, so collection names are banned from
    `feedback`/`missing_parts` (enforced by make_sufficient_context_schema, which
    knows the inventory).

    Verdict semantics: the question is "what retrieval situation are we in?" — a
    state call, not a grade against an ideal answer. An exhausted corpus whose
    findings are thin (only passing mentions) is `исчерпано_есть_упоминания` and
    routes to Synthesis: the honest answer is "the sources contain only …". Zero
    findings are never synthesised — `ничего_не_найдено` stays an honest refusal.
    """
    reason: str = Field(
        description="САМОЕ ПЕРВОЕ. Анализ из двух частей: (1) что по вопросу пользователя (КАК ОН ЗАДАН, без добавления подвопросов, которых пользователь не задавал) найденные фрагменты содержат, а чего в них нет; (2) по статистике поисков — исчерпана ли база по теме вопроса, или есть конкретная причина ожидать от неё большего (необысканная коллекция, неиспробованные формулировки)."
    )
    # The conditionally-empty fields below (draft_answer, missing_parts,
    # feedback) are REQUIRED on purpose — no defaults. A weak GigaChat omits
    # non-required keys outright (observed: a False verdict with no feedback
    # key at all), and requiredness is the one schema property weak models
    # obey reliably. Optional[...] would not help: langchain-gigachat's
    # converter silently DROPS nullable fields from the function schema's
    # `required` list, so "required flat type + empty value" ("" / []) is the
    # only shape whose requiredness actually reaches the model.
    draft_answer: str = Field(
        description="Лучший ответ на вопрос пользователя, построенный ТОЛЬКО из найденного контекста. Если по теме есть лишь отдельные упоминания — приведи их («в источниках об этом только …»): для исчерпанной базы это и есть ответ. Если не найдено вообще ничего — прямо так и напиши; не заполняй поле из общих знаний.",
    )
    missing_parts: list[str] = Field(
        description="Конкретные недостающие ФАКТЫ — каждый элемент короткая именная группа (например «расшифровка аббревиатуры ЭВМ»). НЕ маршруты и НЕ имена коллекций («поиск в коллекции …» — неправильно), НЕ размытое («более подробная информация»). Пустой список ([]), если черновик уже отвечает на вопрос.",
    )

    _coerce_missing = field_validator("missing_parts", mode="before")(_coerce_list)
    retrieval_state: str = Field(
        description="ПЕРЕД вердиктом проговори СОСТОЯНИЕ ПОИСКА по статистике (это факты, а не оценка качества ответа): (а) есть ли в описи правдоподобно-релевантная коллекция, которой НЕТ в статистике поисков (ещё не обыскана); (б) даёт ли хоть одна уже обысканная коллекция НОВЫЕ по теме чанки — или последний поиск везде дал «+0 новых»/чанки не по теме. Опирайся на цифры статистики, не на свою память.",
    )
    verdict: JudgeVerdict = Field(
        description=(
            "ВЕРДИКТ — выбери РОВНО ОДНУ ситуацию, ПОСЛЕ полей выше (по retrieval_state и draft_answer). "
            "Это состояние ПОИСКА, а не оценка качества ответа:\n"
            "• «ответ_найден» — draft_answer отвечает на вопрос, как он задан (даже если можно накопать ещё деталей);\n"
            "• «исчерпано_есть_упоминания» — все правдоподобно-релевантные коллекции обысканы, новые поиски не приносят нового ПО ТЕМЕ (последний поиск «+0 новых» или чанки не о том), а draft_answer собрал всё найденное, пусть и скудное («в источниках об этом есть только …») — это полноценный честный ответ;\n"
            "• «есть_необысканная_коллекция» — в описи есть правдоподобно-релевантная коллекция, которой НЕТ в статистике поисков;\n"
            "• «есть_неиспробованный_угол» — релевантная коллекция ещё отдаёт НОВЫЕ по теме чанки (её последний поиск НЕ «+0») и есть конкретная неиспробованная формулировка; при «+0 новых» эту ситуацию для такой коллекции выбирать НЕЛЬЗЯ;\n"
            "• «ничего_не_найдено» — по теме нет НИ ОДНОГО упоминания, и искать больше негде (все правдоподобные коллекции обысканы)."
        )
    )
    feedback: str = Field(
        description="В САМОМ КОНЦЕ. Непустой ТОЛЬКО при «есть_необысканная_коллекция» или «есть_неиспробованный_угол» — опиши ИНФОРМАЦИОННЫЙ пробел; удобная форма: «Не хватает: …. Найдено вместо этого: …. Альтернативные формулировки: ….» НЕ называй коллекции и не указывай, где искать — выбор источника не твоя задача. Для остальных вердиктов передай пустую строку.",
    )

    @property
    def sufficient(self) -> bool:
        """True iff the verdict routes to Synthesis (corpus has its answer).

        A compatibility shim so node/trace code and tests can keep asking the
        binary question; the model now emits the richer `verdict` instead.
        """
        return self.verdict in SYNTHESIS_VERDICTS

    @model_validator(mode="after")
    def _verdict_is_actionable(self):
        # Validate only what CODE consumes; what an LLM consumes is guided by
        # the prompt, not policed. A blank missing_parts item is never useful
        # (the give_up renderer and the Planner both read them), so reject it
        # on any verdict. feedback's PRESENCE is a code contract: the Planner's
        # iteration mode keys off bool(feedback), and only CONTINUE_VERDICTS
        # route to the Planner — so feedback is required exactly for those, and
        # an empty one would silently demote the re-route to initial-plan mode.
        # The feedback TEMPLATE is guidance, deliberately NOT enforced: a label
        # check rejects semantically-good feedback worded differently, and a
        # re-prompt burned on formatting can escalate a correct verdict into
        # give_up. The check needing node-time context (collection names)
        # lives in make_sufficient_context_schema.
        if any(not part or not part.strip() for part in self.missing_parts):
            raise ValueError(
                "каждый элемент missing_parts должен быть непустой строкой — "
                "короткой именной группой, называющей конкретный отсутствующий факт."
            )
        if self.verdict in CONTINUE_VERDICTS and not self.feedback.strip():
            raise ValueError(
                "Когда вердикт — «есть_необысканная_коллекция» или "
                "«есть_неиспробованный_угол», ОБЯЗАТЕЛЬНО заполни feedback — "
                "опиши информационный пробел: какая информация отсутствует, что "
                "поиски дали вместо неё, какими ещё формулировками она может "
                "называться. Если же продолжать поиск незачем — выбери вердикт, "
                "отвечающий найденному («ответ_найден», "
                "«исчерпано_есть_упоминания» или «ничего_не_найдено»)."
            )
        return self


def make_sufficient_context_schema(
    collection_names: Sequence[str],
) -> type[SufficientContextResult]:
    """SufficientContextResult bound to node-time context (the inventory).

    One constraint can only be checked with information the node has — the actual
    collection names — so it is baked into a dynamic subclass as a validator.
    That keeps every requirement a Pydantic constraint flowing through the one
    uniform generate_structured re-prompt path: no separate node-level retry loop.

    - feedback/missing_parts of a non-synthesis verdict must not name a
      collection or prescribe where to search — the judge states the information
      gap, the Planner owns routing. Weak models violate this regularly, hence a
      mechanical check rather than a prompt-only rule.
    """
    # Tiny names (1–2 chars) would false-positive on ordinary words; routing
    # leakage that matters always quotes a recognizable table name.
    banned = sorted(
        {n.strip().casefold() for n in collection_names if n and len(n.strip()) >= 3}
    )

    class BoundSufficientContextResult(SufficientContextResult):
        @model_validator(mode="after")
        def _no_routes_in_information_fields(self):
            # The ban applies wherever feedback/missing_parts are consumed
            # downstream — the Planner (CONTINUE verdicts) and the give_up
            # renderer (ничего_не_найдено, which shows missing_parts). On the
            # Synthesis verdicts these fields are unused, so skip the check.
            if self.verdict in SYNTHESIS_VERDICTS:
                return self
            fields = [("feedback", self.feedback)]
            fields += [("missing_parts", part) for part in self.missing_parts]
            for field_name, text in fields:
                lowered = (text or "").casefold()
                for name in banned:
                    if name in lowered:
                        raise ValueError(
                            f"в поле {field_name} указан маршрут вместо "
                            f"информационного пробела: «{name}» — это имя "
                            "коллекции. НЕ называй коллекции и не указывай, где "
                            "искать (это решает планировщик); опиши, КАКОЙ ФАКТ "
                            "отсутствует."
                        )
            return self

    # The class name is the function-calling tool name the model sees (and the
    # name quoted in StructuredGenerationError) — keep it stable across binds.
    BoundSufficientContextResult.__name__ = SufficientContextResult.__name__
    BoundSufficientContextResult.__qualname__ = SufficientContextResult.__qualname__
    # The docstring ships as the tool description in the function-calling
    # schema, and a subclass does not inherit __doc__ (it gets None) —
    # langchain-gigachat rejects a tool whose description is empty.
    BoundSufficientContextResult.__doc__ = SufficientContextResult.__doc__
    return BoundSufficientContextResult


# ── Graph State (TypedDict with Annotated reducers) ──

class AgentRAGState(TypedDict):
    """Full state of the Agentic RAG graph.

    Fields with Annotated[list, operator.add] use concatenation reducer —
    updates are accumulated, not overwritten. Essential for iteration loops.
    """

    # Input
    query: str

    # Vector DB scope — which LanceDB to search (per-project isolation).
    # None → global LANCE_DB_PATH (CLI default).
    db_path: Optional[str]

    # Per-project neighbor-stitching overrides for search_fanout:
    # {"expand_padding": int, "bridge_gap": int}. None / missing keys →
    # the global vdb_settings values (CLI default).
    stitch_settings: Optional[dict]

    # Planner
    plan_steps: list[dict]
    # Current route being processed (set by Planner)
    current_route: Optional[dict]

    # Query Rewriter
    rewritten_queries: Annotated[list[str], operator.add]
    # Current-turn search tasks: [{"collection": str, "query": str}], one per
    # planner route. Overwritten each turn (no reducer).
    search_tasks: list[dict]

    # Search Fanout
    search_results: Annotated[list[dict], operator.add]

    # Sufficient Context
    sufficient: Optional[bool]
    sufficient_reason: str
    feedback: str
    missing_parts: list[str]
    draft_answer: str

    # Iteration control
    iteration_count: int
    max_iterations: int

    # Final answer
    final_answer: str

    # Set by llm_failsafe when a node's LLM call fails unrecoverably — give_up
    # reads it to render an honest "model problem" refusal. "" = no LLM failure.
    llm_error: str

    # Audit trail
    trace: Annotated[list[dict], operator.add]


def make_initial_state(
    query: str,
    max_iterations: int = 3,
    db_path: Optional[str] = None,
    stitch_settings: Optional[dict] = None,
) -> AgentRAGState:
    """Create a clean initial state for the graph.

    db_path scopes vector search to one LanceDB (per-project isolation).
    None → global LANCE_DB_PATH (CLI default, backward-compatible).
    stitch_settings carries per-project neighbor-stitching overrides
    (expand_padding/bridge_gap) into search_fanout; None → vdb_settings.
    """
    return AgentRAGState(
        query=query,
        db_path=db_path,
        stitch_settings=stitch_settings,
        plan_steps=[],
        current_route=None,
        rewritten_queries=[],
        search_tasks=[],
        search_results=[],
        sufficient=None,
        sufficient_reason="",
        feedback="",
        missing_parts=[],
        draft_answer="",
        iteration_count=0,
        max_iterations=max_iterations,
        final_answer="",
        llm_error="",
        trace=[],
    )


def make_trace_entry(agent: str, decision: str, detail: str = "", info: str = "") -> dict:
    """Create a trace entry dict (appended to trace via operator.add reducer).

    `info` is optional human-facing context shown as the middle line of a step in
    the web UI (Planner rationale, searched source+query, judge reason/feedback);
    `detail` stays the compact log-line string. Newlines in `info` render as
    separate lines in the UI.
    """
    return {"agent": agent, "decision": decision, "detail": detail, "info": info}
