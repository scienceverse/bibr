"""Tests for Scoped-Hierarchy v5-lite: study-marker detection, provisional
scope assignment, scope closing, scoped hierarchy assignment, per-scope
IMRaD dedup, and the LLM tier-3 scope-context prompt addition."""

import copy
import random
from unittest import mock
from unittest.mock import AsyncMock

from bibr.paper import enforce_imrad_order
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.structure.section_tree import (
    IMRAD_ANCHORS,
    INTERLUDE_TYPES,
    assign_hierarchy_from_top_level,
    assign_provisional_scopes,
    close_scopes,
    detect_study_markers,
    infer_level_from_numbering,
)


def _section(
    sid: int,
    header: str,
    type_: CanonicalSection = CanonicalSection.UNKNOWN,
    is_top: bool | None = None,
    level: int = 2,
    parent: int | None = 0,
) -> PaperSection:
    sec = PaperSection(section_id=sid, header=header, level=level, parent_section_id=parent)
    sec.section_type = type_
    sec.is_top_level_predicted = is_top
    return sec


def _sections(*headers: str) -> list[PaperSection]:
    return [_section(i + 1, h) for i, h in enumerate(headers)]


# ---------------------------------------------------------------------------
# A1. Marker detection
# ---------------------------------------------------------------------------


class TestDetectStudyMarkers:
    def test_basic_numeric_markers(self):
        secs = _sections("Study 1", "Experiment 2", "Exp. 3", "STUDY 4")
        markers = detect_study_markers(secs)
        assert set(markers) == {1, 2, 3, 4}
        assert [markers[i].base for i in (1, 2, 3, 4)] == ["1", "2", "3", "4"]
        assert not any(m.is_separator for m in markers.values())

    def test_roman_numeral_markers(self):
        secs = _sections("Study II", "Experiment IV: Results")
        markers = detect_study_markers(secs)
        assert markers[1].base == "ii"
        assert markers[2].base == "iv"

    def test_substudy_letters_coalesce_to_base(self):
        secs = _sections("Study 1a", "Method", "Study 1b", "Method")
        markers = detect_study_markers(secs)
        assert markers[1].base == "1"
        assert markers[3].base == "1"
        scopes = assign_provisional_scopes(secs, markers)
        # 1a and 1b share one scope — no scope explosion.
        assert scopes[1] == scopes[2] == scopes[3] == scopes[4] == 1

    def test_compound_marker_remainder_alias(self):
        secs = _sections("Experiment 2: Method")
        markers = detect_study_markers(secs)
        assert markers[1].base == "2"
        assert markers[1].remainder_type == CanonicalSection.METHODS

    def test_compound_marker_without_alias_hit(self):
        secs = _sections("Study 2: The Effect of Sleep")
        markers = detect_study_markers(secs)
        assert markers[1].remainder_type is None

    def test_non_marker_headers(self):
        secs = _sections(
            "study participants",
            "Study design",
            "Studying memory",
            "Experimental setup",
            "Study",
            "pilot study results showed",
        )
        assert detect_study_markers(secs) == {}

    def test_separator_full_header_only(self):
        # Separators count only alongside a real numbered marker — see
        # test_lone_separator_ignored for the single-study guard.
        secs = _sections(
            "Study 1", "Pilot Study", "Replication", "Follow-up Study", "Pilot Experiment"
        )
        markers = detect_study_markers(secs)
        assert set(markers) == {1, 2, 3, 4, 5}
        assert all(m.is_separator for sid, m in markers.items() if sid != 1)

    def test_lone_separator_ignored(self):
        # "Pilot Study" as a Method subsection of a single-study paper must
        # not open a scope: with no real numbered/lettered marker anywhere,
        # separators are dropped entirely and the paper stays zero-marker
        # (identity with the unscoped implementation).
        secs = _sections("Introduction", "Method", "Pilot Study", "Participants", "Results")
        assert detect_study_markers(secs) == {}

    def test_lone_separator_ignored_multiple(self):
        secs = _sections("Pilot Study", "Replication", "Follow-up Study")
        assert detect_study_markers(secs) == {}

    def test_root_section_skipped(self):
        root = _section(0, "Study 1", level=0, parent=None)
        assert detect_study_markers([root]) == {}


# ---------------------------------------------------------------------------
# A1. Provisional scope assignment
# ---------------------------------------------------------------------------


class TestAssignProvisionalScopes:
    def test_scope_zero_before_first_marker_then_increments(self):
        secs = _sections("Introduction", "Study 1", "Method", "Study 2", "Method")
        markers = detect_study_markers(secs)
        scopes = assign_provisional_scopes(secs, markers)
        assert scopes == {1: 0, 2: 1, 3: 1, 4: 2, 5: 2}

    def test_separator_opens_new_scope(self):
        secs = _sections("Study 1", "Method", "Pilot Study", "Method")
        markers = detect_study_markers(secs)
        scopes = assign_provisional_scopes(secs, markers)
        assert scopes == {1: 1, 2: 1, 3: 2, 4: 2}

    def test_no_markers_all_scope_zero(self):
        secs = _sections("Introduction", "Method", "Results")
        scopes = assign_provisional_scopes(secs, {})
        assert set(scopes.values()) == {0}


# ---------------------------------------------------------------------------
# A2. Scope closing (post-classification)
# ---------------------------------------------------------------------------


class TestCloseScopes:
    def test_general_discussion_closes_scopes(self):
        secs = [
            _section(1, "Study 1"),
            _section(2, "Method", CanonicalSection.METHODS),
            _section(3, "General Discussion", CanonicalSection.DISCUSSION),
            _section(4, "Constraints on Generality", CanonicalSection.UNKNOWN),
        ]
        markers = detect_study_markers(secs)
        scopes = close_scopes(secs, assign_provisional_scopes(secs, markers), markers)
        assert scopes[1] == 1 and scopes[2] == 1
        # The closing section AND its followers revert to scope 0.
        assert scopes[3] == 0 and scopes[4] == 0

    def test_plain_discussion_stays_in_scope(self):
        secs = [
            _section(1, "Study 2"),
            _section(2, "Discussion", CanonicalSection.DISCUSSION),
        ]
        markers = detect_study_markers(secs)
        scopes = close_scopes(secs, assign_provisional_scopes(secs, markers), markers)
        assert scopes[2] == 1

    def test_back_matter_closes_scopes(self):
        for closing_type in (CanonicalSection.REFERENCES, CanonicalSection.ACKNOWLEDGMENT):
            secs = [
                _section(1, "Study 1"),
                _section(2, "Results", CanonicalSection.RESULTS),
                _section(3, "Back Matter", closing_type),
                _section(4, "Appendix Notes", CanonicalSection.UNKNOWN),
            ]
            markers = detect_study_markers(secs)
            scopes = close_scopes(secs, assign_provisional_scopes(secs, markers), markers)
            assert scopes[3] == 0 and scopes[4] == 0, closing_type

    def test_closing_set_is_exactly_interlude_types(self):
        # Spec A2's closing set must stay in sync with INTERLUDE_TYPES.
        assert (
            frozenset(
                {
                    CanonicalSection.REFERENCES,
                    CanonicalSection.ACKNOWLEDGMENT,
                    CanonicalSection.FUNDING,
                    CanonicalSection.COI,
                    CanonicalSection.ETHICS,
                    CanonicalSection.AUTHOR_CONTRIBUTIONS,
                    CanonicalSection.OPEN_DATA,
                    CanonicalSection.KEYWORDS,
                    CanonicalSection.ENDNOTE,
                    CanonicalSection.APPENDIX,
                }
            )
            == INTERLUDE_TYPES
        )

    def test_marker_reopens_scope_after_close(self):
        secs = [
            _section(1, "Study 1"),
            _section(2, "General Discussion", CanonicalSection.DISCUSSION),
            _section(3, "Study 2"),
            _section(4, "Method", CanonicalSection.METHODS),
        ]
        markers = detect_study_markers(secs)
        scopes = close_scopes(secs, assign_provisional_scopes(secs, markers), markers)
        assert scopes[2] == 0
        assert scopes[3] == 2 and scopes[4] == 2


# ---------------------------------------------------------------------------
# C. Scoped hierarchy assignment
# ---------------------------------------------------------------------------


def _multistudy_fixture() -> tuple[list[PaperSection], dict[int, int], set[int]]:
    secs = [
        _section(1, "Introduction", CanonicalSection.INTRODUCTION),
        _section(2, "Study 1", CanonicalSection.UNKNOWN),
        _section(3, "Method", CanonicalSection.METHODS),
        _section(4, "Results", CanonicalSection.RESULTS),
        _section(5, "Study 2", CanonicalSection.UNKNOWN),
        _section(6, "Method", CanonicalSection.METHODS),
        _section(7, "Results", CanonicalSection.RESULTS),
        _section(8, "General Discussion", CanonicalSection.DISCUSSION),
        _section(9, "References", CanonicalSection.REFERENCES),
    ]
    markers = detect_study_markers(secs)
    scopes = close_scopes(secs, assign_provisional_scopes(secs, markers), markers)
    return secs, scopes, set(markers)


class TestScopedHierarchy:
    def test_study2_methods_gets_own_anchor(self):
        """Per-scope first_of_type: Study 2's Method/Results never fold under
        Study 1's anchors."""
        secs, scopes, marker_ids = _multistudy_fixture()
        assign_hierarchy_from_top_level(secs, scope_ids=scopes, marker_ids=marker_ids)
        by_id = {s.section_id: s for s in secs}
        for sid in (3, 4, 6, 7):
            assert by_id[sid].level == 1, f"section {sid}"
            assert by_id[sid].parent_section_id == 0, f"section {sid}"
        assert by_id[8].level == 1 and by_id[8].parent_section_id == 0
        assert by_id[9].level == 1 and by_id[9].parent_section_id == 0

    def test_marker_exempt_from_unknown_fold(self):
        """An UNKNOWN-typed marker after Study 1's Results must NOT fold under
        the Results anchor — it pins to level=1, parent=0."""
        secs, scopes, marker_ids = _multistudy_fixture()
        assign_hierarchy_from_top_level(secs, scope_ids=scopes, marker_ids=marker_ids)
        by_id = {s.section_id: s for s in secs}
        assert by_id[2].level == 1 and by_id[2].parent_section_id == 0
        assert by_id[5].level == 1 and by_id[5].parent_section_id == 0
        assert by_id[5].section_type == CanonicalSection.UNKNOWN  # type kept

    def test_marker_updates_top_level_not_imrad_anchor(self):
        secs = [
            _section(1, "Study 2", CanonicalSection.UNKNOWN),
            # is_top=False subsection folds under the most recent top level —
            # the study header itself.
            _section(2, "Overview", CanonicalSection.UNKNOWN, is_top=False),
            # UNKNOWN positional fold has no scope-local IMRaD anchor (the
            # marker must not have become one) — left untouched.
            _section(3, "Stray", CanonicalSection.UNKNOWN, level=2, parent=0),
        ]
        scopes = {1: 1, 2: 1, 3: 1}
        assign_hierarchy_from_top_level(secs, scope_ids=scopes, marker_ids={1})
        assert secs[1].level == 2 and secs[1].parent_section_id == 1
        assert secs[2].level == 2 and secs[2].parent_section_id == 0

    def test_unknown_folds_under_scope_local_anchor(self):
        secs = [
            _section(1, "Study 1", CanonicalSection.UNKNOWN),
            _section(2, "Method", CanonicalSection.METHODS),
            _section(3, "Study 2", CanonicalSection.UNKNOWN),
            _section(4, "Method", CanonicalSection.METHODS),
            _section(5, "Apparatus", CanonicalSection.UNKNOWN),
        ]
        scopes = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2}
        assign_hierarchy_from_top_level(secs, scope_ids=scopes, marker_ids={1, 3})
        # Apparatus folds under Study 2's Method (id 4), not Study 1's (id 2).
        assert secs[4].level == 2 and secs[4].parent_section_id == 4


# ---------------------------------------------------------------------------
# Hard invariant: scope_ids None / all-zero ⇒ byte-identical to the frozen
# pre-refactor implementation.
# ---------------------------------------------------------------------------


def _frozen_assign_hierarchy_from_top_level(sections: list[PaperSection]) -> None:
    """Verbatim copy of assign_hierarchy_from_top_level before the scoped
    refactor (main @ 39dc666). Do not edit."""
    most_recent_top_level: int = 0
    most_recent_imrad_anchor: int = 0
    first_of_type: dict[CanonicalSection, int] = {}
    for sec in sections:
        if sec.level == 0:
            continue
        if infer_level_from_numbering(sec.header) is not None:
            if sec.level == 1:
                most_recent_top_level = sec.section_id
                if sec.section_type in IMRAD_ANCHORS:
                    most_recent_imrad_anchor = sec.section_id
                    first_of_type.setdefault(sec.section_type, sec.section_id)
            continue

        is_top = getattr(sec, "is_top_level_predicted", None)

        if is_top is True:
            sec.level = 1
            sec.parent_section_id = 0
            most_recent_top_level = sec.section_id
            if sec.section_type in IMRAD_ANCHORS:
                most_recent_imrad_anchor = sec.section_id
                first_of_type.setdefault(sec.section_type, sec.section_id)
        elif is_top is False:
            sec.level = 2
            sec.parent_section_id = most_recent_top_level
        else:
            if sec.section_type in IMRAD_ANCHORS:
                if sec.section_type in first_of_type:
                    sec.level = 2
                    sec.parent_section_id = first_of_type[sec.section_type]
                else:
                    sec.level = 1
                    sec.parent_section_id = 0
                    most_recent_top_level = sec.section_id
                    most_recent_imrad_anchor = sec.section_id
                    first_of_type[sec.section_type] = sec.section_id
            elif sec.section_type in INTERLUDE_TYPES:
                sec.level = 1
                sec.parent_section_id = 0
                most_recent_top_level = sec.section_id
            elif sec.section_type == CanonicalSection.UNKNOWN and most_recent_imrad_anchor:
                sec.level = 2
                sec.parent_section_id = most_recent_imrad_anchor


def _fixture_section_lists() -> list[list[PaperSection]]:
    """Section lists mirroring every existing test fixture plus generated ones."""
    fixtures: list[list[PaperSection]] = [
        [_section(1, "Method", CanonicalSection.METHODS, True)],
        [
            _section(1, "Method", CanonicalSection.METHODS, True),
            _section(2, "Participants", CanonicalSection.UNKNOWN, False),
            _section(3, "Procedure", CanonicalSection.UNKNOWN, False),
        ],
        [_section(1, "Stray", CanonicalSection.UNKNOWN, False)],
        [_section(1, "1.1 Encoder", CanonicalSection.UNKNOWN, False, level=3, parent=5)],
        [
            _section(1, "Method", CanonicalSection.METHODS, True),
            _section(2, "Participants", CanonicalSection.UNKNOWN, False),
            _section(3, "Results", CanonicalSection.RESULTS, True),
            _section(4, "Subsec", CanonicalSection.UNKNOWN, False),
        ],
        [_section(1, "Method", CanonicalSection.METHODS)],
        [
            _section(1, "Method", CanonicalSection.METHODS),
            _section(2, "Apparatus", CanonicalSection.UNKNOWN),
            _section(3, "Stimuli", CanonicalSection.UNKNOWN),
        ],
        [
            _section(1, "Mystery Heading", CanonicalSection.UNKNOWN),
            _section(2, "Method", CanonicalSection.METHODS),
        ],
        [
            _section(1, "Method", CanonicalSection.METHODS),
            _section(2, "Open Practices", CanonicalSection.OPEN_DATA),
            _section(3, "Apparatus", CanonicalSection.UNKNOWN),
        ],
        [
            _section(1, "Funding", CanonicalSection.FUNDING),
            _section(2, "Conflict of Interest", CanonicalSection.COI),
        ],
        [
            _section(1, "Method", CanonicalSection.METHODS),
            _section(2, "Apparatus", CanonicalSection.UNKNOWN),
            _section(3, "Results", CanonicalSection.RESULTS),
            _section(4, "Power Analysis", CanonicalSection.UNKNOWN),
        ],
        [
            _section(1, "Method", CanonicalSection.METHODS),
            _section(2, "Statistical Analysis", CanonicalSection.METHODS),
            _section(3, "Apparatus", CanonicalSection.UNKNOWN),
            _section(4, "Results", CanonicalSection.RESULTS),
        ],
        [
            _section(1, "Method", CanonicalSection.METHODS),
            _section(2, "Surprise Top-Level", CanonicalSection.UNKNOWN, True),
        ],
    ]

    rng = random.Random(42)  # noqa: S311 — deterministic fixture generation, not crypto
    type_pool = [
        CanonicalSection.UNKNOWN,
        CanonicalSection.METHODS,
        CanonicalSection.RESULTS,
        CanonicalSection.DISCUSSION,
        CanonicalSection.INTRODUCTION,
        CanonicalSection.ABSTRACT,
        CanonicalSection.REFERENCES,
        CanonicalSection.OPEN_DATA,
        CanonicalSection.COI,
        CanonicalSection.FUNDING,
        CanonicalSection.ENDNOTE,
        CanonicalSection.FOOTNOTE,
        CanonicalSection.TITLE,
        CanonicalSection.KEYWORDS,
    ]
    header_pool = [
        "Method",
        "Results",
        "Discussion",
        "Apparatus",
        "1 Introduction",
        "2.1 Subsection",
        "3. Numbered",
        "Random Heading",
        "Stimuli",
        "",
    ]
    for _ in range(100):
        n = rng.randint(1, 12)
        secs = []
        for sid in range(1, n + 1):
            secs.append(
                _section(
                    sid,
                    rng.choice(header_pool),
                    rng.choice(type_pool),
                    rng.choice([True, False, None]),
                    level=rng.choice([0, 1, 2, 3]),
                    parent=rng.choice([0, 0, 1, 5]),
                )
            )
        fixtures.append(secs)
    return fixtures


def _triples(secs: list[PaperSection]) -> list[tuple[int, int, int | None]]:
    return [(s.section_id, s.level, s.parent_section_id) for s in secs]


class TestNoScopeIdentityInvariant:
    def test_scope_ids_none_is_identity(self):
        for fixture in _fixture_section_lists():
            frozen = copy.deepcopy(fixture)
            current = copy.deepcopy(fixture)
            _frozen_assign_hierarchy_from_top_level(frozen)
            assign_hierarchy_from_top_level(current)
            assert _triples(current) == _triples(frozen)

    def test_scope_ids_all_zero_is_identity(self):
        for fixture in _fixture_section_lists():
            frozen = copy.deepcopy(fixture)
            current = copy.deepcopy(fixture)
            _frozen_assign_hierarchy_from_top_level(frozen)
            scope_ids = {s.section_id: 0 for s in current}
            assign_hierarchy_from_top_level(current, scope_ids=scope_ids, marker_ids=set())
            assert _triples(current) == _triples(frozen)


# ---------------------------------------------------------------------------
# enforce_imrad_order: dedup of unique types stays GLOBAL even with scopes.
# A mid-study "Summary" (ABSTRACT alias) must not survive as a second
# abstract — there is no legitimate per-study abstract/references pattern.
# ---------------------------------------------------------------------------


class TestEnforceImradOrderGlobalDedup:
    def test_mid_study_summary_reset_even_across_scopes(self):
        secs = [
            _section(1, "Abstract", CanonicalSection.ABSTRACT),
            _section(2, "Study 1", CanonicalSection.UNKNOWN),
            _section(3, "Summary", CanonicalSection.ABSTRACT),
        ]
        enforce_imrad_order(secs)
        assert secs[0].section_type == CanonicalSection.ABSTRACT
        assert secs[2].section_type == CanonicalSection.UNKNOWN
        assert secs[2].classification_score == 0.0

    def test_duplicate_references_reset(self):
        secs = [
            _section(1, "References", CanonicalSection.REFERENCES),
            _section(2, "Bibliography", CanonicalSection.REFERENCES),
        ]
        enforce_imrad_order(secs)
        assert secs[0].section_type == CanonicalSection.REFERENCES
        assert secs[1].section_type == CanonicalSection.UNKNOWN
        assert secs[1].classification_score == 0.0

    def test_non_unique_types_repeat_freely(self):
        secs = [
            _section(1, "Abstract", CanonicalSection.ABSTRACT),
            _section(2, "Summary", CanonicalSection.ABSTRACT),
            _section(3, "Method", CanonicalSection.METHODS),
            _section(4, "Method", CanonicalSection.METHODS),
        ]
        enforce_imrad_order(secs)
        assert secs[1].section_type == CanonicalSection.UNKNOWN  # duplicate unique type
        assert secs[3].section_type == CanonicalSection.METHODS  # non-unique repeats allowed


# ---------------------------------------------------------------------------
# B. LLM tier-3 scope context
# ---------------------------------------------------------------------------


def _stub_llm(classifications):
    from types import SimpleNamespace

    result_obj = SimpleNamespace(
        classifications=[SimpleNamespace(header=h, section_type=t) for (h, t) in classifications]
    )
    client = mock.MagicMock()
    client.limiter = mock.MagicMock()
    client.limiter.acquire = AsyncMock(return_value=None)
    client.invoke_structured = AsyncMock(return_value=result_obj)
    client.close = AsyncMock(return_value=None)
    return client


class TestLlmScopeContext:
    def _disable_trained_model(self, monkeypatch):
        from bibr.structure import section_classifier

        monkeypatch.setattr(section_classifier, "_model_cache", False)

    async def test_scope_context_lines_prepended(self, monkeypatch):
        self._disable_trained_model(monkeypatch)
        from bibr.structure.section_classifier import classify_headers_batch_async

        client = _stub_llm([("manipulation check", "method"), ("word lists", "method")])
        await classify_headers_batch_async(
            ["Manipulation Check", "Word Lists"],
            llm_client=client,
            scope_context=["Study 1", "Study 1"],
        )
        from bibr.clients.prompts import prompt_text

        prompt = prompt_text(client.invoke_structured.await_args.args[1][0]["content"])
        assert "Headers 1..2 are inside 'Study 1' of a multi-study paper." in prompt

    async def test_multiple_scopes_one_line_each(self, monkeypatch):
        self._disable_trained_model(monkeypatch)
        from bibr.structure.section_classifier import classify_headers_batch_async

        client = _stub_llm([])
        await classify_headers_batch_async(
            ["Norming Oddity", "Manipulation Check", "Word Lists"],
            llm_client=client,
            scope_context=[None, "Study 1", "Study 2"],
        )
        from bibr.clients.prompts import prompt_text

        prompt = prompt_text(client.invoke_structured.await_args.args[1][0]["content"])
        # Indices refer to the LLM batch's own numbering.
        assert "Headers 2..2 are inside 'Study 1' of a multi-study paper." in prompt
        assert "Headers 3..3 are inside 'Study 2' of a multi-study paper." in prompt

    async def test_prompt_byte_identical_without_markers(self, monkeypatch):
        self._disable_trained_model(monkeypatch)
        from bibr.structure.section_classifier import classify_headers_batch_async

        client_a = _stub_llm([])
        await classify_headers_batch_async(["Weird Header"], llm_client=client_a)
        prompt_a = client_a.invoke_structured.await_args.args[1][0]["content"]

        client_b = _stub_llm([])
        await classify_headers_batch_async(
            ["Weird Header"], llm_client=client_b, scope_context=[None]
        )
        prompt_b = client_b.invoke_structured.await_args.args[1][0]["content"]

        assert prompt_a == prompt_b
        assert "multi-study" not in prompt_a


# ---------------------------------------------------------------------------
# End-to-end: _classify_sections (no_llm path — lookup classification only)
# ---------------------------------------------------------------------------


def _multistudy_contents() -> PaperContents:
    headers = [
        ("Root", 0),
        ("Introduction", 1),
        ("Study 1", 1),
        ("Method", 1),
        ("Results", 1),
        ("Study 2", 1),
        ("Method", 1),
        ("Results", 1),
        ("General Discussion", 1),
        ("References", 1),
    ]
    sections = [
        PaperSection(
            section_id=i,
            header=h,
            level=level,
            parent_section_id=None if i == 0 else 0,
        )
        for i, (h, level) in enumerate(headers)
    ]
    return PaperContents(
        sentences=[],
        sections=sections,
        tables=[],
        links=[],
        sections_text=dict.fromkeys(range(len(headers)), ""),
        detected_title=None,
    )


class TestClassifySectionsEndToEnd:
    async def test_multistudy_no_llm(self):
        from bibr.pipeline.stages.post_parse import _classify_sections

        contents = _multistudy_contents()
        await _classify_sections(contents, None, True, None)

        by_id = {s.section_id: s for s in contents.sections}
        # Study 2's Method/Results are fresh level-1 anchors, not folded
        # under Study 1's.
        for sid in (3, 4, 6, 7):
            assert by_id[sid].level == 1 and by_id[sid].parent_section_id == 0
        assert by_id[2].level == 1 and by_id[2].parent_section_id == 0
        assert by_id[5].level == 1 and by_id[5].parent_section_id == 0
        assert by_id[5].section_type == CanonicalSection.UNKNOWN

        # Global dedup only touches unique types — both Results survive.
        enforce_imrad_order(contents.sections)
        assert by_id[4].section_type == CanonicalSection.RESULTS
        assert by_id[7].section_type == CanonicalSection.RESULTS

    async def test_no_markers_no_hierarchy_change(self):
        from bibr.pipeline.stages.post_parse import _classify_sections

        contents = _multistudy_contents()
        contents.sections = [s for s in contents.sections if s.header not in ("Study 1", "Study 2")]
        before = [(s.section_id, s.header) for s in contents.sections]
        await _classify_sections(contents, None, True, None)
        assert [(s.section_id, s.header) for s in contents.sections] == before

    def test_marker_sections_cannot_become_implicit_body_anchors(self):
        # Markers keep their classified type (typically UNKNOWN) — pin that
        # UNKNOWN never counts as a body anchor for implicit-section detection.
        from bibr.structure.implicit_sections import _BODY_TYPES

        assert CanonicalSection.UNKNOWN not in _BODY_TYPES
