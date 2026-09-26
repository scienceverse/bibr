"""Tests for assign_hierarchy_from_top_level."""

from bibr.paper_contents import CanonicalSection, PaperSection
from bibr.structure.section_tree import assign_hierarchy_from_top_level


def _section(sid: int, header: str, type_: CanonicalSection, is_top: bool) -> PaperSection:
    sec = PaperSection(
        section_id=sid,
        header=header,
        level=2,  # placeholder, helper should overwrite
        parent_section_id=0,
    )
    sec.section_type = type_
    sec.is_top_level_predicted = is_top
    return sec


def test_top_level_attaches_to_root():
    secs = [_section(1, "Method", CanonicalSection.METHODS, True)]
    assign_hierarchy_from_top_level(secs)
    assert secs[0].level == 1
    assert secs[0].parent_section_id == 0


def test_subsection_attaches_to_most_recent_top_level():
    secs = [
        _section(1, "Method", CanonicalSection.METHODS, True),
        _section(2, "Participants", CanonicalSection.UNKNOWN, False),
        _section(3, "Procedure", CanonicalSection.UNKNOWN, False),
    ]
    assign_hierarchy_from_top_level(secs)
    assert secs[1].level == 2
    assert secs[1].parent_section_id == 1
    assert secs[2].level == 2
    assert secs[2].parent_section_id == 1


def test_subsection_with_no_prior_top_level_falls_back_to_root():
    secs = [_section(1, "Stray", CanonicalSection.UNKNOWN, False)]
    assign_hierarchy_from_top_level(secs)
    assert secs[0].level == 2
    assert secs[0].parent_section_id == 0


def test_numbered_headings_are_skipped():
    """Headings whose header text starts with '1.', '1.1', etc. keep their existing level/parent."""
    sec = _section(1, "1.1 Encoder", CanonicalSection.UNKNOWN, False)
    sec.level = 3
    sec.parent_section_id = 5
    assign_hierarchy_from_top_level([sec])
    assert sec.level == 3
    assert sec.parent_section_id == 5


def test_top_level_resets_recent_anchor():
    secs = [
        _section(1, "Method", CanonicalSection.METHODS, True),
        _section(2, "Participants", CanonicalSection.UNKNOWN, False),
        _section(3, "Results", CanonicalSection.RESULTS, True),
        _section(4, "Subsec", CanonicalSection.UNKNOWN, False),
    ]
    assign_hierarchy_from_top_level(secs)
    assert secs[3].parent_section_id == 3  # attaches to Results, not Method


# ---------------------------------------------------------------------------
# Positional fallback (supervisor's IMRaD-anchor rule) — fires when
# is_top_level_predicted is unset (None).
# ---------------------------------------------------------------------------


def _typed_section(sid: int, header: str, type_: CanonicalSection) -> PaperSection:
    """Section with section_type set but is_top_level_predicted unset."""
    sec = PaperSection(
        section_id=sid,
        header=header,
        level=2,
        parent_section_id=0,
    )
    sec.section_type = type_
    return sec


def test_imrad_anchor_typed_becomes_top_level():
    """METHODS-typed section with no is_top hint becomes level=1 (anchor)."""
    secs = [_typed_section(1, "Method", CanonicalSection.METHODS)]
    assign_hierarchy_from_top_level(secs)
    assert secs[0].level == 1
    assert secs[0].parent_section_id == 0


def test_unknown_after_imrad_anchor_folds_into_anchor():
    """An UNKNOWN section after an IMRaD anchor attaches as its child."""
    secs = [
        _typed_section(1, "Method", CanonicalSection.METHODS),
        _typed_section(2, "Apparatus", CanonicalSection.UNKNOWN),
        _typed_section(3, "Stimuli", CanonicalSection.UNKNOWN),
    ]
    assign_hierarchy_from_top_level(secs)
    assert secs[0].level == 1
    assert secs[1].level == 2 and secs[1].parent_section_id == 1
    assert secs[2].level == 2 and secs[2].parent_section_id == 1


def test_unknown_before_any_anchor_stays_orphan():
    """An UNKNOWN section with no prior IMRaD anchor is not folded."""
    secs = [
        _typed_section(1, "Mystery Heading", CanonicalSection.UNKNOWN),
        _typed_section(2, "Method", CanonicalSection.METHODS),
    ]
    assign_hierarchy_from_top_level(secs)
    # Mystery section had no anchor to attach to — left at original level/parent.
    assert secs[0].level == 2
    assert secs[0].parent_section_id == 0
    assert secs[1].level == 1


def test_interlude_does_not_reset_anchor():
    """OPEN_DATA between Method and an UNKNOWN should NOT capture the UNKNOWN."""
    secs = [
        _typed_section(1, "Method", CanonicalSection.METHODS),
        _typed_section(2, "Open Practices", CanonicalSection.OPEN_DATA),
        _typed_section(3, "Apparatus", CanonicalSection.UNKNOWN),
    ]
    assign_hierarchy_from_top_level(secs)
    # OPEN_DATA stays as level=1 interlude.
    assert secs[1].level == 1
    assert secs[1].parent_section_id == 0
    # UNKNOWN folds into the IMRaD anchor (Method), not the interlude.
    assert secs[2].level == 2
    assert secs[2].parent_section_id == 1


def test_interlude_promoted_to_top_level():
    """COI/funding/etc. with no is_top hint become level=1 (top-level)."""
    secs = [
        _typed_section(1, "Funding", CanonicalSection.FUNDING),
        _typed_section(2, "Conflict of Interest", CanonicalSection.COI),
    ]
    assign_hierarchy_from_top_level(secs)
    assert secs[0].level == 1
    assert secs[1].level == 1


def test_imrad_anchor_progresses_through_sections():
    """The anchor advances when a new IMRaD section appears."""
    secs = [
        _typed_section(1, "Method", CanonicalSection.METHODS),
        _typed_section(2, "Apparatus", CanonicalSection.UNKNOWN),
        _typed_section(3, "Results", CanonicalSection.RESULTS),
        _typed_section(4, "Power Analysis", CanonicalSection.UNKNOWN),
    ]
    assign_hierarchy_from_top_level(secs)
    assert secs[1].parent_section_id == 1  # under Method
    assert secs[3].parent_section_id == 3  # under Results, not Method


def test_repeat_imrad_type_folds_under_first():
    """A second METHODS-typed section (e.g. 'Statistical Analysis' aliased to
    METHODS) folds under the first METHODS section, not as a new anchor."""
    secs = [
        _typed_section(1, "Method", CanonicalSection.METHODS),
        _typed_section(2, "Statistical Analysis", CanonicalSection.METHODS),
        _typed_section(3, "Apparatus", CanonicalSection.UNKNOWN),
        _typed_section(4, "Results", CanonicalSection.RESULTS),
    ]
    assign_hierarchy_from_top_level(secs)
    assert secs[0].level == 1 and secs[0].parent_section_id == 0
    # Second METHODS folds under first METHODS.
    assert secs[1].level == 2 and secs[1].parent_section_id == 1
    # UNKNOWN folds under the IMRaD anchor (Method, sec 1), not under
    # Statistical Analysis.
    assert secs[2].level == 2 and secs[2].parent_section_id == 1
    assert secs[3].level == 1 and secs[3].parent_section_id == 0


def test_exact_alias_heading_outranks_an_earlier_keyword_anchor():
    """eLife prints a Results subsection "A neural implementation of ..."
    that reads as METHODS by keyword. The later "Materials and methods" is
    exactly the part's name: it starts its own anchor, and its subsections
    fold under it instead of under Discussion."""
    secs = [
        _typed_section(1, "Results", CanonicalSection.RESULTS),
        _typed_section(2, "A neural implementation of oscillation", CanonicalSection.METHODS),
        _typed_section(3, "Discussion", CanonicalSection.DISCUSSION),
        _typed_section(4, "Materials and methods", CanonicalSection.METHODS),
        _typed_section(5, "Strains and culture conditions", CanonicalSection.UNKNOWN),
        _typed_section(6, "Statistical analysis", CanonicalSection.METHODS),
    ]
    for sec, source in zip(
        secs,
        ["exact_alias", "substring_alias", "exact_alias", "exact_alias", None, "substring_alias"],
        strict=True,
    ):
        sec.classification_source = source
    assign_hierarchy_from_top_level(secs)
    assert [(s.section_id, s.level, s.parent_section_id) for s in secs] == [
        (1, 1, 0),
        (2, 1, 0),
        (3, 1, 0),
        (4, 1, 0),
        (5, 2, 4),
        # A later keyword heading folds under the exact one.
        (6, 2, 4),
    ]


def test_exact_part_name_experimental_section_outranks_a_keyword_anchor():
    """Chemistry papers name their Methods part "Experimental Section"; an
    earlier Results subsection read as METHODS by keyword must not swallow it
    and its compound subsections."""
    secs = [
        _typed_section(1, "Results and Discussion", CanonicalSection.RESULTS),
        _typed_section(2, "Development and Implementation of a Library", CanonicalSection.METHODS),
        _typed_section(3, "Summary and Conclusion", CanonicalSection.DISCUSSION),
        _typed_section(4, "Experimental Section", CanonicalSection.METHODS),
        _typed_section(5, "Purification of Products", CanonicalSection.UNKNOWN),
    ]
    for sec, source in zip(
        secs,
        ["exact_alias", "substring_alias", "substring_alias", "exact_alias", None],
        strict=True,
    ):
        sec.classification_source = source
    assign_hierarchy_from_top_level(secs)
    assert [(s.section_id, s.level, s.parent_section_id) for s in secs] == [
        (1, 1, 0),
        (2, 1, 0),
        (3, 1, 0),
        (4, 1, 0),
        (5, 2, 4),
    ]


def test_exact_subsection_names_stay_under_a_keyword_part_heading():
    """Keyword part headings keep their exact-named subsections.

    "Patients and methods" and "Discussion and conclusion" are keyword hits,
    but they are the parts. "Study design", "Statistical analysis" and
    "Limitations" are exact aliases that name subsections, so they fold under
    the part instead of starting one and taking its later subsections."""
    secs = [
        _typed_section(1, "Introduction", CanonicalSection.INTRODUCTION),
        _typed_section(2, "Patients and methods", CanonicalSection.METHODS),
        _typed_section(3, "Study design", CanonicalSection.METHODS),
        _typed_section(4, "Procedure", CanonicalSection.UNKNOWN),
        _typed_section(5, "Statistical analysis", CanonicalSection.METHODS),
        _typed_section(6, "Results", CanonicalSection.RESULTS),
        _typed_section(7, "Discussion and conclusion", CanonicalSection.DISCUSSION),
        _typed_section(8, "Limitations", CanonicalSection.DISCUSSION),
        _typed_section(9, "Implications", CanonicalSection.DISCUSSION),
    ]
    for sec, source in zip(
        secs,
        [
            "exact_alias",
            "substring_alias",
            "exact_alias",
            None,
            "exact_alias",
            "exact_alias",
            "substring_alias",
            "exact_alias",
            "exact_alias",
        ],
        strict=True,
    ):
        sec.classification_source = source
    assign_hierarchy_from_top_level(secs)
    assert [(s.section_id, s.level, s.parent_section_id) for s in secs] == [
        (1, 1, 0),
        (2, 1, 0),
        (3, 2, 2),
        (4, 2, 2),
        (5, 2, 2),
        (6, 1, 0),
        (7, 1, 0),
        (8, 2, 7),
        (9, 2, 7),
    ]


def test_an_exact_part_name_printed_inside_another_part_still_outranks():
    """Known limitation: the rule reads names, not layout. A part name printed
    as a subsection of a keyword part heading, such as "Methods" inside
    "Subjects and methods" or "Conclusions" inside "Discussion and
    conclusion", starts a part of its own and takes the subsections after
    it."""
    secs = [
        _typed_section(1, "Subjects and methods", CanonicalSection.METHODS),
        _typed_section(2, "Methods", CanonicalSection.METHODS),
        _typed_section(3, "Statistics", CanonicalSection.UNKNOWN),
        _typed_section(4, "Results", CanonicalSection.RESULTS),
        _typed_section(5, "Discussion and conclusion", CanonicalSection.DISCUSSION),
        _typed_section(6, "Conclusions", CanonicalSection.DISCUSSION),
        _typed_section(7, "Practical implications", CanonicalSection.UNKNOWN),
    ]
    for sec, source in zip(
        secs,
        [
            "substring_alias",
            "exact_alias",
            None,
            "exact_alias",
            "substring_alias",
            "exact_alias",
            None,
        ],
        strict=True,
    ):
        sec.classification_source = source
    assign_hierarchy_from_top_level(secs)
    assert [(s.section_id, s.level, s.parent_section_id) for s in secs] == [
        (1, 1, 0),
        (2, 1, 0),
        (3, 2, 2),
        (4, 1, 0),
        (5, 1, 0),
        (6, 1, 0),
        (7, 2, 6),
    ]


def test_part_headings_are_exact_imrad_aliases():
    """Every outranking part name must be an exact alias of an IMRaD part,
    or the rule could never see it (it fires on exact alias hits only)."""
    from bibr.structure.section_classifier import _ALIAS_EXACT_LOOKUP
    from bibr.structure.section_tree import _PART_HEADINGS, IMRAD_ANCHORS

    not_parts = sorted(
        name for name in _PART_HEADINGS if _ALIAS_EXACT_LOOKUP.get(name) not in IMRAD_ANCHORS
    )
    assert not_parts == []


def test_repeat_exact_alias_heading_still_folds_under_the_first():
    secs = [
        _typed_section(1, "Methods", CanonicalSection.METHODS),
        _typed_section(2, "Results", CanonicalSection.RESULTS),
        _typed_section(3, "Methods", CanonicalSection.METHODS),
    ]
    for sec in secs:
        sec.classification_source = "exact_alias"
    assign_hierarchy_from_top_level(secs)
    assert (secs[2].level, secs[2].parent_section_id) == (2, 1)


def test_explicit_is_top_overrides_type_based_rule():
    """is_top_level_predicted=True wins over UNKNOWN-type fold."""
    secs = [
        _typed_section(1, "Method", CanonicalSection.METHODS),
        _section(2, "Surprise Top-Level", CanonicalSection.UNKNOWN, True),
    ]
    assign_hierarchy_from_top_level(secs)
    assert secs[1].level == 1
    assert secs[1].parent_section_id == 0
