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


def test_an_exact_subsection_alias_also_outranks_a_keyword_part():
    """Known limitation: the exact alias table also holds subsection names.
    "Statistical analysis" (exact METHODS) outranks "Methods and materials"
    (a keyword hit), so the Results subsections after it fold under it and
    read as METHODS. Restricting the rule to part names such as "Methods"
    lost more than it saved on the PMC sample ("Experimental Section")."""
    secs = [
        _typed_section(1, "Methods and materials", CanonicalSection.METHODS),
        _typed_section(2, "Results", CanonicalSection.RESULTS),
        _typed_section(3, "Statistical analysis", CanonicalSection.METHODS),
        _typed_section(4, "Dose comparison", CanonicalSection.UNKNOWN),
    ]
    for sec, source in zip(
        secs, ["substring_alias", "exact_alias", "exact_alias", None], strict=True
    ):
        sec.classification_source = source
    assign_hierarchy_from_top_level(secs)
    assert [(s.section_id, s.level, s.parent_section_id) for s in secs] == [
        (1, 1, 0),
        (2, 1, 0),
        (3, 1, 0),
        (4, 2, 3),
    ]


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
