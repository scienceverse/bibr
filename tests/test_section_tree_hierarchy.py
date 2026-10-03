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


def test_first_heading_with_no_section_before_it_opens_a_top_level_section():
    secs = [_section(1, "Stray", CanonicalSection.UNKNOWN, False)]
    assign_hierarchy_from_top_level(secs)
    assert secs[0].level == 1
    assert secs[0].parent_section_id == 0


def test_numbered_heading_level_is_its_depth():
    """The parser's level (depth + 1 under the title) gives way to the depth."""
    sec = _section(1, "1.1 Encoder", CanonicalSection.UNKNOWN, False)
    sec.level = 3
    sec.parent_section_id = 5
    assign_hierarchy_from_top_level([sec])
    assert sec.level == 2
    assert sec.parent_section_id == 0


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


def test_unknown_before_any_anchor_opens_a_top_level_section():
    """An UNKNOWN section with no section before it is level 1, not folded."""
    secs = [
        _typed_section(1, "Mystery Heading", CanonicalSection.UNKNOWN),
        _typed_section(2, "Method", CanonicalSection.METHODS),
    ]
    assign_hierarchy_from_top_level(secs)
    assert (secs[0].level, secs[0].parent_section_id) == (1, 0)
    assert (secs[1].level, secs[1].parent_section_id) == (1, 0)


def test_interlude_does_not_reset_anchor():
    """OPEN_DATA between Method and an UNKNOWN should NOT capture the UNKNOWN."""
    secs = [
        _typed_section(1, "Method", CanonicalSection.METHODS),
        _typed_section(2, "Open Practices", CanonicalSection.OPEN_DATA),
        _typed_section(3, "Apparatus", CanonicalSection.UNKNOWN),
    ]
    secs[1].classification_source = "substring_alias"
    assign_hierarchy_from_top_level(secs)
    # OPEN_DATA stays as level=1 interlude.
    assert secs[1].level == 1
    assert secs[1].parent_section_id == 0
    # UNKNOWN folds into the IMRaD anchor (Method), not the interlude.
    assert secs[2].level == 2
    assert secs[2].parent_section_id == 1


def test_interlude_promoted_to_top_level():
    """Alias-typed COI/funding/etc. become level=1 (top-level)."""
    secs = [
        _typed_section(1, "Funding", CanonicalSection.FUNDING),
        _typed_section(2, "Conflict of Interest", CanonicalSection.COI),
    ]
    for sec in secs:
        sec.classification_source = "exact_alias"
    assign_hierarchy_from_top_level(secs)
    assert secs[0].level == 1
    assert secs[1].level == 1


def test_guessed_back_matter_type_does_not_lift_a_body_heading():
    """A model guess of a back-matter type for a heading printed inside
    Methods keeps it a subsection of Methods."""
    secs = [
        _typed_section(1, "Methods", CanonicalSection.METHODS),
        _typed_section(2, "Ethical considerations of the design", CanonicalSection.ETHICS),
        _typed_section(3, "History", CanonicalSection.ABSTRACT),
    ]
    secs[0].classification_source = "exact_alias"
    secs[1].classification_source = "model"
    secs[2].classification_source = "llm"
    assign_hierarchy_from_top_level(secs)
    assert [(s.level, s.parent_section_id) for s in secs] == [(1, 0), (2, 1), (2, 1)]


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
    METHODS) is a subsection of the Methods part it is printed in."""
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
        # A keyword heading stays in the part where it is printed.
        (2, 2, 1),
        (3, 1, 0),
        (4, 1, 0),
        (5, 2, 4),
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
        (2, 2, 1),
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


def test_repeated_part_name_opens_a_new_part():
    """A part name printed again (a second study's "Methods") opens a level-1
    section where it is printed instead of folding back under the first."""
    secs = [
        _typed_section(1, "Methods", CanonicalSection.METHODS),
        _typed_section(2, "Results", CanonicalSection.RESULTS),
        _typed_section(3, "Methods", CanonicalSection.METHODS),
        _typed_section(4, "Procedure", CanonicalSection.UNKNOWN),
    ]
    for sec in secs:
        sec.classification_source = "exact_alias"
    assign_hierarchy_from_top_level(secs)
    assert [(s.level, s.parent_section_id) for s in secs] == [(1, 0), (1, 0), (1, 0), (2, 3)]


def test_explicit_is_top_overrides_type_based_rule():
    """is_top_level_predicted=True wins over UNKNOWN-type fold."""
    secs = [
        _typed_section(1, "Method", CanonicalSection.METHODS),
        _section(2, "Surprise Top-Level", CanonicalSection.UNKNOWN, True),
    ]
    assign_hierarchy_from_top_level(secs)
    assert secs[1].level == 1
    assert secs[1].parent_section_id == 0


# ---------------------------------------------------------------------------
# Numbering and document order (modelled on tester papers, #119)
# ---------------------------------------------------------------------------


def _parsed(sid: int, header: str, type_=CanonicalSection.UNKNOWN, source=None, level=2):
    """A section as the parser leaves it: numbered headings at depth + 1 under
    the title, everything else at its layout level."""
    sec = PaperSection(section_id=sid, header=header, level=level, parent_section_id=1)
    sec.section_type = type_
    sec.classification_source = source
    return sec


def _title(sid: int = 1) -> PaperSection:
    sec = PaperSection(section_id=sid, header="A Paper Title", level=1, parent_section_id=0)
    sec.section_type = CanonicalSection.TITLE
    sec.classification_source = "title"
    return sec


def _tree(secs):
    return [(s.section_id, s.level, s.parent_section_id) for s in secs]


def test_numbered_preprint_sections_take_their_depth_and_prefix_parent():
    """A preprint numbers its parts 1-4 and their subsections 3.1-4.2: the
    parser put every part at level 2 under the title."""
    secs = [
        _title(),
        _parsed(2, "Abstract", CanonicalSection.ABSTRACT, "exact_alias"),
        _parsed(3, "1 Introduction", CanonicalSection.INTRODUCTION, "exact_alias"),
        _parsed(4, "2 Materials and methods", CanonicalSection.METHODS, "exact_alias"),
        _parsed(5, "2.1 Synthesis", CanonicalSection.METHODS, "model", level=3),
        _parsed(6, "3 Results", CanonicalSection.RESULTS, "exact_alias"),
        _parsed(7, "3.1 Morphology", CanonicalSection.RESULTS, "llm", level=3),
        _parsed(8, "3.2 Structural characterization", CanonicalSection.METHODS, "llm", level=3),
        _parsed(9, "Statistical note", CanonicalSection.UNKNOWN, None),
        _parsed(10, "4 Discussion", CanonicalSection.DISCUSSION, "exact_alias"),
        _parsed(11, "4.1 Mechanism", CanonicalSection.RESULTS, "llm", level=3),
        _parsed(12, "Declarations", CanonicalSection.UNKNOWN, None),
        _parsed(13, "Funding", CanonicalSection.FUNDING, "exact_alias"),
        _parsed(14, "References", CanonicalSection.REFERENCES, "exact_alias"),
    ]
    assign_hierarchy_from_top_level(secs)
    assert _tree(secs) == [
        (1, 1, 0),
        (2, 1, 0),
        (3, 1, 0),
        (4, 1, 0),
        (5, 2, 4),
        (6, 1, 0),
        (7, 2, 6),
        (8, 2, 6),
        # An unnumbered heading after 3.2 is its subsection.
        (9, 3, 8),
        (10, 1, 0),
        (11, 2, 10),
        (12, 1, 0),
        (13, 1, 0),
        (14, 1, 0),
    ]


def test_roman_numbered_parts_parent_their_arabic_and_lettered_subsections():
    """An IEEE-style paper: Roman parts, "4.2" under "IV", lettered and
    "ii)" items under the section they are printed in."""
    secs = [
        _title(),
        _parsed(2, "I INTRODUCTION", CanonicalSection.INTRODUCTION, "substring_alias"),
        _parsed(3, "II RELATED WORK", CanonicalSection.INTRODUCTION, "substring_alias"),
        _parsed(4, "III EXISTING SYSTEM", CanonicalSection.METHODS, "llm"),
        _parsed(5, "IV PROPOSED METHOD", CanonicalSection.METHODS, "substring_alias"),
        _parsed(6, "4.1 Haze model", CanonicalSection.METHODS, "llm", level=3),
        _parsed(7, "i) Dark channel", CanonicalSection.METHODS, "llm"),
        _parsed(8, "ii) Fading Section", CanonicalSection.UNKNOWN, None),
        _parsed(9, "4.2 CONTRACT IN HRF", CanonicalSection.METHODS, "llm", level=3),
        _parsed(10, "V MODULE DESCRIPTION", CanonicalSection.METHODS, "llm"),
        _parsed(11, "A. Input module", CanonicalSection.METHODS, "llm"),
        _parsed(12, "VI CONCLUSION", CanonicalSection.DISCUSSION, "substring_alias"),
        _parsed(13, "REFERENCES", CanonicalSection.REFERENCES, "exact_alias"),
    ]
    assign_hierarchy_from_top_level(secs)
    assert _tree(secs) == [
        (1, 1, 0),
        (2, 1, 0),
        (3, 1, 0),
        (4, 1, 0),
        (5, 1, 0),
        (6, 2, 5),
        (7, 3, 6),
        (8, 3, 6),
        (9, 2, 5),
        (10, 1, 0),
        (11, 2, 10),
        (12, 1, 0),
        (13, 1, 0),
    ]


def test_a_lone_roman_looking_word_and_large_integers_are_not_numbers():
    secs = [
        _parsed(1, "Introduction", CanonicalSection.INTRODUCTION, "exact_alias"),
        _parsed(2, "V Model Design", CanonicalSection.UNKNOWN, None),
        _parsed(3, "75 Years of Screening", CanonicalSection.UNKNOWN, None),
        _parsed(4, "Methods", CanonicalSection.METHODS, "exact_alias"),
    ]
    assign_hierarchy_from_top_level(secs)
    assert _tree(secs) == [(1, 1, 0), (2, 2, 1), (3, 2, 1), (4, 1, 0)]


def test_chapters_number_a_thesis_and_contain_their_part_names():
    """A dissertation: "Chapter N" is level 1 and the "Introduction" or
    "Methodology" printed inside a chapter is its subsection."""
    secs = [
        _title(),
        _parsed(2, "Table of Contents", CanonicalSection.RESULTS, "model"),
        _parsed(3, "Chapter 1: Introduction to the Study", CanonicalSection.INTRODUCTION, "llm"),
        _parsed(4, "Problem Statement", CanonicalSection.INTRODUCTION, "model"),
        _parsed(5, "Nature of the Study", CanonicalSection.METHODS, "model"),
        _parsed(
            6, "Chapter 2: Literature Review", CanonicalSection.INTRODUCTION, "substring_alias"
        ),
        _parsed(7, "Introduction", CanonicalSection.INTRODUCTION, "exact_alias"),
        _parsed(8, "Chapter 3: Research Method", CanonicalSection.METHODS, "substring_alias"),
        _parsed(9, "Methodology", CanonicalSection.METHODS, "exact_alias"),
        _parsed(10, "Chapter 4: Results", CanonicalSection.RESULTS, "substring_alias"),
        _parsed(11, "Data Collection", CanonicalSection.METHODS, "exact_alias"),
        _parsed(12, "References", CanonicalSection.REFERENCES, "exact_alias"),
    ]
    assign_hierarchy_from_top_level(secs)
    assert _tree(secs) == [
        (1, 1, 0),
        (2, 1, 0),
        (3, 1, 0),
        (4, 2, 3),
        (5, 2, 3),
        (6, 1, 0),
        (7, 2, 6),
        (8, 1, 0),
        (9, 2, 8),
        (10, 1, 0),
        (11, 2, 10),
        (12, 1, 0),
    ]


def test_all_caps_headings_are_parts_when_the_part_names_are_all_caps():
    """A review prints its parts in capitals and repeats "Overview" in each;
    every Overview belongs to the part it is printed in."""
    secs = [
        _title(),
        _parsed(2, "INTRODUCTION", CanonicalSection.INTRODUCTION, "exact_alias"),
        _parsed(3, "HISTORY", CanonicalSection.ABSTRACT, "model"),
        _parsed(4, "Overview", CanonicalSection.INTRODUCTION, "exact_alias"),
        _parsed(5, "CURATIVE EMBOLIZATION OF DEEP LESIONS", CanonicalSection.TITLE, "model"),
        _parsed(6, "Overview", CanonicalSection.INTRODUCTION, "exact_alias"),
        _parsed(7, "Technique", CanonicalSection.METHODS, "llm"),
        _parsed(8, "CONCLUSIONS", CanonicalSection.DISCUSSION, "exact_alias"),
        _parsed(9, "REFERENCES", CanonicalSection.REFERENCES, "exact_alias"),
    ]
    assign_hierarchy_from_top_level(secs)
    assert _tree(secs) == [
        (1, 1, 0),
        (2, 1, 0),
        (3, 1, 0),
        (4, 2, 3),
        (5, 1, 0),
        (6, 2, 5),
        (7, 2, 5),
        (8, 1, 0),
        (9, 1, 0),
    ]


def test_back_matter_printed_inside_methods_does_not_take_its_subsections():
    secs = [
        _parsed(1, "Methods", CanonicalSection.METHODS, "exact_alias"),
        _parsed(2, "Participants", CanonicalSection.METHODS, "exact_alias"),
        _parsed(3, "Data sharing", CanonicalSection.OPEN_DATA, "exact_alias"),
        _parsed(4, "Statistical analysis", CanonicalSection.METHODS, "exact_alias"),
        _parsed(5, "Running title: Short", CanonicalSection.UNKNOWN, None),
        _parsed(6, "Results", CanonicalSection.RESULTS, "exact_alias"),
    ]
    assign_hierarchy_from_top_level(secs)
    assert _tree(secs) == [(1, 1, 0), (2, 2, 1), (3, 1, 0), (4, 2, 1), (5, 1, 0), (6, 1, 0)]


def test_first_heading_of_a_type_is_a_part_only_when_no_part_name_exists():
    """With "Introduction" printed but no Methods or Results part names, the
    first method- and results-typed headings open the parts."""
    secs = [
        _parsed(1, "Introduction", CanonicalSection.INTRODUCTION, "exact_alias"),
        _parsed(2, "Prior accounts", CanonicalSection.INTRODUCTION, "model"),
        _parsed(3, "Participants", CanonicalSection.METHODS, "exact_alias"),
        _parsed(4, "Procedure", CanonicalSection.UNKNOWN, None),
        _parsed(5, "Accuracy improved with practice", CanonicalSection.RESULTS, "model"),
        _parsed(6, "Speed", CanonicalSection.RESULTS, "model"),
    ]
    assign_hierarchy_from_top_level(secs)
    assert _tree(secs) == [(1, 1, 0), (2, 2, 1), (3, 1, 0), (4, 2, 3), (5, 1, 0), (6, 2, 5)]


def test_cover_sheet_labels_sit_at_level_one_and_contain_nothing():
    """A preprint cover page: the label rows are front matter, so the
    affiliation line after them opens its own section instead of nesting."""
    secs = [
        _parsed(1, "Research Article", CanonicalSection.UNKNOWN, None),
        _parsed(2, "Posted Date: September 29th, 2026", CanonicalSection.UNKNOWN, None),
        _parsed(3, "Corresponding author", CanonicalSection.UNKNOWN, None),
        _parsed(4, "Universidade Federal", CanonicalSection.UNKNOWN, None),
        _parsed(5, "Introduction", CanonicalSection.INTRODUCTION, "exact_alias"),
    ]
    assign_hierarchy_from_top_level(secs)
    assert _tree(secs) == [(1, 1, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0), (5, 1, 0)]
