"""classification_source provenance threading through post-parse (chunk 4b).

Exercises the no-LLM classification path end to end: the title tag, exact-alias
hits, and the appendix-repair pass must each stamp their source onto the
section so the JSON export can surface it.
"""

from __future__ import annotations

from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.pipeline.stages.post_parse import (
    _classify_sections,
    _gate_non_imrad_section_types,
    _inherit_child_section_types,
)


def _sec(sid: int, header: str, level: int = 1) -> PaperSection:
    return PaperSection(section_id=sid, header=header, level=level, parent_section_id=0)


def _contents(sections: list[PaperSection], title: str) -> PaperContents:
    contents = PaperContents(
        sentences=[],
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
    )
    contents.detected_title = title
    return contents


async def test_no_llm_classification_sources_stamped():
    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        _sec(1, "My Great Paper"),
        _sec(2, "Methods"),
        _sec(3, "References"),
        _sec(4, "A Additional Results"),
        _sec(5, "B Proofs"),
    ]
    contents = _contents(sections, title="My Great Paper")

    await _classify_sections(contents, [], no_llm=True, llm_client=None)

    by = {s.section_id: s for s in contents.sections}
    assert by[1].section_type == CanonicalSection.TITLE
    assert by[1].classification_source == "title"
    assert by[2].section_type == CanonicalSection.METHODS
    assert by[2].classification_source == "exact_alias"
    assert by[3].section_type == CanonicalSection.REFERENCES
    assert by[3].classification_source == "exact_alias"
    # A/B appendices re-typed and stamped by the repair pass.
    assert by[4].section_type == CanonicalSection.APPENDIX
    assert by[4].classification_source == "appendix_repair"
    assert by[5].section_type == CanonicalSection.APPENDIX
    assert by[5].classification_source == "appendix_repair"


async def test_no_llm_miss_leaves_source_none():
    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        _sec(1, "Mysterious Nonalias Heading"),
    ]
    contents = _contents(sections, title="Some Title")
    await _classify_sections(contents, [], no_llm=True, llm_client=None)
    by = {s.section_id: s for s in contents.sections}
    assert by[1].section_type == CanonicalSection.UNKNOWN
    assert by[1].classification_source is None


async def test_unknown_child_inherits_body_parent_type_after_hierarchy():
    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        _sec(1, "Methods"),
        _sec(2, "Stimulus Calibration"),
    ]
    contents = _contents(sections, title="Some Title")

    await _classify_sections(contents, [], no_llm=True, llm_client=None)

    by = {s.section_id: s for s in contents.sections}
    assert by[2].parent_section_id == 1
    assert by[2].section_type == CanonicalSection.METHODS
    assert by[2].classification_source == "parent_context"


async def test_parent_context_inheritance_preserves_explicit_child_type():
    parent = _sec(1, "Methods")
    child = _sec(2, "Results", level=2)
    child.parent_section_id = 1
    child.outline_level_authoritative = True
    contents = _contents(
        [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            parent,
            child,
        ],
        title="Some Title",
    )

    await _classify_sections(contents, [], no_llm=True, llm_client=None)

    by = {s.section_id: s for s in contents.sections}
    assert by[2].parent_section_id == 1
    assert by[2].section_type == CanonicalSection.RESULTS
    assert by[2].classification_source == "exact_alias"


def _typed(sid, header, type_, source, parent=0, level=1, score=0.9):
    return PaperSection(
        section_id=sid,
        header=header,
        level=level,
        parent_section_id=parent,
        section_type=type_,
        classification_score=score,
        classification_source=source,
    )


def test_guessed_child_types_follow_a_part_heading_parent():
    secs = [
        _typed(9, "Results", CanonicalSection.RESULTS, "exact_alias", score=1.0),
        _typed(10, "Professional view on engagement", CanonicalSection.METHODS, "model", 9, 2),
        _typed(11, "Engagement measurement method", CanonicalSection.METHODS, "llm", 9, 2),
        _typed(12, "Statistical analysis", CanonicalSection.METHODS, "exact_alias", 9, 2),
        _typed(13, "Further notes", CanonicalSection.UNKNOWN, None, 9, 2, score=0.0),
    ]
    _inherit_child_section_types(secs)
    assert [(s.section_type, s.classification_source) for s in secs[1:]] == [
        (CanonicalSection.RESULTS, "parent_context"),
        (CanonicalSection.RESULTS, "parent_context"),
        # The heading names its own type.
        (CanonicalSection.METHODS, "exact_alias"),
        (CanonicalSection.RESULTS, "parent_context"),
    ]
    assert secs[1].classification_score == 0.75


def test_numbered_children_follow_their_numbered_part():
    """A preprint's "3.2" and "4.1" subsections, typed alone by the LLM."""
    secs = [
        _typed(3, "3 Results", CanonicalSection.RESULTS, "substring_alias"),
        _typed(4, "3.2 Structural characterization", CanonicalSection.METHODS, "llm", 3, 2),
        _typed(5, "4 Discussion", CanonicalSection.DISCUSSION, "substring_alias"),
        _typed(6, "4.1 Adsorption mechanism", CanonicalSection.RESULTS, "llm", 5, 2),
        _typed(7, "4.2 Comparison", CanonicalSection.METHODS, "llm", 5, 2),
    ]
    _inherit_child_section_types(secs)
    assert [s.section_type for s in secs] == [
        CanonicalSection.RESULTS,
        CanonicalSection.RESULTS,
        CanonicalSection.DISCUSSION,
        CanonicalSection.DISCUSSION,
        CanonicalSection.DISCUSSION,
    ]


def test_a_keyword_heading_parent_does_not_retype_its_children():
    secs = [
        _typed(1, "Overview of the findings", CanonicalSection.RESULTS, "model"),
        _typed(2, "Sampling frame", CanonicalSection.METHODS, "model", 1, 2),
    ]
    _inherit_child_section_types(secs)
    assert secs[1].section_type == CanonicalSection.METHODS


def test_guessed_title_on_a_body_heading_becomes_its_parts_type():
    secs = [
        _typed(1, "A Review of Treatment", CanonicalSection.TITLE, "title"),
        _typed(2, "Discussion", CanonicalSection.DISCUSSION, "exact_alias"),
        _typed(3, "CURATIVE EMBOLIZATION", CanonicalSection.TITLE, "model", 2, 2),
        _typed(4, "Walden University", CanonicalSection.TITLE, "model"),
    ]
    _gate_non_imrad_section_types(secs, "empirical", review_body=True)
    assert [(s.section_type, s.classification_source) for s in secs] == [
        (CanonicalSection.TITLE, "title"),
        (CanonicalSection.DISCUSSION, "exact_alias"),
        (CanonicalSection.DISCUSSION, "parent_context"),
        (CanonicalSection.UNKNOWN, "model"),
    ]


def _review_sections():
    return [
        _typed(1, "Introduction", CanonicalSection.INTRODUCTION, "exact_alias"),
        _typed(2, "Scope of this review", CanonicalSection.INTRODUCTION, "model", 1, 2),
        _typed(3, "Definitions", CanonicalSection.INTRODUCTION, "parent_context", 1, 2),
        _typed(4, "Embolization in practice", CanonicalSection.INTRODUCTION, "llm"),
        _typed(5, "Future directions", CanonicalSection.DISCUSSION, "exact_alias"),
        _typed(6, "State of the field", CanonicalSection.INTRODUCTION, "model"),
    ]


def test_review_body_guesses_become_discussion():
    """Introduction guesses after the introduction become discussion. A
    subsection the model itself read as introduction stays one; a heading
    that only took the Introduction's type from its parent is body."""
    secs = _review_sections()
    _gate_non_imrad_section_types(secs, "review", review_body=True)
    assert [(s.section_type, s.classification_source) for s in secs] == [
        (CanonicalSection.INTRODUCTION, "exact_alias"),
        (CanonicalSection.INTRODUCTION, "model"),
        (CanonicalSection.DISCUSSION, "positional"),
        (CanonicalSection.DISCUSSION, "positional"),
        (CanonicalSection.DISCUSSION, "exact_alias"),
        (CanonicalSection.DISCUSSION, "positional"),
    ]


def test_review_body_gate_needs_the_setting_the_paper_type_and_no_methods_heading():
    for paper_type, review_body in (("review", False), ("empirical", True), (None, True)):
        secs = _review_sections()
        _gate_non_imrad_section_types(secs, paper_type, review_body=review_body)
        assert secs[3].section_type == CanonicalSection.INTRODUCTION
    # Any methods or results heading, printed or guessed, means the paper
    # reports a study: "Data" / "Empirical strategy" / "Estimates".
    for source in ("exact_alias", "model", "llm"):
        secs = _review_sections()
        secs.append(_typed(7, "Empirical strategy", CanonicalSection.METHODS, source))
        _gate_non_imrad_section_types(secs, "commentary", review_body=True)
        assert secs[3].section_type == CanonicalSection.INTRODUCTION
    # Systematic and scoping reviews and meta-analyses report a search.
    for text in (
        "Bleeding risk: a systematic review",
        "A scoping review of embolization outcomes",
        "We ran a meta-analysis of 40 trials.",
        "An umbrella review",
    ):
        secs = _review_sections()
        _gate_non_imrad_section_types(secs, "review", review_body=True, title_abstract=text)
        assert secs[3].section_type == CanonicalSection.INTRODUCTION
    # Case studies report methods and results like a research paper.
    secs = _review_sections()
    _gate_non_imrad_section_types(secs, "case-study", review_body=True)
    assert secs[3].section_type == CanonicalSection.INTRODUCTION


def test_review_headings_naming_their_part_keep_it_with_their_subsections():
    secs = [
        _typed(1, "Introduction", CanonicalSection.INTRODUCTION, "exact_alias"),
        _typed(2, "Background of the debate", CanonicalSection.INTRODUCTION, "model"),
        _typed(3, "Historical Background", CanonicalSection.INTRODUCTION, "alias_prior"),
        _typed(4, "Early accounts", CanonicalSection.INTRODUCTION, "parent_context", 3, 2),
        _typed(5, "4. Literature review", CanonicalSection.INTRODUCTION, "exact_alias"),
        _typed(6, "4.1 Supply chains", CanonicalSection.INTRODUCTION, "model", 5, 2),
        _typed(7, "Remaining gaps", CanonicalSection.INTRODUCTION, "model"),
    ]
    _gate_non_imrad_section_types(secs, "review", review_body=True)
    assert [s.section_type for s in secs] == [
        CanonicalSection.INTRODUCTION,
        CanonicalSection.INTRODUCTION,
        CanonicalSection.INTRODUCTION,
        CanonicalSection.INTRODUCTION,
        CanonicalSection.INTRODUCTION,
        CanonicalSection.INTRODUCTION,
        CanonicalSection.DISCUSSION,
    ]


def test_a_results_or_discussion_guess_is_not_pulled_back_into_methods():
    """A lost "3 Results" heading: 3.1/3.2 sit after "2 Methods" but keep
    their results guess; a methods guess under "4 Results" still follows it,
    and between results and discussion the part decides."""
    secs = [
        _typed(1, "2 Methods", CanonicalSection.METHODS, "exact_alias"),
        _typed(2, "2.1 Data", CanonicalSection.METHODS, "model", 1, 2),
        _typed(3, "3.1 Effect of treatment on recovery", CanonicalSection.RESULTS, "model", 1, 2),
        _typed(4, "3.2 Subgroup analyses", CanonicalSection.RESULTS, "model", 1, 2),
        _typed(5, "4 Results", CanonicalSection.RESULTS, "exact_alias"),
        _typed(6, "4.1 Structural characterization", CanonicalSection.METHODS, "model", 5, 2),
        _typed(7, "4.2 What the trend implies", CanonicalSection.DISCUSSION, "model", 5, 2),
        _typed(8, "5 Discussion", CanonicalSection.DISCUSSION, "exact_alias"),
        _typed(9, "5.1 Principal findings", CanonicalSection.RESULTS, "model", 8, 2),
        _typed(10, "Methods", CanonicalSection.METHODS, "exact_alias"),
        _typed(11, "Effect of treatment", CanonicalSection.RESULTS, "model", 10, 2),
    ]
    _inherit_child_section_types(secs)
    assert [s.section_type for s in secs[1:]] == [
        CanonicalSection.METHODS,
        CanonicalSection.RESULTS,
        CanonicalSection.RESULTS,
        CanonicalSection.RESULTS,
        CanonicalSection.RESULTS,
        CanonicalSection.RESULTS,
        CanonicalSection.DISCUSSION,
        CanonicalSection.DISCUSSION,
        CanonicalSection.METHODS,
        CanonicalSection.RESULTS,
    ]


def test_an_untyped_child_takes_any_imrad_parents_type():
    """The model typed the part from a keyword; its untyped subsections follow
    it, while a model guess for a subsection needs a confirmed part."""
    secs = [
        _typed(1, "Engaging the panel in the design work", CanonicalSection.METHODS, "alias_prior"),
        _typed(2, "Recruitment of the panel", CanonicalSection.UNKNOWN, None, 1, 2, score=0.0),
        _typed(3, "Overall experience", CanonicalSection.RESULTS, "model", 1, 2),
    ]
    _inherit_child_section_types(secs)
    assert [(s.section_type, s.classification_source) for s in secs[1:]] == [
        (CanonicalSection.METHODS, "parent_context"),
        (CanonicalSection.RESULTS, "model"),
    ]


def test_an_untyped_child_of_an_untyped_scope_follows_the_sibling_before_it():
    """Under "Study 1" (untyped), the model left some method subsections
    untyped; each takes the IMRaD type of the sibling printed just before it.
    The first child, and a child after an untyped sibling, stay untyped."""
    secs = [
        _typed(1, "Study 1: a pilot study", CanonicalSection.UNKNOWN, None, score=0.0),
        _typed(2, "Overview", CanonicalSection.UNKNOWN, None, 1, 2, score=0.0),
        _typed(3, "Participants", CanonicalSection.METHODS, "exact_alias", 1, 2),
        _typed(4, "Virtual reality scenario", CanonicalSection.UNKNOWN, None, 1, 2, score=0.0),
        _typed(5, "Interview", CanonicalSection.UNKNOWN, None, 1, 2, score=0.0),
        _typed(6, "Research in context", CanonicalSection.UNKNOWN, "exact_alias", 1, 2),
        _typed(7, "Evidence before this study", CanonicalSection.UNKNOWN, None, 1, 2, score=0.0),
    ]
    _inherit_child_section_types(secs)
    assert [(s.section_type, s.classification_source) for s in secs[1:]] == [
        (CanonicalSection.UNKNOWN, None),
        (CanonicalSection.METHODS, "exact_alias"),
        (CanonicalSection.METHODS, "parent_context"),
        (CanonicalSection.METHODS, "parent_context"),
        (CanonicalSection.UNKNOWN, "exact_alias"),
        (CanonicalSection.UNKNOWN, None),
    ]
    # An introduction guess is not followed: "Overview" in a later part is
    # not introduction.
    secs = [
        _typed(1, "4. Proposed scheme", CanonicalSection.UNKNOWN, None, score=0.0),
        _typed(2, "4.1. Overview", CanonicalSection.INTRODUCTION, "model", 1, 2),
        _typed(3, "4.2. Processing of requests", CanonicalSection.UNKNOWN, None, 1, 2, score=0.0),
    ]
    _inherit_child_section_types(secs)
    assert secs[2].section_type == CanonicalSection.UNKNOWN
    # Top-level headings are not siblings in a scope: the level-0 root is not
    # a parent to follow.
    secs = [
        _typed(0, "", None, None, level=0, score=0.0),
        _typed(1, "Methods", CanonicalSection.METHODS, "exact_alias"),
        _typed(2, "Conceptual revisions", CanonicalSection.UNKNOWN, None, score=0.0),
    ]
    _inherit_child_section_types(secs)
    assert secs[2].section_type == CanonicalSection.UNKNOWN
