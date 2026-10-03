"""classification_source provenance threading through post-parse (chunk 4b).

Exercises the no-LLM classification path end to end: the title tag, exact-alias
hits, and the appendix-repair pass must each stamp their source onto the
section so the JSON export can surface it.
"""

from __future__ import annotations

from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.pipeline.stages.post_parse import (
    _classify_sections,
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
