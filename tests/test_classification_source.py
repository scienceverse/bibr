"""classification_source provenance threading through post-parse (chunk 4b).

Exercises the no-LLM classification path end to end: the title tag, exact-alias
hits, and the appendix-repair pass must each stamp their source onto the
section so the JSON export can surface it.
"""

from __future__ import annotations

from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.pipeline.stages.post_parse import (
    _classify_sections,
    _reconcile_result_subsection_types,
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


def test_weak_method_prediction_under_results_uses_parent_context():
    parent = PaperSection(
        section_id=9,
        header="Results",
        level=1,
        parent_section_id=0,
        section_type=CanonicalSection.RESULTS,
        classification_score=1.0,
        classification_source="exact_alias",
    )
    child = PaperSection(
        section_id=11,
        header="Professional view on engagement",
        level=2,
        parent_section_id=9,
        section_type=CanonicalSection.METHODS,
        classification_score=0.704,
        classification_source="model",
    )

    _reconcile_result_subsection_types([parent, child])

    assert child.section_type == CanonicalSection.RESULTS
    assert child.classification_score == 0.8
    assert child.classification_source == "parent_context"


def test_strong_method_prediction_under_results_is_preserved():
    parent = PaperSection(
        section_id=9,
        header="Results",
        level=1,
        parent_section_id=0,
        section_type=CanonicalSection.RESULTS,
    )
    child = PaperSection(
        section_id=11,
        header="Engagement measurement method",
        level=2,
        parent_section_id=9,
        section_type=CanonicalSection.METHODS,
        classification_score=0.91,
        classification_source="model",
    )

    _reconcile_result_subsection_types([parent, child])

    assert child.section_type == CanonicalSection.METHODS
    assert child.classification_source == "model"
