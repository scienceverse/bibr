"""Native title evidence survives institution words without inventing variants."""

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from bibr.extract.core_metadata import prefer_first_printed_native_title
from bibr.extract.front_matter import collect_front_matter_candidates, resolve_front_matter
from bibr.extract.metadata_variants import collect_metadata_variants
from bibr.extract.title_source import SOURCE_QUALIFIED_TITLE_ROLE, native_title_evidence
from bibr.paper_contents import (
    CanonicalSection as C,
)
from bibr.paper_contents import (
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)

TITLE = (
    "Seasonal patterns of respiratory infections among animals in a veterinary teaching hospital"
)
ALTERNATE = "Padrões sazonais de infecções respiratórias em animais de um hospital veterinário"


def _source(*, title=TITLE, alternate=ALTERNATE):
    title_box = (40, 50, 550, 90)
    sections = [
        PaperSection(1, title, 1, None, C.TITLE, provenance=[Provenance(1, title_box)]),
        PaperSection(2, "Abstract", 2, 1, C.ABSTRACT, header_is_synthetic=True),
    ]
    texts = [
        (alternate, "text", 1, (40, 110, 550, 150)),
        ("Mara Quill1, Elian Brook2, Talia Vale3", "text", 1, (40, 170, 550, 200)),
        (
            "Abstract: The study measured seasonal infection patterns in animals.",
            "abstract",
            2,
            (40, 230, 550, 350),
        ),
    ]
    sentences = [
        PaperSentence(index, text, sid, index, 1, provenance=[Provenance(1, bbox)])
        for index, (text, _, sid, bbox) in enumerate(texts, 1)
    ]
    regions = [RegionSummary(1, 0, "doc_title", title_box, content=title, section_id=1)]
    regions.extend(
        RegionSummary(1, index, label, bbox, content=text, section_id=sid)
        for index, (text, label, sid, bbox) in enumerate(texts, 1)
    )
    return PaperContents(
        sentences, sections, [], [], {}, detected_title=title, region_summaries=regions
    )


def test_native_title_keeps_title_role_and_can_prefer_first_printed_source():
    contents = _source()
    resolution, _ = resolve_front_matter(contents, target_required=False)
    title_candidate = next(row for row in resolution.candidates if row.region_label == "doc_title")

    assert SOURCE_QUALIFIED_TITLE_ROLE in title_candidate.roles
    assert "affiliation" not in title_candidate.roles
    assert title_candidate.raw_text == TITLE
    assert len(resolution.blocks) == 1
    title, issue = prefer_first_printed_native_title(ALTERNATE, contents, resolution)
    assert title == TITLE
    assert issue.code == "VAL_TITLE_SOURCE_PREFERRED"
    assert not issue.blocking
    # The adjacent row has not been certified as a translation/presentation.
    assert (
        len([v for v in collect_metadata_variants(contents, resolution) if v.field == "title"]) <= 1
    )


def test_shared_parser_paragraph_preserves_separate_source_region_anatomy():
    contents = _source()
    contents.sentences[1].paragraph_id = contents.sentences[0].paragraph_id
    resolution, _ = resolve_front_matter(contents, target_required=False)

    assert prefer_first_printed_native_title(ALTERNATE, contents, resolution)[0] == TITLE


async def test_native_title_source_proof_survives_real_implicit_section_normalization(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.pipeline.stages.post_parse import _normalize_section_structure
    from bibr.schemas import FrontMatterResult, FrontMatterSegment

    contents = _source()
    # This source paragraph has the same affiliation vocabulary as real
    # abstracts. Its conservative front-matter role leaves it for implicit
    # section detection, which reassigns the sentence to a synthetic section.
    abstract = "Abstract: The study measured seasonal infection patterns at a university hospital."
    contents.sentences[2].text = contents.region_summaries[3].content = abstract
    contents.sections.append(PaperSection(3, "Introduction", 1, None, C.INTRODUCTION))
    contents.sentences.append(
        PaperSentence(4, "We measured seasonal changes over several years.", 3, 4, page_number=1)
    )
    resolution, _ = resolve_front_matter(contents, target_required=False)
    contents.front_matter_resolution = resolution
    assert prefer_first_printed_native_title(ALTERNATE, contents, resolution)[0] == TITLE
    original_section = contents.sentences[2].section_id
    detector = AsyncMock(
        return_value=FrontMatterResult(
            segments=[FrontMatterSegment(first_text_id=3, section_type="abstract")]
        )
    )
    monkeypatch.setattr("bibr.structure.implicit_sections._detect_via_llm", detector)

    await _normalize_section_structure(
        contents,
        False,
        object(),
        "synthetic",
        settings=GlobalSettings(IMPLICIT_SECTION_DETECTION=True),
    )

    detector.assert_awaited_once()
    assert contents.sentences[2].section_id != original_section
    assert contents.region_summaries[3].section_id == original_section
    title, issue = prefer_first_printed_native_title(ALTERNATE, contents, resolution)
    assert title == TITLE
    assert issue.code == "VAL_TITLE_SOURCE_PREFERRED"


@pytest.mark.parametrize(
    "fault", ["missing-provenance", "conflicting-provenance", "foreign-owner", "duplicate-owner"]
)
def test_changed_semantic_section_does_not_relax_original_source_ownership(fault):
    contents = _source()
    resolution, _ = resolve_front_matter(contents, target_required=False)
    contents.sentences[2].section_id = 99
    byline = next(row for row in resolution.candidates if 2 in row.text_ids)
    if fault == "missing-provenance":
        contents.sentences[1].provenance = []
    elif fault == "conflicting-provenance":
        contents.sentences[1].provenance[0].bbox = (20, 400, 550, 450)
    elif fault == "foreign-owner":
        resolution = replace(
            resolution,
            candidates=tuple(
                replace(row, section_id=99) if row is byline else row
                for row in resolution.candidates
            ),
        )
    else:
        duplicate = replace(byline, candidate_id="duplicate-owner")
        block = resolution.blocks[0]
        resolution = replace(
            resolution,
            candidates=(*resolution.candidates, duplicate),
            blocks=(replace(block, candidate_ids=(*block.candidate_ids, duplicate.candidate_id)),),
        )

    assert prefer_first_printed_native_title(ALTERNATE, contents, resolution) == (ALTERNATE, None)


def test_source_region_with_two_candidate_owners_is_not_borrowed():
    contents = _source()
    candidates = collect_front_matter_candidates(contents)
    byline = next(row for row in candidates if 2 in row.text_ids)
    duplicate = replace(byline, candidate_id="duplicate-owner")

    assert native_title_evidence(contents, (*candidates, duplicate)) == ()


@pytest.mark.parametrize(
    "fault",
    [
        "hospital-affiliation",
        "fragment",
        "no-byline",
        "no-abstract",
        "foreign-page",
        "foreign-column",
        "unowned-byline",
        "duplicate-title-region",
        "missing-provenance",
        "conflicting-provenance",
    ],
)
def test_unsupported_native_title_anatomy_does_not_override_scalar(fault):
    contents = _source()
    if fault in {"hospital-affiliation", "fragment"}:
        changed = (
            "Department of Veterinary Medicine, Regional Hospital"
            if fault == "hospital-affiliation"
            else TITLE + ":"
        )
        contents.sections[0].header = contents.detected_title = changed
        contents.region_summaries[0].content = changed
    elif fault == "no-byline":
        contents.sentences[1].text = contents.region_summaries[2].content = "Winter, spring, summer"
    elif fault == "no-abstract":
        contents.region_summaries[3].label = "text"
    elif fault == "foreign-page":
        contents.region_summaries[2].page = 2
    elif fault == "foreign-column":
        contents.region_summaries[2].bbox = (600, 170, 900, 200)
    elif fault == "unowned-byline":
        contents.region_summaries[2].section_id = 99
    elif fault == "missing-provenance":
        contents.sentences[1].provenance = []
    elif fault == "conflicting-provenance":
        contents.sentences[1].provenance[0].bbox = (20, 400, 550, 450)
    else:
        contents.region_summaries.append(replace(contents.region_summaries[0]))
    candidates = collect_front_matter_candidates(contents)

    assert not any(SOURCE_QUALIFIED_TITLE_ROLE in row.roles for row in candidates)
    resolution, _ = resolve_front_matter(contents, target_required=False)
    assert prefer_first_printed_native_title(ALTERNATE, contents, resolution) == (ALTERNATE, None)


def test_complete_join_and_explicit_original_marker_keep_model_choice():
    contents = _source()
    resolution, _ = resolve_front_matter(contents, target_required=False)
    joined = f"{TITLE}: {ALTERNATE}"
    assert prefer_first_printed_native_title(joined, contents, resolution) == (joined, None)
    marker = replace(resolution.candidates[-1], raw_text="Original title: " + ALTERNATE)
    marked = replace(resolution, candidates=resolution.candidates[:-1] + (marker,))
    assert prefer_first_printed_native_title(ALTERNATE, contents, marked) == (ALTERNATE, None)


def test_another_title_or_record_cannot_supply_the_source_preference():
    contents = _source()
    resolution, _ = resolve_front_matter(contents, target_required=False)
    other = replace(
        resolution.candidates[-1],
        raw_text="Another separately developed title",
        roles=frozenset({"title"}),
    )
    ambiguous = replace(resolution, candidates=resolution.candidates[:-1] + (other,))
    assert prefer_first_printed_native_title(ALTERNATE, contents, ambiguous) == (ALTERNATE, None)
    abstained = replace(resolution, selected_block_id=None)
    assert prefer_first_printed_native_title(ALTERNATE, contents, abstained) == (ALTERNATE, None)


def test_separate_articles_with_their_own_native_anatomy_remain_separate():
    from bibr.extract.document_scope import scope_document_records

    contents = _source()
    other = _source(
        title=TITLE.replace("respiratory", "intestinal"),
        alternate=ALTERNATE.replace("respiratórias", "intestinais"),
    )
    for section in other.sections:
        section.section_id += 2
        if section.parent_section_id is not None:
            section.parent_section_id += 2
        for point in section.provenance:
            point.page_no = 2
    for sentence in other.sentences:
        sentence.text_id += 3
        sentence.paragraph_id += 3
        sentence.section_id += 2
        sentence.page_number = 2
        for point in sentence.provenance:
            point.page_no = 2
    for region in other.region_summaries:
        region.page = 2
        region.section_id += 2
    contents.sections.extend(other.sections)
    contents.sentences.extend(other.sentences)
    contents.region_summaries.extend(other.region_summaries)
    resolution, _ = resolve_front_matter(contents, target_required=False)

    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id is None
    scopes = scope_document_records(contents, resolution)
    assert all(scope.contents is not None for scope in scopes)
    assert set(scopes[0].source_text_ids).isdisjoint(scopes[1].source_text_ids)
