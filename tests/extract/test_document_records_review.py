"""Independent identity and source-geometry review of document record recovery."""

from dataclasses import replace

import pytest

from bibr.extract.document_records import refine_document_records
from bibr.paper_contents import (
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)
from tests.extract.test_document_records import _records
from tests.extract.test_document_scope import _candidate, _resolution


def test_multiple_printed_dois_in_one_presentation_veto_contextual_merge():
    original = _records()
    candidates = list(original.candidates)
    candidates[3] = replace(
        candidates[3],
        raw_text=candidates[0].raw_text,
        normalized_text=candidates[0].normalized_text,
    )
    candidates.insert(
        2,
        _candidate(20, "DOI: 10.9999/first; DOI: 10.9999/second", "doi", 1, text_id=20),
    )
    candidates = [replace(row, reading_order=index) for index, row in enumerate(candidates)]

    refined = refine_document_records(_resolution(candidates))

    assert len(refined.blocks) == 2
    assert not any(block.merge_reasons for block in refined.blocks)


@pytest.mark.parametrize("separator", [" and ", ", ", " & "])
def test_pure_and_composite_printed_author_lists_share_identity(separator):
    original = _records("Mara Quill, Elian Brook, Forest University, Hill City, Country")
    candidates = list(original.candidates)
    candidates[3] = replace(
        candidates[3],
        raw_text=candidates[0].raw_text,
        normalized_text=candidates[0].normalized_text,
    )
    pure = f"Mara Quill{separator}Elian Brook"
    candidates[1] = replace(
        candidates[1], raw_text=pure, normalized_text=pure.casefold(), roles=frozenset({"byline"})
    )

    refined = refine_document_records(_resolution(candidates))

    assert len(refined.blocks) == 1
    assert "shared_document_contextual_identity" in refined.blocks[0].merge_reasons
    assert [row.raw_text for row in refined.candidates] == [row.raw_text for row in candidates]


def _column_spanning_records():
    original = _records(composite=True)
    candidates = list(original.candidates)
    candidates[3] = replace(candidates[3], bbox=(50, 20, 950, 370))
    resolution = _resolution(candidates)
    sections = [
        PaperSection(index + 1, candidates[index * 2].raw_text, 1, None) for index in range(2)
    ]
    sentences = [
        PaperSentence(row.text_ids[0], row.raw_text, row.section_id, row.paragraph_id, row.page)
        for row in (candidates[1], candidates[3])
    ]
    regions = []
    for index, row, bbox in [
        (0, sections[0], candidates[0].bbox),
        (1, sentences[0], candidates[1].bbox),
        (3, sections[1], candidates[2].bbox),
        (4, sentences[1], (50, 340, 450, 370)),
        (5, sentences[1], (500, 20, 950, 160)),
    ]:
        row.provenance.append(Provenance(page_no=1, bbox=bbox))
        regions.append(RegionSummary(1, index, "text", bbox, section_id=row.section_id))
    contents = PaperContents(sentences, sections, [], [], {}, region_summaries=regions)
    return contents, resolution


def test_first_source_region_locates_column_spanning_byline_without_rewriting_source():
    contents, resolution = _column_spanning_records()
    assert refine_document_records(resolution) is resolution

    refined = refine_document_records(resolution, contents=contents)

    assert len(refined.blocks) == 2
    assert "document_contextual_byline" in refined.candidates[3].roles
    assert [row.raw_text for row in refined.candidates] == [
        row.raw_text for row in resolution.candidates
    ]
    assert [row.bbox for row in refined.candidates] == [row.bbox for row in resolution.candidates]
    assert [row.text_ids for row in refined.candidates] == [
        row.text_ids for row in resolution.candidates
    ]
    assert len(contents.sentences[1].provenance) == 2


@pytest.mark.parametrize(
    "fault", ["missing-region", "ambiguous-region", "reversed-order", "foreign-page"]
)
def test_unproven_first_region_does_not_repair_a_column_spanning_byline(fault):
    contents, resolution = _column_spanning_records()
    if fault == "missing-region":
        contents.region_summaries = [row for row in contents.region_summaries if row.index != 4]
    elif fault == "ambiguous-region":
        original = contents.region_summaries[-2]
        contents.region_summaries.append(replace(original, index=6))
    elif fault == "reversed-order":
        contents.sentences[1].provenance.reverse()
    else:
        contents.sentences[1].provenance[0].page_no = 2
        contents.region_summaries[-2].page = 2

    refined = refine_document_records(resolution, contents=contents)

    assert refined is resolution
