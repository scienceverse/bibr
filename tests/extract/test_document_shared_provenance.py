"""Paragraph provenance stays intact across real sentence segmentation."""

from dataclasses import replace

import pytest

from bibr.extract.document_records import refine_document_records
from bibr.extract.document_scope import scope_document_records
from bibr.paper_contents import Provenance, RegionSummary
from bibr.structure.assembler import DocumentAssembler
from bibr.structure.parse_text import TextHandlersMixin
from tests.extract.test_document_records_review import _column_spanning_records
from tests.extract.test_document_scope_review import _two_articles


def _segmented_articles(*, cross_page=True):
    contents, resolution = _two_articles()
    regions = []

    def region(page, index, section):
        if not cross_page:
            page, index = 1, len(regions)
        bbox = (20.0, 40.0 + len(regions) * 20, 400.0, 55.0 + len(regions) * 20)
        row = RegionSummary(page, index, "text", bbox, section_id=section)
        regions.append(row)
        return Provenance(page, bbox)

    contents.sections[0].provenance = [region(1, 0, 1)]
    contents.sentences[0].provenance = [region(1, 1, 1)]
    footprint = [region(1, 2, 1), region(2, 0, 1)]
    parts = [
        "We measured the first response before treatment.",
        "The continuation describes the response after treatment.",
        "These observations complete the first abstract.",
    ]
    assembler = DocumentAssembler()
    assembler.append(
        " ".join(parts),
        1,
        1,
        True,
        provenance=footprint,
        page_spans=[(0, 1), (len(parts[0]) + 1, 2)] if cross_page else [],
    )
    siblings, _, _ = assembler.emit(
        [parts],
        sentence_factory=TextHandlersMixin()._make_sentence,
        sentence_counter=20,
        paragraph_counter=1,
    )
    contents.sections[1].provenance = [region(2, 1, 2)]
    contents.sentences[2].provenance = [region(2, 2, 2)]
    contents.sections[2].provenance = [region(2, 3, 3)]
    contents.sentences[3].provenance = [region(2, 4, 3)]
    contents.sections[3].provenance = [region(3, 0, 4)]
    contents.sentences[4].provenance = [region(3, 1, 4)]
    contents.sentences[5].provenance = [region(3, 2, 4)]
    contents.sections[4].provenance = [region(3, 3, 5)]
    contents.sentences[6].provenance = [region(3, 4, 5)]
    contents.sections[5].provenance = [region(3, 5, 6)]
    contents.sentences[7].provenance = [region(3, 6, 6)]
    for sentence in contents.sentences:
        if sentence.provenance:
            sentence.page_number = sentence.provenance[0].page_no
    contents.sentences[1:2] = siblings
    contents.region_summaries = regions
    candidates = list(resolution.candidates)
    for index, candidate in enumerate(candidates):
        candidates[index] = replace(candidate, page=(1 if index < 3 else 3) if cross_page else 1)
    candidates[2] = replace(
        candidates[2],
        page=None if cross_page else 1,
        text_ids=tuple(row.text_id for row in siblings),
        raw_text=" ".join(parts),
        normalized_text=" ".join(parts).casefold(),
    )
    return contents, replace(resolution, candidates=tuple(candidates))


@pytest.mark.parametrize("cross_page", [False, True])
@pytest.mark.parametrize("partial_geometry", [False, True])
def test_segmented_paragraph_is_scoped_without_changing_sentence_pages(
    cross_page, partial_geometry
):
    contents, resolution = _segmented_articles(cross_page=cross_page)
    if partial_geometry:
        contents.sentences[-1].provenance = []
    original = [
        (row.text_id, row.text, row.page_number, list(row.provenance)) for row in contents.sentences
    ]
    assert [row.page_number for row in contents.sentences[1:4]] == (
        [1, 2, 2] if cross_page else [1, 1, 1]
    )

    first, second = scope_document_records(contents, resolution)

    assert first.contents is not None and second.contents is not None
    assert not set(first.source_text_ids) & set(second.source_text_ids)
    assert first.pages == ((1, 2) if cross_page else (1,))
    assert second.pages == ((3,) if cross_page else (1,))
    retained = [
        (row.text_id, row.text, row.page_number, list(row.provenance))
        for scope in (first, second)
        for row in scope.contents.sentences
    ]
    assert retained == original
    assert not any("record 2" in row.text for row in first.contents.sentences)
    assert not any("record 1" in row.text for row in second.contents.sentences)


@pytest.mark.parametrize("partial_geometry", [False, True])
@pytest.mark.parametrize("fault", ["foreign-page", "reordered", "other-paragraph", "crosses-title"])
def test_shared_footprint_cannot_hide_conflicting_physical_evidence(fault, partial_geometry):
    contents, resolution = _segmented_articles(cross_page=False)
    siblings = contents.sentences[1:4]
    if fault == "foreign-page":
        siblings[1].page_number = 99
    elif fault == "reordered":
        for row in siblings:
            row.provenance.reverse()
    elif fault == "other-paragraph":
        siblings[1].paragraph_id = 99
    else:
        for row in siblings:
            row.provenance[-1] = contents.sentences[-1].provenance[0]
    if partial_geometry:
        contents.sentences[-2].provenance = []

    scopes = scope_document_records(contents, resolution)

    assert all(scope.contents is None and scope.reason_flags for scope in scopes)


def test_split_column_spanning_byline_keeps_full_source_and_supports_record_anatomy():
    contents, resolution = _column_spanning_records()
    original = contents.sentences[1]
    # Actual segmentation duplicates the paragraph footprint, including the
    # return to a higher position in the next column, on both sentences.
    first, rest = original.text.split(". ", 1)
    segments = [first + ".", rest]
    assembler = DocumentAssembler()
    assembler.append(
        original.text,
        original.page_number,
        original.section_id,
        True,
        provenance=original.provenance,
    )
    siblings, _, _ = assembler.emit(
        [segments],
        sentence_factory=TextHandlersMixin()._make_sentence,
        sentence_counter=100,
        paragraph_counter=original.paragraph_id - 1,
    )
    contents.sentences[1:] = siblings
    candidate = replace(resolution.candidates[3], text_ids=tuple(row.text_id for row in siblings))
    resolution = replace(resolution, candidates=resolution.candidates[:3] + (candidate,))

    refined = refine_document_records(resolution, contents=contents)

    assert len(refined.blocks) == 2
    assert "document_contextual_byline" in refined.candidates[3].roles
    assert refined.candidates[3].bbox == candidate.bbox
    assert refined.candidates[3].raw_text == original.text
    assert refined.candidates[3].text_ids == candidate.text_ids
    assert all(row.provenance == original.provenance for row in siblings)
