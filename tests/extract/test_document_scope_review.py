"""Independent boundary review: record scopes never borrow another paper's objects."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from bibr.extract.document_scope import scope_document_records
from bibr.extract.front_matter import FrontMatterBlock, FrontMatterCandidate, FrontMatterResolution
from bibr.extract.ref_locator import RefLocator
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    PaperTable,
    PaperTablePart,
    PaperXref,
    Provenance,
    RegionSummary,
)


def _two_articles():
    sections, sentences, candidates, blocks = [], [], [], []
    for index, (title, author) in enumerate(
        [
            ("Shade responses in alpine seedlings", "Mara Quill"),
            ("Water responses in coastal grasses", "Elian Brook"),
        ]
    ):
        page, sid, first = index + 1, index * 3 + 1, index * 4 + 1
        sections.extend(
            [
                PaperSection(sid, title, 1, None, CanonicalSection.TITLE),
                PaperSection(sid + 1, "Methods", 2, sid, CanonicalSection.METHODS),
                PaperSection(sid + 2, "References", 2, sid, CanonicalSection.REFERENCES),
            ]
        )
        rows = [
            (first, author, sid),
            (first + 1, f"We measured response {index + 1}.", sid),
            (first + 2, f"Only record {index + 1} body [1].", sid + 1),
            (first + 3, f"Only record {index + 1} cited work. Journal 2025.", sid + 2),
        ]
        sentences.extend(
            PaperSentence(text_id, text, section_id, text_id, page_number=page)
            for text_id, text, section_id in rows
        )
        member_ids = []
        for source_kind, text, roles, text_ids in [
            ("heading", title, {"title", "heading"}, ()),
            ("paragraph", author, {"byline"}, (first,)),
            ("paragraph", rows[1][1], {"abstract"}, (first + 1,)),
        ]:
            key = f"candidate-{len(candidates) + 1}"
            member_ids.append(key)
            candidates.append(
                FrontMatterCandidate(
                    candidate_id=key,
                    source_kind=source_kind,
                    reading_order=len(candidates),
                    page=page,
                    bbox=None,
                    region_label="doc_title" if source_kind == "heading" else "text",
                    font_size=None,
                    font_bold=None,
                    section_id=sid,
                    text_ids=text_ids,
                    paragraph_id=text_ids[0] if text_ids else None,
                    raw_text=text,
                    normalized_text=text.casefold(),
                    roles=frozenset(roles),
                )
            )
        blocks.append(FrontMatterBlock(f"record-{page}", tuple(member_ids), (member_ids[0],)))
    contents = PaperContents(sentences, sections, [], [], {})
    resolution = FrontMatterResolution(
        tuple(candidates), tuple(blocks), None, "abstained", (), frozenset(), frozenset()
    )
    return contents, resolution


def _table(table_id, *, owner_section, page, text):
    return PaperTable(
        table_id,
        pd.DataFrame([[text]], columns=["Evidence"]),
        f"<table><tr><td>{text}</td></tr></table>",
        owner_section,
        caption=text,
        page_number=page,
        _body_section_id=owner_section,
    )


def test_scoped_reference_search_cannot_select_other_records_bibliography():
    contents, resolution = _two_articles()
    scopes = scope_document_records(contents, resolution)

    assert len(scopes) == 2
    for index, scope in enumerate(scopes):
        assert scope.contents is not None
        references = RefLocator(scope.contents).collect_reference_rows()
        assert references["text_id"].tolist() == [index * 4 + 4]
        assert all(f"record {index + 1}" in text for text in references["text"])


def test_scoped_contents_own_copies_and_clear_document_native_reference_shortcuts():
    contents, resolution = _two_articles()
    contents.native_ref_strings = ["Unowned document-level reference shortcut"]
    contents.xrefs = [PaperXref(1, "bib", "[1]", 3), PaperXref(1, "bib", "[1]", 7)]

    first, second = scope_document_records(contents, resolution)

    assert first.contents is not None and second.contents is not None
    assert first.contents.native_ref_strings is None
    assert second.contents.native_ref_strings is None
    assert first.contents.xrefs == second.contents.xrefs == []
    first.contents.sentences[0].text = "Changed only in first record"
    first.contents.sections[0].header = "Changed first section"
    assert contents.sentences[0].text == "Mara Quill"
    assert contents.sections[0].header == "Shade responses in alpine seedlings"
    assert second.contents.sentences[0].text == "Elian Brook"


def test_duplicate_document_table_ids_cannot_reassign_first_records_object_to_second():
    contents, resolution = _two_articles()
    contents.tables = [
        _table(1, owner_section=2, page=1, text="First record private table"),
        _table(1, owner_section=5, page=2, text="Second record private table"),
    ]

    scopes = scope_document_records(contents, resolution)

    assert all(scope.contents is None for scope in scopes)
    assert all(scope.reason_flags for scope in scopes)


@pytest.mark.parametrize("provenance_only", [False, True])
def test_composite_table_cannot_carry_physical_part_from_other_records_page(provenance_only):
    contents, resolution = _two_articles()
    first_table = _table(1, owner_section=2, page=1, text="First record table")
    first_table.parts = [
        PaperTablePart(
            page_number=None if provenance_only else 2,
            bbox=None if provenance_only else (20.0, 200.0, 500.0, 400.0),
            tbl_html="<table><tr><td>Second record private part</td></tr></table>",
            df=pd.DataFrame([["Second record private part"]], columns=["Evidence"]),
            provenance=[Provenance(page_no=2, bbox=(20.0, 200.0, 500.0, 400.0))],
        )
    ]
    contents.tables = [first_table]

    first, _ = scope_document_records(contents, resolution)

    assert first.contents is None
    assert first.reason_flags


@pytest.mark.parametrize("record_index", [0, 1])
def test_title_anchor_cannot_be_inferred_from_partial_paragraph(record_index):
    contents, resolution = _two_articles()
    candidate_index = record_index * 3
    title = resolution.candidates[candidate_index]
    source = contents.sentences[record_index * 4]
    changed = replace(
        title,
        source_kind="paragraph",
        text_ids=(source.text_id,),
        paragraph_id=source.paragraph_id,
    )
    # A second source sentence belongs to the same paragraph, so an anchor
    # containing only one sentence is not a complete article-title boundary.
    contents.sentences[record_index * 4 + 1].paragraph_id = source.paragraph_id
    candidates = list(resolution.candidates)
    candidates[candidate_index] = changed
    resolution = replace(resolution, candidates=tuple(candidates))

    scopes = scope_document_records(contents, resolution)

    assert all(scope.contents is None for scope in scopes)


@pytest.mark.parametrize("source_kind", ["heading", "paragraph"])
def test_anchor_text_must_match_its_actual_source_row(source_kind):
    contents, resolution = _two_articles()
    first = resolution.candidates[0]
    if source_kind == "paragraph":
        first = replace(
            first,
            source_kind="paragraph",
            text_ids=(1,),
            paragraph_id=1,
        )
    else:
        contents.sections[0].header = "Another article entirely"
    resolution = replace(resolution, candidates=(first,) + resolution.candidates[1:])

    scopes = scope_document_records(contents, resolution)

    assert all(scope.contents is None for scope in scopes)


def test_same_page_object_part_with_other_records_region_evidence_is_not_shared():
    contents, resolution = _two_articles()
    for row in contents.sentences:
        row.page_number = 1
    resolution = replace(
        resolution,
        candidates=tuple(replace(row, page=1) for row in resolution.candidates),
    )
    second_bbox = (20.0, 400.0, 500.0, 600.0)
    contents.sentences[6].provenance = [Provenance(page_no=1, bbox=second_bbox)]
    contents.region_summaries = [
        RegionSummary(
            1, 1, "table", second_bbox, section_id=5, content="Second record private part"
        )
    ]
    first_table = _table(1, owner_section=2, page=1, text="First record table")
    first_table.parts = [
        PaperTablePart(
            page_number=1,
            bbox=second_bbox,
            tbl_html="<table><tr><td>Second record private part</td></tr></table>",
            df=pd.DataFrame([["Second record private part"]], columns=["Evidence"]),
            provenance=[Provenance(page_no=1, bbox=second_bbox)],
        )
    ]
    contents.tables = [first_table]

    first, _ = scope_document_records(contents, resolution)

    assert first.contents is None


def test_reused_abstract_section_id_does_not_override_ordered_source_paragraphs():
    contents, resolution = _two_articles()
    contents.sections.insert(1, PaperSection(7, "Abstract", 2, 1, CanonicalSection.ABSTRACT))
    candidates = list(resolution.candidates)
    for sentence_index, candidate_index in ((1, 2), (5, 5)):
        contents.sentences[sentence_index].section_id = 7
        candidates[candidate_index] = replace(candidates[candidate_index], section_id=7)
    resolution = replace(resolution, candidates=tuple(candidates))

    first, second = scope_document_records(contents, resolution)

    assert first.contents is not None and second.contents is not None
    assert first.source_text_ids == (1, 2, 3, 4)
    assert second.source_text_ids == (5, 6, 7, 8)
    assert "response 2" not in first.contents.sections_text[7]
    assert "response 1" not in second.contents.sections_text[7]


def test_same_page_source_paragraph_interleaving_is_not_repaired_from_candidate_order():
    contents, resolution = _two_articles()
    for sentence in contents.sentences:
        sentence.page_number = 1
    resolution = replace(
        resolution,
        candidates=tuple(replace(candidate, page=1) for candidate in resolution.candidates),
    )
    # Physical/source list order now puts record two's byline before record
    # one's abstract. Reordering by metadata candidate order would hide that.
    contents.sentences[1], contents.sentences[4] = contents.sentences[4], contents.sentences[1]

    scopes = scope_document_records(contents, resolution)

    assert all(scope.contents is None for scope in scopes)


def test_title_candidate_page_cannot_contradict_its_printed_heading_provenance():
    contents, resolution = _two_articles()
    title_bbox = (20.0, 10.0, 500.0, 50.0)
    contents.sections[0].provenance = [Provenance(page_no=1, bbox=title_bbox)]
    first = replace(resolution.candidates[0], page=2, bbox=title_bbox)
    resolution = replace(resolution, candidates=(first,) + resolution.candidates[1:])

    scopes = scope_document_records(contents, resolution)

    assert all(scope.contents is None for scope in scopes)


def test_foreign_byline_text_cannot_be_admitted_under_owned_source_ids():
    contents, resolution = _two_articles()
    candidates = list(resolution.candidates)
    candidates[1] = replace(candidates[1], raw_text="Elian Brook", normalized_text="elian brook")
    resolution = replace(resolution, candidates=tuple(candidates))

    first, _ = scope_document_records(contents, resolution)

    assert first.contents is None


def _physical_shared_abstract_document():
    contents, resolution = _two_articles()
    contents.sections.insert(1, PaperSection(7, "Abstract", 2, 1, CanonicalSection.ABSTRACT))
    candidates = list(resolution.candidates)
    for sentence_index, candidate_index in ((1, 2), (5, 5)):
        contents.sentences[sentence_index].section_id = 7
        candidates[candidate_index] = replace(candidates[candidate_index], section_id=7)
    # A second column starts above the bottom of the first column. Region
    # reading order, not sorting bbox y-coordinates, establishes these anchors.
    sections = {section.section_id: section for section in contents.sections}
    source_rows = [
        sections[1],
        *contents.sentences[:2],
        sections[2],
        contents.sentences[2],
        sections[3],
        contents.sentences[3],
        sections[4],
        *contents.sentences[4:6],
        sections[5],
        contents.sentences[6],
        sections[6],
        contents.sentences[7],
    ]
    for index, row in enumerate(source_rows):
        column, offset = divmod(index, 7)
        left, top = 20.0 + column * 500, 20.0 + offset * 50
        bbox = (left, top, left + 450, top + 40)
        row.provenance = [Provenance(page_no=1, bbox=bbox)]
        if isinstance(row, PaperSentence):
            row.page_number = 1
        contents.region_summaries.append(
            RegionSummary(1, index, "text", bbox, section_id=row.section_id)
        )
    candidates = [replace(candidate, page=1) for candidate in candidates]
    return contents, replace(resolution, candidates=tuple(candidates))


def test_physical_region_order_recovers_shared_abstract_across_two_columns():
    contents, resolution = _physical_shared_abstract_document()

    first, second = scope_document_records(contents, resolution)

    assert first.contents is not None and second.contents is not None
    assert first.source_text_ids == (1, 2, 3, 4)
    assert second.source_text_ids == (5, 6, 7, 8)
    assert first.pages == second.pages == (1,)
    assert "source_region_interval_scope" in first.reason_flags
    assert "response 2" not in first.contents.sections_text[7]
    assert "response 1" not in second.contents.sections_text[7]


@pytest.mark.parametrize("fault", ["interleaved-regions", "crossing-paragraph"])
@pytest.mark.parametrize("complete_provenance", [False, True])
def test_physical_source_evidence_cannot_cross_a_record_anchor(fault, complete_provenance):
    contents, resolution = _physical_shared_abstract_document()
    if fault == "interleaved-regions":
        first_abstract = contents.sentences[1]
        second_byline = contents.sentences[4]
        first_abstract.provenance, second_byline.provenance = (
            second_byline.provenance,
            first_abstract.provenance,
        )
    else:
        # A merged OCR paragraph with provenance on both sides of the next
        # article title cannot be assigned wholly to either paper.
        contents.sentences[1].provenance.extend(contents.sentences[4].provenance)
    if not complete_provenance:
        # Missing geometry on an unrelated bibliography row must not erase
        # contradictory physical evidence that is available for the front matter.
        contents.sentences[-1].provenance = []

    scopes = scope_document_records(contents, resolution)

    assert all(scope.contents is None for scope in scopes)
