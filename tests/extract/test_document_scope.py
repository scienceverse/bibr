"""Synthetic article separation: source ownership, not page-number slicing."""

from dataclasses import replace

import pandas as pd
import pytest

from bibr.extract.document_scope import scope_document_records
from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates
from bibr.extract.front_matter import (
    FrontMatterCandidate,
    FrontMatterResolution,
    group_front_matter_blocks,
)
from bibr.extract.ref_locator import RefLocator
from bibr.paper_contents import (
    CanonicalSection as C,
)
from bibr.paper_contents import (
    PaperContents,
    PaperEquation,
    PaperFigure,
    PaperSection,
    PaperSentence,
    PaperTable,
    PaperURLLink,
    PaperXref,
    Provenance,
    RegionSummary,
)


def _candidate(index, text, role, section, *, text_id=None, page=1):
    return FrontMatterCandidate(
        candidate_id=f"candidate-{index}",
        source_kind="heading" if text_id is None else "paragraph",
        reading_order=index,
        page=page,
        bbox=None,
        region_label="abstract" if role == "abstract" else "text",
        font_size=None,
        font_bold=None,
        section_id=section,
        text_ids=() if text_id is None else (text_id,),
        paragraph_id=text_id,
        raw_text=text,
        normalized_text=text.casefold(),
        roles=frozenset({role}),
    )


def _resolution(candidates):
    candidates = tuple(candidates)
    blocks = group_front_matter_blocks(candidates)
    return FrontMatterResolution(
        candidates,
        blocks,
        blocks[0].block_id if len(blocks) == 1 else None,
        "unique_block" if len(blocks) == 1 else "abstained",
        (),
        frozenset(),
        frozenset(),
    )


def _document(*, count=2, same_page=False):
    sections = [PaperSection(0, "Root", 0, None)]
    sentences, candidates = [], []
    for index in range(count):
        sid, tid, page = index * 4 + 1, index * 10 + 1, 1 if same_page else index * 2 + 1
        title = ["SHADE AND SEEDLING GROWTH", "WATER AND SEEDLING GROWTH"][index]
        rows = [
            (sid, title, C.TITLE),
            (sid + 1, "Abstract", C.ABSTRACT),
            (sid + 2, "Results", C.RESULTS),
            (sid + 3, "References", C.REFERENCES),
        ]
        sections.extend(
            PaperSection(
                key,
                header,
                1 if key == sid else 2,
                0 if key == sid else sid,
                kind,
                provenance=[Provenance(page)],
            )
            for key, header, kind in rows
        )
        content = [
            (
                tid,
                "Mara Quill and Elian Brook" if index == 0 else "Talia Vale and Orin Finch",
                sid,
                "byline",
            ),
            (tid + 1, f"DOI: 10.9999/study-{index + 1}", sid, "doi"),
            (tid + 2, "The controlled study measured seedling development.", sid + 1, "abstract"),
            (tid + 3, f"Article {index + 1} body [1] and Table 1.", sid + 2, None),
            (tid + 4, f"1. River A. Study {index + 1}. Forest Journal. 2020;1:1-9.", sid + 3, None),
        ]
        candidates.append(_candidate(len(candidates), title, "title", sid, page=page))
        for text_id, text, section, role in content:
            sentences.append(PaperSentence(text_id, text, section, text_id, page))
            if role:
                candidates.append(
                    _candidate(len(candidates), text, role, section, text_id=text_id, page=page)
                )
    contents = PaperContents(sentences, sections, [], [], {}, detected_title=sections[1].header)
    return contents, _resolution(candidates)


def _shared_section():
    section = PaperSection(1, "Abstracts", 1, None, C.ABSTRACT)
    sentences, candidates = [], []
    for index, title in enumerate(["SHADE AND SEEDLING GROWTH", "WATER AND SEEDLING GROWTH"]):
        for text, role in [
            (title, "title"),
            ("Mara Quill and Elian Brook" if index == 0 else "Talia Vale and Orin Finch", "byline"),
            (f"DOI: 10.9999/shared-{index + 1}", "doi"),
            (f"Study {index + 1} measured seedling development.", "abstract"),
        ]:
            text_id = len(sentences) + 1
            sentences.append(PaperSentence(text_id, text, 1, text_id, 1))
            candidates.append(_candidate(len(candidates), text, role, 1, text_id=text_id))
    return PaperContents(sentences, [section], [], [], {}), _resolution(candidates)


@pytest.mark.parametrize("same_page", [False, True])
def test_separate_articles_keep_their_own_doi_body_and_bibliography(same_page):
    contents, resolution = _document(same_page=same_page)

    scopes = scope_document_records(contents, resolution)

    assert [scope.record_id for scope in scopes] == [block.block_id for block in resolution.blocks]
    assert [scope.source_text_ids for scope in scopes] == [(1, 2, 3, 4, 5), (11, 12, 13, 14, 15)]
    for index, scope in enumerate(scopes):
        assert scope.contents is not None
        doi = select_doi_candidates(collect_doi_candidates(scope.contents))
        assert doi.selected.normalized == f"10.9999/study-{index + 1}"
        assert RefLocator(scope.contents).collect_reference_rows().text_id.tolist() == [
            index * 10 + 5
        ]
        assert scope.contents.front_matter_resolution.selected_block_id == scope.record_id
        assert len(scope.contents.front_matter_resolution.blocks) == 1
        assert set(scope.contents.front_matter_resolution.allowed_text_ids).issubset(
            scope.source_text_ids
        )


def test_exact_paragraph_anchors_split_one_abstract_section_on_one_page():
    contents, resolution = _shared_section()

    first, second = scope_document_records(contents, resolution)

    assert first.source_text_ids == (1, 2, 3, 4)
    assert second.source_text_ids == (5, 6, 7, 8)
    assert first.pages == second.pages == (1,)
    assert first.source_section_ids == second.source_section_ids == (1,)
    assert first.contents.sections[0].header == second.contents.sections[0].header == ""
    assert "Study 1" in first.contents.sections_text[1]
    assert "Study 1" not in second.contents.sections_text[1]


@pytest.mark.parametrize("fault", ["partial-paragraph", "duplicate-anchor", "missing-anchor"])
def test_ambiguous_paragraph_anchors_never_fall_back_to_whole_document(fault):
    contents, resolution = _shared_section()
    rows = list(resolution.candidates)
    if fault == "partial-paragraph":
        contents.sentences[5].paragraph_id = 5
    elif fault == "duplicate-anchor":
        rows[4] = replace(rows[4], text_ids=(1,), paragraph_id=1)
    else:
        rows[4] = replace(rows[4], text_ids=(), paragraph_id=None)
    resolution = replace(resolution, candidates=tuple(rows))

    scopes = scope_document_records(contents, resolution)

    assert len(scopes) == 2
    assert all(scope.contents is None and scope.reason_flags for scope in scopes)


def test_interleaved_source_sections_are_explicitly_unresolved():
    contents, resolution = _document()
    contents.sentences[3], contents.sentences[6] = contents.sentences[6], contents.sentences[3]

    scopes = scope_document_records(contents, resolution)

    assert all(scope.contents is None for scope in scopes)
    assert all("nonmonotonic_source_order" in scope.reason_flags for scope in scopes)


def test_interleaved_candidate_reading_order_is_unresolved():
    contents, resolution = _document()
    candidates = list(resolution.candidates)
    candidates[2], candidates[5] = candidates[5], candidates[2]

    scopes = scope_document_records(contents, replace(resolution, candidates=tuple(candidates)))

    assert all(scope.contents is None for scope in scopes)


def test_later_byline_before_title_is_not_assigned_to_previous_article():
    contents, resolution = _shared_section()
    candidates = list(resolution.candidates)
    candidates[4], candidates[5] = candidates[5], candidates[4]
    contents.sentences[4], contents.sentences[5] = contents.sentences[5], contents.sentences[4]
    candidates = [
        replace(candidate, reading_order=index) for index, candidate in enumerate(candidates)
    ]
    resolution = replace(resolution, candidates=tuple(candidates))

    assert all(scope.contents is None for scope in scope_document_records(contents, resolution))


def test_objects_and_appended_caption_footnote_text_follow_source_links():
    contents, resolution = _document()
    contents.sections.extend(
        [
            PaperSection(9, "Table 1", 1, 0, C.TABLE),
            PaperSection(10, "Figure 2", 1, 0, C.FIGURE),
            PaperSection(11, "Footnote 1", 1, 0, C.FOOTNOTE),
        ]
    )
    contents.tables = [
        PaperTable(
            1, pd.DataFrame({"count": [3]}), "<table/>", 9, "Table 1. Counts", 1, _body_section_id=3
        )
    ]
    contents.figures = [PaperFigure(2, 10, "image", "Figure 2. Growth", 3, _body_section_id=7)]
    contents.sentences.extend(
        [
            PaperSentence(21, "Table 1. Counts", 9, 21, 1),
            PaperSentence(22, "Figure 2. Growth", 10, 22, 3),
            PaperSentence(23, "A note for article one.", 11, 23, 1),
        ]
    )
    contents.links = [
        PaperURLLink("https://example.org/one", 3, 4, 4),
        PaperURLLink("https://example.org/two", 7, 14, 14),
    ]
    contents.equations = [
        PaperEquation(4, 1, "p", "=", "0.1"),
        PaperEquation(14, 2, "p", "=", "0.2"),
    ]
    contents.xrefs = [
        PaperXref(1, "table", "Table 1", 4),
        PaperXref(2, "figure", "Figure 2", 14),
        PaperXref(1, "foot", "1", 4),
        PaperXref(1, "bib", "[1]", 4),
    ]

    first, second = scope_document_records(contents, resolution)

    assert first.contents is not None and second.contents is not None
    assert first.source_text_ids == (1, 2, 3, 4, 5, 21, 23)
    assert second.source_text_ids == (11, 12, 13, 14, 15, 22)
    assert [table.table_id for table in first.contents.tables] == [1]
    assert second.contents.tables == []
    assert [figure.figure_id for figure in second.contents.figures] == [2]
    assert [link.text_id for link in first.contents.links] == [4]
    assert [eq.grp_id for eq in second.contents.equations] == [2]
    assert {xref.xref_type for xref in first.contents.xrefs} == {"table", "foot"}


def test_shared_section_object_is_not_assigned_using_page_alone():
    contents, resolution = _shared_section()
    contents.tables = [PaperTable(1, pd.DataFrame({"count": [3]}), "<table/>", 1, page_number=1)]

    scopes = scope_document_records(contents, resolution)

    assert all(scope.contents is None for scope in scopes)
    assert all("ambiguous_object_ownership" in scope.reason_flags for scope in scopes)


def test_unowned_appended_object_text_is_not_silently_lost():
    contents, resolution = _document()
    contents.sections.append(PaperSection(9, "Table 1", 1, 0, C.TABLE))
    contents.sentences.append(PaperSentence(21, "Table content without its object", 9, 21, 1))

    scopes = scope_document_records(contents, resolution)

    assert all(scope.contents is None for scope in scopes)
    assert all("unowned_auxiliary_content" in scope.reason_flags for scope in scopes)


def test_cross_record_object_link_blocks_the_source_record():
    contents, resolution = _document()
    contents.figures = [PaperFigure(2, 7, None, "Growth", 3)]
    contents.xrefs = [PaperXref(2, "figure", "Figure 2", 4)]

    first, second = scope_document_records(contents, resolution)

    assert first.contents is None
    assert "unresolved_or_cross_record_link" in first.reason_flags
    assert second.contents is not None


def test_copies_detach_foreign_parents_and_do_not_share_mutation_or_global_metadata():
    contents, resolution = _document()
    contents.sections[5].parent_section_id = 1
    contents.detected_headers = ["DOI: 10.9999/global-header"]
    contents.detected_footers = ["Global footer"]
    contents.ref_line_geometry = [{"text": "all references"}]
    contents.native_ref_strings = ["global reference"]
    contents.region_summaries = [
        RegionSummary(1, 0, "text", None, section_id=3),
        RegionSummary(3, 1, "text", None, section_id=7),
    ]
    contents.processing_warnings = ["other record failed"]
    contents.reference_boundary_reason_flags = ["global receipt"]
    _ = contents.sentences_df

    first, second = scope_document_records(contents, resolution)
    second.contents.sentences[0].text = "Changed"
    second.contents.sections[0].header = "Changed section"

    assert contents.sentences[5].text == "Talia Vale and Orin Finch"
    assert "Changed" not in first.contents.sentences_df.text.tolist()
    assert second.contents.sections[0].parent_section_id is None
    for scope in (first, second):
        assert scope.contents.detected_headers == scope.contents.detected_footers == []
        assert scope.contents.ref_line_geometry is None
        assert scope.contents.native_ref_strings is None
        assert scope.contents.processing_warnings == []
        assert scope.contents.reference_boundary_reason_flags == []
        assert len(scope.contents.region_summaries) == 1


@pytest.mark.parametrize("reason", ["toc", "identity"])
def test_single_unresolved_or_toc_record_is_retained_without_contents(reason):
    contents, resolution = _document(count=1)
    resolution = replace(
        resolution, selected_block_id=None, reason_flags=("toc_listing",) if reason == "toc" else ()
    )

    scopes = scope_document_records(contents, resolution)

    assert len(scopes) == 1 and scopes[0].contents is None


def test_one_selected_record_retains_full_source_in_an_independent_copy():
    contents, resolution = _document(count=1)
    contents.sentences.insert(0, PaperSentence(50, "Cover information", 0, 50, 1))
    contents.reference_boundary_reason_flags = ["old"]

    (scope,) = scope_document_records(contents, resolution)

    assert scope.source_text_ids == (50, 1, 2, 3, 4, 5)
    assert scope.contents.front_matter_resolution.selected_block_id == scope.record_id
    assert scope.contents.reference_boundary_reason_flags == []
    scope.contents.sentences[0].text = "changed"
    assert contents.sentences[0].text == "Cover information"


@pytest.mark.parametrize("extra_role", ["byline", "affiliation"])
def test_single_selected_record_does_not_require_a_pure_title_role(extra_role):
    contents, resolution = _document(count=1)
    title, *remaining = resolution.candidates
    resolution = replace(
        resolution,
        candidates=(replace(title, roles=title.roles | {extra_role}), *remaining),
    )

    (scope,) = scope_document_records(contents, resolution)

    assert scope.contents is not None
    assert scope.source_text_ids == tuple(row.text_id for row in contents.sentences)
    assert scope.contents.front_matter_resolution.selected_block_id == scope.record_id


def test_single_selected_record_without_any_source_title_stays_unresolved():
    contents, resolution = _document(count=1)
    title, *remaining = resolution.candidates
    resolution = replace(
        resolution, candidates=(replace(title, roles=frozenset({"heading"})), *remaining)
    )

    (scope,) = scope_document_records(contents, resolution)

    assert scope.contents is None
    assert scope.reason_flags == ("missing_record_title",)


def test_cross_page_paragraph_preserves_every_source_page_and_region():
    contents, resolution = _document()
    first_bbox, continuation_bbox = (10.0, 500.0, 450.0, 700.0), (10.0, 20.0, 450.0, 100.0)
    contents.sentences[4].provenance = [
        Provenance(1, first_bbox),
        Provenance(2, continuation_bbox),
    ]
    contents.region_summaries = [
        RegionSummary(1, 10, "text", first_bbox, section_id=4),
        RegionSummary(2, 0, "text", continuation_bbox, section_id=4),
    ]

    first, second = scope_document_records(contents, resolution)

    assert first.contents is not None and second.contents is not None
    assert first.pages == (1, 2)
    assert second.pages == (3,)
    assert [region.page for region in first.contents.region_summaries] == [1, 2]
    assert [p.page_no for p in first.contents.sentences[-1].provenance] == [1, 2]
