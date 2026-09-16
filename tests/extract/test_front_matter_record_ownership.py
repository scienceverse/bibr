"""Conservative record identity across printed variants and body headings.

All titles, authors, identifiers and prose are synthetic. These tests retain
distinct proceedings records while allowing repeated front matter belonging
to one explicitly supported article identity.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bibr.extract.front_matter import (
    FrontMatterCandidate,
    group_front_matter_blocks,
    resolve_front_matter,
)
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)


def _candidate(index: int, text: str, role: str, *, page: int = 1) -> FrontMatterCandidate:
    return FrontMatterCandidate(
        candidate_id=f"candidate-{index}",
        source_kind="paragraph",
        reading_order=index,
        page=page,
        bbox=(60.0, 40.0 + index * 25, 500.0, 60.0 + index * 25),
        region_label="doc_title" if role == "title" else role,
        font_size=16.0 if role == "title" else 10.0,
        font_bold=role == "title",
        section_id=0,
        text_ids=(index,),
        paragraph_id=index,
        raw_text=text,
        normalized_text=" ".join(text.casefold().split()),
        roles=frozenset({role}),
    )


def _record(
    offset: int,
    title: str,
    *,
    byline: str = "Mara Quill and Elian Brook",
    doi: str | None = None,
    abstract: str = "We measured the relationship between shade and seedling growth.",
) -> list[FrontMatterCandidate]:
    rows = [(title, "title"), (byline, "byline"), (abstract, "abstract")]
    if doi is not None:
        rows.append((f"https://doi.org/{doi}", "doi"))
    return [_candidate(offset + index, text, role) for index, (text, role) in enumerate(rows)]


def test_translated_complete_front_matter_with_shared_doi_and_byline_is_one_record():
    candidates = tuple(
        _record(1, "SHADE AND SEEDLING GROWTH", doi="10.9999/shade-study")
        + _record(
            5,
            "SOMBRA E CRESCIMENTO DE MUDAS",
            doi="10.9999/shade-study",
            abstract="Medimos a relação entre sombra e crescimento de mudas.",
        )
    )

    blocks = group_front_matter_blocks(candidates)

    assert len(blocks) == 1
    assert blocks[0].title_candidate_ids == ("candidate-1", "candidate-5")
    assert blocks[0].candidate_ids == tuple(candidate.candidate_id for candidate in candidates)


def test_adjacent_repeated_title_and_byline_without_doi_is_one_record():
    candidates = tuple(
        _record(1, "SHADE AND SEEDLING GROWTH") + _record(4, "SHADE AND SEEDLING GROWTH")
    )

    blocks = group_front_matter_blocks(candidates)

    assert len(blocks) == 1
    assert blocks[0].title_candidate_ids == ("candidate-1", "candidate-4")
    assert len(blocks[0].candidate_ids) == len(candidates)


@pytest.mark.parametrize(("first_page", "second_page"), [(1, 8), (8, 1)])
def test_repeated_identity_does_not_merge_distant_or_reversed_pages(first_page, second_page):
    first = [replace(row, page=first_page) for row in _record(1, "SHADE AND SEEDLING GROWTH")]
    second = [replace(row, page=second_page) for row in _record(4, "SHADE AND SEEDLING GROWTH")]
    assert len(group_front_matter_blocks(tuple(first + second))) == 2


@pytest.mark.parametrize(("section_id", "label"), [(42, "text"), (0, "reference_content")])
def test_shared_cited_doi_cannot_merge_distinct_articles_by_same_authors(section_id, label):
    first = _record(1, "SHADE AND SEEDLING GROWTH", doi="10.9999/shared-citation")
    second = _record(5, "WATER AND SEEDLING GROWTH", doi="10.9999/shared-citation")
    for rows in (first, second):
        rows[-1] = replace(rows[-1], section_id=section_id, region_label=label)
    assert len(group_front_matter_blocks(tuple(first + second))) == 2


@pytest.mark.parametrize(
    ("second_title", "second_byline", "first_doi", "second_doi"),
    [
        ("SOIL AND SEEDLING GROWTH", "Mara Quill and Elian Brook", None, None),
        ("SHADE AND SEEDLING GROWTH", "Talia Vale and Orin Finch", None, None),
        (
            "SHADE AND SEEDLING GROWTH",
            "Mara Quill and Elian Brook",
            "10.9999/shade-one",
            "10.9999/shade-two",
        ),
        (
            "SOMBRA E CRESCIMENTO DE MUDAS",
            "Talia Vale and Orin Finch",
            "10.9999/shared",
            "10.9999/shared",
        ),
    ],
    ids=["same-authors-only", "same-title-only", "conflicting-dois", "same-doi-different-authors"],
)
def test_partial_identity_agreement_does_not_merge_independent_records(
    second_title, second_byline, first_doi, second_doi
):
    first = _record(1, "SHADE AND SEEDLING GROWTH", doi=first_doi)
    second = _record(
        len(first) + 1,
        second_title,
        byline=second_byline,
        doi=second_doi,
    )

    blocks = group_front_matter_blocks(tuple(first + second))

    assert len(blocks) == 2
    assert blocks[0].candidate_ids == tuple(candidate.candidate_id for candidate in first)
    assert blocks[1].candidate_ids == tuple(candidate.candidate_id for candidate in second)


def _section(section_id, header, section_type, *, page=1, parent=None, level=1):
    return PaperSection(
        section_id=section_id,
        header=header,
        level=level,
        parent_section_id=parent,
        section_type=section_type,
        provenance=[Provenance(page_no=page, bbox=(60.0, 50.0, 500.0, 80.0))],
    )


def _sentence(text_id, text, section_id, *, page=1, label="text"):
    return PaperSentence(
        text_id=text_id,
        text=text,
        section_id=section_id,
        paragraph_id=text_id,
        page_number=page,
        provenance=[Provenance(page_no=page, bbox=(60.0, 100.0, 500.0, 160.0))],
        region_meta={"region_type": label, "font_size": 10.0},
    )


def _contents(sections, sentences, *, detected_title, region_summaries=()):
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={section.section_id: "" for section in sections},
        detected_title=detected_title,
        region_summaries=list(region_summaries),
    )


def test_numbered_unknown_descendant_body_headings_cannot_root_article_records():
    title = "SHADE AND SEEDLING GROWTH"
    body_headers = [
        "2. EFFECTS OF SOIL WATER ON THE DEVELOPMENT OF YOUNG PLANTS",
        "3. IMPLICATIONS FOR THE CONSERVATION OF FRAGILE ECOSYSTEMS",
    ]
    sections = [
        _section(1, title, CanonicalSection.TITLE),
        _section(2, "Abstract", CanonicalSection.ABSTRACT, parent=1, level=2),
        _section(3, body_headers[0], CanonicalSection.UNKNOWN, page=2, parent=1, level=2),
        _section(4, body_headers[1], CanonicalSection.UNKNOWN, page=3, parent=1, level=2),
    ]
    contents = _contents(
        sections,
        [
            _sentence(1, "Mara Quill and Elian Brook", 1),
            _sentence(2, "We measured shade and seedling growth in a controlled study.", 2),
            _sentence(
                3, "Earlier work is available at https://doi.org/10.9999/earlier-soil.", 3, page=2
            ),
            _sentence(
                4, "We compare the results with https://doi.org/10.9999/earlier-water.", 4, page=3
            ),
        ],
        detected_title=title,
    )

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is not None
    assert len(resolution.blocks) == 1
    assert not any(issue.blocking for issue in issues)
    assert all(
        "title" not in candidate.roles
        for candidate in resolution.candidates
        if candidate.raw_text in body_headers
    )


def test_numbered_explicit_document_titles_still_create_distinct_records():
    first_title = "12. SHADE AND SEEDLING GROWTH"
    second_title = "13. WATER AND SEEDLING GROWTH"
    sections = [
        _section(1, first_title, CanonicalSection.TITLE),
        _section(2, "Abstract", CanonicalSection.ABSTRACT, parent=1, level=2),
        _section(3, second_title, CanonicalSection.TITLE, page=2),
        _section(4, "Abstract", CanonicalSection.ABSTRACT, page=2, parent=3, level=2),
    ]
    summaries = [
        RegionSummary(
            page=page,
            index=0,
            label="doc_title",
            bbox=(60.0, 50.0, 500.0, 80.0),
            section_id=section_id,
        )
        for section_id, page in [(1, 1), (3, 2)]
    ]
    contents = _contents(
        sections,
        [
            _sentence(1, "Mara Quill and Elian Brook", 1),
            _sentence(2, "The first study examined shade and seedling growth.", 2),
            _sentence(3, "Talia Vale and Orin Finch", 3, page=2),
            _sentence(4, "The second study examined water and seedling growth.", 4, page=2),
        ],
        detected_title=first_title,
        region_summaries=summaries,
    )

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id is None
    assert any(issue.code == "VAL_METADATA_MULTI_ITEM" and issue.blocking for issue in issues)


@pytest.mark.parametrize(("parent", "level"), [(0, 1), (1, 2)])
def test_numbered_unknown_proceedings_title_with_own_anatomy_stays_independent(parent, level):
    first_title = "12. SHADE AND SEEDLING GROWTH"
    second_title = "13. WATER AND SEEDLING GROWTH"
    contents = _contents(
        [
            _section(0, "Root", CanonicalSection.UNKNOWN, level=0),
            _section(1, first_title, CanonicalSection.TITLE, parent=0),
            _section(2, "Abstract", CanonicalSection.ABSTRACT, parent=1, level=2),
            _section(3, second_title, CanonicalSection.UNKNOWN, page=2, parent=parent, level=level),
            _section(4, "Abstract", CanonicalSection.ABSTRACT, page=2, parent=3, level=level + 1),
        ],
        [
            _sentence(1, "Mara Quill and Elian Brook", 1),
            _sentence(2, "The first study examined shade and seedling growth.", 2),
            _sentence(3, "Talia Vale and Orin Finch", 3, page=2),
            _sentence(4, "The second study examined water and seedling growth.", 4, page=2),
        ],
        detected_title=first_title,
    )

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id is None
    assert any(issue.blocking for issue in issues)
    assert "title" in next(
        candidate.roles for candidate in resolution.candidates if candidate.raw_text == second_title
    )
