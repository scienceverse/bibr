"""Synthetic cases for metadata evidence loss and bibliography ownership."""

import pytest

from bibr.extract.core_metadata import grounded_authors_in_context
from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates
from bibr.extract.ref_locator import RefLocator
from bibr.paper import PaperAuthor
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
from bibr.structure.section_tree import repair_appendix_hierarchy


def _contents(sections, rows=(), *, headers=()):
    return PaperContents(
        sections=sections,
        sentences=[
            PaperSentence(
                text_id=i,
                text=text,
                section_id=sid,
                paragraph_id=i,
                page_number=page,
                region_meta={"region_type": "text"},
            )
            for i, (sid, text, page) in enumerate(rows, 1)
        ],
        tables=[],
        links=[],
        sections_text={},
        detected_headers=list(headers),
    )


@pytest.mark.parametrize("marker", ["2a", "1b", "12c", "²ᵃ", "1b,3c"])
def test_grounding_keeps_names_with_digit_letter_affiliation_markers(marker):
    author = PaperAuthor(author_id=1, given="Alice", family="Mori", affiliation="")
    kept, rejected = grounded_authors_in_context([author], f"Alice Mori{marker}, Bob Smith3")
    assert kept == [author]
    assert rejected == []


@pytest.mark.parametrize("printed", ["Alice Moria", "Alice Mori2abc", "Alice Morrison"])
def test_grounding_does_not_strip_ordinary_surname_letters(printed):
    author = PaperAuthor(author_id=1, given="Alice", family="Mori", affiliation="")
    assert grounded_authors_in_context([author], printed) == ([], [author])


@pytest.mark.parametrize("source", ["detected_title", "llm"])
def test_appendix_repair_preserves_article_titles_after_citation_sidebar(source):
    title = PaperSection(2, "A framework for studying communities", 1, 0, CanonicalSection.TITLE)
    title.classification_source = source
    sections = [
        PaperSection(1, "CITATION", 1, 0, CanonicalSection.REFERENCES),
        title,
        PaperSection(3, "Introduction", 1, 0, CanonicalSection.INTRODUCTION),
    ]
    assert 2 not in repair_appendix_hierarchy(sections)
    assert title.section_type == CanonicalSection.TITLE
    assert title.classification_source == source
    assert (title.level, title.parent_section_id) == (1, 0)


def test_citation_sidebar_cannot_anchor_an_untyped_a_heading_as_appendix():
    heading = PaperSection(2, "A framework for studying communities", 1, 0)
    sections = [
        PaperSection(1, "CITATION", 1, 0, CanonicalSection.REFERENCES),
        heading,
        PaperSection(3, "Introduction", 1, 0, CanonicalSection.INTRODUCTION),
    ]
    assert 2 not in repair_appendix_hierarchy(sections)
    assert heading.section_type != CanonicalSection.APPENDIX


@pytest.mark.parametrize("location", ["header", "footer", "header_region"])
def test_bare_spaced_doi_uses_furniture_evidence_and_preserves_raw_spelling(location):
    raw = "10. 4321/example.26-7-s12"
    text = f"Clinical Reports 2026 Vol 26, No 7: s12 {raw}"
    contents = _contents([], headers=[text] if location == "header" else [])
    if location == "footer":
        contents.detected_footers = [text]
    elif location == "header_region":
        contents = _contents([PaperSection(1, "Root", 0, None)], [(1, text, 1)])
        contents.sentences[0].region_meta["region_type"] = "header"
    selection = select_doi_candidates(collect_doi_candidates(contents))
    assert selection.selected is not None
    assert selection.selected.normalized == "10.4321/example.26-7-s12"
    assert selection.selected.raw == raw


@pytest.mark.parametrize(
    "text", ["10. 123/example.1", "10. 1234567890/example.1", "10. 4321 /example.1"]
)
def test_bare_spaced_doi_in_header_requires_intact_registrant_and_slash(text):
    assert collect_doi_candidates(_contents([], headers=[text])) == ()


def test_bare_spaced_doi_is_not_repaired_in_body_or_reference_text():
    contents = _contents(
        [
            PaperSection(1, "Results", 1, 0, CanonicalSection.RESULTS),
            PaperSection(2, "References", 1, 0, CanonicalSection.REFERENCES),
        ],
        [(1, "See section 10. 4321/2026 for details.", 1), (2, "10. 4321/other.1", 5)],
    )
    assert collect_doi_candidates(contents) == ()


def test_reference_doi_in_header_cannot_become_article_identity():
    contents = _contents([], headers=["Reference DOI: 10. 4321/other.1"])
    selection = select_doi_candidates(collect_doi_candidates(contents))
    assert selection.selected is None


def test_bibliography_keeps_adjacent_translated_and_continuation_sections():
    contents = _contents(
        [
            PaperSection(1, "Discussion", 1, 0, CanonicalSection.DISCUSSION),
            PaperSection(2, "Referencias", 1, 0, CanonicalSection.REFERENCES),
            PaperSection(3, "References", 1, 0, CanonicalSection.REFERENCES),
        ],
        [
            (1, "Body prose.", 8),
            (2, "Smith, A. (2020). First source.", 9),
            (3, "Doe, B. (2021). Second source.", 10),
        ],
    )
    contents.sections[2].header_is_synthetic = True
    assert RefLocator(contents).collect_reference_rows()["text_id"].tolist() == [2, 3]


def test_bibliography_span_includes_subsections_but_excludes_earlier_footnote_and_tail():
    contents = _contents(
        [
            PaperSection(1, "References", 1, 0, CanonicalSection.REFERENCES),
            PaperSection(2, "Discussion", 1, 0, CanonicalSection.DISCUSSION),
            PaperSection(3, "List of References", 1, 0),
            PaperSection(4, "Books", 1, 0),
            PaperSection(5, "Articles and Research Papers", 1, 0),
            PaperSection(6, "Cases", 1, 0),
            PaperSection(7, "Acknowledgments", 1, 0, CanonicalSection.ACKNOWLEDGMENT),
        ],
        [
            (1, "Smith, A. (1999). A body footnote.", 2),
            (2, "More body prose.", 3),
            (4, "1. Jones, B. (2020). A book.", 4),
            (5, "1. Doe, C. (2021). A paper.", 5),
            (6, "1. State v. Example (2022).", 6),
            (7, "We thank our colleagues.", 6),
        ],
    )
    contents.sections[0].header_is_synthetic = True
    assert RefLocator(contents).collect_reference_rows()["text_id"].tolist() == [3, 4, 5]


def test_bibliography_does_not_cross_body_between_two_reference_lists():
    contents = _contents(
        [
            PaperSection(1, "References", 1, 0, CanonicalSection.REFERENCES),
            PaperSection(2, "Introduction", 1, 0, CanonicalSection.INTRODUCTION),
            PaperSection(3, "Bibliography", 1, 0, CanonicalSection.REFERENCES),
        ],
        [
            (1, "Smith, A. (2000). First list.", 2),
            (2, "A different article begins.", 3),
            (3, "Doe, B. (2001). Second list.", 5),
        ],
    )
    assert RefLocator(contents).collect_reference_rows()["text_id"].tolist() == [3]


@pytest.mark.parametrize("header", ["Literaturverzeichnis", "引用文献", "6. Referências"])
def test_printed_bibliography_heading_does_not_need_layout_or_classifier_hints(header):
    contents = _contents(
        [PaperSection(1, header, 1, 0, CanonicalSection.UNKNOWN)],
        [(1, "Smith, A. (2020). A reference.", 4)],
    )
    assert RefLocator(contents).collect_reference_rows()["text_id"].tolist() == [1]
    assert contents.sections[0].section_type == CanonicalSection.REFERENCES


def test_bibliography_owns_unlabelled_children_but_stops_at_typed_body_section():
    contents = _contents(
        [
            PaperSection(1, "References", 1, 0, CanonicalSection.REFERENCES),
            PaperSection(2, "Primary sources", 2, 1, CanonicalSection.UNKNOWN),
            PaperSection(3, "Further discussion", 2, 1, CanonicalSection.DISCUSSION),
        ],
        [
            (2, "Smith, A. (2020). A reference.", 4),
            (3, "Body prose after the bibliography.", 5),
        ],
    )
    assert RefLocator(contents).collect_reference_rows()["text_id"].tolist() == [1]
