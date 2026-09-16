"""Bibliography ownership across translated headings and bounded subsections."""

import pytest

from bibr.extract.ref_locator import RefLocator
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence


def _contents(specs):
    """Build independent source sections, with stable IDs and reading order."""
    sections = []
    sentences = []
    for section_id, header, kind, parent, page, texts in specs:
        sections.append(
            PaperSection(
                section_id=section_id,
                header=header,
                level=2 if parent is not None else 1,
                parent_section_id=parent,
                section_type=kind,
            )
        )
        for text in texts:
            text_id = len(sentences) + 1
            sentences.append(
                PaperSentence(
                    text_id=text_id,
                    text=text,
                    section_id=section_id,
                    paragraph_id=text_id,
                    page_number=page,
                )
            )
    return PaperContents(sentences, sections, [], [], {})


FIRST = "1. Meyer A. A study of river sediments. Journal of Hydrology. 2018;7:10-20."
SECOND = "2. Becker B. Field measurements of river flow. Water Research. 2019;8:21-30."
THIRD = "3. Chen C. Estimating seasonal rainfall. Climate Research. 2020;9:31-40."


@pytest.mark.parametrize("header", ["Referencias", "Bibliografía", "参考文献", "Список литературы"])
def test_printed_translated_heading_and_synthetic_english_section_share_ownership(header):
    contents = _contents(
        [
            (1, "Discussion", CanonicalSection.DISCUSSION, None, 3, ["Body text."]),
            (2, header, CanonicalSection.UNKNOWN, None, 4, [FIRST]),
            (3, "References", CanonicalSection.REFERENCES, None, 5, [SECOND, THIRD]),
        ]
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert rows.text.tolist() == [FIRST, SECOND, THIRD]
    assert rows.text_id.tolist() == [2, 3, 4]
    assert contents.sections[1].section_type == CanonicalSection.REFERENCES
    assert contents.reference_boundary_reason_flags == ["adjacent_bibliography_sections"]


def test_unknown_translated_heading_is_sufficient_without_layout_or_model():
    contents = _contents(
        [(1, "Literaturverzeichnis", CanonicalSection.UNKNOWN, None, 4, [FIRST, SECOND])]
    )

    assert RefLocator(contents).collect_reference_rows().text.tolist() == [FIRST, SECOND]


def test_explicit_page_continuation_is_joined_in_source_order():
    contents = _contents(
        [
            (1, "References", CanonicalSection.REFERENCES, None, 4, [FIRST, SECOND]),
            (2, "References (continued)", CanonicalSection.UNKNOWN, None, 5, [THIRD]),
        ]
    )

    assert RefLocator(contents).collect_reference_rows().text.tolist() == [FIRST, SECOND, THIRD]


def test_explicit_continuation_page_may_contain_only_a_citation_tail():
    tail = "Journal of Hydrology, volume 7, pages 10–20."
    contents = _contents(
        [
            (1, "References", CanonicalSection.REFERENCES, None, 4, [FIRST, SECOND]),
            (2, "References (continued)", CanonicalSection.UNKNOWN, None, 5, [tail]),
        ]
    )

    assert RefLocator(contents).collect_reference_rows().text.tolist() == [FIRST, SECOND, tail]


def test_explicit_bibliography_child_can_restart_numbering():
    restarted = "1. Rivera D. A regional archive of rainfall. University Press. 2021."
    contents = _contents(
        [
            (1, "References", CanonicalSection.REFERENCES, None, 4, [FIRST, SECOND]),
            (2, "Books", CanonicalSection.UNKNOWN, 1, 5, [restarted]),
            (3, "Appendix", CanonicalSection.APPENDIX, None, 5, ["Supplementary results."]),
        ]
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert rows.text.tolist() == [FIRST, SECOND, restarted]
    assert contents.sections[1].section_type == CanonicalSection.REFERENCES
    assert contents.sections[2].section_type == CanonicalSection.APPENDIX


@pytest.mark.parametrize(
    ("header", "kind"),
    [
        ("Notes", CanonicalSection.FOOTNOTE),
        ("Appendix", CanonicalSection.APPENDIX),
        ("Authors and Affiliations", CanonicalSection.UNKNOWN),
        ("Acknowledgments", CanonicalSection.ACKNOWLEDGMENT),
    ],
)
def test_adjacent_non_bibliography_section_is_not_claimed_even_with_citations(header, kind):
    contents = _contents(
        [
            (1, "References", CanonicalSection.REFERENCES, None, 4, [FIRST, SECOND]),
            (2, header, kind, 1, 4, [THIRD]),
        ]
    )

    assert RefLocator(contents).collect_reference_rows().text.tolist() == [FIRST, SECOND]


def test_numbering_without_bibliography_ancestry_does_not_establish_ownership():
    contents = _contents(
        [
            (1, "References", CanonicalSection.REFERENCES, None, 4, [FIRST, SECOND]),
            (2, "Historical notes", CanonicalSection.UNKNOWN, None, 4, [THIRD]),
        ]
    )

    assert RefLocator(contents).collect_reference_rows().text.tolist() == [FIRST, SECOND]


def test_repeated_heading_does_not_join_bibliographies_across_a_body_section():
    contents = _contents(
        [
            (1, "References", CanonicalSection.REFERENCES, None, 4, [FIRST, SECOND]),
            (2, "Supplementary methods", CanonicalSection.METHODS, None, 5, ["Body text."]),
            (3, "References", CanonicalSection.REFERENCES, None, 5, [THIRD]),
        ]
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert rows.text.tolist() == [THIRD]
    assert rows.section_id.tolist() == [3]


def test_last_bibliography_wins_when_an_earlier_list_has_a_translated_heading():
    contents = _contents(
        [
            (1, "References", CanonicalSection.REFERENCES, None, 4, [FIRST]),
            (2, "Bibliografía", CanonicalSection.UNKNOWN, None, 4, [SECOND]),
            (3, "Supplementary methods", CanonicalSection.METHODS, None, 5, ["Body text."]),
            (4, "References", CanonicalSection.REFERENCES, None, 5, [THIRD]),
        ]
    )

    assert RefLocator(contents).collect_reference_rows().text.tolist() == [THIRD]


def test_repeated_heading_does_not_bridge_distant_pages():
    contents = _contents(
        [
            (1, "Referencias", CanonicalSection.REFERENCES, None, 4, [FIRST, SECOND]),
            (2, "References", CanonicalSection.REFERENCES, None, 9, [THIRD]),
        ]
    )

    assert RefLocator(contents).collect_reference_rows().text.tolist() == [THIRD]


def test_terminal_row_inside_a_reference_section_prevents_backward_join():
    contents = _contents(
        [
            (1, "Referencias", CanonicalSection.REFERENCES, None, 4, [FIRST, "Author information"]),
            (2, "References", CanonicalSection.REFERENCES, None, 4, [SECOND, THIRD]),
        ]
    )

    assert RefLocator(contents).collect_reference_rows().text.tolist() == [SECOND, THIRD]


def test_terminal_spill_is_still_trimmed_after_joining_sections():
    contents = _contents(
        [
            (1, "Referencias", CanonicalSection.UNKNOWN, None, 4, [FIRST]),
            (2, "References", CanonicalSection.REFERENCES, None, 5, [SECOND, "Author information"]),
        ]
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert rows.text.tolist() == [FIRST, SECOND]
    assert "terminal_boundary_trimmed" in contents.reference_boundary_reason_flags


def test_header_merely_mentioning_references_does_not_override_canonical_section():
    contents = _contents(
        [
            (1, "Sources", CanonicalSection.REFERENCES, None, 4, [FIRST, SECOND]),
            (2, "References to earlier experiments", CanonicalSection.DISCUSSION, None, 5, [THIRD]),
        ]
    )

    assert RefLocator(contents).collect_reference_rows().text.tolist() == [FIRST, SECOND]
