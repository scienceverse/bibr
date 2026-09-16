"""Deferred source rows must not erase the selected bibliography's earlier run."""

import pytest

from bibr.extract.ref_locator import RefLocator
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence

FIRST = "Meyer, A. (2018). River sediments. Journal of Hydrology 7:10–20."
SECOND = "Becker, B. (2019). River flow. Water Research 8:21–30."
TAIL = "Watson, M. (2020). Seasonal rainfall. Climate Research 9:31–40."


def _contents(tail, *, translated=False):
    sections = [
        PaperSection(1, "Results", 1, None, section_type=CanonicalSection.RESULTS),
        PaperSection(2, "References", 1, None, section_type=CanonicalSection.REFERENCES),
    ]
    rows = [(2, 7, FIRST), (2, 8, SECOND), (1, 6, "Deferred table caption."), (2, 9, tail)]
    if translated:
        sections.append(
            PaperSection(3, "Referencias", 1, None, section_type=CanonicalSection.UNKNOWN)
        )
        rows.insert(0, (3, 7, "Chen, C. (2017). Rainfall measurement. Climate Research 5:1–9."))
    sentences = [
        PaperSentence(i, text, section_id, i, page_number=page)
        for i, (section_id, page, text) in enumerate(rows, 1)
    ]
    return PaperContents(sentences, sections, [], [], {})


@pytest.mark.parametrize("tail", [TAIL, "Professor Watson studies seasonal rainfall."])
def test_same_section_identity_recovers_earlier_run_without_deferred_body(tail):
    contents = _contents(tail)

    rows = RefLocator(contents).collect_reference_rows()

    assert rows.text.tolist() == [FIRST, SECOND, tail]
    assert rows.section_id.tolist() == [2, 2, 2]
    assert "same_section_disjoint_runs" in contents.reference_boundary_reason_flags


def test_disjoint_owner_still_joins_adjacent_translated_heading():
    contents = _contents(TAIL, translated=True)

    rows = RefLocator(contents).collect_reference_rows()

    assert rows.section_id.tolist() == [3, 2, 2, 2]
    assert "Deferred table caption." not in rows.text.tolist()
    assert "adjacent_bibliography_sections" in contents.reference_boundary_reason_flags


def test_distinct_section_identity_with_same_heading_remains_separate():
    contents = _contents(TAIL)
    contents.sections.append(
        PaperSection(4, "References", 1, None, section_type=CanonicalSection.REFERENCES)
    )
    contents.sentences[-1].section_id = 4

    rows = RefLocator(contents).collect_reference_rows()

    assert rows.text.tolist() == [TAIL]
    assert rows.section_id.tolist() == [4]


def test_terminal_row_is_still_trimmed_inside_disjoint_owner():
    contents = _contents("Author information")

    rows = RefLocator(contents).collect_reference_rows()

    assert rows.text.tolist() == [FIRST, SECOND]
    assert "terminal_boundary_trimmed" in contents.reference_boundary_reason_flags
