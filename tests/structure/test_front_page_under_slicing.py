"""Page slicing must not disable every front-matter heuristic.

OCR pads ``json_result`` with empty pages so a list index stays the absolute
PDF page — export provenance depends on that. The cost was that every
front-matter heuristic phrased as ``page_number == 1`` silently never fired
under ``--pages 5-12`` / ``chew(pages=…)`` / serve ``start_page``: the title
was never captured, the running-header keep-first-occurrence exemption was
defeated so the printed title got demoted as furniture, byline affiliation
markers went unstripped, and the page-1 decoration filter went inert.

``PDFParser`` now takes ``first_page_index`` and asks ``_is_front_page``.
"""

import pandas as pd
import pytest

from bibr.extract.core_metadata import _front_page
from bibr.structure.pdf_parser import PDFParser

TITLE = "Attention Is Not All You Need"


def _page(*regions):
    return list(regions)


def _heading(text, label="doc_title"):
    return {"label": label, "content": text, "bbox_2d": [0, 0, 100, 20]}


def _text(text):
    return {"label": "text", "content": text, "bbox_2d": [0, 30, 100, 60]}


def _sliced_regions(first_index: int):
    """Regions as OCR hands them over for ``--pages`` starting at first_index."""
    pages = [[] for _ in range(first_index)]
    pages.append(_page(_heading(TITLE), _text("Body text begins here.")))
    # The journal reprints the title as a running head on the next page.
    pages.append(_page(_heading(TITLE), _text("More body text follows.")))
    return pages


def test_title_is_detected_on_the_first_processed_page():
    parser = PDFParser(_sliced_regions(4), first_page_index=4)
    parser.parse()

    assert parser._detected_title == TITLE


def test_title_detection_is_lost_without_the_slice_offset():
    """Control: the same regions with the default offset reproduce the bug."""
    parser = PDFParser(_sliced_regions(4))
    parser.parse()

    assert parser._detected_title is None


def test_unsliced_documents_are_unchanged():
    parser = PDFParser(_sliced_regions(0), first_page_index=0)
    parser.parse()

    assert parser._detected_title == TITLE


def test_first_occurrence_of_a_repeated_title_is_not_demoted_as_furniture():
    """The keep-first-occurrence exemption must follow the slice."""
    parser = PDFParser(_sliced_regions(4), first_page_index=4)
    parser._mark_running_headers()

    # Page index 4 is the first processed page: its title survives, the
    # page-5 reprint is demoted.
    assert (4, 0) not in parser._running_header_regions
    assert (5, 0) in parser._running_header_regions


def test_first_occurrence_is_demoted_without_the_slice_offset():
    """Control: unexempted, the printed title itself is deleted at parse."""
    parser = PDFParser(_sliced_regions(4))
    parser._mark_running_headers()

    assert (4, 0) in parser._running_header_regions


@pytest.mark.parametrize("first_index", [0, 1, 7])
def test_is_front_page_tracks_the_slice(first_index):
    parser = PDFParser([], first_page_index=first_index)

    assert parser._is_front_page(first_index + 1)
    assert not parser._is_front_page(first_index + 2)


def test_negative_first_page_index_is_clamped():
    assert PDFParser([], first_page_index=-3)._first_page_index == 0


def test_affiliation_markers_are_stripped_on_the_sliced_front_page():
    parser = PDFParser([], first_page_index=4)
    parser._current_section_id = 0
    parser._handle_content("Jane Doe^{1,2} and John Roe^{3}", 5, bbox=[0, 0, 1, 1])
    parser._flush_carry_over()

    assert [e.text for e in parser.assembler.entries] == ["Jane Doe and John Roe"]


class TestFrontPageHelper:
    """``core_metadata._front_page`` — the same notion for the extract stage."""

    def test_unsliced_is_page_one(self):
        assert _front_page(pd.DataFrame({"page_number": [1, 2, 3]})) == 1

    def test_sliced_anchors_on_the_lowest_page(self):
        assert _front_page(pd.DataFrame({"page_number": [5, 6, 7]})) == 5

    def test_missing_column_falls_back_to_one(self):
        assert _front_page(pd.DataFrame({"text": ["a"]})) == 1

    def test_empty_frame_falls_back_to_one(self):
        assert _front_page(pd.DataFrame({"page_number": []})) == 1
