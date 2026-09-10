"""Tests for the PDF-outline (bookmarks) extractor and title→heading matcher.

The extractor is ported from Docling's ``extract_outline_from_pdfium``; the
matcher ports Docling's bookmark cascade *semantics* (marker stripping, fuzzy
match, page constraint, greedy one-to-one, contiguous level compression).
None of the fixtures ships an embedded outline, so the pdfium walk is exercised
via lightweight mock objects; ``extract_pdf_outline`` is smoke-tested against a
real (outline-less) fixture.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2.raw as pdfium_c
import pytest

from bibr.input.pdf_outline import (
    HeadingRef,
    OutlineItem,
    _walk_pdfium_outline,
    extract_pdf_outline,
    match_outline_to_headings,
)

_FIXTURES = Path(__file__).parent.parent / "fixtures"


# --------------------------------------------------------------------------- matcher


def test_exact_match_single_level():
    outline = [OutlineItem(title="Introduction", level=0, page_no=1)]
    headings = [HeadingRef(text="Introduction", page_no=1)]
    assert match_outline_to_headings(outline, headings) == {0: 1}


def test_fuzzy_match_above_threshold():
    # Minor OCR-style typo; similarity stays > 0.8.
    outline = [OutlineItem(title="Materials and Methods", level=0, page_no=2)]
    headings = [HeadingRef(text="Materials and Methads", page_no=2)]
    assert match_outline_to_headings(outline, headings) == {0: 1}


def test_fuzzy_match_below_threshold_rejected():
    outline = [OutlineItem(title="Introduction", level=0, page_no=1)]
    headings = [HeadingRef(text="Acknowledgements", page_no=1)]
    assert match_outline_to_headings(outline, headings) == {}


def test_page_constraint_rejects_far_page():
    # Titles match perfectly but the heading sits many pages from the bookmark.
    outline = [OutlineItem(title="Results", level=0, page_no=2)]
    headings = [HeadingRef(text="Results", page_no=9)]
    assert match_outline_to_headings(outline, headings) == {}


def test_page_tolerance_allows_off_by_one():
    outline = [OutlineItem(title="Results", level=0, page_no=2)]
    headings = [HeadingRef(text="Results", page_no=3)]
    assert match_outline_to_headings(outline, headings) == {0: 1}


def test_marker_stripping_matches_numbered_heading():
    # Bookmark carries no marker; the on-page heading carries "2.1".
    outline = [OutlineItem(title="Participants", level=0, page_no=3)]
    headings = [HeadingRef(text="2.1 Participants", page_no=3)]
    assert match_outline_to_headings(outline, headings) == {0: 1}


def test_roman_and_letter_markers_stripped():
    outline = [
        OutlineItem(title="Methods", level=0, page_no=1),
        OutlineItem(title="Analysis", level=1, page_no=1),
    ]
    headings = [
        HeadingRef(text="IV. Methods", page_no=1),
        HeadingRef(text="A. Analysis", page_no=1),
    ]
    assert match_outline_to_headings(outline, headings) == {0: 1, 1: 2}


def test_depth_compression_to_contiguous_levels():
    # Raw bookmark depths 0 and 2 (a level skipped) compress to 1 and 2.
    outline = [
        OutlineItem(title="Part One", level=0, page_no=1),
        OutlineItem(title="Deep Subsection", level=2, page_no=1),
    ]
    headings = [
        HeadingRef(text="Part One", page_no=1),
        HeadingRef(text="Deep Subsection", page_no=1),
    ]
    assert match_outline_to_headings(outline, headings) == {0: 1, 1: 2}


def test_greedy_one_to_one_distinct_claims():
    # Two bookmarks, two candidate headings with identical text on the same
    # page: each bookmark claims a distinct heading (document order), and the
    # y-position tie-break steers each to its nearest heading.
    outline = [
        OutlineItem(title="Study", level=0, page_no=1, y_top=0.1),
        OutlineItem(title="Study", level=1, page_no=1, y_top=0.6),
    ]
    headings = [
        HeadingRef(text="Study", page_no=1, y_top=0.12),
        HeadingRef(text="Study", page_no=1, y_top=0.58),
    ]
    assert match_outline_to_headings(outline, headings) == {0: 1, 1: 2}


def test_tie_break_prefers_target_page_over_duplicate():
    # Two headings with identical text on pages 4 and 5, both within page
    # tolerance of a page-5 bookmark. y_top is page-relative, so comparing it
    # across pages is meaningless; the page-proximity tie-break must claim the
    # page-5 heading (index 1), not the page-4 one.
    outline = [OutlineItem(title="Results", level=0, page_no=5, y_top=0.30)]
    headings = [
        HeadingRef(text="Results", page_no=4, y_top=0.50),
        HeadingRef(text="Results", page_no=5, y_top=0.50),
    ]
    assert match_outline_to_headings(outline, headings) == {1: 1}


def test_tie_break_page_beats_smaller_y_distance():
    # Adversarial: the WRONG-page (page 4) duplicate is vertically closer to the
    # bookmark than the correct page-5 heading. A y-only tie-break (the old
    # behavior) would wrongly pick page 4; the page-first tuple tie-break picks
    # the bookmark's target page 5.
    outline = [OutlineItem(title="Results", level=0, page_no=5, y_top=0.30)]
    headings = [
        HeadingRef(text="Results", page_no=4, y_top=0.31),  # y-closer, wrong page
        HeadingRef(text="Results", page_no=5, y_top=0.60),  # correct page, y-far
    ]
    assert match_outline_to_headings(outline, headings) == {1: 1}


def test_containment_boost_for_truncated_bookmark():
    # A truncated bookmark is contained in the full heading; the containment
    # boost lifts an otherwise sub-threshold ratio over the line.
    outline = [
        OutlineItem(
            title="General Discussion",
            level=0,
            page_no=5,
        )
    ]
    headings = [
        HeadingRef(
            text="General Discussion and Broader Theoretical Implications",
            page_no=5,
        )
    ]
    assert match_outline_to_headings(outline, headings) == {0: 1}


def test_cross_page_bookmark_needs_stronger_match():
    # No page constraint (page_no None) raises the threshold; a weak fuzzy
    # match that would pass at 0.8 is rejected at 0.9.
    outline = [OutlineItem(title="Discusion", level=0, page_no=None)]
    headings = [HeadingRef(text="Conclusions", page_no=4)]
    assert match_outline_to_headings(outline, headings) == {}


def test_empty_inputs():
    assert match_outline_to_headings([], [HeadingRef(text="X", page_no=1)]) == {}
    assert match_outline_to_headings([OutlineItem("X", 0, 1)], []) == {}


# --------------------------------------------------------------------------- extractor mocks


class _FakeDest:
    def __init__(self, index, view):
        self._index = index
        self._view = view

    def get_index(self):
        return self._index

    def get_view(self):
        return self._view


class _FakeBookmark:
    def __init__(self, title, level, dest):
        self._title = title
        self.level = level
        self._dest = dest

    def get_title(self):
        return self._title

    def get_dest(self):
        return self._dest


class _FakePage:
    def __init__(self, height):
        self._height = height
        self.closed = False

    def get_height(self):
        return self._height

    def close(self):
        self.closed = True


class _FakeDoc:
    def __init__(self, toc, page_height=1000.0):
        self._toc = toc
        self._page_height = page_height
        self.opened_pages: list[_FakePage] = []

    def get_toc(self):
        return list(self._toc)

    def __getitem__(self, index):
        page = _FakePage(self._page_height)
        self.opened_pages.append(page)
        return page


def test_walk_extracts_title_level_page_and_ytop():
    # XYZ view: pos = [x, y_pdf, zoom]; top index 1. page_height 1000, y_pdf 900
    # -> top-left fraction = 1 - 900/1000 = 0.1.
    dest = _FakeDest(0, (pdfium_c.PDFDEST_VIEW_XYZ, [72.0, 900.0, 0.0]))
    doc = _FakeDoc([_FakeBookmark("Introduction", 0, dest)])
    items = _walk_pdfium_outline(doc)
    assert len(items) == 1
    it = items[0]
    assert it.title == "Introduction"
    assert it.level == 0
    assert it.page_no == 1  # 1-based
    assert it.y_top == pytest.approx(0.1, abs=1e-6)
    # The transiently opened page is always closed.
    assert all(p.closed for p in doc.opened_pages)


def test_walk_skips_empty_titles():
    dest = _FakeDest(0, (pdfium_c.PDFDEST_VIEW_XYZ, [0.0, 500.0, 0.0]))
    doc = _FakeDoc(
        [
            _FakeBookmark("   ", 0, dest),
            _FakeBookmark("Real", 1, dest),
        ]
    )
    items = _walk_pdfium_outline(doc)
    assert [it.title for it in items] == ["Real"]


def test_walk_view_without_top_leaves_ytop_none():
    # FIT view carries no vertical position; y_top stays None but page is set.
    dest = _FakeDest(2, (pdfium_c.PDFDEST_VIEW_FIT, []))
    doc = _FakeDoc([_FakeBookmark("Chapter", 0, dest)])
    items = _walk_pdfium_outline(doc)
    assert items[0].page_no == 3
    assert items[0].y_top is None
    # No page was opened because there was no vertical coordinate to convert.
    assert doc.opened_pages == []


def test_walk_no_dest_yields_no_page():
    doc = _FakeDoc([_FakeBookmark("Orphan", 0, None)])
    items = _walk_pdfium_outline(doc)
    assert items[0].page_no is None
    assert items[0].y_top is None


def test_walk_fith_view_uses_index_zero():
    # FITH view: pos = [y]; top index 0.
    dest = _FakeDest(0, (pdfium_c.PDFDEST_VIEW_FITH, [250.0]))
    doc = _FakeDoc([_FakeBookmark("Sec", 0, dest)], page_height=1000.0)
    items = _walk_pdfium_outline(doc)
    assert items[0].y_top == pytest.approx(0.75, abs=1e-6)


def test_extract_pdf_outline_no_outline_returns_empty():
    # Real pdfium open of an outline-less fixture returns [].
    pdf_bytes = (_FIXTURES / "native_text_sample.pdf").read_bytes()
    assert extract_pdf_outline(pdf_bytes) == []


def test_extract_pdf_outline_defensive_on_bad_bytes():
    # Corrupt input must never raise — best-effort returns [].
    assert extract_pdf_outline(b"not a pdf") == []
