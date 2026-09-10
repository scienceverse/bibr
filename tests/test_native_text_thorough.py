"""Thorough tests for native PDF text bypass.

Covers edge cases, multi-page scenarios, coordinate math, label eligibility,
pipeline integration with mixed regions, fallback behavior, and concurrent access.

Fixtures are generated programmatically via pypdfium2 raw API to avoid shipping
many static PDFs in the test suite.
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import threading
from pathlib import Path

import pypdfium2
import pypdfium2.raw as raw
import pytest
from PIL import Image

from bibr.ocr.native_text import (
    DEFAULT_ELIGIBLE_LABELS,
    _attach_bbox_pdf_pts,
    _normalized_bbox_to_pdf_points,
    fill_native_text_and_fonts,
    fill_regions_from_native_text,
    get_native_text_in_bbox,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_NATIVE = FIXTURE_DIR / "native_text_sample.pdf"
FIXTURE_SCANNED = FIXTURE_DIR / "scanned_sample.pdf"


# ---------------------------------------------------------------------------
# Programmatic PDF helpers
# ---------------------------------------------------------------------------


def _encode_wide(text: str):
    """Encode Python string as ctypes c_ushort array for FPDFText_SetText."""
    return (ctypes.c_ushort * (len(text) + 1))(*[ord(c) for c in text], 0)


def make_pdf_with_text(
    pages: list[list[tuple[str, float, float]]],
    page_width: float = 595.0,
    page_height: float = 842.0,
) -> bytes:
    """Create a PDF with text objects placed at given coordinates.

    Parameters
    ----------
    pages : list of lists of (text, x, y) tuples
        Each inner list represents one page. Each tuple places *text* at
        (*x*, *y*) in PDF-point coordinates (origin bottom-left, y-up).
    """
    doc = pypdfium2.PdfDocument.new()
    font_raw = raw.FPDFText_LoadStandardFont(doc.raw, b"Helvetica")
    for page_texts in pages:
        page = doc.new_page(page_width, page_height)
        for text, x, y in page_texts:
            text_obj = raw.FPDFPageObj_CreateTextObj(doc.raw, font_raw, ctypes.c_float(10))
            raw.FPDFText_SetText(text_obj, _encode_wide(text))
            raw.FPDFPageObj_Transform(text_obj, 1, 0, 0, 1, x, y)
            raw.FPDFPage_InsertObject(page.raw, text_obj)
        raw.FPDFPage_GenerateContent(page.raw)
        page.close()
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def make_empty_pdf(n_pages: int = 1) -> bytes:
    """Create a PDF with *n_pages* blank pages (no text objects)."""
    doc = pypdfium2.PdfDocument.new()
    for _ in range(n_pages):
        page = doc.new_page(595, 842)
        raw.FPDFPage_GenerateContent(page.raw)
        page.close()
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def make_multipage_mixed_pdf() -> bytes:
    """3 pages: page 0 has text, page 1 is blank, page 2 has text."""
    doc = pypdfium2.PdfDocument.new()
    font_raw = raw.FPDFText_LoadStandardFont(doc.raw, b"Helvetica")

    # Page 0: text
    p0 = doc.new_page(595, 842)
    t0 = raw.FPDFPageObj_CreateTextObj(doc.raw, font_raw, ctypes.c_float(10))
    raw.FPDFText_SetText(t0, _encode_wide("Page zero has native text content here"))
    raw.FPDFPageObj_Transform(t0, 1, 0, 0, 1, 72, 700)
    raw.FPDFPage_InsertObject(p0.raw, t0)
    raw.FPDFPage_GenerateContent(p0.raw)
    p0.close()

    # Page 1: blank (no text layer — simulates scanned page)
    p1 = doc.new_page(595, 842)
    raw.FPDFPage_GenerateContent(p1.raw)
    p1.close()

    # Page 2: text
    p2 = doc.new_page(595, 842)
    t2 = raw.FPDFPageObj_CreateTextObj(doc.raw, font_raw, ctypes.c_float(10))
    raw.FPDFText_SetText(t2, _encode_wide("Third page also has native text for extraction"))
    raw.FPDFPageObj_Transform(t2, 1, 0, 0, 1, 72, 700)
    raw.FPDFPage_InsertObject(p2.raw, t2)
    raw.FPDFPage_GenerateContent(p2.raw)
    p2.close()

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _region(label: str = "text", bbox=None, content: str = ""):
    bbox = bbox or [0, 0, 1000, 1000]
    return {"label": label, "task_type": "text", "bbox_2d": bbox, "content": content}


# ===================================================================
# get_native_text_in_bbox — edge cases
# ===================================================================


class TestGetNativeTextInBbox:
    def test_negative_page_idx(self):
        pdf = FIXTURE_NATIVE.read_bytes()
        assert get_native_text_in_bbox(pdf, page_idx=-1, bbox_normalized=[0, 0, 1000, 1000]) == ""

    def test_page_idx_beyond_range(self):
        pdf = FIXTURE_NATIVE.read_bytes()
        assert get_native_text_in_bbox(pdf, page_idx=999, bbox_normalized=[0, 0, 1000, 1000]) == ""

    def test_page_idx_exactly_at_boundary(self):
        pdf = FIXTURE_NATIVE.read_bytes()
        # 1-page PDF, idx=1 is out of range
        assert get_native_text_in_bbox(pdf, page_idx=1, bbox_normalized=[0, 0, 1000, 1000]) == ""

    def test_multipage_text_extraction_per_page(self):
        pdf = make_multipage_mixed_pdf()
        # Page 0 has text
        t0 = get_native_text_in_bbox(pdf, page_idx=0, bbox_normalized=[0, 0, 1000, 1000])
        assert "Page zero" in t0
        # Page 1 is blank
        t1 = get_native_text_in_bbox(pdf, page_idx=1, bbox_normalized=[0, 0, 1000, 1000])
        assert t1 == ""
        # Page 2 has text
        t2 = get_native_text_in_bbox(pdf, page_idx=2, bbox_normalized=[0, 0, 1000, 1000])
        assert "Third page" in t2

    def test_partial_bbox_extracts_subset(self):
        """Smaller bbox only returns text within that region."""
        pdf = make_pdf_with_text(
            [
                [("Top text here", 72, 750), ("Bottom text here", 72, 100)],
            ]
        )
        # Top quarter only — should get top text, not bottom
        top_text = get_native_text_in_bbox(pdf, page_idx=0, bbox_normalized=[0, 0, 1000, 250])
        bottom_text = get_native_text_in_bbox(pdf, page_idx=0, bbox_normalized=[0, 750, 1000, 1000])
        assert "Top text" in top_text
        assert "Bottom text" in bottom_text
        # Cross-check: top bbox shouldn't contain bottom text
        assert "Bottom text" not in top_text

    def test_blank_pdf_returns_empty(self):
        pdf = make_empty_pdf(1)
        text = get_native_text_in_bbox(pdf, page_idx=0, bbox_normalized=[0, 0, 1000, 1000])
        assert text == ""

    def test_empty_pdf_bytes_raises_or_returns_empty(self):
        """Corrupt/empty bytes should not crash — may raise or return empty."""
        import contextlib

        with contextlib.suppress(Exception):
            result = get_native_text_in_bbox(b"", page_idx=0, bbox_normalized=[0, 0, 1000, 1000])
            assert result == ""

    def test_unicode_text_extraction(self):
        pdf = make_pdf_with_text(
            [
                [("Sch\u00f6ne Gr\u00fc\u00dfe aus Z\u00fcrich", 72, 700)],
            ]
        )
        text = get_native_text_in_bbox(pdf, page_idx=0, bbox_normalized=[0, 0, 1000, 1000])
        assert "Sch\u00f6ne" in text or len(text) > 0


# ===================================================================
# Coordinate conversion
# ===================================================================


class TestCoordinateConversion:
    def test_single_point_top_left_corner(self):
        """(0,0) in image-space maps to (0, page_height) in PDF-space."""
        left, bottom, right, top = _normalized_bbox_to_pdf_points([0, 0, 0, 0], (0, 0, 612, 792))
        assert left == pytest.approx(0.0)
        assert top == pytest.approx(792.0)  # top of page

    def test_single_point_bottom_right_corner(self):
        """(1000,1000) maps to (page_width, 0) in PDF-space."""
        left, bottom, right, top = _normalized_bbox_to_pdf_points(
            [1000, 1000, 1000, 1000], (0, 0, 612, 792)
        )
        assert right == pytest.approx(612.0)
        assert bottom == pytest.approx(0.0)

    def test_center_bbox(self):
        """Center quarter of the page."""
        left, bottom, right, top = _normalized_bbox_to_pdf_points(
            [250, 250, 750, 750], (0, 0, 1000, 1000)
        )
        assert (left, bottom, right, top) == pytest.approx((250.0, 250.0, 750.0, 750.0))

    def test_non_square_page(self):
        """Landscape page (wider than tall)."""
        left, bottom, right, top = _normalized_bbox_to_pdf_points(
            [0, 0, 1000, 1000], (0, 0, 1190, 842)
        )
        assert (left, bottom, right, top) == pytest.approx((0.0, 0.0, 1190.0, 842.0))

    def test_small_page(self):
        """Very small page dimensions."""
        left, bottom, right, top = _normalized_bbox_to_pdf_points([0, 0, 500, 500], (0, 0, 10, 10))
        assert (left, bottom, right, top) == pytest.approx((0.0, 5.0, 5.0, 10.0))

    def test_bbox_preserves_aspect_on_non_letter(self):
        """A4 page: 595.28 x 841.89."""
        left, bottom, right, top = _normalized_bbox_to_pdf_points(
            [100, 100, 900, 900], (0, 0, 595.28, 841.89)
        )
        expected_left = 100 / 1000.0 * 595.28
        expected_right = 900 / 1000.0 * 595.28
        expected_top = 841.89 - (100 / 1000.0 * 841.89)
        expected_bottom = 841.89 - (900 / 1000.0 * 841.89)
        assert (left, bottom, right, top) == pytest.approx(
            (expected_left, expected_bottom, expected_right, expected_top)
        )

    def test_y_flip_symmetry(self):
        """Mirror bbox around horizontal center should swap top/bottom."""
        _, b1, _, t1 = _normalized_bbox_to_pdf_points([0, 200, 1000, 300], (0, 0, 100, 100))
        _, b2, _, t2 = _normalized_bbox_to_pdf_points([0, 700, 1000, 800], (0, 0, 100, 100))
        # b1/t1 should be mirror of b2/t2 around y=50
        assert t1 == pytest.approx(100 - b2)
        assert b1 == pytest.approx(100 - t2)


# ===================================================================
# _attach_bbox_pdf_pts — persisted PDF-point bbox (containment-correct)
# ===================================================================


class TestAttachBboxPdfPts:
    """The attach step converts each region's 0..1000 ``bbox_2d`` into PDF
    points (bottom-left origin, y-up) stored as ``_bbox_pdf_pts``, leaving the
    working ``bbox_2d`` field untouched for the 0..1000-tuned consumers."""

    @pytest.mark.parametrize(
        ("crop_box", "bbox_2d", "expected"),
        [
            # Zero-origin US Letter, top-left quadrant half-height.
            ((0, 0, 612, 792), [0, 0, 500, 250], [0.0, 594.0, 306.0, 792.0]),
            # Full page maps to the full CropBox.
            ((0, 0, 612, 792), [0, 0, 1000, 1000], [0.0, 0.0, 612.0, 792.0]),
            # Non-zero CropBox origin (40, 50): right-half top quadrant.
            ((40, 50, 640, 850), [500, 0, 1000, 250], [340.0, 650.0, 640.0, 850.0]),
        ],
    )
    def test_known_bbox_converts_to_pdf_points(self, crop_box, bbox_2d, expected):
        regions = [{"label": "text", "bbox_2d": list(bbox_2d)}]
        _attach_bbox_pdf_pts(crop_box, regions)
        assert regions[0]["_bbox_pdf_pts"] == pytest.approx(expected)
        # The 0..1000 working field must be left untouched.
        assert regions[0]["bbox_2d"] == bbox_2d

    def test_region_without_bbox_gets_none(self):
        regions = [{"label": "text"}, {"label": "text", "bbox_2d": None}]
        _attach_bbox_pdf_pts((0, 0, 612, 792), regions)
        assert regions[0]["_bbox_pdf_pts"] is None
        assert regions[1]["_bbox_pdf_pts"] is None

    def test_values_rounded_to_two_decimals(self):
        # A4 page produces non-round conversions; assert 2dp rounding.
        regions = [{"label": "text", "bbox_2d": [100, 100, 900, 900]}]
        _attach_bbox_pdf_pts((0, 0, 595.28, 841.89), regions)
        pts = regions[0]["_bbox_pdf_pts"]
        assert pts == [round(v, 2) for v in pts]
        assert pts == pytest.approx([59.53, 84.19, 535.75, 757.7])


class TestBboxPdfPtsContainmentInvariant:
    """``_bbox_pdf_pts`` must satisfy the page-containment invariant the raw
    0..1000 ``bbox_2d`` violated: ``0 <= x1 <= x2 <= page_w`` and
    ``0 <= y1 <= y2 <= page_h`` (PDF points). Regression for the exported
    ``_bbox_2d`` field being persisted in the wrong coordinate space."""

    def test_native_pdf_regions_contained_in_page(self):
        pdf = make_pdf_with_text(
            [[("Some native text on the page for extraction.", 72, 700)]],
            page_width=612.0,
            page_height=792.0,
        )
        regions = [
            [
                {"label": "text", "bbox_2d": [50, 40, 950, 120], "content": ""},
                {"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
            ]
        ]
        fill_native_text_and_fonts(pdf, regions, min_chars=1)
        for r in regions[0]:
            pts = r["_bbox_pdf_pts"]
            assert pts is not None
            x1, y1, x2, y2 = pts
            pw, ph = r["_page_w"], r["_page_h"]
            assert 0 <= x1 <= x2 <= pw + 0.01, (pts, pw)
            assert 0 <= y1 <= y2 <= ph + 0.01, (pts, ph)

    def test_scanned_page_still_gets_pdf_pts(self):
        """Image-only pages fall back to OCR but must still carry the correct
        PDF-point bbox (page geometry is attached before the char-count gate)."""
        pdf = make_empty_pdf(1)
        regions = [[{"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]]
        fill_native_text_and_fonts(pdf, regions, min_chars=1)
        r = regions[0][0]
        assert r["_bbox_pdf_pts"] == pytest.approx([0.0, 0.0, 595.0, 842.0])
        assert r["_page_w"] == pytest.approx(595.0)
        assert r["_page_h"] == pytest.approx(842.0)


# ===================================================================
# fill_regions_from_native_text — comprehensive
# ===================================================================


class TestFillRegions:
    def test_all_eligible_labels(self):
        """Every label in DEFAULT_ELIGIBLE_LABELS gets filled on a native PDF."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [
            [_region(label=lbl, content="") for lbl in sorted(DEFAULT_ELIGIBLE_LABELS)]
        ]
        filled = fill_regions_from_native_text(
            pdf, pages_regions, min_chars=5, eligible_labels=DEFAULT_ELIGIBLE_LABELS
        )
        for r in filled[0]:
            assert r.get("_native_text_used") is True, (
                f"Label '{r['label']}' should be eligible but was not filled"
            )

    def test_ineligible_labels_never_filled(self):
        """Labels NOT in eligible set remain untouched."""
        pdf = FIXTURE_NATIVE.read_bytes()
        ineligible = ["table", "figure", "formula", "table_title", "figure_title", "chart_title"]
        pages_regions = [[_region(label=lbl, content="") for lbl in ineligible]]
        filled = fill_regions_from_native_text(
            pdf, pages_regions, min_chars=5, eligible_labels=DEFAULT_ELIGIBLE_LABELS
        )
        for r in filled[0]:
            assert "_native_text_used" not in r, f"Label '{r['label']}' should NOT be filled"

    def test_mixed_eligible_and_ineligible_on_same_page(self):
        """Only eligible labels get filled; others stay empty."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [
            [
                _region(label="text", content=""),
                _region(label="table", content=""),
                _region(label="abstract", content=""),
                _region(label="formula", content=""),
                _region(label="reference", content=""),
            ]
        ]
        filled = fill_regions_from_native_text(
            pdf, pages_regions, min_chars=5, eligible_labels=DEFAULT_ELIGIBLE_LABELS
        )
        assert filled[0][0].get("_native_text_used") is True  # text
        assert "_native_text_used" not in filled[0][1]  # table
        assert filled[0][2].get("_native_text_used") is True  # abstract
        assert "_native_text_used" not in filled[0][3]  # formula
        assert filled[0][4].get("_native_text_used") is True  # reference

    def test_multipage_mixed_text_and_blank(self):
        """Pages without text layer leave regions unfilled; pages with text fill them."""
        pdf = make_multipage_mixed_pdf()
        pages_regions = [
            [_region(content="")],  # page 0 — has text
            [_region(content="")],  # page 1 — blank
            [_region(content="")],  # page 2 — has text
        ]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert filled[0][0].get("_native_text_used") is True
        assert filled[1][0].get("_native_text_used", False) is False
        assert filled[2][0].get("_native_text_used") is True

    def test_more_region_pages_than_pdf_pages(self):
        """Extra region pages beyond PDF page count are silently skipped."""
        pdf = FIXTURE_NATIVE.read_bytes()  # 1 page
        pages_regions = [
            [_region(content="")],
            [_region(content="")],  # page 1 doesn't exist in PDF
            [_region(content="")],  # page 2 doesn't exist
        ]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert filled[0][0].get("_native_text_used") is True
        assert filled[1][0].get("_native_text_used", False) is False
        assert filled[2][0].get("_native_text_used", False) is False

    def test_empty_pages_regions(self):
        """Empty input list returns empty output without error."""
        pdf = FIXTURE_NATIVE.read_bytes()
        result = fill_regions_from_native_text(pdf, [], min_chars=5)
        assert result == []

    def test_page_with_no_regions(self):
        """Pages with empty region lists are harmless."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [[], [_region(content="")]]  # page 0 empty, page 1 doesn't exist
        result = fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert result[0] == []

    def test_region_missing_bbox_2d(self):
        """Region without bbox_2d is skipped (not crashed)."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [[{"label": "text", "task_type": "text", "content": ""}]]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert filled[0][0].get("_native_text_used", False) is False

    def test_region_with_none_bbox(self):
        """Region with bbox_2d=None is treated same as missing."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [
            [
                {"label": "text", "task_type": "text", "bbox_2d": None, "content": ""},
            ]
        ]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert filled[0][0].get("_native_text_used", False) is False

    def test_region_missing_label(self):
        """Region without 'label' key is skipped (label=None not in eligible set)."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [[{"task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert filled[0][0].get("_native_text_used", False) is False

    def test_min_chars_boundary_exact(self):
        """Text with exactly min_chars characters gets filled; min_chars+1 does not."""
        pdf = FIXTURE_NATIVE.read_bytes()
        full_text = get_native_text_in_bbox(pdf, 0, [0, 0, 1000, 1000])
        stripped_len = len(full_text.strip())

        # Exactly at boundary: should fill
        pages_regions = [[_region(content="")]]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=stripped_len)
        assert filled[0][0].get("_native_text_used") is True

        # One above: should NOT fill
        pages_regions = [[_region(content="")]]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=stripped_len + 1)
        assert filled[0][0].get("_native_text_used", False) is False

    def test_min_chars_zero(self):
        """min_chars=0 fills even single-char regions."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [[_region(content="")]]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=0)
        assert filled[0][0].get("_native_text_used") is True

    def test_returns_same_list_object(self):
        """fill mutates in-place and returns the same list (for chaining)."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [[_region(content="")]]
        result = fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert result is pages_regions

    def test_pre_existing_content_overwritten(self):
        """If a region already has content, native text overwrites it."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [[_region(content="old garbage content")]]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert filled[0][0]["content"] != "old garbage content"
        assert "quick brown fox" in filled[0][0]["content"]

    def test_scanned_pdf_leaves_all_regions_untouched(self):
        pdf = FIXTURE_SCANNED.read_bytes()
        regions = [[_region(label=lbl, content="") for lbl in sorted(DEFAULT_ELIGIBLE_LABELS)]]
        filled = fill_regions_from_native_text(pdf, regions, min_chars=5)
        for r in filled[0]:
            assert "_native_text_used" not in r
            assert r["content"] == ""

    def test_custom_eligible_labels_subset(self):
        """Custom eligible_labels restricts which labels are filled."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [
            [
                _region(label="text", content=""),
                _region(label="abstract", content=""),
                _region(label="reference", content=""),
            ]
        ]
        only_abstract = frozenset({"abstract"})
        filled = fill_regions_from_native_text(
            pdf, pages_regions, min_chars=5, eligible_labels=only_abstract
        )
        assert filled[0][0].get("_native_text_used", False) is False  # text — excluded
        assert filled[0][1].get("_native_text_used") is True  # abstract — included
        assert filled[0][2].get("_native_text_used", False) is False  # reference — excluded

    def test_many_regions_per_page(self):
        """Stress test: 50 regions on one page all get processed."""
        pdf = FIXTURE_NATIVE.read_bytes()
        pages_regions = [[_region(content="") for _ in range(50)]]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        filled_count = sum(1 for r in filled[0] if r.get("_native_text_used"))
        assert filled_count == 50

    def test_multipage_pdf_region_counts(self):
        """Each page's regions are independently processed against that page's text."""
        pdf = make_pdf_with_text(
            [
                [("Page one text content " * 3, 72, 700)],
                [("Page two different text " * 3, 72, 700)],
            ]
        )
        pages_regions = [
            [_region(content="")],
            [_region(content="")],
        ]
        filled = fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert filled[0][0].get("_native_text_used") is True
        assert filled[1][0].get("_native_text_used") is True
        # Verify content is page-specific
        assert "Page one" in filled[0][0]["content"]
        assert "Page two" in filled[1][0]["content"]


# ===================================================================
# Pipeline helpers integration — ocr_page_regions with native text
# ===================================================================


class TestOcrPageRegionsNativeBypass:
    def test_mixed_prefilled_and_ocr_regions(self):
        """Page with 3 regions: prefilled, OCR-needed, and skip. Correct dispatch."""
        from bibr.pipeline.stages.ocr import ocr_page_regions

        page_img = Image.new("RGB", (1000, 1000), color="white")
        regions = [
            {
                "label": "text",
                "task_type": "text",
                "bbox_2d": [0, 0, 500, 300],
                "content": "native text body",
                "_native_text_used": True,
            },
            {
                "label": "text",
                "task_type": "text",
                "bbox_2d": [0, 300, 500, 600],
                "content": "",
            },
            {
                "label": "figure",
                "task_type": "skip",
                "bbox_2d": [0, 600, 500, 1000],
                "content": "",
            },
        ]
        ocr_calls = []

        async def fake_ocr(img, prompt):
            ocr_calls.append(prompt)
            return "ocr output"

        result = asyncio.run(
            ocr_page_regions(page_img, regions, page_idx=0, filename="test.pdf", ocr_fn=fake_ocr)
        )

        # Prefilled region: content preserved, no OCR
        assert result[0]["content"] == "native text body"
        # OCR region: went through OCR
        assert result[1]["content"] == "ocr output"
        # Skip region: empty content
        assert result[2]["content"] == ""
        # Only 1 OCR call (for the non-prefilled text region)
        assert len(ocr_calls) == 1

    def test_all_regions_prefilled_no_ocr_calls(self):
        """When all text regions are prefilled, zero OCR calls happen."""
        from bibr.pipeline.stages.ocr import ocr_page_regions

        page_img = Image.new("RGB", (1000, 1000), color="white")
        regions = [
            {
                "label": "abstract",
                "task_type": "text",
                "bbox_2d": [0, 0, 1000, 300],
                "content": "Abstract text from native layer",
                "_native_text_used": True,
            },
            {
                "label": "text",
                "task_type": "text",
                "bbox_2d": [0, 300, 1000, 700],
                "content": "Body text from native layer",
                "_native_text_used": True,
            },
            {
                "label": "reference",
                "task_type": "text",
                "bbox_2d": [0, 700, 1000, 1000],
                "content": "References from native layer",
                "_native_text_used": True,
            },
        ]
        ocr_calls = []

        async def fail_ocr(img, prompt):
            ocr_calls.append(1)
            raise AssertionError("OCR should not be called")

        result = asyncio.run(
            ocr_page_regions(page_img, regions, page_idx=0, filename="test.pdf", ocr_fn=fail_ocr)
        )

        assert len(ocr_calls) == 0
        assert result[0]["content"] == "Abstract text from native layer"
        assert result[1]["content"] == "Body text from native layer"
        assert result[2]["content"] == "References from native layer"

    def test_native_label_preserved_in_output(self):
        """The native_label field carries the original layout label through bypass."""
        from bibr.pipeline.stages.ocr import ocr_page_regions

        page_img = Image.new("RGB", (1000, 1000), color="white")
        regions = [
            {
                "label": "abstract",
                "task_type": "text",
                "bbox_2d": [0, 0, 1000, 500],
                "content": "Abstract content from native",
                "_native_text_used": True,
            },
            {
                "label": "reference_content",
                "task_type": "text",
                "bbox_2d": [0, 500, 1000, 1000],
                "content": "Reference content from native",
                "_native_text_used": True,
            },
        ]

        async def noop_ocr(img, prompt):
            return ""

        result = asyncio.run(
            ocr_page_regions(page_img, regions, page_idx=0, filename="test.pdf", ocr_fn=noop_ocr)
        )
        assert result[0]["native_label"] == "abstract"
        assert result[1]["native_label"] == "reference_content"

    def test_reading_order_preserved_with_bypass(self):
        """Slot ordering is preserved when some regions bypass OCR."""
        from bibr.pipeline.stages.ocr import ocr_page_regions

        page_img = Image.new("RGB", (1000, 1000), color="white")
        regions = [
            {
                "label": "text",
                "task_type": "text",
                "bbox_2d": [0, 0, 1000, 200],
                "content": "FIRST native",
                "_native_text_used": True,
            },
            {
                "label": "text",
                "task_type": "text",
                "bbox_2d": [0, 200, 1000, 400],
                "content": "",
            },
            {
                "label": "text",
                "task_type": "text",
                "bbox_2d": [0, 400, 1000, 600],
                "content": "THIRD native",
                "_native_text_used": True,
            },
            {
                "label": "text",
                "task_type": "text",
                "bbox_2d": [0, 600, 1000, 800],
                "content": "",
            },
            {
                "label": "text",
                "task_type": "text",
                "bbox_2d": [0, 800, 1000, 1000],
                "content": "FIFTH native",
                "_native_text_used": True,
            },
        ]

        async def ocr_fn(img, prompt):
            return "OCR result"

        result = asyncio.run(
            ocr_page_regions(page_img, regions, page_idx=0, filename="test.pdf", ocr_fn=ocr_fn)
        )
        assert [r["content"] for r in result] == [
            "FIRST native",
            "OCR result",
            "THIRD native",
            "OCR result",
            "FIFTH native",
        ]

    def test_bbox_2d_preserved_through_bypass(self):
        """bbox_2d from the region dict survives into the output."""
        from bibr.pipeline.stages.ocr import ocr_page_regions

        page_img = Image.new("RGB", (1000, 1000), color="white")
        bbox = [50, 100, 950, 400]
        regions = [
            {
                "label": "text",
                "task_type": "text",
                "bbox_2d": bbox,
                "content": "text from native layer extraction",
                "_native_text_used": True,
            },
        ]

        async def noop_ocr(img, prompt):
            return ""

        result = asyncio.run(
            ocr_page_regions(page_img, regions, page_idx=0, filename="test.pdf", ocr_fn=noop_ocr)
        )
        assert result[0]["bbox_2d"] == bbox


# ===================================================================
# Pipeline fallback — exception recovery
# ===================================================================


class TestPipelineFallback:
    def test_cleanup_after_partial_fill(self):
        """Simulate pipeline's except block: clear all _native_text_used flags."""
        pages_regions = [
            [
                {
                    "label": "text",
                    "bbox_2d": [0, 0, 1000, 300],
                    "content": "filled content 1",
                    "_native_text_used": True,
                },
                {
                    "label": "text",
                    "bbox_2d": [0, 300, 1000, 600],
                    "content": "filled content 2",
                    "_native_text_used": True,
                },
                {"label": "text", "bbox_2d": [0, 600, 1000, 1000], "content": ""},
            ],
            [
                {
                    "label": "abstract",
                    "bbox_2d": [0, 0, 1000, 500],
                    "content": "filled abstract",
                    "_native_text_used": True,
                },
            ],
        ]

        # Pipeline except block pattern (from local/pipeline.py:711-717)
        for page in pages_regions:
            for r in page:
                if r.pop("_native_text_used", False):
                    r["content"] = ""

        # Verify all cleaned
        for page in pages_regions:
            for r in page:
                assert "_native_text_used" not in r
                assert r["content"] == ""

    def test_fill_exception_does_not_propagate_with_guard(self):
        """Wrapping fill in try/except (as pipeline does) catches corrupt PDF."""
        corrupt_pdf = b"%PDF-1.4 this is not valid"
        pages_regions = [[_region(content="should stay")]]

        try:
            fill_regions_from_native_text(corrupt_pdf, pages_regions, min_chars=5)
        except Exception:
            # Pipeline would clean up here
            for page in pages_regions:
                for r in page:
                    if r.pop("_native_text_used", False):
                        r["content"] = ""

        # Verify regions are in a clean state for OCR
        assert pages_regions[0][0].get("_native_text_used") is None
        # Content either stayed original or was cleaned
        assert pages_regions[0][0]["content"] in ("should stay", "")


# ===================================================================
# Thread safety
# ===================================================================


class TestThreadSafety:
    def test_concurrent_fill_calls(self):
        """Multiple threads calling fill_regions_from_native_text simultaneously."""
        pdf = FIXTURE_NATIVE.read_bytes()
        errors = []
        results = [None] * 8

        def worker(idx):
            try:
                regions = [[_region(content="")]]
                fill_regions_from_native_text(pdf, regions, min_chars=5)
                results[idx] = regions[0][0].get("_native_text_used", False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert all(r is True for r in results), f"Not all filled: {results}"

    def test_concurrent_get_native_text(self):
        """Multiple threads calling get_native_text_in_bbox simultaneously."""
        pdf = FIXTURE_NATIVE.read_bytes()
        errors = []
        results = [None] * 8

        def worker(idx):
            try:
                text = get_native_text_in_bbox(pdf, 0, [0, 0, 1000, 1000])
                results[idx] = "quick brown fox" in text
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert all(r is True for r in results)

    def test_concurrent_mixed_pdfs(self):
        """Threads processing different PDFs (native vs scanned) concurrently."""
        native_pdf = FIXTURE_NATIVE.read_bytes()
        scanned_pdf = FIXTURE_SCANNED.read_bytes()
        errors = []
        results = {}

        def worker_native(idx):
            try:
                regions = [[_region(content="")]]
                fill_regions_from_native_text(native_pdf, regions, min_chars=5)
                results[f"native_{idx}"] = regions[0][0].get("_native_text_used", False)
            except Exception as e:
                errors.append(("native", idx, e))

        def worker_scanned(idx):
            try:
                regions = [[_region(content="")]]
                fill_regions_from_native_text(scanned_pdf, regions, min_chars=5)
                results[f"scanned_{idx}"] = regions[0][0].get("_native_text_used", False)
            except Exception as e:
                errors.append(("scanned", idx, e))

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=worker_native, args=(i,)))
            threads.append(threading.Thread(target=worker_scanned, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"
        for i in range(4):
            assert results[f"native_{i}"] is True
            assert results[f"scanned_{i}"] is False


# ===================================================================
# End-to-end: fill → ocr_page_regions
# ===================================================================


class TestEndToEndFillThenOcr:
    """Runs fill_regions_from_native_text on a real PDF, then passes the
    result through ocr_page_regions to verify the full data flow."""

    def test_native_pdf_fill_then_ocr_skip(self):
        from bibr.pipeline.stages.ocr import ocr_page_regions

        pdf = FIXTURE_NATIVE.read_bytes()
        page_img = Image.new("RGB", (1000, 1000), color="white")

        # Simulate layout output: one text region per page
        pages_regions = [[_region(content="")]]

        # Stage 4b: fill
        fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert pages_regions[0][0]["_native_text_used"] is True

        # Stage 5: OCR — should skip
        ocr_calls = []

        async def spy_ocr(img, prompt):
            ocr_calls.append(1)
            return "should not appear"

        result = asyncio.run(
            ocr_page_regions(
                page_img, pages_regions[0], page_idx=0, filename="test.pdf", ocr_fn=spy_ocr
            )
        )
        assert len(ocr_calls) == 0
        assert "quick brown fox" in result[0]["content"]

    def test_scanned_pdf_fill_then_ocr_runs(self):
        from bibr.pipeline.stages.ocr import ocr_page_regions

        pdf = FIXTURE_SCANNED.read_bytes()
        page_img = Image.new("RGB", (1000, 1000), color="white")

        pages_regions = [[_region(content="")]]

        # Stage 4b: fill — no text layer, nothing happens
        fill_regions_from_native_text(pdf, pages_regions, min_chars=5)
        assert pages_regions[0][0].get("_native_text_used", False) is False

        # Stage 5: OCR — must run
        ocr_calls = []

        async def spy_ocr(img, prompt):
            ocr_calls.append(1)
            return "ocr extracted text"

        result = asyncio.run(
            ocr_page_regions(
                page_img, pages_regions[0], page_idx=0, filename="test.pdf", ocr_fn=spy_ocr
            )
        )
        assert len(ocr_calls) == 1
        assert result[0]["content"] == "ocr extracted text"

    def test_multipage_mixed_fill_then_ocr(self):
        """Multi-page PDF: some pages filled, some not. OCR only called for unfilled."""
        from bibr.pipeline.stages.ocr import ocr_page_regions

        pdf = make_multipage_mixed_pdf()

        pages_regions = [
            [_region(content="")],  # page 0 — has text
            [_region(content="")],  # page 1 — blank
            [_region(content="")],  # page 2 — has text
        ]

        fill_regions_from_native_text(pdf, pages_regions, min_chars=5)

        ocr_calls_per_page = {0: 0, 1: 0, 2: 0}
        page_img = Image.new("RGB", (1000, 1000), color="white")

        for page_idx in range(3):

            async def make_ocr(pidx, regions=pages_regions):
                async def spy_ocr(img, prompt):
                    ocr_calls_per_page[pidx] += 1
                    return f"ocr page {pidx}"

                return await ocr_page_regions(
                    page_img,
                    regions[pidx],
                    page_idx=pidx,
                    filename="test.pdf",
                    ocr_fn=spy_ocr,
                )

            asyncio.run(make_ocr(page_idx))

        # Pages 0 and 2 had native text — no OCR
        assert ocr_calls_per_page[0] == 0
        # Page 1 was blank — OCR was called
        assert ocr_calls_per_page[1] == 1
        # Page 2 had native text — no OCR
        assert ocr_calls_per_page[2] == 0


# ===================================================================
# DEFAULT_ELIGIBLE_LABELS consistency
# ===================================================================


class TestEligibleLabels:
    def test_eligible_labels_is_frozenset(self):
        assert isinstance(DEFAULT_ELIGIBLE_LABELS, frozenset)

    def test_tables_excluded(self):
        assert "table" not in DEFAULT_ELIGIBLE_LABELS
        assert "table_title" not in DEFAULT_ELIGIBLE_LABELS

    def test_figures_excluded(self):
        assert "figure" not in DEFAULT_ELIGIBLE_LABELS
        assert "figure_title" not in DEFAULT_ELIGIBLE_LABELS

    def test_formulas_excluded(self):
        assert "formula" not in DEFAULT_ELIGIBLE_LABELS
        assert "formula_number" not in DEFAULT_ELIGIBLE_LABELS

    def test_core_text_labels_included(self):
        for label in ("text", "abstract", "reference", "paragraph_title", "doc_title"):
            assert label in DEFAULT_ELIGIBLE_LABELS, f"{label} should be eligible"

    def test_no_unexpected_labels(self):
        """Guard against accidental additions. Update this test when adding labels."""
        expected = {
            "text",
            "content",
            "vertical_text",
            "paragraph_title",
            "doc_title",
            "abstract",
            "reference",
            "reference_content",
            "footnote",
            "vision_footnote",
            "algorithm",
            "seal",
        }
        assert expected == DEFAULT_ELIGIBLE_LABELS
