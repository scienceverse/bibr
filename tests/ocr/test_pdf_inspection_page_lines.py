"""The page lines and links the PDF inspection captures for the reference line stream."""

from __future__ import annotations

import pytest

from bibr.ocr.native_text import _normalized_bbox_to_pdf_points, _pdf_points_to_normalized_bbox
from bibr.ocr.pdf_inspection import _break_wrapped_lines, inspect_pdf
from tests.ocr.test_pdf_links import _pdf_with_links


def test_inspection_maps_links_into_the_layout_frame():
    pdf = _pdf_with_links([((10, 20, 110, 40), "https://doi.org/10.1234/abc.5")])
    inspection = inspect_pdf(
        pdf,
        [[]],
        fill_native_text=False,
        include_outline=False,
        include_ref_geometry=True,
        min_chars=10,
        min_printable_ratio=0.85,
    )

    assert inspection.uri_links == [
        {"page": 1, "bbox": [50.0, 800.0, 550.0, 900.0], "uri": "https://doi.org/10.1234/abc.5"}
    ]


def test_links_keep_the_pdf_page_number_under_a_page_slice():
    pdf = _pdf_with_links(
        [((10, 20, 110, 40), "https://doi.org/10.1234/abc.5")], blank_pages_before=1
    )
    inspection = inspect_pdf(
        pdf,
        [[]],
        page_indices=[1],
        fill_native_text=False,
        include_outline=False,
        include_ref_geometry=True,
        min_chars=10,
        min_printable_ratio=0.85,
    )

    # Region summaries number pages from the start of the PDF, not the slice.
    assert [link["page"] for link in inspection.uri_links] == [2]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_point_to_layout_conversion_inverts_layout_to_point(rotation):
    crop_box = (12.0, 30.0, 612.0, 822.0)
    layout = (100.0, 200.0, 400.0, 260.0)
    points = _normalized_bbox_to_pdf_points(list(layout), crop_box, rotation)

    assert _pdf_points_to_normalized_bbox(points, crop_box, rotation) == pytest.approx(layout)


def test_wrapped_line_joined_by_pdfium_is_broken_at_the_new_baseline():
    # "meta" + U+FFFE (the hyphen PDFium reads at a line end), then the next
    # printed line starting further left and lower, with no newline between.
    chars = [
        ("m", (500.0, 724.0, 509.0, 730.0)),
        ("e", (509.0, 724.0, 514.0, 730.0)),
        ("t", (514.0, 724.0, 517.0, 731.0)),
        ("a", (517.0, 724.0, 522.0, 730.0)),
        ("\ufffe", (522.0, 726.0, 525.0, 727.0)),
        ("a", (100.0, 696.0, 105.0, 702.0)),
        ("n", (105.0, 696.0, 111.0, 702.0)),
        ("x", (111.0, 694.0, 116.0, 702.0)),
        # A subscript sits lower but to the right: same line.
        ("2", (116.0, 692.0, 119.0, 697.0)),
    ]
    broken = _break_wrapped_lines(chars)

    assert "".join(char for char, _ in broken) == "meta-\nanx2"
