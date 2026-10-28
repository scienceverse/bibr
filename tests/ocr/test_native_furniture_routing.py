"""Text-layer headers and footers go to OCR where the text layer misses or garbles them.

With OCR_NATIVE_TEXT_HEADER_FOOTER on, a header or footer stays with OCR when
its text is in a right-to-left script (pdfium's visual order turned "(14)"
into ")41(" in an Arabic running head), when it is a first-page masthead
drawn partly as an image (gesec's raster "DOI: 10.7769/..." line, of which the
text layer has only the editor line), on a page whose text is mostly an
invisible OCR layer, and on a page a scan image covers whatever the render mode
of its OCR layer. Body text keeps its route in every case.
"""

from __future__ import annotations

import pytest

from bibr.ocr import native_text
from bibr.ocr.native_text import fill_regions_from_native_text, resolve_eligible_labels
from tests.ocr.test_invisible_text_layer import _image, _pdf, _text
from tests.ocr.test_watermark_text import _width

_PAGE_W, _PAGE_H = 612.0, 792.0
_BODY = [
    "Body text of the article runs down the page in lines.",
    "A second line of the body keeps the page from looking empty.",
]


def _region(label: str, left: float, bottom: float, right: float, top: float) -> dict:
    return {
        "label": label,
        "task_type": "text",
        "bbox_2d": [
            left / _PAGE_W * 1000.0,
            (_PAGE_H - top) / _PAGE_H * 1000.0,
            right / _PAGE_W * 1000.0,
            (_PAGE_H - bottom) / _PAGE_H * 1000.0,
        ],
        "content": "",
    }


def _fill(pdf_bytes: bytes, pages: list[list[dict]]) -> list[list[dict]]:
    return fill_regions_from_native_text(
        pdf_bytes, pages, min_chars=5, eligible_labels=resolve_eligible_labels(True, False)
    )


def _masthead_page(*, image: bool) -> dict:
    """An editor line under a raster masthead band (gesec's first page)."""
    band = _image(270, 50, 30, 740) if image else b""
    return {
        "content": band
        + _text(["Editora Cientifica: Cibele Barsalini Martins"], x=34, y=744, size=8)
        + _text(_BODY, y=600)
    }


def test_first_page_masthead_over_an_image_goes_to_ocr():
    pdf_bytes = _pdf([_masthead_page(image=True), _masthead_page(image=True)])
    pages = [
        [_region("header", 30, 740, 300, 790), _region("text", 60, 560, 560, 615)],
        [_region("header", 30, 740, 300, 790)],
    ]

    filled = _fill(pdf_bytes, pages)

    assert "_native_text_used" not in filled[0][0]
    assert filled[0][1]["_native_text_used"] is True
    # Later pages keep the text-layer header: the rule is about the masthead.
    assert filled[1][0]["content"] == "Editora Cientifica: Cibele Barsalini Martins"


def test_first_page_box_the_glyphs_barely_fill_goes_to_ocr():
    """One 8 pt line in a 50 pt masthead box: the rest is not in the text layer."""
    pdf_bytes = _pdf([_masthead_page(image=False)])

    tall = _fill(pdf_bytes, [[_region("header", 30, 740, 300, 790)]])[0][0]
    tight = _fill(pdf_bytes, [[_region("header", 30, 740, 300, 754)]])[0][0]

    assert "_native_text_used" not in tall
    assert tight["content"] == "Editora Cientifica: Cibele Barsalini Martins"


@pytest.mark.parametrize(
    ("label", "filled"), [("header", False), ("footer", False), ("text", True)]
)
def test_right_to_left_furniture_goes_to_ocr(monkeypatch, label, filled):
    """An Arabic journal's running head as pdfium returns it."""
    monkeypatch.setattr(
        native_text,
        "_text_from_bbox_on_textpage",
        lambda *a, **k: "المجلد )41( 1 202",
    )
    pdf_bytes = _pdf([{"content": _text(_BODY, y=600)}, {"content": _text(_BODY, y=600)}])

    result = _fill(pdf_bytes, [[], [_region(label, 30, 560, 560, 615)]])

    assert result[1][0].get("_native_text_used", False) is filled


def test_mostly_invisible_page_keeps_its_furniture_on_ocr():
    """An OCR layer over a page the scan check did not accept: body as before, header to OCR."""
    layer = _text(["Journal of Things 12 (1985) 3-9"], mode=3, y=760, size=9)
    layer += _text(_BODY * 4, mode=3, y=600)
    pdf_bytes = _pdf([{"content": layer}, {"content": layer}])
    pages = [[], [_region("header", 60, 750, 560, 772), _region("text", 60, 440, 560, 615)]]

    filled = _fill(pdf_bytes, pages)

    assert "_native_text_used" not in filled[1][0]
    assert filled[1][1]["_native_text_used"] is True


def test_visible_ocr_layer_over_a_scan_keeps_its_furniture_on_ocr():
    """A scanning service's OCR layer drawn visibly over the page image (an evaluation-set scan)."""
    layer = _image(612, 792)
    layer += _text(["Althof ef af.i Electronic portal imaging device"], x=60, y=760, size=9)
    layer += _text(_BODY * 4, y=600)
    pdf_bytes = _pdf([{"content": layer}, {"content": layer}])
    pages = [[], [_region("header", 60, 750, 560, 772), _region("text", 60, 440, 560, 615)]]

    filled = _fill(pdf_bytes, pages)

    assert "_native_text_used" not in filled[1][0]
    assert filled[1][1]["_native_text_used"] is True


_DOI_LINE = "https://doi.org/10.1177/11779322231160397"


def _doi_header_page() -> dict:
    return {"content": _text([_DOI_LINE], x=34, y=744, size=8) + _text(_BODY, y=600)}


def test_box_that_cuts_the_line_marks_the_region():
    """A SAGE paper's header box ended the DOI at '...311603'."""
    cut_at = 34 + _width("https://doi.org/10.1177/117793222311603", 8.0) + 0.5
    pdf_bytes = _pdf([_doi_header_page(), _doi_header_page()])

    cut = _fill(pdf_bytes, [[], [_region("header", 30, 740, cut_at, 754)]])[1][0]
    whole = _fill(pdf_bytes, [[], [_region("header", 30, 740, 300, 754)]])[1][0]

    assert cut["content"].endswith("311603")
    assert cut["_native_text_edge_cut"] is True
    assert whole["content"] == _DOI_LINE
    assert "_native_text_edge_cut" not in whole


def test_box_whose_bottom_edge_cuts_a_line_marks_the_region():
    page = {
        "content": _text([_DOI_LINE, "Bioinformatics and Biology Insights"], x=34, y=744, size=8)
    }
    page["content"] += _text(_BODY, y=600)
    pdf_bytes = _pdf([page, page])

    # The second line's baseline is at 732; its glyphs reach about 738.
    cut = _fill(pdf_bytes, [[], [_region("header", 30, 735, 300, 754)]])[1][0]

    assert cut["_native_text_edge_cut"] is True
