"""Text that never prints stays out of the native text layer (rule hidden_text/2).

Springer PDFs carry an invisible run of the font subset's digits and
punctuation in the running head, "(0123456789().,-volV)" at 0.1 pt in white,
and "Vol.:(0123456789)" at 8 pt in white at the foot of the first page. The
text layer glued the run onto the DOI ("...0654-7(0123456789("). Other PDFs
print repository stamps or whole abstracts in white on white, or hide a
generator line in render mode 3. The fixtures are minimal hand-built PDFs with
those shapes; visible text that merely looks like a glyph-set run, white text
on a coloured band and a scan's hidden OCR layer must survive.
"""

from __future__ import annotations

import pypdfium2
import pytest

from bibr.extract.pdf_doi_evidence import read_pdf_doi_evidence
from bibr.ocr import native_text
from bibr.ocr.native_text import fill_native_text_and_fonts, open_text_page
from bibr.ocr.utils import pdfium_lock
from tests.ocr.test_invisible_text_layer import _image
from tests.ocr.test_invisible_text_layer import _pdf as _layer_pdf
from tests.ocr.test_invisible_text_layer import _text as _layer_text
from tests.ocr.test_watermark_text import _PAGE_H, _PAGE_W, _bt, _pdf, _width

_HEAD_Y = 760.0
_FOOT_Y = 40.0
_DOI_HEAD = "J Relig Health https://doi.org/10.1007/s10943-018-0654-7"


def _white(content: bytes) -> bytes:
    return b"q 1 1 1 rg\n" + content + b"Q\n"


def _tiny_run(x: float, y: float, text: bytes = rb"\(0123456789\(\).,-volV\)") -> bytes:
    """Springer's head run: Tf 1 under a 0.1 matrix, in white."""
    return _white(b"BT /F1 1 Tf 0.1 0 0 0.1 %.4f %.4f Tm (" % (x, y) + text + b") Tj ET\n")


def _region(label: str, y: float, *, height: float = 14.0) -> dict:
    top, bottom = y + height - 3.0, y - 4.0
    return {
        "label": label,
        "bbox_2d": [
            30.0 / _PAGE_W * 1000.0,
            (_PAGE_H - top) / _PAGE_H * 1000.0,
            590.0 / _PAGE_W * 1000.0,
            (_PAGE_H - bottom) / _PAGE_H * 1000.0,
        ],
        "content": "",
    }


def _read(pdf_bytes: bytes, label: str, y: float) -> str:
    regions = [_region(label, y)]
    fill_native_text_and_fonts(
        pdf_bytes,
        [regions],
        min_chars=3,
        eligible_labels=native_text.resolve_eligible_labels(True, False),
    )
    return regions[0]["content"]


def _page_text(pdf_bytes: bytes, furniture: list | None = None) -> str:
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            page = doc[0]
            textpage = (
                open_text_page(page) if furniture is None else open_text_page(page, [], furniture)
            )
            try:
                return textpage.get_text_range()
            finally:
                textpage.close()
                page.close()
        finally:
            doc.close()


def _body() -> bytes:
    return b"".join(
        _bt(f"Body text line {n} of the article.", 72.0, 600.0 - 20.0 * n) for n in range(8)
    )


def test_springer_head_run_does_not_glue_onto_the_doi():
    pdf_bytes = _pdf(
        _bt(_DOI_HEAD, 46.0, _HEAD_Y, size=8.0)
        + _tiny_run(46.0 + _width(_DOI_HEAD, 8.0), _HEAD_Y)
        + _tiny_run(46.0 + _width(_DOI_HEAD, 8.0) + 1.0, _HEAD_Y)
        + _body()
    )

    assert _read(pdf_bytes, "header", _HEAD_Y) == _DOI_HEAD
    evidence = read_pdf_doi_evidence(pdf_bytes, [1])
    head = [line.text for line in evidence.lines if "doi.org" in line.text]
    assert head == [_DOI_HEAD]


def test_white_vol_marker_leaves_the_footer():
    """'Vol.:(0123456789)' in white at 8 pt beside the visible footer.

    It ends before the page number: white glyphs painted over black ones would
    erase ink, which the render counts as showing.
    """
    marker = "Vol.:(0123456789)"
    pdf_bytes = _pdf(
        _body()
        + _bt("1 3", 520.0, _FOOT_Y, size=8.0)
        + _white(_bt(marker, 510.0 - _width(marker, 8.0), _FOOT_Y, size=8.0))
    )

    assert _read(pdf_bytes, "footer", _FOOT_Y) == "1 3"


def test_en_dash_run_is_dropped_too():
    """The run with an en dash (StandardEncoding 0xB1) instead of the hyphen."""
    pdf_bytes = _pdf(
        _bt("Journal of Religion and Health", 46.0, _HEAD_Y, size=8.0)
        + _tiny_run(300.0, _HEAD_Y, rb"\(0123456789\(\).,\261volV\)")
        + _body()
    )

    assert _read(pdf_bytes, "header", _HEAD_Y) == "Journal of Religion and Health"


def test_visible_text_after_the_run_is_kept_whole():
    """The string guard took 'Vol' from '(0123456789().,-volV)Vol. 12'."""
    pdf_bytes = _pdf(_tiny_run(46.0, _HEAD_Y) + _bt("Vol. 12", 48.5, _HEAD_Y, size=8.0) + _body())

    assert _read(pdf_bytes, "header", _HEAD_Y) == "Vol. 12"


def test_visible_digit_run_is_text():
    """A printed '0123456789' (a reference's page field) is not a glyph-set run."""
    line = "Development, 0123456789, 1-4."
    pdf_bytes = _pdf(_body() + _bt(line, 72.0, 300.0))

    assert _read(pdf_bytes, "reference_content", 300.0) == line


def test_white_text_on_a_coloured_band_is_kept():
    band = b"q 0.1 0.2 0.5 rg 30 750 552 30 re f Q\n"
    pdf_bytes = _pdf(band + _white(_bt("Open Access Journal", 46.0, _HEAD_Y, size=10.0)) + _body())

    assert _read(pdf_bytes, "header", _HEAD_Y) == "Open Access Journal"


def test_white_text_half_on_a_coloured_band_is_kept():
    """A band behind the top of the box only: most of the box is white page.

    A median of the box judged it background and dropped the whole line; only
    text whose drawing changes almost none of its box is hidden.
    """
    band = b"q 0.1 0.2 0.5 rg 30 763.4 552 30 re f Q\n"
    pdf_bytes = _pdf(band + _white(_bt("Open Access Journal", 46.0, _HEAD_Y, size=10.0)) + _body())

    assert _read(pdf_bytes, "header", _HEAD_Y) == "Open Access Journal"


def test_white_text_under_a_logo_painted_over_it_leaves_the_footer():
    """Springer's marker sits under its logo: the box shows the logo, not white."""
    marker = "Vol.:(0123456789)"
    logo = b"q 0.3 0.3 0.3 rg 96 %.2f %.2f 14 re f Q\n" % (_FOOT_Y - 4.0, _width(marker, 8.0) + 8.0)
    pdf_bytes = _pdf(
        _body()
        + _bt("1 3", 520.0, _FOOT_Y, size=8.0)
        + _white(_bt(marker, 100.0, _FOOT_Y, size=8.0))
        + logo
    )

    assert _read(pdf_bytes, "footer", _FOOT_Y) == "1 3"


def test_white_text_on_white_is_dropped():
    pdf_bytes = _pdf(
        _white(_bt("Abstract citation ID: ckae144.904", 72.0, 700.0)) + _body(),
    )

    text = _page_text(pdf_bytes)
    assert "ckae144" not in text
    assert "Body text line 0" in text


def test_a_white_on_white_doi_is_hidden_doi_evidence_not_a_printed_line():
    """osf_mvfth: the postprint banner's DOI is white on white."""
    banner = "Final article available at: https://doi.org/10.1037/cou0000324"
    pdf_bytes = _pdf(_white(_bt(banner, 72.0, 700.0, size=8.0)) + _body())

    evidence = read_pdf_doi_evidence(pdf_bytes, [1])

    assert not [line for line in evidence.lines if "10.1037" in line.text]
    assert [(item.page, item.text) for item in evidence.hidden] == [(1, banner)]


def test_white_text_with_a_dark_outline_is_kept():
    """Fill and stroke (render mode 2): the black outline prints on white."""
    outlined = b"q 1 1 1 rg 0 0 0 RG BT 2 Tr /F1 18 Tf 1 0 0 1 72 700 Tm (Outlined Title) Tj ET Q\n"

    assert "Outlined Title" in _page_text(_pdf(outlined + _body()))


def test_light_grey_text_is_kept():
    """Grey well off the background (a 'Downloaded from' stamp) prints."""
    pdf_bytes = _pdf(
        b"q 0.7 0.7 0.7 rg\n" + _bt("Downloaded from x", 72.0, 700.0) + b"Q\n" + _body()
    )

    assert "Downloaded from x" in _page_text(pdf_bytes)


def test_invisible_generator_line_is_dropped_on_a_born_digital_page():
    hidden = b"q BT 3 Tr /F1 6 Tf 1 0 0 1 72 20 Tm (Powered by TCPDF) Tj ET Q\n"
    pdf_bytes = _pdf(_body() + hidden)

    text = _page_text(pdf_bytes)
    assert "TCPDF" not in text
    assert "Body text line 7" in text


def test_hidden_ocr_layer_of_a_scan_is_kept():
    """Mostly render-mode-3 text is a scan's OCR layer: the invisible-layer check needs it."""
    layer = b"".join(
        b"q BT 3 Tr /F1 12 Tf 1 0 0 1 72 %d Tm (OCR line %d) Tj ET Q\n" % (700 - 20 * n, n)
        for n in range(10)
    )
    pdf_bytes = _pdf(layer + _bt("Stamp", 72.0, 40.0))

    text = _page_text(pdf_bytes)
    assert "OCR line 0" in text
    assert "OCR line 9" in text


def _scan_route(pdf_bytes: bytes) -> tuple[str, bool]:
    """The page text after the furniture strip, and whether the scan check still holds."""
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            page = doc[0]
            textpage = open_text_page(page)
            try:
                crop_box = native_text._page_crop_box(page)
                return (
                    textpage.get_text_range(),
                    native_text._is_invisible_text_layer_page(page, textpage, crop_box),
                )
            finally:
                textpage.close()
                page.close()
        finally:
            doc.close()


_OCR_LINES = [
    f"Hidden OCR line {n} of a scanned article, read by the legacy engine long ago"
    for n in range(6)
]


def _stamps() -> bytes:
    """Ten visible one-glyph marks: page numbers, a stamp's letters."""
    return b"".join(_layer_text([str(n)], x=40 + 50 * n, y=20, size=8) for n in range(10))


def test_scan_with_long_ocr_lines_beside_one_glyph_stamps_keeps_its_scan_route():
    """6 invisible OCR lines and 10 visible one-glyph objects over a page image.

    By object count the layer was a minority (6 of 16) and was stripped, so
    the scan check saw no invisible text and the page lost its scan route. By
    characters, as the scan check counts, the layer is most of the page, and a
    page images cover keeps it whatever the share.
    """
    layer = b"".join(
        _layer_text([line], mode=3, x=40, y=700 - 20 * n, size=10)
        for n, line in enumerate(_OCR_LINES)
    )
    pdf_bytes = _layer_pdf([{"content": _image(612, 792) + layer + _stamps()}])

    text, scan = _scan_route(pdf_bytes)
    assert all(line in text for line in _OCR_LINES)
    assert scan


def test_mostly_invisible_characters_stay_on_a_page_without_a_scan_image():
    """The same text with no page image: the invisible share is by characters."""
    layer = b"".join(
        _layer_text([line], mode=3, x=40, y=700 - 20 * n, size=10)
        for n, line in enumerate(_OCR_LINES)
    )
    pdf_bytes = _layer_pdf([{"content": layer + _stamps()}])

    text, scan = _scan_route(pdf_bytes)
    assert all(line in text for line in _OCR_LINES)
    assert not scan


def test_a_little_invisible_text_on_a_born_digital_page_still_goes():
    """A few invisible characters on a born-digital page still go."""
    hidden = _layer_text(["Powered by TCPDF"], mode=3, x=72, y=20, size=6)
    pdf_bytes = _layer_pdf([{"content": _layer_text(["Visible body text"] * 4) + hidden}])

    text, scan = _scan_route(pdf_bytes)
    assert "TCPDF" not in text
    assert not scan


def test_white_ocr_layer_over_a_scan_is_kept():
    """A scan's OCR layer filled white over the page image (a scanned Lancet paper) is its text.

    The render behind it is the scan's paper, so the background test would
    keep only the words over ink; a page images cover keeps the layer whole.
    """
    lines = ["Proceedings of the Society 41 (1962)", "An old scan with a white text layer"]
    layer = b"q 1 1 1 rg\n" + _layer_text(lines, x=40, y=740, size=10) + b"Q\n"
    pdf_bytes = _layer_pdf([{"content": _image(612, 792) + layer}])

    text = _page_text(pdf_bytes)
    assert all(line in text for line in lines)


def test_text_off_the_page_is_dropped():
    pdf_bytes = _pdf(_bt("Slug 2017-09-13 15:45", -300.0, 30.0) + _body())

    assert "Slug" not in _page_text(pdf_bytes)


def test_removed_objects_are_reported_as_hidden_text_furniture():
    pdf_bytes = _pdf(_white(_bt("Vol.:(0123456789)", 470.0, _FOOT_Y, size=8.0)) + _body())

    furniture: list = []
    _page_text(pdf_bytes, furniture)
    assert [(kind, text) for kind, _box, text in furniture] == [
        ("hidden_text", "Vol.:(0123456789)")
    ]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (_tiny_run(46.0, _HEAD_Y), "tiny"),
        (_white(_bt("Vol.:(0123456789)", 470.0, _FOOT_Y, size=8.0)), "background"),
        (b"q BT 3 Tr /F1 6 Tf 1 0 0 1 72 20 Tm (Powered by) Tj ET Q\n", "invisible"),
        (_bt("Slug", -300.0, 30.0), "off_page"),
    ],
)
def test_hidden_text_reasons(content, reason):
    import pypdfium2.raw as pdfium_c

    with pdfium_lock:
        doc = pypdfium2.PdfDocument(_pdf(content + _body()))
        try:
            page = doc[0]
            try:
                objects: list = []
                top = [
                    pdfium_c.FPDFPage_GetObject(page.raw, index)
                    for index in range(pdfium_c.FPDFPage_CountObjects(page.raw))
                ]
                native_text._collect_text_objects(pdfium_c, top, None, None, 0, objects)
                found = native_text._find_hidden_text(pdfium_c, page, objects)
            finally:
                page.close()
        finally:
            doc.close()
    assert [found_reason for _entry, found_reason, _box in found] == [reason]


def test_fake_bold_copies_collapse_to_one_line():
    """'Sensors 2024, 24, 572' drawn three times 0.12 pt apart for a bold look."""
    line = "Sensors 2024, 24, 572"
    pdf_bytes = _pdf(
        b"".join(_bt(line, 46.0 + 0.12 * k, _HEAD_Y, size=8.0) for k in range(3)) + _body()
    )

    assert _read(pdf_bytes, "header", _HEAD_Y) == line


def test_repeated_text_side_by_side_is_kept():
    """Copies that do not overlap are separate words, not overprint."""
    pdf_bytes = _pdf(
        _bt("12", 46.0, _HEAD_Y, size=8.0) + _bt("12", 80.0, _HEAD_Y, size=8.0) + _body()
    )

    assert _read(pdf_bytes, "header", _HEAD_Y) == "12 12"
