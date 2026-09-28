import ctypes
import io
import json
import unicodedata
from pathlib import Path

import pypdfium2
import pypdfium2.raw as pdfium_raw
import pytest

from bibr.ocr.native_text import (
    _build_page_char_records,
    _is_native_text_usable,
    _normalized_bbox_to_pdf_points,
    _printable_ratio,
    _reconstruct_text_from_records,
    fill_font_metadata,
    fill_regions_from_native_text,
    get_native_text_in_bbox,
)

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "native_text_sample.pdf"
PUA_SOURCE_FIXTURE = Path(__file__).parent / "fixtures" / "native_text_pua.json"


class _FakeTextPage:
    raw = object()

    def __init__(self, boxes):
        self._boxes = boxes

    def count_chars(self):
        return len(self._boxes)

    def get_charbox(self, index):
        return self._boxes[index]


def test_page_char_records_combine_utf16_surrogate_pair(monkeypatch):
    code_units = [0xD835, 0xDC46]
    monkeypatch.setattr(
        pypdfium2.raw,
        "FPDFText_GetUnicode",
        lambda _raw, index: code_units[index],
    )

    records = _build_page_char_records(_FakeTextPage([(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 4.0, 2.0)]))

    assert records == [("𝑆", 2.0, 1.0, False)]
    assert all(unicodedata.category(ch) != "Cs" for ch, *_rest in records)
    assert records[0][0].encode("utf-8") == b"\xf0\x9d\x91\x86"


@pytest.mark.parametrize("code_unit", [0xD835, 0xDC46])
def test_page_char_records_replace_unpaired_surrogate(monkeypatch, code_unit):
    monkeypatch.setattr(
        pypdfium2.raw,
        "FPDFText_GetUnicode",
        lambda _raw, _index: code_unit,
    )

    records = _build_page_char_records(_FakeTextPage([(1.0, 2.0, 3.0, 4.0)]))

    assert records == [("\ufffd", 2.0, 3.0, False)]


class _BoxTextPage:
    """Fake textpage serving recorded ``(char, tight box, loose box)`` triples."""

    raw = object()

    def __init__(self, chars):
        self._chars = chars

    def count_chars(self):
        return len(self._chars)

    def get_charbox(self, index, loose=False):
        return self._chars[index][2 if loose else 1]


def _line_text(monkeypatch, chars, left, bottom, right, top):
    monkeypatch.setattr(
        pypdfium2.raw, "FPDFText_GetUnicode", lambda _raw, index: ord(chars[index][0])
    )
    records = _build_page_char_records(_BoxTextPage(chars))
    return _reconstruct_text_from_records(records, left, bottom, right, top)


# Char boxes recorded with pypdfium2 from the dev-set PDFs each comment describes.
_SEX_TRAFFICKING = [  # a born-digital title: "ffi" ligature, then a space inside it
    ("S", (43.64, 527.02, 58.56, 548.3), (42.52, 522.23, 59.52, 549.14)),
    ("e", (60.64, 527.13, 73.8, 541.16), (59.74, 522.23, 74.36, 549.14)),
    ("x", (74.33, 527.41, 89.64, 540.82), (74.08, 522.23, 89.78, 549.14)),
    (" ", (89.78, 527.41, 97.06, 527.44), (89.78, 522.23, 97.06, 549.14)),
    ("T", (97.23, 527.41, 117.14, 548.16), (97.06, 522.23, 117.31, 549.14)),
    ("r", (116.47, 527.41, 128.28, 541.13), (115.77, 522.23, 128.42, 549.14)),
    ("a", (129.24, 527.21, 143.71, 541.13), (128.14, 522.23, 143.71, 549.14)),
    ("f", (144.3, 527.41, 168.52, 548.97), (143.6, 522.23, 169.22, 549.14)),
    ("f", (144.3, 527.41, 168.52, 548.97), (143.6, 522.23, 169.22, 549.14)),
    ("i", (144.3, 527.41, 168.52, 548.97), (143.6, 522.23, 169.22, 549.14)),
    (" ", (152.14, 527.41, 159.42, 527.44), (152.14, 522.23, 159.42, 549.14)),
    ("c", (170.12, 527.13, 183.16, 541.16), (169.22, 522.23, 183.44, 549.14)),
    ("k", (183.78, 527.41, 199.82, 549.0), (183.22, 522.23, 200.1, 549.14)),
    ("i", (200.92, 527.41, 208.45, 547.79), (200.08, 522.23, 209.15, 549.14)),
    ("n", (209.82, 527.41, 225.72, 541.18), (209.12, 522.23, 226.28, 549.14)),
    ("g", (226.68, 521.44, 242.44, 541.24), (226.26, 521.44, 242.44, 549.14)),
]
_SOFI_OKSANEN = [  # same PDF, page 3: the space after a word-final "fi" ligature is real
    ("S", (88.91, 612.5, 94.1, 619.84), (88.41, 610.81, 94.63, 619.84)),
    ("o", (95.19, 612.5, 99.53, 617.75), (94.63, 610.81, 100.09, 619.78)),
    ("f", (100.36, 612.62, 105.28, 619.9), (100.04, 610.81, 106.24, 619.9)),
    ("i", (100.36, 612.62, 105.28, 619.9), (100.04, 610.81, 106.24, 619.9)),
    (" ", (103.14, 612.62, 105.29, 612.63), (103.14, 610.81, 105.29, 619.78)),
    ("O", (108.97, 612.5, 114.39, 619.84), (108.39, 610.81, 114.97, 619.84)),
    ("k", (115.96, 612.62, 120.15, 619.84), (114.97, 610.81, 120.33, 619.84)),
]
_OF_PERVASIVE = [  # a paper's abstract: an italic "f" hangs over the space
    ("o", (461.81, 363.21, 465.7, 367.34), (461.55, 360.55, 466.05, 372.52)),
    ("f", (464.49, 361.38, 470.28, 369.65), (464.49, 360.55, 470.28, 372.52)),
    (" ", (468.55, 363.31, 470.8, 363.32), (468.55, 360.55, 470.8, 372.52)),
    ("p", (469.59, 361.39, 474.83, 367.34), (469.59, 360.55, 475.08, 372.52)),
    ("e", (475.35, 363.21, 478.95, 367.34), (475.08, 360.55, 479.07, 372.52)),
]
_MODEL_ANALYSIS = [  # same PDF, title, last line
    ("M", (190.76, 626.86, 210.12, 643.09), (190.4, 619.74, 210.41, 648.49)),
    ("O", (211.78, 626.65, 224.16, 643.33), (210.41, 619.74, 225.53, 648.49)),
    ("D", (225.92, 626.86, 239.74, 643.09), (225.53, 619.74, 241.11, 648.49)),
    ("E", (241.49, 626.86, 252.89, 643.09), (241.11, 619.74, 254.02, 648.49)),
    ("L", (254.43, 626.86, 265.37, 643.09), (254.02, 619.74, 266.04, 648.49)),
    (" ", (266.04, 626.86, 273.17, 626.89), (266.04, 619.74, 273.17, 648.49)),
    ("A", (272.96, 626.86, 289.88, 643.33), (272.96, 619.74, 289.88, 648.49)),
    ("N", (290.12, 626.86, 305.38, 643.09), (289.64, 619.74, 305.64, 648.49)),
]
_AFTER_A_COMMA = [  # another paper's byline: a comma hangs below the baseline
    ("i", (111.25, 323.24, 113.49, 330.28), (110.96, 320.17, 113.74, 333.64)),
    ("n", (114.02, 323.24, 118.92, 327.91), (113.96, 320.17, 118.96, 333.64)),
    ("a", (119.42, 323.16, 123.48, 327.91), (119.06, 320.17, 123.5, 333.64)),
    ("1", (123.82, 326.57, 125.34, 330.57), (123.14, 324.78, 126.05, 332.63)),
    (",", (126.25, 325.6, 127.09, 327.15), (125.93, 324.78, 127.39, 332.63)),
    ("2", (127.69, 326.57, 130.24, 330.57), (127.57, 324.78, 130.48, 332.63)),
    (",", (131.2, 321.58, 132.65, 324.23), (130.66, 320.17, 133.16, 333.64)),
    (" ", (133.16, 323.24, 135.66, 323.25), (133.16, 320.17, 135.66, 333.64)),
    ("R", (135.83, 323.24, 142.42, 329.96), (135.66, 320.17, 142.42, 333.64)),
    (".", (143.18, 323.11, 144.26, 324.2), (142.47, 320.17, 144.97, 333.64)),
    ("I", (145.19, 323.24, 148.03, 329.96), (144.94, 320.17, 148.27, 333.64)),
]
_AFTER_A_QUOTE = [  # another paper's table title: a closing quote sits high
    ("t", (416.37, 587.09, 420.06, 594.55), (416.16, 584.76, 420.7, 595.73)),
    ("a", (421.22, 587.08, 426.48, 592.52), (420.7, 584.76, 426.55, 595.73)),
    ("n", (426.93, 587.2, 432.78, 592.46), (426.55, 584.76, 433.04, 595.73)),
    ("t", (432.93, 587.09, 436.61, 594.55), (432.72, 584.76, 437.26, 595.73)),
    ("”", (437.39, 592.03, 441.19, 595.5), (437.26, 584.76, 441.19, 595.73)),
    (" ", (446.03, 587.19, 446.03, 587.19), (446.03, 587.19, 446.03, 587.19)),
    ("o", (446.38, 587.09, 451.52, 592.53), (446.03, 584.76, 451.88, 595.73)),
    ("r", (452.22, 587.2, 456.1, 592.46), (451.88, 584.76, 456.42, 595.73)),
    (" ", (460.32, 587.19, 460.32, 587.19), (460.32, 587.19, 460.32, 587.19)),
    ("H", (460.81, 587.2, 468.59, 595.36), (460.32, 584.76, 469.08, 595.73)),
]

# A body line set with LaTeX (recorded from a dev-set PDF): a bracketed
# superscript citation, then the space pdfium generates on the superscript's
# own raised baseline.
_SENSING_WHERE = [
    ("i", (420.7, 548.37, 422.84, 554.53), (420.51, 546.07, 423.04, 555.09)),
    ("n", (423.23, 548.37, 428.07, 552.54), (423.04, 546.07, 428.24, 555.09)),
    ("g", (428.54, 546.08, 432.61, 552.54), (428.24, 546.07, 432.79, 555.09)),
    (",", (433.29, 547.19, 434.49, 549.3), (432.79, 546.07, 435.03, 555.09)),
    ("[", (435.44, 550.06, 436.35, 555.46), (435.03, 549.98, 436.52, 555.99)),
    ("7", (437.36, 551.24, 439.81, 555.62), (437.03, 549.98, 440.02, 555.99)),
    ("]", (440.69, 550.06, 441.6, 555.46), (440.52, 549.98, 442.01, 555.99)),
    (" ", (440.77, 551.51, 440.77, 551.51), (440.77, 551.51, 440.77, 551.51)),
    ("w", (444.98, 548.26, 451.31, 552.43), (444.98, 546.07, 451.31, 555.09)),
    ("h", (451.38, 548.37, 456.22, 554.98), (451.31, 546.07, 456.4, 555.09)),
    ("e", (456.69, 548.26, 460.25, 552.54), (456.4, 546.07, 460.53, 555.09)),
]
# A line of a scan's text layer (recorded from a dev-set PDF): the spaces sit
# on the baseline, above the short descenders of "y" and "p".
_OS_Y_DE_PUBLIC = [
    ("o", (311.04, 362.9, 315.3, 367.49), (310.51, 360.58, 315.85, 371.09)),
    ("s", (315.93, 362.95, 319.77, 367.47), (315.18, 360.58, 320.52, 371.09)),
    (" ", (320.52, 363.06, 325.86, 363.07), (320.52, 360.58, 325.86, 371.09)),
    ("y", (322.49, 361.5, 327.7, 367.32), (322.43, 360.58, 327.77, 371.09)),
    (" ", (327.77, 363.06, 333.11, 363.07), (327.77, 360.58, 333.11, 371.09)),
    ("d", (331.85, 362.92, 336.71, 369.35), (331.61, 360.58, 336.95, 371.09)),
    ("e", (337.69, 362.95, 341.9, 367.53), (337.13, 360.58, 342.47, 371.09)),
    (" ", (342.47, 363.06, 347.81, 363.07), (342.47, 360.58, 347.81, 371.09)),
    ("p", (344.73, 361.5, 349.59, 367.4), (344.67, 360.58, 350.01, 371.09)),
    ("u", (349.51, 362.92, 354.33, 367.32), (349.25, 360.58, 354.59, 371.09)),
    ("b", (354.01, 362.92, 359.0, 369.35), (353.84, 360.58, 359.18, 371.09)),
]
# The end of a reference line in a scan's text layer (recorded from a dev-set
# PDF): pdfium generates the space after "London." with no line break, so the
# glyph after it opens the next line.
_LONDON_THEN_MILES = [
    ("n", (473.41, 692.34, 476.99, 696.54), (473.25, 689.98, 477.16, 699.97)),
    (".", (478.66, 692.21, 479.58, 693.38), (477.16, 689.98, 481.08, 699.97)),
    (" ", (482.86, 692.34, 482.86, 692.34), (482.86, 692.34, 482.86, 692.34)),
    ("M", (308.28, 685.98, 311.37, 691.43), (308.16, 683.57, 311.49, 693.77)),
    ("i", (312.02, 685.98, 314.29, 692.35), (311.49, 683.57, 314.82, 693.77)),
]


def test_space_drawn_inside_a_ligature_does_not_split_the_word(monkeypatch):
    text = _line_text(monkeypatch, _SEX_TRAFFICKING, 0, 500, 400, 560)

    assert text == "Sex Trafficking"


def test_code_point_past_unicode_after_a_ligature_space_reads_as_a_replacement(monkeypatch):
    # A broken ToUnicode map can hand pdfium a code point above U+10FFFF. The
    # inner-space check reads the glyph after the space as well; it must see
    # the replacement glyph the page records keep, not raise and send the page
    # to OCR.
    codes = [ord(ch) for ch, _tight, _loose in _SEX_TRAFFICKING]
    codes[11] = 0x110000  # the "c" after the space inside the "ffi" ligature
    monkeypatch.setattr(pypdfium2.raw, "FPDFText_GetUnicode", lambda _raw, index: codes[index])

    records = _build_page_char_records(_BoxTextPage(_SEX_TRAFFICKING))

    assert _reconstruct_text_from_records(records, 0, 500, 400, 560) == "Sex Traffi\ufffdking"


def test_word_space_after_a_word_final_ligature_is_kept(monkeypatch):
    text = _line_text(monkeypatch, _SOFI_OKSANEN, 0, 600, 400, 630)

    assert text == "Sofi Ok"


def test_space_under_an_italic_f_overhang_is_kept(monkeypatch):
    text = _line_text(monkeypatch, _OF_PERVASIVE, 400, 350, 560, 380)

    assert text == "of pe"


def test_flat_space_box_stays_with_its_line(monkeypatch):
    # The title region's bottom edge (626.91 pt) sits just above the
    # baseline the space box is flat on, below the letters' centres.
    text = _line_text(monkeypatch, _MODEL_ANALYSIS, 78.236, 626.91, 493.795, 728.702)

    assert text == "MODEL AN"


def test_flat_space_box_is_not_lowered_to_a_comma_below_the_baseline(monkeypatch):
    # The byline region's bottom edge (323.15 pt) sits between the comma's
    # centre and the baseline: the comma falls out, the space after it stays.
    text = _line_text(monkeypatch, _AFTER_A_COMMA, 65.48, 323.15, 516.1, 346.96)

    assert text == "ina1,2 R.I"


def test_flat_space_box_is_not_raised_to_a_quote_above_the_baseline(monkeypatch):
    # The table title region's top edge (591.62 pt) sits between the letters'
    # centres and the closing quote's: the quote falls out, the space after it
    # stays.
    text = _line_text(monkeypatch, _AFTER_A_QUOTE, 110.77, 538.56, 497.56, 591.62)

    assert text == "tant or H"


def test_space_after_a_superscript_is_not_raised_above_the_letters_after_it(monkeypatch):
    # A region top edge (552.0 pt) that cuts the superscript citation off the
    # line, above the space's own centre and the letters': the citation falls
    # out, the space between the words stays.
    text = _line_text(monkeypatch, _SENSING_WHERE, 400.0, 540.0, 480.0, 552.0)

    assert text == "ing, whe"


def test_space_before_a_descender_stays_with_the_glyph_before_it(monkeypatch):
    # The region's bottom edge (364.5 pt) cuts the descenders of "y" and "p"
    # off the line, above the spaces' baseline. A space on the line's baseline
    # still rises to the glyph before it, whatever the glyph after it.
    text = _line_text(monkeypatch, _OS_Y_DE_PUBLIC, 290.0, 364.5, 380.0, 380.0)

    assert text == "os de ub"


def test_space_before_the_next_line_stays_with_the_glyph_before_it(monkeypatch):
    # The glyph after the space is on the line below, so the space is not on
    # a raised baseline: it still rises to the full stop, above the region's
    # bottom edge (692.5 pt).
    text = _line_text(monkeypatch, _LONDON_THEN_MILES, 460.0, 692.5, 490.0, 705.0)

    assert text == "n. "


def _encode_wide(text: str):
    """Encode a Python str as the ctypes c_ushort array FPDFText_SetText wants."""
    return (ctypes.c_ushort * (len(text) + 1))(*[ord(c) for c in text], 0)


def _make_single_text_pdf(
    text: str,
    *,
    font_name: bytes = b"Helvetica",
    page_width: float = 595.0,
    page_height: float = 842.0,
    with_text: bool = True,
) -> bytes:
    """Build a one-page PDF with *text* set in *font_name* (a standard-14 font).

    ``with_text=False`` produces a blank page (no text layer) with the given
    dimensions — used to prove page-size metadata is attached even when a page
    would fall back to OCR.
    """
    doc = pypdfium2.PdfDocument.new()
    page = doc.new_page(page_width, page_height)
    if with_text:
        font_raw = pdfium_raw.FPDFText_LoadStandardFont(doc.raw, font_name)
        text_obj = pdfium_raw.FPDFPageObj_CreateTextObj(doc.raw, font_raw, ctypes.c_float(12))
        pdfium_raw.FPDFText_SetText(text_obj, _encode_wide(text))
        # Place near page centre so a full-page bbox's centre-probe
        # (FPDFText_GetCharIndexAtPos) lands on the text.
        pdfium_raw.FPDFPageObj_Transform(text_obj, 1, 0, 0, 1, page_width * 0.2, page_height * 0.5)
        pdfium_raw.FPDFPage_InsertObject(page.raw, text_obj)
    pdfium_raw.FPDFPage_GenerateContent(page.raw)
    page.close()
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_fill_font_metadata_sets_is_italic_true_for_oblique_font():
    """Standard-14 oblique/italic fonts expose italic only via the font name
    (pdfium leaves the descriptor Italic flag unset for them), so name-based
    detection must flag the region as italic."""
    pdf_bytes = _make_single_text_pdf("Italic words here now", font_name=b"Helvetica-Oblique")
    regions = [[{"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]]
    fill_font_metadata(pdf_bytes, regions)
    assert regions[0][0]["_is_italic"] is True


def test_fill_font_metadata_sets_is_italic_false_for_regular_font():
    pdf_bytes = _make_single_text_pdf("Regular upright words here", font_name=b"Helvetica")
    regions = [[{"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]]
    fill_font_metadata(pdf_bytes, regions)
    assert regions[0][0]["_is_italic"] is False


def test_fill_font_metadata_sets_page_dimensions():
    pdf_bytes = _make_single_text_pdf("Some page text here", page_width=612.0, page_height=792.0)
    regions = [[{"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]]
    fill_font_metadata(pdf_bytes, regions)
    assert regions[0][0]["_page_w"] == pytest.approx(612.0)
    assert regions[0][0]["_page_h"] == pytest.approx(792.0)


def test_fill_font_metadata_sets_page_dimensions_even_on_textless_page():
    """Page dimensions are available regardless of a text layer; a scanned /
    image-only page that will fall back to OCR must still carry page_w/page_h."""
    pdf_bytes = _make_single_text_pdf("", page_width=500.0, page_height=700.0, with_text=False)
    regions = [[{"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]]
    fill_font_metadata(pdf_bytes, regions)
    assert regions[0][0]["_page_w"] == pytest.approx(500.0)
    assert regions[0][0]["_page_h"] == pytest.approx(700.0)


def test_extracts_text_from_region_bbox():
    """A bbox covering the full first page should yield the page's native text."""
    pdf_bytes = FIXTURE_PDF.read_bytes()
    bbox = [0, 0, 1000, 1000]
    text = get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=bbox)
    assert "The quick brown fox" in text


def test_empty_bbox_returns_empty_string():
    pdf_bytes = FIXTURE_PDF.read_bytes()
    text = get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=[0, 0, 1, 1])
    assert text == ""


def test_scanned_pdf_returns_empty_string():
    """A scanned/image-only PDF has no text layer."""
    scanned = Path(__file__).parent / "fixtures" / "scanned_sample.pdf"
    text = get_native_text_in_bbox(
        scanned.read_bytes(), page_idx=0, bbox_normalized=[0, 0, 1000, 1000]
    )
    assert text == ""


def test_coordinate_conversion_matches_expected_values():
    """Sanity-check the y-flip math on a known-size page with origin (0,0).

    CropBox (0, 0, 612, 792) (US Letter), bbox [0, 0, 500, 250] in image-space
    (top-left quadrant half-height).
    Expected PDF-point coords: left=0, top=792, right=306, bottom=594.
    """
    left, bottom, right, top = _normalized_bbox_to_pdf_points(
        [0, 0, 500, 250], crop_box=(0, 0, 612, 792)
    )
    assert left == pytest.approx(0.0)
    assert right == pytest.approx(306.0)
    assert top == pytest.approx(792.0)  # y-flip: image y=0 (top) -> PDF y=crop top
    assert bottom == pytest.approx(594.0)  # 792 - 250/1000*792 = 594


def test_coordinate_conversion_full_page():
    """Full-page normalized bbox produces full-page PDF-point bbox."""
    left, bottom, right, top = _normalized_bbox_to_pdf_points(
        [0, 0, 1000, 1000], crop_box=(0, 0, 612, 792)
    )
    assert (left, bottom, right, top) == pytest.approx((0.0, 0.0, 612.0, 792.0))


def test_coordinate_conversion_offsets_by_crop_origin():
    """Pages with a non-zero CropBox origin must offset bboxes by that origin.

    pypdfium2 renders the CropBox to the image the layout bboxes index into,
    but text coords live in the page's native (MediaBox) space. Assuming a
    (0,0) origin shifts every box, slicing text columns. Regression test for
    native-text garbling on publisher PDFs (e.g. Elsevier) whose CropBox
    origin is non-zero.
    """
    # CropBox origin (40, 50), 600 wide x 800 tall. Right-half top quadrant.
    left, bottom, right, top = _normalized_bbox_to_pdf_points(
        [500, 0, 1000, 250], crop_box=(40, 50, 640, 850)
    )
    assert left == pytest.approx(340.0)  # 40 + 500/1000*600
    assert right == pytest.approx(640.0)  # 40 + 1000/1000*600
    assert top == pytest.approx(850.0)  # crop top (y1); image y=0
    assert bottom == pytest.approx(650.0)  # 850 - 250/1000*800


def test_nonzero_crop_origin_does_not_slice_columns():
    """End-to-end: a PDF whose MediaBox origin is (100,100) must not have its
    text columns sliced. The fixture places ALPHA near the left and OMEGA near
    the right edge; a right-half bbox must return OMEGA, not empty.
    """
    fixture = Path(__file__).parent / "fixtures" / "cropbox_offset_sample.pdf"
    pdf_bytes = fixture.read_bytes()
    right_half = get_native_text_in_bbox(
        pdf_bytes, page_idx=0, bbox_normalized=[600, 0, 1000, 1000]
    )
    assert "OMEGA" in right_half
    assert "ALPHA" not in right_half
    left_half = get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=[0, 0, 400, 1000])
    assert "ALPHA" in left_half
    assert "OMEGA" not in left_half


def _inherited_mediabox_pdf(lines: list[tuple[float, str]]) -> bytes:
    """An A4 page whose MediaBox sits on the page tree, not the page dictionary.

    *lines* are ``(baseline y in points, text)`` pairs set in 12 pt Helvetica.
    """
    stream = "".join(f"BT /F1 12 Tf 72 {y} Td ({text}) Tj ET\n" for y, text in lines).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 595.28 841.89] >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def test_inherited_mediabox_maps_regions_onto_the_rendered_page():
    """``get_cropbox()`` reads the page dictionary alone and falls back to US
    Letter; the page renders at its inherited A4 MediaBox. The Letter box put
    the byline's text under the title region (a preprint
    exported the byline "DanSnow1" as its title)."""
    pdf_bytes = _inherited_mediabox_pdf([(742, "Biased Perceptions"), (698, "Dan Snow")])
    page_height = 841.89
    # The title line's band, 88-104 pt below the top of the rendered page.
    title_band = [0, 88 / page_height * 1000, 1000, 104 / page_height * 1000]

    text = get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=title_band)

    assert text == "Biased Perceptions"


_BLEED_FIXTURE = Path(__file__).parent / "fixtures" / "native_text_bbox_bleed_sample.pdf"


def test_bbox_intersection_does_not_leak_clipped_neighboring_line():
    """A region bbox whose top edge slices through the line above it must not
    pull in the partial glyphs of that neighboring line as garbage.

    Regression test: ``get_text_bounded`` treats a char as "inside" whenever
    its bounding box merely INTERSECTS the query rect, so a bbox edge that
    slices through the line above the region bleeds in stray partial-glyph
    garbage (the fixture's upper-line descender fragments, ``pyy``) before
    the real text. The committed synthetic fixture replaces the uncommitted
    ``data/hrv.pdf`` sample with the same geometry: a query whose top edge
    cuts through the upper line's glyph boxes while their centers stay out.
    """
    pdf_bytes = _BLEED_FIXTURE.read_bytes()
    text = get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=[0, 117, 1000, 154])
    assert text == "Second line of body text content here"


@pytest.mark.parametrize("bottom_pt", [420.0, 421.3, 423.0, 425.0])
def test_region_edge_above_the_baseline_keeps_the_spaces(bottom_pt):
    """pdfium's box for a space is flat on the baseline (no ink). A region
    whose bottom edge crosses the line between the baseline (421 pt here) and
    the letters' centres (~425 pt) kept the letters but dropped the space
    ("MODELANALYSIS" in the title of the _MODEL_ANALYSIS paper)."""
    pdf_bytes = _make_single_text_pdf("MODEL ANALYSIS")
    bbox = [0, (842.0 - 440.0) / 842.0 * 1000, 1000, (842.0 - bottom_pt) / 842.0 * 1000]

    text = get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=bbox)

    assert text == "MODEL ANALYSIS"


def test_fill_populates_text_regions_from_native_pdf():
    """Text regions get their `content` pre-filled with native text."""
    pdf_bytes = FIXTURE_PDF.read_bytes()
    regions_per_page = [
        [  # page 0
            {"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
        ],
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes,
        regions_per_page,
        min_chars=5,
        eligible_labels=frozenset({"text"}),
    )
    assert filled[0][0]["_native_text_used"] is True
    assert "The quick brown fox" in filled[0][0]["content"]


def test_fill_skips_table_regions():
    pdf_bytes = FIXTURE_PDF.read_bytes()
    regions = [
        [{"label": "table", "task_type": "table", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=5, eligible_labels=frozenset({"text"})
    )
    assert "_native_text_used" not in filled[0][0]
    assert filled[0][0]["content"] == ""


def test_eligible_labels_include_header_footer():
    """x-performance-3 behind OCR_NATIVE_TEXT_HEADER_FOOTER (default off).

    Default output is unchanged (header/footer go to OCR); opting in adds
    them to the eligible set. Short running heads keep the short-text
    allowance in both cases (it only matters once they are eligible).
    """
    from bibr.config import GlobalSettings
    from bibr.ocr.native_text import (
        _SHORT_NATIVE_TEXT_LABELS,
        DEFAULT_ELIGIBLE_LABELS,
        HEADER_FOOTER_LABELS,
        resolve_eligible_labels,
    )

    assert GlobalSettings().ocr.native_text_header_footer is False
    assert "header" not in DEFAULT_ELIGIBLE_LABELS
    assert "footer" not in DEFAULT_ELIGIBLE_LABELS
    assert resolve_eligible_labels(False) == DEFAULT_ELIGIBLE_LABELS
    assert resolve_eligible_labels(True) == DEFAULT_ELIGIBLE_LABELS | HEADER_FOOTER_LABELS
    assert "header" in HEADER_FOOTER_LABELS
    assert "footer" in HEADER_FOOTER_LABELS
    assert "header" in _SHORT_NATIVE_TEXT_LABELS
    assert "footer" in _SHORT_NATIVE_TEXT_LABELS


def test_header_footer_setting_defaults_off_but_enables_fill(monkeypatch):
    """The setting gates the behavior: off skips header/footer, on fills."""
    from bibr.config import GlobalSettings
    from bibr.ocr.native_text import resolve_eligible_labels

    monkeypatch.setenv("OCR_NATIVE_TEXT_HEADER_FOOTER", "true")
    assert GlobalSettings().ocr.native_text_header_footer is True

    pdf_bytes = FIXTURE_PDF.read_bytes()
    regions_off = [
        [
            {"label": "header", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
        ]
    ]
    filled_off = fill_regions_from_native_text(
        pdf_bytes, regions_off, min_chars=5, eligible_labels=resolve_eligible_labels(False)
    )
    assert "_native_text_used" not in filled_off[0][0]

    regions_on = [
        [
            {"label": "header", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
        ]
    ]
    filled_on = fill_regions_from_native_text(
        pdf_bytes, regions_on, min_chars=5, eligible_labels=resolve_eligible_labels(True)
    )
    assert filled_on[0][0]["_native_text_used"] is True


def test_fill_populates_header_footer_from_native_pdf():
    """Header/footer regions fill from the text layer like body text (same
    printable-ratio gate): every such region otherwise costs an OCR call."""
    from bibr.ocr.native_text import resolve_eligible_labels

    pdf_bytes = FIXTURE_PDF.read_bytes()
    regions = [
        [
            {"label": "header", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
            {"label": "footer", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
            {"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
        ]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=5, eligible_labels=resolve_eligible_labels(True)
    )
    for region in filled[0]:
        assert region["_native_text_used"] is True, region["label"]
        assert _printable_ratio(region["content"]) >= 0.85
    assert "The quick brown fox" in filled[0][0]["content"]
    assert filled[0][0]["content"] == filled[0][2]["content"]


def test_fill_accepts_short_header_footer_below_body_min_chars(monkeypatch):
    """Running headers/footers ("Cell Biology", "Benartzi et al.") are shorter
    than the body min_chars: the short-text allowance covers them."""
    import bibr.ocr.native_text as native_mod
    from bibr.ocr.native_text import resolve_eligible_labels

    pdf_bytes = FIXTURE_PDF.read_bytes()
    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: "Cell Biology")

    regions = [
        [
            {"label": "header", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
            {"label": "footer", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
            {"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
        ]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=20, eligible_labels=resolve_eligible_labels(True)
    )

    assert filled[0][0]["_native_text_used"] is True
    assert filled[0][0]["content"] == "Cell Biology"
    assert filled[0][1]["_native_text_used"] is True
    assert filled[0][2].get("_native_text_used", False) is False


def test_fill_header_footer_still_reject_corrupt_text(monkeypatch):
    """Guard: the printable-ratio gate still applies to headers/footers, so a
    broken text layer falls back to OCR instead of ingesting mojibake."""
    import bibr.ocr.native_text as native_mod
    from bibr.ocr.native_text import resolve_eligible_labels

    pdf_bytes = FIXTURE_PDF.read_bytes()
    monkeypatch.setattr(
        native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: "(cid:1) (cid:2) \x01 soup"
    )

    regions = [
        [
            {"label": "header", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
            {"label": "footer", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
        ]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=5, eligible_labels=resolve_eligible_labels(True)
    )

    for region in filled[0]:
        assert "_native_text_used" not in region, region["label"]


def test_fill_respects_min_chars_threshold():
    """fill_regions_from_native_text skips / fills based on the min_chars threshold."""
    pdf_bytes = FIXTURE_PDF.read_bytes()
    full_page_bbox = [0, 0, 1000, 1000]
    total_chars = len(
        get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=full_page_bbox)
    )
    assert total_chars > 0, "fixture PDF must have native text"

    def _make_regions():
        return [[{"label": "text", "task_type": "text", "bbox_2d": full_page_bbox, "content": ""}]]

    # min_chars one above total: should NOT fill
    filled = fill_regions_from_native_text(
        pdf_bytes, _make_regions(), min_chars=total_chars + 1, eligible_labels=frozenset({"text"})
    )
    assert filled[0][0].get("_native_text_used", False) is False

    # min_chars ten below total: should fill
    filled = fill_regions_from_native_text(
        pdf_bytes, _make_regions(), min_chars=total_chars - 10, eligible_labels=frozenset({"text"})
    )
    assert filled[0][0]["_native_text_used"] is True


def test_fill_accepts_short_heading_native_text_below_body_min_chars(monkeypatch):
    """Short layout headings should use the native PDF text instead of OCR fallback."""
    import bibr.ocr.native_text as native_mod

    pdf_bytes = FIXTURE_PDF.read_bytes()
    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: "Method")

    regions = [
        [
            {
                "label": "paragraph_title",
                "task_type": "text",
                "bbox_2d": [0, 0, 1000, 1000],
                "content": "",
            },
            {"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
        ]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes,
        regions,
        min_chars=20,
        eligible_labels=frozenset({"paragraph_title", "text"}),
    )

    assert filled[0][0]["_native_text_used"] is True
    assert filled[0][0]["content"] == "Method"
    assert filled[0][1].get("_native_text_used", False) is False
    assert filled[0][1]["content"] == ""


def test_fill_is_idempotent_on_scanned_pdf():
    scanned = (FIXTURE_PDF.parent / "scanned_sample.pdf").read_bytes()
    regions = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        scanned, regions, min_chars=5, eligible_labels=frozenset({"text"})
    )
    assert filled[0][0].get("_native_text_used", False) is False
    assert filled[0][0]["content"] == ""


def test_printable_ratio_clean_ascii_text_is_high():
    text = "The quick brown fox jumps over the lazy dog. It was a bright cold day in April."
    assert _printable_ratio(text) == pytest.approx(1.0)


def test_printable_ratio_clean_multilingual_scientific_text_is_high():
    """Diacritics, Greek letters, math symbols, dashes, curly quotes are all 'reasonable'."""
    text = (
        "Müller, J., & Garcia-López, M. (2023). Föhn effects on café résumé—a "
        "pilot study of α, β, and Ω coefficients (p < .001, χ² = 4.2, r = .85 ± .03). "
        "The effect size (Cohen’s d = 0.62) was “significant” across ∑ conditions "
        "and ∫ f(x)dx converged for n = 30 participants."
    )
    assert _printable_ratio(text) > 0.9


def test_printable_ratio_control_char_soup_is_low():
    text = "".join(chr(c) for c in range(1, 20)) * 5
    assert _printable_ratio(text) < 0.3


def test_printable_ratio_cid_artifacts_counted_as_printable_chars():
    """(cid:N) runs are ASCII letters/digits/punctuation — caught by a separate
    artifact check, not by the printable-ratio calculation itself."""
    text = "(cid:72)(cid:88)(cid:94)" * 10
    assert _printable_ratio(text) == pytest.approx(1.0)


def test_printable_ratio_empty_text_is_treated_as_usable():
    assert _printable_ratio("") == 1.0


def test_is_native_text_usable_accepts_clean_scientific_text():
    text = (
        "Müller, J., & Garcia-López, M. (2023). Föhn effects on café résumé—a "
        "pilot study of α, β, and Ω coefficients (p < .001, χ² = 4.2, r = .85 ± .03). "
        "The effect size (Cohen’s d = 0.62) was “significant” across ∑ conditions."
    )
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is True


def test_is_native_text_usable_rejects_cid_artifact_runs():
    text = "(cid:72)(cid:88)(cid:94)(cid:98)(cid:100)(cid:103)" * 5
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False


def test_is_native_text_usable_rejects_fffd_heavy_text():
    text = "abc���def���ghi���" * 5
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False


def test_is_native_text_usable_rejects_control_char_soup():
    text = "".join(chr(c) for c in range(1, 20)) * 5
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False


def test_is_native_text_usable_rejects_symbol_gibberish():
    text = "".join(chr(c) for c in range(0xE000, 0xE030)) * 5  # Private Use Area
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False


def test_is_native_text_usable_threshold_is_configurable():
    """A noisy-but-not-garbage text is rejected at a strict threshold but
    accepted once the caller relaxes it via config."""
    text = "clean words here" + "★☆♦♣" * 5
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False
    assert _is_native_text_usable(text, min_printable_ratio=0.2) is True


def test_is_native_text_usable_accepts_normal_whitespace_controls():
    """Tab/newline/CR/FF/VT are legitimate whitespace, not corruption."""
    text = "Line one.\nLine two.\tTabbed.\r\nCRLF too.\x0cFF.\x0bVT." * 3
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is True


def test_is_native_text_usable_rejects_single_c0_control_char():
    """A single non-whitespace C0 control char (e.g. a broken CID leak, like
    U+000F from a LaTeX math font's broken ToUnicode CMap) must reject the
    whole native-text region, even though it barely dents the printable
    ratio in an otherwise clean paragraph."""
    text = (
        "This is a clean paragraph of scientific prose with plenty of words "
        "so that a single stray control character would never trip the "
        "overall printable-ratio gate on its \x0fown, yet it must still be "
        "rejected because genuine text never contains raw control bytes."
    )
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False


def test_is_native_text_usable_accepts_clean_paragraph_without_control_chars():
    text = (
        "This is a clean paragraph of scientific prose with plenty of words "
        "and no control characters anywhere, so it should be accepted as "
        "genuine native text extracted from the PDF's text layer."
    )
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is True


def test_fill_skips_region_with_corrupt_cid_native_text(monkeypatch):
    """A region whose native text is corrupt (cid artifacts) must be left
    unfilled so the OCR stage falls back to GLM-OCR for it, even though the
    text clears the min_chars threshold."""
    import bibr.ocr.native_text as native_mod

    pdf_bytes = FIXTURE_PDF.read_bytes()
    corrupt_text = "(cid:72)(cid:88)(cid:94)(cid:98)(cid:100)" * 10

    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: corrupt_text)

    regions = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=5, eligible_labels=frozenset({"text"})
    )
    assert filled[0][0].get("_native_text_used", False) is False
    assert filled[0][0]["content"] == ""


def test_fill_resolves_stx_soft_hyphens_before_corruption_gate(monkeypatch):
    """PDFium uses STX for benign line-end hyphens in some native text layers.

    Resolve the marker with the shared OCR-artifact logic instead of rejecting
    the whole region as a corrupt CMap. The resolver joins wrapped words while
    preserving genuine compound hyphens.
    """
    import bibr.ocr.native_text as native_mod

    pdf_bytes = FIXTURE_PDF.read_bytes()
    native_text = "Interviews were con\x02ducted with self\x02proclaimed users."
    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: native_text)

    regions = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=5, eligible_labels=frozenset({"text"})
    )

    assert filled[0][0]["_native_text_used"] is True
    assert filled[0][0]["content"] == "Interviews were conducted with self-proclaimed users."


@pytest.mark.parametrize(
    "critical_label",
    ["text", "content", "vertical_text", "reference", "reference_content"],
)
def test_fill_rejects_source_backed_private_use_critical_region_only(monkeypatch, critical_label):
    """A single PUA glyph in a citation-critical region forces OCR, while a
    clean sibling remains on the native path with its exact source geometry."""
    import bibr.ocr.native_text as native_mod

    source = json.loads(PUA_SOURCE_FIXTURE.read_text(encoding="utf-8"))
    clean, corrupt = source["regions"]
    by_bbox = {
        tuple(clean["bbox_2d"]): clean["native_text"],
        tuple(corrupt["bbox_2d"]): corrupt["native_text"],
    }
    monkeypatch.setattr(
        native_mod,
        "_text_from_bbox_on_textpage",
        lambda _textpage, _crop_box, bbox, **_kwargs: by_bbox[tuple(bbox)],
    )
    regions = [
        [
            {"label": clean["label"], "task_type": "text", "bbox_2d": clean["bbox_2d"]},
            {"label": critical_label, "task_type": "text", "bbox_2d": corrupt["bbox_2d"]},
        ]
    ]

    fill_regions_from_native_text(FIXTURE_PDF.read_bytes(), regions, min_chars=5)

    assert source["source_page"] == 3
    assert regions[0][0]["_native_text_used"] is True
    assert regions[0][0]["content"] == clean["native_text"]
    assert "_native_text_candidate" not in regions[0][0]
    assert regions[0][1].get("_native_text_used") is None
    assert regions[0][1]["_native_text_candidate"] == corrupt["native_text"]
    assert regions[0][1]["_native_text_rejection_reason"] == "private_use"


def test_private_use_fallback_is_label_scoped_not_global(monkeypatch):
    """PUA fallback must not alter the general usability predicate or broaden
    beyond citation-critical labels."""
    import bibr.ocr.native_text as native_mod

    text = "A sufficiently long document title with one private glyph  in context"
    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *_a, **_k: text)
    regions = [[{"label": "doc_title", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000]}]]

    fill_regions_from_native_text(FIXTURE_PDF.read_bytes(), regions, min_chars=5)

    assert _is_native_text_usable(text, min_printable_ratio=0.85) is True
    assert regions[0][0]["_native_text_used"] is True
    assert regions[0][0]["content"] == text
    assert "_native_text_rejection_reason" not in regions[0][0]


def test_fill_skips_region_with_fffd_heavy_native_text(monkeypatch):
    import bibr.ocr.native_text as native_mod

    pdf_bytes = FIXTURE_PDF.read_bytes()
    corrupt_text = "�" * 40

    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: corrupt_text)

    regions = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=5, eligible_labels=frozenset({"text"})
    )
    assert filled[0][0].get("_native_text_used", False) is False


def test_fill_respects_custom_min_printable_ratio(monkeypatch):
    """Callers can relax the printable-ratio gate (e.g. via config) to accept
    noisier-but-not-garbage text."""
    import bibr.ocr.native_text as native_mod

    pdf_bytes = FIXTURE_PDF.read_bytes()
    noisy_text = "clean words here" + "★☆♦♣" * 5

    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: noisy_text)

    regions = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes,
        regions,
        min_chars=5,
        eligible_labels=frozenset({"text"}),
        min_printable_ratio=0.85,
    )
    assert filled[0][0].get("_native_text_used", False) is False

    filled = fill_regions_from_native_text(
        pdf_bytes,
        regions,
        min_chars=5,
        eligible_labels=frozenset({"text"}),
        min_printable_ratio=0.2,
    )
    assert filled[0][0]["_native_text_used"] is True


def test_fill_partial_mutation_can_be_cleaned_up():
    """If fill is interrupted mid-iteration, the caller can restore regions
    to an OCR-eligible state by clearing the _native_text_used flag and content."""
    # Simulate a partial mutation: first region flagged, second not.
    pages_regions = [
        [
            {
                "label": "text",
                "bbox_2d": [0, 0, 1000, 500],
                "content": "pre-filled",
                "_native_text_used": True,
            },
            {"label": "text", "bbox_2d": [0, 500, 1000, 1000], "content": ""},
        ]
    ]
    # Caller cleanup (what the pipeline hooks' except blocks do):
    for page in pages_regions:
        for r in page:
            if r.pop("_native_text_used", False):
                r["content"] = ""

    assert pages_regions[0][0].get("_native_text_used") is None
    assert pages_regions[0][0]["content"] == ""
    assert pages_regions[0][1].get("_native_text_used") is None
    assert pages_regions[0][1]["content"] == ""


# --- Page /Rotate (audit [16]) ----------------------------------------------


def _rotated_pdf(rotate: int) -> bytes:
    """A 600x800 page with ALPHA near the top-left and OMEGA near the
    bottom-right in *unrotated* user space, carrying ``/Rotate rotate``."""
    content = b"BT /F1 24 Tf 50 750 Td (ALPHA) Tj ET\nBT /F1 24 Tf 460 50 Td (OMEGA) Tj ET\n"
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        (
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 600 800]/Rotate "
            + str(rotate).encode()
            + b"/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>"
        ),
        b"<</Length " + str(len(content)).encode() + b">>stream\n" + content + b"endstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(index).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer<</Size "
        + str(len(objects) + 1).encode()
        + b"/Root 1 0 R>>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


_ROTATION_QUADRANTS = {
    # (rotation) -> (quadrant holding ALPHA, quadrant holding OMEGA) in the
    # rendered image the layout detector indexes. Verified against
    # page.render(): rotation moves the ink, and the text layer does not move
    # with it.
    0: ("top-left", "bottom-right"),
    90: ("top-right", "bottom-left"),
    180: ("bottom-right", "top-left"),
    270: ("bottom-left", "top-right"),
}

_QUADRANT_BBOX = {
    "top-left": [0, 0, 300, 300],
    "top-right": [700, 0, 1000, 300],
    "bottom-left": [0, 700, 300, 1000],
    "bottom-right": [700, 700, 1000, 1000],
}


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_native_text_follows_page_rotation(rotation):
    """``page.render()`` applies ``/Rotate``, so on a 90/270 page the image is
    transposed while the CropBox and char boxes stay unrotated. Mapping the
    layout bbox through the CropBox alone returned '' for every text region."""
    pdf_bytes = _rotated_pdf(rotation)
    alpha_quadrant, omega_quadrant = _ROTATION_QUADRANTS[rotation]

    alpha = get_native_text_in_bbox(
        pdf_bytes, page_idx=0, bbox_normalized=_QUADRANT_BBOX[alpha_quadrant]
    )
    omega = get_native_text_in_bbox(
        pdf_bytes, page_idx=0, bbox_normalized=_QUADRANT_BBOX[omega_quadrant]
    )

    assert "ALPHA" in alpha
    assert "OMEGA" not in alpha
    assert "OMEGA" in omega
    assert "ALPHA" not in omega


# --- CropBox origin (audit [33]) --------------------------------------------


def test_bbox_pdf_pts_shares_the_frame_page_dimensions_describe():
    """``page.get_size()`` reports the CropBox extent while ``_bbox_pdf_pts``
    included the CropBox origin, so ``0 <= x1 <= x2 <= page_w`` was false on
    any page whose CropBox does not start at (0, 0) — and
    ``front_role_features.from_pdf_bbox`` divides by exactly those dimensions.
    """
    import pypdfium2

    from bibr.ocr.native_text import (
        _attach_bbox_pdf_pts,
        _attach_page_dimensions,
        _page_crop_box,
        _page_rotation,
    )

    fixture = Path(__file__).parent / "fixtures" / "cropbox_offset_sample.pdf"
    doc = pypdfium2.PdfDocument(fixture.read_bytes())
    try:
        page = doc[0]
        regions = [{"label": "text", "bbox_2d": [0, 0, 1000, 1000]}]
        _attach_page_dimensions(page, regions)
        _attach_bbox_pdf_pts(_page_crop_box(page), regions, _page_rotation(page))
        page.close()
    finally:
        doc.close()

    region = regions[0]
    x1, y1, x2, y2 = region["_bbox_pdf_pts"]
    assert 0 <= x1 <= x2 <= region["_page_w"] + 0.01
    assert 0 <= y1 <= y2 <= region["_page_h"] + 0.01
