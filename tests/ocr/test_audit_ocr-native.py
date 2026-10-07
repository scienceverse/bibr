"""Audit fixes in the PDF native-text pass.

The references-header regex, the word-boundary repair and the line-number
column detection answer exactly as before in linear time; font sampling no
longer costs a page its native text; reference lines keep chars beyond the
BMP. The timing guards use inputs the old code spent tens of seconds on while
the new code takes milliseconds, so their budget stays loose on a busy machine.
"""

from __future__ import annotations

import time
from copy import deepcopy

import pypdfium2
import pytest

from bibr.config import GlobalSettings
from bibr.document import views
from bibr.document.rebuild import render_budget
from bibr.ocr import native_text, pdf_inspection
from bibr.ocr.native_text import _line_number_column, _sample_font_metadata_in_bbox
from bibr.ocr.pdf_inspection import inspect_pdf
from bibr.ocr.ref_geometry import _extract_page_chars, recover_reference_lines
from bibr.ocr.ref_patterns import _REF_HEADER_RE
from tests.document import _pdfs
from tests.ocr.test_native_word_boundaries import _repair_line

# Budget for an input the old code took 15 s or more on.
_BUDGET_S = 1.0
_WHOLE_PAGE = {"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}
# Around the first line _lines draws, where font sampling finds its glyphs.
_FIRST_LINE = {"label": "text", "bbox_2d": [100, 95, 700, 125], "content": ""}


def _seconds(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def _pdf(content: bytes, to_unicode: dict[int, str] | None = None) -> bytes:
    """A one-page Helvetica PDF; *to_unicode* maps a code to UTF-16BE hex in a ToUnicode CMap."""
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
    ]
    if to_unicode:
        pairs = " ".join(f"<{code:02X}> <{target}>" for code, target in to_unicode.items())
        cmap = (
            "/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /X def "
            "1 begincodespacerange <00> <FF> endcodespacerange "
            f"{len(to_unicode)} beginbfchar {pairs} endbfchar endcmap "
            "CMapName currentdict /CMap defineresource pop end end"
        ).encode()
        objects.append(font + b" /ToUnicode 6 0 R >>")
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(cmap), cmap))
    else:
        objects.append(font + b" >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def _lines(*lines: str, size: float = 12.0) -> bytes:
    return b"".join(
        b"BT /F1 %.1f Tf 72 %.1f Td (%s) Tj ET\n" % (size, 700.0 - 20.0 * index, line.encode())
        for index, line in enumerate(lines)
    )


def _inspect(pdf_bytes: bytes, region: dict = _WHOLE_PAGE, **overrides):
    kwargs = {
        "fill_native_text": True,
        "include_outline": False,
        "include_ref_geometry": True,
        "min_chars": 10,
        "min_printable_ratio": 0.85,
    }
    kwargs.update(overrides)
    return inspect_pdf(pdf_bytes, [[deepcopy(region)]], **kwargs)


# ---------------------------------------------------------------------------
# References header
# ---------------------------------------------------------------------------

_HEADERS = [
    "References",
    "REFERENCES",
    "Reference",
    "References and Notes",
    "Literature Cited",
    "6 REFERENCES",
    "IV. Bibliography",
    "E. Referensi",
    "■ References",
    "• 7. References",
    "References:",
    "References :",
    "References.",
    "References»»»",
    "References : —",
    "Список використаних джерел:",
    "参考文献",
    "  References  ",
    "　References ",
    "References\t",
]


@pytest.mark.parametrize("header", _HEADERS)
def test_reference_headers_match_on_their_own_line_and_in_page_text(header):
    assert _REF_HEADER_RE.match(header)
    # pdfium's page text breaks lines with "\r\n".
    assert _REF_HEADER_RE.search(f"Body text ends here.\r\n{header}\r\n[1] Smith, J. (2020).")
    assert _REF_HEADER_RE.search(f"Body text.\n\n{header}\n")


def test_leading_blank_lines_still_open_a_matched_header():
    assert _REF_HEADER_RE.match("\n\n  References\n")
    assert _REF_HEADER_RE.match("\r\n6 References")


@pytest.mark.parametrize(
    "text",
    [
        "References to the literature are given below",
        "Reference list of things",
        "See bibliography for details",
        "References x",
        "1. Introduction",
        "References;",
    ],
)
def test_prose_is_still_not_a_references_header(text):
    assert not _REF_HEADER_RE.match(text)
    assert not _REF_HEADER_RE.search(f"Body.\r\n{text}\r\nMore body.")


@pytest.mark.parametrize(
    "text",
    [
        # Three adjacent whitespace runs after the word: cubic in the spaces.
        pytest.param("References" + " " * 2000 + "x", id="spaces"),
        pytest.param("References" + " " * 2000 + "x", id="em-spaces"),
        # Two after the colon: quadratic.
        pytest.param("References :" + " " * 40000 + "x", id="colon"),
        # Every line start running over all the blank lines below it.
        pytest.param("\n" * 12000, id="blank-lines"),
        pytest.param(" \r\n" * 8000, id="crlf-lines"),
    ],
)
def test_references_header_regex_is_linear(text):
    assert _seconds(_REF_HEADER_RE.search, text) < _BUDGET_S
    assert not _REF_HEADER_RE.search(text)


def test_em_space_padded_line_does_not_stall_pdf_inspection():
    # A 3 KB page whose text is "References", 2,000 em spaces and "x".
    pdf = _pdf(_lines("References" + "~" * 2000 + "x", size=0.2), {0x7E: "2003"})

    start = time.perf_counter()
    inspection = _inspect(pdf, fill_native_text=False)

    assert time.perf_counter() - start < _BUDGET_S
    assert inspection.reference_lines == []


# ---------------------------------------------------------------------------
# Font sampling never discards the native fill
# ---------------------------------------------------------------------------

_BODY = "The participants completed the survey in two sessions."


class _CharsTextPage:
    raw = object()

    def __init__(self, boxes):
        self._boxes = boxes

    def count_chars(self):
        return len(self._boxes)

    def get_charbox(self, index, loose=False):
        return self._boxes[index]


def test_font_sampling_reads_a_code_past_unicode_as_a_glyph(monkeypatch):
    # A broken ToUnicode map can hand pdfium a code above U+10FFFF.
    codes = [0x41, 0x110000, 0x43]
    boxes = [(10.0, 10.0, 20.0, 20.0), (21.0, 10.0, 30.0, 22.0), (31.0, 10.0, 40.0, 20.0)]
    monkeypatch.setattr(pypdfium2.raw, "FPDFText_GetUnicode", lambda _raw, i: codes[i])
    monkeypatch.setattr(pypdfium2.raw, "FPDFText_GetCharIndexAtPos", lambda *_args: 0)
    monkeypatch.setattr(pypdfium2.raw, "FPDFText_GetFontWeight", lambda *_args: 400)
    monkeypatch.setattr(native_text, "_char_is_italic", lambda _tp, _i: False)

    assert _sample_font_metadata_in_bbox(_CharsTextPage(boxes), 3, 0, 0, 100, 100) == (
        10.0,
        400,
        False,
    )


def test_code_past_unicode_keeps_the_page_native_text(monkeypatch):
    pdf = _pdf(_lines(_BODY))
    get_unicode = pypdfium2.raw.FPDFText_GetUnicode
    broken = _BODY.index("survey")
    monkeypatch.setattr(
        pypdfium2.raw,
        "FPDFText_GetUnicode",
        lambda raw, i: 0x110000 if i == broken else get_unicode(raw, i),
    )

    inspection = _inspect(pdf, _FIRST_LINE)

    region = inspection.layout_results[0][0]
    assert inspection.component_errors == {}
    assert region["_native_text_used"] is True
    assert region["content"] == _BODY.replace("survey", "�urvey")
    assert region["_font_size"] > 0


def test_font_sampling_failure_keeps_the_native_fill(monkeypatch):
    def fail(*_args):
        raise RuntimeError("font sampling failed")

    monkeypatch.setattr(pdf_inspection, "_sample_page_font_metadata", fail)

    inspection = _inspect(_pdf(_lines(_BODY)), _FIRST_LINE)

    region = inspection.layout_results[0][0]
    assert region["_native_text_used"] is True
    assert region["content"] == _BODY
    assert "_font_size" not in region
    assert set(inspection.component_errors) == {"font_metadata:0"}


# ---------------------------------------------------------------------------
# Word-boundary repair
# ---------------------------------------------------------------------------


def test_long_line_of_inserted_spaces_is_repaired_in_linear_time():
    # "~" is a word-sized gap with nothing in it; each one re-walked the line.
    line = "x;~" * 8000 + "x"

    assert _seconds(_repair_line, line) < _BUDGET_S
    assert _repair_line(line) == "x; " * 8000 + "x"


def test_long_link_run_stays_closed_in_linear_time():
    line = "https://osf.io/" + "a;~" * 6000 + "b"

    assert _seconds(_repair_line, line) < _BUDGET_S
    assert _repair_line(line) == "https://osf.io/" + "a;" * 6000 + "b"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("x;~x;~x|https://osf.io/a;~b", "x; x; x https://osf.io/a;b"),
        ("x;~(https://osf.io/a;~b);~[1];~y", "x; (https://osf.io/a;b); [1]; y"),
        ("x;~x|jane@uni.edu;~x", "x; x jane@uni.edu; x"),
        ("mail|jane@uni.edu,~x;~y", "mail jane@uni.edu, x; y"),
        ("a;~b|me@x.org;~c@d.ef;~g", "a; b me@x.org; c@d.ef; g"),
    ],
)
def test_links_on_a_repaired_line_still_keep_their_gaps_closed(line, expected):
    # Several runs on one line, each with its own links.
    assert _repair_line(line) == expected


# ---------------------------------------------------------------------------
# Line-number column
# ---------------------------------------------------------------------------


def test_line_number_column_is_found_in_linear_time():
    count = 6000
    pitch = 790.0 / count

    def band(i: int, left: float, right: float, shift: float = 0.0):
        bottom = 790.0 - (i + 1) * pitch + shift * pitch
        return (left, bottom, right, bottom + 0.8 * pitch)

    numbers = [(i + 1, band(i, 20.0, 30.0)) for i in range(count)]
    # Text 10 pt from the column (never touching it), and margin text further
    # out that sits between the numbers' lines.
    others = [band(i, 40.0, 500.0) for i in range(count)]
    others += [band(i, 2.0, 8.0, shift=0.5) for i in range(count)]

    assert _seconds(_line_number_column, numbers, others, 0.0, 612.0) < _BUDGET_S
    assert _line_number_column(numbers, others, 0.0, 612.0) == set(range(count))


def test_text_touching_most_numbers_still_keeps_the_column():
    numbers = [(i + 1, (20.0, 700.0 - 14.0 * i, 30.0, 708.0 - 14.0 * i)) for i in range(12)]
    apart = [(40.0, 700.0 - 14.0 * i, 500.0, 708.0 - 14.0 * i) for i in range(12)]
    touching = [(32.0, 700.0 - 14.0 * i, 500.0, 708.0 - 14.0 * i) for i in range(12)]
    further_out = [(2.0, 700.0 - 14.0 * i, 8.0, 708.0 - 14.0 * i) for i in range(12)]

    assert _line_number_column(numbers, apart, 0.0, 612.0) == set(range(12))
    assert _line_number_column(numbers, touching, 0.0, 612.0) == set()
    assert _line_number_column(numbers, apart + further_out, 0.0, 612.0) == set()


# ---------------------------------------------------------------------------
# Reference lines beyond the BMP
# ---------------------------------------------------------------------------

_MATH_S = "\U0001d446"  # MATHEMATICAL ITALIC CAPITAL S


def _surrogate_pdf() -> bytes:
    # "Q" reads as U+1D446, which pdfium holds as a UTF-16 surrogate pair.
    content = _lines("References", "Smith, J. (2020). The Q statistic.")
    return _pdf(content, {0x51: "D835DC46"})


def test_reference_lines_keep_a_char_held_as_a_surrogate_pair():
    lines = recover_reference_lines(_surrogate_pdf())

    assert [line.text for line in lines] == [f"Smith, J. (2020). The {_MATH_S} statistic."]


def test_reference_chars_keep_a_char_held_as_one_code():
    line = f"Let {_pdfs.MATH_ALPHA_CODE} be small"
    pdf = _pdfs.build_pdf([_pdfs.PageSpec(_pdfs.text(line, 72.0, 700.0, size=12.0, font="F4"))])
    doc = pypdfium2.PdfDocument(pdf)
    try:
        textpage = doc[0].get_textpage()
        try:
            chars = _extract_page_chars(textpage)
        finally:
            textpage.close()
    finally:
        doc.close()

    assert "".join(ch for ch, _box in chars) == line.replace(
        _pdfs.MATH_ALPHA_CODE, _pdfs.MATH_ALPHA
    )


def test_document_layer_lines_match_the_line_stream_for_a_surrogate_pair():
    inspection = _inspect(
        _surrogate_pdf(), include_doc_layer=True, render_budget=render_budget(GlobalSettings())
    )

    lines = [line for line in inspection.page_lines if line["page"] == 1]
    assert [line["text"] for line in lines] == [
        "References",
        f"Smith, J. (2020). The {_MATH_S} statistic.",
    ]
    assert views.page_lines(inspection.document.page(0)) == lines
