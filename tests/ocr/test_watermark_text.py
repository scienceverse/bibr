"""Diagonal watermarks stay out of the native text layer.

Review copies and stamped PDFs draw a large diagonal string over every page.
Its glyphs used to land in whatever region box they crossed ("e\\r\\nPotential
confounding variables", "IV PROPOSED METHOD\\r\\nR"), skew region font sizes,
and, drawn first, make pdfium break the line before every one-glyph text object
("buyer\\r\\n-supplier"). The fixtures are minimal hand-built PDFs that copy the
shapes of the real cases: a 48 pt string at -51.5 degrees drawn before word-group
text objects, a 100 pt stamp at 50 degrees drawn last, and a stamp set upright
inside a rotated form. Small rotated chart labels, 90 degree side stamps and
sheared (synthetic italic) text are not watermarks and must survive.
"""

from __future__ import annotations

import math

import pypdfium2

from bibr.ocr.native_text import fill_native_text_and_fonts, get_native_text_in_bbox
from bibr.ocr.pdf_inspection import inspect_pdf, inspection_from_dict, inspection_to_dict

_PAGE_W, _PAGE_H = 612.0, 792.0

# Helvetica advance widths (1/1000 em) for the characters the fixtures use.
_WIDTHS = dict.fromkeys("abcdeghnopqu0123456789", 556)
_WIDTHS.update({"f": 278, "i": 222, "j": 222, "l": 222, "m": 833, "r": 333, "s": 500, "t": 278})
_WIDTHS.update({"v": 500, "w": 722, "x": 500, "y": 500, "k": 500, " ": 278, "-": 333})
_WIDTHS.update({",": 278, ".": 278, ":": 278, "(": 333, ")": 333})
_WIDTHS.update({"/": 278, "@": 1015, "?": 556, "=": 584, "_": 556})
_UPPER_WIDTHS = (667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833)
_UPPER_WIDTHS += (722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611)
_WIDTHS.update(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", _UPPER_WIDTHS, strict=True))

# Word groups drawn as separate text objects, the way the review copies draw them.
_LINES = [
    ["such as buyer", "-", "supplier projects", ", are exposed"],
    ["to supplier opportunism due to the uncertain"],
    ["nature of the collaboration process. Some pre", "-"],
    ["existing contexts matter for the perspective", "s", ", we argue"],
]
_LINE_COUNT = 24
_TOP_BASELINE = 720.0
_LEADING = 27.6


def _num(value: float) -> bytes:
    return f"{value:.4f}".encode()


def _bt(text: str, x: float, y: float, *, size: float = 12.0, angle: float = 0.0, shear=0.0):
    a = math.radians(angle)
    matrix = (math.cos(a), math.sin(a), shear - math.sin(a), math.cos(a), x, y)
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return (
        b"BT /F1 "
        + _num(size)
        + b" Tf "
        + b" ".join(_num(v) for v in matrix)
        + b" Tm ("
        + escaped.encode("latin-1")
        + b") Tj ET\n"
    )


def _width(text: str, size: float = 12.0) -> float:
    return sum(_WIDTHS.get(c, 667) for c in text) / 1000.0 * size


def _baseline(index: int) -> float:
    return _TOP_BASELINE - _LEADING * index


def _body() -> bytes:
    out = b""
    for index in range(_LINE_COUNT):
        x = 72.0
        for group in _LINES[index % len(_LINES)]:
            out += _bt(group, x, _baseline(index))
            x += _width(group)
    return out


def _pdf(
    content: bytes,
    form: bytes | None = None,
    *,
    pages: int = 1,
    extra_pages: tuple[bytes, ...] = (),
) -> bytes:
    """A PDF of *pages* identical pages, then one page per *extra_pages* content.

    *form* is the content of a Form XObject ``/Fm1``.
    """
    objects: list[bytes] = [b"", b""]

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    xobjects = b""
    if form is not None:
        form_id = add(
            b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Length %d >>\nstream\n"
            % (font, len(form))
            + form
            + b"\nendstream"
        )
        xobjects = b" /XObject << /Fm1 %d 0 R >>" % form_id
    streams = [
        add(b"<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream")
        for body in (content, *extra_pages)
    ]
    kids = [
        add(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >>"
            % font
            + xobjects
            + b" >> /Contents %d 0 R >>" % stream
        )
        for stream in [streams[0]] * pages + streams[1:]
    ]
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % kid for kid in kids),
        len(kids),
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def _line_region(index: int, label: str = "text") -> dict:
    """A layout box around one body line, wide enough to cross the watermark."""
    top = _baseline(index) + 11.0
    bottom = _baseline(index) - 4.0
    return {
        "label": label,
        "bbox_2d": [
            50.0 / _PAGE_W * 1000.0,
            (_PAGE_H - top) / _PAGE_H * 1000.0,
            560.0 / _PAGE_W * 1000.0,
            (_PAGE_H - bottom) / _PAGE_H * 1000.0,
        ],
        "content": "",
    }


def _line_texts(pdf_bytes: bytes) -> list[dict]:
    regions = [_line_region(index) for index in range(_LINE_COUNT)]
    fill_native_text_and_fonts(pdf_bytes, [regions], min_chars=3)
    return regions


# ("For Review Only", 48 pt at -51.5 degrees, drawn before the body: a ScholarOne review copy).
_REVIEW_WATERMARK = _bt("For Review Only", 177.8, 640.0, size=48.0, angle=-51.5)


def test_first_drawn_watermark_leaves_the_text_of_the_clean_page():
    clean = _line_texts(_pdf(_body()))
    stamped = _line_texts(_pdf(_REVIEW_WATERMARK + _body()))

    assert [r["content"] for r in stamped] == [r["content"] for r in clean]
    assert stamped[0]["content"] == "such as buyer-supplier projects, are exposed"


def test_first_drawn_watermark_does_not_break_lines_before_one_glyph_objects():
    texts = [r["content"] for r in _line_texts(_pdf(_REVIEW_WATERMARK + _body()))]

    assert not any("\r\n" in text for text in texts)
    assert "the perspectives, we argue" in texts[3]


def test_watermark_glyphs_do_not_skew_region_font_size():
    clean = _line_texts(_pdf(_body()))
    stamped = _line_texts(_pdf(_REVIEW_WATERMARK + _body()))

    assert [r.get("_font_size") for r in stamped] == [r.get("_font_size") for r in clean]
    assert all(r.get("_font_size") is None or r["_font_size"] < 12.0 for r in stamped)


def test_last_drawn_stamp_is_not_appended_to_headings():
    """A 100 pt "RETRACTED" stamp at 50 degrees, drawn after the page (a retracted paper)."""
    heading = _bt("IV PROPOSED METHOD", 72.0, _baseline(6))
    stamp = _bt("RETRACTED", 130.0, 150.0, size=100.0, angle=50.0)
    pdf_bytes = _pdf(_body() + heading + stamp)
    regions = [_line_region(6, "paragraph_title")]

    fill_native_text_and_fonts(pdf_bytes, [regions], min_chars=3)

    assert "R\r\n" not in regions[0]["content"]
    assert regions[0]["content"].endswith("IV PROPOSED METHOD")


def test_upright_stamp_inside_a_rotated_form_is_removed():
    """ "ARTICLE IN PRESS" set at Tf 1 inside a form the page draws rotated by 35 degrees."""
    form = b"BT /F1 1 Tf 30 0 0 30 0 0 Tm (ARTICLE IN PRESS) Tj ET"
    a = math.radians(35.0)
    rotate = b"q " + b" ".join(
        _num(v) for v in (math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 110.0, 200.0)
    )
    pdf_bytes = _pdf(_body() + rotate + b" cm /Fm1 Do Q\n", form=form)

    assert [r["content"] for r in _line_texts(pdf_bytes)] == [
        r["content"] for r in _line_texts(_pdf(_body()))
    ]


def _figure_region() -> dict:
    return {"label": "text", "bbox_2d": [50.0, 50.0, 950.0, 950.0], "content": ""}


def test_small_rotated_chart_labels_are_kept():
    labels = b"".join(
        _bt(name, 120.0 + 60.0 * i, 300.0, size=8.0, angle=45.0)
        for i, name in enumerate(["Baseline", "Week four", "Follow-up"])
    )
    regions = [_figure_region()]

    fill_native_text_and_fonts(_pdf(labels), [regions], min_chars=3)

    for name in ("Baseline", "Week four", "Follow-up"):
        assert name in regions[0]["content"]


def test_side_stamp_at_ninety_degrees_is_kept():
    stamp = _bt("arXiv side stamp kept", 30.0, 200.0, size=20.0, angle=90.0)

    text = get_native_text_in_bbox(_pdf(_body() + stamp), 0, [0.0, 0.0, 1000.0, 1000.0])

    assert "arXiv side stamp kept" in text


def test_large_sheared_italic_title_is_kept():
    """Synthetic italics shear the text matrix; FPDFText_GetCharAngle reads that as rotation."""
    title = _bt("A sheared display title", 72.0, 760.0, size=24.0, shear=0.33)

    text = get_native_text_in_bbox(_pdf(title + _body()), 0, [0.0, 0.0, 1000.0, 1000.0])

    assert "A sheared display title" in text


def _inspect(pdf_bytes: bytes):
    return inspect_pdf(
        pdf_bytes,
        [[_line_region(0)]],
        fill_native_text=True,
        include_outline=False,
        include_ref_geometry=True,
        min_chars=3,
        min_printable_ratio=0.85,
    )


def test_inspection_records_the_removed_watermark_and_keeps_it_out_of_page_lines():
    inspection = _inspect(_pdf(_REVIEW_WATERMARK + _body()))

    assert inspection.pages[0].watermarks == ("For Review Only",)
    assert inspection.layout_results[0][0]["content"] == (
        "such as buyer-supplier projects, are exposed"
    )
    line_texts = [line["text"] for line in inspection.page_lines]
    assert line_texts[0] == "such as buyer-supplier projects, are exposed"
    assert not any("Review" in text for text in line_texts)


def test_inspection_of_a_clean_page_records_no_watermark():
    assert _inspect(_pdf(_body())).pages[0].watermarks == ()


def test_watermarks_survive_the_cache_round_trip():
    inspection = _inspect(_pdf(_REVIEW_WATERMARK + _body()))

    restored = inspection_from_dict(
        inspection_to_dict(inspection), metadata=None, outline=None, reference_lines=None
    )

    assert restored is not None
    assert restored.pages[0].watermarks == ("For Review Only",)


def test_stripping_changes_only_the_loaded_page():
    pdf_bytes = _pdf(_REVIEW_WATERMARK + _body())

    _line_texts(pdf_bytes)

    doc = pypdfium2.PdfDocument(pdf_bytes)
    try:
        assert "For Review Only" in doc[0].get_textpage().get_text_range()
    finally:
        doc.close()
