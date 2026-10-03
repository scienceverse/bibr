"""Spacing accents set over or under a letter are composed with it.

TeX and some other generators set an accented letter as the letter plus a
spacing accent glyph moved onto it, and pdfium reports the two chars in either
order: reference and author names came out as "Bas¸kaya", "Brˇci´c",
"Ca´rcel". The fixtures draw each glyph as its own Helvetica object with
StandardEncoding codes for the accents, centred on the letter.
"""

from __future__ import annotations

from bibr.ocr.native_text import get_native_text_in_bbox
from bibr.ocr.pdf_inspection import inspect_pdf
from tests.ocr.test_watermark_text import _bt, _pdf, _width

_WHOLE_PAGE = [0.0, 0.0, 1000.0, 1000.0]
# StandardEncoding codes and Helvetica widths of the accent glyphs.
_ACCENTS = {
    "´": ("\xc2", 333),
    "ˇ": ("\xcf", 333),
    "¸": ("\xcb", 333),
    "¨": ("\xc8", 333),
    "`": ("\xc1", 333),
    "ˆ": ("\xc3", 333),
    "˜": ("\xc4", 333),
    "¯": ("\xc5", 333),
}
_DOTLESS_I = ("\xf5", 278)


def _name(
    parts: list[tuple[str, str | None]],
    *,
    x: float = 72.0,
    y: float = 700.0,
    accent_first: bool = False,
    size: float = 12.0,
) -> bytes:
    """Draw *parts* as (letter, accent or None); an accent is centred on its letter."""
    out = b""
    for letter, accent in parts:
        code, width = _DOTLESS_I if letter == "ı" else (letter, None)
        advance = (width / 1000.0 * size) if width else _width(letter, size)
        letter_obj = _bt(code, x, y, size=size)
        accent_obj = b""
        if accent is not None:
            accent_code, accent_width = _ACCENTS[accent]
            ax = x + (advance - accent_width / 1000.0 * size) / 2.0
            accent_obj = _bt(accent_code, ax, y, size=size)
        out += accent_obj + letter_obj if accent_first else letter_obj + accent_obj
        x += advance
    return out


def _plain(text: str) -> list[tuple[str, str | None]]:
    return [(ch, None) for ch in text]


def _text(content: bytes) -> str:
    return get_native_text_in_bbox(_pdf(content), 0, _WHOLE_PAGE)


def test_caron_and_acute_before_their_letters():
    parts = _plain("Br") + [("c", "ˇ"), ("i", None), ("c", "´")]

    assert _text(_name(parts, accent_first=True)) == "Brčić"


def test_cedilla_after_its_letter():
    parts = _plain("Ba") + [("s", "¸")] + _plain("kaya")

    assert _text(_name(parts)) == "Başkaya"


def test_acute_after_its_letter():
    parts = [("C", None), ("a", "´")] + _plain("rcel")

    assert _text(_name(parts)) == "Cárcel"


def test_acute_on_a_dotless_i():
    parts = _plain("Mart") + [("ı", "´")] + _plain("nez")

    assert _text(_name(parts)) == "Martínez"


def test_acute_typed_as_an_apostrophe_stays():
    """The accent sits in its own advance between two letters, over neither."""
    content = _bt("don", 72.0, 700.0) + _bt("\xc2", 72.0 + _width("don"), 700.0)
    content += _bt("t", 72.0 + _width("don") + 4.0, 700.0)

    assert _text(content) == "don´t"


def test_reference_page_lines_are_composed():
    parts = _plain("Ba") + [("s", "¸")] + _plain("kaya, A. (2020). Title.")
    inspection = inspect_pdf(
        _pdf(_name(parts)),
        [[{"label": "text", "bbox_2d": [0.0, 0.0, 1000.0, 1000.0], "content": ""}]],
        fill_native_text=True,
        include_outline=False,
        include_ref_geometry=True,
        min_chars=3,
        min_printable_ratio=0.85,
    )

    assert [line["text"] for line in inspection.page_lines] == ["Başkaya, A. (2020). Title."]


def test_tilde_after_its_letter():
    parts = [*_plain("Pe"), ("n", "˜"), ("a", None)]

    assert _text(_name(parts)) == "Peña"


def test_circumflex_and_acute_before_their_letters():
    parts = [("C", None), ("o", "ˆ"), ("t", None), ("e", "´")]

    assert _text(_name(parts, accent_first=True)) == "Côté"


def test_grave_after_its_letter():
    parts = [*_plain("Universit"), ("a", "`")]

    assert _text(_name(parts)) == "Università"


def test_macron_after_its_letter():
    parts = [*_plain("T"), ("o", "¯"), *_plain("kyo")]

    assert _text(_name(parts)) == "Tōkyo"


def test_accent_at_a_line_end_does_not_land_on_the_next_line():
    """A trailing accent over the first letter of the line below stays put."""
    line = _name(_plain("Peoples"), y=700.0)
    end = 72.0 + _width("Peoples", 12.0)
    accent = _bt("\xc2", end, 700.0, size=12.0)
    below = _name(_plain("et al."), x=end - 1.0, y=688.0)

    text = _text(line + accent + below)

    assert "ét" not in text
    assert "et al." in text


def test_generated_space_after_a_composed_capital_is_dropped():
    parts = [("E", "´"), *_plain("rica")]

    assert _text(_name(parts)) == "Érica"
