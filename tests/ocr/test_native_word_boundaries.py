"""Word gaps and mid-line breaks are repaired from glyph geometry.

pdfium decides whitespace per text object. A PDF that draws one glyph per
positioned object (submission-system cover sheets, some generators) can come
out with no space at a word gap ("arterialand") or with a generated line break
in the middle of a line ("buyer\\r\\n-supplier", "perspective\\r\\ns"). The
fixtures draw the same shapes with Helvetica: each glyph its own object, word
gaps without a space character, word groups split around punctuation.
"""

from __future__ import annotations

import pypdfium2
import pytest

from bibr.ocr.native_text import (
    _build_page_char_records,
    _reconstruct_text_from_records,
    _repair_word_boundaries,
    get_native_text_in_bbox,
)
from tests.ocr.test_watermark_text import (
    _REVIEW_WATERMARK,
    _bt,
    _line_region,
    _pdf,
    _width,
)
from tests.ocr.test_watermark_text import _body as _word_group_page

_WHOLE_PAGE = [0.0, 0.0, 1000.0, 1000.0]


def _glyph_line(text: str, x: float, y: float, *, gap: float, size: float = 10.0) -> bytes:
    """Each glyph of *text* is its own text object; a space advances by *gap* and draws nothing."""
    out = b""
    for ch in text:
        if ch == " ":
            x += gap
            continue
        out += _bt(ch, x, y, size=size)
        x += _width(ch, size)
    return out


def _letter_spaced(text: str, x: float, y: float, *, spacing: float, size: float = 10.0) -> bytes:
    out = b""
    for ch in text:
        out += _bt(ch, x, y, size=size)
        x += _width(ch, size) + spacing
    return out


def _page_text(pdf_bytes: bytes) -> str:
    """Region text of the whole page from a text page built without the watermark strip."""
    doc = pypdfium2.PdfDocument(pdf_bytes)
    try:
        textpage = doc[0].get_textpage()
        try:
            records = _build_page_char_records(textpage)
        finally:
            textpage.close()
    finally:
        doc.close()
    return _reconstruct_text_from_records(records, 0.0, 0.0, 612.0, 792.0)


def test_word_gap_without_a_space_glyph_gets_a_space():
    """A cover sheet's title line: pdfium emits "majorarterialandvenous"."""
    pdf_bytes = _pdf(_glyph_line("major arterial and venous diseases: a cohort", 72, 700, gap=1.6))

    assert get_native_text_in_bbox(pdf_bytes, 0, _WHOLE_PAGE) == (
        "major arterial and venous diseases: a cohort"
    )


def test_generated_breaks_between_one_glyph_objects_are_joined():
    """pdfium breaks the line before every one-glyph object of the same line."""
    pdf_bytes = _pdf(_glyph_line("major arterial and venous", 72, 700, gap=2.9))

    assert get_native_text_in_bbox(pdf_bytes, 0, _WHOLE_PAGE) == "major arterial and venous"


def test_generated_break_before_punctuation_on_the_same_baseline_is_joined():
    """The review-copy shape: "buyer\\r\\n-supplier", "perspective\\r\\ns, we argue"."""
    text = _page_text(_pdf(_REVIEW_WATERMARK + _word_group_page()))

    assert "buyer-supplier projects, are exposed" in text
    assert "the perspectives, we argue" in text
    assert "\r\n-" not in text
    assert "perspective\r\ns" not in text


@pytest.mark.parametrize("spacing", [1.45, 1.6, 3.0])
def test_letter_spaced_heading_gets_no_extra_space(spacing):
    """Every gap is word-sized, but no in-word gap is next to it: nothing to split."""
    pdf_bytes = _pdf(_letter_spaced("ABSTRACT", 72, 700, spacing=spacing))
    doc = pypdfium2.PdfDocument(pdf_bytes)
    try:
        pdfium_text = doc[0].get_textpage().get_text_range()
    finally:
        doc.close()

    assert get_native_text_in_bbox(pdf_bytes, 0, _WHOLE_PAGE) == pdfium_text


def test_tight_glyphs_of_one_word_are_not_split():
    pdf_bytes = _pdf(_glyph_line("Association", 72, 700, gap=0.0))

    assert get_native_text_in_bbox(pdf_bytes, 0, _WHOLE_PAGE) == "Association"


def test_affiliation_superscript_stays_attached():
    """A raised, smaller "1" before "Department" is not on the word's line."""
    marker = _bt("1", 72.0, 704.0, size=6.0)
    word = _bt("Department of Psychology", 72.0 + _width("1", 6.0) + 1.5, 700.0)

    text = get_native_text_in_bbox(_pdf(marker + word), 0, _WHOLE_PAGE)

    assert "1 Department" not in text
    assert text.replace("\r\n", "") == "1Department of Psychology"


def test_printed_spaces_and_line_breaks_are_untouched():
    lines = _bt("First printed line of text.", 72, 700) + _bt("Second line.", 72, 686)

    assert get_native_text_in_bbox(_pdf(lines), 0, _WHOLE_PAGE) == (
        "First printed line of text.\r\nSecond line."
    )


def test_region_fill_of_a_glyph_by_glyph_title():
    from bibr.ocr.native_text import fill_regions_from_native_text

    pdf_bytes = _pdf(_glyph_line("Association of COVID-19 with venous", 72, _line_y(0), gap=1.6))
    regions = [_line_region(0, "doc_title")]

    fill_regions_from_native_text(pdf_bytes, [regions], min_chars=3)

    assert regions[0]["content"] == "Association of COVID-19 with venous"


def _line_y(index: int) -> float:
    from tests.ocr.test_watermark_text import _baseline

    return _baseline(index)


def _spaced_glyphs(text: str, x: float, y: float, *, loose: str, gap: float, size: float = 10.0):
    """One object per glyph: *gap* points around the glyphs in *loose*, 0.2 pt elsewhere."""
    out = b""
    for index, ch in enumerate(text):
        out += _bt(ch, x, y, size=size)
        following = text[index + 1] if index + 1 < len(text) else ""
        x += _width(ch, size) + (gap if ch in loose or following in loose else 0.2)
    return out


def test_loosely_set_doi_link_gets_no_extra_space():
    """A justified reference line spaces the dots and slashes of its DOI link:
    "https ://doi . org / 10 . 0000 /" broke the DOI."""
    link = "https://doi.org/10.0000/0000000000000000"
    pdf_bytes = _pdf(_spaced_glyphs(link, 72, 700, loose=".:/", gap=1.6))
    doc = pypdfium2.PdfDocument(pdf_bytes)
    try:
        pdfium_text = doc[0].get_textpage().get_text_range()
    finally:
        doc.close()

    assert get_native_text_in_bbox(pdf_bytes, 0, _WHOLE_PAGE) == pdfium_text
    assert "doi.org" in pdfium_text


def test_glued_sentence_after_a_full_stop_still_splits():
    pdf_bytes = _pdf(_glyph_line("of toddler temperament. As a part", 72, 700, gap=1.6))

    assert get_native_text_in_bbox(pdf_bytes, 0, _WHOLE_PAGE) == "of toddler temperament. As a part"


@pytest.mark.parametrize(
    ("link", "loose"),
    [
        ("10.0000/S0000-0000(20)30183-5", ".:/-()"),
        ("first.last@uni-example.edu", ".@-"),
        ("https://example.org/abcde/?view_only=0123", ".:/?_="),
    ],
)
def test_loosely_set_dois_urls_and_addresses_stay_whole(link, loose):
    """url.sty stretches the space around - ( ) @ _ ? = as well as . : /."""
    pdf_bytes = _pdf(_spaced_glyphs(link, 72, 700, loose=loose, gap=1.6))
    doc = pypdfium2.PdfDocument(pdf_bytes)
    try:
        pdfium_text = doc[0].get_textpage().get_text_range()
    finally:
        doc.close()

    assert get_native_text_in_bbox(pdf_bytes, 0, _WHOLE_PAGE) == pdfium_text
    assert pdfium_text.replace(" ", "") == link


def test_link_on_the_next_line_leaves_the_words_above_alone():
    """A link run ends at the line end, so the address below does not glue "increased risk"."""
    content = _glyph_line("children are at increased risk", 72, _line_y(0), gap=1.6)
    content += _glyph_line("first.last@example.edu", 72, _line_y(1), gap=1.6)

    text = get_native_text_in_bbox(_pdf(content), 0, _WHOLE_PAGE)

    assert "children are at increased risk" in text
    assert "first.last@example.edu" in text


def _repair_line(text: str) -> str:
    """Repair one 12 pt line of 6 pt wide glyphs; "|" is a generated line break at a 3 pt gap."""
    records, glyphs, breaks, x = [], [], set(), 72.0
    for ch in text:
        if ch == "|":
            breaks.update((len(records), len(records) + 1))
            records += [("\r", 0.0, 0.0, True), ("\n", 0.0, 0.0, True)]
            x += 3.0
            continue
        glyphs.append((len(records), ch, (x, 698.0, x + 6.0, 710.0)))
        records.append((ch, x + 3.0, 704.0, False))
        x += 6.0
    return "".join(record[0] for record in _repair_word_boundaries(records, glyphs, breaks))


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("see Fig.|3 for", "see Fig. 3 for"),
        ("6-2-|Funding", "6-2- Funding"),
        ("10.0000/S0000-|0000", "10.0000/S0000-0000"),
    ],
)
def test_generated_break_keeps_its_space_after_a_dot_or_hyphen_outside_links(line, expected):
    """A line break pdfium set at a word gap becomes a space, except inside a link."""
    assert _repair_line(line) == expected


def test_link_run_ends_at_a_bracket_after_punctuation():
    """The space in "[77] (http" is not part of the URL after it."""
    line = "analysis [77] (http://www.example.org/tool)"
    pdf_bytes = _pdf(_glyph_line(line, 72, 700, gap=1.6))

    assert get_native_text_in_bbox(pdf_bytes, 0, _WHOLE_PAGE) == line
