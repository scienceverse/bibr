"""Manuscript line numbers stay out of the native text layer.

A line-numbered manuscript prints a number in the margin beside every line.
A layout box that reaches over the margin used to take the numbers with the
text: a heading "668 References", reference titles with "675" inside them.
The fixtures copy the shape of the reported preprint's reference page (12 pt
numbers right-aligned at x = 53, text from x = 73, 25 pt line pitch), on two
pages as a manuscript numbers page after page, and of the numbered columns
that must survive: a bare-numbered reference list, labels that touch their
text, a table's restarting row numbers or prefixed ids, one numbered page.
"""

from __future__ import annotations

from bibr.ocr import native_text
from bibr.ocr.native_text import fill_native_text_and_fonts
from bibr.ocr.pdf_inspection import inspect_pdf
from tests.ocr.test_watermark_text import _bt, _pdf, _width

_PAGE_W, _PAGE_H = 612.0, 792.0
_NUMBER_RIGHT = 53.3
_TEXT_LEFT = 73.0


def _number(value: int, y: float, *, right: float = _NUMBER_RIGHT, size: float = 12.0) -> bytes:
    text = str(value)
    return _bt(text, right - _width(text, size), y, size=size)


def _region(top: float, bottom: float, label: str = "text", left: float = 30.0) -> dict:
    """A layout box in PDF points (y up), reaching over the left margin."""
    return {
        "label": label,
        "bbox_2d": [
            left / _PAGE_W * 1000.0,
            (_PAGE_H - top) / _PAGE_H * 1000.0,
            580.0 / _PAGE_W * 1000.0,
            (_PAGE_H - bottom) / _PAGE_H * 1000.0,
        ],
        "content": "",
    }


def _fill(pdf_bytes: bytes, regions: list[dict]) -> list[str]:
    fill_native_text_and_fonts(pdf_bytes, [regions], min_chars=3)
    return [region["content"] for region in regions]


_REFERENCE_LINES = [
    "Author A, Author B. A transcriptome atlas of healthy",
    "tissue across the lifespan. J Example 12, 1-9 (2020).",
    "Author C. Single-cell signatures of chronic disease.",
    "Example Reports 4, 10-19 (2021).",
    "Author D, Author E. Cohort profile: a population study.",
    "Int J Example 50, 20-29 (2022).",
    "Author F. Methods for spatial transcriptomics. Nat",
    "Example 3, 30-39 (2023).",
    "Author G, Author H. Ageing and the immune response.",
    "Example Medicine 8, 40-49 (2024).",
    "Author I. A reference genome for model organisms.",
]


def _reference_page(*, first: int = 668, pitch: float = 25.0) -> bytes:
    """The heading and reference entries, every printed line numbered in the margin."""
    content = b""
    top = 700.0
    lines = ["References", *_REFERENCE_LINES]
    for index, line in enumerate(lines):
        y = top - pitch * index
        content += _number(first + index, y) + _bt(line, _TEXT_LEFT, y)
    return _pdf(content, pages=2)


def test_heading_and_reference_lines_lose_the_margin_numbers():
    pdf_bytes = _reference_page()
    regions = [_region(712.0, 696.0, "paragraph_title"), _region(687.0, 440.0, "reference_content")]

    heading, references = _fill(pdf_bytes, regions)

    assert heading == "References"
    assert references.splitlines()[0] == _REFERENCE_LINES[0]
    assert not any(ch.isdigit() for ch in references.split("(2020)")[0].replace("12, 1-9", ""))
    for value in range(669, 680):
        assert str(value) not in references


def test_page_lines_for_the_reference_stream_lose_the_margin_numbers():
    pdf_bytes = _reference_page()
    inspection = inspect_pdf(
        pdf_bytes,
        [[_region(712.0, 696.0, "paragraph_title")]],
        fill_native_text=True,
        include_outline=False,
        include_ref_geometry=True,
        min_chars=3,
        min_printable_ratio=0.85,
    )

    texts = [line["text"] for line in inspection.page_lines]
    assert texts[0] == "References"
    assert texts[1:3] == _REFERENCE_LINES[:2]


def test_numbers_every_fifth_line_are_removed():
    content = b""
    for index in range(60):
        y = 760.0 - 12.0 * index
        if (index + 1) % 5 == 0:
            content += _number(index + 1, y, size=8.0)
        content += _bt(f"Body text line {index + 1} of the manuscript.", _TEXT_LEFT, y, size=10.0)

    (text,) = _fill(_pdf(content, pages=2), [_region(770.0, 30.0)])

    assert text.splitlines()[4] == "Body text line 5 of the manuscript."
    assert "55 Body" not in text


def test_right_hand_column_is_removed():
    content = b""
    for index in range(12):
        y = 700.0 - 25.0 * index
        content += _bt(f"Line {index} of a right-numbered page.", _TEXT_LEFT, y)
        content += _bt(str(index + 1), 572.0, y)

    (text,) = _fill(_pdf(content, pages=2), [_region(712.0, 420.0, left=60.0)])

    assert text.splitlines() == [f"Line {index} of a right-numbered page." for index in range(12)]


def test_numbers_drawn_inside_a_form_are_removed():
    form = b"".join(_number(668 + index, 700.0 - 25.0 * index) for index in range(12))
    body = b"".join(
        _bt(line, _TEXT_LEFT, 700.0 - 25.0 * index)
        for index, line in enumerate(["References", *_REFERENCE_LINES])
    )

    (heading,) = _fill(_pdf(body + b"/Fm1 Do\n", form=form, pages=2), [_region(712.0, 696.0)])

    assert heading == "References"


def test_bare_numbered_reference_list_keeps_its_numbers():
    """One number per two-line entry: the continuation line sits between the numbers."""
    content = b""
    for entry in range(10):
        y = 700.0 - 24.0 * entry
        content += _number(entry + 1, y, size=10.0)
        content += _bt(
            f"Author{entry} A. Title of work {entry} in a journal,", _TEXT_LEFT, y, size=10.0
        )
        content += _bt("J Example 1, 1-9 (2020).", _TEXT_LEFT, y - 12.0, size=10.0)

    (text,) = _fill(_pdf(content, pages=2), [_region(712.0, 450.0)])

    assert "3 Author2 A." in text
    assert "10 Author9 A." in text


def test_numbers_touching_their_text_are_kept():
    content = b""
    for index in range(12):
        y = 700.0 - 25.0 * index
        content += _number(index + 1, y) + _bt(f"Item {index} of a list.", _NUMBER_RIGHT + 2.0, y)

    (text,) = _fill(_pdf(content, pages=2), [_region(712.0, 420.0)])

    assert text.splitlines()[0].startswith("1")
    assert "12" in text


def test_short_column_is_kept():
    content = b""
    for index in range(7):
        y = 700.0 - 25.0 * index
        content += _number(index + 1, y) + _bt(f"Line {index} of a short page.", _TEXT_LEFT, y)

    (text,) = _fill(_pdf(content, pages=2), [_region(712.0, 530.0)])

    assert text.splitlines()[0].startswith("1")


def test_restarting_table_row_numbers_are_kept():
    """A table's first column repeats 2-6 for each block: not a line count."""
    content = b""
    values = [2, 3, 4, 5, 6] * 3
    for index, value in enumerate(values):
        y = 700.0 - 14.0 * index
        content += _number(value, y, size=10.0) + _bt("0.52  0.48  0.61", _TEXT_LEFT, y, size=10.0)

    (text,) = _fill(_pdf(content, pages=2), [_region(712.0, 480.0)])

    assert text.splitlines()[0].startswith("2")


def test_table_ids_with_a_prefix_further_out_are_kept():
    """Sample ids "ID-99", "ID-100", ... drawn as a prefix and a number: a table column."""
    content = b""
    for index in range(12):
        y = 700.0 - 19.0 * index
        content += _bt("ID-", 30.0, y, size=10.0) + _number(99 + index, y, size=10.0)
        content += _bt("3060.5  2446.5  6048.5", _TEXT_LEFT, y, size=10.0)

    (text,) = _fill(_pdf(content, pages=2), [_region(712.0, 480.0, left=20.0)])

    assert text.splitlines()[0].startswith("ID-99")
    assert "ID-110" in text


def test_numbered_column_on_a_single_page_is_kept():
    """A manuscript numbers page after page; one numbered page is a table or list."""
    content = b""
    for index in range(12):
        y = 700.0 - 25.0 * index
        content += _number(index + 1, y) + _bt(f"Row {index} of a numbered table.", _TEXT_LEFT, y)

    (text,) = _fill(_pdf(content), [_region(712.0, 420.0)])

    assert text.splitlines()[0].startswith("1")
    assert "12" in text


def test_numbered_table_on_two_pages_of_a_longer_paper_is_kept():
    """Items 1-12 on two pages of a six-page paper: a questionnaire, not line numbers."""
    content = b""
    for index in range(12):
        y = 700.0 - 25.0 * index
        content += _number(index + 1, y) + _bt(f"Item {index} of the questionnaire.", _TEXT_LEFT, y)
    prose = b"".join(
        _bt("Body text of an unnumbered page.", _TEXT_LEFT, 700.0 - 14.0 * i) for i in range(20)
    )

    (text,) = _fill(_pdf(content, pages=2, extra_pages=(prose,) * 4), [_region(712.0, 420.0)])

    assert text.splitlines()[0].startswith("1")
    assert "12" in text


def test_furniture_failure_falls_back_to_the_plain_text_layer(monkeypatch):
    def broken(page):
        raise RuntimeError("furniture pass failed")

    monkeypatch.setattr(native_text, "strip_furniture_objects", broken)

    (text,) = _fill(_reference_page(), [_region(687.0, 440.0, "reference_content")])

    assert "transcriptome atlas" in text


def test_document_scan_skips_pages_that_fail_to_load():
    class _BrokenDocument:
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int):
            raise RuntimeError("page failed to load")

    assert native_text._document_is_line_numbered(_BrokenDocument()) is False
