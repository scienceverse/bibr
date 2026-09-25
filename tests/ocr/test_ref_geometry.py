from pathlib import Path

import pytest

from bibr.ocr.ref_geometry import (
    LineRecord,
    group_chars_into_lines,
    record_to_dict,
    records_from_dicts,
    recover_reference_lines,
    reference_lines_from_pages,
)


def _chars(s, y, x0=72.0, dx=6.0):
    """Lay out string s as chars on one text-line at vertical position y."""
    out = []
    x = x0
    for ch in s:
        out.append((ch, (x, y, x + dx, y + 10.0)))
        x += dx
    return out


def test_group_chars_splits_on_newline_and_preserves_indent():
    chars = _chars("Aknin, L. (2013).", y=700, x0=72.0)
    chars.append(("\n", (0.0, 0.0, 0.0, 0.0)))
    chars += _chars("    continued line", y=688, x0=108.0)  # indented continuation
    lines = group_chars_into_lines(chars, page=5)
    assert [ln.text for ln in lines] == ["Aknin, L. (2013).", "continued line"]
    assert lines[0].x0 == pytest.approx(72.0)
    assert lines[1].x0 == pytest.approx(108.0)  # hanging indent preserved
    assert lines[0].page == 5
    assert lines[0].font_size == pytest.approx(10.0)


def test_group_chars_skips_blank_lines():
    chars = _chars("Real text", y=700)
    chars.append(("\n", (0, 0, 0, 0)))
    chars += [("   ", (0, 0, 0, 0))]  # whitespace-only -> dropped
    lines = group_chars_into_lines(chars, page=0)
    assert [ln.text for ln in lines] == ["Real text"]


def test_record_dict_round_trip():
    r = LineRecord("Aknin, L.", 5, 72.0, 710.0, 180.0, 700.0, 10.0)
    back = records_from_dicts([record_to_dict(r)])
    assert back == [r]


def _line(text: str, page: int, y_top: float) -> LineRecord:
    return LineRecord(text, page, 72.0, y_top, 400.0, y_top - 10.0, 10.0)


def test_reference_lines_remove_repeated_running_head_with_page_number():
    page_lines = {
        0: [
            _line("WILLINGNESS TO PRE-REGISTER 22", 0, 760.0),
            _line("References", 0, 700.0),
            _line("Smith, A. (2020). First.", 0, 660.0),
        ],
        1: [
            _line("WILLINGNESS TO PRE-REGISTER 23", 1, 760.0),
            _line("continues here.", 1, 700.0),
        ],
        2: [
            _line("WILLINGNESS TO PRE-REGISTER 24", 2, 760.0),
            _line("Jones, B. (2021). Second.", 2, 700.0),
        ],
    }

    lines = reference_lines_from_pages(page_lines, header_page=0)

    assert [line.text for line in lines] == [
        "Smith, A. (2020). First.",
        "continues here.",
        "Jones, B. (2021). Second.",
    ]


def test_reference_lines_keep_nonrepeated_top_page_continuation():
    page_lines = {
        0: [
            _line("References", 0, 700.0),
            _line("Smith, A. (2020). First.", 0, 660.0),
        ],
        1: [
            _line("continues here.", 1, 760.0),
            _line("Jones, B. (2021). Second.", 1, 700.0),
        ],
    }

    lines = reference_lines_from_pages(page_lines, header_page=0)

    assert "continues here." in [line.text for line in lines]


def test_recover_reference_lines_from_gold_pdf():
    pdf = Path(__file__).parent.parent / "fixtures" / "ref_geometry_hanging_indent_sample.pdf"
    lines = recover_reference_lines(pdf.read_bytes())
    assert len(lines) > 20
    assert len({round(ln.x0) for ln in lines}) >= 2  # hanging-indent x0 levels


def test_recover_reference_lines_no_text_layer_returns_empty():
    pdf = Path(__file__).parent.parent / "fixtures" / "scanned_sample.pdf"
    assert recover_reference_lines(pdf.read_bytes()) == []
