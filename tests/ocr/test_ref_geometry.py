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


def test_reference_lines_keep_mid_page_lines_that_match_edge_furniture():
    # Gutter-aligned labels emitted as their own lines: "1." and "4." sit at a
    # page edge on two pages, so "#." is furniture there, but the labels
    # inside the pages are bibliography lines. So is a year-only
    # continuation that matches the page-number key.
    page_texts = {
        5: [
            "References",
            "1.",
            "Smith, J. A study. J 1, 1",
            "2015",
            "2.",
            "Roe, R. Another. J 2, 2 (2002).",
            "3.",
            "Poe, P. Third. J 3, 3 (2003).",
            "12",
        ],
        6: ["4.", "Lee, L. Fourth. J 4, 4 (2004).", "5.", "Kim, K. Fifth. J 5, 5 (2005).", "13"],
    }
    page_lines = {
        page: [_line(text, page, 760.0 - 20.0 * i) for i, text in enumerate(texts)]
        for page, texts in page_texts.items()
    }

    lines = reference_lines_from_pages(page_lines, header_page=5)

    assert [line.text for line in lines] == [
        "Smith, J. A study. J 1, 1",
        "2015",
        "2.",
        "Roe, R. Another. J 2, 2 (2002).",
        "3.",
        "Poe, P. Third. J 3, 3 (2003).",
        "Lee, L. Fourth. J 4, 4 (2004).",
        "5.",
        "Kim, K. Fifth. J 5, 5 (2005).",
    ]


@pytest.mark.slow
def test_recover_reference_lines_from_gold_pdf():
    pdf = Path("data/psych_science_pdf_oa/09567976211052476.pdf")
    if not pdf.exists():
        pytest.skip("gold PDF not present")
    lines = recover_reference_lines(pdf.read_bytes())
    assert len(lines) > 20
    assert len({round(ln.x0) for ln in lines}) >= 2  # hanging-indent x0 levels


@pytest.mark.slow
def test_recover_reference_lines_no_text_layer_returns_empty():
    pdf = Path("tests/fixtures/scanned_sample.pdf")
    if not pdf.exists():
        pytest.skip("fixture absent")
    assert recover_reference_lines(pdf.read_bytes()) == []
