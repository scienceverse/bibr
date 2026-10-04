"""A table printed in several parts without its caption repeated is one table."""

from __future__ import annotations

from bibr.structure.pdf_parser import PDFParser


def _region(index, label, content="", bbox=None):
    return {"index": index, "label": label, "content": content, "bbox_2d": bbox}


_HEAD = "| Outcome | HR | 95% CI |\n|---|---|---|\n"
_BODY = "Stroke and related arterial events in the first weeks of follow-up after diagnosis."


def _two_page_table(continuation: str, between: list[dict] | None = None) -> list[list[dict]]:
    """A captioned table filling a page to its foot, continued at the top of
    the next page with no caption of its own (one report's Table 3, pp. 37-38)."""
    return [
        [
            _region(
                0, "figure_title", "Table 3. Hazard ratios by outcome", bbox=[80, 154, 899, 223]
            ),
            _region(1, "table", _HEAD + "| Stroke | 1.2 | 1.0-1.4 |", bbox=[78, 226, 922, 843]),
            *(between or []),
            _region(9, "footer", "Preprint", bbox=[21, 915, 66, 929]),
        ],
        [
            _region(0, "table", continuation, bbox=[78, 151, 922, 383]),
            _region(1, "footer", "Preprint", bbox=[21, 916, 66, 929]),
        ],
    ]


def test_uncaptioned_table_at_the_top_of_the_next_page_continues_the_table():
    contents = PDFParser(_two_page_table(_HEAD + "| Embolism | 2.1 | 1.7-2.6 |")).parse()

    (table,) = contents.tables
    assert table.caption == "Table 3. Hazard ratios by outcome"
    assert [part.page_number for part in table.parts] == [1, 2]
    assert table.contents == [
        ["Outcome", "HR", "95% CI"],
        ["Stroke", "1.2", "1.0-1.4"],
        ["Embolism", "2.1", "1.7-2.6"],
    ]


def test_continuation_without_a_repeated_header_keeps_its_first_row():
    continuation = "| Embolism | 2.1 | 1.7-2.6 |\n|---|---|---|\n| Bleeding | 0.9 | 0.7-1.1 |"

    (table,) = PDFParser(_two_page_table(continuation)).parse().tables

    assert len(table.parts) == 2
    assert table.contents == [
        ["Outcome", "HR", "95% CI"],
        ["Stroke", "1.2", "1.0-1.4"],
        ["Embolism", "2.1", "1.7-2.6"],
        ["Bleeding", "0.9", "0.7-1.1"],
    ]


def test_uncaptioned_table_with_another_column_count_stays_separate():
    contents = PDFParser(_two_page_table("| Site | n |\n|---|---|\n| North | 12 |")).parse()

    assert [len(table.parts) for table in contents.tables] == [1, 1]
    assert contents.tables[1].caption is None


def test_a_table_note_and_body_text_between_end_the_first_table():
    """A two-column page: the table's note and the next column's paragraphs
    are read between the two tables, so the second is a table of its own
    whose caption the layout model missed."""
    between = [
        _region(
            2, "vision_footnote", "HR, hazard ratio; CI, confidence interval.", [80, 846, 600, 860]
        ),
        _region(3, "text", _BODY, bbox=[80, 862, 900, 905]),
    ]
    pages = _two_page_table(_HEAD + "| Embolism | 2.1 | 1.7-2.6 |", between)

    assert [len(table.parts) for table in PDFParser(pages).parse().tables] == [1, 1]


def _stacked_blocks(gap_region: dict | None = None) -> list[list[dict]]:
    """One caption over a table printed as two stacked blocks (an appendix
    table split by a parameter range)."""
    head = "| alpha | Accuracy | Loss |\n|---|---|---|\n"
    return [
        [
            _region(0, "figure_title", "Table 15. Results by alpha", bbox=[150, 100, 850, 130]),
            _region(1, "table", head + "| 256 | 0.91 | 0.30 |", bbox=[150, 135, 850, 400]),
            *([gap_region] if gap_region else []),
            _region(3, "table", head + "| 2048 | 0.94 | 0.21 |", bbox=[150, 412, 850, 690]),
            _region(4, "text", _BODY, bbox=[100, 720, 900, 800]),
        ]
    ]


def test_stacked_uncaptioned_block_is_a_part_of_the_captioned_table_above():
    contents = PDFParser(_stacked_blocks()).parse()

    (table,) = contents.tables
    assert table.caption == "Table 15. Results by alpha"
    assert len(table.parts) == 2
    assert [row[0] for row in table.contents] == ["alpha", "256", "2048"]


def test_stacked_block_after_a_paragraph_stays_a_table_of_its_own():
    paragraph = _region(2, "text", _BODY, bbox=[150, 402, 850, 410])

    tables = PDFParser(_stacked_blocks(paragraph)).parse().tables

    assert [len(table.parts) for table in tables] == [1, 1]
