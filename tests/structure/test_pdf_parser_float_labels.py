"""PDF floats carry the label their caption prints, and mentions find them by it."""

from __future__ import annotations

from bibr.paper_contents import PaperSentence
from bibr.structure.pdf_parser import PDFParser
from bibr.structure.xref_utils import detect_xrefs

_TABLE = "| A | B |\n|---|---|\n| 1 | 2 |"


def _region(index, label, content="", bbox=None, image_b64=None):
    value = {"index": index, "label": label, "content": content, "bbox_2d": bbox}
    if image_b64 is not None:
        value["image_b64"] = image_b64
    return value


def _captioned_table(caption: str) -> list[dict]:
    return [
        _region(0, "figure_title", caption, bbox=[0, 100, 500, 130]),
        _region(1, "table", _TABLE, bbox=[0, 140, 500, 400]),
    ]


def test_captions_give_their_printed_labels():
    contents = PDFParser(
        [
            _captioned_table("Table 3.1. Participants"),
            _captioned_table("Table S2: Robustness checks"),
            _captioned_table("TABLE IV. Items"),
            _captioned_table("Supplementary Table 4. Extra items"),
            [
                _region(0, "chart", bbox=[0, 100, 500, 300], image_b64="plot"),
                _region(1, "figure_title", "Figure A1. Sensitivity", bbox=[0, 305, 500, 345]),
            ],
        ]
    ).parse()

    by_page = {table.parts[0].page_number: table for table in contents.tables}
    assert {page: table.label for page, table in by_page.items()} == {
        1: "3.1",
        2: "S2",
        3: "IV",
        4: "S4",
    }
    # A "Table S2" caption the layout model calls a figure title is a table's.
    assert by_page[2].caption == "Table S2: Robustness checks"
    assert [(figure.label, figure.caption) for figure in contents.figures] == [
        ("A1", "Figure A1. Sensitivity")
    ]

    sentence = PaperSentence(
        text_id=1,
        text="Table S2, Supplementary Table 4, Table IV and Table 3.1 agree with Figure A1.",
        section_id=0,
        paragraph_id=1,
    )
    xrefs = detect_xrefs([sentence], contents.tables, contents.figures)
    assert [(xref.xref_type, xref.xref_id) for xref in xrefs] == [
        ("table", by_page[2].table_id),
        ("table", by_page[4].table_id),
        ("table", by_page[3].table_id),
        ("table", by_page[1].table_id),
        ("figure", contents.figures[0].figure_id),
    ]


def test_differently_labelled_tables_on_adjacent_pages_stay_apart():
    """Tables 3.1 and 3.2 share the number 3, not the label: merging them as
    one table continued across pages lost the second."""
    contents = PDFParser(
        [
            [
                _region(0, "figure_title", "Table 3.1 Participants", bbox=[0, 10, 500, 40]),
                _region(1, "table", _TABLE, bbox=[0, 50, 500, 900]),
            ],
            [
                _region(0, "figure_title", "Table 3.2 Participants", bbox=[0, 10, 500, 40]),
                _region(1, "table", _TABLE, bbox=[0, 50, 500, 900]),
            ],
        ]
    ).parse()

    assert [(table.label, len(table.parts)) for table in contents.tables] == [
        ("3.1", 1),
        ("3.2", 1),
    ]


def test_the_same_label_still_continues_across_pages():
    contents = PDFParser(
        [
            [
                _region(0, "figure_title", "Table 3.1 Participants", bbox=[0, 10, 500, 40]),
                _region(1, "table", _TABLE, bbox=[0, 50, 500, 900]),
            ],
            [
                _region(0, "figure_title", "Table 3.1 (continued)", bbox=[0, 10, 500, 40]),
                _region(1, "table", _TABLE, bbox=[0, 50, 500, 900]),
            ],
        ]
    ).parse()

    (table,) = contents.tables
    assert (table.label, len(table.parts)) == ("3.1", 2)
