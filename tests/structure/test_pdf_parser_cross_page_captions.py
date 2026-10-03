"""A caption whose cross-page edge would be vetoed still owns its same-page figure (#133)."""

from __future__ import annotations

from bibr.structure.pdf_parser import PDFParser


def _region(index, label, content="", bbox=None, image_b64=None):
    value = {"index": index, "label": label, "content": content, "bbox_2d": bbox}
    if image_b64 is not None:
        value["image_b64"] = image_b64
    return value


def test_table_caption_is_not_lost_to_a_table_on_the_previous_page():
    """An uncaptioned cover-sheet table shifts the provisional table ids by
    one, so "Table 2" matched the number of the previous page's table, which
    ends at the page foot. That cross-page edge won the solve and was then
    vetoed, leaving Table 2 uncaptioned (a 54-page preprint; issue #133)."""
    table = "| Test | Result |\n|---|---|\n| Bartlett | p < .001 |"
    note = "*p<0.05, **p<0.01, ***p<0.001"
    contents = PDFParser(
        [
            [_region(0, "table", table, [109, 315, 891, 765])],
            [
                _region(0, "figure_title", "Table 1: Descriptive statistics", [82, 119, 248, 140]),
                _region(1, "table", table, [61, 141, 874, 830]),
                _region(2, "vision_footnote", note, [83, 831, 324, 848]),
            ],
            [
                _region(
                    0, "figure_title", "Table 2: Validation tests and results", [82, 121, 293, 139]
                ),
                _region(1, "table", table, [77, 137, 922, 643]),
            ],
        ]
    ).parse()

    assert [(item.label, item.caption) for item in contents.tables] == [
        (None, None),
        ("1", "Table 1: Descriptive statistics"),
        ("2", "Table 2: Validation tests and results"),
    ]
