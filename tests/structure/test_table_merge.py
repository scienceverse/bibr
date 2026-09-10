from copy import deepcopy

import pytest

from bibr.structure.table_merge import merge_table_contents
from tests.structure.test_pdf_parser_table_html import _HTML_TABLE, _parse, _region


def test_continuation_preserves_header_topology_and_source_parts():
    second = _HTML_TABLE.replace("Age", "Weight")
    pages = [
        [_region(0, "figure_title", "Table 1. Values"), _region(1, "table", _HTML_TABLE)],
        [_region(0, "figure_title", "Table 1. Values (Continued)"), _region(1, "table", second)],
    ]
    (table,) = _parse(pages).tables
    assert 'rowspan="2"' in table.tbl_html
    assert 'colspan="2"' in table.tbl_html
    assert table.tbl_html.count(">Var<") == 1
    assert [p.tbl_html for p in table.parts] == [_HTML_TABLE, second]
    assert [p.page_number for p in table.parts] == [1, 2]
    assert table.df.iloc[:, 0].tolist() == ["Age", "Weight"]


def test_three_page_table_keeps_headerless_first_data_row_and_inline_markup():
    middle = "<table><tr><td><em>Weight</em></td><td>007</td><td>2</td></tr></table>"
    final = _HTML_TABLE.replace("Age", "Height")
    pages = [
        [_region(0, "figure_title", "Table 1. Values"), _region(1, "table", _HTML_TABLE)],
        [_region(0, "figure_title", "Table 1. Values (Continued)"), _region(1, "table", middle)],
        [_region(0, "figure_title", "Table 1. Values (Continued)"), _region(1, "table", final)],
    ]
    (table,) = _parse(pages).tables
    assert [p.page_number for p in table.parts] == [1, 2, 3]
    assert [p.tbl_html for p in table.parts] == [_HTML_TABLE, middle, final]
    assert table.df.iloc[:, 0].tolist() == ["Age", "Weight", "Height"]
    assert "<em>Weight</em>" in table.tbl_html
    assert "007" in table.df.iloc[1].tolist()


@pytest.mark.parametrize("change", ["header", "topology", "overflow", "nested"])
def test_ambiguous_cell_structure_is_not_mutated(change):
    (left,) = _parse([[_region(0, "table", _HTML_TABLE)]]).tables
    html = {
        "header": _HTML_TABLE.replace("Group A", "Group B"),
        "topology": _HTML_TABLE.replace('rowspan="2"', 'colspan="2"'),
        "overflow": _HTML_TABLE.replace('rowspan="2"', 'rowspan="99"'),
        "nested": _HTML_TABLE.replace("Age", "<table><tr><td>nested</td></tr></table>"),
    }[change]
    right = deepcopy(left)
    right.tbl_html = html
    before = (left.tbl_html, len(left.parts), left.df.copy())
    assert not merge_table_contents(left, right)
    assert (left.tbl_html, len(left.parts)) == before[:2]
    assert left.df.equals(before[2])


def test_changed_explicit_headers_stay_separate_in_parser():
    second = _HTML_TABLE.replace("Group A", "Group B")
    result = _parse(
        [
            [_region(0, "figure_title", "Table 1. Values"), _region(1, "table", _HTML_TABLE)],
            [
                _region(0, "figure_title", "Table 1. Values (Continued)"),
                _region(1, "table", second),
            ],
        ]
    )
    assert len(result.tables) == 2
    assert sum(len(t.parts) for t in result.tables) == 2


def test_trailing_continued_caption_cannot_override_heading_barrier():
    result = _parse(
        [
            [_region(0, "figure_title", "Table 1. Values"), _region(1, "table", _HTML_TABLE)],
            [
                _region(0, "paragraph_title", "Results"),
                _region(1, "figure_title", "Table 1. Values (Continued)"),
                _region(2, "table", _HTML_TABLE),
            ],
        ]
    )
    assert len(result.tables) == 2
