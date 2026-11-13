"""HTML table edge cases."""

from bibr.structure.html_table import html_table_frame


def test_a_table_nested_in_a_cell_adds_no_rows_to_the_outer_table():
    frame = html_table_frame(
        "<table><tr><th>A</th><th>B</th></tr>"
        "<tr><td>1</td><td><table><tr><td>x</td></tr><tr><td>y</td></tr></table></td></tr>"
        "<tr><td>2</td><td>3</td></tr></table>"
    )
    assert frame is not None
    assert frame.values.tolist() == [["1", "xy"], ["2", "3"]]


def test_a_nested_table_in_a_header_cell_adds_no_header_rows():
    frame = html_table_frame(
        "<table><thead><tr><th>A</th><th><table><tr><td>x</td></tr></table></th></tr></thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
    )
    assert frame is not None
    assert list(frame.columns) == ["A", "x"]
    assert frame.values.tolist() == [["1", "2"]]
