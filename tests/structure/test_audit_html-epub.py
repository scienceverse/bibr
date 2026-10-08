"""HTML table span expansion is bounded (audit 2026-10-06, §1.4).

``colspan`` and ``rowspan`` were capped per cell only, and a rowspan past the
last row added a row of copies for every row it reached, so the 64-byte
``<td colspan="1000" rowspan="4000">`` built a 4,000 x 1,000 frame and a
rowspan of 65534 ran out of memory. A colspan now ends at the width the
table's own markup allows, and a table out of proportion all the same is not
read.
"""

from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import pytest

from bibr.input.html_native import HtmlParser
from bibr.structure import html_table
from bibr.structure.html_table import html_table_frame


class TestRowspansEndAtTheLastRow:
    def test_a_rowspan_past_the_last_row_adds_no_rows(self):
        """read_html added an ("a", "") row for each row the span reached past
        the table; a browser draws the cell down to the last row only."""
        df = html_table_frame(
            '<table><tr><th>A</th><th>B</th></tr><tr><td rowspan="3">a</td><td>b</td></tr></table>'
        )

        assert df is not None
        assert df.values.tolist() == [["a", "b"]]

    def test_a_rowspan_past_the_footer_stops_after_it(self):
        df = html_table_frame(
            "<table><thead><tr><th>A</th><th>B</th></tr></thead>"
            '<tbody><tr><td rowspan="5">a</td><td>b</td></tr></tbody>'
            "<tfoot><tr><td>f</td></tr></tfoot></table>"
        )

        assert df is not None
        assert df.values.tolist() == [["a", "b"], ["a", "f"]]

    def test_a_huge_rowspan_over_a_narrow_table_reads_its_rows(self):
        df = html_table_frame(
            '<table><tr><td rowspan="65534">a</td><td>b</td></tr><tr><td>c</td></tr></table>'
        )

        assert df is not None
        assert df.values.tolist() == [["a", "b"], ["a", "c"]]

    @pytest.mark.parametrize(
        "html",
        [
            # From the header rows of a table without <thead> into the body.
            '<table><tr><th rowspan="2">Var</th><th>A</th></tr><tr><td>x</td></tr>'
            "<tr><td>y</td><td>z</td></tr></table>",
            # From the body into the footer.
            "<table><thead><tr><th>A</th><th>B</th></tr></thead>"
            '<tbody><tr><td rowspan="2">a</td><td>b</td></tr></tbody>'
            "<tfoot><tr><td>f</td></tr></tfoot></table>",
            # From <thead> into the body.
            '<table><thead><tr><th rowspan="2">H</th><th>I</th></tr></thead>'
            "<tbody><tr><td>p</td></tr><tr><td>q</td><td>r</td></tr></tbody></table>",
        ],
    )
    def test_spans_within_the_table_still_match_read_html(self, html):
        expected = pd.read_html(StringIO(html), flavor="html5lib")[0].fillna("")

        df = html_table_frame(html)

        assert df is not None
        assert list(df.columns) == list(expected.columns)
        assert df.values.tolist() == expected.values.tolist()


# One wide row padding many one-cell rows: 300 x 300 slots from 600 cells.
_PADDING = "<table><tr>" + "<td>w</td>" * 300 + "</tr>" + "<tr><td>n</td></tr>" * 300 + "</table>"


class TestExpansionStaysInProportion:
    @pytest.mark.parametrize(
        ("html", "shape"),
        [
            # The report's 64 bytes: 4,000 x 1,000 slots.
            ('<table><tr><td colspan="1000" rowspan="4000">x</td></tr></table>', (1, 22)),
            # Ten such cells: 2,000 x 10,000 slots from 424 bytes.
            (
                "<table><tr>" + '<td colspan="1000" rowspan="2000">x</td>' * 10 + "</tr></table>",
                (1, 40),
            ),
            # A span down rows that are present: 50 x 100 slots from one cell.
            (
                '<table><tr><td colspan="100" rowspan="300">x</td></tr>'
                + "<tr></tr>" * 49
                + "</table>",
                (50, 22),
            ),
            # 1,000 cells of colspan 1000 (25 KB): a 1 x 1,000,000 frame took 66 s.
            ("<table><tr>" + '<td colspan="1000">x</td>' * 1000 + "</tr></table>", (1, 2020)),
        ],
        ids=["report-bomb", "ten-cells", "rows-present", "wide-cells"],
    )
    def test_a_colspan_past_the_table_width_ends_there(self, html, shape):
        df = html_table_frame(html)

        assert df is not None
        assert df.shape == shape
        assert set(df.values.ravel()) == {"x"}

    @pytest.mark.parametrize(
        "html",
        [
            # Every row opens another wide span over the rows below.
            "<table>" + '<tr><td colspan="5" rowspan="9">x</td></tr>' * 8 + "</table>",
            _PADDING,
        ],
        ids=["growing-spans", "padding"],
    )
    def test_a_table_far_larger_than_its_markup_is_not_read(self, html, caplog):
        with caplog.at_level(logging.WARNING, logger="bibr.structure.html_table"):
            assert html_table_frame(html) is None

        assert "expand far past its markup" in caplog.text

    def test_spans_may_widen_a_table_to_twice_its_widest_row_plus_some(self):
        widest = "<tr><td>a</td><td>b</td><td>c</td></tr>"
        limit = 2 * 3 + html_table._FREE_COLUMNS

        df = html_table_frame(f'<table><tr><td colspan="{limit}">T</td></tr>{widest}</table>')
        wider = html_table_frame(
            f'<table><tr><td colspan="{limit + 1}">T</td></tr>{widest}</table>'
        )

        assert df is not None
        assert df.shape == (2, limit)
        assert wider is not None
        assert wider.values.tolist() == df.values.tolist()

    def test_a_colspan_ends_with_room_for_the_rest_of_its_row(self):
        df = html_table_frame(
            "<table><tr><td>a</td><td>b</td><td>c</td></tr>"
            '<tr><td rowspan="2">r</td><td>s</td><td colspan="999">t</td><td>u</td></tr>'
            "<tr><td>v</td></tr></table>"
        )

        assert df is not None
        # Widest row: 4 cells, so 2 x 4 + 20 columns; "t" leaves one for "u".
        assert df.shape == (3, 28)
        assert df.values.tolist()[1] == ["r", "s"] + ["t"] * 25 + ["u"]
        assert df.values.tolist()[2] == ["r", "v"] + [""] * 26

    def test_a_colspan_within_proportion_is_still_capped_at_1000(self):
        df = html_table_frame(
            '<table><tr><td colspan="99999999">a</td></tr><tr>'
            + "<td>b</td>" * 500
            + "</tr></table>"
        )

        assert df is not None
        assert df.shape == (2, 1000)

    @pytest.mark.parametrize("colspan", ["100", "100%", "31"])
    def test_a_footnote_row_spanning_every_column_keeps_the_table(self, colspan):
        """The "span every column" idiom over five columns: the footnote's
        colspan ends at 30 columns and every data cell is read, where a span
        past that width dropped the whole table."""
        data = [["A", "10", "1.2", "0.3", "0.01"], ["B", "12", "1.4", "0.2", "0.04"]] * 5
        html = (
            "<table><thead><tr><th>Group</th><th>N</th><th>Mean</th><th>SD</th><th>p</th></tr>"
            "</thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in data)
            + f'</tbody><tfoot><tr><td colspan="{colspan}">Note. Means.</td></tr></tfoot></table>'
        )

        df = html_table_frame(html)

        assert df is not None
        width = min(int(colspan.rstrip("%")), 2 * 5 + html_table._FREE_COLUMNS)
        assert df.shape == (11, width)
        assert list(df.columns[:5]) == ["Group", "N", "Mean", "SD", "p"]
        assert df.iloc[:10, :5].values.tolist() == data
        assert df.iloc[10].tolist() == ["Note. Means."] * width

    def test_an_uncaptioned_html_table_with_a_spanning_footnote_is_kept(self):
        html = (
            b"<html><body><article><h1>T</h1><p>Body.</p>"
            b"<table><tr><th>Group</th><th>N</th><th>Mean</th></tr>"
            b"<tr><td>A</td><td>10</td><td>1.2</td></tr>"
            b'<tr><td colspan="100">Source: survey 2024</td></tr></table>'
            b"</article></body></html>"
        )

        contents = HtmlParser(html).parse()

        assert len(contents.tables) == 1
        assert contents.tables[0].df.iloc[0, :3].tolist() == ["A", "10", "1.2"]

    @pytest.mark.parametrize(
        ("attrs", "shape"),
        [
            # Capped at 1000, then at the table's width of 2 x 2 + 20.
            ('colspan="' + "9" * 5000 + '"', (2, 24)),
            # Leading zeros do not count towards the digits.
            ('rowspan="' + "0" * 5000 + '2"', (2, 2)),
        ],
        ids=["many-digits", "leading-zeros"],
    )
    def test_a_span_of_thousands_of_digits_is_read(self, attrs, shape):
        """int() raises on more than 4,300 digits, which dropped the table."""
        df = html_table_frame(
            f"<table><tr><td {attrs}>a</td><td>b</td></tr><tr><td>c</td></tr></table>"
        )

        assert df is not None
        assert df.shape == shape

    def test_slots_are_bounded_by_the_cells_and_rows_in_the_markup(self):
        # A row of 100 cells pads 48 one-cell rows to 49 x 100 = 4,900 slots,
        # within 20 for each of the 197 cells and rows plus 1,000; one more
        # one-cell row is not.
        wide = "<tr>" + "<td>w</td>" * 100 + "</tr>"

        df = html_table_frame("<table>" + wide + "<tr><td>n</td></tr>" * 48 + "</table>")

        assert df is not None
        assert df.shape == (49, 100)
        assert html_table_frame("<table>" + wide + "<tr><td>n</td></tr>" * 49 + "</table>") is None

    def test_an_ordinary_table_with_merged_cells_still_matches_read_html(self):
        html = (
            '<table><tr><th rowspan="2">Group</th><th colspan="39">Measures</th></tr><tr>'
            + "".join(f"<th>m{i}</th>" for i in range(39))
            + "</tr>"
            + ("<tr>" + '<td colspan="2">g</td>' + "<td>v</td>" * 38 + "</tr>") * 60
            + "</table>"
        )
        expected = pd.read_html(StringIO(html), flavor="html5lib")[0].fillna("")

        df = html_table_frame(html)

        assert df is not None
        assert df.shape == (60, 40)
        assert list(df.columns) == list(expected.columns)
        assert df.values.tolist() == expected.values.tolist()

    def test_the_first_table_out_of_proportion_is_not_replaced_by_the_next(self):
        html = _PADDING + "<table><tr><th>T</th></tr><tr><td>v</td></tr></table>"

        assert html_table_frame(html) is None

    def test_an_html_table_out_of_proportion_keeps_its_caption_without_contents(self):
        html = (
            b"<html><body><article><h1>T</h1><p>Body.</p>"
            + _PADDING.replace("<table>", "<table><caption>Table 1. Padded</caption>").encode()
            + b"</article></body></html>"
        )

        contents = HtmlParser(html).parse()

        assert len(contents.tables) == 1
        assert contents.tables[0].label is not None
        assert contents.tables[0].df.empty


class TestHeaderRowCap:
    def test_more_header_rows_than_the_cap_are_read_as_data(self):
        """Every <th> row was a MultiIndex level, at about a millisecond each:
        8,000 one-cell rows (150 KB) took 9 s."""
        rows = html_table._MAX_HEADER_ROWS + 20
        html = "<table>" + "<tr><th>h</th></tr>" * rows + "<tr><td>x</td></tr></table>"

        df = html_table_frame(html)

        assert df is not None
        assert list(df.columns) == [0]
        assert df[0].tolist() == ["h"] * rows + ["x"]

    def test_a_header_at_the_cap_is_still_a_multiindex(self):
        rows = html_table._MAX_HEADER_ROWS
        html = "<table>" + "<tr><th>h</th></tr>" * rows + "<tr><td>x</td></tr></table>"

        df = html_table_frame(html)

        assert df is not None
        assert df.columns.nlevels == rows
        assert df.values.tolist() == [["x"]]
