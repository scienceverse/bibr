"""HTML table markup -> DataFrame of the printed cell text.

``pandas.read_html`` inferred column types, so ``table[].contents`` carried
numbers the paper never printed ("2.50" -> "2.5", "1,5" -> "15", an empty
cell -> "nan"). :func:`html_table_frame` keeps every cell as its text while
keeping ``read_html``'s frame shape and column labels.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from bibr.structure.html_table import html_table_frame


def _grid(df: pd.DataFrame) -> list[list[str]]:
    return [[str(c) for c in df.columns], *df.values.tolist()]


class TestCellsStayAsPrinted:
    def test_numbers_keep_their_printed_form(self):
        df = html_table_frame(
            "<table><tr><th>N</th><th>Code</th><th>Count</th><th>M</th></tr>"
            "<tr><td>12</td><td>007</td><td>1,234</td><td>2.50</td></tr>"
            "<tr><td></td><td>010</td><td>56</td><td>0.10</td></tr></table>"
        )

        assert df is not None
        assert _grid(df) == [
            ["N", "Code", "Count", "M"],
            # read_html: "12.0", "7.0", "1234.0", "2.5", with "nan" for the
            # empty cell below — every column became numbers, and the int
            # column with an empty cell floats.
            ["12", "007", "1,234", "2.50"],
            ["", "010", "56", "0.10"],
        ]

    def test_a_decimal_comma_is_not_a_thousands_separator(self):
        """read_html's default ``thousands=","`` read "1,5" as 15."""
        df = html_table_frame(
            "<table><tr><th>Gruppe</th><th>M</th><th>SD</th></tr>"
            "<tr><td>A</td><td>1,5</td><td>0,25</td></tr>"
            "<tr><td>B</td><td>2,75</td><td>1,0</td></tr></table>"
        )

        assert df is not None
        assert df.values.tolist() == [["A", "1,5", "0,25"], ["B", "2,75", "1,0"]]

    def test_na_and_boolean_words_are_text(self):
        df = html_table_frame(
            "<table><tr><th>x</th><th>y</th><th>z</th></tr>"
            "<tr><td>NA</td><td>TRUE</td><td>n/a</td></tr>"
            "<tr><td>1</td><td>False</td><td>None</td></tr></table>"
        )

        assert df is not None
        assert df.values.tolist() == [["NA", "TRUE", "n/a"], ["1", "False", "None"]]

    def test_every_cell_is_a_string(self):
        df = html_table_frame("<table><tr><td>1</td><td></td></tr><tr><td>2</td><td>3</td></tr>")

        assert df is not None
        assert all(isinstance(v, str) for row in df.values.tolist() for v in row)


class TestReadHtmlLayoutIsKept:
    """Header, span and whitespace handling are ``read_html``'s."""

    @pytest.mark.parametrize(
        "html",
        [
            # Multi-level header with a stub rowspan and a group colspan.
            '<table><tr><th rowspan="2">Var</th><th colspan="2">Group A</th></tr>'
            "<tr><th>M</th><th>SD</th></tr><tr><td>Age</td><td>x</td><td>y</td></tr></table>",
            # Explicit sections, footer and a rowspan running past the body.
            "<table><thead><tr><th>A</th><th></th></tr></thead>"
            '<tbody><tr><td rowspan="3">a</td><td>b</td></tr></tbody>'
            "<tfoot><tr><td>f</td></tr></tfoot></table>",
            # No header row; ragged rows are padded.
            "<table><tr><td>a</td></tr><tr><td>b</td><td>c</td></tr></table>",
            # Duplicate and empty header names.
            "<table><tr><th>A</th><th>A</th><th></th></tr><tr><td>x</td><td>y</td><td>z</td></tr>"
            "</table>",
            # <br>, runs of whitespace, hidden cells and <style> inside cells.
            "<table><tr><th>a<br>b</th><th>c  d\ne</th><th style='display: none'>h</th></tr>"
            "<tr><td>x<style>.s{}</style></td><td> y </td></tr></table>",
            # The first table without text is skipped.
            "<table><tr><td></td></tr></table><table><tr><th>T</th></tr><tr><td>v</td></tr></table>",
            # A rowspan in the last column carries into the rows below.
            "<table><tr><th>G</th><th>p</th></tr>"
            '<tr><td>a</td><td rowspan="2">x</td></tr><tr><td>b</td></tr></table>',
            # A header row with no text adds no level to the MultiIndex.
            "<table><thead><tr><th>A</th><th>B</th></tr><tr><th></th><th></th></tr>"
            "<tr><th>x</th><th>y</th></tr></thead><tbody><tr><td>p</td><td>q</td></tr></tbody>"
            "</table>",
            # A hidden first table is skipped.
            "<table style='display:none'><tr><td>h</td></tr></table>"
            "<table><tr><th>T</th></tr><tr><td>v</td></tr></table>",
            # A first table with only a caption has no rows, so the next is read.
            "<table><caption>c</caption></table>"
            "<table><tr><th>T</th></tr><tr><td>v</td></tr></table>",
        ],
    )
    def test_shape_labels_and_text_match_read_html(self, html):
        """With no cell pandas would convert, the result is read_html's."""
        expected = pd.read_html(StringIO(html), flavor="html5lib")[0].fillna("")

        df = html_table_frame(html)

        assert df is not None
        assert list(df.columns) == list(expected.columns)
        assert df.values.tolist() == expected.values.tolist()

    def test_spans_copy_the_text_into_every_slot(self):
        df = html_table_frame(
            '<table><tr><th rowspan="2">Var</th><th colspan="2">Group A</th></tr>'
            "<tr><th>M</th><th>SD</th></tr>"
            '<tr><td rowspan="2">Age</td><td>30</td><td>5.0</td></tr>'
            "<tr><td>31</td><td>4.0</td></tr></table>"
        )

        assert df is not None
        assert list(df.columns) == [
            ("Var", "Var"),
            ("Group A", "M"),
            ("Group A", "SD"),
        ]
        assert df.values.tolist() == [["Age", "30", "5.0"], ["Age", "31", "4.0"]]

    def test_no_table_with_text_is_none(self):
        assert html_table_frame("<table><tr><td></td></tr></table>") is None
        assert html_table_frame("<table><tr><td></td><td></td></tr></table>") is None
        assert html_table_frame("<p>no table</p>") is None
        assert html_table_frame("<table><caption>Only a caption</caption></table>") is None


class TestParsedElements:
    def test_a_parsed_table_is_read_without_being_modified(self):
        soup = BeautifulSoup(
            "<table><tr><th>A<br>B</th><th style='display:none'>h</th></tr>"
            "<tr><td>0.50</td><td>x</td></tr></table>",
            "html5lib",
        )
        before = str(soup)

        df = html_table_frame(soup.table)

        assert df is not None
        assert _grid(df) == [["A B", "Unnamed: 1"], ["0.50", "x"]]
        assert str(soup) == before


class TestMarkupReadHtmlCouldNotRead:
    def test_colspan_zero_keeps_the_cell(self):
        """read_html repeated the cell zero times, so its text was lost."""
        df = html_table_frame(
            '<table><tr><td colspan="0">a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>'
        )

        assert df is not None
        assert df.values.tolist() == [["a", "b"], ["c", "d"]]

    def test_a_span_with_a_unit_counts_its_digits(self):
        """``int("2px")`` made read_html fail the whole table."""
        df = html_table_frame(
            '<table><tr><th colspan="2px">G</th></tr><tr><td>1</td><td>2</td></tr></table>'
        )

        assert df is not None
        assert list(df.columns) == ["G", "G.1"]
        assert df.values.tolist() == [["1", "2"]]

    def test_header_rows_without_text_read_as_no_header(self):
        """read_html raised IndexError on several all-empty header rows."""
        df = html_table_frame(
            "<table><thead><tr><th></th><th></th></tr><tr><th></th><th></th></tr></thead>"
            "<tbody><tr><td>1</td><td>x</td></tr></tbody></table>"
        )

        assert df is not None
        assert list(df.columns) == [0, 1]
        assert df.values.tolist() == [["1", "x"]]

    def test_spans_are_capped_at_the_html_limits(self):
        df = html_table_frame(
            '<table><tr><td colspan="99999999">a</td></tr><tr><td>b</td></tr></table>'
        )

        assert df is not None
        assert df.shape == (2, 1000)
