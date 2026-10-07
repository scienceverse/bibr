"""HTML table text is bounded, not only its slots (final review, 2026-10-07).

The span limits bounded how many slots a table expands to, but every slot a
cell spans holds its text again: one long cell spanning 20 columns down 5,000
bare ``<tr>`` made 40 KB of HTML into 2 GB of ``contents``. A table is now
limited by its size as rendered (escaped text and cell markup in every slot,
at the bytes per character its widest character needs), and so are the size
and slots of a document's tables. pandas' naming of a one-row header, which
grew with the square of its width, is done here in linear time.
"""

from __future__ import annotations

import logging
import random

import pytest
from pandas.io.parsers import TextParser  # type: ignore[attr-defined]  # no stub

from bibr.processing_warnings import WarningCode
from bibr.structure import html_table
from bibr.structure.html_table import TableBudget, TableSize, html_table_frame

M = html_table._CELL_MARKUP_CHARS


def _spanned(text: str, colspan: int, rowspan: int) -> str:
    """One cell spanning *colspan* columns down *rowspan* rows, all but the
    first of them bare ``<tr>``."""
    return (
        f'<table><tr><td colspan="{colspan}" rowspan="{rowspan}">{text}</td></tr>'
        + "<tr>" * (rowspan - 1)
        + "</table>"
    )


class TestTableText:
    @pytest.mark.parametrize(
        ("colspan", "rowspan", "chars"),
        # 20 x 50 slots of 20,000 characters, and 100 rows of 200,000: both
        # 20 M characters of contents from 20 to 200 KB.
        [(20, 50, 20_000), (1, 100, 200_000)],
        ids=["colspan-and-rowspan", "rowspan"],
    )
    def test_a_spanned_cell_repeating_long_text_is_not_read(self, colspan, rowspan, chars, caplog):
        budget = TableBudget()

        with caplog.at_level(logging.WARNING, logger="bibr.structure.html_table"):
            df = html_table_frame(_spanned("x" * chars, colspan, rowspan), budget)

        assert df is None
        assert "text passes the table size limits" in caplog.text
        assert (budget.refused, budget.bytes, budget.slots) == (1, 0, 0)

    def test_a_table_at_the_size_limit_is_read(self, monkeypatch):
        monkeypatch.setattr(html_table, "MAX_TABLE_BYTES", 20 * (5 + M))

        df = html_table_frame(_spanned("x" * 5, 20, 1))
        over = html_table_frame(_spanned("x" * 6, 20, 1))

        assert df is not None
        assert df.shape == (1, 20)
        assert over is None

    def test_the_limits_fit_long_printed_tables(self):
        # 2,000 rows of twenty 400-character cells, and the DOCX limits.
        size = html_table.TableSize(html_table.MAX_TABLE_BYTES)
        assert size.add(["x" * 400] * 20 * 2_000)
        assert html_table.MAX_TABLE_BYTES <= html_table.MAX_DOCUMENT_TABLE_BYTES == 64 * 2**20
        assert html_table.MAX_DOCUMENT_TABLE_SLOTS == 4_000_000

    @pytest.mark.parametrize(
        "cell",
        ["\U0001f600" * 10, "&amp;" * 10, "&lt;" * 10],
        ids=["wide", "ampersand", "angle-bracket"],
    )
    def test_a_header_cell_repeated_in_wide_or_escaped_text_is_refused(self, monkeypatch, cell):
        """Ten characters across 20 columns over 20 empty cells render to 840
        ASCII characters, but 3,360 bytes at 4 bytes a character, and 1,640
        or 1,440 characters escaped."""
        monkeypatch.setattr(html_table, "MAX_TABLE_BYTES", 1_000)

        def table(text: str) -> str:
            return (
                f'<table><thead><tr><th colspan="20">{text}</th></tr></thead>'
                "<tbody><tr>" + "<td></td>" * 20 + "</tr></tbody></table>"
            )

        budget = TableBudget()
        assert html_table_frame(table("x" * 10), budget) is not None
        assert budget.bytes == 20 * (10 + M) + 20 * M
        assert html_table_frame(table(cell), budget) is None
        assert budget.refused == 1

    def test_merged_cells_in_an_ordinary_table_still_read(self):
        html = (
            '<table><tr><th rowspan="2">Group</th><th colspan="3">Measures</th></tr>'
            "<tr><th>a</th><th>b</th><th>c</th></tr>"
            + ('<tr><td colspan="2">' + "g" * 300 + "</td><td>1</td><td>2</td></tr>") * 50
            + "</table>"
        )
        budget = TableBudget()

        df = html_table_frame(html, budget)

        assert df is not None
        assert df.shape == (50, 4)
        header = 2 * (5 + M) + 3 * (8 + M) + 3 * (1 + M)
        assert budget.bytes == header + 50 * (2 * (300 + M) + 2 * (1 + M))
        assert budget.slots == 52 * 4


class TestRenderedSize:
    def test_escapes_and_character_width_are_counted(self):
        assert html_table.rendered_size("a&b<c>") == (6 + 4 + 3 + 3, 1)
        assert html_table.rendered_size("café") == (4, 1)
        assert html_table.rendered_size("naïve Ω") == (7, 2)
        assert html_table.rendered_size("x\U0001f600") == (2, 4)
        assert html_table.rendered_size("") == (0, 1)

    def test_the_widest_character_sets_the_width_of_the_table(self):
        size = TableSize(10**6)

        assert size.add(["abc", "abc"])
        assert size.bytes == 2 * (3 + M)
        assert size.add(["Ω"])
        assert size.bytes == 2 * (2 * (3 + M) + 1 + M)


class TestHeaderNames:
    def test_the_names_are_pandas_names(self):
        """Blank cells are "Unnamed: <column>", named cells are named first,
        and a repeated name takes the first ".<n>" no other column holds."""
        rng = random.Random(7)  # noqa: S311 - deterministic test fixture, not crypto
        choices = ["", "", "A", "A", "A.1", "A.2", "A.1.1", "B", "Unnamed: 1", "Unnamed: 0.1"]
        for _ in range(500):
            row = [rng.choice(choices) for _ in range(rng.randint(2, 8))]
            with TextParser([row, [""] * len(row)], header=0, dtype=str, na_filter=False) as p:
                expected = list(p.read().columns)

            assert html_table._header_names(row) == expected, row

    def test_pandas_is_given_unique_names(self, monkeypatch):
        """pandas looks every repeated name up in the list of names, and every
        column up in the list of blank ones: naming 20,000 columns of one name
        took 3 s, growing with the square of the width."""
        headers = []
        text_parser = html_table.TextParser

        def recording(rows, **kwargs):
            headers.append(rows[0])
            return text_parser(rows, **kwargs)

        monkeypatch.setattr(html_table, "TextParser", recording)
        html = (
            "<table><tr>" + '<th colspan="1000">x</th>' * 3 + "<th></th>" * 2 + "</tr>"
            "<tr>" + "<td>v</td>" * 1_510 + "</tr></table>"
        )

        df = html_table_frame(html)

        assert df is not None
        assert df.shape == (1, 3_002)
        (header,) = headers
        assert len(set(header)) == len(header)
        assert list(df.columns[:3]) == ["x", "x.1", "x.2"]
        assert list(df.columns[-3:]) == ["x.2999", "Unnamed: 3000", "Unnamed: 3001"]

    @pytest.mark.parametrize(
        "head",
        [
            "<tr>{names}</tr><tr></tr>",
            "<tr></tr><tr>{names}</tr>",
            "<tr></tr><tr>{names}</tr><tr></tr>",
        ],
        ids=["blank-after", "blank-before", "blank-around"],
    )
    def test_a_header_row_among_blank_ones_is_named_as_pandas_names_it(self, monkeypatch, head):
        # Blank head rows make the header [k], which pandas names with the
        # same quadratic loop as a header of one row.
        headers = []
        text_parser = html_table.TextParser

        def recording(rows, **kwargs):
            header = kwargs["header"]
            headers.append(rows[header if isinstance(header, int) else header[0]])
            return text_parser(rows, **kwargs)

        names = "<th>x</th><th></th><th>x</th><th>x.1</th><th></th><th>y</th>"
        html = (
            "<table><thead>" + head.format(names=names) + "</thead>"
            "<tbody><tr>" + "<td>v</td>" * 6 + "</tr></tbody></table>"
        )
        raw = [["x", "", "x", "x.1", "", "y"], ["v"] * 6]
        with text_parser(raw, header=0, dtype=str, na_filter=False) as parser:
            expected = list(parser.read().columns)

        monkeypatch.setattr(html_table, "TextParser", recording)
        df = html_table_frame(html)

        assert df is not None
        assert list(df.columns) == expected
        (header,) = headers
        assert header == expected

    def test_one_blank_header_cell_is_still_skipped_as_pandas_skips_it(self):
        html = "<table><thead><tr><th></th></tr></thead><tbody><tr><td>a</td></tr></tbody></table>"

        df = html_table_frame(html)

        assert df is not None
        assert list(df.columns) == ["a"]
        assert df.empty


class TestDocumentBudget:
    def test_tables_share_the_document_size_limit(self, monkeypatch):
        monkeypatch.setattr(html_table, "MAX_DOCUMENT_TABLE_BYTES", 2 * (40 + M) + 2 + M)
        budget = TableBudget()
        table = "<table><tr><td>" + "x" * 40 + "</td></tr></table>"

        frames = [html_table_frame(table, budget) for _ in range(3)]

        assert [df is not None for df in frames] == [True, True, False]
        assert (budget.bytes, budget.refused) == (2 * (40 + M), 1)
        # A table that still fits is read.
        fits = html_table_frame("<table><tr><td>xx</td></tr></table>", budget)
        assert fits is not None

    def test_tables_share_the_document_cell_limit(self, monkeypatch, caplog):
        monkeypatch.setattr(html_table, "MAX_DOCUMENT_TABLE_SLOTS", 10)
        budget = TableBudget()
        table = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td></tr></table>"

        with caplog.at_level(logging.WARNING, logger="bibr.structure.html_table"):
            frames = [html_table_frame(table, budget) for _ in range(3)]

        assert [df is not None for df in frames] == [True, True, False]
        assert budget.slots == 8
        assert "table cell limit" in caplog.text

    def test_a_table_out_of_proportion_is_counted_as_refused(self):
        budget = TableBudget()
        padding = "<table><tr>" + "<td>w</td>" * 300 + "</tr>" + "<tr><td>n</td></tr>" * 300

        assert html_table_frame(padding + "</table>", budget) is None
        assert budget.refused == 1
        assert [w.code for w in budget.warnings()] == [WarningCode.TABLE_CONTENTS_OMITTED]

    def test_no_warning_without_a_refused_table(self):
        budget = TableBudget()

        assert html_table_frame("<table><tr><td>a</td></tr></table>", budget) is not None
        assert budget.warnings() == []
