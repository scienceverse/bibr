"""HTML table text is bounded, not only its slots (final review, 2026-10-07).

The span limits bounded how many slots a table expands to, but every slot a
cell spans holds its text again: one long cell spanning 20 columns down 5,000
bare ``<tr>`` made 40 KB of HTML into 2 GB of ``contents``. A table's slot
text is now limited, and so are the text and slots of a document's tables.
"""

from __future__ import annotations

import logging

import pytest

from bibr.processing_warnings import WarningCode
from bibr.structure import html_table
from bibr.structure.html_table import TableBudget, html_table_frame


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
        assert "text passes the table text limits" in caplog.text
        assert (budget.refused, budget.chars, budget.slots) == (1, 0, 0)

    def test_a_table_at_the_text_limit_is_read(self, monkeypatch):
        monkeypatch.setattr(html_table, "MAX_TABLE_CHARS", 100)

        df = html_table_frame(_spanned("x" * 5, 20, 1))
        over = html_table_frame(_spanned("x" * 6, 20, 1))

        assert df is not None
        assert df.shape == (1, 20)
        assert over is None

    def test_the_limits_fit_long_printed_tables(self):
        # 2,000 rows of twenty 400-character cells, and the DOCX cell limit.
        assert html_table.MAX_TABLE_CHARS >= 2_000 * 20 * 400
        assert html_table.MAX_TABLE_CHARS <= html_table.MAX_DOCUMENT_TABLE_CHARS
        assert html_table.MAX_DOCUMENT_TABLE_SLOTS == 4_000_000

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
        assert budget.chars == 50 * (2 * 300 + 2) + 2 * len("Group") + 3 * len("Measures") + 3
        assert budget.slots == 52 * 4


class TestDocumentBudget:
    def test_tables_share_the_document_text_limit(self, monkeypatch):
        monkeypatch.setattr(html_table, "MAX_DOCUMENT_TABLE_CHARS", 100)
        budget = TableBudget()
        table = "<table><tr><td>" + "x" * 40 + "</td></tr></table>"

        frames = [html_table_frame(table, budget) for _ in range(3)]

        assert [df is not None for df in frames] == [True, True, False]
        assert (budget.chars, budget.refused) == (80, 1)
        # A table that still fits is read.
        fits = html_table_frame("<table><tr><td>" + "x" * 20 + "</td></tr></table>", budget)
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
