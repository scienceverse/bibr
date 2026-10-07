"""Table limits for HTML, ePub and JATS input (final review, 2026-10-07).

HTML and ePub tables bounded their slots but not the text each slot repeats,
so 40 KB of HTML made 2 GB of contents. JATS tables padded every row to the
widest with no limit at all (one row of 4,000 ``<td/>`` over 4,000 empty
``<tr/>``: 40 KB, 135 s and 2 GB), and read a nested table's rows once for
every table around it. Both now share the HTML table limits and one budget
per document, and record the tables left without contents as
``TABLE_CONTENTS_OMITTED``.
"""

from __future__ import annotations

import io
import logging
import zipfile

import pytest

from bibr.input.epub_native import EpubParser
from bibr.input.html_native import HtmlParser
from bibr.input.jats_native import JatsParser
from bibr.processing_warnings import WarningCode
from bibr.structure import html_table


def _codes(contents) -> list[str]:
    return [w.code for w in contents.processing_warnings]


def _html(*tables: str) -> bytes:
    return (
        "<html><body><article><h1>A Study</h1><p>Body text.</p>"
        + "".join(tables)
        + "</article></body></html>"
    ).encode()


def _article(*wraps: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><article><front><article-meta><title-group>'
        "<article-title>A Study</article-title></title-group></article-meta></front>"
        "<body><sec><title>Results</title><p>See Table 1.</p>"
        + "".join(wraps)
        + "</sec></body></article>"
    ).encode()


def _wrap(table: str, label: str = "Table 1") -> str:
    head = f"<label>{label}</label><caption><p>Data.</p></caption>" if label else ""
    return f"<table-wrap>{head}{table}</table-wrap>"


def _padded(width: int, rows: int) -> str:
    """One row of *width* empty cells over *rows* empty rows."""
    return "<table><tr>" + "<td/>" * width + "</tr>" + "<tr/>" * rows + "</table>"


class TestHtmlInput:
    def test_a_long_spanned_cell_keeps_its_caption_without_contents(self):
        """20 x 50 slots of a 20,000-character cell: 20 M characters."""
        table = (
            '<table><caption>Table 1. Data</caption><tr><td colspan="20" rowspan="50">'
            + "word " * 4_000
            + "</td></tr>"
            + "<tr>" * 49
            + "</table>"
        )

        contents = HtmlParser(_html(table)).parse()

        assert len(contents.tables) == 1
        assert contents.tables[0].label is not None
        assert contents.tables[0].df.empty
        assert contents.tables[0].contents == []
        assert _codes(contents) == [WarningCode.TABLE_CONTENTS_OMITTED]

    def test_the_tables_of_a_document_share_one_text_limit(self, monkeypatch):
        monkeypatch.setattr(html_table, "MAX_DOCUMENT_TABLE_CHARS", 100)
        table = (
            "<table><caption>Table {}. Data</caption><tr><th>A</th></tr><tr><td>"
            + "x" * 40
            + "</td></tr></table>"
        )

        contents = HtmlParser(_html(*(table.format(n) for n in (1, 2, 3)))).parse()

        assert [t.df.shape for t in contents.tables] == [(1, 1), (1, 1), (0, 0)]
        assert [t.contents for t in contents.tables] == [[["A"], ["x" * 40]]] * 2 + [[]]
        (warning,) = contents.processing_warnings
        assert warning.code == WarningCode.TABLE_CONTENTS_OMITTED
        assert "1 table(s)" in warning.message

    def test_an_uncaptioned_table_over_the_limits_is_dropped(self, monkeypatch):
        monkeypatch.setattr(html_table, "MAX_TABLE_CHARS", 10)

        contents = HtmlParser(_html("<table><tr><td>" + "x" * 11 + "</td></tr></table>")).parse()

        assert contents.tables == []
        assert _codes(contents) == [WarningCode.TABLE_CONTENTS_OMITTED]

    def test_ordinary_tables_record_no_warning(self):
        table = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"

        contents = HtmlParser(_html(table, table)).parse()

        assert [t.contents for t in contents.tables] == [[["A", "B"], ["1", "2"]]] * 2
        assert contents.processing_warnings == []


def _epub(*chapters: str) -> bytes:
    manifest = "".join(
        f'<item id="c{i}" href="ch{i}.xhtml" media-type="application/xhtml+xml"/>'
        for i in range(len(chapters))
    )
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
        )
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>",
        )
        for i, body in enumerate(chapters):
            zf.writestr(
                f"OEBPS/ch{i}.xhtml",
                '<html xmlns="http://www.w3.org/1999/xhtml"><body><section>'
                f"<h1>Chapter {i}</h1><p>Text.</p>{body}</section></body></html>",
            )
    return buf.getvalue()


def test_the_chapters_of_an_epub_share_one_table_budget(monkeypatch):
    monkeypatch.setattr(html_table, "MAX_DOCUMENT_TABLE_SLOTS", 6)
    table = (
        "<table><caption>Table {}. Data</caption><tr><th>A</th><th>B</th></tr>"
        "<tr><td>a</td><td>b</td></tr></table>"
    )

    contents = EpubParser(_epub(table.format(1), table.format(2))).parse()

    assert [t.contents for t in contents.tables] == [[["A", "B"], ["a", "b"]], []]
    assert _codes(contents) == [WarningCode.TABLE_CONTENTS_OMITTED]


class TestJatsTables:
    def test_one_wide_row_padding_many_empty_rows_is_not_read(self, caplog):
        """300 x 300 padded cells from 600 elements; at 4,000 x 4,000 the
        padding took 135 s and 2 GB."""
        with caplog.at_level(logging.WARNING, logger="bibr.input.jats_native"):
            contents = JatsParser(_article(_wrap(_padded(300, 300)))).parse()

        (table,) = contents.tables
        assert table.caption == "Table 1 Data."
        assert table.df.empty
        assert table.contents == []
        assert table.tbl_html == ""
        assert _codes(contents) == [WarningCode.TABLE_CONTENTS_OMITTED]
        assert "pads its rows far past its cells" in caplog.text

    def test_an_uncaptioned_table_over_the_limits_is_dropped(self):
        contents = JatsParser(_article(_wrap(_padded(300, 300), label=""))).parse()

        assert contents.tables == []
        assert _codes(contents) == [WarningCode.TABLE_CONTENTS_OMITTED]

    def test_padding_is_bounded_as_html_tables_are(self):
        """A row of 100 cells over 48 one-cell rows is 4,900 cells, within 20
        for each of the 197 cells and rows plus 1,000; one more row is not."""
        wide = "<tr>" + "<td>w</td>" * 100 + "</tr>"

        def parse(rows: int):
            table = "<table>" + wide + "<tr><td>n</td></tr>" * rows + "</table>"
            return JatsParser(_article(_wrap(table))).parse()

        within, past = parse(48), parse(49)

        assert within.tables[0].df.shape == (48, 100)
        assert within.processing_warnings == []
        assert past.tables[0].df.empty
        assert _codes(past) == [WarningCode.TABLE_CONTENTS_OMITTED]

    def test_a_ragged_printed_table_still_reads(self):
        rows = "".join(
            "<tr><td>Group</td></tr>"
            if i % 10 == 0
            else "<tr>" + "".join(f"<td>{i}.{j}</td>" for j in range(8)) + "</tr>"
            for i in range(200)
        )
        table = "<table><thead><tr>" + "<th>h</th>" * 8 + "</tr></thead>" + rows + "</table>"

        contents = JatsParser(_article(_wrap(table))).parse()

        assert contents.tables[0].df.shape == (200, 8)
        assert contents.tables[0].df.iloc[0].tolist() == ["Group"] + [""] * 7
        assert contents.processing_warnings == []

    def test_a_nested_table_is_read_once(self):
        """Its rows were read with the table around it and again on their
        own, after every row of the outer table."""
        table = (
            "<table><tr><th>A</th><th>B</th></tr><tr><td>x</td><td>"
            "<table><tr><td>i1</td></tr><tr><td>i2</td></tr></table></td></tr></table>"
        )

        contents = JatsParser(_article(_wrap(table))).parse()

        assert contents.tables[0].df.values.tolist() == [["x", "i1 i2"], ["i1", ""], ["i2", ""]]

    def test_deeply_nested_tables_are_not_read_quadratically(self):
        """30 tables nested in one another gave 465 rows, each holding the
        text of every table inside it."""
        inner = "y" * 1_000
        for depth in range(30):
            inner = f"<table><tr><td>{depth}</td><td>{inner}</td></tr></table>"

        contents = JatsParser(_article(_wrap(inner))).parse()

        df = contents.tables[0].df
        assert df.shape == (29, 2)
        assert sum(len(v) for row in df.values.tolist() for v in row) < 30 * 1_100

    def test_a_table_over_the_text_limit_keeps_its_caption(self, monkeypatch, caplog):
        monkeypatch.setattr(html_table, "MAX_TABLE_CHARS", 50)
        table = "<table><tr><th>A</th></tr>" + "<tr><td>0123456789</td></tr>" * 5 + "</table>"

        with caplog.at_level(logging.WARNING, logger="bibr.input.jats_native"):
            contents = JatsParser(_article(_wrap(table))).parse()

        assert contents.tables[0].df.empty
        assert _codes(contents) == [WarningCode.TABLE_CONTENTS_OMITTED]
        assert "text passes the table text limits" in caplog.text

    @pytest.mark.parametrize(
        ("limit", "name"),
        [(10, "MAX_DOCUMENT_TABLE_SLOTS"), (20, "MAX_DOCUMENT_TABLE_CHARS")],
        ids=["cells", "text"],
    )
    def test_the_tables_of_a_document_share_one_budget(self, monkeypatch, limit, name):
        monkeypatch.setattr(html_table, name, limit)
        table = "<table><tr><th>A</th><th>B</th></tr><tr><td>1234</td><td>5678</td></tr></table>"

        contents = JatsParser(
            _article(*(_wrap(table, label=f"Table {n}") for n in (1, 2, 3)))
        ).parse()

        assert [t.df.shape for t in contents.tables] == [(1, 2), (1, 2), (0, 0)]
        (warning,) = contents.processing_warnings
        assert warning.code == WarningCode.TABLE_CONTENTS_OMITTED
        assert "1 table(s)" in warning.message
