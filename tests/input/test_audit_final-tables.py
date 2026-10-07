"""Table limits for HTML, ePub and JATS input (final review, 2026-10-07).

HTML and ePub tables bounded their slots but not the text each slot repeats,
so 40 KB of HTML made 2 GB of contents. JATS tables padded every row to the
widest with no limit at all (one row of 4,000 ``<td/>`` over 4,000 empty
``<tr/>``: 40 KB, 135 s and 2 GB), and read a nested table's rows once for
every table around it, each enclosing cell holding (and walking) the text of
the tables inside it again. Both now share the HTML table limits, measured
as the table HTML renders, and one budget per document, and record the
tables left without contents as ``TABLE_CONTENTS_OMITTED``.
"""

from __future__ import annotations

import io
import logging
import zipfile

import pytest
from lxml import etree

from bibr.input import jats_native
from bibr.input.epub_native import EpubParser
from bibr.input.html_native import HtmlParser
from bibr.input.jats_native import JatsParser
from bibr.processing_warnings import WarningCode
from bibr.structure import html_table

M = html_table._CELL_MARKUP_CHARS


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

    def test_the_tables_of_a_document_share_one_size_limit(self, monkeypatch):
        # Two tables of "A" over 40 characters fit, a third does not.
        monkeypatch.setattr(html_table, "MAX_DOCUMENT_TABLE_BYTES", 2 * (1 + 40 + 2 * M) + 10)
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
        monkeypatch.setattr(html_table, "MAX_TABLE_BYTES", 10 + M)

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
        own, after every row of the outer table, and the cell holding it
        repeated its text. Its rows now follow that row, once."""
        table = (
            "<table><tr><th>A</th><th>B</th></tr><tr><td>x</td><td>"
            "<table><tr><td>i1</td></tr><tr><td>i2</td></tr></table></td></tr></table>"
        )

        contents = JatsParser(_article(_wrap(table))).parse()

        assert contents.tables[0].df.values.tolist() == [["x", ""], ["i1", ""], ["i2", ""]]

    def test_a_nested_table_keeps_the_words_around_it_apart(self):
        table = (
            "<table><tr><th>A</th></tr><tr><td>before<table><tr><td>in</td></tr></table>after"
            "</td></tr></table>"
        )

        contents = JatsParser(_article(_wrap(table))).parse()

        assert contents.tables[0].df.values.tolist() == [["before after"], ["in"]]

    def test_deeply_nested_tables_are_not_read_quadratically(self):
        """30 tables nested in one another gave 465 rows, each holding the
        text of every table inside it."""
        inner = "y" * 1_000
        for depth in range(30):
            inner = f"<table><tr><td>{depth}</td><td>{inner}</td></tr></table>"

        contents = JatsParser(_article(_wrap(inner))).parse()

        rows = contents.tables[0].df.values.tolist()
        assert len(rows) == 29
        assert rows[-1] == ["0", "y" * 1_000]
        assert sum(len(v) for row in rows for v in row) == 1_000 + len("".join(map(str, range(29))))

    def test_each_element_of_nested_tables_is_walked_once(self, monkeypatch):
        """Every cell up the nesting walked the tables inside it again: 60
        levels over 200,000 empty elements (800 KB) took 15 s."""
        calls = 0
        walk = jats_native._Walker.walk

        def counting(self, *args):
            nonlocal calls
            calls += 1
            return walk(self, *args)

        monkeypatch.setattr(jats_native._Walker, "walk", counting)
        inner = "<x/>" * 1_000
        for depth in range(30):
            inner = f"<table><tr><td>{depth}</td><td>{inner}</td></tr></table>"

        contents = JatsParser(_article(_wrap(inner))).parse()

        assert contents.tables[0].df.shape == (29, 2)
        assert calls < 2 * 1_000

    def test_the_nested_table_check_looks_only_at_table_ancestors(self):
        """It walked every ancestor of every <table> in Python: many tables
        deep in one wrap took twelve times as long."""
        yielded = 0

        class Counting(etree.ElementBase):
            def iterancestors(self, *args, **kwargs):
                nonlocal yielded
                for el in super().iterancestors(*args, **kwargs):
                    yielded += 1
                    yield el

        parser = etree.XMLParser()
        parser.set_element_class_lookup(etree.ElementDefaultClassLookup(element=Counting))
        tables = "".join(f"<table><tr><td>{i}</td></tr></table>" for i in range(20))
        wrap = etree.fromstring(
            "<table-wrap>" + "<x>" * 50 + tables + "</x>" * 50 + "</table-wrap>", parser
        )

        df = JatsParser._tables_to_df([el for el in wrap.iter() if el.tag == "table"])

        assert df is not None
        assert df.shape == (19, 1)
        assert yielded == 0

    @pytest.mark.parametrize(
        "cell",
        ["&amp;" * 250, "\U0001f600" * 250],
        ids=["escaped", "wide"],
    )
    def test_text_is_charged_at_its_rendered_size(self, monkeypatch, cell):
        """250 characters render to 1,250 escaped, or take 4 bytes each in
        the whole table HTML, so they pass 1,000 bytes where plain text does
        not."""
        monkeypatch.setattr(html_table, "MAX_TABLE_BYTES", 1_000)

        def parse(text: str):
            table = f"<table><tr><th>A</th></tr><tr><td>{text}</td></tr></table>"
            return JatsParser(_article(_wrap(table))).parse()

        plain, rendered = parse("a" * 250), parse(cell)

        assert plain.tables[0].df.shape == (1, 1)
        assert plain.processing_warnings == []
        assert rendered.tables[0].df.empty
        assert rendered.tables[0].tbl_html == ""
        assert _codes(rendered) == [WarningCode.TABLE_CONTENTS_OMITTED]

    def test_a_table_over_the_size_limit_keeps_its_caption(self, monkeypatch, caplog):
        monkeypatch.setattr(html_table, "MAX_TABLE_BYTES", 100)
        table = "<table><tr><th>A</th></tr>" + "<tr><td>0123456789</td></tr>" * 5 + "</table>"

        with caplog.at_level(logging.WARNING, logger="bibr.input.jats_native"):
            contents = JatsParser(_article(_wrap(table))).parse()

        assert contents.tables[0].df.empty
        assert _codes(contents) == [WarningCode.TABLE_CONTENTS_OMITTED]
        assert "text passes the table size limits" in caplog.text

    @pytest.mark.parametrize(
        ("limit", "name"),
        # A table of "A", "B" over "1234", "5678" renders to 74 characters.
        [(10, "MAX_DOCUMENT_TABLE_SLOTS"), (150, "MAX_DOCUMENT_TABLE_BYTES")],
        ids=["cells", "size"],
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
