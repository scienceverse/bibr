"""Table HTML fidelity.

An OCR/VLM table region emitted as HTML carries structure the flat
DataFrame cannot represent: ``rowspan``/``colspan`` merged cells and
multi-level headers. The parser must preserve that HTML **verbatim** in
``tbl_html`` (round-tripping it through ``pandas.read_html`` ->
``DataFrame.to_html`` drops ``rowspan``, mangles stub headers into
``Unnamed: 0_level_0`` and injects ``class="dataframe"``/``halign`` noise).
The DataFrame is still used for the flattened ``contents`` grid.

Markdown-input tables have no source HTML, so they keep the
DataFrame-rendered ``tbl_html`` as before.
"""

from __future__ import annotations


def _region(index, label, content, bbox=None):
    return {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": bbox or [0, 0, 100, 100],
    }


def _parse(json_result):
    from bibr.structure.pdf_parser import PDFParser

    return PDFParser(json_result).parse()


# Multi-level header with BOTH a rowspan (stub header) and colspan (group).
_HTML_TABLE = (
    "<table>"
    '<tr><th rowspan="2">Var</th><th colspan="2">Group A</th></tr>'
    "<tr><th>M</th><th>SD</th></tr>"
    "<tr><td>Age</td><td>30</td><td>5</td></tr>"
    "</table>"
)


class TestHtmlTableFidelity:
    def test_html_table_preserved_verbatim(self):
        """An HTML table region keeps its original markup in ``tbl_html``."""
        contents = _parse([[_region(0, "table", _HTML_TABLE, bbox=[0, 0, 100, 100])]])
        assert len(contents.tables) == 1
        tbl_html = contents.tables[0].tbl_html
        # Verbatim preservation: the exact VLM markup, no lossy round-trip.
        assert tbl_html == _HTML_TABLE
        # Structural guarantees the round-trip destroys today:
        assert 'rowspan="2"' in tbl_html
        assert "Unnamed" not in tbl_html
        assert 'class="dataframe"' not in tbl_html

    def test_html_table_still_yields_contents_grid(self):
        """The flattened ``contents`` grid is still derived from the DataFrame."""
        contents = _parse([[_region(0, "table", _HTML_TABLE, bbox=[0, 0, 100, 100])]])
        grid = contents.tables[0].contents
        assert grid, "expected a non-empty flattened contents grid"
        # Data row survives the flatten.
        flat = [cell for row in grid for cell in row]
        assert "Age" in flat

    def test_markdown_table_still_renders_html_from_dataframe(self):
        """Markdown input has no source HTML; keep the DataFrame-rendered table."""
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        contents = _parse([[_region(0, "table", md, bbox=[0, 0, 100, 100])]])
        assert len(contents.tables) == 1
        tbl_html = contents.tables[0].tbl_html
        assert tbl_html.lstrip().startswith("<table")
        assert "A" in tbl_html and "B" in tbl_html
