"""Table HTML fidelity.

An OCR/VLM table region emitted as HTML carries structure the flat
DataFrame cannot represent: ``rowspan``/``colspan`` merged cells and
multi-level headers. The parser must preserve that HTML **verbatim** in
``tbl_html`` (round-tripping it through the DataFrame ->
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


def _export_contents(contents):
    """``table[].contents`` of the validated JSON export of *contents*."""
    from pathlib import Path

    from bibr.export import export_paper_to_json
    from bibr.input.file import InputFile, InputFormat
    from bibr.models import PaperMetadata
    from bibr.paper import Paper

    paper = Paper(
        input_file=InputFile(
            path=Path("/tmp/paper.pdf"),
            file_hash="hash",
            input_format=InputFormat(
                file_extension=".pdf",
                detected_mime_type="application/pdf",
                file_type="pdf",
            ),
        ),
        metadata=PaperMetadata(title="Paper", doi=""),
        contents=contents,
    )
    return [table["contents"] for table in export_paper_to_json(paper, validate=True)["table"]]


class TestTableCellsKeepPrintedText:
    """``read_html`` type inference rewrote numeric columns: "2.50" -> "2.5",
    "007" -> "7", "1,234" -> "1234", a decimal comma "1,5" -> "15", and an int
    column with an empty cell -> "12.0". ``table[].contents`` is cell text."""

    def test_ocr_html_cells_are_exported_as_printed(self):
        html = (
            "<table><tr><th>N</th><th>Code</th><th>Count</th><th>M</th><th>p</th></tr>"
            "<tr><td>12</td><td>007</td><td>1,234</td><td>2.50</td><td>.050</td></tr>"
            "<tr><td></td><td>010</td><td>56</td><td>1,5</td><td>0.10</td></tr></table>"
        )
        contents = _parse([[_region(0, "table", html)]])

        expected = [
            ["N", "Code", "Count", "M", "p"],
            ["12", "007", "1,234", "2.50", ".050"],
            ["", "010", "56", "1,5", "0.10"],
        ]
        assert contents.tables[0].contents == expected
        assert _export_contents(contents) == [expected]

    def test_headerless_ocr_table_promotes_its_first_row_as_printed(self):
        """Paddle's OTSL decodes to <td>-only tables; the first row becomes the
        header. A numeric column with an empty cell turned its header "2019"
        into "2019.0" and its cells into floats."""
        html = (
            "<table><tr><td>Region</td><td>2019</td><td>2020</td></tr>"
            "<tr><td>Nord</td><td>1,5</td><td>0,25</td></tr>"
            "<tr><td>Süd</td><td></td><td>1,0</td></tr></table>"
        )
        contents = _parse([[_region(0, "table", html)]])

        assert contents.tables[0].contents == [
            ["Region", "2019", "2020"],
            ["Nord", "1,5", "0,25"],
            ["Süd", "", "1,0"],
        ]
