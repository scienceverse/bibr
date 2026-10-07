"""DOCX parser resource limits and inline text (audit 2026-10-06, §1.2, §1.3, §4.4).

- Table cells came from python-docx ``row.cells``, which repeats a cell once
  per spanned grid column (``w:gridSpan`` is a free integer) and resolves each
  ``w:vMerge`` continuation by recursing up its column: a 36 KB file ran for
  days, and tall merged columns ended in RecursionError. With spans clamped, a
  wide table's columns and a merged cell's repeated text still multiplied.
- Every picture showing one image part got its own base64 copy of the image.
- Text inside inline content controls, simple fields, smart tags, custom XML
  and moves was dropped, from body paragraphs and from headings, captions and
  table cells.
"""

from __future__ import annotations

import base64
import copy
import io
import os
import struct
import zlib

import pytest

pytest.importorskip("docx")

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

import bibr.input.docx_native as docx_native
from bibr.input.docx_native import DocxParser
from bibr.processing_warnings import WarningCode

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
HYPERLINK = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _save(doc) -> bytes:
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _parse(data: bytes):
    parser = DocxParser(data)
    return parser, parser.parse()


def _codes(contents) -> list[str]:
    return [w.code for w in contents.processing_warnings]


# ----- Tables -----


def _tbl(rows_xml: str) -> etree._Element:
    return etree.fromstring(f'<w:tbl xmlns:w="{W}"><w:tblPr/><w:tblGrid/>{rows_xml}</w:tbl>')


def _tc(text: str = "", props: str = "") -> str:
    run = f"<w:r><w:t>{text}</w:t></w:r>" if text else ""
    return f"<w:tc><w:tcPr>{props}</w:tcPr><w:p>{run}</w:p></w:tc>"


def _span(columns: int | str) -> str:
    return f'<w:gridSpan w:val="{columns}"/>'


def _doc_with_table(rows_xml: str, *, tables: int = 1) -> bytes:
    doc = Document()
    doc.add_paragraph("Body text.")
    for _ in range(tables):
        doc.element.body.insert(0, _tbl(rows_xml))
    return _save(doc)


def _merged_column(rows: int, span: int) -> str:
    """A table of *rows* rows holding one cell *span* columns wide, merged
    down the whole table."""
    restart = _tc("Merged", _span(span) + '<w:vMerge w:val="restart"/>')
    cont = _tc("", _span(span) + "<w:vMerge/>")
    return f"<w:tr>{restart}</w:tr>" + f"<w:tr>{cont}</w:tr>" * (rows - 1)


def test_grid_span_is_clamped():
    """python-docx repeated the cell once per spanned column: 5,000 columns
    here, 2e9 (days of CPU) for the same few bytes."""
    rows = f"<w:tr>{_tc('Wide', _span(5000))}</w:tr><w:tr>{_tc('x')}</w:tr>"
    _, contents = _parse(_doc_with_table(rows))

    (table,) = contents.tables
    assert table.df.shape == (1, docx_native._MAX_GRID_SPAN)
    assert set(table.df.columns) == {"Wide"}
    assert table.df.iloc[0].tolist() == ["x"] + [""] * (docx_native._MAX_GRID_SPAN - 1)


def test_huge_grid_span_is_bounded():
    rows = f"<w:tr>{_tc('Wide', _span(2_000_000_000))}</w:tr>"
    _, contents = _parse(_doc_with_table(rows))

    assert contents.tables[0].df.shape == (0, docx_native._MAX_GRID_SPAN)


def test_tall_merged_column_reads_without_recursion():
    """python-docx followed each continuation up the column recursively:
    quadratic time, then RecursionError near 1,500 rows."""
    _, contents = _parse(_doc_with_table(_merged_column(3000, 2)))

    (table,) = contents.tables
    assert table.df.shape == (2999, 2)
    assert (table.df.to_numpy() == "Merged").all()


def test_table_over_the_cell_limit_is_dropped_with_its_caption_kept_as_text():
    """1,001 rows of a 1,000-column merged cell: a few KB for 1,001,000 grid cells."""
    doc = Document()
    doc.add_paragraph("Table 1. Too large to keep.", style="Caption")
    doc.add_paragraph("After.")
    doc.paragraphs[0]._p.addnext(_tbl(_merged_column(1001, 1000)))
    parser, contents = _parse(_save(doc))

    assert contents.tables == []
    assert _codes(contents) == [WarningCode.DOCX_TABLE_DROPPED]
    # The caption is not taken along: it stays in the body text.
    assert [e.text for e in parser.assembler.entries] == ["Table 1. Too large to keep.", "After."]


def test_tables_past_the_document_cell_limit_are_dropped(monkeypatch):
    # A 10 x 10 table counts its 100 cells and _COLUMN_CELLS for each column.
    size = (10 + docx_native._COLUMN_CELLS) * 10
    monkeypatch.setattr(docx_native, "_MAX_TABLE_CELLS", size)
    monkeypatch.setattr(docx_native, "_MAX_DOCUMENT_TABLE_CELLS", size * 5 // 2)
    row = "<w:tr>" + _tc("c") * 10 + "</w:tr>"
    _, contents = _parse(_doc_with_table(row * 10, tables=3))

    assert len(contents.tables) == 2
    (warning,) = contents.processing_warnings
    assert warning.code == WarningCode.DOCX_TABLE_DROPPED
    assert warning.message.startswith("Dropped 1 table(s)")


def test_a_wide_table_is_charged_for_its_columns():
    """pandas pays per column what it pays for some 20 cells: one row of 1,000
    cells spanning 1,000 columns each (36 KB) passed the cell limit as
    1,000,000 cells and took three minutes. Twenty such cells pass it now."""
    row = "<w:tr>" + _tc("", _span(1000)) * 20 + "</w:tr>"
    _, contents = _parse(_doc_with_table(row))

    assert contents.tables == []
    assert _codes(contents) == [WarningCode.DOCX_TABLE_DROPPED]


def test_wide_tables_are_charged_for_their_columns_per_document(monkeypatch):
    """Spread over one-cell tables, the same columns passed the document limit:
    4,000 tables of one 1,000-column cell (340 KB) took eight minutes."""
    monkeypatch.setattr(docx_native, "_MAX_DOCUMENT_TABLE_CELLS", 250_000)
    row = f"<w:tr>{_tc('', _span(1000))}</w:tr>"
    _, contents = _parse(_doc_with_table(row, tables=5))

    assert [table.df.shape for table in contents.tables] == [(0, 1000)] * 2
    (warning,) = contents.processing_warnings
    assert warning.message.startswith("Dropped 3 table(s)")


def test_merged_cell_text_counts_in_every_grid_cell_it_fills():
    """The HTML and the export write a merged cell's text into each grid cell
    it covers: 10 KB of text spanning 1,000 columns and merged down 7 rows (an
    11 KB document.xml) made 70 MB of table HTML."""
    restart = _tc("x" * 10_000, _span(1000) + '<w:vMerge w:val="restart"/>')
    cont = _tc("", _span(1000) + "<w:vMerge/>")
    rows = f"<w:tr>{restart}</w:tr>" + f"<w:tr>{cont}</w:tr>" * 6
    _, contents = _parse(_doc_with_table(rows))

    assert contents.tables == []
    assert _codes(contents) == [WarningCode.DOCX_TABLE_DROPPED]


def test_tables_past_the_document_text_limit_are_dropped(monkeypatch):
    monkeypatch.setattr(docx_native, "_MAX_DOCUMENT_TABLE_CHARS", 250)
    # 3 rows of 10 four-character cells: 120 characters a table.
    row = "<w:tr>" + _tc("cell") * 10 + "</w:tr>"
    _, contents = _parse(_doc_with_table(row * 3, tables=3))

    assert len(contents.tables) == 2
    (warning,) = contents.processing_warnings
    assert warning.code == WarningCode.DOCX_TABLE_DROPPED
    assert warning.message.startswith("Dropped 1 table(s)")


def test_a_continuation_with_nothing_above_reads_as_its_own_cell():
    """python-docx raised ValueError, failing the whole parse."""
    rows = f"<w:tr>{_tc('Top', '<w:vMerge/>')}{_tc('B')}</w:tr><w:tr>{_tc('1')}{_tc('2')}</w:tr>"
    _, contents = _parse(_doc_with_table(rows))

    assert contents.tables[0].df.columns.tolist() == ["Top", "B"]
    assert contents.tables[0].df.iloc[0].tolist() == ["1", "2"]


def test_a_malformed_grid_span_reads_as_one_column():
    """python-docx raised ValueError on the non-numeric value."""
    rows = f"<w:tr>{_tc('A', _span('wide'))}{_tc('B')}</w:tr>"
    _, contents = _parse(_doc_with_table(rows))

    assert contents.tables[0].df.columns.tolist() == ["A", "B"]


def test_merged_tables_read_as_python_docx_reads_them():
    """Well-formed tables, merged cells and short rows included, keep the
    cells python-docx's ``row.cells`` gave them."""
    doc = Document()
    table = doc.add_table(rows=5, cols=5)
    for r in range(5):
        for c in range(5):
            table.cell(r, c).text = f"r{r}c{c}"
    table.cell(0, 0).merge(table.cell(0, 2))
    table.cell(1, 1).merge(table.cell(3, 2))
    table.cell(2, 4).merge(table.cell(4, 4))
    table.cell(4, 0).merge(table.cell(4, 1))
    table.cell(3, 0).text = "two\nlines"
    # A row that starts one grid column late.
    last = table._tbl.tr_lst[-1]
    last.remove(last.tc_lst[0])
    grid_before = etree.SubElement(last.get_or_add_trPr(), qn("w:gridBefore"))
    grid_before.set(qn("w:val"), "2")
    expected = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    width = max(len(row) for row in expected)
    expected = [row + [""] * (width - len(row)) for row in expected]

    _, contents = _parse(_save(doc))

    (parsed,) = contents.tables
    assert [parsed.df.columns.tolist(), *parsed.df.to_numpy().tolist()] == expected


# ----- Figures -----


def _png(side: int) -> bytes:
    raw = b"".join(b"\x00" + os.urandom(side * 3) for _ in range(side))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 0))
        + chunk(b"IEND", b"")
    )


def _doc_with_pictures(image: bytes, count: int) -> bytes:
    """*count* pictures, each in its own paragraph, all showing one image part."""
    doc = Document()
    doc.add_heading("Results", 1)
    paragraph = doc.add_paragraph()
    paragraph.add_run().add_picture(io.BytesIO(image))
    for _ in range(count - 1):
        paragraph._p.addnext(copy.deepcopy(paragraph._p))
    return _save(doc)


def test_an_image_shown_many_times_is_encoded_once():
    _, contents = _parse(_doc_with_pictures(_PNG_BYTES, 5))

    assert len(contents.figures) == 5
    first = contents.figures[0].image_b64
    assert base64.b64decode(first) == _PNG_BYTES
    assert all(f.image_b64 is first and f.parts[0].image_b64 is first for f in contents.figures)
    assert contents.processing_warnings == []


def test_figure_image_data_is_capped_per_document():
    """A 1 MiB image shown 140 times: 140 MiB of image data, counted once
    per figure, from a 1 MB file."""
    image = _png(600)
    _, contents = _parse(_doc_with_pictures(image, 140))

    with_image = [f for f in contents.figures if f.image_b64 is not None]
    assert len(contents.figures) == 140
    assert len(with_image) == docx_native._MAX_FIGURE_IMAGE_BYTES // len(image)
    assert all(f.parts[0].image_b64 is None for f in contents.figures[len(with_image) :])
    (warning,) = contents.processing_warnings
    assert warning.code == WarningCode.DOCX_FIGURE_IMAGES_OMITTED
    assert warning.message.startswith(f"Kept {140 - len(with_image)} figure(s) without")


def test_figures_past_the_limit_are_dropped(monkeypatch):
    monkeypatch.setattr(docx_native, "_MAX_FIGURES", 3)
    _, contents = _parse(_doc_with_pictures(_PNG_BYTES, 5))

    assert [f.figure_id for f in contents.figures] == [1, 2, 3]
    (warning,) = contents.processing_warnings
    assert warning.code == WarningCode.DOCX_FIGURES_DROPPED
    assert warning.message.startswith("Dropped 2 picture(s)")


# ----- Inline wrappers -----


def _append(paragraph, fragment: str) -> None:
    root = etree.fromstring(
        f'<w:root xmlns:w="{W}" xmlns:m="{M}" xmlns:r="{R}">{fragment}</w:root>'
    )
    for element in root:
        paragraph._p.append(element)


def _paragraph_with(fragment: str) -> list[str]:
    doc = Document()
    _append(doc.add_paragraph("Prior work "), fragment)
    parser, _ = _parse(_save(doc))
    return [e.text for e in parser.assembler.entries]


def test_inline_wrappers_keep_their_text():
    fragment = (
        "<w:sdt><w:sdtPr><w:citation/></w:sdtPr>"
        "<w:sdtContent><w:r><w:t>(Smith, 2020)</w:t></w:r></w:sdtContent></w:sdt>"
        '<w:r><w:t xml:space="preserve"> showed </w:t></w:r>'
        '<w:fldSimple w:instr=" SEQ Figure "><w:r><w:t>Figure 1</w:t></w:r></w:fldSimple>'
        '<w:r><w:t xml:space="preserve"> in </w:t></w:r>'
        '<w:smartTag w:uri="x" w:element="place"><w:r><w:t>Boston</w:t></w:r></w:smartTag>'
        '<w:r><w:t xml:space="preserve"> and </w:t></w:r>'
        '<w:customXml w:element="loc"><w:r><w:t>Paris</w:t></w:r></w:customXml>'
        '<w:moveTo w:id="1"><w:r><w:t xml:space="preserve"> lately</w:t></w:r></w:moveTo>'
        '<w:r><w:t xml:space="preserve">.</w:t></w:r>'
    )
    assert _paragraph_with(fragment) == [
        "Prior work (Smith, 2020) showed Figure 1 in Boston and Paris lately."
    ]


def test_nested_inline_wrappers_keep_their_text():
    fragment = (
        "<w:ins w:id='1'><w:sdt><w:sdtContent>"
        '<w:fldSimple w:instr=" CITATION "><w:r><w:t>[1]</w:t></w:r></w:fldSimple>'
        "</w:sdtContent></w:sdt></w:ins>"
        '<w:customXml w:element="a"><w:smartTag w:element="b">'
        '<w:r><w:t xml:space="preserve"> and [2]</w:t></w:r></w:smartTag></w:customXml>'
    )
    assert _paragraph_with(fragment) == ["Prior work [1] and [2]"]


def test_deleted_and_moved_from_text_stays_out():
    fragment = (
        "<w:sdt><w:sdtContent>"
        '<w:del w:id="1"><w:r><w:delText>gone</w:delText></w:r></w:del>'
        '<w:moveFrom w:id="2"><w:r><w:t>moved away</w:t></w:r></w:moveFrom>'
        "<w:r><w:t>kept</w:t></w:r>"
        "</w:sdtContent></w:sdt>"
        '<w:moveFrom w:id="3"><w:r><w:t> moved away</w:t></w:r></w:moveFrom>'
    )
    assert _paragraph_with(fragment) == ["Prior work kept"]


def test_hyperlink_text_inside_a_wrapper_counts_toward_its_link_text():
    doc = Document()
    paragraph = doc.add_paragraph("See ")
    rel_id = paragraph.part.relate_to(
        "https://example.org/data",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    paragraph._p.append(
        etree.fromstring(
            f'<w:hyperlink xmlns:w="{W}" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            f'r:id="{rel_id}"><w:smartTag w:element="x"><w:r><w:t>the data</w:t></w:r>'
            "</w:smartTag></w:hyperlink>"
        )
    )
    parser, contents = _parse(_save(doc))
    parser.apply_segmentation(contents, [["See the data"]])

    assert [e.text for e in parser.assembler.entries] == ["See the data"]
    assert [(link.url, link.link_text) for link in contents.links] == [
        ("https://example.org/data", "the data")
    ]


def test_math_inside_inline_wrappers_is_kept():
    omath = "<m:oMath><m:r><m:t>x=1</m:t></m:r></m:oMath>"
    fragment = (
        f"<w:ins w:id='1'>{omath}</w:ins>"
        '<w:r><w:t xml:space="preserve"> and </w:t></w:r>'
        f"<w:sdt><w:sdtContent><w:hyperlink w:anchor='a'>{omath}</w:hyperlink>"
        "</w:sdtContent></w:sdt>"
        f"<w:sdt><w:sdtContent><m:oMathPara>{omath}</m:oMathPara></w:sdtContent></w:sdt>"
    )
    assert _paragraph_with(fragment) == ["Prior work $x=1$ and $x=1$", "$$x=1$$"]


def test_a_hyperlink_inside_a_wrapper_keeps_its_url():
    """Reference managers put a citation, link included, in a content control."""
    doc = Document()
    paragraph = doc.add_paragraph("See ")
    rel_id = paragraph.part.relate_to("https://example.org/data", HYPERLINK, is_external=True)
    _append(
        paragraph,
        f'<w:sdt><w:sdtContent><w:hyperlink r:id="{rel_id}"><w:r><w:t>the data</w:t></w:r>'
        "</w:hyperlink></w:sdtContent></w:sdt>",
    )
    parser, contents = _parse(_save(doc))
    parser.apply_segmentation(contents, [["See the data"]])

    assert [(link.url, link.link_text) for link in contents.links] == [
        ("https://example.org/data", "the data")
    ]


def test_heading_text_inside_a_wrapper_is_kept():
    """A title in a content control (Word's Title quick part) was lost, with
    its section."""
    doc = Document()
    _append(
        doc.add_paragraph(style="Title"),
        "<w:sdt><w:sdtPr><w:alias w:val='Title'/></w:sdtPr><w:sdtContent>"
        "<w:r><w:t>Cognitive load and recall</w:t></w:r></w:sdtContent></w:sdt>",
    )
    doc.add_paragraph("Body.")
    _, contents = _parse(_save(doc))

    assert contents.detected_title == "Cognitive load and recall"
    assert "Cognitive load and recall" in [section.header for section in contents.sections]


def test_caption_number_in_a_simple_field_is_kept():
    """A caption numbered by a SEQ field, "Table 1: Results", read "Table :
    Results", without a label."""
    doc = Document()
    _append(
        doc.add_paragraph("Table ", style="Caption"),
        '<w:fldSimple w:instr=" SEQ Table \\* ARABIC "><w:r><w:t>1</w:t></w:r></w:fldSimple>'
        "<w:r><w:t>: Results</w:t></w:r>",
    )
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "a"
    _, contents = _parse(_save(doc))

    assert [(table.caption, table.label) for table in contents.tables] == [
        ("Table 1: Results", "1")
    ]


def test_cell_text_inside_a_wrapper_is_kept():
    doc = Document()
    table = doc.add_table(rows=2, cols=1)
    table.cell(0, 0).text = "Head"
    _append(
        table.cell(1, 0).paragraphs[0],
        "<w:ins w:id='9'><w:r><w:t>inserted</w:t></w:r></w:ins>",
    )
    _, contents = _parse(_save(doc))

    assert contents.tables[0].df.to_numpy().tolist() == [["inserted"]]
