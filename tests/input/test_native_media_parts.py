"""Natively parsed media objects export as whole-object rows (v12)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest

from bibr.export import export_paper_to_json
from bibr.input.docx_native import DocxParser
from bibr.input.file import InputFile, InputFormat
from bibr.input.html_native import HtmlParser
from bibr.input.jats_native import JatsParser
from bibr.models import PaperMetadata
from bibr.paper import Paper

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _export(contents, *, suffix: str, mime_type: str, file_type: str) -> dict:
    paper = Paper(
        input_file=InputFile(
            path=Path(f"/tmp/native{suffix}"),
            file_hash="native-hash",
            input_format=InputFormat(
                file_extension=suffix,
                detected_mime_type=mime_type,
                file_type=file_type,
            ),
        ),
        metadata=PaperMetadata(title="Native media", doi=""),
        contents=contents,
    )
    return export_paper_to_json(paper, validate=False)


def _assert_single_parts_match_legacy_payload(contents, output: dict) -> None:
    """Natively parsed media keep one internal part, but a part with no page or
    box says nothing the whole-object row does not: v12 exports no ``parts``
    on the rows and no ``extraction.float_parts`` entry for it."""
    source_figure = contents.figures[0]
    source_table = contents.tables[0]
    figure = output["figure"][0]
    table = output["table"][0]

    assert len(source_figure.parts) == 1
    assert source_figure.parts[0].image_b64 == source_figure.image_b64
    assert source_figure.parts[0].page_number == source_figure.page_number
    assert source_figure.parts[0].provenance == source_figure.provenance
    # Exported as a data URI that names the embedded image's media type.
    assert figure["image"] == (
        "data:image/png;base64," + source_figure.image_b64 if source_figure.image_b64 else None
    )
    assert "parts" not in figure

    assert len(source_table.parts) == 1
    assert source_table.parts[0].tbl_html == source_table.tbl_html
    assert source_table.parts[0].df.equals(source_table.df)
    assert source_table.parts[0].page_number == source_table.page_number
    assert source_table.parts[0].provenance == source_table.provenance
    assert table["html"] == source_table.tbl_html
    assert "parts" not in table

    located = [
        part
        for part in (*source_figure.parts, *source_table.parts)
        if part.page_number is not None or part.bbox
    ]
    assert len(output["extraction"].get("float_parts") or []) == len(located)


def test_docx_native_export_has_one_part_per_media_object():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Results", level=1)
    document.add_picture(io.BytesIO(_PNG_BYTES))
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = "B"
    table.cell(1, 0).text = "1"
    table.cell(1, 1).text = "2"
    buffer = io.BytesIO()
    document.save(buffer)

    contents = DocxParser(buffer.getvalue()).parse()
    output = _export(
        contents,
        suffix=".docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        file_type="docx",
    )

    _assert_single_parts_match_legacy_payload(contents, output)


def test_html_native_export_has_one_part_per_media_object():
    contents = HtmlParser(
        b"""
        <html><body><article><h1>Results</h1>
          <figure><img alt="Plot"><figcaption>Figure 1. Plot.</figcaption></figure>
          <table><tr><th>A</th></tr><tr><td>1</td></tr></table>
        </article></body></html>
        """
    ).parse()
    output = _export(
        contents,
        suffix=".html",
        mime_type="text/html",
        file_type="html",
    )

    _assert_single_parts_match_legacy_payload(contents, output)


def test_jats_native_export_has_one_part_per_media_object():
    contents = JatsParser(
        b"""
        <article><front><article-meta><title-group><article-title>Native</article-title>
        </title-group></article-meta></front><body><sec><title>Results</title>
          <fig><label>Figure 1</label><caption><p>Plot.</p></caption><graphic/></fig>
          <table-wrap><label>Table 1</label><caption><p>Values.</p></caption>
            <table><tr><th>A</th></tr><tr><td>1</td></tr></table>
          </table-wrap>
        </sec></body></article>
        """
    ).parse()
    output = _export(
        contents,
        suffix=".xml",
        mime_type="application/xml",
        file_type="xml",
    )

    _assert_single_parts_match_legacy_payload(contents, output)
