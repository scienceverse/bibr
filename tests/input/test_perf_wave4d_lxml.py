"""Wave 4D tests: opt-in lxml HTML parser option and direct lxml.html table fast path."""

from __future__ import annotations

import codecs
import io
import zipfile
from pathlib import Path

import pytest

from bibr.config import GlobalSettings, PipelineOptions
from bibr.input.epub_native import EpubParser
from bibr.input.html_native import HtmlParser, inspect_html
from bibr.input.validate import validate_input_file
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.html import HtmlHandlingStage
from bibr.pipeline.stages.validate import ValidateStage
from bibr.pipeline.state import FileState
from bibr.structure.html_table import TableBudget, html_table_frame


def test_html_parser_setting_defaults_to_html5lib_and_accepts_lxml(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_HTML_PARSER", raising=False)
    assert PipelineOptions().html_parser == "html5lib"

    monkeypatch.setenv("PIPELINE_HTML_PARSER", "lxml")
    assert PipelineOptions().html_parser == "lxml"


def test_invalid_html_parser_name_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported html_parser"):
        html_table_frame("<table><tr><td>1</td></tr></table>", parser="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported html_parser"):
        inspect_html(b"<html><body><p>hi</p></body></html>", parser="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "html",
    [
        # Multi-level header with rowspan/colspan and printed numbers.
        '<table><tr><th rowspan="2">Var</th><th colspan="2">Group A</th></tr>'
        "<tr><th>M</th><th>SD</th></tr><tr><td>Age</td><td>007</td><td>1,234</td></tr></table>",
        # Explicit thead/tbody/tfoot with rowspan crossing into tfoot.
        "<table><thead><tr><th>A</th><th>B</th></tr></thead>"
        '<tbody><tr><td rowspan="2">a</td><td>2.50</td></tr></tbody>'
        "<tfoot><tr><td>f</td></tr></tfoot></table>",
        # Hidden style/span with trailing text (.tail must survive!), <br>, comments.
        "<table><tr><th>a<br>b</th><th>c  d\ne</th><th style='display: none'>h</th></tr>"
        "<tr><td>x<style>.s{}</style>y<!-- note -->z</td>"
        "<td><span style='display:none'>skip</span>kept</td></tr></table>",
        # Bare <th> without enclosing <tr> (malformed OCR table markup).
        "<table><th>H1<th>H2<tr><td>1<td>2</table>",
        # Hidden / empty first tables skipped.
        "<table style='display:none'><tr><td>h</td></tr></table>"
        "<table><tr><td></td></tr></table>"
        "<table><tr><th>T</th></tr><tr><td>v</td></tr></table>",
        # Nested table inside a cell.
        "<table><tr><th>A</th><th>B</th></tr>"
        "<tr><td>outer<table><tr><td>inner</td></tr></table></td><td>right</td></tr></table>",
        # colspan=0 and ragged rows.
        '<table><tr><td colspan="0">a</td><td>b</td></tr><tr><td>c</td></tr></table>',
        # Non-table / empty / comment-only strings.
        "",
        "   ",
        "<!-- comment only -->",
        "<!doctype html>",
        "<p>no table</p>",
        "<table><caption>Only a caption</caption></table>",
    ],
)
def test_lxml_html_table_frame_matches_html5lib(html: str) -> None:
    df_h5 = html_table_frame(html, parser="html5lib")
    df_lx = html_table_frame(html, parser="lxml")

    if df_h5 is None:
        assert df_lx is None
    else:
        assert df_lx is not None
        assert list(df_lx.columns) == list(df_h5.columns)
        assert df_lx.values.tolist() == df_h5.values.tolist()


def test_lxml_html_table_frame_enforces_span_and_budget_bounds() -> None:
    bomb = '<table><tr><td colspan="1000" rowspan="4000">x</td></tr></table>'
    df_h5 = html_table_frame(bomb, parser="html5lib")
    df_lx = html_table_frame(bomb, parser="lxml")
    assert df_h5 is not None and df_lx is not None
    assert df_lx.shape == df_h5.shape == (1, 22)

    padding = (
        "<table><tr>" + "<td>w</td>" * 300 + "</tr>" + "<tr><td>n</td></tr>" * 300 + "</table>"
    )
    budget_lx = TableBudget()
    assert html_table_frame(padding, budget_lx, parser="lxml") is None
    assert budget_lx.refused == 1


def test_html_parser_and_epub_parser_lxml_match_html5lib() -> None:
    html_text = (
        "<!doctype html><html lang='de'><head><meta charset='utf-8'>"
        "<meta name='citation_title' content='Der α-Helix Effekt'>"
        "<meta name='citation_author' content='Müller, José'>"
        "</head><body><article><h1>Der α-Helix Effekt</h1>"
        "<h2>Methods</h2><p>Dissolved H<sub>2</sub>S and m<sup>6</sup>A "
        "(<a href='#f1'>Figure 1</a>) at 37°C<!-- note -->.<br>Next line.</p>"
        "<table><caption>Table 1. Values.</caption>"
        "<thead><tr><th>Metric</th><th>Val</th></tr></thead>"
        "<tbody><tr><td>N</td><td>007</td></tr><tr><td>M</td><td>1,5</td></tr></tbody></table>"
        "<h2>References</h2><ol><li>Doe J. (2024). Study.</li></ol>"
        "</article></body></html>"
    )
    for raw in (html_text.encode("utf-8"), codecs.BOM_UTF16_LE + html_text.encode("utf-16-le")):
        ok_h5, soup_h5 = inspect_html(raw, parser="html5lib")
        ok_lx, soup_lx = inspect_html(raw, parser="lxml")
        assert ok_h5 is True and ok_lx is True
        p_h5 = HtmlParser(raw, parsed_soup=soup_h5, parser="html5lib")
        p_lx = HtmlParser(raw, parsed_soup=soup_lx, parser="lxml")
        c_h5, c_lx = p_h5.parse(), p_lx.parse()

        assert [e.text for e in p_lx.assembler.entries] == [e.text for e in p_h5.assembler.entries]
        assert c_lx.preparsed_metadata == c_h5.preparsed_metadata
        assert c_lx.native_ref_strings == c_h5.native_ref_strings
        assert len(c_lx.tables) == len(c_h5.tables) == 1
        assert c_lx.tables[0].df.values.tolist() == c_h5.tables[0].df.values.tolist()

    # ePub parity with html_parser="lxml"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            b'<?xml version="1.0"?><container version="1.0" '
            b'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            b'<rootfiles><rootfile full-path="OEBPS/content.opf" '
            b'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        zf.writestr(
            "OEBPS/content.opf",
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            b'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            b"<dc:title>Book Title</dc:title></metadata>"
            b'<manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>'
            b'<spine><itemref idref="c1"/></spine></package>',
        )
        zf.writestr(
            "OEBPS/ch1.xhtml",
            b'<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Ch 1</h1>'
            b"<p>Temperature 37\xc2\xb0C in M\xc3\xbcnchen.</p></body></html>",
        )
    epub_bytes = buf.getvalue()
    ep_h5 = EpubParser(epub_bytes, html_parser="html5lib")
    ep_lx = EpubParser(epub_bytes, html_parser="lxml")
    ep_h5.parse()
    ep_lx.parse()
    assert ep_lx._deferred_texts == ep_h5._deferred_texts


@pytest.mark.asyncio
async def test_validate_and_html_stages_wire_pipeline_html_parser(monkeypatch) -> None:
    from bibr.input import html_native

    seen_parsers: list[str] = []
    real_parse_soup = html_native._parse_soup

    def recording(html_bytes, *, parser="html5lib"):
        seen_parsers.append(parser)
        return real_parse_soup(html_bytes, parser=parser)

    monkeypatch.setattr(html_native, "_parse_soup", recording)

    settings = GlobalSettings()
    settings.pipeline.html_parser = "lxml"
    fs = FileState(
        path=Path("paper.html"),
        pdf_bytes=b"<!doctype html><html><body><article><h1>T</h1><p>Body text.</p></article></body></html>",
    )
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=object(),
        config=RunConfig(),
        settings=settings,
    )

    await ValidateStage().run(ctx)
    assert fs.error is None
    assert seen_parsers == ["lxml"]

    await HtmlHandlingStage().run(ctx)
    assert fs.error is None
    assert fs.contents is not None
    # Reused DOM from ValidateStage: no second parse.
    assert seen_parsers == ["lxml"]


def test_validate_input_file_accepts_html_parser_kwarg() -> None:
    res = validate_input_file(
        "paper.html",
        b"<html><body><article><h1>T</h1><p>Hello world.</p></article></body></html>",
        html_parser="lxml",
    )
    assert res.is_valid
    assert res.native_artifact is not None


def test_parse_media_forwards_lxml_html_parser_setting(monkeypatch) -> None:
    from bibr.structure import parse_media
    from bibr.structure.pdf_parser import PDFParser

    seen: list[str] = []
    real_frame = parse_media.html_table_frame

    def recording(source, budget=None, *, parser="html5lib"):
        seen.append(parser)
        return real_frame(source, budget, parser=parser)

    monkeypatch.setattr(parse_media, "html_table_frame", recording)

    settings = GlobalSettings()
    settings.pipeline.html_parser = "lxml"
    parser = PDFParser([], settings=settings)
    parser._handle_table(
        "<table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>",
        page_number=1,
    )
    assert seen == ["lxml"]
    assert len(parser.tables) == 1
    assert parser.tables[0].df.values.tolist() == [["1", "2"]]
