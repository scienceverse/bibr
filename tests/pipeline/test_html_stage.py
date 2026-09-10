"""HtmlHandlingStage — native HTML/ePub parse."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.html import HtmlHandlingStage
from bibr.pipeline.state import FileState


def _ctx(file_states):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(),
    )


@pytest.mark.asyncio
async def test_pdf_docx_and_xml_files_are_untouched():
    for suffix in (".pdf", ".docx", ".xml"):
        fs = FileState(path=Path(f"x{suffix}"))
        fs.pdf_bytes = b"bytes"
        ctx = _ctx([fs])
        await HtmlHandlingStage().run(ctx)
        assert fs.contents is None
        assert fs.error is None


@pytest.mark.asyncio
async def test_native_html_parses_directly():
    fs = FileState(path=Path("x.html"))
    fs.pdf_bytes = b"<html><body><p>Text.</p></body></html>"
    contents = MagicMock()
    parser = MagicMock()
    parser.parse.return_value = contents
    with patch("bibr.input.html_native.HtmlParser", return_value=parser) as html_parser:
        await HtmlHandlingStage().run(_ctx([fs]))
    html_parser.assert_called_once_with(b"<html><body><p>Text.</p></body></html>")
    assert fs.contents is contents
    assert fs._native_parser is parser
    assert fs.error is None


@pytest.mark.asyncio
async def test_native_html_reuses_validation_artifact():
    fs = FileState(path=Path("x.html"))
    fs.pdf_bytes = b"<html><body><p>Text.</p></body></html>"
    artifact = object()
    fs.native_validation_artifact = artifact
    parser = MagicMock()
    with patch("bibr.input.html_native.HtmlParser", return_value=parser) as html_parser:
        await HtmlHandlingStage().run(_ctx([fs]))
    html_parser.assert_called_once_with(fs.pdf_bytes, parsed_soup=artifact)
    assert fs.native_validation_artifact is None


@pytest.mark.asyncio
async def test_native_html_parse_runs_off_event_loop_thread():
    main_thread = threading.get_ident()
    called_on = None
    parser = MagicMock()

    def construct(data):
        nonlocal called_on
        called_on = threading.get_ident()
        return parser

    fs = FileState(path=Path("x.html"))
    fs.pdf_bytes = b"<html/>"
    with patch("bibr.input.html_native.HtmlParser", side_effect=construct):
        await HtmlHandlingStage().run(_ctx([fs]))

    assert called_on != main_thread


@pytest.mark.asyncio
async def test_native_epub_parses_directly():
    fs = FileState(path=Path("x.epub"))
    fs.pdf_bytes = b"epub"
    contents = MagicMock()
    parser = MagicMock()
    parser.parse.return_value = contents
    with patch("bibr.input.epub_native.EpubParser", return_value=parser) as epub_parser:
        await HtmlHandlingStage().run(_ctx([fs]))
    epub_parser.assert_called_once_with(b"epub")
    assert fs.contents is contents
    assert fs._native_parser is parser
    assert fs.error is None


@pytest.mark.asyncio
async def test_native_epub_reuses_validation_artifact():
    fs = FileState(path=Path("x.epub"))
    fs.pdf_bytes = b"epub"
    artifact = object()
    fs.native_validation_artifact = artifact
    parser = MagicMock()
    with patch("bibr.input.epub_native.EpubParser", return_value=parser) as epub_parser:
        await HtmlHandlingStage().run(_ctx([fs]))
    epub_parser.assert_called_once_with(fs.pdf_bytes, document=artifact)
    assert fs.native_validation_artifact is None


@pytest.mark.asyncio
async def test_native_parse_failure_sets_error():
    fs = FileState(path=Path("x.htm"))
    fs.pdf_bytes = b"bad"
    with patch("bibr.input.html_native.HtmlParser", side_effect=ValueError("boom")):
        await HtmlHandlingStage().run(_ctx([fs]))
    assert fs.error is not None
    assert fs.error_code == "parse_failed"
    assert fs.failed_stage == "html"
