"""DocxHandlingStage — native-DOCX parse."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.docx import DocxHandlingStage
from bibr.pipeline.state import FileState


def _ctx(file_states):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(),
    )


@pytest.mark.asyncio
async def test_pdf_files_are_untouched():
    fs = FileState(path=Path("x.pdf"))
    fs.pdf_bytes = b"%PDF"
    ctx = _ctx([fs])
    await DocxHandlingStage().run(ctx)
    assert fs.contents is None
    assert fs.pdf_bytes == b"%PDF"
    assert fs.error is None


@pytest.mark.asyncio
async def test_native_docx_parses_directly():
    fs = FileState(path=Path("x.docx"))
    fs.pdf_bytes = b"docx-bytes"
    contents = MagicMock()
    parser = MagicMock()
    parser.parse.return_value = contents
    with patch("bibr.input.docx_native.DocxParser", return_value=parser) as DP:
        await DocxHandlingStage().run(_ctx([fs]))
    DP.assert_called_once_with(b"docx-bytes")
    assert fs.contents is contents
    assert fs._native_parser is parser
    assert fs.error is None


@pytest.mark.asyncio
async def test_native_docx_parse_runs_off_event_loop_thread():
    main_thread = threading.get_ident()
    called_on = None
    parser = MagicMock()

    def construct(data):
        nonlocal called_on
        called_on = threading.get_ident()
        return parser

    fs = FileState(path=Path("x.docx"))
    fs.pdf_bytes = b"docx"
    with patch("bibr.input.docx_native.DocxParser", side_effect=construct):
        await DocxHandlingStage().run(_ctx([fs]))

    assert called_on != main_thread


@pytest.mark.asyncio
async def test_native_parse_failure_sets_error():
    fs = FileState(path=Path("x.docx"))
    fs.pdf_bytes = b"bad"
    with patch("bibr.input.docx_native.DocxParser", side_effect=ValueError("boom")):
        await DocxHandlingStage().run(_ctx([fs]))
    assert fs.error is not None
    assert fs.error_code == "docx_parse_failed"
    assert fs.failed_stage == "docx"
