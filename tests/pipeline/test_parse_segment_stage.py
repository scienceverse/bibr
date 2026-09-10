"""ParseSegmentStage — PDFParser + wtpsplit segmentation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.parse_segment import ParseSegmentStage
from bibr.pipeline.state import FileState


def _ctx(file_states, rm):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(),
    )


@pytest.mark.asyncio
async def test_pdf_path_calls_parser_then_segmenter():
    fs = FileState(path=Path("x.pdf"))
    ocr_regions = [[{"content": "hello world.", "task_type": "text"}]]
    fs.ocr_regions = ocr_regions
    contents = MagicMock()
    parser = MagicMock()
    parser.parse.return_value = contents
    parser._deferred_texts = []
    rm = MagicMock()
    rm.segmenter = MagicMock(segment_batch=MagicMock(return_value=[]))
    ctx = _ctx([fs], rm)

    with patch("bibr.structure.pdf_parser.PDFParser", return_value=parser) as PP:
        await ParseSegmentStage().run(ctx)

    PP.assert_called_once_with(ocr_regions, outline=None, settings=ctx.settings, first_page_index=0)
    parser.create_content_sections.assert_called_once_with(contents)
    assert fs.contents is contents


@pytest.mark.asyncio
async def test_native_docx_path_reuses_attached_parser():
    fs = FileState(path=Path("x.docx"))
    parser = MagicMock()
    parser._deferred_texts = []
    fs._native_parser = parser
    fs.contents = MagicMock()
    rm = MagicMock()
    rm.segmenter = MagicMock(segment_batch=MagicMock(return_value=[]))

    with patch("bibr.structure.pdf_parser.PDFParser") as PP:
        await ParseSegmentStage().run(_ctx([fs], rm))

    PP.assert_not_called()
    parser.create_content_sections.assert_called_once_with(fs.contents)


@pytest.mark.asyncio
async def test_parser_failure_sets_error():
    fs = FileState(path=Path("x.pdf"))
    fs.ocr_regions = [[]]
    rm = MagicMock()

    with patch("bibr.structure.pdf_parser.PDFParser", side_effect=ValueError("boom")):
        await ParseSegmentStage().run(_ctx([fs], rm))

    assert fs.error is not None
    assert fs.error_code == "parse_failed"
    assert fs.failed_stage == "parse"
