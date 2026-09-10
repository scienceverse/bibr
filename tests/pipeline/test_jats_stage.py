"""JatsHandlingStage — native JATS-XML parse."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.jats import JatsHandlingStage
from bibr.pipeline.stages.validate import ValidateStage
from bibr.pipeline.state import FileState

JATS_FIXTURE = Path(__file__).parents[1] / "fixtures" / "jats" / "PMC4383902.xml"


def _ctx(file_states):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(),
    )


@pytest.mark.asyncio
async def test_pdf_and_docx_files_are_untouched():
    for suffix in (".pdf", ".docx"):
        fs = FileState(path=Path(f"x{suffix}"))
        fs.pdf_bytes = b"bytes"
        ctx = _ctx([fs])
        await JatsHandlingStage().run(ctx)
        assert fs.contents is None
        assert fs.error is None


@pytest.mark.asyncio
async def test_native_xml_parses_directly():
    fs = FileState(path=Path("x.xml"))
    fs.pdf_bytes = b"<article/>"
    contents = MagicMock()
    parser = MagicMock()
    parser.parse.return_value = contents
    with patch("bibr.input.jats_native.JatsParser", return_value=parser) as JP:
        await JatsHandlingStage().run(_ctx([fs]))
    JP.assert_called_once_with(b"<article/>")
    assert fs.contents is contents
    assert fs._native_parser is parser
    assert fs.error is None


@pytest.mark.asyncio
async def test_native_xml_parse_runs_off_event_loop_thread():
    main_thread = threading.get_ident()
    called_on = None
    parser = MagicMock()

    def construct(data):
        nonlocal called_on
        called_on = threading.get_ident()
        return parser

    fs = FileState(path=Path("x.xml"))
    fs.pdf_bytes = b"<article/>"
    with patch("bibr.input.jats_native.JatsParser", side_effect=construct):
        await JatsHandlingStage().run(_ctx([fs]))

    assert called_on != main_thread


@pytest.mark.asyncio
async def test_real_jats_fixture_loads_as_pipeline_input():
    fs = FileState(path=JATS_FIXTURE)
    ctx = _ctx([fs])

    await ValidateStage().run(ctx)
    await JatsHandlingStage().run(ctx)

    assert fs.error is None
    assert fs.contents is not None
    assert fs.contents.preparsed_metadata.title == (
        "Europe PMC: a full-text literature database for the life sciences and platform "
        "for innovation"
    )
    assert fs.contents.preparsed_metadata.doi == "10.1093/nar/gku1061"
    assert len(fs.contents.sections) == 32
    assert len(fs.contents.figures) == 3


@pytest.mark.asyncio
async def test_native_parse_failure_sets_error():
    fs = FileState(path=Path("x.xml"))
    fs.pdf_bytes = b"bad"
    with patch("bibr.input.jats_native.JatsParser", side_effect=ValueError("boom")):
        await JatsHandlingStage().run(_ctx([fs]))
    assert fs.error is not None
    assert fs.error_code == "parse_failed"
    assert fs.failed_stage == "jats"
