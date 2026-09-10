"""ValidateStage — input file validation."""

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.validate import ValidateStage
from bibr.pipeline.state import FileState


def _ctx(file_states):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(),
    )


@pytest.mark.asyncio
async def test_validates_and_populates_pdf_bytes(tmp_path):
    p = tmp_path / "doc.pdf"
    # Minimal valid PDF structure to pass validation (must end with %%EOF)
    pdf_content = b"%PDF-1.4\n%%EOF"
    p.write_bytes(pdf_content)
    fs = FileState(path=p)
    ctx = _ctx([fs])

    await ValidateStage().run(ctx)

    # Valid file populates pdf_bytes and hash, no error. The hash is a real
    # FileState field — not smuggled through the stage_times dict.
    assert fs.pdf_bytes == pdf_content
    import hashlib

    assert fs.file_hash == hashlib.sha256(pdf_content).hexdigest()[:16]
    assert "_file_hash" not in fs.stage_times
    assert fs.error is None


@pytest.mark.asyncio
async def test_sets_error_on_unreadable_path():
    fs = FileState(path=Path("/nonexistent/path.pdf"))
    ctx = _ctx([fs])

    await ValidateStage().run(ctx)

    assert fs.error is not None
    assert fs.failed_stage == "validate"


@pytest.mark.asyncio
async def test_sets_error_on_unsupported_format(tmp_path):
    p = tmp_path / "doc.xyz"
    p.write_bytes(b"not a pdf")
    fs = FileState(path=p)
    ctx = _ctx([fs])

    await ValidateStage().run(ctx)

    assert fs.error is not None
    assert fs.error_code in ("unsupported_format", "corrupted_file", "unknown")
    assert fs.failed_stage == "validate"


def test_stage_name():
    assert ValidateStage().name == "validate"


@pytest.mark.asyncio
async def test_validation_runs_off_event_loop_thread(monkeypatch):
    main_thread = threading.get_ident()
    called_on = None
    valid = MagicMock(is_valid=True)

    def fake_validate(*args, **kwargs):
        nonlocal called_on
        called_on = threading.get_ident()
        return valid

    monkeypatch.setattr("bibr.input.validate.validate_input_file", fake_validate)
    fs = FileState(path=Path("x.pdf"))
    fs.pdf_bytes = b"%PDF"
    await ValidateStage().run(_ctx([fs]))

    assert called_on != main_thread


@pytest.mark.asyncio
async def test_validation_carries_native_parse_artifact(monkeypatch):
    artifact = object()
    valid = MagicMock(is_valid=True, native_artifact=artifact)
    monkeypatch.setattr("bibr.input.validate.validate_input_file", lambda *args, **kwargs: valid)
    fs = FileState(path=Path("x.html"), pdf_bytes=b"<html/>")

    await ValidateStage().run(_ctx([fs]))

    assert fs.native_validation_artifact is artifact


@pytest.mark.asyncio
async def test_validation_reuses_precomputed_sha256(monkeypatch):
    valid = MagicMock(is_valid=True)
    seen = {}

    def fake_validate(name, content, *, file_hash):
        seen["file_hash"] = file_hash
        return valid

    def forbidden_sha256(*args, **kwargs):
        raise AssertionError("validation must not hash bytes again")

    monkeypatch.setattr("bibr.input.validate.validate_input_file", fake_validate)
    monkeypatch.setattr("bibr.pipeline.stages.validate.hashlib.sha256", forbidden_sha256)
    fs = FileState(path=Path("x.pdf"), pdf_bytes=b"%PDF", content_sha256="a" * 64)

    await ValidateStage().run(_ctx([fs]))

    assert fs.file_hash == "a" * 16
    assert seen["file_hash"] == "a" * 16
