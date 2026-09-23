"""ExportStage — serialize Paper to JSON, free state."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.export import ExportStage, _build_engines
from bibr.pipeline.state import FileState


def _ctx(file_states, rm=None, config=None):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=rm or MagicMock(),
        config=config or RunConfig(),
    )


@pytest.mark.asyncio
async def test_populates_result_json():
    fs = FileState(path=Path("x.pdf"))
    fs.paper = MagicMock()
    fs.paper.export_to_json.return_value = {"ok": True}

    await ExportStage().run(_ctx([fs]))

    assert fs.result_json == {"ok": True}
    assert fs.contents is None  # free_all called


@pytest.mark.asyncio
async def test_sets_error_on_export_failure():
    fs = FileState(path=Path("x.pdf"))
    fs.paper = MagicMock()
    fs.paper.export_to_json.side_effect = RuntimeError("boom")

    await ExportStage().run(_ctx([fs]))

    assert fs.error is not None
    assert fs.error_code == "export_failed"


@pytest.mark.asyncio
async def test_does_not_shut_down_llm_server_itself():
    """LLM server shutdown is handled by LocalPipeline.process_chunk, not ExportStage."""
    fs = FileState(path=Path("x.pdf"))
    fs.paper = MagicMock()
    fs.paper.export_to_json.return_value = {}
    rm = MagicMock()

    await ExportStage().run(_ctx([fs], rm=rm))

    rm.shutdown_llm_server.assert_not_called()


@pytest.mark.asyncio
async def test_gc_collect_throttled(monkeypatch):
    """gc.collect() blocks the event loop (GIL held for the whole pass), so
    back-to-back chunks — every serve request is one — must not each pay it."""
    import bibr.pipeline.stages.export as mod

    calls = []
    monkeypatch.setattr(mod.gc, "collect", lambda: calls.append(1))

    stage = ExportStage()
    for _ in range(5):
        fs = FileState(path=Path("x.pdf"))
        fs.paper = MagicMock()
        fs.paper.export_to_json.return_value = {}
        await stage.run(_ctx([fs]))

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_gc_throttle_is_per_instance(monkeypatch):
    """Throttle state lives on the stage instance, not module-level globals."""
    import bibr.pipeline.stages.export as mod

    calls = []
    monkeypatch.setattr(mod.gc, "collect", lambda: calls.append(1))

    for _ in range(2):
        fs = FileState(path=Path("x.pdf"))
        fs.paper = MagicMock()
        fs.paper.export_to_json.return_value = {}
        await ExportStage().run(_ctx([fs]))

    assert len(calls) == 2


def test_ocr_engine_reports_backend_specific_rapid_mlx_model(monkeypatch):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.ocr, "model", "numind/NuExtract3-mlx-8bits")
    monkeypatch.setattr(Settings.ocr, "rapid_mlx_model", "mlx-community/GLM-OCR-8bit")
    ctx = _ctx([], config=RunConfig(ocr_backend="glm-rapid-mlx", llm_backend="rapid-mlx"))

    assert _build_engines(ctx)[0]["model"] == "mlx-community/GLM-OCR-8bit"


def test_ocr_engine_reports_default_sglang_glm_model(monkeypatch):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.ocr, "model", None)
    monkeypatch.setattr(Settings.ocr, "local_model", "zai-org/GLM-OCR")
    # glm-http talks to an externally-managed server, so provenance must record
    # the alias that was actually requested over the wire — not
    # ``ocr.local_model``, which is the HuggingFace repo a *local* runtime would
    # load weights from. Falling through to the repo id also made the readiness
    # gate poll for a model id the server never advertises.
    ctx = _ctx([], config=RunConfig(ocr_backend="glm-http"))

    assert _build_engines(ctx)[0]["model"] == "glm-ocr"


def test_ocr_engine_uses_concrete_runtime_identity_from_scratch():
    from bibr.ocr.profiles import OcrRuntimeIdentity

    ctx = _ctx([], config=RunConfig(ocr_backend="serve-http"))
    ctx.scratch["ocr_runtime_identity"] = OcrRuntimeIdentity(
        backend="serve-http",
        model="paddle-ocr-vl-1.6",
        profile="paddle",
        normalizer_version="paddle-canonical-v1",
    )

    config = _build_engines(ctx)[0]

    assert config == {
        "backend": "serve-http",
        "model": "paddle-ocr-vl-1.6",
        "profile": "paddle",
    }


def test_ocr_engine_resolves_concrete_identity_without_ocr_startup(monkeypatch):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.ocr, "paddle_served_model", "paddle-ocr-vl-1.6")
    ctx = _ctx([], config=RunConfig(ocr_backend="serve-http", ocr_profile="paddle"))

    config = _build_engines(ctx)[0]

    assert config == {
        "backend": "serve-http",
        "model": "paddle-ocr-vl-1.6",
        "profile": "paddle",
    }


def test_ocr_engine_serve_http_default_profile_follows_served_model(monkeypatch):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.ocr, "model", None)
    monkeypatch.setattr(Settings.ocr, "profile", None)
    ctx = _ctx([], config=RunConfig(ocr_backend="serve-http"))

    config = _build_engines(ctx)[0]

    assert config == {
        "backend": "serve-http",
        "model": "glm-ocr",
        "profile": "glm",
    }
