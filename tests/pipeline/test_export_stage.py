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
    monkeypatch.setattr(Settings.ocr, "local_model", "THUDM/GLM-OCR")
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


# --- audit pipeline-stages-12: checkpointed (-o) vs unsinked parity -------


def _sinked_run(fs, config):
    """Run ExportStage with a bare-bones context (mirrors test_consolidate)."""
    import asyncio
    from types import SimpleNamespace

    from bibr.config import GlobalSettings

    ctx = SimpleNamespace(
        progress=_NullProgress(),
        config=config,
        settings=GlobalSettings(),
        alive=lambda: [fs],
    )
    asyncio.run(ExportStage().run(ctx))
    return fs.result_json


class _NullProgress:
    def stage_start(self, name):
        pass

    def stage_end(self, name):
        pass


class _CannedPaper:
    """A paper whose export is a fixed payload (no models involved)."""

    def __init__(self, payload):
        import copy

        self._payload = copy.deepcopy(payload)
        self.processing_warnings = []
        self.llm_usage_labels = {}
        self.llm_trace = []
        self.text_quality = None
        self.metadata = None
        self.extraction = None

    def export_to_json(self, *, include_regions=False, include_region_meta=False):
        import copy

        return copy.deepcopy(self._payload)


def _checkpoint_core() -> dict:
    from bibr.pipeline.artifacts import CORE_SCHEMA_VERSION

    return {
        "schema_version": CORE_SCHEMA_VERSION,
        "bib": [{"bib_id": 1, "doi": None}],
        "bib_match": [],
        "extraction": {
            "warnings": [],
            "timings": {"stages": {"parse": 1.0}, "total_seconds": 1.0},
        },
    }


def _boom_on_error(*a, **k):
    raise AssertionError(f"export errored: {a} {k}")


def test_checkpoint_path_consolidates_like_unsinked():
    """With consolidate on and enrichment off, the -o (checkpoint) export must
    carry the same CONSOLIDATE_WITHOUT_ENRICHMENT warning as the unsinked
    export. Fails on base (checkpoint path skips consolidation)."""
    import copy
    from types import SimpleNamespace

    config = RunConfig(consolidate="fill", crossref=False)

    plain_fs = SimpleNamespace(
        paper=_CannedPaper(_checkpoint_core()),
        warnings=[],
        result_json=None,
        path=Path("x.pdf"),
        free_all=lambda: None,
        set_error=_boom_on_error,
    )
    plain_out = _sinked_run(plain_fs, config)

    materialized = {}
    fs = SimpleNamespace(
        paper=_CannedPaper(_checkpoint_core()),
        warnings=[],
        result_json=copy.deepcopy(_checkpoint_core()),
        artifact_sink=SimpleNamespace(materialize=lambda fs, p: materialized.setdefault("p", p)),
        core_sha256="deadbeef",
        enrichment_state=None,
        path=Path("x.pdf"),
        free_all=lambda: None,
        set_error=_boom_on_error,
    )
    sinked_out = _sinked_run(fs, config)

    assert sinked_out["extraction"]["warnings"] == plain_out["extraction"]["warnings"]
    assert any(
        w["code"] == "CONSOLIDATE_WITHOUT_ENRICHMENT" for w in sinked_out["extraction"]["warnings"]
    )
    # The -o file is the materialized payload, so the warning reaches disk.
    assert materialized["p"]["extraction"]["warnings"] == sinked_out["extraction"]["warnings"]


def test_checkpoint_path_does_not_rematerialize_when_consolidation_is_off():
    """Guard: with the default consolidate-off config there is nothing to
    merge, so the checkpoint path must not rewrite the -o file a second time.
    Fails while materialize runs unconditionally after the no-op consolidate."""
    import copy
    from types import SimpleNamespace

    calls = []
    fs = SimpleNamespace(
        paper=_CannedPaper(_checkpoint_core()),
        warnings=[],
        result_json=copy.deepcopy(_checkpoint_core()),
        artifact_sink=SimpleNamespace(materialize=lambda fs, p: calls.append(p)),
        core_sha256="deadbeef",
        enrichment_state=None,
        path=Path("x.pdf"),
        free_all=lambda: None,
        set_error=_boom_on_error,
    )
    out = _sinked_run(fs, RunConfig())

    assert calls == []
    assert out == _checkpoint_core()


def test_enrichment_replay_carries_enrich_timings():
    """The replayed (-o) payload's extraction.timings must include the enrich
    stage that ran, matching the unsinked export. Fails on base (core
    timings, no enrich key)."""
    import copy
    from types import SimpleNamespace

    from bibr.pipeline.artifacts import RunState, canonical_json_sha256

    core = _checkpoint_core()
    enriched = copy.deepcopy(core)
    enriched["bib_match"] = [{"bib_id": 1, "service": "crossref", "doi": "10.1234/ref"}]
    enriched["extraction"]["timings"] = {
        "stages": {"parse": 1.0, "enrich": 2.0},
        "total_seconds": 3.0,
    }

    stored: dict = {}

    def _materialize(fs, payload):
        stored["payload"] = payload

    sink = SimpleNamespace(
        read_core=lambda fs: copy.deepcopy(core),
        write_enrichment=lambda fs, sidecar: stored.setdefault("sidecar", sidecar),
        read_enrichment=lambda fs: stored["sidecar"],
        materialize=_materialize,
        record=lambda *a, **k: None,
    )
    fs = SimpleNamespace(
        paper=_CannedPaper(enriched),
        warnings=[],
        result_json=copy.deepcopy(core),
        artifact_sink=sink,
        core_sha256=canonical_json_sha256(core),
        enrichment_state=RunState.ENRICHMENT_COMPLETE,
        path=Path("x.pdf"),
        free_all=lambda: None,
        set_error=_boom_on_error,
    )
    out = _sinked_run(fs, RunConfig())

    assert out["bib_match"] == enriched["bib_match"]  # replay succeeded, no fallback
    assert out["extraction"]["timings"] == enriched["extraction"]["timings"]
    assert stored["payload"]["extraction"]["timings"] == enriched["extraction"]["timings"]
