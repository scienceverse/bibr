"""The text-layer OCR backend through the real pipeline stages.

Layout and sentence segmentation are the smoke-test fakes and the run has no
LLM, so nothing loads a model; the OCR registry must never be asked for a
backend that would run one.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.config import Settings
from bibr.exceptions import ProcessingError
from bibr.ocr import registry
from tests import test_pipeline_smoke as smoke

SCANNED = Path(__file__).resolve().parents[1] / "fixtures" / "scanned_sample.pdf"


@pytest.fixture
def created_backends(monkeypatch):
    """Every backend the OCR registry is asked to create (none should run OCR)."""
    created: list[str] = []
    real_create = registry.create

    def create(name, **kwargs):
        created.append(name)
        if name != registry.TEXT_LAYER_BACKEND:
            raise AssertionError(f"OCR backend {name!r} must not be created")
        return real_create(name, **kwargs)

    monkeypatch.setattr(registry, "create", create)
    return created


@pytest.fixture
def no_ocr_runtime(monkeypatch):
    """A machine where the automatic chain is llama.cpp alone and it is missing."""
    real = registry.resolve_backend_candidates
    llama = (registry.OcrBackendCandidate("glm-llama", "glm-ocr", "glm"),)
    monkeypatch.setattr(
        registry,
        "resolve_backend_candidates",
        lambda backend, settings: llama if backend == "paddle" else real(backend, settings),
    )
    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: None)
    monkeypatch.setattr(
        "bibr.local.llama_cpp.missing_binary_message", lambda **kwargs: "llama.cpp not installed"
    )


def _pipeline(monkeypatch, **kwargs):
    from bibr.local.pipeline import LocalPipeline
    from bibr.pipeline.resources import ResourceManager

    monkeypatch.setattr(Settings.ml, "section_classifier_model_id", None)
    pipeline = LocalPipeline(crossref=False, no_llm=True, **kwargs)
    rm = ResourceManager(
        memory_mode=pipeline.memory_mode,
        ocr_backend=pipeline.ocr_backend,
        ocr_url=pipeline.ocr_url,
        ocr_model=pipeline.ocr_model,
        device=pipeline.device,
        settings=pipeline.settings,
        layout=smoke.FakeLayout(),
        segmenter=smoke.FakeSegmenter(),
    )
    pipeline._resources = rm
    return pipeline, rm


async def _run(pipeline, path: Path):
    try:
        return await pipeline.process_file(path)
    finally:
        await pipeline.aclose()


def _codes(result) -> list[str]:
    return [warning["code"] for warning in result["extraction"]["warnings"]]


async def test_text_layer_run_reads_every_region_without_ocr(
    tmp_path, monkeypatch, created_backends
):
    from bibr.export.json_export import validate_export
    from bibr.ocr import text_layer

    reads = []
    real_read = text_layer.read_text_layer

    def read_text_layer(*args, **kwargs):
        reads.append(real_read(*args, **kwargs))
        return reads[-1]

    monkeypatch.setattr(text_layer, "read_text_layer", read_text_layer)
    pdf = tmp_path / "smoke.pdf"
    pdf.write_bytes(smoke._build_pdf(smoke._LINES))
    pipeline, rm = _pipeline(monkeypatch, ocr_backend="text-layer")
    rm.start_ocr_preload = MagicMock(side_effect=AssertionError("no OCR preload"))

    result = await _run(pipeline, pdf)

    ocr = result["extraction"]["ocr"]
    assert (ocr["backend"], ocr["model"], ocr["profile"]) == ("text-layer", "pdf-text-layer", None)
    # The short regions the native pass leaves to OCR (the author line, the
    # headings) were read from the text layer.
    [read] = reads
    assert read.pages[0].ok
    assert sum(read.filled.values()) == len(smoke._REGIONS)
    assert result["metadata"]["title"] == smoke._TITLE
    headings = [section["header"] for section in result["section"]]
    assert "Introduction" in headings
    assert validate_export(result) == []
    assert not [code for code in _codes(result) if code.startswith("TEXT_LAYER_")]
    rm.start_ocr_preload.assert_not_called()
    assert set(created_backends) <= {registry.TEXT_LAYER_BACKEND}


async def test_scanned_pdf_fails_as_needing_ocr(tmp_path, monkeypatch, created_backends):
    pipeline, _ = _pipeline(monkeypatch, ocr_backend="text-layer")

    with pytest.raises(ProcessingError) as raised:
        await _run(pipeline, SCANNED)

    assert raised.value.error_code == "scan_needs_ocr"
    assert "scanned PDF, which needs OCR" in str(raised.value)
    # Asked for: the file's own verdict, which a resumed batch does not retry.
    assert raised.value.outage is False
    assert set(created_backends) <= {registry.TEXT_LAYER_BACKEND}


async def test_automatic_backend_falls_back_when_no_runtime_can_start(
    tmp_path, monkeypatch, created_backends, no_ocr_runtime
):
    from bibr.pipeline.resources import ResourceManager

    monkeypatch.setattr(Settings.ocr, "fallback_to_text_layer", "auto")
    start = AsyncMock(side_effect=AssertionError("no engine start is attempted"))
    monkeypatch.setattr(ResourceManager, "await_ocr", start)
    pdf = tmp_path / "smoke.pdf"
    pdf.write_bytes(smoke._build_pdf(smoke._LINES))
    pipeline, rm = _pipeline(monkeypatch)

    result = await _run(pipeline, pdf)

    start.assert_not_awaited()
    assert result["extraction"]["ocr"]["backend"] == "text-layer"
    [warning] = [
        w for w in result["extraction"]["warnings"] if w["code"] == "OCR_BACKEND_UNAVAILABLE"
    ]
    assert "glm-llama: llama.cpp not installed" in warning["message"]
    assert rm.ocr_text_layer_fallback == "glm-llama: llama.cpp not installed"
    assert result["metadata"]["title"] == smoke._TITLE


async def test_a_runtime_that_fails_to_start_is_not_a_reason_to_fall_back(
    tmp_path, monkeypatch, created_backends
):
    """A machine that can run OCR fails the file, as without the setting.

    A startup failure there (a busy GPU, a server that died) is transient:
    the export must not silently become a text-layer read, and the run must
    not stay on the text layer for every later file.
    """
    from bibr.pipeline.resources import ResourceManager

    monkeypatch.setattr(Settings.ocr, "fallback_to_text_layer", "auto")
    monkeypatch.setattr(registry, "automatic_chain_blockers", lambda settings: [])
    monkeypatch.setattr(
        ResourceManager, "await_ocr", AsyncMock(side_effect=RuntimeError("CUDA out of memory"))
    )
    pdf = tmp_path / "smoke.pdf"
    pdf.write_bytes(smoke._build_pdf(smoke._LINES))
    pipeline, rm = _pipeline(monkeypatch)

    with pytest.raises(ProcessingError) as raised:
        await _run(pipeline, pdf)

    assert raised.value.error_code == "ocr_failed"
    assert rm.ocr_text_layer_fallback is None


async def test_a_scanned_pdf_under_the_fallback_fails_as_an_outage(
    monkeypatch, created_backends, no_ocr_runtime
):
    """Not the file's verdict: a resumed batch reads it again once OCR is installed."""
    monkeypatch.setattr(Settings.ocr, "fallback_to_text_layer", "auto")
    pipeline, _ = _pipeline(monkeypatch)

    with pytest.raises(ProcessingError) as raised:
        await _run(pipeline, SCANNED)

    assert raised.value.error_code == "scan_needs_ocr"
    assert raised.value.outage is True


async def test_a_cached_text_layer_read_under_the_fallback_says_why(
    tmp_path, monkeypatch, created_backends, no_ocr_runtime
):
    """A cache entry from an explicit text-layer run carries no fallback warning."""
    monkeypatch.setattr(Settings.cache, "ocr", True)
    monkeypatch.setattr(Settings.cache, "ocr_dir", str(tmp_path / "cache"))
    pdf = tmp_path / "smoke.pdf"
    pdf.write_bytes(smoke._build_pdf(smoke._LINES))
    explicit, _ = _pipeline(monkeypatch, ocr_backend="text-layer")
    first = await _run(explicit, pdf)
    assert "OCR_BACKEND_UNAVAILABLE" not in _codes(first)
    assert list((tmp_path / "cache").rglob("*.json")), "the explicit run stores its read"

    monkeypatch.setattr(Settings.ocr, "fallback_to_text_layer", "auto")
    automatic, rm = _pipeline(monkeypatch)
    read = MagicMock(side_effect=AssertionError("served from the cache"))
    monkeypatch.setattr("bibr.ocr.text_layer.read_text_layer", read)

    second = await _run(automatic, pdf)

    assert rm.ocr_text_layer_fallback is not None
    assert second["extraction"]["ocr"]["backend"] == "text-layer"
    assert _codes(second).count("OCR_BACKEND_UNAVAILABLE") == 1


def test_the_fallback_check_needs_auto_the_automatic_backend_and_no_url(monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    blocked = MagicMock(return_value=["glm-llama: llama.cpp not installed"])
    monkeypatch.setattr(registry, "automatic_chain_blockers", blocked)

    def check(fallback: str, **kwargs) -> str | None:
        monkeypatch.setattr(Settings.ocr, "fallback_to_text_layer", fallback)
        rm = ResourceManager(memory_mode="balanced", settings=Settings, **kwargs)
        return rm.check_text_layer_fallback()

    assert check("off", ocr_backend="paddle") is None
    assert check("auto", ocr_backend="glm-llama") is None
    assert check("auto", ocr_backend="paddle", ocr_url="http://ocr.invalid:8080") is None
    blocked.assert_not_called()
    assert check("auto", ocr_backend="paddle") == "glm-llama: llama.cpp not installed"


def test_the_fallback_check_runs_once(monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    blocked = MagicMock(return_value=[])
    monkeypatch.setattr(registry, "automatic_chain_blockers", blocked)
    monkeypatch.setattr(Settings.ocr, "fallback_to_text_layer", "auto")
    rm = ResourceManager(memory_mode="balanced", ocr_backend="paddle", settings=Settings)

    assert rm.check_text_layer_fallback() is None
    assert rm.check_text_layer_fallback() is None
    blocked.assert_called_once()


def test_automatic_chain_blockers_names_every_candidate_or_none(monkeypatch):
    candidates = (
        registry.OcrBackendCandidate("paddle-vllm", "paddle-ocr", "paddle"),
        registry.OcrBackendCandidate("glm-llama", "glm-ocr", "glm"),
    )
    monkeypatch.setattr(registry, "resolve_backend_candidates", lambda backend, s: candidates)
    reasons = {"paddle-vllm": "no NVIDIA GPU", "glm-llama": "llama.cpp not installed"}
    monkeypatch.setattr(registry, "local_runtime_unavailable_reason", reasons.get)

    assert registry.automatic_chain_blockers(Settings) == [
        "paddle-vllm: no NVIDIA GPU",
        "glm-llama: llama.cpp not installed",
    ]
    reasons["glm-llama"] = None
    assert registry.automatic_chain_blockers(Settings) == []


async def test_automatic_backend_without_fallback_fails_the_file(
    tmp_path, monkeypatch, created_backends
):
    from bibr.pipeline.resources import ResourceManager

    monkeypatch.setattr(Settings.ocr, "fallback_to_text_layer", "off")
    monkeypatch.setattr(
        ResourceManager, "await_ocr", AsyncMock(side_effect=RuntimeError("no OCR runtime"))
    )
    pdf = tmp_path / "smoke.pdf"
    pdf.write_bytes(smoke._build_pdf(smoke._LINES))
    pipeline, rm = _pipeline(monkeypatch)

    with pytest.raises(ProcessingError) as raised:
        await _run(pipeline, pdf)

    assert raised.value.error_code == "ocr_failed"
    assert rm.ocr_text_layer_fallback is None


async def test_fallback_is_kept_for_later_chunks():
    from bibr.ocr.profiles import text_layer_identity
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.ocr import resolve_ocr_identity

    rm = MagicMock()
    rm.ocr_text_layer_fallback = "glm-llama: llama.cpp not installed"
    rm.await_ocr = AsyncMock(side_effect=AssertionError("must not retry the engine"))
    ctx = PipelineContext(file_states=[], progress=NullProgress(), resources=rm, config=RunConfig())

    resolution = await resolve_ocr_identity(ctx, needs_backend=True, fail_targets=[])

    assert resolution.identity == text_layer_identity()
    assert ctx.scratch["ocr_runtime_identity"] == text_layer_identity()


async def test_layout_stage_does_not_preload_ocr_in_text_layer_mode():
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.layout import LayoutStage
    from bibr.pipeline.state import FileState

    fs = FileState(path=Path("x.pdf"))
    fs.pdf_bytes = b"%PDF"
    rm = MagicMock()
    rm.layout = MagicMock(detect_batch=AsyncMock(return_value=[[]]))
    rm.ocr = None
    rm.ocr_text_layer_fallback = None
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(memory_mode="balanced", ocr_backend="text-layer"),
    )
    with patch(
        "bibr.pipeline.stages.layout._iter_pdf_pages",
        return_value=iter([(0, MagicMock(width=800, height=1000))]),
    ):
        await LayoutStage().run(ctx)

    rm.start_ocr_preload.assert_not_called()
    assert ctx.signals.preloading_ocr is False


def test_text_layer_identity_has_its_own_cache_key(tmp_path):
    from bibr.ocr.profiles import OcrRuntimeIdentity, text_layer_identity
    from bibr.pipeline import ocr_cache
    from bibr.pipeline.context import RunConfig

    identity = text_layer_identity()
    assert (identity.backend, identity.model, identity.profile) == (
        "text-layer",
        "pdf-text-layer",
        None,
    )
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 text layer")
    from bibr.pipeline.state import FileState

    fs = FileState(path=pdf, pdf_bytes=pdf.read_bytes())
    config = RunConfig(ocr_backend="text-layer")
    settings = Settings
    glm = OcrRuntimeIdentity(
        backend="glm-llama",
        model="glm-ocr",
        profile="glm",
        normalizer_version=identity.normalizer_version,
    )
    assert ocr_cache._key(fs, config, identity, settings) != ocr_cache._key(
        fs, config, glm, settings
    )


def test_the_fallback_warning_is_recorded_once_and_only_for_the_fallback():
    from bibr.ocr.text_layer import note_text_layer_fallback
    from bibr.pipeline.state import FileState
    from bibr.processing_warnings import WarningCode

    fs = FileState(path=Path("x.pdf"))
    note_text_layer_fallback(fs, None)  # an explicit --ocr text-layer run
    assert fs.warnings == []

    note_text_layer_fallback(fs, "glm-llama: llama.cpp not installed")
    note_text_layer_fallback(fs, "glm-llama: llama.cpp not installed")

    [warning] = fs.warnings
    assert warning.code == WarningCode.OCR_BACKEND_UNAVAILABLE
    assert "glm-llama: llama.cpp not installed" in warning.message
