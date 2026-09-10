"""Tests for ``bibr demo`` reference-strategy wiring (``--refs``)."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

gr = pytest.importorskip("gradio")  # demo is gated behind gradio
if not hasattr(gr, "Blocks"):
    pytest.skip("gradio not fully installed", allow_module_level=True)

import bibr.config
import bibr.demo.local_app as local_app
from bibr.demo.local_app import _build_status_md


def test_status_md_includes_refs_strategy():
    md = _build_status_md("glm-mlx", "google", "gemini-flash", refs_strategy="ner")
    assert "ner" in md


def _capture_pipeline_kwargs(monkeypatch):
    captured: dict = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        return object()

    # LocalPipeline is imported lazily inside create_local_demo; patch it at the
    # source module so the in-function import resolves to the stub.
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _stub)
    return captured


@pytest.fixture
def _cache_ocr_state():
    """Save/restore Settings.cache.ocr's value and explicit-set membership."""
    section = bibr.config.Settings.cache
    original_value = section.ocr
    was_explicit = "ocr" in section.model_fields_set
    yield section
    section.ocr = original_value
    if not was_explicit:
        section.model_fields_set.discard("ocr")


def test_create_local_demo_defaults_ocr_cache_on(monkeypatch, caplog, _cache_ocr_state):
    """No CACHE_OCR set by the user — the demo turns the disk cache on."""
    _capture_pipeline_kwargs(monkeypatch)
    section = _cache_ocr_state
    section.ocr = False
    section.model_fields_set.discard("ocr")

    with caplog.at_level(logging.INFO):
        local_app.create_local_demo(ocr_backend="glm-mlx")

    assert section.ocr is True
    assert any("OCR disk cache" in r.message for r in caplog.records)


def test_create_local_demo_respects_explicit_ocr_cache_off(monkeypatch, caplog, _cache_ocr_state):
    """CACHE_OCR=0 (or any explicit value) set by the user is left alone."""
    _capture_pipeline_kwargs(monkeypatch)
    section = _cache_ocr_state
    section.ocr = False
    section.model_fields_set.add("ocr")

    with caplog.at_level(logging.INFO):
        local_app.create_local_demo(ocr_backend="glm-mlx")

    assert section.ocr is False
    assert not any("OCR disk cache" in r.message for r in caplog.records)


def test_create_local_demo_refs_rides_pipeline_kwargs(monkeypatch):
    """``refs='ner'`` must ride LocalPipeline's per-run config, not Settings."""
    captured = _capture_pipeline_kwargs(monkeypatch)
    monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "llm")

    local_app.create_local_demo(ocr_backend="glm-mlx", refs="ner")

    assert captured["ref_parse_strategy"] == "ner"
    assert bibr.config.Settings.REF_PARSE_STRATEGY == "llm"  # global untouched


def test_create_local_demo_default_leaves_refs_strategy_unset(monkeypatch):
    captured = _capture_pipeline_kwargs(monkeypatch)

    local_app.create_local_demo(ocr_backend="glm-mlx")

    assert captured.get("ref_parse_strategy") is None


def test_create_local_demo_passes_llm_backend_to_pipeline(monkeypatch):
    captured = _capture_pipeline_kwargs(monkeypatch)

    local_app.create_local_demo(ocr_backend="glm-mlx", llm_backend="llama-cpp")

    assert captured["llm_backend"] == "llama-cpp"


def test_create_local_demo_default_leaves_memory_mode_unset(monkeypatch):
    captured = _capture_pipeline_kwargs(monkeypatch)

    local_app.create_local_demo(ocr_backend="glm-mlx")

    assert captured["memory_mode"] is None


def test_create_local_demo_passes_explicit_memory_mode(monkeypatch):
    captured = _capture_pipeline_kwargs(monkeypatch)

    local_app.create_local_demo(ocr_backend="glm-mlx", memory_mode="aggressive")

    assert captured["memory_mode"] == "aggressive"


async def test_preset_replacement_applies_full_settings_and_closes_old_pipeline():
    helper = getattr(local_app, "_replace_demo_pipeline_with_preset", None)
    assert helper is not None

    settings = SimpleNamespace(
        llm=SimpleNamespace(provider="google", model="old", backend="cloud"),
        ocr=SimpleNamespace(backend="glm-mlx"),
        OCR_BASE_URL="http://old",
        REF_PARSE_STRATEGY="ner",
    )

    class Manager:
        def apply_to_settings(self, name, target):
            assert name == "gpu"
            target.llm.provider = "openai"
            target.llm.model = "numind/NuExtract3-FP8"
            target.llm.backend = "rapid-mlx"
            target.ocr.backend = "glm-http"
            target.OCR_BASE_URL = "http://gpu-box:8002"
            return []

    old_pipeline = SimpleNamespace(aclose=AsyncMock())
    state = {
        "pipeline": old_pipeline,
        "ocr_backend": "glm-mlx",
        "memory_mode": "balanced",
        "llm_backend": None,
    }
    captured = {}

    def pipeline_factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(llm_backend=kwargs["llm_backend"])

    status = await helper(
        "gpu",
        manager=Manager(),
        settings=settings,
        pipeline_state=state,
        pipeline_factory=pipeline_factory,
        normalize_ocr_backend=lambda value: value,
        refs=None,
    )

    assert captured == {
        "memory_mode": "balanced",
        "ocr_backend": "glm-http",
        "llm_backend": "rapid-mlx",
        "ref_parse_strategy": None,
    }
    assert settings.OCR_BASE_URL == "http://gpu-box:8002"
    assert state["pipeline"] is not old_pipeline
    old_pipeline.aclose.assert_awaited_once()
    assert "openai/numind/NuExtract3-FP8" in status


async def test_failed_preset_replacement_keeps_old_pipeline_open():
    helper = getattr(local_app, "_replace_demo_pipeline_with_preset", None)
    assert helper is not None

    settings = SimpleNamespace(
        llm=SimpleNamespace(provider="google", model="old", backend="cloud"),
        ocr=SimpleNamespace(backend="glm-mlx"),
        REF_PARSE_STRATEGY="ner",
    )

    class Manager:
        def apply_to_settings(self, name, target):
            target.llm.model = "partially-applied"
            target.ocr.backend = "glm-http"
            return []

    old_pipeline = SimpleNamespace(aclose=AsyncMock())
    state = {
        "pipeline": old_pipeline,
        "ocr_backend": "glm-mlx",
        "memory_mode": None,
        "llm_backend": None,
    }

    def fail_factory(**kwargs):
        raise RuntimeError("replacement failed")

    with pytest.raises(RuntimeError, match="replacement failed"):
        await helper(
            "broken",
            manager=Manager(),
            settings=settings,
            pipeline_state=state,
            pipeline_factory=fail_factory,
            normalize_ocr_backend=lambda value: value,
            refs=None,
        )

    assert state["pipeline"] is old_pipeline
    old_pipeline.aclose.assert_not_awaited()
    assert settings.llm.model == "old"
    assert settings.ocr.backend == "glm-mlx"
