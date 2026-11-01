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
    """No CACHE_OCR set, and the demo keeps its files — it turns the disk cache on."""
    _capture_pipeline_kwargs(monkeypatch)
    monkeypatch.setenv("DEMO_CACHE_TTL_SECONDS", "0")
    section = _cache_ocr_state
    section.ocr = False
    section.model_fields_set.discard("ocr")

    with caplog.at_level(logging.INFO):
        local_app.create_local_demo(ocr_backend="glm-mlx")

    assert section.ocr is True
    assert any("OCR disk cache" in r.message for r in caplog.records)


def test_create_local_demo_leaves_ocr_cache_off_when_deleting_files(
    monkeypatch, caplog, _cache_ocr_state
):
    """The cache keeps each paper's OCR text with no expiry, so a demo that
    deletes uploads (the default) does not turn it on."""
    _capture_pipeline_kwargs(monkeypatch)
    monkeypatch.delenv("DEMO_CACHE_TTL_SECONDS", raising=False)
    section = _cache_ocr_state
    section.ocr = False
    section.model_fields_set.discard("ocr")

    with caplog.at_level(logging.INFO):
        local_app.create_local_demo(ocr_backend="glm-mlx")

    assert section.ocr is False
    assert not any("OCR disk cache" in r.message for r in caplog.records)


@pytest.mark.parametrize("ttl", ["0", "3600"])
@pytest.mark.parametrize("explicit", [False, True])
def test_create_local_demo_respects_explicit_ocr_cache(
    monkeypatch, caplog, _cache_ocr_state, ttl, explicit
):
    """An explicit CACHE_OCR set by the user is left alone, whatever the cleanup."""
    _capture_pipeline_kwargs(monkeypatch)
    monkeypatch.setenv("DEMO_CACHE_TTL_SECONDS", ttl)
    section = _cache_ocr_state
    section.ocr = explicit
    section.model_fields_set.add("ocr")

    with caplog.at_level(logging.INFO):
        local_app.create_local_demo(ocr_backend="glm-mlx")

    assert section.ocr is explicit
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


# --- run mode: --fast / --full -----------------------------------------------


@pytest.fixture
def _no_llm(monkeypatch):
    """A demo started on a machine with no LLM configured."""
    from bibr.config import GlobalSettings

    for var in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BACKEND",
        "PIPELINE_MODE",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = GlobalSettings()
    monkeypatch.setattr("bibr.config.Settings", settings)
    return settings


def test_create_local_demo_passes_fast_mode_to_the_pipeline(monkeypatch, _no_llm):
    captured = _capture_pipeline_kwargs(monkeypatch)

    local_app.create_local_demo(ocr_backend="glm-mlx", mode="fast")

    assert captured["mode"] == "fast"


def test_create_local_demo_with_no_llm_configured_runs_fast(monkeypatch, _no_llm):
    captured = _capture_pipeline_kwargs(monkeypatch)

    local_app.create_local_demo(ocr_backend="glm-mlx")

    assert captured["mode"] == "fast"


def test_create_local_demo_with_a_key_runs_full(monkeypatch, _no_llm):
    captured = _capture_pipeline_kwargs(monkeypatch)
    _no_llm.GOOGLE_API_KEY = "test-key"

    local_app.create_local_demo(ocr_backend="glm-mlx")

    assert captured["mode"] == "full"


def test_create_local_demo_full_flag_beats_auto(monkeypatch, _no_llm):
    captured = _capture_pipeline_kwargs(monkeypatch)

    local_app.create_local_demo(ocr_backend="glm-mlx", mode="full")

    assert captured["mode"] == "full"


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [({"llm_backend": "llama-cpp"}, "--llm llama-cpp"), ({"refs": "llm"}, "--refs llm")],
)
def test_create_local_demo_fast_refuses_an_llm_option(monkeypatch, _no_llm, kwargs, named):
    _capture_pipeline_kwargs(monkeypatch)

    with pytest.raises(ValueError, match="calls an LLM, which fast mode"):
        local_app.create_local_demo(ocr_backend="glm-mlx", mode="fast", **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"llm_backend": "llama-cpp"}, {"refs": "llm"}], ids=["llm", "refs"]
)
def test_create_local_demo_an_llm_option_asks_for_full_under_auto(monkeypatch, _no_llm, kwargs):
    captured = _capture_pipeline_kwargs(monkeypatch)

    local_app.create_local_demo(ocr_backend="glm-mlx", **kwargs)

    assert captured["mode"] == "full"


def test_status_md_names_the_mode_and_no_llm_for_a_fast_run():
    md = _build_status_md("glm-mlx", "google", "gemini-flash", refs_strategy="ner", mode="fast")
    assert "**Mode** `fast`" in md
    assert "none (fast mode)" in md
    assert "gemini-flash" not in md
    assert "google" not in md


def test_status_md_names_the_mode_and_llm_for_a_full_run():
    md = _build_status_md("glm-mlx", "google", "gemini-flash", refs_strategy="llm", mode="full")
    assert "**Mode** `full`" in md
    assert "`google/gemini-flash`" in md


def test_status_md_without_a_mode_is_a_full_banner():
    assert "**Mode** `full`" in _build_status_md("glm-mlx", "google", "gemini-flash")


def test_effective_refs_shows_the_local_parser_for_a_fast_run():
    settings = SimpleNamespace(REF_PARSE_STRATEGY="llm")
    assert local_app._effective_refs(None, settings, "fast") == "ner"
    assert local_app._effective_refs(None, settings, "full") == "llm"
    assert local_app._effective_refs("ner", settings, "full") == "ner"


async def test_a_preset_swap_carries_an_explicit_mode():
    settings = SimpleNamespace(
        llm=SimpleNamespace(provider="google", model="m", backend="cloud"),
        ocr=SimpleNamespace(backend="glm-mlx"),
        REF_PARSE_STRATEGY="llm",
    )

    class Manager:
        def apply_to_settings(self, name, target):  # noqa: ARG002
            return []

    captured = {}

    def pipeline_factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(llm_backend=None, mode="fast")

    state = {
        "pipeline": SimpleNamespace(aclose=AsyncMock()),
        "ocr_backend": "glm-mlx",
        "memory_mode": None,
        "llm_backend": None,
        "mode": "fast",
    }
    status = await local_app._replace_demo_pipeline_with_preset(
        "p",
        manager=Manager(),
        settings=settings,
        pipeline_state=state,
        pipeline_factory=pipeline_factory,
        normalize_ocr_backend=lambda value: value,
        refs=None,
    )

    assert captured["mode"] == "fast"
    # The banner follows the replacement's mode, and a fast run parses with the local model.
    assert "**Mode** `fast`" in status
    assert "none (fast mode)" in status
    assert "**Refs** `ner`" in status


async def test_a_preset_swap_in_an_automatic_demo_leaves_the_mode_to_the_settings():
    settings = SimpleNamespace(
        llm=SimpleNamespace(provider="google", model="m", backend="cloud"),
        ocr=SimpleNamespace(backend="glm-mlx"),
        REF_PARSE_STRATEGY="ner",
    )

    class Manager:
        def apply_to_settings(self, name, target):  # noqa: ARG002
            return []

    captured = {}

    def pipeline_factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(llm_backend=None, mode="full")

    state = {
        "pipeline": SimpleNamespace(aclose=AsyncMock()),
        "ocr_backend": "glm-mlx",
        "memory_mode": None,
        "llm_backend": None,
        "mode": None,
    }
    status = await local_app._replace_demo_pipeline_with_preset(
        "p",
        manager=Manager(),
        settings=settings,
        pipeline_state=state,
        pipeline_factory=pipeline_factory,
        normalize_ocr_backend=lambda value: value,
        refs=None,
    )

    assert "mode" not in captured
    assert "**Mode** `full`" in status


def _demo_main(monkeypatch, argv, create):
    import sys

    from bibr.demo.server import main

    launched = []

    class _Demo:
        def launch(self, **kwargs):
            launched.append(kwargs)

    monkeypatch.setattr(sys, "argv", ["bibr-demo", *argv])
    monkeypatch.setattr(local_app, "create_local_demo", lambda **kw: create(_Demo(), kw))
    main()
    return launched


@pytest.mark.parametrize(
    ("flags", "expected"), [(["--fast"], "fast"), (["--full"], "full"), ([], None)]
)
def test_demo_main_passes_the_run_mode(monkeypatch, flags, expected):
    seen = {}

    def create(demo, kwargs):
        seen.update(kwargs)
        return demo

    assert len(_demo_main(monkeypatch, flags, create)) == 1
    assert seen["mode"] == expected


def test_demo_main_reports_fast_with_an_llm_option_as_a_usage_error(monkeypatch, capsys):
    def create(_demo, _kwargs):
        raise ValueError("--refs llm calls an LLM, which fast mode (--fast) never does.")

    with pytest.raises(SystemExit) as exc_info:
        _demo_main(monkeypatch, ["--fast", "--refs", "llm"], create)

    assert exc_info.value.code == 2
    assert "calls an LLM, which fast mode" in " ".join(capsys.readouterr().err.split())


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
