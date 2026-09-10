"""``ServePipeline`` — serve flavour of the shared ``Pipeline`` base class."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


def _build():
    from bibr.serve.pipeline import ServePipeline

    layout = MagicMock()
    segmenter = MagicMock()
    http = MagicMock()
    breaker = MagicMock()
    return ServePipeline(
        layout=layout,
        segmenter=segmenter,
        http_client=http,
        ocr_base_url="http://ocr.local",
        ocr_sem_global=asyncio.Semaphore(16),
        ocr_breaker=breaker,
    )


def test_subclasses_pipeline_base():
    from bibr.pipeline.pipeline import Pipeline

    pl = _build()
    assert isinstance(pl, Pipeline)


def test_stage_list_contains_expected_names():
    pl = _build()
    names = [s.name for s in pl._stages]
    assert "validate" in names
    assert "docx" in names
    assert "render_ocr" in names
    assert "extract" in names
    assert "export" in names


def test_resources_have_serve_fields_populated():
    pl = _build()
    rm = pl._resources
    assert rm.http_client is not None
    assert rm.ocr_sem_global is not None
    assert rm.ocr_breaker is not None
    assert rm.classifiers is not None


def test_config_ocr_backend_is_serve_http():
    pl = _build()
    assert pl._config.ocr_backend == "serve-http"


def test_ocr_model_from_settings_flows_to_resources_and_config(monkeypatch):
    # OCR_MODEL lets the serve OCR backend target a non-``glm-ocr`` served
    # model (e.g. a NuExtract vLLM server). It must reach both the
    # ResourceManager (→ backend payload) and the RunConfig.
    from bibr.config import Settings

    monkeypatch.setattr(Settings.ocr, "model", "numind/NuExtract-2.0-2B")
    pl = _build()
    assert pl._resources.ocr_model == "numind/NuExtract-2.0-2B"
    assert pl._config.ocr_model == "numind/NuExtract-2.0-2B"


def test_config_per_request_fields_unset_on_pipeline():
    # Per-request fields (start_page / end_page / include_figures) are no
    # longer baked into ServePipeline — they ride on a fresh RunConfig built
    # by ``BibrPipelineAPI.predict`` so a single pipeline can be shared
    # across concurrent requests.
    pl = _build()
    assert pl._config.include_figures is None
    assert pl._config.start_page is None
    assert pl._config.end_page is None


def test_serve_ocr_defaults_follow_the_backend_family():
    """``OCR_BACKEND=paddle-http`` (the deployment guide's Paddle recipe) selects
    the Paddle served alias and profile; every other value keeps the GLM defaults."""
    from bibr.config import GlobalSettings
    from bibr.serve.pipeline import serve_ocr_defaults

    assert serve_ocr_defaults(GlobalSettings(ocr={"backend": "paddle"})) == (None, None)
    assert serve_ocr_defaults(GlobalSettings(ocr={"backend": "glm-http"})) == (None, None)
    assert serve_ocr_defaults(GlobalSettings(ocr={"backend": "paddle-http"})) == (
        "paddle-ocr-vl-1.6",
        "paddle",
    )
    explicit = GlobalSettings(ocr={"backend": "paddle-http", "model": "my/alias", "profile": "glm"})
    assert serve_ocr_defaults(explicit) == ("my/alias", "glm")


def test_serve_pipeline_requests_the_paddle_alias_under_paddle_http():
    from bibr.config import GlobalSettings
    from bibr.serve.pipeline import ServePipeline

    pl = ServePipeline(
        layout=MagicMock(),
        segmenter=MagicMock(),
        http_client=MagicMock(),
        ocr_base_url="http://ocr.local",
        ocr_sem_global=asyncio.Semaphore(16),
        ocr_breaker=MagicMock(),
        settings=GlobalSettings(ocr={"backend": "paddle-http"}),
    )
    assert pl._config.ocr_model == "paddle-ocr-vl-1.6"
    assert pl._config.ocr_profile == "paddle"
