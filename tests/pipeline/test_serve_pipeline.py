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
    assert "layout" in names
    assert "ocr" in names
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


def _enrichment_stage(pl):
    from bibr.pipeline.stages.enrich import EnrichmentStage

    (stage,) = [s for s in pl._stages if isinstance(s, EnrichmentStage)]
    return stage


def _build_with(settings):
    from bibr.serve.pipeline import ServePipeline

    return ServePipeline(
        layout=MagicMock(),
        segmenter=MagicMock(),
        http_client=MagicMock(),
        ocr_base_url="http://ocr.local",
        ocr_sem_global=asyncio.Semaphore(16),
        ocr_breaker=MagicMock(),
        settings=settings,
    )


def test_enricher_is_built_even_when_crossref_enrich_setting_is_off():
    """Enrichment is a per-request switch on serve: the stage must exist for a
    ``crossref=true`` request even though CROSSREF_ENRICH is off (the default)."""
    from bibr.config import GlobalSettings
    from bibr.pipeline.enricher import CrossrefEnricher
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage

    settings = GlobalSettings()
    settings.crossref.enrich = False
    pl = _build_with(settings)

    enrichers = _enrichment_stage(pl)._enrichers
    assert len(enrichers) == 1 and isinstance(enrichers[0], CrossrefEnricher)
    (checkpoint,) = [s for s in pl._stages if isinstance(s, CoreCheckpointStage)]
    assert checkpoint._enrichment_requested is True
    assert pl._config.crossref is None


async def test_refs_off_default_allows_a_request_to_enable_parsing_and_enrichment():
    from dataclasses import replace
    from pathlib import Path
    from unittest.mock import AsyncMock

    from bibr.config import GlobalSettings
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.context import PipelineContext
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.state import FileState

    settings = GlobalSettings()
    settings.crossref.enrich = True
    settings.REF_PARSE_STRATEGY = "off"

    pl = _build_with(settings)
    stage = _enrichment_stage(pl)
    (enricher,) = stage._enrichers
    enricher.enrich = AsyncMock(return_value=None)
    fs = FileState(path=Path("paper.xml"), paper=MagicMock())
    ctx = PipelineContext([fs], NullProgress(), pl._resources, pl._config, settings)
    await stage.run(ctx)
    enricher.enrich.assert_not_awaited()
    assert fs.enrichment_state is None

    ctx.config = replace(pl._config, ref_parse_strategy="ner", crossref=True)
    await stage.run(ctx)
    enricher.enrich.assert_awaited_once_with(fs)
    assert fs.enrichment_state == RunState.ENRICHMENT_COMPLETE

    enricher.enrich.reset_mock()
    ctx.config = replace(pl._config, ref_parse_strategy="off", crossref=True)
    await stage.run(ctx)
    enricher.enrich.assert_not_awaited()


async def _run_enrichment(pl, *, crossref, setting):
    """Drive the serve pipeline's EnrichmentStage with a per-request RunConfig."""
    import dataclasses
    from pathlib import Path
    from unittest.mock import AsyncMock

    from bibr.pipeline.context import PipelineContext
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.state import FileState

    stage = _enrichment_stage(pl)
    (enricher,) = stage._enrichers
    enricher.enrich = AsyncMock(return_value=None)
    settings = pl.settings
    settings.crossref.enrich = setting
    fs = FileState(path=Path("paper.pdf"), paper=MagicMock())
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=MagicMock(),
        config=dataclasses.replace(pl._config, crossref=crossref),
        settings=settings,
    )
    await stage.run(ctx)
    return enricher.enrich


async def test_request_crossref_true_runs_enrichment_when_setting_is_off():
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.crossref.enrich = False
    pl = _build_with(settings)

    enrich = await _run_enrichment(pl, crossref=True, setting=False)
    enrich.assert_awaited_once()


async def test_request_crossref_false_skips_enrichment_when_setting_is_on():
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.crossref.enrich = True
    pl = _build_with(settings)

    enrich = await _run_enrichment(pl, crossref=False, setting=True)
    enrich.assert_not_awaited()


async def test_request_without_crossref_follows_setting():
    from bibr.config import GlobalSettings

    pl = _build_with(GlobalSettings())
    assert not (await _run_enrichment(pl, crossref=None, setting=False)).await_count
    assert (await _run_enrichment(pl, crossref=None, setting=True)).await_count == 1
