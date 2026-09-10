"""Serve pipeline — the ``Pipeline`` subclass used by ``BibrPipelineAPI``.

Builds the same stage list as ``LocalPipeline`` but with an
``OcrBackend`` of type ``"serve-http"`` and a ``ResourceManager``
populated with HTTP client + semaphores + breaker injected from the
LitServe worker.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from bibr.config import snapshot_settings
from bibr.pipeline.context import RunConfig
from bibr.pipeline.pipeline import Pipeline
from bibr.pipeline.plans import build_stage_plan
from bibr.pipeline.resources import ResourceManager

if TYPE_CHECKING:
    import httpx

    from bibr.config import GlobalSettings
    from bibr.utils.circuit_breaker import AsyncCircuitBreaker

logger = logging.getLogger(__name__)


def serve_ocr_defaults(settings: GlobalSettings) -> tuple[str | None, str | None]:
    """``(served_model, profile)`` that ``bibr serve`` assumes for its OCR endpoint.

    ``bibr serve`` always proxies over HTTP, so ``OCR_BACKEND`` cannot pick a
    runtime — but ``paddle-http`` still says which *family* of server sits at
    ``OCR_BASE_URL``. Then the served alias defaults to ``OCR_PADDLE_SERVED_MODEL``
    (``paddle-ocr-vl-1.6``) and the profile to ``paddle``, instead of the GLM
    ``glm-ocr`` defaults every other value keeps (vLLM 404s on a wrong alias).
    An explicit ``OCR_MODEL`` / ``OCR_PROFILE`` always wins.
    """
    ocr = settings.ocr
    model = ocr.model
    profile = ocr.profile
    if (ocr.backend or "").lower() == "paddle-http":
        model = model or ocr.paddle_served_model
        profile = profile or "paddle"
    return model, profile


class ServePipeline(Pipeline):
    """Serve-flavoured stage pipeline.

    Layout + segmenter are pre-constructed by the LitServe worker (loaded
    once per process) and handed to the ``ResourceManager`` as-is. OCR
    runs over HTTP against a co-located OCR server via
    ``BibrServeOcrBackend`` registered as ``"serve-http"``.
    """

    def __init__(
        self,
        *,
        layout,
        segmenter,
        http_client: httpx.AsyncClient,
        ocr_base_url: str,
        ocr_sem_global: asyncio.Semaphore,
        ocr_breaker: AsyncCircuitBreaker,
        crossref: bool = True,
        equations: bool = True,
        ocr_profile: str | None = None,
        settings: GlobalSettings | None = None,
    ) -> None:
        # Ensure BibrServeOcrBackend is imported so its ``@register``
        # decorator runs before ResourceManager tries to instantiate it.
        from bibr.serve import ocr_backend  # noqa: F401

        # Served-model name for the co-located OCR server: the ``glm-ocr``
        # alias, or the Paddle alias under ``OCR_BACKEND=paddle-http``; set
        # ``OCR_MODEL`` to target any other served model — vLLM 404s on a
        # mismatch.
        settings_snapshot = snapshot_settings(settings)
        ocr_model, default_ocr_profile = serve_ocr_defaults(settings_snapshot)

        # Per-request fields (start_page / end_page / include_figures) are
        # supplied via ``process_file(config=...)`` so a single ServePipeline
        # instance can be shared across concurrent requests on one worker.
        config = RunConfig(
            memory_mode="keep_all",
            ocr_backend="serve-http",
            ocr_url=ocr_base_url,
            ocr_model=ocr_model,
            ocr_profile=ocr_profile if ocr_profile is not None else default_ocr_profile,
            device=None,
            crossref=crossref,
            equations=equations,
            llm_backend="cloud",
        )
        resources = ResourceManager(
            memory_mode="keep_all",
            ocr_backend="serve-http",
            ocr_url=ocr_base_url,
            ocr_model=ocr_model,
            ocr_profile=config.ocr_profile,
            http_client=http_client,
            ocr_sem_global=ocr_sem_global,
            ocr_breaker=ocr_breaker,
            managed_vllm_fraction=0.0,
            settings=settings_snapshot,
            layout=layout,
            segmenter=segmenter,
        )

        # Mirror LocalPipeline: refs=off produces an empty bib table, so
        # Crossref enrichment has nothing to enrich. Serve consulted only
        # ``crossref.enrich`` and kept the stage, so a deployment configured
        # with ``REF_PARSE_STRATEGY=off`` still ran an enrich stage that idled
        # on every request — and diverged from the local pipeline's stage list
        # for identical settings.
        refs_off = (settings_snapshot.REF_PARSE_STRATEGY or "").lower() == "off"

        enrichers: list = []
        if crossref and settings_snapshot.crossref.enrich and not refs_off:
            from bibr.pipeline.enricher import CrossrefEnricher

            enrichers.append(CrossrefEnricher(settings=settings_snapshot))

        # NOTE: ServePipeline omits LlmServerStage (which LocalPipeline includes
        # at bibr/local/pipeline.py). The serve worker manages its own
        # vllm-mlx subprocess lifecycle out-of-band — adding LlmServerStage here
        # would double-start the LLM server on the first request.
        stages = build_stage_plan(mode="serve", stream_backhalf=False, enrichers=enrichers)
        super().__init__(
            stages=stages,
            resources=resources,
            config=config,
            settings=settings_snapshot,
        )
