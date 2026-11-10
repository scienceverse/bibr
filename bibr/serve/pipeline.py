"""Serve pipeline — the ``Pipeline`` subclass used by ``BibrPipelineAPI``.

Builds the same stage list as ``LocalPipeline`` but with an
``OcrBackend`` of type ``"serve-http"`` and a ``ResourceManager``
populated with HTTP client + semaphores + breaker injected from the
LitServe worker. A deployment on the text-layer route
(:func:`serve_ocr_route`) uses the ``"text-layer"`` backend instead and
contacts no OCR server.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from bibr.config import snapshot_settings
from bibr.ocr.registry import TEXT_LAYER_BACKEND, ocr_route_chosen
from bibr.pipeline.context import RunConfig
from bibr.pipeline.pipeline import Pipeline
from bibr.pipeline.plans import build_stage_plan
from bibr.pipeline.resources import ResourceManager
from bibr.run_mode import resolve_mode

if TYPE_CHECKING:
    import httpx

    from bibr.config import GlobalSettings
    from bibr.utils.circuit_breaker import AsyncCircuitBreaker

logger = logging.getLogger(__name__)


def serve_ocr_defaults(settings: GlobalSettings) -> tuple[str | None, str | None]:
    """``(served_model, profile)`` that ``bibr serve`` assumes for its OCR endpoint.

    ``bibr serve`` always proxies over HTTP, so ``OCR_BACKEND`` cannot pick a
    runtime — but ``paddle-http`` still says which *family* of server sits at
    ``OCR_BASE_URL``, as does an explicit ``OCR_PROFILE=paddle``. Then the
    served alias defaults to ``OCR_PADDLE_SERVED_MODEL``
    (``paddle-ocr-vl-1.6``) and the profile to ``paddle``, instead of the GLM
    ``glm-ocr`` defaults every other value keeps (vLLM 404s on a wrong alias).
    An explicit ``OCR_MODEL`` / ``OCR_PROFILE`` always wins.
    """
    from bibr.ocr.profiles import resolve_served_family

    ocr = settings.ocr
    model = ocr.model
    profile = ocr.profile
    if (
        resolve_served_family(
            requested_backend=(ocr.backend or "serve-http").lower(),
            explicit_profile=ocr.profile,
        )
        == "paddle"
    ):
        model = model or ocr.paddle_served_model
        profile = profile or "paddle"
    return model, profile


#: The OCR backend of a deployment that proxies to an OCR server.
SERVE_HTTP_BACKEND = "serve-http"


def serve_ocr_route(settings: GlobalSettings, mode: str | None = None) -> str:
    """The OCR backend a ``bibr serve`` deployment runs: ``"text-layer"`` or ``"serve-http"``.

    The text layer (no OCR server, born-digital PDFs only; a scan is refused
    with ``scan_needs_ocr``) when ``OCR_BACKEND=text-layer`` asks for it, and
    for a fast deployment where nobody chose an OCR route: none of
    ``OCR_BACKEND``, ``OCR_MODEL``, ``OCR_PROFILE`` or ``OCR_BASE_URL`` is set.
    That is the rule ``bibr chew --fast`` follows, with ``OCR_BASE_URL`` in the
    place of ``--ocr-url``. Every other deployment sends regions to the server
    at ``OCR_BASE_URL``.

    The route belongs to the deployment: *mode* is the deployment's (the
    argument, else ``PIPELINE_MODE``), and a request's ``mode`` field does not
    move it.
    """
    if (settings.ocr.backend or "").lower() == TEXT_LAYER_BACKEND:
        return TEXT_LAYER_BACKEND
    if resolve_mode(settings, mode, backend="cloud") != "fast":
        return SERVE_HTTP_BACKEND
    if ocr_route_chosen(settings) or "OCR_BASE_URL" in settings.model_fields_set:
        return SERVE_HTTP_BACKEND
    return TEXT_LAYER_BACKEND


class ServePipeline(Pipeline):
    """Serve-flavoured stage pipeline.

    Layout + segmenter are pre-constructed by the LitServe worker (loaded
    once per process) and handed to the ``ResourceManager`` as-is. OCR
    runs over HTTP against a co-located OCR server via
    ``BibrServeOcrBackend`` registered as ``"serve-http"``, or not at all on
    the text-layer route (:func:`serve_ocr_route`), which reads each PDF's own
    text and leaves the HTTP client, semaphore and breaker unused.
    """

    def __init__(
        self,
        *,
        layout,
        segmenter,
        http_client: httpx.AsyncClient | None,
        ocr_base_url: str,
        ocr_sem_global: asyncio.Semaphore,
        ocr_breaker: AsyncCircuitBreaker,
        crossref: bool | None = None,
        equations: bool = True,
        ocr_profile: str | None = None,
        settings: GlobalSettings | None = None,
        mode: str | None = None,
    ) -> None:
        # Ensure BibrServeOcrBackend is imported so its ``@register``
        # decorator runs before ResourceManager tries to instantiate it.
        from bibr.serve import ocr_backend  # noqa: F401

        # Served-model name for the co-located OCR server: the ``glm-ocr``
        # alias, or the Paddle alias under ``OCR_BACKEND=paddle-http`` or
        # ``OCR_PROFILE=paddle``; set ``OCR_MODEL`` to target any other served model — vLLM 404s on a
        # mismatch.
        settings_snapshot = snapshot_settings(settings)
        run_mode = resolve_mode(settings_snapshot, mode, backend="cloud")
        self.ocr_route = serve_ocr_route(settings_snapshot, run_mode)
        if self.ocr_route == TEXT_LAYER_BACKEND:
            # No server and no model: nothing to address or to profile.
            ocr_url, ocr_model, run_ocr_profile = None, None, None
        else:
            ocr_url = ocr_base_url
            ocr_model, default_ocr_profile = serve_ocr_defaults(settings_snapshot)
            run_ocr_profile = ocr_profile if ocr_profile is not None else default_ocr_profile

        # Per-request fields (start_page / end_page / include_figures /
        # crossref / mode) are supplied via ``process_file(config=...)`` so a
        # single ServePipeline instance can be shared across concurrent requests
        # on one worker. ``crossref`` here is the deployment-level tri-state
        # (None = follow CROSSREF_ENRICH); a request's ``crossref`` form field
        # overrides it per run. ``mode`` is the deployment's (the argument, else
        # PIPELINE_MODE, which under auto is fast when no LLM is configured);
        # a request's ``mode`` field overrides it the same way.
        config = RunConfig(
            memory_mode="keep_all",
            ocr_backend=self.ocr_route,
            ocr_url=ocr_url,
            ocr_model=ocr_model,
            ocr_profile=run_ocr_profile,
            device=None,
            crossref=crossref,
            equations=equations,
            llm_backend="cloud",
            mode=run_mode,
        )
        resources = ResourceManager(
            memory_mode="keep_all",
            ocr_backend=self.ocr_route,
            ocr_url=ocr_url,
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

        # Enrichment is decided per request (the ``crossref`` form field can
        # switch it on for a deployment whose CROSSREF_ENRICH is off, and off
        # for one where it is on), so the enricher is built whenever some
        # request could need it and ``EnrichmentStage`` gates each run on
        # ``RunConfig.enrichment_enabled``. A request can also override a
        # deployment whose reference parser defaults to off, so that default
        # must not permanently remove the enricher.
        from bibr.pipeline.enricher import CrossrefEnricher, RorEnricher

        enrichers = [CrossrefEnricher(settings=settings_snapshot)]
        if settings_snapshot.ror.enrich:
            enrichers.append(RorEnricher(settings=settings_snapshot))

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
