"""BibrPipelineAPI — LitServe inference API wrapping the bibr pipeline.

Delegates to :class:`bibr.serve.pipeline.ServePipeline`. This module owns
the HTTP surface (multipart decode, error-code mapping, Redis response
cache) and nothing else.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import litserve as ls
from fastapi import HTTPException

from bibr.config import GlobalSettings, snapshot_settings
from bibr.exceptions import ProcessingError, SafeLlmDiagnostics
from bibr.serve.ingress import UploadIntegrityError, consume_upload_descriptor
from bibr.serve.jobs import _sanitize_request_id
from bibr.serve.logsetup import configure_worker_logging

if TYPE_CHECKING:
    from bibr.cache import ResponseCache

logger = logging.getLogger(__name__)

# Per-extraction usage metering (D2). One JSON line per extraction on the shared
# ``bibr.serve.metering`` logger (same logger the HTTP middleware in serve.app
# uses for per-request records).
_metering_logger = logging.getLogger("bibr.serve.metering")


def _link_id(raw: object) -> str | None:
    """Sanitize an optional linkage id from an upload descriptor.

    The API wrote these ids itself (request id, job id); anything unexpected
    becomes None rather than failing the request.
    """
    return _sanitize_request_id(raw) if isinstance(raw, str) else None


def _emit_extract_metric(
    *,
    file_hash: str,
    filename: str,
    duration_ms: int,
    cache_hit: bool,
    success: bool,
    error_kind: str | None,
    result_json: dict | None,
    settings: GlobalSettings,
    error_code: str | None = None,
    safe_diagnostics: SafeLlmDiagnostics | None = None,
    request_id: str | None = None,
    job_id: str | None = None,
) -> None:
    """Emit one ``event="extract"`` JSON line for an extraction attempt.

    Gated on ``METER_ENABLED``. ``llm_usage_totals``/``llm_tokens_total`` are
    reported only for fresh (non-cache) successes — a cache hit spends no LLM
    tokens, so reporting the cached run's usage would double-count spend.

    ``llm_usage_totals`` is flat pipeline-wide totals
    (``calls``/``input_tokens``/``cached_input_tokens``/``output_tokens``/
    ``total_tokens``). It replaces the pre-v11 ``llm_usage`` key, which was
    keyed by model; the key was renamed rather than reshaped so consumers fail
    loudly on a missing key instead of silently misreading a changed one.
    """
    if not settings.metering.enabled:
        return
    llm_usage_totals: dict | None = None
    tokens_total: int | None = None
    failure_diagnostics: dict[str, object] | None = None
    if success and not cache_hit and result_json:
        # v11: per-paper LLM spend is ``extraction.usage.totals`` (already
        # aggregated over every (label, provider, model) row).
        llm_usage_totals = ((result_json.get("extraction") or {}).get("usage") or {}).get(
            "totals"
        ) or None
        if llm_usage_totals:
            try:
                tokens_total = int(llm_usage_totals.get("total_tokens", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                tokens_total = None
    elif not success and type(safe_diagnostics) is SafeLlmDiagnostics:
        failure_diagnostics = safe_diagnostics.to_dict()
        tokens_total = safe_diagnostics.total_tokens
    _metering_logger.info(
        json.dumps(
            {
                "event": "extract",
                "request_id": request_id,
                "job_id": job_id,
                "file_hash": file_hash,
                "filename": filename,
                "duration_ms": duration_ms,
                "cache_hit": cache_hit,
                "success": success,
                "error_kind": error_kind,
                "error_code": error_code,
                "llm_usage_totals": llm_usage_totals,
                "llm_tokens_total": tokens_total,
                "llm_failure_diagnostics": failure_diagnostics,
            }
        )
    )


def _emit_singleflight_metric(
    outcome: str,
    *,
    cache_key: str,
    settings: GlobalSettings,
) -> None:
    """Record distributed-flight outcomes without exposing document contents."""
    if not settings.metering.enabled:
        return
    _metering_logger.info(
        json.dumps(
            {
                "event": "cache_singleflight",
                "outcome": outcome,
                "cache_key": cache_key,
            }
        )
    )


_ERROR_KIND_TO_HTTP = {
    "input_validation": 400,
    "upstream_service": 502,
    "processing": 422,
    "unexpected": 500,
}

# Per-request reference-strategy overrides accepted on the wire. Mirror the
# CLI (`--refs` / `--ref-seg`) and library (`chew(refs=, ref_seg=)`) choices so
# a caller can, e.g., request `refs=off` for a fast core-metadata-only pass or
# `refs=llm` for full-precision parsing without redeploying the server.
_VALID_REF_PARSE = frozenset({"ner", "llm", "llm-chunked", "off"})
_VALID_REF_SEG = frozenset({"geom", "region", "llm", "crf"})

# Default in-flight LLM concurrency for serve. The library default
# (LLM_MAX_CONCURRENCY=0 = unlimited) suits CLI/cloud batch, but a single
# high-fan-out paper (many refs/sections parsed in parallel) can otherwise
# fire dozens of concurrent calls and self-induce provider 503 storms.
SERVE_DEFAULT_LLM_CONCURRENCY = 6


class _NullAsyncGate:
    """No-op async context manager used when the in-flight cap is disabled.

    A single shared instance is safe to enter concurrently — it holds no
    state — so ``predict`` can ``async with`` it without per-call allocation.
    """

    async def __aenter__(self) -> _NullAsyncGate:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


_NULL_GATE = _NullAsyncGate()


@dataclasses.dataclass
class _FlightEntry:
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    users: int = 0


class _KeyedAsyncLockPool:
    """Reference-counted per-key locks without an unbounded key registry."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._entries: dict[str, _FlightEntry] = {}

    @asynccontextmanager
    async def hold(self, key: str):
        async with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _FlightEntry()
                self._entries[key] = entry
            entry.users += 1

        try:
            await entry.lock.acquire()
            try:
                yield
            finally:
                entry.lock.release()
        finally:
            async with self._guard:
                entry.users -= 1
                if entry.users == 0 and self._entries.get(key) is entry:
                    self._entries.pop(key, None)


def _apply_serve_llm_concurrency_default(settings) -> bool:
    """Bound in-flight LLM requests for serve when the operator hasn't set
    ``LLM_MAX_CONCURRENCY``.

    Respects an explicit value (e.g. ``=1`` for a local single-device LLM
    server). Returns ``True`` when a default was applied.
    """
    if "max_concurrency" in settings.llm.model_fields_set:
        return False
    settings.llm.max_concurrency = SERVE_DEFAULT_LLM_CONCURRENCY
    return True


def _resolve_layout_device(device: str | None, *, use_gpu: bool | None) -> str | None:
    """Resolve the torch device string for the serve ``LayoutDetector``.

    LitServe's ``accelerator="auto"`` only returns a GPU backend when torch is
    already imported in the *master* process (see litserve
    ``Connector._choose_auto_accelerator``). bibr lazy-imports torch inside the
    worker's ``setup()``, so ``device`` arrives as ``"cpu"`` even on a GPU box —
    and pinning PP-DocLayoutV3 there is the dramatically-slower path. So a CPU
    verdict from LitServe is treated as "auto-detect", not "force CPU":

    - ``use_gpu is False`` → ``"cpu"`` (force CPU on VRAM-tight, OCR-co-located
      boxes; mirrors ``SEGMENTER_USE_GPU``).
    - LitServe gave ``None``/``"auto"``/``"cpu"`` → ``None`` so the detector
      auto-detects via the shared ``cuda→mps→cpu`` ladder.
    - an explicit non-CPU device (e.g. ``"cuda:1"`` from a multi-GPU LitServe,
      or ``"mps"``) is honored as-is.
    """
    if use_gpu is False:
        return "cpu"
    if device in (None, "auto", "cpu"):
        return None
    return device


def _release_llm_server(resources) -> None:
    """Sync teardown for weakref.finalize — releases vllm-mlx if running."""
    try:
        resources.shutdown_llm_server()
    except Exception:  # noqa: BLE001, S110  # finalizers must not raise
        pass


def _raise_for_required_classifier_failure(statuses) -> None:
    from bibr.pipeline.classifier_resources import ClassifierState

    failures = [
        f"{name}: {status.error or 'load failed'}"
        for name, status in statuses.items()
        if status.state is ClassifierState.FAILED_REQUIRED
    ]
    if failures:
        raise RuntimeError("Required classifier startup failed: " + "; ".join(failures))


class BibrPipelineAPI(ls.LitAPI):
    """Full bibr extraction pipeline exposed as a single LitServe API.

    Consumes disk-backed upload descriptors produced by ``bibr.serve.ingress``.
    HTTP multipart parsing remains owned by the public ingress layer. Descriptor
    fields include:
      - ``upload_id`` (required): opaque owned-file identifier
      - ``filename`` / ``size`` / ``sha256`` (required): verified upload metadata
      - ``start_page`` (optional int, 0-indexed inclusive)
      - ``end_page``   (optional int, 0-indexed inclusive)
      - ``include_figures`` (optional bool, default false)
      - ``include_regions`` (optional bool, default false): emit the
        ``extraction.regions`` layout debug payload; off by default since standard
        consumers (Metacheck) don't read it.
      - ``crossref`` (optional bool): run Crossref/resolver reference
        enrichment for this request (``true``) or skip it (``false``);
        absent → the server-side CROSSREF_ENRICH setting, which is off by
        default. The response cache keys on the effective value, so an
        enriched and an unenriched result never collide.
      - ``consolidate`` (optional, ``fill``/``replace``): merge accepted
        bib_match data into bib rows (absent → server-side
        CROSSREF_CONSOLIDATE setting; no per-request 'off' override)
      - ``refs`` (optional, ``ner``/``llm``/``llm-chunked``/``off``): per-request
        reference parse strategy (absent → server-side REF_PARSE_STRATEGY).
        ``off`` skips reference extraction entirely for a fast core-metadata pass.
      - ``ref_seg`` (optional, ``geom``/``region``/``llm``/``crf``): per-request
        reference segmentation strategy (absent → server-side REF_SEG_STRATEGY).
    """

    def __init__(
        self,
        *args,
        upload_root: str | Path,
        settings: GlobalSettings | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._settings = snapshot_settings(settings)
        self._upload_root = str(upload_root)

    def setup(self, device: str) -> None:
        import httpx

        # This runs in the spawned inference worker, a fresh interpreter with no
        # logging configured: give it the serve sink and the metering sink.
        configure_worker_logging(self._settings)

        from bibr.serve.deployments.layout import LayoutDetector
        from bibr.serve.deployments.segmenter import SentenceSegmenter
        from bibr.serve.pipeline import ServePipeline
        from bibr.utils.circuit_breaker import AsyncCircuitBreaker

        # Bound LLM concurrency for serve unless the operator set it explicitly
        # — unbounded fan-out on a big paper self-induces provider 503 storms.
        if _apply_serve_llm_concurrency_default(self._settings):
            logger.info(
                "LLM concurrency defaulted to %d for serve (override with LLM_MAX_CONCURRENCY)",
                SERVE_DEFAULT_LLM_CONCURRENCY,
            )

        # Cap concurrent in-flight pipeline runs per worker (0 = unlimited) so an
        # upload flood can't exhaust host RAM with page images. GPU peak is
        # bounded separately by the layout/segmenter GpuBatcher.
        n_inflight = self._settings.pipeline.max_inflight_requests
        self._inflight_sem = asyncio.Semaphore(n_inflight) if n_inflight > 0 else None

        # Don't trust LitServe's device verdict for the GPU-capable layout model:
        # accelerator="auto" resolves to "cpu" when torch isn't pre-imported in
        # the master (which bibr never does), so this would otherwise pin layout
        # to CPU on a GPU box. Auto-detect in the worker instead (see
        # _resolve_layout_device); LAYOUT_USE_GPU=false forces CPU.
        layout_device = _resolve_layout_device(device, use_gpu=self._settings.layout.use_gpu)
        self.layout = LayoutDetector(device=layout_device, settings=self._settings)
        # Segmenter auto-detects the GPU (SEGMENTER_USE_GPU=None). CPU
        # segmentation is the dominant cost on large papers (~197s vs ~0.7s on
        # GPU); set SEGMENTER_USE_GPU=false to force CPU on VRAM-tight boxes.
        self.segmenter = SentenceSegmenter(
            use_gpu=self._settings.SEGMENTER_USE_GPU,
            settings=self._settings,
        )

        self._ocr_breaker = AsyncCircuitBreaker(
            failure_threshold=self._settings.cb.failure_threshold,
            reset_timeout=self._settings.cb.reset_timeout_seconds,
            name="ocr",
            failure_dedup_window=self._settings.cb.failure_dedup_window,
        )
        # Genuinely server-wide: bibr serve pins one inference worker, so this
        # semaphore is the only gate between the process and the OCR endpoint.
        self._ocr_sem = asyncio.Semaphore(self._settings.ocr.max_concurrent_regions)
        self.ocr_base_url = self._settings.OCR_BASE_URL.rstrip("/")

        from bibr.ocr.http_security import ocr_request_headers

        ocr_headers = ocr_request_headers(
            self.ocr_base_url,
            self._settings.ocr.api_key,
            allow_insecure_http=self._settings.ocr.allow_insecure_http,
        )

        self._http_client = httpx.AsyncClient(
            headers=ocr_headers,
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(self._settings.ocr.request_timeout),
                write=30.0,
                pool=10.0,
            ),
            http2=True,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=50,
                keepalive_expiry=30.0,
            ),
        )

        # Build the pipeline once per worker. Per-request fields (start_page /
        # end_page / include_figures) ride along on a fresh RunConfig in
        # ``predict``, not in the pipeline itself.
        self._pipeline = ServePipeline(
            layout=self.layout,
            segmenter=self.segmenter,
            http_client=self._http_client,
            ocr_base_url=self.ocr_base_url,
            ocr_sem_global=self._ocr_sem,
            ocr_breaker=self._ocr_breaker,
            ocr_profile=self._settings.ocr.profile,
            settings=self._settings,
        )
        # LitAPI setup runs before the worker accepts requests. Loading here
        # verifies baked artifacts and prevents first-request downloads.
        classifier_start = self._pipeline._resources.start_classifiers()
        if asyncio.iscoroutine(classifier_start):
            asyncio.run(classifier_start)
        _raise_for_required_classifier_failure(self._pipeline._resources.classifiers.status())

        self._cache: ResponseCache | None = None
        self._cache_inited = False
        self._cache_flights = _KeyedAsyncLockPool()

        # Best-effort cleanup of the long-lived vllm-mlx subprocess on worker
        # exit. This runs at GC or interpreter shutdown via weakref.finalize.
        # Async resources (httpx, OCR HTTP client) are reclaimed by OS process
        # exit — there's no reliable way to await them from a finalizer.
        self._finalizer = weakref.finalize(
            self,
            _release_llm_server,
            self._pipeline._resources,
        )

        logger.info(
            "BibrPipelineAPI ready (ocr via HTTP at %s, OCR sem=%d)",
            self.ocr_base_url,
            self._settings.ocr.max_concurrent_regions,
        )

    def _ensure_cache(self) -> None:
        if self._cache_inited:
            return
        self._cache_inited = True
        try:
            from bibr.cache import ResponseCache

            if self._settings.cache.enabled and self._settings.redis.url:
                from bibr.config import cache_namespace

                self._cache = ResponseCache(
                    redis_url=self._settings.redis.url,
                    ttl_seconds=self._settings.cache.ttl_seconds,
                    connect_timeout=self._settings.redis.connect_timeout_seconds,
                    socket_timeout=self._settings.redis.socket_timeout_seconds,
                    # Code hash + behavior fingerprint: a config change (model,
                    # ref strategy, enrichment mode...) invalidates the cache
                    # the same way a code change does.
                    prefix=cache_namespace(self._settings),
                )
                if self._settings.cache.ttl_seconds > 0:
                    logger.info(
                        "Response cache enabled (TTL=%ds)",
                        self._settings.cache.ttl_seconds,
                    )
                else:
                    logger.info("Response cache enabled (no expiry)")
        except Exception as e:
            logger.warning("Response cache not available: %s", e)

    async def decode_request(self, request: dict) -> dict:
        """Consume a verified private upload descriptor into pipeline inputs."""
        try:
            content, content_hash = await asyncio.to_thread(
                consume_upload_descriptor,
                self._upload_root,
                request,
                max_size=self._settings.pipeline.max_file_size,
            )
        except UploadIntegrityError:
            raise HTTPException(status_code=500, detail="Upload handoff failed") from None

        # Bound the descriptor filename before it reaches metering/logs (audit
        # L5); 255 matches the common filesystem limit. Task 2 validates
        # descriptor filename shape; this remains a defensive normalization.
        filename = str(request.get("filename") or "")[:255]
        if not filename:
            raise HTTPException(status_code=400, detail="Filename required")

        def _opt_int(name: str, v):
            if v is None or v == "":
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"{name} must be an integer") from None

        start_page = _opt_int("start_page", request.get("start_page"))
        end_page = _opt_int("end_page", request.get("end_page"))
        if start_page is not None and start_page < 0:
            raise HTTPException(status_code=400, detail="start_page must be >= 0")
        if end_page is not None and end_page < 0:
            raise HTTPException(status_code=400, detail="end_page must be >= 0")
        if start_page is not None and end_page is not None and start_page > end_page:
            raise HTTPException(
                status_code=400,
                detail=f"start_page ({start_page}) must be <= end_page ({end_page})",
            )

        def _opt_bool(name: str, v: object) -> bool:
            normalized = str(v).lower()
            if normalized in ("true", "1", "yes"):
                return True
            if normalized in ("false", "0", "no"):
                return False
            raise HTTPException(
                status_code=400,
                detail=f"{name} must be a boolean (true/false, 1/0, yes/no)",
            )

        include_figures = _opt_bool("include_figures", request.get("include_figures", "false"))
        include_regions = _opt_bool("include_regions", request.get("include_regions", "false"))
        # Tri-state: absent/empty → None (defer to CROSSREF_ENRICH at run time).
        crossref_raw = request.get("crossref")
        crossref = None if crossref_raw in (None, "") else _opt_bool("crossref", crossref_raw)

        consolidate = request.get("consolidate")
        if consolidate in (None, ""):
            consolidate = None
        else:
            consolidate = str(consolidate).lower()
            if consolidate not in ("fill", "replace"):
                raise HTTPException(
                    status_code=400, detail="consolidate must be 'fill' or 'replace'"
                )

        refs = self._opt_choice(
            request.get("refs"),
            _VALID_REF_PARSE,
            "refs must be one of: " + ", ".join(sorted(_VALID_REF_PARSE)),
        )
        ref_seg = self._opt_choice(
            request.get("ref_seg"),
            _VALID_REF_SEG,
            "ref_seg must be one of: " + ", ".join(sorted(_VALID_REF_SEG)),
        )

        return {
            "filename": filename,
            "content": content,
            "content_hash": content_hash,
            "start_page": start_page,
            "end_page": end_page,
            "include_figures": include_figures,
            "include_regions": include_regions,
            "crossref": crossref,
            "consolidate": consolidate,
            "refs": refs,
            "ref_seg": ref_seg,
            "request_id": _link_id(request.get("request_id")),
            "job_id": _link_id(request.get("job_id")),
        }

    @staticmethod
    def _opt_choice(value: object, valid: frozenset[str], error: str) -> str | None:
        """Lower-case + validate an optional enum form field. Absent → None."""
        if value in (None, ""):
            return None
        normalized = str(value).lower()
        if normalized not in valid:
            raise HTTPException(status_code=400, detail=error)
        return normalized

    async def predict(self, inputs: dict) -> dict:
        self._ensure_cache()

        filename = inputs["filename"]
        content = inputs["content"]
        start_page = inputs["start_page"]
        end_page = inputs["end_page"]
        include_figures = inputs["include_figures"]
        include_regions = inputs["include_regions"]
        # Effective mode (request field wins, else server-side setting) is used
        # for the cache key so cached entries can't leak across deployments that
        # differ only in CROSSREF_CONSOLIDATE. The RunConfig keeps the raw
        # request value — None defers to the setting at export time.
        consolidate = inputs["consolidate"]
        effective_consolidate = consolidate or self._settings.crossref.consolidate
        # Same for enrichment: the cache key carries the *effective* switch so
        # a ``crossref=true`` request never reads back a cached unenriched
        # result (or vice versa) under the same file hash.
        effective_crossref = self._effective_crossref(inputs.get("crossref"))
        # Per-request ref-strategy overrides (None → server-side settings). The
        # RunConfig keeps the raw request values (None defers at run time); the
        # cache key uses the *effective* resolved strategies so a refs=off pass
        # can't collide with a full-reference result under the same file hash.
        refs = inputs.get("refs")
        ref_seg = inputs.get("ref_seg")
        from bibr.extract.ref_extractor import _resolve_ref_strategies

        eff_ref_seg, eff_refs = _resolve_ref_strategies(ref_seg, refs)

        # decode_request computes this while streaming. Direct embedders that
        # bypass decode retain a compatibility fallback.
        digest: str | None = inputs.get("content_hash")
        if digest is None:
            import hashlib

            digest = await asyncio.to_thread(lambda: hashlib.sha256(content).hexdigest())
        file_hash = digest[:16]
        # The cache is shared by every caller, so it is keyed on the full
        # SHA-256: 64 bits can be collided on purpose, and the colliding upload
        # would then be answered with the first one's extraction. The extension
        # picks the parser, so the same bytes as .html and .xml are two results.
        cache_key = self._cache_key(
            digest,
            start_page,
            end_page,
            include_figures,
            include_regions,
            effective_consolidate,
            refs=eff_refs,
            ref_seg=eff_ref_seg,
            crossref=effective_crossref,
            input_format=Path(filename).suffix,
        )

        if not self._cache:
            return await self._run_pipeline(inputs, file_hash=file_hash, cache_key=cache_key)

        cache_start = time.perf_counter()
        cached_response = await self._read_cached_response(
            cache_key,
            file_hash=file_hash,
            filename=filename,
            started_at=cache_start,
            request_id=inputs.get("request_id"),
            job_id=inputs.get("job_id"),
        )
        if cached_response is not None:
            return cached_response

        # Coalesce only identical misses. The second lookup after acquiring the
        # key lock lets waiters consume the first request's newly cached result;
        # unrelated keys retain full concurrency. Entries are reference-counted
        # and removed after the final waiter, bounding registry memory.
        flights = getattr(self, "_cache_flights", None)
        if flights is None:  # compatibility for tests/embedders that bypass setup()
            flights = _KeyedAsyncLockPool()
            self._cache_flights = flights
        async with flights.hold(cache_key):
            cached_response = await self._read_cached_response(
                cache_key,
                file_hash=file_hash,
                filename=filename,
                started_at=cache_start,
                request_id=inputs.get("request_id"),
                job_id=inputs.get("job_id"),
            )
            if cached_response is not None:
                return cached_response
            return await self._run_distributed_singleflight(
                inputs,
                file_hash=file_hash,
                cache_key=cache_key,
                filename=filename,
                started_at=cache_start,
            )

    async def _run_distributed_singleflight(
        self,
        inputs: dict,
        *,
        file_hash: str,
        cache_key: str,
        filename: str,
        started_at: float,
    ) -> dict:
        """Coordinate cache misses across workers while remaining fail-open."""
        if not self._settings.cache.distributed_singleflight:
            _emit_singleflight_metric(
                "disabled",
                cache_key=cache_key,
                settings=self._settings,
            )
            return await self._run_pipeline(inputs, file_hash=file_hash, cache_key=cache_key)

        assert self._cache is not None  # predict() returns early without a cache
        try:
            lease = await asyncio.wait_for(
                self._cache.try_acquire_lease(
                    cache_key,
                    ttl_seconds=self._settings.cache.singleflight_lease_ttl_seconds,
                ),
                timeout=self._settings.cache.operation_timeout_seconds,
            )
        except Exception:
            logger.warning(
                "Distributed cache lease unavailable; extracting normally", exc_info=True
            )
            _emit_singleflight_metric(
                "redis_error_fallback",
                cache_key=cache_key,
                settings=self._settings,
            )
            return await self._run_pipeline(inputs, file_hash=file_hash, cache_key=cache_key)

        if lease is not None:
            _emit_singleflight_metric(
                "owner",
                cache_key=cache_key,
                settings=self._settings,
            )
            renew_task = asyncio.create_task(self._renew_cache_lease(lease, cache_key))
            try:
                return await self._run_pipeline(inputs, file_hash=file_hash, cache_key=cache_key)
            finally:
                renew_task.cancel()
                await asyncio.gather(renew_task, return_exceptions=True)
                try:
                    await asyncio.wait_for(
                        lease.release(),
                        timeout=self._settings.cache.operation_timeout_seconds,
                    )
                except Exception:
                    logger.warning("Failed to release distributed cache lease", exc_info=True)

        # A real extraction runs up to PIPELINE_TIMEOUT (plus queueing on the
        # in-flight semaphore, which the owner holds its lease through), so a
        # waiter that gives up after the flat singleflight_wait_seconds almost
        # always pays the wait AND a duplicate extraction. Unless the operator
        # set an explicit wait, the budget follows the pipeline timeout; the
        # owner's lease (renewed every 30 s against a 120 s TTL) is the
        # liveness signal, and its disappearance means the owner died without
        # publishing — take over at once instead of waiting out the budget.
        if "singleflight_wait_seconds" in self._settings.cache.model_fields_set:
            wait_budget = self._settings.cache.singleflight_wait_seconds
        else:
            wait_budget = float(self._settings.pipeline.timeout)
        deadline = time.monotonic() + wait_budget
        poll_seconds = self._settings.cache.singleflight_poll_interval_ms / 1000
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_seconds)
            try:
                cached_response = await self._read_cached_response(
                    cache_key,
                    file_hash=file_hash,
                    filename=filename,
                    started_at=started_at,
                    request_id=inputs.get("request_id"),
                    job_id=inputs.get("job_id"),
                )
            except Exception:
                logger.warning("Distributed cache wait failed; extracting normally", exc_info=True)
                _emit_singleflight_metric(
                    "redis_error_fallback",
                    cache_key=cache_key,
                    settings=self._settings,
                )
                return await self._run_pipeline(inputs, file_hash=file_hash, cache_key=cache_key)
            if cached_response is not None:
                _emit_singleflight_metric(
                    "waiter_hit",
                    cache_key=cache_key,
                    settings=self._settings,
                )
                return cached_response
            try:
                lease_alive = await asyncio.wait_for(
                    self._cache.lease_alive(cache_key),
                    timeout=self._settings.cache.operation_timeout_seconds,
                )
            except Exception:
                logger.warning(
                    "Distributed cache lease check failed; extracting normally", exc_info=True
                )
                _emit_singleflight_metric(
                    "redis_error_fallback",
                    cache_key=cache_key,
                    settings=self._settings,
                )
                return await self._run_pipeline(inputs, file_hash=file_hash, cache_key=cache_key)
            if not lease_alive:
                # The owner's publish and lease release may both have landed
                # between this poll's cache read and the lease check — re-read
                # once before falling back, or a result already in Redis pays
                # for a duplicate extraction.
                try:
                    cached_response = await self._read_cached_response(
                        cache_key,
                        file_hash=file_hash,
                        filename=filename,
                        started_at=started_at,
                        request_id=inputs.get("request_id"),
                        job_id=inputs.get("job_id"),
                    )
                except Exception:
                    logger.warning(
                        "Distributed cache wait failed; extracting normally", exc_info=True
                    )
                    _emit_singleflight_metric(
                        "redis_error_fallback",
                        cache_key=cache_key,
                        settings=self._settings,
                    )
                    return await self._run_pipeline(
                        inputs, file_hash=file_hash, cache_key=cache_key
                    )
                if cached_response is not None:
                    _emit_singleflight_metric(
                        "waiter_hit",
                        cache_key=cache_key,
                        settings=self._settings,
                    )
                    return cached_response
                # Only one waiter takes over: the lease acquisition is atomic,
                # so losers keep polling for the winner's result instead of
                # every waiter extracting at once.
                try:
                    takeover = await asyncio.wait_for(
                        self._cache.try_acquire_lease(
                            cache_key,
                            ttl_seconds=self._settings.cache.singleflight_lease_ttl_seconds,
                        ),
                        timeout=self._settings.cache.operation_timeout_seconds,
                    )
                except Exception:
                    logger.warning(
                        "Distributed cache lease takeover failed; extracting normally",
                        exc_info=True,
                    )
                    _emit_singleflight_metric(
                        "redis_error_fallback",
                        cache_key=cache_key,
                        settings=self._settings,
                    )
                    return await self._run_pipeline(
                        inputs, file_hash=file_hash, cache_key=cache_key
                    )
                if takeover is None:
                    continue
                _emit_singleflight_metric(
                    "owner_gone_fallback",
                    cache_key=cache_key,
                    settings=self._settings,
                )
                renew_task = asyncio.create_task(self._renew_cache_lease(takeover, cache_key))
                try:
                    return await self._run_pipeline(
                        inputs, file_hash=file_hash, cache_key=cache_key
                    )
                finally:
                    renew_task.cancel()
                    await asyncio.gather(renew_task, return_exceptions=True)
                    try:
                        await asyncio.wait_for(
                            takeover.release(),
                            timeout=self._settings.cache.operation_timeout_seconds,
                        )
                    except Exception:
                        logger.warning("Failed to release distributed cache lease", exc_info=True)

        _emit_singleflight_metric(
            "timeout_fallback",
            cache_key=cache_key,
            settings=self._settings,
        )
        return await self._run_pipeline(inputs, file_hash=file_hash, cache_key=cache_key)

    async def _renew_cache_lease(self, lease, cache_key: str) -> None:
        """Keep an owned lease alive; losing it never cancels extraction."""
        interval = self._settings.cache.singleflight_renew_interval_seconds
        while True:
            await asyncio.sleep(interval)
            try:
                if not await lease.renew():
                    _emit_singleflight_metric(
                        "lease_lost",
                        cache_key=cache_key,
                        settings=self._settings,
                    )
                    return
            except Exception:
                logger.warning("Failed to renew distributed cache lease", exc_info=True)
                _emit_singleflight_metric(
                    "lease_lost",
                    cache_key=cache_key,
                    settings=self._settings,
                )
                return

    async def _bounded_cache_call(self, operation, *, what: str):
        """Run one cache coroutine under ``CACHE_OPERATION_TIMEOUT_SECONDS``.

        The Redis client already carries socket timeouts; this is the second
        fence, so a misconfigured or wedged connection pool degrades to a cache
        miss instead of holding the request (and its admission slot) forever.
        """
        timeout = self._settings.cache.operation_timeout_seconds
        try:
            return await asyncio.wait_for(operation, timeout=timeout)
        except TimeoutError:
            logger.warning(
                "Response cache %s took longer than %.1fs; continuing without the cache",
                what,
                timeout,
            )
            return None

    async def _read_cached_response(
        self,
        cache_key: str,
        *,
        file_hash: str,
        filename: str,
        started_at: float,
        request_id: str | None = None,
        job_id: str | None = None,
    ) -> dict | None:
        cached = await self._bounded_cache_call(self._cache.get(cache_key), what="get")
        if not cached:
            return None
        # Deserialize off-loop: a multi-MB json.loads shouldn't stall other
        # in-flight requests on the worker's event loop.
        try:
            paper_json = await asyncio.to_thread(json.loads, cached)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            logger.warning("Corrupt response cache entry %s; deleting", cache_key, exc_info=True)
            await self._bounded_cache_call(self._cache.delete(cache_key), what="delete")
            return None
        # v11: the input artifact's identity lives at the root ``source`` block.
        source = paper_json.get("source") if isinstance(paper_json, dict) else None
        if isinstance(source, dict):
            source["file_name"] = filename
        _emit_extract_metric(
            file_hash=file_hash,
            filename=filename,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            cache_hit=True,
            success=True,
            error_kind=None,
            result_json=paper_json,
            settings=self._settings,
            request_id=request_id,
            job_id=job_id,
        )
        return {
            "success": True,
            "paper_json": paper_json,
            "error": None,
            "error_kind": None,
        }

    async def _run_pipeline(self, inputs: dict, *, file_hash: str, cache_key: str) -> dict:
        filename = inputs["filename"]
        content = inputs["content"]
        start_page = inputs["start_page"]
        end_page = inputs["end_page"]
        include_figures = inputs["include_figures"]
        include_regions = inputs["include_regions"]
        consolidate = inputs["consolidate"]
        refs = inputs.get("refs")
        ref_seg = inputs.get("ref_seg")
        crossref = inputs.get("crossref")

        # Per-request fields ride on a fresh RunConfig; the pipeline is shared.
        overrides: dict[str, object] = {
            "start_page": start_page,
            "end_page": end_page,
            "include_figures": include_figures,
            "include_regions": include_regions,
            "consolidate": consolidate,
            "ref_seg_strategy": ref_seg,
            "ref_parse_strategy": refs,
        }
        # Only an explicit request value replaces the deployment's tri-state
        # ``crossref``; an absent field keeps whatever the pipeline was built
        # with (normally None → CROSSREF_ENRICH).
        if crossref is not None:
            overrides["crossref"] = crossref
        config = dataclasses.replace(self._pipeline._config, **overrides)

        # Gate entry into the expensive pipeline run. The async loop dispatches
        # unbounded concurrent predicts per worker; without this, a flood of
        # uploads would all render page images at once and exhaust host RAM.
        # The slot is released as soon as process_file returns (cache write +
        # encode happen outside the gate).
        inflight_sem = getattr(self, "_inflight_sem", None)
        gate = inflight_sem if inflight_sem is not None else _NULL_GATE
        run_start = time.perf_counter()
        try:
            async with gate:
                process_kwargs = {
                    "paper_id": file_hash,
                    "content": content,
                    "config": config,
                }
                if inputs.get("content_hash") is not None:
                    process_kwargs["content_hash"] = inputs["content_hash"]
                result_json = await asyncio.wait_for(
                    self._pipeline.process_file(filename, **process_kwargs),
                    timeout=self._settings.pipeline.timeout,
                )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:  # noqa: UP041  # forward-compat: keep prefixed form
            _emit_extract_metric(
                file_hash=file_hash,
                filename=filename,
                duration_ms=int((time.perf_counter() - run_start) * 1000),
                cache_hit=False,
                success=False,
                error_kind="timeout",
                result_json=None,
                settings=self._settings,
                request_id=inputs.get("request_id"),
                job_id=inputs.get("job_id"),
            )
            raise HTTPException(status_code=504, detail="Pipeline processing timed out") from None
        except HTTPException:
            raise
        except Exception as e:
            failure = self._translate_error(filename, e)
            safe_diagnostics = e.safe_diagnostics if isinstance(e, ProcessingError) else None
            _emit_extract_metric(
                file_hash=file_hash,
                filename=filename,
                duration_ms=int((time.perf_counter() - run_start) * 1000),
                cache_hit=False,
                success=False,
                error_kind=failure.get("error_kind"),
                result_json=None,
                settings=self._settings,
                error_code=failure.get("error_code"),
                safe_diagnostics=safe_diagnostics,
                request_id=inputs.get("request_id"),
                job_id=inputs.get("job_id"),
            )
            return failure

        run_duration_ms = int((time.perf_counter() - run_start) * 1000)
        if self._cache:
            # Serialize off-loop: a multi-MB json.dumps shouldn't stall other
            # in-flight requests on the worker's event loop.
            encoded = await asyncio.to_thread(lambda: json.dumps(result_json).encode())
            await self._bounded_cache_call(self._cache.set(cache_key, encoded), what="set")

        _emit_extract_metric(
            file_hash=file_hash,
            filename=filename,
            duration_ms=run_duration_ms,
            cache_hit=False,
            success=True,
            error_kind=None,
            result_json=result_json,
            settings=self._settings,
            request_id=inputs.get("request_id"),
            job_id=inputs.get("job_id"),
        )
        return {
            "success": True,
            "paper_json": result_json,
            "error": None,
            "error_kind": None,
        }

    async def encode_response(self, output: dict):
        if not output["success"]:
            status = _ERROR_KIND_TO_HTTP.get(output.get("error_kind") or "", 500)
            message = output.get("error") or "Internal server error"
            error_code = output.get("error_code")
            detail = {"message": message, "error_code": error_code} if error_code else message
            raise HTTPException(
                status_code=status,
                detail=detail,
            )
        # The producing bibr version is carried in the export schema at
        # ``extraction.producer.version`` (set at export time, so it reflects
        # the bibr that produced the result even on cache hits). No serve-side
        # stamp.
        return output["paper_json"]

    def _effective_crossref(self, requested: bool | None) -> bool:
        """Resolve a request's tri-state ``crossref`` the way the pipeline will."""
        if requested is not None:
            return bool(requested)
        pipeline = getattr(self, "_pipeline", None)
        config = getattr(pipeline, "_config", None)
        deployment_default = getattr(config, "crossref", None)
        if deployment_default is not None:
            return bool(deployment_default)
        return bool(self._settings.crossref.enrich)

    @staticmethod
    def _cache_key(
        content_hash: str,
        start_page: int | None,
        end_page: int | None,
        include_figures: bool,
        include_regions: bool,
        consolidate: str | None,
        refs: str | None = None,
        ref_seg: str | None = None,
        crossref: bool = False,
        input_format: str | None = None,
    ) -> str:
        """Response-cache key: the full SHA-256 of the upload (never the 16-hex
        ``file_hash`` display id) plus every option that changes the export."""
        key = f"json:{content_hash}"
        if start_page is not None:
            key += f":sp{start_page}"
        if end_page is not None:
            key += f":ep{end_page}"
        if include_figures:
            key += ":fig"
        if include_regions:
            key += ":reg"
        if consolidate and consolidate != "off":
            key += f":con:{consolidate}"
        if refs:
            key += f":refs:{refs}"
        if ref_seg:
            key += f":rseg:{ref_seg}"
        if crossref:
            key += ":enrich"
        if input_format:
            key += f":fmt:{input_format.lstrip('.').lower()}"
        return key

    @staticmethod
    def _translate_error(filename: str, exc: Exception) -> dict:
        """Map pipeline exceptions back to the wire error-kind taxonomy."""
        from bibr.exceptions import (
            InputValidationError,
            ProcessingError,
            UpstreamServiceError,
        )
        from bibr.utils.redact import redact_urls

        if isinstance(exc, InputValidationError):
            kind = "input_validation"
            message = str(exc)
        elif isinstance(exc, UpstreamServiceError):
            kind = "upstream_service"
            message = str(exc)
        elif isinstance(exc, ProcessingError):
            kind = "processing"
            message = str(exc)
        else:
            kind = "unexpected"
            message = "Internal processing error"
        error_code = exc.error_code if isinstance(exc, ProcessingError) else None
        safe_diagnostics = exc.safe_diagnostics if isinstance(exc, ProcessingError) else None
        if error_code:
            logger.error("[%s] %s (%s): %s", filename, kind, error_code, message)
        else:
            logger.exception("[%s] %s: %s", filename, kind, exc)
        return {
            "success": False,
            "paper_json": None,
            # The 4xx/5xx body and the job error reach the caller: they never
            # name an internal endpoint (the log line above keeps it).
            "error": redact_urls(message),
            "error_kind": kind,
            "error_code": error_code,
            "safe_diagnostics": (
                safe_diagnostics.to_dict() if type(safe_diagnostics) is SafeLlmDiagnostics else None
            ),
        }
