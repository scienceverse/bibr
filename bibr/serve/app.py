"""LitServe application entry point — ``bibr serve`` CLI.

Builds a single ``litserve.LitServer`` hosting the full bibr pipeline. The
server exposes:
  - ``POST /papers/extract``           — multipart upload, returns the bibr JSON schema
  - ``POST /papers/jobs``              — submit an async extraction job (202)
  - ``GET  /papers/jobs/{id}``         — poll job status
  - ``GET  /papers/jobs/{id}/result``  — fetch the paper_json once succeeded
  - ``GET  /health``                   — LitServe's built-in liveness check
  - ``GET  /ready``                    — custom readiness probe (OCR + Redis + job store)

Every response carries ``x-request-id`` (echoed from the request when a sane
one is supplied, else generated) and ``x-bibr-duration-ms``; one structured
JSON record per request is emitted on the ``bibr.serve.metering`` logger. Both
the async-job API (``JOBS_ENABLED``) and metering (``METER_ENABLED``) are
on by default and independently toggleable.
"""

import functools
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


#: How long a non-``ok`` classifier verdict stays cached on ``/ready``. A probe
#: that arrives while the worker is still downloading weights must not pin the
#: process at 503 forever; a cached ``ok`` is never re-probed on the request
#: path (no model load there), while failures are retried after this interval.
_CLASSIFIER_CHECK_RETRY_SECONDS = 30.0


def classifier_artifact_readiness(settings, *, snapshot_download=None) -> tuple[str, bool]:
    """Verify configured classifier snapshots exist locally without downloading."""
    if snapshot_download is None:
        from huggingface_hub import snapshot_download

    configured = (
        (
            settings.ml.paper_classifier_model_id,
            settings.ml.paper_classifier_revision,
        ),
        (
            settings.ml.section_classifier_model_id,
            settings.ml.section_classifier_revision,
        ),
    )
    try:
        for model_id, revision in configured:
            if not model_id:
                continue
            # A local directory or file counts as present when the
            # configured runtime could load from it (resolve_runtime)
            # without loading any model: torch loads the directory
            # itself, so the ONNX bundle is required only when
            # ML_RUNTIME=onnx. snapshot_download rejects
            # filesystem paths outright, so without this a baked-in
            # classifier is reported missing. Local-only: no Hub access,
            # so this never slows the readiness probe.
            if Path(str(model_id)).expanduser().exists():
                from bibr.utils.ml_runtime import find_onnx_bundle, resolve_runtime

                resolve_runtime(
                    "classifier",
                    settings=settings,
                    bundle=functools.partial(find_onnx_bundle, model_id, revision),
                    bundle_hint="publish an onnx/ bundle or run a torch runtime",
                )
                continue
            snapshot_download(model_id, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - readiness reports missing/corrupt cache
        if settings.ml.classifiers_required:
            return ("failed_required", False)
        return ("degraded", True)
    return ("ok", True)


def _request_metering_line(
    *, request_id: str, method: str, path: str, status: int, duration_ms: int, outstanding: int
) -> str:
    """One JSON metering line, shared by the normal and 500 request paths."""
    return json.dumps(
        {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": duration_ms,
            "inference_outstanding": outstanding,
        }
    )


def readiness_payload(
    status: str, checks: dict[str, str], settings, *, include_detail: bool = True
) -> dict[str, object]:
    """Build the readiness body.

    ``/ready`` is unauthenticated (probes carry no credentials), so per-service
    health and the deployment ``build_sha`` are disclosed only to authenticated
    callers — anonymous callers get just the overall status (audit L1). Probes
    rely on the HTTP status code, not the body, so nothing operational is lost.
    """
    if not include_detail:
        return {"status": status}
    return {
        "status": status,
        "checks": checks,
        "build_sha": settings.BIBR_BUILD_SHA,
    }


# Metering records (one JSON line each) go through ``bibr.serve.logsetup``; the
# names stay importable from here for the middleware below and for tests.
from bibr.serve.logsetup import (  # noqa: E402
    configure_metering_logging as _configure_metering_logging,
)
from bibr.serve.logsetup import metering_logger  # noqa: E402


def _safe_cors_credentials(origins: list[str], allow_credentials: bool) -> bool:
    """Never combine wildcard origins with credentials.

    Starlette's CORSMiddleware, given ``allow_origins=["*"]`` and
    ``allow_credentials=True``, reflects the request ``Origin`` back — letting
    *any* site make credentialed cross-origin requests. Production already rejects
    ``*`` outright; this closes the same hole in dev mode (audit L3).
    """
    if allow_credentials and "*" in origins:
        logger.warning(
            "CORS: allow_credentials with wildcard origins would reflect any origin; "
            "disabling credentials. Set CORS_ORIGINS to explicit origins to allow them."
        )
        return False
    return allow_credentials


def _compose_lifespan_cleanup(app, cleanup) -> None:
    """Run router and bibr cleanup inside LitServe's custom app lifespan."""
    from contextlib import asynccontextmanager

    litserve_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _lifespan(lifespan_app):
        async with litserve_lifespan(lifespan_app):
            try:
                await lifespan_app.router._startup()
                yield
            finally:
                try:
                    await lifespan_app.router._shutdown()
                finally:
                    await cleanup()

    app.router.lifespan_context = _lifespan


def build_server():
    """Construct the configured LitServer and clean partial build resources."""
    upload_stores = []
    try:
        return _build_server(upload_stores)
    except BaseException:
        for upload_store in upload_stores:
            try:
                upload_store.close_sync()
            except OSError:
                logger.exception("Failed to clean upload root after server construction error")
        raise


def _build_server(upload_stores):
    """Construct the configured LitServer instance."""
    import litserve as ls
    import litserve.server as litserve_server
    from fastapi.middleware.cors import CORSMiddleware

    from bibr.config import Settings, validate_production_settings
    from bibr.serve.admission import base64_envelope
    from bibr.serve.auth import PUBLIC_PATHS, check_bearer
    from bibr.serve.deployments.pipeline import BibrPipelineAPI
    from bibr.serve.ingress import (
        INTERNAL_INFERENCE_PATH,
        InferenceDispatchTracker,
        UploadStore,
        configure_multipart_spooling,
        register_extract_route,
        resolve_litserve_dispatch,
    )
    from bibr.serve.paths import MCP_MOUNT_PATH

    # LitServe 0.2.17 gates its own MCP connector on the official `mcp` package
    # being importable (litserve.server._MCP_AVAILABLE) but builds that connector
    # from the third-party `fastmcp` package: litserve/mcp.py binds `MCPServer`
    # only when fastmcp is installed, so with bibr's `mcp` extra alone
    # `server.run()` dies with `NameError: name 'MCPServer' is not defined`
    # before uvicorn starts. bibr mounts its own endpoint (bibr.serve.mcp) and
    # never wants LitServe's, so switch the detection off for this process
    # regardless of what is installed. Must happen before the LitServer is
    # constructed and before run(): both read the module flag.
    litserve_server._MCP_AVAILABLE = False

    # Fail fast on misconfigured production deployments (missing Redis
    # password / API key, wildcard CORS) before any model loads.
    validate_production_settings(Settings)

    # The outer body cap is the file limit plus multipart headroom — except that the
    # MCP endpoint carries its file base64-encoded inside a JSON-RPC body, so with
    # MCP enabled the cap must fit a max_file_size file in that form (4/3 inflation)
    # or chew_paper would refuse files well under the advertised limit.
    max_payload_size = Settings.pipeline.max_file_size + Settings.pipeline.multipart_overhead_bytes
    if Settings.mcp.enabled:
        max_payload_size = max(
            max_payload_size,
            base64_envelope(Settings.pipeline.max_file_size)
            + Settings.pipeline.multipart_overhead_bytes,
        )

    upload_store = UploadStore.create(
        max_size=Settings.pipeline.max_file_size,
        spool_memory_bytes=Settings.pipeline.upload_spool_memory_bytes,
        stale_after_seconds=max(Settings.pipeline.timeout + 60, 120),
    )
    upload_stores.append(upload_store)
    api = BibrPipelineAPI(
        api_path=INTERNAL_INFERENCE_PATH,
        enable_async=True,
        upload_root=upload_store.root,
        settings=Settings,
    )
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices=1,
        # Exactly one inference worker, by design and not by default. One worker
        # already serves many requests concurrently on the async loop, and the
        # layout/segmenter GpuBatcher coalesces their GPU work into single
        # forward passes — a second worker would duplicate the models (~1.5 GB
        # RSS each), give each its own batcher (so batches shrink as workers
        # rise), and parallelize only GIL-bound Python, since torch/ONNX
        # intra-op threads already use every core from one process. Scale with
        # PIPELINE_MAX_INFLIGHT_REQUESTS and the batch-timeout knobs instead.
        workers_per_device=1,
        max_payload_size=max_payload_size,
        timeout=Settings.pipeline.timeout,
        restart_workers=Settings.pipeline.restart_workers,
    )

    configure_multipart_spooling(Settings.pipeline.upload_spool_memory_bytes)
    inference_tracker = InferenceDispatchTracker(
        dispatch=resolve_litserve_dispatch(server.app),
        store=upload_store,
    )
    server.app.state.upload_store = upload_store
    server.app.state.inference_tracker = inference_tracker
    register_extract_route(server.app, upload_store, inference_tracker)

    # Gate every other route too (/info, /openapi.json, /docs, …) — LitServe
    # and FastAPI metadata endpoints leak deployment details. Middleware so
    # future routes are covered by default; only the probe paths stay public.
    # Exceptions raised in middleware bypass FastAPI's handlers, so respond
    # directly instead of raising HTTPException.
    from fastapi.responses import JSONResponse

    # This gate is registered before the auth middleware so auth remains
    # outside it in the request stack. Authenticated upload floods are rejected
    # before Starlette parses multipart bodies or LitServe buffers file bytes.
    from bibr.serve.admission import add_upload_admission

    admission_gate = add_upload_admission(
        server.app,
        Settings.pipeline.max_active_uploads,
        # A large /mcp body holds a slot while it is received; the chew tools take
        # their own slot for the extraction (bibr.serve.mcp).
        body_gated_paths=(MCP_MOUNT_PATH, MCP_MOUNT_PATH + "/") if Settings.mcp.enabled else (),
        body_threshold=Settings.pipeline.upload_spool_memory_bytes,
        max_body=max_payload_size,
        max_file_size=Settings.pipeline.max_file_size,
    )
    server.app.state.upload_admission_gate = admission_gate

    @server.app.middleware("http")
    async def _auth_gate(request, call_next):  # pyright: ignore[reportUnusedFunction]
        if request.url.path == INTERNAL_INFERENCE_PATH:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        if request.url.path not in PUBLIC_PATHS:
            detail = check_bearer(request.headers.get("authorization"))
            if detail is not None:
                return JSONResponse(
                    {"detail": detail},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    # Per-request usage metering (D2). Registered AFTER the auth gate so it is
    # the OUTER of the two — 401s from the gate are still timed and logged —
    # but BEFORE CORS, which stays outermost. Skips /health and /ready (noise).
    from bibr.serve.jobs import _sanitize_request_id

    _configure_metering_logging(Settings)

    @server.app.middleware("http")
    async def _metering(request, call_next):  # pyright: ignore[reportUnusedFunction]
        if not Settings.metering.enabled or request.url.path in ("/health", "/ready"):
            return await call_next(request)
        request_id = _sanitize_request_id(request.headers.get("x-request-id")) or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # An unhandled route failure still emits its per-request metering
            # record (status 500, with the request id) before Starlette's
            # ServerErrorMiddleware turns it into the 500 response. The 500
            # body itself is left to the error mapping, so no x-request-id
            # header can be attached there without changing error body shapes.
            duration_ms = int((time.perf_counter() - start) * 1000)
            metering_logger.info(
                _request_metering_line(
                    request_id=request_id,
                    method=request.method,
                    path=request.url.path,
                    status=500,
                    duration_ms=duration_ms,
                    outstanding=getattr(
                        request.state,
                        "inference_outstanding",
                        server.app.state.inference_tracker.outstanding,
                    ),
                )
            )
            raise
        duration_ms = int((time.perf_counter() - start) * 1000)
        response.headers["x-request-id"] = request_id
        response.headers["x-bibr-duration-ms"] = str(duration_ms)
        metering_logger.info(
            _request_metering_line(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
                outstanding=getattr(
                    request.state,
                    "inference_outstanding",
                    server.app.state.inference_tracker.outstanding,
                ),
            )
        )
        return response

    # Async job API (D1). JOBS_STORE picks the in-process store (lives on the
    # single API-server process pinned in main()) or the Redis store shared
    # between replicas; build_job_store fails fast on a Redis store without a
    # URL or the redis package. Routes are auth-gated by the middleware above
    # (not in PUBLIC_PATHS). The dispatcher closes in the router's shutdown
    # handlers first; the store closes after it, once no runner can write.
    if Settings.jobs.enabled:
        from bibr.serve.jobs import build_job_store, register_job_routes

        job_store = build_job_store(Settings)
        server.app.state.job_store = job_store
        register_job_routes(
            server.app,
            store=job_store,
            upload_store=upload_store,
            tracker=inference_tracker,
        )
        server.app.router.add_event_handler("shutdown", job_store.close)

    # MCP endpoint (opt-in): streamable-HTTP Model Context Protocol tools at
    # /mcp, riding the same upload store and inference dispatch as
    # /papers/extract. Covered by the auth-gate middleware above (/mcp is not
    # in PUBLIC_PATHS); its session manager is composed into the app lifespan.
    if Settings.mcp.enabled:
        try:
            from bibr.serve.mcp import mount_mcp
        except ModuleNotFoundError as e:
            if e.name and e.name.split(".")[0] == "mcp":
                from bibr.exceptions import ConfigurationError

                raise ConfigurationError(
                    "MCP_ENABLED=true requires the optional 'mcp' dependency — install "
                    "it with 'uv sync --extra mcp' (source checkout) or "
                    "pip install 'bibr[mcp]'."
                ) from e
            raise
        mount_mcp(
            server.app,
            Settings,
            upload_store=upload_store,
            tracker=inference_tracker,
            admission_gate=admission_gate,
        )

    # CORS added last so it wraps the auth gate — 401s from it still get
    # Access-Control-Allow-Origin, and preflights short-circuit before auth.
    if Settings.cors.origins:
        server.app.add_middleware(
            CORSMiddleware,
            allow_origins=Settings.cors.origins,
            allow_credentials=_safe_cors_credentials(
                Settings.cors.origins, Settings.cors.allow_credentials
            ),
            allow_methods=Settings.cors.allow_methods,
            allow_headers=Settings.cors.allow_headers,
        )

    # Scrub credential-shaped substrings from server logs, including SDK
    # tracebacks logged with exc_info that can embed ?key=… URLs (audit L12).
    # main() installs the serve sink and keeps LitServe's rebuilt handler covered.
    from bibr.serve.logsetup import scrub_library_handlers

    scrub_library_handlers()

    _register_readiness_route(server, Settings)

    async def _close_ingress_resources() -> None:
        try:
            await inference_tracker.close()
        finally:
            await upload_store.close()

    _compose_lifespan_cleanup(server.app, _close_ingress_resources)
    return server


def _register_readiness_route(server, Settings) -> None:
    """Mount a /ready probe that checks downstream services from the master.

    Layout/segmenter readiness is covered by LitServe's own worker lifecycle
    (requests simply queue until workers are ready), so this only probes
    external dependencies: the SGLang OCR server, Redis (when the cache is
    enabled), and the shared job store (when ``JOBS_STORE=redis``).
    """
    import asyncio
    import json

    import httpx
    from fastapi import Request, Response

    from bibr.ocr.http_security import normalize_ocr_base_url, ocr_request_headers
    from bibr.ocr.profiles import GLM_SERVED_MODEL_ALIAS
    from bibr.serve.auth import check_bearer
    from bibr.serve.pipeline import serve_ocr_defaults

    ocr_base_url = normalize_ocr_base_url(Settings.OCR_BASE_URL)
    ocr_headers = ocr_request_headers(
        ocr_base_url,
        Settings.ocr.api_key,
        allow_insecure_http=Settings.ocr.allow_insecure_http,
    )
    # The served-model alias extraction requires on /v1/models — the same
    # alias ServePipeline asks the server for, so /ready fails while the
    # server is still loading it or serves a different name.
    expected_ocr_model, _ = serve_ocr_defaults(Settings)
    expected_ocr_model = expected_ocr_model or GLM_SERVED_MODEL_ALIAS
    state: dict[str, Any] = {}

    async def _get_http_client() -> httpx.AsyncClient:
        client = state.get("http_client")
        if not isinstance(client, httpx.AsyncClient) or client.is_closed:
            client = httpx.AsyncClient(timeout=5.0, headers=ocr_headers)
            state["http_client"] = client
        return client

    async def _get_cache():
        if "cache_inited" in state:
            return state.get("cache")
        state["cache_inited"] = True
        try:
            from bibr.cache import ResponseCache

            if Settings.cache.enabled and Settings.redis.url:
                state["cache"] = ResponseCache(
                    redis_url=Settings.redis.url,
                    ttl_seconds=Settings.cache.ttl_seconds,
                    connect_timeout=Settings.redis.connect_timeout_seconds,
                    socket_timeout=Settings.redis.socket_timeout_seconds,
                    prefix=f"bibr:{Settings.cache.version}",
                )
        except Exception as e:
            logger.warning("Readiness cache init failed: %s", e)
        return state.get("cache")

    @server.app.get("/ready")
    async def ready(request: Request):
        # Disclose per-service health + build_sha only to authenticated callers.
        # check_bearer returns None both when auth is disabled (dev, single-tenant)
        # and when a valid bearer is presented; a non-None detail string means the
        # anonymous caller gets status-only (audit L1).
        include_detail = check_bearer(request.headers.get("authorization")) is None
        checks: dict[str, str] = {}
        check_results: list[bool] = []
        client = await _get_http_client()
        try:
            resp = await client.get(f"{ocr_base_url}/health")
            if resp.status_code != 200:
                checks["ocr"] = f"unhealthy ({resp.status_code})"
                check_results.append(False)
            else:
                # /health answers without auth and before any model loads;
                # extraction needs /v1/models to list the served alias with
                # the same headers the backend sends, so probe that too.
                try:
                    models_resp = await client.get(f"{ocr_base_url}/v1/models")
                    if models_resp.status_code == 401:
                        checks["ocr"] = "unauthorized (401; check OCR_API_KEY)"
                        check_results.append(False)
                    elif models_resp.status_code != 200:
                        checks["ocr"] = f"unhealthy ({models_resp.status_code})"
                        check_results.append(False)
                    else:
                        try:
                            served_ids = [item["id"] for item in models_resp.json()["data"]]
                        except Exception:
                            checks["ocr"] = "unhealthy (bad /v1/models body)"
                            check_results.append(False)
                        else:
                            if expected_ocr_model in served_ids:
                                checks["ocr"] = "ok"
                                check_results.append(True)
                            else:
                                checks["ocr"] = f"model_missing ({expected_ocr_model})"
                                check_results.append(False)
                except Exception:
                    logger.warning("Readiness OCR model check failed", exc_info=True)
                    checks["ocr"] = "unreachable"
                    check_results.append(False)
        except Exception:
            logger.warning("Readiness OCR check failed", exc_info=True)
            checks["ocr"] = "unreachable"
            check_results.append(False)

        classifier_check = state.get("classifier_check")
        classifier_checked_at = state.get("classifier_check_at")
        # A cached ``ok`` stands: re-probing it on every request would put
        # model resolution on the probe path. Anything else is retried after a
        # bounded interval, so a probe that raced worker startup (or a local
        # path that has since appeared) recovers without a restart.
        if (
            not isinstance(classifier_check, tuple)
            or not isinstance(classifier_checked_at, float)
            or (
                classifier_check[0] != "ok"
                and time.monotonic() - classifier_checked_at >= _CLASSIFIER_CHECK_RETRY_SECONDS
            )
        ):
            classifier_check = await asyncio.to_thread(classifier_artifact_readiness, Settings)
            state["classifier_check"] = classifier_check
            state["classifier_check_at"] = time.monotonic()
        classifier_status, classifier_ok = classifier_check
        checks["classifiers"] = classifier_status
        check_results.append(classifier_ok)

        cache = await _get_cache()
        if cache is not None:
            try:
                # Bound the ping so an unresponsive (vs unreachable) Redis
                # can't hang the readiness probe indefinitely.
                await asyncio.wait_for(cache._redis.ping(), timeout=3.0)
                checks["redis"] = "ok"
                check_results.append(True)
            except Exception:
                logger.warning("Readiness Redis check failed", exc_info=True)
                checks["redis"] = "error"
                check_results.append(False)

        # The shared job store is a hard dependency of the job routes when it is
        # configured: a replica that cannot reach it answers 503 on every job call,
        # so it must not receive traffic. The memory store needs no probe.
        job_store = getattr(server.app.state, "job_store", None)
        if Settings.jobs.enabled and Settings.jobs.store == "redis" and job_store is not None:
            try:
                await asyncio.wait_for(job_store.ping(), timeout=3.0)
                checks["jobs_store"] = "ok"
                check_results.append(True)
            except Exception:
                logger.warning("Readiness job-store check failed", exc_info=True)
                checks["jobs_store"] = "error"
                check_results.append(False)

        all_ok = all(check_results)
        status = "ready" if all_ok else "not_ready"
        return Response(
            content=json.dumps(
                readiness_payload(status, checks, Settings, include_detail=include_detail)
            ),
            media_type="application/json",
            status_code=200 if all_ok else 503,
        )


def main():
    """CLI entry point for ``bibr serve``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="bibr serve",
        description="bibr HTTP API — scientific paper extraction pipeline (LitServe)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    from bibr.config import Settings
    from bibr.serve.auth import validate_bind_auth
    from bibr.serve.logsetup import configure_serve_logging, install_litserve_logging_hook

    # The CLI's own logging setup never runs for `serve`; install the serve sink
    # first so every warning below (and every bibr.* record) is formatted and
    # scrubbed rather than falling through logging.lastResort.
    configure_serve_logging(Settings)
    install_litserve_logging_hook()

    validate_bind_auth(args.host, Settings.auth.api_key)

    server = build_server()
    logger.info("bibr serving on %s:%d", args.host, args.port)

    # Upload-root ownership, dispatch tracking, admission, readiness state, and
    # optional jobs are process-local. LitServe otherwise defaults this count to
    # the inference-worker count, making API processes race over shared cleanup.
    # log_config=None keeps uvicorn from installing its own handlers, so its
    # records propagate to the serve sink (formatted, scrubbed) instead.
    server.run(
        host=args.host,
        port=args.port,
        generate_client_file=False,
        num_api_servers=1,
        log_level=Settings.SERVE_LOG_LEVEL,
        log_config=None,
    )
