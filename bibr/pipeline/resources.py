"""Model lifecycle for the local pipeline.

Owns the lifetime of layout, segmenter, OCR, and LLM-server resources.
``LocalPipeline`` delegates load/unload/preload to this class so that
stages can request resources via ``ctx.resources`` without knowing which
backend is behind each.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import inspect
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from bibr.utils.async_tasks import await_owned

if TYPE_CHECKING:
    from bibr.clients.llm import LLMClient
    from bibr.config import GlobalSettings
    from bibr.local.layout import LayoutDetector
    from bibr.local.llm import VllmMlxLlmServer
    from bibr.local.segmenter import SentenceSegmenter
    from bibr.ocr.backend import OcrBackend

logger = logging.getLogger(__name__)


def _make_llm_client(settings: GlobalSettings | None = None):
    if settings is None:
        from bibr.config import snapshot_settings

        settings = snapshot_settings()
    from bibr.clients.llm import LLMClient

    return LLMClient(settings=settings)


def _accepts_positional_arg(callable_obj) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        or parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters
    )


def _accepts_keyword(callable_obj, name: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == name or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _make_settings_aware(callable_obj, settings: GlobalSettings, **kwargs):
    """Call a production constructor with settings while tolerating old fakes."""
    kwargs = {
        name: value
        for name, value in kwargs.items()
        if value is not None and _accepts_keyword(callable_obj, name)
    }
    if _accepts_keyword(callable_obj, "settings"):
        kwargs["settings"] = settings
    return callable_obj(**kwargs)


class ResourceManager:
    """Manages layout, segmenter, OCR engine, and LLM server lifecycle."""

    def __init__(
        self,
        *,
        memory_mode: str = "balanced",
        ocr_backend: str = "paddle",
        ocr_url: str | None = None,
        ocr_model: str | None = None,
        ocr_profile: str | None = None,
        device: str | None = None,
        http_client=None,
        ocr_sem_global=None,
        ocr_breaker=None,
        managed_vllm_fraction: float = 0.0,
        settings: GlobalSettings | None = None,
        layout=None,
        segmenter=None,
        layout_factory: Callable | None = None,
        segmenter_factory: Callable | None = None,
        classifier_resources=None,
        owns_models: bool = True,
    ) -> None:
        from bibr.config import snapshot_settings

        self._settings = settings if settings is not None else snapshot_settings()
        self.memory_mode = memory_mode
        # Preserve the original selector and explicit overrides across the
        # lifetime of any concrete candidate it selects. The public fields
        # expose the active candidate after startup for compatibility, so they
        # cannot also be the source of truth for a later automatic restart.
        self._requested_ocr_backend = ocr_backend
        self._requested_ocr_model = ocr_model
        self._requested_ocr_profile = ocr_profile
        self.ocr_backend = ocr_backend
        self.ocr_url = ocr_url
        self.ocr_model = ocr_model
        self.ocr_profile = ocr_profile
        self.ocr_runtime_identity = None
        self.ocr_fallback_reason: str | None = None
        self.device = device

        self.http_client = http_client
        self.ocr_sem_global = ocr_sem_global
        self.ocr_breaker = ocr_breaker
        self._managed_vllm_fraction = managed_vllm_fraction
        self._layout_factory = layout_factory
        self._segmenter_factory = segmenter_factory
        # Injected models transfer ownership by default. Embedders sharing
        # models across pipelines can retain their lifecycle explicitly.
        self._owns_models = owns_models

        self._layout: LayoutDetector | None = layout
        self._segmenter: SentenceSegmenter | None = segmenter
        self._front_role = None
        self._front_role_resolved = False
        self._ocr: OcrBackend | None = None
        self._ocr_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._ocr_future: concurrent.futures.Future[OcrBackend] | None = None
        self._ocr_init_lock = asyncio.Lock()
        self._llm_server: VllmMlxLlmServer | None = None
        self._llm_init_lock = asyncio.Lock()
        self._llm_client: LLMClient | None = None
        if classifier_resources is None:
            from bibr.pipeline.classifier_resources import ClassifierResources

            classifier_resources = ClassifierResources(
                self._settings,
                memory_mode=memory_mode,
                managed_vllm_fraction=managed_vllm_fraction,
            )
        self._classifiers = classifier_resources

    async def start_classifiers(self) -> None:
        """Load this resource manager's owned classifiers once."""
        await self._classifiers.start()

    async def close_classifiers(self) -> None:
        await self._classifiers.close()

    @property
    def classifiers(self):
        return self._classifiers

    # -- Layout --

    def ensure_layout(self) -> None:
        """Load layout detector if not already loaded."""
        if self._layout is not None and getattr(self._layout, "loaded", False):
            return
        if self._layout_factory is not None:
            self._layout = _make_settings_aware(
                self._layout_factory,
                self._settings,
                device=self.device,
            )
            return
        from bibr.layout.registry import create
        from bibr.local import layout as _local_layout  # noqa: F401 — triggers @register

        self._layout = create("local", device=self.device, settings=self._settings)

    def unload_layout(self) -> None:
        """Unload layout detector to free GPU memory."""
        if self._layout is not None and self._layout.loaded and hasattr(self._layout, "unload"):
            self._layout.unload()

    @property
    def layout(self):
        return self._layout

    # -- Segmenter --

    def ensure_segmenter(self) -> None:
        """Load sentence segmenter if not already loaded."""
        if self._segmenter is not None and getattr(self._segmenter, "loaded", False):
            return
        if self._segmenter_factory is not None:
            self._segmenter = _make_settings_aware(
                self._segmenter_factory,
                self._settings,
            )
            return
        from bibr.local import segmenter as _local_segmenter  # noqa: F401 — triggers @register
        from bibr.segmenter.registry import create

        self._segmenter = create("local", settings=self._settings)

    def unload_segmenter(self) -> None:
        """Unload sentence segmenter."""
        if (
            self._segmenter is not None
            and self._segmenter.loaded
            and hasattr(self._segmenter, "unload")
        ):
            self._segmenter.unload()

    @property
    def segmenter(self):
        return self._segmenter

    def unload_ner_parser(self) -> None:
        """Drop the NER reference-parser singleton reference, if one is loaded.

        The parser lives in ``bibr.extract.ref_extractor`` outside the
        layout/segmenter lifecycle, so aggressive mode never unloads it
        without this hook (``PostParseStage`` calls it after post-parse).
        Only the reference is dropped; in-flight holders keep the object
        alive. Reload is lazy on next use.
        """
        from bibr.extract.ref_extractor import unload_ner_parser

        unload_ner_parser()

    async def close_models(self) -> None:
        """Release resident models even when the closed manager stays reachable."""
        models = (self._layout, self._segmenter)
        self._layout = None
        self._segmenter = None
        if self._front_role is not None:
            from bibr.extract.front_role import release_front_role_classifier

            release_front_role_classifier(self._front_role)
        self._front_role = None
        self._front_role_resolved = False
        if not self._owns_models:
            return
        # The NER parser singleton is process-global rather than an owned
        # model object. Release it in aggressive mode only: balanced
        # one-shot chew() calls share the process, and unloading there
        # would reload the ~1 GB parser on every file.
        if self.memory_mode == "aggressive":
            self.unload_ner_parser()
        for model in models:
            if model is None:
                continue
            try:
                if hasattr(model, "unload"):
                    await await_owned(asyncio.to_thread(model.unload))
                elif hasattr(model, "aclose"):
                    await await_owned(model.aclose())
            except Exception:  # best-effort: one failure cannot retain the other model
                logger.warning("Resident model shutdown failed", exc_info=True)

    # -- Front-role classifier (optional prior for front-matter ownership) --

    def ensure_front_role(self):
        """Return the front-role classifier, loading it on first use.

        ``None`` when disabled (``ML_FRONT_ROLE_MODEL_ID`` unset) or when the
        bundle cannot be loaded — the loader logs once and the pipeline keeps
        going on heuristics alone.
        """
        if self._front_role_resolved:
            return self._front_role
        from bibr.extract.front_role import load_front_role_classifier

        self._front_role = load_front_role_classifier(self._settings)
        self._front_role_resolved = True
        return self._front_role

    @property
    def front_role(self):
        return self._front_role

    # -- OCR --

    def _resolve_ocr_candidates(self):
        """Resolve startup candidates, retaining explicit request overrides."""
        from bibr.ocr.registry import resolve_backend_candidates

        candidates = resolve_backend_candidates(self._requested_ocr_backend, self._settings)
        if len(candidates) != 1 or self._requested_ocr_backend == "paddle":
            return candidates
        candidate = candidates[0]
        # A concrete backend is an explicit user choice: preserve its model
        # and profile through construction, identity, cache, and export. The
        # vLLM Paddle endpoint is intentionally addressed by its served alias;
        # The requested model remains the source model_path passed to vLLM.
        model = (
            candidate.model
            if candidate.backend == "paddle-vllm"
            else self._requested_ocr_model or candidate.model
        )
        profile = self._requested_ocr_profile or self._settings.ocr.profile or candidate.profile
        return (replace(candidate, model=model, profile=profile),)

    def _ocr_kwargs(self, backend_name: str | None = None, candidate=None) -> dict:
        """Pack constructor kwargs shared across OCR backends.

        Each backend consumes only the subset it cares about (the rest land
        in ``**_kw`` and are ignored).
        """
        backend_name = backend_name or self._requested_ocr_backend

        kwargs: dict = {
            "base_url": self.ocr_url,
            # Served-model name for HTTP backends. Local engine backends use
            # ``model_path`` and ignore this key via ``**_kw``.
            "model": self._requested_ocr_model,
            "model_path": self._requested_ocr_model,
            "device": self.device,
            "settings": self._settings,
        }
        if backend_name in {
            "glm-http",
            "paddle-http",
            "paddle-vllm",
            "paddle-rapid-mlx",
            "paddle-mlx-vlm",
            "serve-http",
        }:
            from bibr.ocr.profiles import resolve_ocr_profile, resolve_ocr_runtime_identity
            from bibr.pipeline.context import RunConfig

            identity = resolve_ocr_runtime_identity(
                RunConfig(
                    ocr_backend=(
                        candidate.backend if candidate is not None else self._requested_ocr_backend
                    ),
                    ocr_url=self.ocr_url,
                    ocr_model=(
                        candidate.model if candidate is not None else self._requested_ocr_model
                    ),
                    ocr_profile=(
                        candidate.profile if candidate is not None else self._requested_ocr_profile
                    ),
                ),
                self._settings,
            )
            kwargs["model"] = identity.model
            kwargs["profile"] = resolve_ocr_profile(
                explicit=identity.profile,
                backend=identity.backend,
                model=identity.model,
                max_tokens=self._settings.ocr.generation_max_tokens,
                temperature=self._settings.ocr.generation_temperature,
            )
            if backend_name == "paddle-vllm":
                # vLLM loads the configured Hugging Face model from model_path,
                # but its OpenAI endpoint must be addressed by the served alias.
                kwargs["model"] = self._settings.ocr.paddle_served_model
                if candidate is not None and self._requested_ocr_model is None:
                    kwargs["model_path"] = self._settings.ocr.paddle_model
        if self._requested_ocr_backend == "serve-http":
            kwargs.update(
                {
                    "http_client": self.http_client,
                    "sem_global": self.ocr_sem_global,
                    "breaker": self.ocr_breaker,
                }
            )
        return kwargs

    def _resolve_ocr_backend_name(self, backend_name: str | None = None) -> str:
        """Legacy shim: ``--ocr-url`` implicitly forces the HTTP backend."""
        backend_name = backend_name or self._requested_ocr_backend
        if backend_name in self._VISION_BACKENDS:
            from bibr.local import ocr_cloud  # noqa: F401 — triggers @register
        elif backend_name != "serve-http":
            from bibr.local import ocr as _ocr  # noqa: F401 — triggers @register
            from bibr.local import rapid_mlx as _rapid_mlx  # noqa: F401 — triggers @register

            if backend_name == "paddle-vllm":
                from bibr.local import vllm_ocr as _vllm_ocr  # noqa: F401 — triggers @register
                from bibr.ocr.registry import register

                register(_vllm_ocr.PaddleVllmOcrClient)
            elif backend_name == "paddle-mlx-vlm":
                importlib.import_module("bibr.local.mlx_vlm_ocr")

        from bibr.ocr.registry import known_backends

        known = known_backends()
        if backend_name != "serve-http" and backend_name not in known:
            raise ValueError(f"Unknown OCR backend: {backend_name!r}. Known: {known}")

        if self.ocr_url and backend_name not in ("glm-http", "paddle-http", "serve-http"):
            return "glm-http"
        return backend_name

    _VISION_BACKENDS = frozenset({"gemini", "openai", "anthropic"})

    def _create_ocr_client(self, candidate=None):
        """Create an OCR client (blocking — run in thread executor)."""
        candidate = candidate or getattr(self, "_startup_ocr_candidate", None)
        requested_name = candidate.backend if candidate is not None else None
        backend_name = self._resolve_ocr_backend_name(requested_name)
        if backend_name in self._VISION_BACKENDS:
            from bibr.local import ocr_cloud  # noqa: F401 — triggers @register
        else:
            from bibr.local import ocr as _ocr  # noqa: F401 — triggers @register
            from bibr.local import rapid_mlx as _rapid_mlx  # noqa: F401 — triggers @register

            if backend_name == "paddle-vllm":
                from bibr.local import vllm_ocr as _vllm_ocr  # noqa: F401 — triggers @register
                from bibr.ocr.registry import register

                register(_vllm_ocr.PaddleVllmOcrClient)
            elif backend_name == "paddle-mlx-vlm":
                importlib.import_module("bibr.local.mlx_vlm_ocr")
        from bibr.ocr.registry import create

        return create(backend_name, **self._ocr_kwargs(backend_name, candidate))

    def _create_ocr_client_for(self, candidate):
        """Construct one candidate while keeping the legacy factory seam stable."""
        self._startup_ocr_candidate = candidate
        try:
            return self._create_ocr_client()
        finally:
            self._startup_ocr_candidate = None

    def start_ocr_preload(self) -> None:
        """Start OCR engine initialization in a background thread."""
        if self._ocr is not None and self._ocr.loaded:
            return
        if self._ocr_future is not None:
            return
        candidates = self._resolve_ocr_candidates()
        if len(candidates) > 1:
            # A fallback chain is transactional only after the candidate has
            # passed readiness checks, so preload cannot safely select one.
            return
        # Import the backend module on the main thread before submitting to
        # the worker so registration completes before the client factory runs.
        backend_name = self._resolve_ocr_backend_name(candidates[0].backend)
        if backend_name in self._VISION_BACKENDS:
            from bibr.local import ocr_cloud  # noqa: F401 — triggers @register
        else:
            from bibr.local import ocr, rapid_mlx  # noqa: F401 — triggers @register

            if backend_name == "paddle-vllm":
                from bibr.local import vllm_ocr  # noqa: F401 — triggers @register
                from bibr.ocr.registry import register

                register(vllm_ocr.PaddleVllmOcrClient)
            elif backend_name == "paddle-mlx-vlm":
                importlib.import_module("bibr.local.mlx_vlm_ocr")
        logger.info("Pre-loading OCR engine in background...")
        self._ocr_executor = concurrent.futures.ThreadPoolExecutor(1)
        self._ocr_future = self._ocr_executor.submit(self._create_ocr_client_for, candidates[0])

    async def _shutdown_client(self, client) -> None:
        if client is None or not hasattr(client, "shutdown"):
            return
        result = await asyncio.to_thread(client.shutdown)
        if inspect.isawaitable(result):
            await result

    async def _construct_ocr_candidate(self, candidate):
        """Collect a constructor without losing a late result to cancellation."""
        loop = asyncio.get_running_loop()
        if self._ocr_future is None:
            return await await_owned(
                loop.run_in_executor(None, self._create_ocr_client_for, candidate),
                on_cancel=self._shutdown_client,
            )
        try:
            return await await_owned(
                asyncio.wrap_future(self._ocr_future),
                on_cancel=self._shutdown_client,
            )
        finally:
            # await_owned has settled the constructor, including cancellation
            # cleanup, before we forget the preload handle.
            self._ocr_future = None
            if self._ocr_executor is not None:
                self._ocr_executor.shutdown(wait=False)
                self._ocr_executor = None

    async def _await_ocr_client_ready(self, client) -> None:
        if not hasattr(client, "wait_for_server"):
            return
        result = client.wait_for_server()
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _startup_failure_reason(candidate) -> str:
        """Safe, actionable startup diagnostics without exception disclosure."""
        remedies = {
            "paddle-vllm": "install vLLM and verify the Paddle model is available",
            "paddle-rapid-mlx": "install rapid-mlx and make its executable available",
            "paddle-mlx-vlm": "install MLX-VLM dependencies and the Paddle model",
            "glm-rapid-mlx": "install rapid-mlx and verify the GLM model is available",
            "glm-llama": "install llama.cpp and verify the GLM GGUF model is available",
        }
        return f"{candidate.backend}: {remedies.get(candidate.backend, 'check local OCR runtime setup')}"

    def _set_ocr_runtime_identity(self, candidate) -> None:
        from bibr.ocr.profiles import OcrRuntimeIdentity, resolve_ocr_profile

        profile = resolve_ocr_profile(
            explicit=candidate.profile,
            backend=candidate.backend,
            model=candidate.model,
            max_tokens=self._settings.ocr.generation_max_tokens,
            temperature=self._settings.ocr.generation_temperature,
        )
        self.ocr_backend = candidate.backend
        self.ocr_model = candidate.model
        self.ocr_profile = candidate.profile
        self.ocr_runtime_identity = OcrRuntimeIdentity(
            backend=candidate.backend,
            model=candidate.model,
            profile=candidate.profile,
            normalizer_version=profile.normalizer_version,
        )

    async def await_ocr(self) -> None:
        """Wait for background OCR preload, or load on a worker thread.

        Always runs the (blocking) OCR constructor off the event loop, so
        nothing in the async pipeline can stall on engine startup.
        """
        async with self._ocr_init_lock:
            if self._ocr is not None and getattr(self._ocr, "loaded", False):
                return
            from bibr.exceptions import UpstreamServiceError

            candidates = self._resolve_ocr_candidates()
            if self._ocr_future is not None:
                # Preload is only supported for one concrete candidate. Keep
                # that contract for embedders that supply their own future.
                candidates = candidates[:1]
            failures: list[str] = []
            for candidate in candidates:
                client = None
                published = False
                try:
                    client = await self._construct_ocr_candidate(candidate)
                    if client is None:
                        raise RuntimeError("factory returned no client")
                    try:
                        await self._await_ocr_client_ready(client)
                    except Exception:
                        # Only a backend that owns a cross-request readiness
                        # cooldown publishes its failed instance, so later
                        # requests fail fast on the cooldown instead of
                        # rebuilding and re-polling for the full timeout. A
                        # published instance stays live — it is never shut
                        # down here. Anything else is discarded, never reused
                        # (every OCR client defines wait_for_server, so that
                        # gate cannot tell them apart).
                        if client is not None and getattr(
                            client, "keeps_readiness_cooldown", False
                        ):
                            self._ocr = client
                            published = True
                            try:
                                self._set_ocr_runtime_identity(candidate)
                            except Exception:
                                # Provenance only; never mask the readiness error.
                                logger.warning("OCR runtime identity unavailable", exc_info=True)
                        raise
                except BaseException as exc:  # dispose unpublished clients on cancellation too
                    try:
                        # A published instance stays live for its cooldown —
                        # never shut it down here.
                        if not published:
                            await await_owned(self._shutdown_client(client))
                    except Exception:
                        if not isinstance(exc, Exception):
                            logger.warning(
                                "Server cleanup failed after cancellation", exc_info=True
                            )
                            raise exc from None
                        raise
                    if not isinstance(exc, Exception):
                        raise
                    if len(candidates) == 1:
                        if client is None and isinstance(exc, RuntimeError):
                            raise RuntimeError(
                                f"OCR backend factory for {candidate.backend!r} returned no client"
                            ) from exc
                        raise
                    failures.append(self._startup_failure_reason(candidate))
                    continue
                self._ocr = client
                self._set_ocr_runtime_identity(candidate)
                self.ocr_fallback_reason = "; ".join(failures) or None
                logger.info(
                    "Selected OCR backend %s model %s profile %s%s",
                    candidate.backend,
                    candidate.model,
                    candidate.profile,
                    f" after fallback ({self.ocr_fallback_reason})" if failures else "",
                )
                return
            self.ocr_fallback_reason = "; ".join(failures)[:500]
            raise UpstreamServiceError(
                "ocr",
                f"No OCR startup candidate succeeded: {self.ocr_fallback_reason}",
            )

    async def _drain_ocr_preload(self) -> None:
        """Reclaim a preloaded engine that ``await_ocr`` never came to collect.

        The preload submits the blocking OCR-backend constructor — which spawns
        and health-waits a managed inference subprocess — to an executor, and
        only ``await_ocr``, reached from ``OcrStage``, transfers the result. If
        an earlier stage raises (classically: the layout model OOMs on the GPU
        the preload just claimed), the future is never drained and the
        subprocess outlives the run.
        """
        future, self._ocr_future = self._ocr_future, None
        executor, self._ocr_executor = self._ocr_executor, None
        try:
            if future is None:
                return
            future.cancel()
            client = None
            try:
                client = await asyncio.wrap_future(future)
            except (concurrent.futures.CancelledError, asyncio.CancelledError):
                return
            except Exception as exc:  # noqa: BLE001 - a failed preload has nothing to reclaim
                logger.debug("OCR preload failed before it was collected: %r", exc)
                return
            if client is not None and client is not self._ocr:
                logger.info("Shutting down an OCR engine preloaded but never claimed")
                await self._shutdown_client(client)
        finally:
            if executor is not None:
                executor.shutdown(wait=False)

    async def shutdown_ocr(self) -> None:
        """Shutdown OCR client to free GPU memory."""
        await await_owned(self._shutdown_ocr())

    async def _shutdown_ocr(self) -> None:
        async with self._ocr_init_lock:
            await self._shutdown_ocr_locked()

    async def _shutdown_ocr_locked(self) -> None:
        await self._drain_ocr_preload()
        client = self._ocr
        try:
            if client is not None and hasattr(client, "shutdown"):
                result = await asyncio.to_thread(client.shutdown)
                if asyncio.iscoroutine(result):
                    await result
        finally:
            # The client and concrete identity form one lifecycle. Clearing
            # them together makes a later startup rerun the original selector
            # instead of pairing new inference with stale cache/provenance.
            self._ocr = None
            self.ocr_runtime_identity = None
            self.ocr_fallback_reason = None

    @property
    def ocr(self):
        return self._ocr

    # -- LLM server (vllm-mlx on Apple Silicon, vllm on Linux/CUDA) --

    async def _construct_llm_server(self, factory):
        return await await_owned(
            asyncio.get_running_loop().run_in_executor(None, factory),
            on_cancel=self._shutdown_client,
        )

    async def start_llm_server(self, backend: str) -> None:
        """Start local LLM server if backend requires one (idempotent).

        A managed server binds the pipeline's fixed port and is long-lived:
        ``process_chunk`` reuses it across chunks and never tears it down. Re-
        invoking for an already-running server would spawn a second subprocess
        that fails to bind the port — hard-failing every file in later chunks —
        so no-op when a server is already up.
        """
        async with self._llm_init_lock:
            if self._llm_server is not None:
                return
            server = None
            if backend == "vllm-mlx":
                from bibr.local.llm import VllmMlxLlmServer

                server = await self._construct_llm_server(
                    lambda: _make_settings_aware(VllmMlxLlmServer, self._settings),
                )
            elif backend == "rapid-mlx":
                import bibr.local.rapid_mlx as rapid_mlx

                server = await self._construct_llm_server(
                    lambda: _make_settings_aware(rapid_mlx.RapidMlxLlmServer, self._settings),
                )
            elif backend == "vllm":
                # Module (not class) import so tests can monkeypatch the symbol.
                import bibr.local.vllm_llm as vllm_llm

                server = await self._construct_llm_server(
                    lambda: _make_settings_aware(
                        vllm_llm.VllmLlmServer,
                        self._settings,
                        mem_fraction=self._managed_vllm_fraction or None,
                    ),
                )
            elif backend == "llama-cpp":
                from bibr.local.llama_cpp import LlamaCppLlmServer

                server = await self._construct_llm_server(
                    lambda: _make_settings_aware(LlamaCppLlmServer, self._settings),
                )
            elif backend == "llmster":
                from bibr.local.llmster import LlmsterLlmServer

                server = await self._construct_llm_server(
                    lambda: _make_settings_aware(LlmsterLlmServer, self._settings),
                )

            if server is None:
                return
            try:
                server.configure_llm_client()
            except Exception:
                server.shutdown()
                raise
            self._llm_server = server

    def shutdown_llm_server(self) -> None:
        """Synchronous finalizer; async owners should use close_llm_server."""
        if self._llm_server is None:
            return
        server, self._llm_server = self._llm_server, None
        server.shutdown()

    async def close_llm_server(self) -> None:
        """Wait for startup before shutdown so close cannot miss a late server."""

        async def close() -> None:
            async with self._llm_init_lock:
                await asyncio.to_thread(self.shutdown_llm_server)

        await await_owned(close())

    @property
    def llm_client(self):
        """Lazy-initialized LLMClient shared across the pipeline run."""
        if self._llm_client is None:
            if _accepts_positional_arg(_make_llm_client):
                self._llm_client = _make_llm_client(self._settings)
            else:
                self._llm_client = _make_llm_client()
        return self._llm_client

    async def close_llm_client(self) -> None:
        """Close the shared LLMClient (idempotent)."""
        if self._llm_client is None:
            return
        try:
            await self._llm_client.close()
        except Exception as exc:
            logger.warning("LLMClient.close failed: %s", exc)
        self._llm_client = None
