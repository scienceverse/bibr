"""HTTPX-backed OCR backend for the serve layer.

Registers against the OCR registry under the name ``"serve-http"``.
``OcrStage`` handles per-file concurrency; this backend adds global
concurrency limiting (shared across all concurrent requests), circuit
breaker fault tolerance, and HTTPX retry logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import ClassVar

import httpx

from bibr.local.ocr_transport import BaseHttpOcrClient
from bibr.ocr.backend import OcrText
from bibr.ocr.otsl import check_otsl_completeness
from bibr.ocr.profiles import (
    PADDLE_TABLE_RECOVERY_MAX_TOKENS,
    OcrProfile,
    resolve_ocr_profile,
)
from bibr.ocr.registry import register
from bibr.utils.circuit_breaker import AsyncCircuitBreaker

logger = logging.getLogger(__name__)


_OCR_MAX_RETRIES = 2
_OCR_RETRY_BACKOFF_BASE = 0.5
_OCR_RETRY_BACKOFF_MAX = 8.0
_OCR_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Once a readiness wait fails, refuse further wait_for_server calls for this
# long before retrying. Avoids both (a) blocking every request on a long
# timeout when the OCR server is down, and (b) the original "set _ready=True
# on timeout" bug that silently treated a never-started server as healthy.
_OCR_READY_FAILURE_COOLDOWN = 30.0


@register
class BibrServeOcrBackend:
    """``OcrBackend`` implementation that proxies to a remote GLM-OCR server."""

    name: ClassVar[str] = "serve-http"

    #: A failed readiness poll keeps this instance published on the
    #: ``ResourceManager`` so later requests fail fast on its cooldown
    #: instead of rebuilding and re-polling for the full timeout. Backends
    #: without this flag are discarded (and shut down) on readiness failure,
    #: never reused.
    keeps_readiness_cooldown: ClassVar[bool] = True

    #: Served-model alias used when the caller passes no model. The documented
    #: convention for externally-managed GLM-OCR servers; a server serving a
    #: different model (e.g. a NuExtract vLLM instance) must be pointed at via
    #: ``OCR_MODEL`` — vLLM's OpenAI endpoint 404s on an unknown model name.
    _DEFAULT_MODEL: ClassVar[str] = "glm-ocr"

    def __init__(
        self,
        *,
        base_url: str,
        http_client: httpx.AsyncClient,
        sem_global: asyncio.Semaphore,
        breaker: AsyncCircuitBreaker,
        model: str | None = None,
        profile: OcrProfile | None = None,
        ready_poll_interval: float = 2.0,
        ready_timeout: float | None = None,
        settings=None,
        **_kw: object,
    ) -> None:
        from bibr.config import snapshot_settings
        from bibr.ocr.http_security import normalize_ocr_base_url

        self._settings = settings if settings is not None else snapshot_settings()
        self._base_url = normalize_ocr_base_url(base_url)
        self._http = http_client
        self._sem_per_worker = sem_global
        self._breaker = breaker
        self._model = model or self._DEFAULT_MODEL
        self._profile = profile or resolve_ocr_profile(
            explicit=self._settings.ocr.profile,
            backend=self.name,
            model=self._model,
            max_tokens=self._settings.ocr.generation_max_tokens,
            temperature=self._settings.ocr.generation_temperature,
        )
        self._ready_poll_interval = ready_poll_interval
        self._ready_timeout = float(
            ready_timeout
            if ready_timeout is not None
            else self._settings.pipeline.deployment_ready_timeout
        )
        self._ready = False
        self._cooldown_until = 0.0
        # Serializes concurrent first ``wait_for_server`` callers so they
        # don't all run the polling loop in parallel (duplicate /health
        # probes) or race over ``_cooldown_until`` writes on timeout.
        self._ready_lock: asyncio.Lock | None = None

    @property
    def loaded(self) -> bool:  # OcrBackend contract
        return True

    async def wait_for_server(self) -> None:
        from bibr.exceptions import UpstreamServiceError

        if self._ready:
            return
        if self._ready_lock is None:
            self._ready_lock = asyncio.Lock()
        async with self._ready_lock:
            if self._ready:
                return
            now = time.monotonic()
            # Client-facing messages (they become 502 bodies) name neither the
            # internal endpoint nor what it serves; the log lines carry both.
            if now < self._cooldown_until:
                logger.warning(
                    "OCR server at %s: readiness check still cooling down for %.1fs",
                    self._base_url,
                    self._cooldown_until - now,
                )
                raise UpstreamServiceError(
                    "ocr",
                    "OCR server unavailable "
                    f"(readiness is retried in {self._cooldown_until - now:.0f}s)",
                )

            deadline = now + self._ready_timeout
            logger.info(
                "Waiting up to %ds for OCR server at %s…", self._ready_timeout, self._base_url
            )
            host_was_unreachable = False
            last_status: int | None = None
            observed_model_ids: list[str] = []
            while time.monotonic() < deadline:
                try:
                    resp = await self._http.get(f"{self._base_url}/v1/models", timeout=5.0)
                    # Any response proves the host is up, whatever its status.
                    host_was_unreachable = False
                    if resp.status_code == 200:
                        data = resp.json()
                        observed_model_ids = [item["id"] for item in data["data"]]
                        # A 200 answers the key question in the affirmative —
                        # forget any earlier 401 so the failure names the
                        # model the server keeps not listing, not a key the
                        # server has since accepted.
                        last_status = None
                        if self._model in observed_model_ids:
                            self._ready = True
                            logger.info("OCR server ready at %s", self._base_url)
                            return
                    else:
                        # A non-200 answer still proves the host is up; remember
                        # which status so the failure names the key (401), not
                        # the model alias.
                        last_status = resp.status_code
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    host_was_unreachable = True
                except Exception:  # noqa: S110
                    host_was_unreachable = False
                    pass
                await asyncio.sleep(self._ready_poll_interval)

            self._cooldown_until = time.monotonic() + _OCR_READY_FAILURE_COOLDOWN
            if host_was_unreachable:
                logger.error(
                    "OCR server at %s is unreachable (no connection within %.0fs)",
                    self._base_url,
                    self._ready_timeout,
                )
                raise UpstreamServiceError("ocr", "OCR server is unreachable.")
            if last_status == 401:
                logger.error(
                    "OCR server at %s returned 401 (unauthorized — check OCR_API_KEY); "
                    "observed model ids: %r",
                    self._base_url,
                    observed_model_ids,
                )
                raise UpstreamServiceError(
                    "ocr",
                    "OCR server unauthorized (401): check OCR_API_KEY.",
                )
            logger.error(
                "OCR server at %s did not serve model %r within %.0fs; observed model ids: %r",
                self._base_url,
                self._model,
                self._ready_timeout,
                observed_model_ids,
            )
            raise UpstreamServiceError(
                "ocr",
                f"OCR server did not become ready within {self._ready_timeout:.0f}s "
                f"for model {self._model!r}.",
            )

    async def shutdown(self) -> None:
        # HTTPX client lifecycle is owned by ``ServePipeline``; this backend
        # does not close it.
        return

    async def recognize(self, image, prompt: str) -> str:
        from bibr.exceptions import UpstreamServiceError
        from bibr.ocr.image_utils import encode_region_for_ocr
        from bibr.utils.circuit_breaker import CircuitOpenError

        # Resize + JPEG-encode + base64 is synchronous Pillow work, and it ran
        # as the coroutine's first statement — before the semaphore — so it was
        # neither offloaded nor bounded. On serve's single async worker that is
        # head-of-line blocking for every co-resident request (max loop
        # lateness 300 ms inline vs 15 ms threaded on 800 real crops); doing it
        # inside the semaphore also bounds how many encoded payloads are
        # resident at once. Pillow releases the GIL, so wall time improves too.
        try:
            async with self._sem_per_worker, self._breaker:
                image_b64 = await asyncio.to_thread(
                    encode_region_for_ocr, image, self._profile.image
                )
                result = await self._post_with_retry(image_b64, prompt)
                task = self._profile.task_for_prompt(prompt)
                if self._profile.name != "paddle" or task != "table":
                    return result

                completeness = check_otsl_completeness(result)
                if result.finish_reason != "length" and completeness.complete:
                    return result

                logger.warning(
                    "Paddle table output incomplete; retrying once at %d tokens "
                    "(finish_reason=%s, reasons=%s)",
                    PADDLE_TABLE_RECOVERY_MAX_TOKENS,
                    result.finish_reason,
                    completeness.reasons,
                )
                try:
                    recovered = await self._post_with_retry(
                        image_b64,
                        prompt,
                        max_tokens=PADDLE_TABLE_RECOVERY_MAX_TOKENS,
                    )
                except Exception:
                    logger.warning(
                        "Paddle table recovery failed; retaining initial output",
                        exc_info=True,
                    )
                    return result
                if not recovered:
                    logger.warning(
                        "Paddle table recovery returned empty output; retaining initial output"
                    )
                    return result
                return recovered
        except CircuitOpenError as exc:
            # The breaker only counts server-side failures (5xx/connection/
            # timeout), so an OPEN breaker means the OCR server is down. Surface
            # it as a systemic upstream error (a BibrError) so OcrStage fails the
            # page/file cleanly (→ HTTP 502) instead of silently blanking the
            # region's content.
            raise UpstreamServiceError("ocr", str(exc), original_error=exc) from exc

    async def _post_with_retry(
        self,
        image_b64: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> OcrText:
        payload = BaseHttpOcrClient.build_request_payload(
            model=self._model,
            profile=self._profile,
            image_b64=image_b64,
            prompt=prompt,
        )
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        url = f"{self._base_url}/v1/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(1 + _OCR_MAX_RETRIES):
            try:
                resp = await self._http.post(url, json=payload)
                if resp.status_code in _OCR_RETRYABLE_STATUS and attempt < _OCR_MAX_RETRIES:
                    delay = min(
                        _OCR_RETRY_BACKOFF_BASE * (2**attempt),
                        _OCR_RETRY_BACKOFF_MAX,
                    )
                    logger.warning(
                        "OCR retry %d/%d (status %d), backoff %.1fs",
                        attempt + 1,
                        _OCR_MAX_RETRIES,
                        resp.status_code,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                try:
                    choice = data["choices"][0]
                    content: str = choice["message"]["content"]
                    return OcrText(content, finish_reason=choice.get("finish_reason"))
                except (KeyError, IndexError, TypeError) as exc:
                    raise ValueError(
                        f"Unexpected OCR response structure: {exc} "
                        f"(keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__})"
                    ) from exc
            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                last_exc = e
                if attempt < _OCR_MAX_RETRIES:
                    delay = min(
                        _OCR_RETRY_BACKOFF_BASE * (2**attempt),
                        _OCR_RETRY_BACKOFF_MAX,
                    )
                    logger.warning(
                        "OCR retry %d/%d (error: %s: %s), backoff %.1fs",
                        attempt + 1,
                        _OCR_MAX_RETRIES,
                        type(e).__name__,
                        e or "(no message)",
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        assert last_exc is not None  # noqa: S101 — loop sets it on every retry path
        raise last_exc
