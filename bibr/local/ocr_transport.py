"""Shared HTTP transport for OpenAI-compatible OCR servers.

``HttpOcrClient`` and other concrete clients speak an OpenAI-compatible
``/v1/chat/completions`` endpoint and share identical machinery: httpx client
setup, ``/v1/models`` polling with fast-fail on an unreachable host,
retry/backoff on retryable statuses, and teardown.

This base class owns that machinery. Subclasses supply only the parts that
genuinely differ between concrete OCR clients:

- ``_DEFAULT_MODEL`` — served-model alias when the caller passes no model.
- ``_SERVICE_LABEL`` — the human-readable name used in log lines and error
  strings.
- ``_clean_output`` — optional post-processing of the returned text.
"""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar

from bibr.config import GlobalSettings, snapshot_settings
from bibr.exceptions import UpstreamServiceError
from bibr.ocr.backend import OcrText
from bibr.ocr.image_utils import encode_region_for_ocr
from bibr.ocr.profiles import GLM_SERVED_MODEL_ALIAS, OcrProfile, resolve_ocr_profile

logger = logging.getLogger(__name__)


class BaseHttpOcrClient:
    """Base for HTTP OCR clients speaking the OpenAI chat-completions API.

    Not registered itself — concrete subclasses declare a ``name`` and apply
    ``@register``.
    """

    name: ClassVar[str]
    _MAX_RETRIES = 2
    _RETRY_BACKOFF_BASE = 0.5
    _RETRY_BACKOFF_MAX = 8.0
    _RETRYABLE_STATUS: ClassVar[set[int]] = {429, 500, 502, 503, 504}

    #: Served-model alias used when the caller doesn't pass one explicitly.
    #: The documented convention for externally-managed servers; callers that
    #: manage their own server under a different name (e.g. the vllm-mlx
    #: subprocess, served under its real HF repo id) must pass it explicitly.
    _DEFAULT_MODEL: ClassVar[str] = GLM_SERVED_MODEL_ALIAS
    #: Human-readable label used in log messages and error strings.
    _SERVICE_LABEL: ClassVar[str] = "OCR"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        profile: OcrProfile | None = None,
        settings: GlobalSettings | None = None,
        **_kw: object,
    ):
        import httpx

        from bibr.ocr.http_security import normalize_ocr_base_url, ocr_request_headers

        self._settings = settings if settings is not None else snapshot_settings()
        self._base_url = normalize_ocr_base_url(base_url or self._settings.OCR_BASE_URL)
        timeout_val = timeout if timeout is not None else self._settings.ocr.request_timeout
        self._model = model or self._DEFAULT_MODEL
        self._profile = profile or resolve_ocr_profile(
            explicit=self._settings.ocr.profile,
            backend=self.name,
            model=self._model,
            max_tokens=(
                max_tokens if max_tokens is not None else self._settings.ocr.generation_max_tokens
            ),
            temperature=self._settings.ocr.generation_temperature,
        )
        headers = ocr_request_headers(
            self._base_url,
            self._settings.ocr.api_key,
            allow_insecure_http=self._settings.ocr.allow_insecure_http,
        )

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(timeout_val),
                write=30.0,
                pool=10.0,
            ),
            http2=True,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )
        self._loaded = True
        logger.info("%s ready (url=%s)", type(self).__name__, self._base_url)

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def wait_for_server(self) -> None:
        """Poll the model listing until the configured server alias is ready.

        A TCP connection-refused / DNS-failure on the very first attempt is
        treated as fatal — there's no point sleeping 2 minutes hoping a host
        that's down will spontaneously come up. A response (any status) or a
        timeout means the host is reachable and we wait the full deadline.
        """
        import time

        import httpx

        poll_interval = 2.0
        timeout = self._settings.pipeline.deployment_ready_timeout
        deadline = time.monotonic() + timeout
        label = self._SERVICE_LABEL
        logger.info("Waiting up to %ds for %s server at %s...", timeout, label, self._base_url)

        host_was_unreachable = False
        observed_model_ids: list[str] = []
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            try:
                resp = await self._client.get(f"{self._base_url}/v1/models", timeout=5.0)
                host_was_unreachable = False
                if resp.status_code == 200:
                    data = resp.json()
                    model_data = data["data"]
                    observed_model_ids = [item["id"] for item in model_data]
                    if self._model in observed_model_ids:
                        logger.info("%s server ready at %s", label, self._base_url)
                        return
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                # Host unreachable (refused, no route, DNS failure). Sleep briefly
                # in case the server is mid-restart, but bail fast — most of the
                # 120s default deadline would just be wasted polling.
                host_was_unreachable = True
                if attempts >= 3:
                    raise UpstreamServiceError(
                        "ocr",
                        (
                            f"{label} server at {self._base_url} is unreachable "
                            f"({type(e).__name__}). Check the URL and that the "
                            "server is running."
                        ),
                    ) from e
            except Exception:  # noqa: S110, BLE001
                # Other errors (5xx, malformed response) — host is up but not
                # ready yet. Keep polling.
                host_was_unreachable = False
            await asyncio.sleep(poll_interval)

        if host_was_unreachable:
            raise UpstreamServiceError(
                "ocr",
                f"{label} server at {self._base_url} is unreachable.",
            )
        raise UpstreamServiceError(
            "ocr",
            (
                f"{label} server at {self._base_url} did not become ready within {timeout}s "
                f"for model {self._model!r}; observed model ids: {observed_model_ids!r}"
            ),
        )

    async def recognize(self, image, prompt: str) -> str:
        """OCR a single cropped region image via HTTP.

        Args:
            image: PIL Image of the cropped region.
            prompt: OCR task prompt (e.g. "Text Recognition:").

        Returns:
            Recognized text content.
        """
        image_b64 = encode_region_for_ocr(image, self._profile.image)
        return await self._send_request(image_b64, prompt)

    def _build_payload(self, image_b64: str, prompt: str) -> dict:
        """Build a profile-aware chat-completions request body."""
        return self.build_request_payload(
            model=self._model,
            profile=self._profile,
            image_b64=image_b64,
            prompt=prompt,
        )

    @staticmethod
    def build_request_payload(
        *, model: str, profile: OcrProfile, image_b64: str, prompt: str
    ) -> dict:
        """Build the shared OpenAI-compatible OCR request shape."""
        request = profile.request
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": request.max_tokens_for(profile.task_for_prompt(prompt)),
            "temperature": request.temperature,
        }
        if request.top_k is not None:
            payload["top_k"] = request.top_k
        if request.repetition_penalty is not None:
            payload["repetition_penalty"] = request.repetition_penalty
        return payload

    def _clean_output(self, text: str) -> str:
        """Post-process the recognized text. Default: identity."""
        return text

    async def _send_request(self, image_b64: str, prompt: str) -> str:
        """Send OCR request with retry logic."""
        import httpx

        payload = self._build_payload(image_b64, prompt)
        label = self._SERVICE_LABEL
        url = f"{self._base_url}/v1/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(1 + self._MAX_RETRIES):
            try:
                resp = await self._client.post(url, json=payload)
                if resp.status_code in self._RETRYABLE_STATUS and attempt < self._MAX_RETRIES:
                    delay = min(
                        self._RETRY_BACKOFF_BASE * (2**attempt),
                        self._RETRY_BACKOFF_MAX,
                    )
                    logger.warning(
                        "%s retry %d/%d (status %d), backoff %.1fs",
                        label,
                        attempt + 1,
                        self._MAX_RETRIES,
                        resp.status_code,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                try:
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise ValueError(f"Unexpected {label} response structure: {exc}") from exc
                return OcrText(
                    self._clean_output(content),
                    finish_reason=choice.get("finish_reason"),
                )
            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                last_exc = e
                if attempt < self._MAX_RETRIES:
                    delay = min(
                        self._RETRY_BACKOFF_BASE * (2**attempt),
                        self._RETRY_BACKOFF_MAX,
                    )
                    logger.warning(
                        "%s retry %d/%d (error: %s), backoff %.1fs",
                        label,
                        attempt + 1,
                        self._MAX_RETRIES,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        assert last_exc is not None  # noqa: S101 — loop sets it on every retry path
        raise last_exc

    async def shutdown(self):
        """Close the HTTP client."""
        if not self._loaded:
            return
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._loaded = False
        logger.info("%s shut down", type(self).__name__)
