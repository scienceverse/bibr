"""Cloud vision-LLM OCR backends (Gemini, OpenAI, Anthropic).

Each provider is registered as a separate OCR backend via the standard
``bibr.ocr.registry``. The underlying ``CloudOcrClient`` uses Instructor's
``from_provider()`` with multimodal image support to transcribe cropped
document regions into text, HTML tables, or LaTeX formulas.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, ClassVar

from pydantic import BaseModel

from bibr.config import snapshot_settings
from bibr.ocr.registry import register

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vision prompts — richer instructions than GLM-OCR's terse task tags
# ---------------------------------------------------------------------------

_VISION_PROMPTS = {
    "text": (
        "Transcribe all text from this image region verbatim. Preserve line breaks "
        "exactly as they appear. Return only the transcribed text — no commentary, "
        "no markdown code fences, no explanations."
    ),
    "table": (
        "Transcribe this table as valid HTML using <table>, <tr>, <td>, <th> tags. "
        "Preserve cell contents, row/column structure, and any merged cells "
        "(rowspan/colspan). Return only the HTML — no commentary, no markdown code fences."
    ),
    "formula": (
        "Transcribe this mathematical formula as LaTeX. Return only the LaTeX source "
        "— no $$ delimiters, no commentary, no markdown code fences."
    ),
}

_PROMPT_MAP = {
    "Text Recognition:": "text",
    "Table Recognition:": "table",
    "Formula Recognition:": "formula",
}


class OcrResult(BaseModel):
    """Thin wrapper for structured output from the vision LLM."""

    text: str


# ---------------------------------------------------------------------------
# API key resolution per provider
# ---------------------------------------------------------------------------


def _api_key_for_provider(provider: str, settings: Any) -> str | None:
    if provider in ("google", "gemini"):
        return settings.llm.api_key or settings.GOOGLE_API_KEY
    if provider == "anthropic":
        return settings.ANTHROPIC_API_KEY
    if provider == "openai":
        # Preserve the env-driven path when no explicit key is configured, but
        # let injected settings own credentials for per-pipeline isolation.
        return settings.llm.api_key
    return None


_PROVIDER_STRINGS = {
    "gemini": "google",
    "google": "google",
    "openai": "openai",
    "anthropic": "anthropic",
}


class CloudOcrClient:
    """OCR backend that delegates to a cloud vision LLM via Instructor.

    Satisfies the ``OcrBackend`` protocol. Sends cropped region images
    through ``instructor.from_provider()`` with ``Image.from_raw_base64()``.
    """

    name: ClassVar[str] = "cloud"  # overridden by factory subclasses

    _MAX_RETRIES: ClassVar[int] = 3
    _RETRY_BASE_DELAY: ClassVar[float] = 1.0
    _RETRY_MAX_DELAY: ClassVar[float] = 8.0
    _RETRYABLE_STATUS: ClassVar[set[int]] = {429, 500, 502, 503, 504}
    # Systemic config/auth failures — every region will fail the same way.
    # Raise instead of degrading silently to blank content.
    _FATAL_STATUS: ClassVar[set[int]] = {401, 403, 404}

    def __init__(self, provider: str = "google", settings=None, **_kw: object):
        self._settings = settings if settings is not None else snapshot_settings()
        self._provider = provider
        self._client = None  # lazy-init
        self._init_lock = asyncio.Lock()
        self._loaded = True

        from bibr.utils.circuit_breaker import AsyncCircuitBreaker
        from bibr.utils.rate_limiter import AsyncLocalRateLimiter

        cfg = self._settings.ocr_vision
        interval = 60.0 / cfg.rate_limit_rpm

        self._rate_limiter = AsyncLocalRateLimiter(
            resource_id="ocr_vision",
            max_requests=1,
            window_seconds=interval,
        )
        self._breaker = AsyncCircuitBreaker(
            failure_threshold=self._settings.cb.failure_threshold,
            reset_timeout=self._settings.cb.reset_timeout_seconds,
            name="ocr_vision",
        )
        logger.info("CloudOcrClient ready (provider=%s)", provider)

    @property
    def _effective_settings(self):
        settings = getattr(self, "_settings", None)
        return settings if settings is not None else snapshot_settings()

    async def _aget_client(self):
        """Lazily create the async Instructor client (race-safe)."""
        if self._client is not None:
            return self._client
        async with self._init_lock:
            if self._client is not None:
                return self._client
            import instructor

            settings = self._effective_settings
            cfg = settings.ocr_vision
            instructor_provider = _PROVIDER_STRINGS.get(self._provider, self._provider)
            model_string = f"{instructor_provider}/{cfg.model}"

            kwargs: dict[str, Any] = {"async_client": True}
            api_key = _api_key_for_provider(self._provider, settings)
            if api_key is not None:
                kwargs["api_key"] = api_key
            if cfg.base_url:
                from bibr.utils.hosts import refuse_plaintext_llm_key

                # The vision endpoint gets the LLM provider's key, so the LLM
                # opt-out governs it (OCR_ALLOW_INSECURE_HTTP, on by default in
                # docker-compose, is for the OCR server's own token).
                refuse_plaintext_llm_key(
                    cfg.base_url, api_key, allow_insecure_http=settings.llm.allow_insecure_http
                )
                kwargs["base_url"] = cfg.base_url

            self._client = instructor.from_provider(model_string, **kwargs)
        return self._client

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _resolve_prompt(self, glm_prompt: str) -> str:
        """Map a GLM-OCR task prompt to a verbose vision-LLM prompt."""
        key = _PROMPT_MAP.get(glm_prompt, "text")
        return _VISION_PROMPTS[key]

    async def recognize(self, image: Any, prompt: str) -> str:
        """OCR a single cropped region image via the cloud vision LLM.

        Args:
            image: PIL Image of the cropped region.
            prompt: GLM-OCR task prompt (e.g. "Text Recognition:").

        Returns:
            Recognized text, or ``""`` on unrecoverable failure.
        """
        from instructor.processing.multimodal import Image as InstructorImage

        from bibr.ocr.image_utils import pil_to_base64_glmocr

        image_b64 = pil_to_base64_glmocr(image)
        vision_prompt = self._resolve_prompt(prompt)
        img = InstructorImage.from_raw_base64(image_b64)

        cfg = self._effective_settings.ocr_vision
        hard_timeout = float(cfg.timeout_seconds)
        client = await self._aget_client()

        # Google GenAI: Instructor translates max_tokens only when it's
        # inside generation_config (not as a top-level kwarg).  Also,
        # Instructor auto-injects HARM_CATEGORY_IMAGE_* safety settings
        # for image content, but the v1beta endpoint rejects them.
        if self._provider in ("google", "gemini"):
            extra_kwargs: dict = {
                "generation_config": {"max_tokens": cfg.max_tokens},
                "safety_settings": [],
            }
        else:
            extra_kwargs = {"max_tokens": cfg.max_tokens}

        last_exc: BaseException | None = None
        for attempt in range(self._MAX_RETRIES):
            is_last = attempt == self._MAX_RETRIES - 1
            try:
                # Re-acquire per attempt: a retry is a fresh API call, and
                # skipping the limiter on retries hammers the API exactly
                # when it is telling us to slow down (429).
                await self._rate_limiter.acquire()
                async with self._breaker:
                    result = await asyncio.wait_for(
                        client.create(
                            response_model=OcrResult,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [vision_prompt, img],
                                }
                            ],
                            **extra_kwargs,
                        ),
                        timeout=hard_timeout,
                    )
                    text: str = result.text
                    return text
            except Exception as exc:
                from bibr.clients.llm import http_status_in_chain

                last_exc = exc
                # Check if transient (retryable). Instructor wraps the SDK's
                # error, so the status is on the error it wraps.
                status = http_status_in_chain(exc)
                transient = isinstance(exc, TimeoutError) or (
                    status is not None and status in self._RETRYABLE_STATUS
                )
                fatal = status is not None and status in self._FATAL_STATUS
                if fatal:
                    from bibr.exceptions import UpstreamServiceError

                    logger.error(
                        "CloudOcrClient(%s) fatal config/auth error (status=%s): %s",
                        self._provider,
                        status,
                        exc,
                    )
                    raise UpstreamServiceError(
                        f"ocr_vision/{self._provider}",
                        f"fatal status {status} — backend misconfigured or unauthorized",
                        original_error=exc,
                    ) from exc
                if not transient or is_last:
                    logger.warning(
                        "CloudOcrClient(%s) failed after %d attempt(s): %s",
                        self._provider,
                        attempt + 1,
                        exc,
                    )
                    return ""

                delay = min(
                    self._RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 0.5),  # noqa: S311
                    self._RETRY_MAX_DELAY,
                )
                logger.warning(
                    "CloudOcrClient(%s) retry %d/%d: %s — backoff %.1fs",
                    self._provider,
                    attempt + 1,
                    self._MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        # Unreachable, but satisfies type checker
        logger.warning("CloudOcrClient(%s) exhausted retries: %s", self._provider, last_exc)
        return ""

    async def wait_for_server(self) -> None:
        """No-op — cloud providers are always available."""

    async def shutdown(self) -> None:
        """Release resources."""
        self._loaded = False
        self._client = None
        await self._rate_limiter.close()
        logger.info("CloudOcrClient(%s) shut down", self._provider)


# ---------------------------------------------------------------------------
# Register one backend per cloud provider
# ---------------------------------------------------------------------------


def _make_provider_backend(provider_name: str) -> type:
    """Create a CloudOcrClient subclass bound to a specific provider."""

    class _Cls(CloudOcrClient):
        name: ClassVar[str] = provider_name

        def __init__(self, **kw: object):
            super().__init__(provider=provider_name, **kw)

    _Cls.__name__ = f"CloudOcrClient_{provider_name}"
    _Cls.__qualname__ = f"CloudOcrClient_{provider_name}"
    return _Cls


for _provider in ("gemini", "openai", "anthropic"):
    register(_make_provider_backend(_provider))
