"""Managed local LLM server for Apple Silicon via vllm-mlx.

Starts a ``VllmMlxServer`` serving the configured LLM model, then configures
its owning pipeline settings to route requests to the local endpoint.
"""

import logging

from bibr.config import snapshot_settings
from bibr.local.http_runtime import MANAGED_LOCAL_LLM_RATE_LIMIT_RPM

logger = logging.getLogger(__name__)


_LOCAL_VLLM_MLX_DEFAULT_MAX_TOKENS = 8192


class VllmMlxLlmServer:
    """Managed vllm-mlx server for local LLM inference.

    Created by ``LocalPipeline`` AFTER OCR is done — they share unified
    memory on Apple Silicon, so sequential loading is safer.
    """

    def __init__(
        self,
        model: str | None = None,
        settings=None,
    ):
        import shlex

        from bibr.local.ocr import VllmMlxServer

        self._settings = settings if settings is not None else snapshot_settings()
        self._model = model or self._settings.llm.local_model
        if not self._settings.llm.vllm_mlx_continuous_batching:
            logger.warning(
                "Local vllm-mlx LLM inference is slow on Apple Silicon (SimpleEngine, "
                "single-digit tokens/sec typical) — a full paper extraction can take "
                "15-30+ minutes. This is expected, not a hang."
            )
        port = self._settings.llm.vllm_mlx_port
        extra_args = shlex.split(self._settings.llm.vllm_mlx_extra_args or "")

        self._server = VllmMlxServer(
            model=self._model,
            port=port,
            continuous_batching=self._settings.llm.vllm_mlx_continuous_batching,
            multimodal=False,
            extra_args=extra_args,
            settings=self._settings,
        )

    def configure_llm_client(self):
        """Point the owning pipeline's LLM client at the local vllm-mlx server.

        Mutates only the settings object supplied at construction.
        """
        self._settings.llm.provider = "openai"
        self._settings.llm.base_url = self._server.base_url + "/v1"
        self._settings.llm.api_key = "not-needed"
        self._settings.llm.model = self._model
        if "max_tokens" not in self._settings.llm.model_fields_set:
            # Generic self-hosted OpenAI endpoints omit max_completion_tokens by
            # default because their max_model_len is unknown. Here bibr owns the
            # vllm-mlx endpoint, and leaving every structured request uncapped
            # lets the 0.4 BatchedEngine reserve/run toward its 32k default.
            self._settings.llm.max_tokens = min(
                self._settings.llm.max_tokens,
                _LOCAL_VLLM_MLX_DEFAULT_MAX_TOKENS,
            )
            self._settings.llm.model_fields_set.add("max_tokens")
            logger.info(
                "Local vllm-mlx backend — capping llm.max_tokens to %d",
                self._settings.llm.max_tokens,
            )
        if "timeout_seconds" not in self._settings.llm.model_fields_set:
            # Default (30s, ->60s hard timeout) is sized for cloud APIs. MLX SimpleEngine
            # on Apple Silicon serializes every request and can run single-digit tok/s
            # under concurrent load, so calls with non-trivial output (references,
            # authors) were getting killed by asyncio.wait_for before the model
            # finished, surfacing as httpx.ConnectError / empty results.
            self._settings.llm.timeout_seconds = 300
            logger.info("Local vllm-mlx backend — raising llm.timeout_seconds 30s->300s")
        if "max_concurrency" not in self._settings.llm.model_fields_set:
            # bibr fires several LLM calls concurrently (core metadata + equation
            # fallback + optional reference batches via asyncio.gather), but keep
            # managed local inference serialized by default. Measured 2026-07-09
            # on M4/16GB: 3 concurrent bibr-shaped calls ran 0.71x the speed of
            # serial — the calls are prefill-heavy, prefill serializes on the GPU,
            # and the extra in-flight KV only adds memory pressure. SimpleEngine
            # additionally crashes the Metal context when a queued call outlives
            # the server request timeout, and simultaneous constrained-JSON
            # requests can stall the BatchedEngine. Operators with more unified
            # memory can raise LLM_MAX_CONCURRENCY explicitly.
            self._settings.llm.max_concurrency = 1
            logger.info("Local vllm-mlx backend — capping llm.max_concurrency to 1")
        if "rate_limit_rpm" not in self._settings.llm.model_fields_set:
            # Serialized calls are already the bound here; the cloud-shaped RPM
            # default only adds dead time between them.
            self._settings.llm.rate_limit_rpm = MANAGED_LOCAL_LLM_RATE_LIMIT_RPM
            logger.info(
                "Local vllm-mlx backend — raising llm.rate_limit_rpm to %d",
                MANAGED_LOCAL_LLM_RATE_LIMIT_RPM,
            )
        logger.info(
            "LLM configured: provider=openai, model=%s, base_url=%s",
            self._model,
            self._settings.llm.base_url,
        )

    def shutdown(self):
        """Shutdown the vllm-mlx LLM server."""
        if self._server:
            self._server.shutdown()
