"""OCR clients for local pipeline.

HTTP clients serve external GLM/Paddle endpoints; ``LlamaCppOcrClient``
manages llama.cpp. ``VllmMlxServer`` supports the managed LLM
backend, while its retired OCR client only reports an actionable error.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

from bibr.config import GlobalSettings, snapshot_settings
from bibr.local.http_runtime import LocalHttpError, guard_managed_server_port, request_bytes
from bibr.local.ocr_transport import BaseHttpOcrClient
from bibr.ocr.registry import register

logger = logging.getLogger(__name__)


class VllmMlxServer:
    """Managed vllm-mlx subprocess server.

    Starts ``vllm-mlx serve <model>`` as a subprocess and polls ``/health``
    until the server is ready.  On ``shutdown()``, sends SIGTERM with a
    10-second grace period before SIGKILL — the OS reclaims all unified
    memory from the terminated process.

    Constructor is **blocking** — designed to run in a thread executor
    during layout detection.
    """

    def __init__(
        self,
        model: str,
        port: int,
        continuous_batching: bool = False,
        multimodal: bool = True,
        extra_args: list[str] | None = None,
        settings: GlobalSettings | None = None,
    ):
        self._settings = settings if settings is not None else snapshot_settings()
        self._port = port
        self._model = model
        self._multimodal = multimodal
        self._process: subprocess.Popen | None = None
        self._stderr_fh = None
        self._stderr_log: Path | None = None
        self._reused = guard_managed_server_port(
            "ocr" if multimodal else "llm",
            base_url=self.base_url,
            model=model,
            server_label="vllm-mlx",
            request_fn=request_bytes,
        )
        if self._reused:
            return

        # Pre-flight: check that ``vllm_mlx.server`` is runnable before spawning
        # the subprocess. Stale namespace directories can make
        # ``find_spec("vllm_mlx")`` pass even when the package is gone.
        from bibr.local.vllm_mlx_runtime import vllm_mlx_unavailable_reason

        unavailable_reason = vllm_mlx_unavailable_reason()
        if unavailable_reason is not None:
            from bibr.exceptions import UpstreamServiceError

            raise UpstreamServiceError(
                "ocr",
                "vllm-mlx is not installed or is incomplete "
                f"({unavailable_reason}). Install with one of:\n"
                "  uv sync --extra local-mlx       # if using uv (recommended)\n"
                "  uv pip install vllm-mlx         # if using uv pip\n"
                "  pip install vllm-mlx            # if using plain pip",
            )

        # Use ``python -m vllm_mlx.server`` which supports ``--mllm`` (force
        # multimodal/VLM inference path via mlx-vlm).  The ``vllm-mlx serve``
        # CLI subcommand does not expose ``--mllm`` in all versions. Text-only
        # LLM callers must leave this off so they use mlx-lm rather than paying
        # for the vision-model path; OCR callers keep the historical default.
        # Note: ``server.py`` does not support ``--cache-memory-mb`` — the VLM
        # path loads the full model into unified memory without explicit cap.
        cmd = [
            sys.executable,
            "-m",
            # Stable shim around ``vllm_mlx.server``. It keeps conservative
            # hybrid-cache handling for stale environments while matching the
            # validated 0.4.x server entrypoint. See bibr/local/_vllm_mlx_server.py.
            "bibr.local._vllm_mlx_server",
            "--model",
            model,
            "--port",
            str(port),
        ]
        if multimodal:
            cmd.append("--mllm")
        if continuous_batching:
            cmd.append("--continuous-batching")
        if extra_args:
            cmd.extend(extra_args)

        # vllm-mlx emits ~5 log lines per OCR request via uvicorn + FastAPI
        # + its own logger. Routing stderr through ``subprocess.PIPE`` (and
        # never reading it) deadlocks the server: once the OS pipe buffer
        # fills (~16 KB on macOS), the server's logging.write() blocks the
        # request-handler thread inside a write(2) syscall, and the next
        # chat completion never returns. Write to a file instead so logs
        # accumulate without blocking and remain available for diagnostics.
        from bibr.utils.secure_temp import open_subprocess_log

        self._stderr_log, self._stderr_fh = open_subprocess_log("vllm-mlx", port)

        logger.info("Starting vllm-mlx server: %s", " ".join(cmd))
        logger.info("vllm-mlx stderr -> %s", self._stderr_log)
        self._process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_fh,
        )

        # Poll /health until ready or timeout
        timeout = self._settings.vllm_mlx.startup_timeout
        deadline = time.monotonic() + timeout
        poll_interval = 2.0
        health_url = f"http://localhost:{port}/health"

        while time.monotonic() < deadline:
            # Check process didn't die during startup
            if self._process.poll() is not None:
                rc = self._process.returncode
                stderr_tail = self._read_stderr_tail()
                self._close_stderr_fh()
                self._process = None
                raise RuntimeError(
                    f"vllm-mlx process exited during startup (code {rc}): {stderr_tail[:500]}"
                )
            try:
                status, _reason, raw_body = request_bytes(health_url, timeout=5)
                if status == 200:
                    body = json.loads(raw_body.decode("utf-8"))
                    # /health returns 200 as soon as the FastAPI app is up,
                    # but the MLX engine may still be mmapping weights. Wait
                    # for ``model_loaded: true`` — otherwise the first OCR
                    # request pays the full paging cost and can blow past
                    # the per-request read-timeout.
                    if body.get("model_loaded"):
                        logger.info("vllm-mlx server ready (model=%s, port=%d)", model, port)
                        self._warmup_model()
                        return
            except (LocalHttpError, json.JSONDecodeError):
                pass
            time.sleep(poll_interval)

        # Timeout — kill the process, but never let cleanup mask the TimeoutError.
        try:
            self.shutdown()
        except Exception as cleanup_err:  # noqa: BLE001
            logger.warning("vllm-mlx cleanup failed during timeout: %s", cleanup_err)
        raise TimeoutError(f"vllm-mlx server did not become ready within {timeout}s")

    def _warmup_model(self) -> None:
        """Force the MLX engine to actually page weights into compute memory.

        ``model_loaded: true`` only guarantees ``mlx.load()`` finished (which is
        mmap, not paging). For multimodal OCR models the vision encoder pages
        in only when the request contains an image, so a text-only warmup
        leaves the first real OCR call paying the 30-60 s cost — long enough
        to exceed the 90 s per-region timeout when stacked behind layout work.

        We send one tiny warmup chat-completion mirroring the real request
        shape for this server — image+text for multimodal (OCR) servers,
        text-only for text (LLM) servers. Sending an image to a server started
        without ``--mllm`` (no vision encoder loaded) has previously hung the
        request for the full ``warmup_timeout`` (300s) instead of failing
        fast — every ``--llm local`` run was paying that tax on top of actual
        inference.
        """
        if self._multimodal:
            import base64
            import io as _io

            from PIL import Image

            buf = _io.BytesIO()
            Image.new("RGB", (32, 32), "white").save(buf, format="JPEG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            content: object = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": "Text Recognition:"},
            ]
        else:
            content = "OK"

        warmup_url = f"http://localhost:{self._port}/v1/chat/completions"
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 4,
                "temperature": 0.0,
            }
        ).encode("utf-8")
        warmup_timeout = self._settings.vllm_mlx.warmup_timeout
        t0 = time.monotonic()
        try:
            status, reason, _body = request_bytes(
                warmup_url,
                method="POST",
                body=payload,
                headers={"Content-Type": "application/json"},
                timeout=warmup_timeout,
            )
        except LocalHttpError as e:
            logger.warning("vllm-mlx warmup request failed: %s — first request may be slow", e)
            return
        if 500 <= status < 600:
            # 5xx during warmup means the engine crashed processing the
            # warmup request. Subsequent requests will fail the same way, so
            # tear down and raise — don't pretend we're ready.
            stderr_tail = self._read_stderr_tail()
            logger.warning(
                "vllm-mlx warmup returned %d %s — engine appears broken; tearing down",
                status,
                reason,
            )
            from bibr.exceptions import UpstreamServiceError

            self.shutdown()
            what = "images" if self._multimodal else "requests"
            raise UpstreamServiceError(
                "ocr" if self._multimodal else "llm",
                (
                    f"vllm-mlx warmup failed with HTTP {status} ({reason}). "
                    f"The engine started but cannot process {what}. "
                    f"Last stderr from {self._stderr_log}:\n{stderr_tail[-500:]}"
                ),
            )
        if status >= 400:
            logger.warning(
                "vllm-mlx warmup returned %d %s — first request may be slow",
                status,
                reason,
            )
            return
        logger.info("vllm-mlx warmup complete (took %.1fs)", time.monotonic() - t0)

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def loaded(self) -> bool:
        return self._reused or (self._process is not None and self._process.poll() is None)

    def _read_stderr_tail(self, n_bytes: int = 8192) -> str:
        """Read the last ``n_bytes`` of the stderr log file for error reporting."""
        if self._stderr_log is None or not self._stderr_log.exists():
            return ""
        try:
            with open(self._stderr_log, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - n_bytes))
                return f.read().decode(errors="replace")
        except Exception:  # noqa: BLE001
            return ""

    def _close_stderr_fh(self) -> None:
        """Close the stderr log file handle (idempotent)."""
        if self._stderr_fh is None:
            return
        try:
            self._stderr_fh.close()
        except Exception:  # noqa: BLE001, S110
            pass
        self._stderr_fh = None

    def shutdown(self):
        """Terminate the vllm-mlx subprocess."""
        if self._process is None:
            self._close_stderr_fh()
            return
        logger.info("Shutting down vllm-mlx server (port=%d)...", self._port)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("vllm-mlx did not exit in 10s, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None
        self._close_stderr_fh()
        logger.info("vllm-mlx server shut down (port=%d)", self._port)


def raise_vllm_mlx_ocr_disabled(backend_name: str) -> None:
    """Refuse to construct the disabled vllm-mlx-backed GLM OCR client.

    The vllm-mlx ``--mllm`` OCR path is disabled entirely, not just warned
    about: on 2026-07-09 the validated vllm-mlx 0.4 line emitted NUL-riddled,
    character-lossy region text on a real page (glm-rapid-mlx OCR'd the same
    page correctly), and independently its MLLM vision-embedding cache
    retains every request's pixel tensors under a 100-entry count cap with no
    byte cap — bibr's unique-crop-per-region workload never hits it, so ~100
    dead ~34-72MB entries accumulate to ~3.4GB and OCR then fails silently
    while ``/health`` stays green. Use ``glm-rapid-mlx`` (rapid-mlx) instead.
    """
    from bibr.exceptions import UpstreamServiceError

    raise UpstreamServiceError(
        "ocr",
        f"The {backend_name!r} OCR backend (vllm-mlx --mllm path) is disabled: it has "
        "produced NUL-riddled corrupted text and, independently, leaks ~3.4GB through an "
        "uncapped vision-embedding cache that never gets a hit on bibr's unique-crop-per-"
        "region workload. Use --ocr glm-rapid-mlx instead: pip install 'rapid-mlx[guided]'",
    )


@register
class VllmMlxOcrClient:
    """Compatibility entry point for the retired ``glm-mlx`` OCR backend."""

    name: ClassVar[str] = "glm-mlx"

    def __init__(self, **_kw: object):
        raise_vllm_mlx_ocr_disabled(self.name)


@register
class LlamaCppOcrClient:
    """Native Windows and low-VRAM GLM-OCR backend through llama.cpp."""

    name: ClassVar[str] = "glm-llama"

    def __init__(
        self,
        model_path: str | None = None,
        *,
        settings: GlobalSettings | None = None,
        **_kw: object,
    ):
        from bibr.local.llama_cpp import LlamaCppServer

        effective = settings if settings is not None else snapshot_settings()
        model = model_path or effective.ocr.llama_cpp_model
        self._server = LlamaCppServer(
            model=model,
            port=effective.ocr.llama_cpp_port,
            context_size=effective.ocr.llama_cpp_context_size,
            startup_timeout=effective.ocr.llama_cpp_startup_timeout,
            extra_args=effective.ocr.llama_cpp_extra_args,
            role="ocr",
        )

        # The managed llama.cpp OCR server serves exactly one slot
        # (--parallel 1, see LlamaCppServer's "ocr" role). Client-side
        # concurrency deeper than that just queues regions server-side
        # against the 90s read timeout; a timed-out request is then retried
        # by resending the same image, amplifying load on the single slot.
        # Clamp both concurrency knobs to 1 — but only where the user hasn't
        # pinned them explicitly, since explicit env settings win. Gate on
        # ``user_set_concurrency`` (snapshotted before construction-time
        # auto-tuning), NOT ``model_fields_set`` — the auto-tune assignments
        # in ``GlobalSettings.compute_ocr_concurrency`` mark these fields as
        # set on every real Settings instance.
        user_set = effective.ocr.user_set_concurrency
        clamped_fields = []
        for field in ("concurrent_regions_per_file", "max_concurrent_regions"):
            if field in user_set or getattr(effective.ocr, field) == 1:
                continue
            setattr(effective.ocr, field, 1)
            clamped_fields.append(field)
        if clamped_fields:
            logger.info(
                "llama.cpp OCR server has a single slot — clamping %s to 1",
                " and ".join(clamped_fields),
            )

        self._http_client = HttpOcrClient(
            base_url=self._server.base_url,
            model=model,
            max_tokens=min(4096, effective.ocr.llama_cpp_context_size // 2),
            settings=effective,
        )

    @property
    def loaded(self) -> bool:
        return self._server.loaded

    async def recognize(self, image, prompt: str) -> str:
        return await self._http_client.recognize(image, prompt)

    async def wait_for_server(self) -> None:
        return None

    async def shutdown(self) -> None:
        await self._http_client.shutdown()
        self._server.shutdown()


@register
class HttpOcrClient(BaseHttpOcrClient):
    """HTTP-based OCR client for external servers.

    Talks to an external SGLang/mlx-vlm/Ollama server over HTTP using the
    OpenAI-compatible chat completions API.  Includes retry logic with
    exponential backoff and circuit breaker integration.

    Used when ``--ocr-url`` is specified, and is the supported route to a
    self-hosted SGLang server now that bibr does not depend on sglang.
    """

    name: ClassVar[str] = "glm-http"
    # "glm-ocr" is the documented alias convention for externally-managed
    # servers (sglang/ollama/etc. started with that served name). Callers
    # that manage their own server under a different name (e.g. the vllm-mlx
    # subprocess, served under its real HF repo id) must pass it explicitly —
    # see VllmMlxOcrClient.
    _DEFAULT_MODEL: ClassVar[str] = "glm-ocr"
    _SERVICE_LABEL: ClassVar[str] = "OCR"


@register
class PaddleHttpOcrClient(BaseHttpOcrClient):
    """HTTP client for external PaddleOCR-VL OpenAI-compatible servers."""

    name: ClassVar[str] = "paddle-http"
    _DEFAULT_MODEL: ClassVar[str] = "paddle-ocr-vl-1.6"
    _SERVICE_LABEL: ClassVar[str] = "Paddle OCR"
