"""Managed Rapid-MLX runtime for Apple Silicon.

This module owns the local ``rapid-mlx serve`` subprocess used by the
experimental Apple-Silicon fast path:

- ``glm-rapid-mlx`` OCR serves GLM-OCR 8-bit through the existing HTTP OCR client.
- ``rapid-mlx`` LLM serves Qwen3.5 4B 4-bit with no-thinking defaults.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import ClassVar

from bibr.config import GlobalSettings, snapshot_settings
from bibr.local.http_runtime import (
    MANAGED_LOCAL_LLM_RATE_LIMIT_RPM,
    LocalHttpError,
    guard_managed_server_port,
    request_bytes,
)
from bibr.local.ocr import HttpOcrClient, PaddleHttpOcrClient
from bibr.ocr.registry import register

logger = logging.getLogger(__name__)

_TERM_GRACE_S = 10
_LOCAL_RAPID_MLX_DEFAULT_MAX_TOKENS = 8192
_RAPID_MLX_OCR_MAX_TOKENS = 16384

# Rapid-MLX's native MTP speculative decoding (vendored mlx-lm PR #990) is Qwen3.5/3.6-only,
# and additionally requires the checkpoint to ship MTP layers (mtp_num_hidden_layers >= 1 in
# config.json) — e.g. the default mlx-community/Qwen3.5-4B-MLX-4bit quant does NOT. In "auto"
# mode the launch falls back to --no-spec-decode when the server rejects the checkpoint.
_QWEN_MTP_MARKERS = ("qwen3.5", "qwen3.6", "qwen35", "qwen36")
_MTP_UNSUPPORTED_MARKERS = ("mtp_num_hidden_layers", "requires a qwen3.5 / qwen3.6 checkpoint")


def _resolve_spec_decode_args(model: str, spec_decode: str) -> list[str]:
    """Resolve ``--spec-decode`` flags for the text (non-multimodal) Rapid-MLX server."""
    mode = (spec_decode or "").strip().lower()
    if mode == "auto":
        model_lc = model.lower()
        if any(marker in model_lc for marker in _QWEN_MTP_MARKERS):
            return ["--spec-decode", "mtp"]
        return ["--no-spec-decode"]
    if mode == "mtp":
        return ["--spec-decode", "mtp"]
    if mode != "none":
        logger.warning(
            "Unrecognized rapid_mlx.spec_decode=%r; disabling speculative decoding", spec_decode
        )
    return ["--no-spec-decode"]


def _is_mtp_unsupported_error(exc: BaseException) -> bool:
    # The message excerpt is truncated for readability; match against the full
    # captured stderr tail when present so the marker survives INFO-log noise.
    text = f"{exc} {getattr(exc, 'stderr_tail', '')}".lower()
    return any(marker in text for marker in _MTP_UNSUPPORTED_MARKERS)


def _resolve_executable(
    executable: str | None = None,
    *,
    settings: GlobalSettings | None = None,
) -> str | None:
    effective = settings if settings is not None else snapshot_settings()
    raw = (executable or effective.rapid_mlx.executable).strip()
    if not raw:
        return None
    if "/" in raw:
        path = Path(raw).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
        return None
    return shutil.which(raw)


def rapid_mlx_unavailable_reason(
    executable: str | None = None,
    *,
    settings: GlobalSettings | None = None,
) -> str | None:
    """Return why Rapid-MLX cannot be launched, or ``None`` when it looks usable."""
    effective = settings if settings is not None else snapshot_settings()
    raw = (executable or effective.rapid_mlx.executable).strip() or "rapid-mlx"
    if _resolve_executable(raw, settings=effective) is None:
        return f"{raw!r} was not found or is not executable"
    return None


class RapidMlxServer:
    """Managed ``rapid-mlx serve`` subprocess.

    Constructor is blocking and should run in a worker thread. The subprocess
    inherits the parent environment, including ``HF_HOME`` / ``HF_HUB_CACHE`` so
    users can keep model files on an external drive.
    """

    def __init__(
        self,
        *,
        model: str,
        served_model_name: str | None = None,
        port: int,
        multimodal: bool,
        no_thinking: bool | None = None,
        prefill_step_size: int | None = None,
        max_tokens: int | None = None,
        extra_args: list[str] | None = None,
        strict_ocr_smoke: bool = False,
        settings=None,
    ) -> None:
        self._settings = settings if settings is not None else snapshot_settings()
        self._model = model
        self._served_model_name = served_model_name or model
        self._port = port
        self._multimodal = multimodal
        self._strict_ocr_smoke = strict_ocr_smoke
        self._process: subprocess.Popen | None = None
        self._stderr_fh = None
        self._stderr_log: Path | None = None
        self._reused = guard_managed_server_port(
            "ocr" if multimodal else "llm",
            base_url=self.base_url,
            model=self._served_model_name,
            server_label="Rapid-MLX",
            request_fn=request_bytes,
        )
        if self._reused:
            # A model-listing match proves only the alias. Paddle candidates
            # still need to prove image+prompt completion support; shutdown()
            # is a no-op here because this process is not ours.
            if self._strict_ocr_smoke:
                self._warmup_model()
            return

        exe = _resolve_executable(
            self._settings.rapid_mlx.executable,
            settings=self._settings,
        )
        if exe is None:
            from bibr.exceptions import UpstreamServiceError

            raise UpstreamServiceError(
                "rapid-mlx",
                "Rapid-MLX is not installed or is not executable "
                f"({rapid_mlx_unavailable_reason(self._settings.rapid_mlx.executable, settings=self._settings)}). "
                "Install with: pip install 'rapid-mlx[guided]'",
            )

        if no_thinking is None:
            no_thinking = not multimodal
        prefill_step_size = prefill_step_size or self._settings.rapid_mlx.prefill_step_size

        cmd = [
            exe,
            "serve",
            model,
            "--served-model-name",
            self._served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "INFO",
            "--max-num-seqs",
            str(self._settings.rapid_mlx.max_num_seqs),
            "--max-concurrent-requests",
            str(self._settings.rapid_mlx.max_concurrent_requests),
            "--prefill-step-size",
            str(prefill_step_size),
        ]
        if max_tokens is not None:
            cmd.extend(["--max-tokens", str(max_tokens)])
        if self._settings.rapid_mlx.force_disk_check:
            cmd.append("--force-disk-check")
        cmd.append("--mllm" if multimodal else "--no-mllm")
        if no_thinking:
            cmd.append("--no-thinking")
        spec_decode_mode = (self._settings.rapid_mlx.spec_decode or "").strip().lower()
        spec_args: list[str] = []
        if not multimodal:
            cmd.extend(["--no-tool-call-parser", "--no-reasoning-parser"])
            spec_args = _resolve_spec_decode_args(model, self._settings.rapid_mlx.spec_decode)
            cmd.extend(spec_args)
        if self._settings.rapid_mlx.pin_system_prompt:
            cmd.append("--pin-system-prompt")
        if extra_args:
            cmd.extend(extra_args)

        try:
            self._spawn_and_wait(cmd)
        except RuntimeError as exc:
            auto_mtp = spec_decode_mode == "auto" and spec_args == ["--spec-decode", "mtp"]
            if not (auto_mtp and _is_mtp_unsupported_error(exc)):
                raise
            logger.warning(
                "Rapid-MLX rejected MTP speculative decoding for %s (checkpoint has no MTP "
                "layers) — relaunching without it. Set RAPID_MLX_SPEC_DECODE=none to skip "
                "this attempt.",
                model,
            )
            idx = cmd.index("--spec-decode")
            cmd[idx : idx + 2] = ["--no-spec-decode"]
            self._spawn_and_wait(cmd)

    def _spawn_and_wait(self, cmd: list[str]) -> None:
        from bibr.utils.secure_temp import open_subprocess_log

        self._stderr_log, self._stderr_fh = open_subprocess_log("rapid-mlx", self._port)

        logger.info("Starting Rapid-MLX server: %s", " ".join(cmd))
        logger.info("Rapid-MLX stderr -> %s", self._stderr_log)
        self._process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_fh,
            start_new_session=True,
            env=self._subprocess_env(),
        )
        try:
            self._wait_until_healthy()
        except BaseException:
            # BaseException, not Exception: the child runs in its own session
            # and never sees the terminal's Ctrl-C, so a KeyboardInterrupt or
            # task cancellation out of the health wait must still shut it down.
            self.shutdown()
            raise

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def loaded(self) -> bool:
        return self._reused or (self._process is not None and self._process.poll() is None)

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self._settings.rapid_mlx.hf_home:
            env["HF_HOME"] = self._settings.rapid_mlx.hf_home
        if self._settings.rapid_mlx.hf_hub_cache:
            env["HF_HUB_CACHE"] = self._settings.rapid_mlx.hf_hub_cache
        if self._settings.rapid_mlx.home:
            env["HOME"] = self._settings.rapid_mlx.home
        return env

    def _wait_until_healthy(self) -> None:
        assert self._process is not None  # noqa: S101
        timeout = self._settings.rapid_mlx.startup_timeout
        deadline = time.monotonic() + timeout
        health_url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                rc = self._process.returncode
                tail = self._read_stderr_tail()
                self._close_stderr_fh()
                self._process = None
                err = RuntimeError(
                    f"Rapid-MLX process exited during startup (code {rc}): {tail[-500:]}"
                )
                err.stderr_tail = tail
                raise err
            try:
                status, _reason, _body = request_bytes(health_url, timeout=5)
                if status == 200:
                    self._warmup_model()
                    logger.info(
                        "Rapid-MLX server ready (model=%s, port=%d)",
                        self._model,
                        self._port,
                    )
                    return
            except LocalHttpError:
                pass
            time.sleep(2.0)

        try:
            self.shutdown()
        except Exception as cleanup_err:  # noqa: BLE001
            logger.warning("Rapid-MLX cleanup failed during timeout: %s", cleanup_err)
        raise TimeoutError(f"Rapid-MLX server did not become ready within {timeout}s")

    def _warmup_model(self) -> None:
        if self._multimodal:
            from PIL import Image, ImageDraw

            buf = io.BytesIO()
            image = Image.new("RGB", (128, 48), "white")
            if self._strict_ocr_smoke:
                ImageDraw.Draw(image).text((8, 16), "OCR OK", fill="black")
            image.save(buf, format="JPEG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            content: object = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": "OCR:" if self._strict_ocr_smoke else "Text Recognition:"},
            ]
        else:
            content = "OK"

        payload = json.dumps(
            {
                "model": self._served_model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 8,
                "temperature": 0.0,
            }
        ).encode("utf-8")
        warmup_url = f"{self.base_url}/v1/chat/completions"
        try:
            status, reason, _body = request_bytes(
                warmup_url,
                method="POST",
                body=payload,
                headers={"Content-Type": "application/json"},
                timeout=self._settings.rapid_mlx.warmup_timeout,
            )
        except LocalHttpError as exc:
            if self._strict_ocr_smoke:
                self._raise_strict_smoke_error(f"request failed: {exc}")
            logger.warning("Rapid-MLX warmup request failed: %s — first request may be slow", exc)
            return
        if 500 <= status < 600:
            if self._strict_ocr_smoke:
                self._raise_strict_smoke_error(f"returned HTTP {status} ({reason})")
            stderr_tail = self._read_stderr_tail()
            self.shutdown()
            from bibr.exceptions import UpstreamServiceError

            raise UpstreamServiceError(
                "ocr" if self._multimodal else "llm",
                (
                    f"Rapid-MLX warmup failed with HTTP {status} ({reason}). "
                    f"Last stderr from {self._stderr_log}:\n{stderr_tail[-500:]}"
                ),
            )
        if self._strict_ocr_smoke:
            if status >= 400:
                self._raise_strict_smoke_error(f"returned HTTP {status} ({reason})")
            self._validate_strict_smoke_response(_body)
        if status >= 400:
            logger.warning(
                "Rapid-MLX warmup returned %d %s — first request may be slow",
                status,
                reason,
            )

    def _raise_strict_smoke_error(self, detail: str) -> None:
        """Fail closed when a candidate cannot prove image OCR compatibility."""
        self.shutdown()
        from bibr.exceptions import UpstreamServiceError

        raise UpstreamServiceError("ocr", f"Rapid-MLX Paddle OCR smoke failed: {detail}")

    def _validate_strict_smoke_response(self, body: bytes) -> None:
        try:
            envelope = json.loads(body.decode("utf-8"))
            model = envelope["model"]
            content = envelope["choices"][0]["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            self._raise_strict_smoke_error(f"malformed response envelope: {exc}")
        if model != self._served_model_name:
            self._raise_strict_smoke_error(
                f"served wrong model {model!r}; expected {self._served_model_name!r}"
            )
        if (
            not isinstance(content, str)
            or "ocr" not in content.casefold()
            or "ok" not in content.casefold()
        ):
            self._raise_strict_smoke_error("response did not recognize the OCR OK image")

    def _read_stderr_tail(self, n_bytes: int = 8192) -> str:
        if self._stderr_log is None or not self._stderr_log.exists():
            return ""
        try:
            with open(self._stderr_log, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - n_bytes))
                return f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _close_stderr_fh(self) -> None:
        if self._stderr_fh is None:
            return
        try:
            self._stderr_fh.close()
        except OSError:
            pass
        self._stderr_fh = None

    def shutdown(self) -> None:
        if self._process is None:
            self._close_stderr_fh()
            return
        proc, self._process = self._process, None
        logger.info("Shutting down Rapid-MLX server (port=%d)...", self._port)
        try:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=_TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                proc.wait(timeout=5)
        finally:
            self._close_stderr_fh()
        logger.info("Rapid-MLX server shut down (port=%d)", self._port)


class _ManagedRapidMlxOcrClient:
    """Reusable lifecycle for managed multimodal Rapid-MLX OCR clients."""

    _model_option = "rapid_mlx_model"
    _port_option = "rapid_mlx_port"
    _extra_args_option = "rapid_mlx_extra_args"
    _strict_ocr_smoke = False

    def __init__(
        self,
        model_path: str | None = None,
        model: str | None = None,
        profile=None,
        settings=None,
        **_kw: object,
    ) -> None:
        self._settings = settings if settings is not None else snapshot_settings()
        # The factory passes the winning candidate's model as `model` and the
        # raw requested model as `model_path`: prefer the candidate, so a
        # fallback-chain winner serves its own model (same fix as
        # PaddleMlxVlmOcrClient).
        self._model = model or model_path or getattr(self._settings.ocr, self._model_option)
        self._profile = profile
        self._server, self._http_client = self._start_generation()
        # rapid-mlx's MLLM vision-embedding cache (vllm_mlx MLLMBatchGenerator ->
        # VisionEmbeddingCache) retains full float32 pixel tensors for up to 100
        # requests, evicted by entry COUNT only — no byte cap. bibr sends a unique
        # crop per region, so the cache never gets a hit and just accumulates ~100
        # dead ~34-72MB entries (~3.4GB) before OCR starts failing silently while
        # /health stays green. Recycling the subprocess periodically is the only
        # release valve short of a rapid-mlx-side fix. See
        # https://github.com/raullenchai/Rapid-MLX (vision_embedding_cache.py,
        # engine/batched.py hardcodes vision_cache_size=100 with no override).
        self._recycle_after = self._settings.ocr.rapid_mlx_recycle_after
        self._request_count = 0
        self._inflight = 0
        self._recycle_lock = asyncio.Lock()
        self._drain_event = asyncio.Event()
        self._drain_event.set()
        # Last restart failure, for waiters parked on the drain while a
        # recycle fails beneath them (cleared by every successful start).
        self._restart_error: Exception | None = None

    def _spawn_server(self) -> RapidMlxServer:
        return RapidMlxServer(
            model=self._model,
            served_model_name=self._model,
            port=getattr(self._settings.ocr, self._port_option),
            multimodal=True,
            no_thinking=False,
            max_tokens=_RAPID_MLX_OCR_MAX_TOKENS,
            extra_args=shlex.split(getattr(self._settings.ocr, self._extra_args_option) or ""),
            strict_ocr_smoke=self._strict_ocr_smoke,
            settings=self._settings,
        )

    def _spawn_http_client(self, server: RapidMlxServer) -> HttpOcrClient:
        if self._strict_ocr_smoke:
            return PaddleHttpOcrClient(
                base_url=server.base_url,
                model=self._model,
                profile=self._profile,
                settings=self._settings,
            )
        return HttpOcrClient(
            base_url=server.base_url,
            model=self._model,
            max_tokens=_RAPID_MLX_OCR_MAX_TOKENS,
            settings=self._settings,
        )

    def _start_generation(self) -> tuple[RapidMlxServer, HttpOcrClient]:
        """Construct one complete server/client generation transactionally."""
        server = self._spawn_server()
        try:
            return server, self._spawn_http_client(server)
        except Exception:
            server.shutdown()
            raise

    @property
    def loaded(self) -> bool:
        return self._server is not None and self._server.loaded and self._http_client is not None

    async def _ensure_generation(self) -> None:
        """Recover a missing/dead generation before accepting another request."""
        if self.loaded:
            return
        async with self._recycle_lock:
            if self.loaded:
                return
            loop = asyncio.get_running_loop()
            try:
                server, http_client = await loop.run_in_executor(None, self._start_generation)
            except Exception as exc:
                from bibr.exceptions import UpstreamServiceError

                raise UpstreamServiceError(
                    "rapid-mlx",
                    f"OCR restart failed: {exc}",
                    original_error=exc,
                ) from exc
            self._server, self._http_client = server, http_client
            self._request_count = 0
            self._restart_error = None

    async def recognize(self, image, prompt: str) -> str:
        await self._ensure_generation()
        await self._drain_event.wait()
        if self._http_client is None:
            # Parked on the drain while a recycle failed beneath us: the
            # recycler already surfaced the failure, and no generation is
            # usable — fail loudly instead of the old bare AssertionError.
            from bibr.exceptions import UpstreamServiceError

            raise UpstreamServiceError(
                "rapid-mlx",
                f"OCR restart failed while parked: {self._restart_error}",
            )
        self._inflight += 1
        try:
            result = await self._http_client.recognize(image, prompt)
        finally:
            self._inflight -= 1
        self._request_count += 1
        if self._recycle_after and self._request_count >= self._recycle_after:
            # The regions already transcribed stay valid even if the restart
            # below fails: recycle first, then return OUR result. A failed
            # restart leaves no usable generation, so only that case raises.
            await self._recycle()
        return result

    async def _recycle(self) -> None:
        """Restart the managed server to release rapid-mlx's uncapped vision cache."""
        async with self._recycle_lock:
            if self._request_count < self._recycle_after:
                return  # a concurrent caller already recycled
            self._drain_event.clear()
            try:
                while self._inflight > 0:
                    await asyncio.sleep(0.05)
                logger.info(
                    "Recycling Rapid-MLX OCR server after %d regions "
                    "(vision-cache leak mitigation, port=%d)",
                    self._request_count,
                    getattr(self._settings.ocr, self._port_option),
                )
                old_http = self._http_client
                old_server = self._server
                self._http_client = None
                self._server = None
                if old_http is not None:
                    await old_http.shutdown()
                loop = asyncio.get_running_loop()
                if old_server is not None:
                    await loop.run_in_executor(None, old_server.shutdown)
                try:
                    server, http_client = await loop.run_in_executor(None, self._start_generation)
                except Exception as exc:
                    from bibr.exceptions import UpstreamServiceError

                    failure = UpstreamServiceError(
                        "rapid-mlx",
                        f"OCR restart failed: {exc}",
                        original_error=exc,
                    )
                    # Parked waiters wake to a None client; stash the failure
                    # so they can name it instead of hitting a bare assert.
                    self._restart_error = failure
                    raise failure from exc
                self._server, self._http_client = server, http_client
                self._request_count = 0
                self._restart_error = None
            finally:
                self._drain_event.set()

    async def wait_for_server(self) -> None:
        await self._ensure_generation()

    async def shutdown(self) -> None:
        http_client = self._http_client
        server = self._server
        self._http_client = None
        self._server = None
        if http_client is not None:
            await http_client.shutdown()
        if server is not None:
            server.shutdown()


@register
class RapidMlxOcrClient(_ManagedRapidMlxOcrClient):
    """GLM OCR client using a managed Rapid-MLX subprocess on Apple Silicon."""

    name: ClassVar[str] = "glm-rapid-mlx"


@register
class PaddleRapidMlxOcrClient(_ManagedRapidMlxOcrClient):
    """PaddleOCR-VL client gated by a strict Rapid-MLX image smoke test."""

    name: ClassVar[str] = "paddle-rapid-mlx"
    _model_option = "paddle_rapid_mlx_model"
    # NOTE: the port/extra-args options below are shared with MlxVlmOcrServer's
    # own server (paddle-mlx-vlm backend): both read paddle_mlx_port /
    # paddle_mlx_extra_args, so running both backends in one process collides
    # on the port. Separate PADDLE_RAPID_MLX_* settings are deferred to an
    # owner decision (new knobs need docs + wizard surfacing pre-freeze); the
    # pipeline never selects both backends in one run today.
    _port_option = "paddle_mlx_port"
    _extra_args_option = "paddle_mlx_extra_args"
    _strict_ocr_smoke = True


class RapidMlxLlmServer:
    """Managed Rapid-MLX server for local LLM inference."""

    def __init__(self, model: str | None = None, settings=None) -> None:
        self._settings = settings if settings is not None else snapshot_settings()
        # An explicitly-set LLM_LOCAL_MODEL (e.g. written by `bibr setup`) wins
        # over the rapid-mlx alias default, so the served model never silently
        # diverges from what the user picked; unset defers to the alias.
        self._model = model or self._settings.llm.local_model or self._settings.llm.rapid_mlx_model
        max_tokens = (
            self._settings.llm.max_tokens
            if "max_tokens" in self._settings.llm.model_fields_set
            else _LOCAL_RAPID_MLX_DEFAULT_MAX_TOKENS
        )
        self._server = RapidMlxServer(
            model=self._model,
            served_model_name=self._model,
            port=self._settings.llm.rapid_mlx_port,
            multimodal=False,
            no_thinking=True,
            prefill_step_size=self._settings.rapid_mlx.prefill_step_size,
            max_tokens=max_tokens,
            extra_args=shlex.split(self._settings.llm.rapid_mlx_extra_args or ""),
            settings=self._settings,
        )

    def configure_llm_client(self) -> None:
        self._settings.llm.provider = "openai"
        self._settings.llm.base_url = self._server.base_url + "/v1"
        self._settings.llm.api_key = "not-needed"
        self._settings.llm.model = self._model
        if "max_tokens" not in self._settings.llm.model_fields_set:
            self._settings.llm.max_tokens = min(
                self._settings.llm.max_tokens,
                _LOCAL_RAPID_MLX_DEFAULT_MAX_TOKENS,
            )
            self._settings.llm.model_fields_set.add("max_tokens")
        if "timeout_seconds" not in self._settings.llm.model_fields_set:
            self._settings.llm.timeout_seconds = 300
        if "rate_limit_rpm" not in self._settings.llm.model_fields_set:
            # Loopback: the limiter must not throttle local inference.
            self._settings.llm.rate_limit_rpm = MANAGED_LOCAL_LLM_RATE_LIMIT_RPM
            logger.info(
                "Local rapid-mlx backend — raising llm.rate_limit_rpm to %d",
                MANAGED_LOCAL_LLM_RATE_LIMIT_RPM,
            )
        if "max_concurrency" not in self._settings.llm.model_fields_set:
            # Serialized by default despite server-side continuous batching: measured
            # 2026-07-09 on M4/16GB, 3 concurrent bibr-shaped calls ran 0.71x the speed
            # of serial — the calls are prefill-heavy, prefill serializes on the GPU,
            # and extra in-flight KV adds memory pressure. The server still admits 4
            # requests so operators with more unified memory can raise
            # LLM_MAX_CONCURRENCY without relaunching.
            self._settings.llm.max_concurrency = 1
        self._settings.llm.instructor_mode = "json"
        self._settings.llm.model_fields_set.add("instructor_mode")
        for field in ("reasoning_effort", "reasoning_effort_authors", "reasoning_effort_citations"):
            setattr(self._settings.llm, field, None)
        logger.info(
            "LLM configured: provider=openai, model=%s, base_url=%s",
            self._model,
            self._settings.llm.base_url,
        )

    def shutdown(self) -> None:
        if self._server:
            self._server.shutdown()
