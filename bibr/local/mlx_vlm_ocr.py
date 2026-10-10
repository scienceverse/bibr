"""Managed PaddleOCR-VL runtime for Apple Silicon through MLX-VLM."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar

from bibr.config import GlobalSettings, snapshot_settings
from bibr.local.http_runtime import (
    LocalHttpError,
    check_startup_stop,
    guard_managed_server_port,
    kill_process_group,
    pause_startup_poll,
    request_bytes,
)
from bibr.local.ocr import PaddleHttpOcrClient
from bibr.ocr.registry import register

logger = logging.getLogger(__name__)

_TERM_GRACE_S = 10
_MLX_VLM_VERSION = "0.6.3"


class MlxVlmOcrServer:
    """Own a PaddleOCR-VL MLX-VLM server and prove its image OCR capability."""

    # Set by the owner to stop the startup wait early (see ResourceManager).
    _stop_event: threading.Event | None = None

    def __init__(
        self,
        model: str | None = None,
        port: int | None = None,
        settings: GlobalSettings | None = None,
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._stop_event = stop_event
        self._settings = settings if settings is not None else snapshot_settings()
        self._model = model or self._settings.ocr.paddle_mlx_model
        self._port = port if port is not None else self._settings.ocr.paddle_mlx_port
        self._process: subprocess.Popen | None = None
        self._stderr_fh = None
        self._stderr_log: Path | None = None
        self._reused = guard_managed_server_port(
            "ocr",
            base_url=self.base_url,
            model=self._model,
            server_label="PaddleOCR-VL MLX-VLM",
            request_fn=request_bytes,
            refuse_reuse=self._rapid_mlx_listener_refusal,
        )
        if self._reused:
            # A model-listing match alone does not prove image+prompt OCR.
            # Do not call shutdown here: this listener is owned elsewhere.
            self._run_smoke()
            return

        cmd = self._resolve_launch_cmd(self._model)
        cmd.extend(["--host", "127.0.0.1", "--port", str(self._port)])
        cmd.extend(shlex.split(self._settings.ocr.paddle_mlx_extra_args or ""))

        from bibr.utils.secure_temp import open_subprocess_log

        self._stderr_log, self._stderr_fh = open_subprocess_log("mlx-vlm-ocr", self._port)
        logger.info("Starting PaddleOCR-VL MLX-VLM server: %s", " ".join(cmd))
        logger.info("PaddleOCR-VL MLX-VLM stderr -> %s", self._stderr_log)
        try:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_fh,
                start_new_session=True,
            )
            self._wait_until_ready()
        except BaseException:
            self.shutdown()
            raise

    def _rapid_mlx_listener_refusal(self) -> str | None:
        """Refuse a Rapid-MLX or vllm-mlx server that holds the shared Paddle port.

        paddle-rapid-mlx uses the same port and model, and refuses a server it
        did not start because only its restarts relieve Rapid-MLX's vision-cache
        leak. Adopting that server here, with no restarts, would bring the leak
        back. Their ``/health`` reports an ``engine_type``; MLX-VLM's does not.
        """
        try:
            status, _reason, body = request_bytes(f"{self.base_url}/health", timeout=5)
            health = json.loads(body.decode("utf-8")) if status == 200 else None
        except (LocalHttpError, UnicodeDecodeError, ValueError):
            return None
        if not isinstance(health, dict) or "engine_type" not in health:
            return None
        return (
            "it is a Rapid-MLX or vllm-mlx server (its /health reports an engine_type), "
            "not MLX-VLM, most likely a paddle-rapid-mlx server left over from an earlier "
            "run. Its vision cache leaks memory until OCR fails unless the server is "
            "restarted regularly, which only paddle-rapid-mlx does, and only for a server "
            "it started. `--ocr paddle-rapid-mlx` with OCR_RAPID_MLX_RECYCLE_AFTER=0 uses "
            "it anyway, without restarts."
        )

    @staticmethod
    def _resolve_launch_cmd(model: str) -> list[str]:
        """Prefer an installed server, otherwise isolate the audited MLX-VLM tool."""
        executable = shutil.which("mlx_vlm.server")
        if executable is not None:
            return [executable, "--model", model]
        try:
            installed = importlib.util.find_spec("mlx_vlm.server") is not None
        except ModuleNotFoundError:
            installed = False
        if installed:
            # -P: never import from the user's working directory.
            return [sys.executable, "-P", "-m", "mlx_vlm.server", "--model", model]
        if shutil.which("uv") is not None:
            return [
                "uv",
                "tool",
                "run",
                "--from",
                f"mlx-vlm=={_MLX_VLM_VERSION}",
                "mlx_vlm.server",
                "--model",
                model,
            ]

        from bibr.exceptions import UpstreamServiceError

        raise UpstreamServiceError(
            "ocr",
            "mlx-vlm is not installed and uv is unavailable. Install uv so bibr can launch "
            "the pinned isolated mlx-vlm==0.6.3 runtime, or install mlx-vlm separately.",
        )

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def loaded(self) -> bool:
        return self._reused or (self._process is not None and self._process.poll() is None)

    def _wait_until_ready(self) -> None:
        assert self._process is not None  # noqa: S101
        deadline = time.monotonic() + self._settings.ocr.paddle_mlx_startup_timeout
        health_url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            check_startup_stop(self._stop_event, "PaddleOCR-VL MLX-VLM")
            if self._process.poll() is not None:
                rc = self._process.returncode
                tail = self._read_stderr_tail()
                kill_process_group(self._process.pid)
                self._process = None
                self._close_stderr_fh()
                raise RuntimeError(
                    f"PaddleOCR-VL MLX-VLM process exited during startup (code {rc}): {tail[-500:]}"
                )
            try:
                status, _reason, _body = request_bytes(health_url, timeout=5)
                if status == 200:
                    self._run_smoke()
                    logger.info(
                        "PaddleOCR-VL MLX-VLM ready (model=%s, port=%d)", self._model, self._port
                    )
                    return
            except LocalHttpError:
                pass
            pause_startup_poll(2.0, self._stop_event, "PaddleOCR-VL MLX-VLM")
        raise TimeoutError(
            f"PaddleOCR-VL MLX-VLM server did not become ready within "
            f"{self._settings.ocr.paddle_mlx_startup_timeout}s"
        )

    def _run_smoke(self) -> None:
        """Require image+prompt OCR, rather than mistaking a listening port for support."""
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (128, 48), "white")
        ImageDraw.Draw(image).text((8, 16), "OCR OK", fill="black")

        async def _smoke_request() -> str:
            # The client's httpx pool binds to the loop that created it, so
            # its teardown must run in that same loop. Closing it from a
            # second ``asyncio.run`` raised "Event loop is closed" out of the
            # ``finally``, replacing a successful smoke result and making the
            # backend impossible to start.
            # MLX-VLM runs without a key: never send it the user's OCR_API_KEY.
            client = PaddleHttpOcrClient(
                base_url=self.base_url, model=self._model, settings=self._settings, api_key=""
            )
            try:
                return await client.recognize(image, "OCR:")
            finally:
                await client.shutdown()

        try:
            result = asyncio.run(_smoke_request())
        except Exception as exc:
            self._raise_smoke_error(f"request failed: {exc}")
        if (
            not isinstance(result, str)
            or "ocr" not in result.casefold()
            or "ok" not in result.casefold()
        ):
            self._raise_smoke_error("response did not recognize the OCR OK image")

    def _raise_smoke_error(self, detail: str) -> None:
        from bibr.exceptions import UpstreamServiceError

        raise UpstreamServiceError("ocr", f"MLX-VLM Paddle OCR smoke failed: {detail}")

    def _read_stderr_tail(self, n_bytes: int = 8192) -> str:
        if self._stderr_log is None or not self._stderr_log.exists():
            return ""
        try:
            return self._stderr_log.read_bytes()[-n_bytes:].decode("utf-8", errors="replace")
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
        process, self._process = self._process, None
        try:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                process.terminate()
            try:
                process.wait(timeout=_TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    process.kill()
                process.wait(timeout=5)
        finally:
            self._close_stderr_fh()
        logger.info("PaddleOCR-VL MLX-VLM server shut down")


@register
class PaddleMlxVlmOcrClient:
    """Managed MLX-VLM PaddleOCR-VL backend."""

    name: ClassVar[str] = "paddle-mlx-vlm"

    def __init__(
        self,
        model_path: str | None = None,
        model: str | None = None,
        profile=None,
        settings: GlobalSettings | None = None,
        stop_event: threading.Event | None = None,
        **_kw: object,
    ) -> None:
        effective = settings if settings is not None else snapshot_settings()
        # The factory passes the raw requested model as `model_path` and the
        # winning candidate's model as `model`: the explicit `--ocr-model`
        # request wins, as on main; the candidate is the default when nothing
        # was requested.
        requested_model = model_path or model
        self._server = MlxVlmOcrServer(
            model=requested_model, settings=effective, stop_event=stop_event
        )
        try:
            self._http_client = PaddleHttpOcrClient(
                base_url=self._server.base_url,
                model=requested_model or effective.ocr.paddle_mlx_model,
                profile=profile,
                settings=effective,
                api_key="",
            )
        except BaseException:
            self._server.shutdown()
            raise

    @property
    def loaded(self) -> bool:
        return self._server.loaded

    async def recognize(self, image, prompt: str) -> str:
        return await self._http_client.recognize(image, prompt)

    async def wait_for_server(self) -> None:
        return None

    async def shutdown(self) -> None:
        try:
            await self._http_client.shutdown()
        finally:
            self._server.shutdown()
