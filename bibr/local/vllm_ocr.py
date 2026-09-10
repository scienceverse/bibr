"""Managed local PaddleOCR-VL server for Linux/CUDA via vLLM."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

from bibr.config import GlobalSettings, snapshot_settings
from bibr.local.http_runtime import LocalHttpError, guard_managed_server_port, request_bytes
from bibr.local.ocr import PaddleHttpOcrClient
from bibr.ocr.registry import register

logger = logging.getLogger(__name__)

_TERM_GRACE_S = 10


class VllmOcrServer:
    """Own a PaddleOCR-VL vLLM process and wait for its exact served alias."""

    def __init__(
        self,
        model: str | None = None,
        port: int | None = None,
        settings: GlobalSettings | None = None,
    ) -> None:
        self._settings = settings if settings is not None else snapshot_settings()
        self._model = model or self._settings.ocr.paddle_model
        self._served_model = self._settings.ocr.paddle_served_model
        self._port = port if port is not None else self._settings.ocr.paddle_vllm_port
        self._process: subprocess.Popen | None = None
        self._stderr_fh = None
        self._stderr_log: Path | None = None
        self._reused = guard_managed_server_port(
            "ocr",
            base_url=self.base_url,
            model=self._served_model,
            server_label="PaddleOCR-VL vLLM",
            request_fn=request_bytes,
        )
        if self._reused:
            return

        from bibr.ocr.registry import paddle_vllm_unavailable_reason

        hardware_blocker = paddle_vllm_unavailable_reason()
        if hardware_blocker is not None:
            from bibr.exceptions import UpstreamServiceError

            raise UpstreamServiceError(
                "ocr",
                f"Cannot start the managed PaddleOCR-VL vLLM server: {hardware_blocker}. "
                "Use --ocr glm-llama (llama.cpp), point --ocr-url at an external OCR "
                "server, or choose a cloud vision backend (--ocr gemini|openai|anthropic).",
            )

        cmd = self._resolve_launch_cmd(self._model)
        cmd.extend(
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(self._port),
                "--revision",
                self._settings.ocr.paddle_revision,
                "--served-model-name",
                self._served_model,
                "--gpu-memory-utilization",
                "0.92",
                "--max-model-len",
                "16384",
                "--max-num-seqs",
                "12",
                "--max-num-batched-tokens",
                "16384",
                "--no-enable-prefix-caching",
                "--mm-processor-cache-gb",
                "0",
            ]
        )
        cmd.extend(shlex.split(self._settings.ocr.paddle_vllm_extra_args or ""))

        from bibr.utils.secure_temp import open_subprocess_log

        self._stderr_log, self._stderr_fh = open_subprocess_log("vllm-ocr", self._port)
        logger.info("Starting PaddleOCR-VL vLLM server: %s", " ".join(cmd))
        logger.info("PaddleOCR-VL vLLM stderr -> %s", self._stderr_log)
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

    @staticmethod
    def _resolve_launch_cmd(model: str) -> list[str]:
        """Resolve a vLLM launcher, preferring an installed vLLM module."""
        if importlib.util.find_spec("vllm") is not None:
            return [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model,
            ]
        if shutil.which("uv") is not None:
            logger.warning(
                "vLLM is not installed in this environment; launching PaddleOCR-VL through an "
                "isolated `uv tool run --from vllm==0.27.0` environment instead. The first run "
                "downloads several GB and can take minutes before OCR starts. Install it once "
                "with `uv sync --extra vllm` to skip this bootstrap."
            )
            cmd = ["uv", "tool", "run"]
            if sys.version_info >= (3, 14):
                # vllm==0.27.0 publishes no 3.14 wheels; the isolated tool
                # environment can run a managed 3.13 interpreter instead.
                cmd.extend(["--python", "3.13"])
            cmd.extend(
                ["--from", "vllm==0.27.0", "--with", "openai>=2.54.0,<3", "vllm", "serve", model]
            )
            return cmd

        from bibr.exceptions import UpstreamServiceError

        raise UpstreamServiceError(
            "ocr",
            "vllm is not installed and uv is unavailable. Install uv (recommended) so bibr "
            "can launch PaddleOCR-VL in an isolated vLLM environment, or install vllm into "
            "this environment.",
        )

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def loaded(self) -> bool:
        return self._reused or (self._process is not None and self._process.poll() is None)

    def _wait_until_ready(self) -> None:
        """Poll the vLLM model listing until it advertises our served alias."""
        assert self._process is not None  # noqa: S101
        timeout = self._settings.ocr.paddle_vllm_startup_timeout
        deadline = time.monotonic() + timeout
        models_url = f"{self.base_url}/v1/models"

        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                rc = self._process.returncode
                tail = self._read_stderr_tail()
                self._process = None
                self._close_stderr_fh()
                raise RuntimeError(
                    f"PaddleOCR-VL vLLM process exited during startup (code {rc}): {tail[:500]}"
                )
            try:
                status, _reason, body = request_bytes(models_url, timeout=5)
                if status == 200:
                    data = json.loads(body.decode("utf-8"))
                    ids = [
                        str(item["id"])
                        for item in data.get("data", [])
                        if isinstance(item, dict) and "id" in item
                    ]
                    if self._served_model in ids:
                        logger.info(
                            "PaddleOCR-VL vLLM ready (model=%s, alias=%s, port=%d)",
                            self._model,
                            self._served_model,
                            self._port,
                        )
                        return
            except (LocalHttpError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                pass
            time.sleep(2.0)

        try:
            self.shutdown()
        except Exception as cleanup_err:  # noqa: BLE001
            logger.warning(
                "PaddleOCR-VL vLLM cleanup failed during startup timeout: %s", cleanup_err
            )
        raise TimeoutError(f"PaddleOCR-VL vLLM server did not become ready within {timeout}s")

    def _read_stderr_tail(self, n: int = 2000) -> str:
        if self._stderr_log is None or not self._stderr_log.exists():
            return ""
        try:
            return self._stderr_log.read_bytes()[-n:].decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _close_stderr_fh(self) -> None:
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except OSError:
                pass
            self._stderr_fh = None

    def shutdown(self) -> None:
        """Terminate the owned process group and close its log handle."""
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
        logger.info("PaddleOCR-VL vLLM server shut down")


@register
class PaddleVllmOcrClient:
    """Managed PaddleOCR-VL vLLM backend using the common Paddle HTTP client."""

    name: ClassVar[str] = "paddle-vllm"

    def __init__(
        self,
        model_path: str | None = None,
        model: str | None = None,
        profile=None,
        settings: GlobalSettings | None = None,
        **_kw: object,
    ) -> None:
        effective = settings if settings is not None else snapshot_settings()
        self._server = VllmOcrServer(model=model_path, settings=effective)
        try:
            self._http_client = PaddleHttpOcrClient(
                base_url=self._server.base_url,
                model=model or effective.ocr.paddle_served_model,
                profile=profile,
                settings=effective,
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
        await self._http_client.wait_for_server()

    async def shutdown(self) -> None:
        try:
            await self._http_client.shutdown()
        finally:
            self._server.shutdown()
