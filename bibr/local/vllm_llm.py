"""Managed local LLM server for Linux/CUDA via vLLM.

The managed CUDA backend for ``--llm local``. Starts ``python -m
vllm.entrypoints.openai.api_server`` as a subprocess (OpenAI-compatible
``/v1``), then points its owning pipeline settings at the local endpoint.
"""

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

from bibr.config import snapshot_settings
from bibr.local.http_runtime import (
    MANAGED_LOCAL_LLM_RATE_LIMIT_RPM,
    LocalHttpError,
    guard_managed_server_port,
    request_bytes,
)

logger = logging.getLogger(__name__)

# Grace period between SIGTERM and SIGKILL on shutdown.
_TERM_GRACE_S = 10


class VllmLlmServer:
    """Managed vLLM subprocess for local LLM inference.

    Created by ``LlmServerStage`` AFTER OCR so a single-GPU box loads the
    two models sequentially. Constructor is **blocking** — run it in a
    thread executor.
    """

    def __init__(
        self,
        model: str | None = None,
        port: int | None = None,
        mem_fraction: float | None = None,
        settings=None,
    ):
        self._settings = settings if settings is not None else snapshot_settings()
        self._model = model or self._settings.llm.local_model
        self._port = port if port is not None else self._settings.llm.vllm_port
        if mem_fraction is None:
            mem_fraction = self._settings.llm.local_mem_fraction
        self._process: subprocess.Popen | None = None
        self._stderr_fh = None
        self._stderr_log: Path | None = None
        self._reused = guard_managed_server_port(
            "llm",
            base_url=self.base_url,
            model=self._model,
            server_label="vLLM",
            request_fn=request_bytes,
        )
        if self._reused:
            return

        cmd = self._resolve_launch_cmd(self._model)
        cmd.extend(
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(self._port),
                "--gpu-memory-utilization",
                str(mem_fraction),
                # vLLM 0.24 defaults structured outputs to the mutable "auto"
                # policy. bibr sends stable repeated JSON schemas, so pin the
                # supported XGrammar backend and let user extra args override
                # it only when explicitly requested.
                "--structured-outputs-config",
                '{"backend":"xgrammar"}',
            ]
        )
        # Registry-supplied args for this model (e.g. NuExtract3 MTP config),
        # before user extra args so user flags win on repeated keys.
        from bibr.local.llm_models import registry_server_args

        cmd.extend(registry_server_args(self._model))
        cmd.extend(shlex.split(self._settings.llm.vllm_extra_args or ""))

        # stderr to a file, never PIPE — an unread pipe deadlocks the server
        # once the OS buffer fills (same failure mode as vllm-mlx/sglang).
        from bibr.utils.secure_temp import open_subprocess_log

        self._stderr_log, self._stderr_fh = open_subprocess_log("vllm-llm", self._port)

        logger.info("Starting vLLM LLM server: %s", " ".join(cmd))
        logger.info("vLLM stderr -> %s", self._stderr_log)
        # New session so shutdown() can kill the whole process group — vLLM
        # spawns worker children that would otherwise survive the launcher and
        # hold VRAM.
        self._process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_fh,
            start_new_session=True,
        )
        self._wait_until_healthy()

    @staticmethod
    def _resolve_launch_cmd(model: str) -> list[str]:
        """Resolve the vLLM launcher prefix + model argument.

        bibr only ever launches vLLM as a subprocess, and the ``vllm`` extra is
        heavy and CUDA-specific, so an isolated ``uv tool`` environment is used
        whenever the ``vllm`` module is not importable in-process.
        """
        import importlib.util

        if importlib.util.find_spec("vllm") is not None:
            return [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model,
            ]

        if shutil.which("uv") is not None:
            logger.info(
                "vllm not importable in-process; launching from an isolated 'uv tool' "
                "environment. The first run downloads/installs vLLM into uv's cache "
                "(this can take a while)."
            )
            # `vllm serve` takes the model as a positional argument (not --model).
            # Match the audited `vllm` extra exactly. This fallback executes a
            # downloaded tool, so a floating lower bound is inappropriate.
            return ["uv", "tool", "run", "--from", "vllm==0.25.1", "vllm", "serve", model]

        from bibr.exceptions import UpstreamServiceError

        raise UpstreamServiceError(
            "llm",
            "vllm is not installed and no launcher is available. Choose one:\n"
            "  1. Install uv (recommended — keeps vLLM isolated from bibr's env):\n"
            "       curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "     Then bibr launches vLLM automatically via `uv tool run`.\n"
            "  2. Install vllm into bibr's env:\n"
            "       uv sync --extra vllm    # if using uv\n"
            "       pip install vllm        # if using plain pip",
        )

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def _effective_settings(self):
        settings = getattr(self, "_settings", None)
        return settings if settings is not None else snapshot_settings()

    def _wait_until_healthy(self) -> None:
        """Poll ``/health`` until the engine is up (vLLM binds after load)."""
        assert self._process is not None  # noqa: S101 — set by __init__ before this call
        timeout = self._effective_settings.llm.vllm_startup_timeout
        deadline = time.monotonic() + timeout
        health_url = f"{self.base_url}/health"

        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                rc = self._process.returncode
                tail = self._read_stderr_tail()
                self._close_stderr_fh()
                self._process = None
                raise RuntimeError(f"vLLM process exited during startup (code {rc}): {tail[:500]}")
            try:
                status, _reason, _body = request_bytes(health_url, timeout=5)
                if status == 200:
                    logger.info(
                        "vLLM LLM server ready (model=%s, port=%d)",
                        self._model,
                        self._port,
                    )
                    return
            except (LocalHttpError, json.JSONDecodeError):
                pass
            time.sleep(2.0)

        try:
            self.shutdown()
        except Exception as cleanup_err:  # noqa: BLE001
            logger.warning("vLLM cleanup failed during startup timeout: %s", cleanup_err)
        raise TimeoutError(f"vLLM server did not become ready within {timeout}s")

    def configure_llm_client(self) -> None:
        """Point the owning pipeline's LLM client at the local vLLM server."""
        settings = self._effective_settings
        settings.llm.provider = "openai"
        settings.llm.base_url = self.base_url + "/v1"
        settings.llm.api_key = "not-needed"
        settings.llm.model = self._model
        if "rate_limit_rpm" not in settings.llm.model_fields_set:
            settings.llm.rate_limit_rpm = MANAGED_LOCAL_LLM_RATE_LIMIT_RPM
            logger.info(
                "Local vLLM backend — raising llm.rate_limit_rpm to %d",
                MANAGED_LOCAL_LLM_RATE_LIMIT_RPM,
            )
        logger.info(
            "LLM configured: provider=openai, model=%s, base_url=%s",
            self._model,
            settings.llm.base_url,
        )

    def _read_stderr_tail(self, n: int = 2000) -> str:
        if self._stderr_log is None or not self._stderr_log.exists():
            return ""
        try:
            data = self._stderr_log.read_bytes()
            return data[-n:].decode("utf-8", errors="replace")
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
        """SIGTERM the process group, escalate to SIGKILL after a grace period."""
        if self._process is None:
            self._close_stderr_fh()
            return
        proc, self._process = self._process, None
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
        logger.info("vLLM LLM server shut down")
