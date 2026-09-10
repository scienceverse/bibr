"""Optional LM Studio/llmster deployment backend for local LLM inference.

The adapter manages only resources it creates. It never installs llmster or
downloads a model; operators must perform those trust and storage decisions.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

from bibr.config import snapshot_settings
from bibr.exceptions import UpstreamServiceError

logger = logging.getLogger(__name__)

CommandRunner = Callable[..., Any]


def find_lms() -> str | None:
    """Return the LM Studio CLI path when it is already installed."""

    return shutil.which("lms")


def run_lms_command(lms: str, args: list[str], *, json_output: bool = False) -> Any:
    """Run one non-interactive `lms` command and optionally decode JSON."""

    command = [lms, *args]
    completed = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        rendered = shlex.join(command)
        raise UpstreamServiceError(
            "llmster",
            f"`{rendered}` failed with exit code {completed.returncode}: {detail}",
        )
    if not json_output:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise UpstreamServiceError(
            "llmster", f"`{shlex.join(command)}` returned invalid JSON: {completed.stdout[:500]}"
        ) from exc


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("models", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _model_values(item: dict[str, Any]) -> set[str]:
    keys = ("modelKey", "model_key", "key", "path", "identifier", "id")
    return {str(item[key]) for key in keys if item.get(key)}


class LlmsterLlmServer:
    """Own an optional llmster daemon/server/model instance conservatively."""

    def __init__(
        self,
        model: str | None = None,
        *,
        identifier: str | None = None,
        port: int | None = None,
        context_length: int | None = None,
        load_args: str | None = None,
        runner: CommandRunner | None = None,
        settings=None,
    ) -> None:
        self._settings = settings if settings is not None else snapshot_settings()
        lms = find_lms()
        if lms is None:
            raise UpstreamServiceError(
                "llmster",
                "LM Studio's `lms` CLI is not installed. Install llmster manually with "
                "`curl -fsSL https://lmstudio.ai/install.sh | bash`, then pre-download the "
                "configured model. bibr will not install it or accept its terms for you.",
            )

        self._lms = lms
        self._model = model or getattr(self._settings.llm, "llmster_model", "")
        self._identifier = identifier or getattr(
            self._settings.llm, "llmster_model_id", "bibr-local"
        )
        self._port = port if port is not None else getattr(self._settings.llm, "llmster_port", 1234)
        self._context_length = (
            context_length
            if context_length is not None
            else getattr(self._settings.llm, "llmster_context_length", 32768)
        )
        self._load_args = (
            load_args
            if load_args is not None
            else getattr(self._settings.llm, "llmster_load_args", "")
        )
        self._runner = runner or self._run
        self._started_daemon = False
        self._started_server = False
        self._loaded_identifier: str | None = None

        try:
            self._ensure_ready()
        except Exception:
            self.shutdown()
            raise

    @property
    def _effective_settings(self):
        settings = getattr(self, "_settings", None)
        return settings if settings is not None else snapshot_settings()

    def _run(self, args: list[str], *, json_output: bool = False) -> Any:
        return run_lms_command(self._lms, args, json_output=json_output)

    def _ensure_ready(self) -> None:
        daemon = self._runner(["daemon", "status", "--json"], json_output=True)
        if not isinstance(daemon, dict) or daemon.get("status") != "running":
            self._runner(["daemon", "up", "--json"], json_output=True)
            self._started_daemon = True

        server = self._runner(["server", "status", "--json", "--quiet"], json_output=True)
        if isinstance(server, dict) and server.get("running"):
            if server.get("port") is not None:
                self._port = int(server["port"])
        else:
            self._runner(["server", "start", "--port", str(self._port)])
            self._started_server = True

        local_models = self._runner(["ls", "--json"], json_output=True)
        if not any(self._model in _model_values(item) for item in _items(local_models)):
            raise UpstreamServiceError(
                "llmster",
                f"Model {self._model!r} is not downloaded in LM Studio. Run "
                f"`lms get {self._model}` explicitly, review the selected format, then retry.",
            )

        loaded_models = self._runner(["ps", "--json"], json_output=True)
        if any(self._identifier in _model_values(item) for item in _items(loaded_models)):
            logger.info("Reusing pre-existing llmster model identifier %s", self._identifier)
            return

        command = [
            "load",
            self._model,
            "--identifier",
            self._identifier,
            "--context-length",
            str(self._context_length),
        ]
        command.extend(shlex.split(self._load_args or ""))
        self._runner(command)
        self._loaded_identifier = self._identifier

    def configure_llm_client(self) -> None:
        """Route bibr's existing Instructor client through LM Studio."""

        settings = self._effective_settings
        settings.llm.provider = "openai"
        settings.llm.base_url = f"http://127.0.0.1:{self._port}/v1"
        settings.llm.api_key = "lm-studio"
        settings.llm.model = self._identifier
        logger.info(
            "LLM configured: backend=llmster, model=%s, base_url=%s",
            self._identifier,
            settings.llm.base_url,
        )

    def shutdown(self) -> None:
        """Release only model/server/daemon resources created by this instance."""

        actions: list[list[str]] = []
        if self._loaded_identifier is not None:
            actions.append(["unload", self._loaded_identifier])
            self._loaded_identifier = None
        if self._started_server:
            actions.append(["server", "stop"])
            self._started_server = False
        if self._started_daemon:
            actions.append(["daemon", "down"])
            self._started_daemon = False
        for command in actions:
            try:
                self._runner(command)
            except Exception:  # noqa: BLE001
                logger.warning("llmster cleanup failed for %s", command, exc_info=True)
