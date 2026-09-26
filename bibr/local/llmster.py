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
from bibr.local.http_runtime import MANAGED_LOCAL_LLM_RATE_LIMIT_RPM, request_bytes

logger = logging.getLogger(__name__)

CommandRunner = Callable[..., Any]

# Every `lms` invocation funnels through run_lms_command, so one explicit
# timeout covers daemon/status/server/ls/ps/load/unload alike.
_LMS_COMMAND_TIMEOUT_S = 120


def find_lms() -> str | None:
    """Return the LM Studio CLI path when it is already installed."""

    return shutil.which("lms")


def run_lms_command(lms: str, args: list[str], *, json_output: bool = False) -> Any:
    """Run one non-interactive `lms` command and optionally decode JSON."""

    command = [lms, *args]
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=_LMS_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        # Name the hung subcommand: `load` of a big model is the usual
        # suspect, but daemon/server/ps probes share this path.
        rendered = shlex.join(command)
        subcommand = " ".join(command[1:])
        if command[1:2] == ["load"]:
            hint = (
                "The model may still be loading — retry, or lower "
                "LLM_LLMSTER_CONTEXT_LENGTH / pick a smaller model so the load fits."
            )
        else:
            hint = "Retry once the LM Studio daemon settles."
        raise UpstreamServiceError(
            "llmster",
            f"`{rendered}` timed out after {_LMS_COMMAND_TIMEOUT_S}s "
            f"(subcommand: {subcommand}). {hint}",
        ) from exc
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
        matching = [
            item for item in _items(loaded_models) if self._identifier in _model_values(item)
        ]
        if matching:
            self._check_reused_identifier(matching[0])
            if not self._identifier_served():
                raise UpstreamServiceError(
                    "llmster",
                    f"Identifier {self._identifier!r} is loaded in LM Studio but the "
                    f"server on port {self._port} is not serving it — the daemon or "
                    f"server was restarted, or the identifier was taken over. Run "
                    f"`lms unload {self._identifier}` and retry.",
                )
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

    def _check_reused_identifier(self, item: dict[str, Any]) -> None:
        """Reject a pre-existing identifier that provably serves another model.

        `lms ps` items *may* expose the backing model (modelKey/key/path) next
        to the identifier. When they do and none of those values is the
        configured model, the identifier was taken over — reusing it would
        silently run every request on the wrong model. When the item carries
        no model identity (identifier-only entries), there is nothing to check
        against and the liveness probe below is the only guard.
        """
        model_keys = {
            str(item[key]) for key in ("modelKey", "model_key", "key", "path") if item.get(key)
        }
        if model_keys and self._model not in model_keys | _model_values(item):
            raise UpstreamServiceError(
                "llmster",
                f"Identifier {self._identifier!r} is already loaded but points at "
                f"{sorted(model_keys)[0]!r}, not the configured model {self._model!r}. "
                f"Run `lms unload {self._identifier}` and retry — bibr will not "
                "evict another model's identifier for you.",
            )

    def _identifier_served(self) -> bool:
        """Whether the local server currently serves our identifier.

        One loopback ``GET /v1/models`` per pipeline run (not per request).
        Unreachable server → trust the CLI state: the server may still be
        coming up after `server start`, and failing here would break the
        normal cold-start path.
        """
        url = f"http://127.0.0.1:{self._port}/v1/models"
        try:
            status, _reason, body = request_bytes(url, timeout=10)
            if status != 200:
                raise ValueError(f"HTTP {status}")
            payload = json.loads(body.decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 — any transport failure means "unknown"
            logger.debug("llmster liveness probe of %s failed; trusting `lms ps`", url)
            return True
        return any(self._identifier in _model_values(item) for item in _items(payload))

    def configure_llm_client(self) -> None:
        """Route bibr's existing Instructor client through LM Studio."""

        settings = self._effective_settings
        settings.llm.provider = "openai"
        settings.llm.base_url = f"http://127.0.0.1:{self._port}/v1"
        settings.llm.api_key = "lm-studio"
        settings.llm.model = self._identifier
        if "timeout_seconds" not in settings.llm.model_fields_set:
            # Cloud-shaped 30 s (60 s hard) kills local generations that emit
            # thousands of tokens at laptop tok/s; serialize like the other
            # managed local backends unless the user pinned the fields.
            settings.llm.timeout_seconds = 300
            logger.info("Local llmster backend — raising llm.timeout_seconds 30s->300s")
        if "max_concurrency" not in settings.llm.model_fields_set:
            settings.llm.max_concurrency = 1
            logger.info("Local llmster backend — capping llm.max_concurrency to 1")
        if "rate_limit_rpm" not in settings.llm.model_fields_set:
            # Our own server has no external quota to protect; the cloud 60
            # rpm default would only add dead time between serialized calls.
            settings.llm.rate_limit_rpm = MANAGED_LOCAL_LLM_RATE_LIMIT_RPM
            logger.info(
                "Local llmster backend — raising llm.rate_limit_rpm to %d",
                MANAGED_LOCAL_LLM_RATE_LIMIT_RPM,
            )
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
