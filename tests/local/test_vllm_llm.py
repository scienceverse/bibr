"""VllmLlmServer: command construction, health polling, and shutdown."""

import signal
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from bibr.config import Settings


def _mk_server(vllm_installed=True, uv_present=True, **kwargs):
    from bibr.local import vllm_llm

    with (
        patch("importlib.util.find_spec", return_value=object() if vllm_installed else None),
        patch.object(vllm_llm.shutil, "which", return_value="/usr/bin/uv" if uv_present else None),
        patch.object(vllm_llm.subprocess, "Popen") as popen,
        patch.object(vllm_llm.VllmLlmServer, "_wait_until_healthy"),
    ):
        popen.return_value = MagicMock()
        server = vllm_llm.VllmLlmServer(**kwargs)
        return server, popen


def test_missing_vllm_and_uv_raises_actionable_error():
    from bibr.exceptions import UpstreamServiceError
    from bibr.local.vllm_llm import VllmLlmServer

    with (
        patch("importlib.util.find_spec", return_value=None),
        patch("shutil.which", return_value=None),
        pytest.raises(UpstreamServiceError) as exc,
    ):
        VllmLlmServer(model="org/m")
    msg = str(exc.value)
    assert "uv" in msg
    assert "vllm" in msg


def test_uv_fallback_when_vllm_absent():
    server, popen = _mk_server(
        vllm_installed=False, uv_present=True, model="org/m", port=9999, mem_fraction=0.5
    )
    cmd = popen.call_args[0][0]
    assert cmd[:6] == ["uv", "tool", "run", "--from", "vllm==0.25.1", "vllm"]
    assert cmd[6] == "serve"
    assert cmd[7] == "org/m"  # model positional
    assert "--model" not in cmd
    assert "9999" in cmd
    assert "0.5" in cmd
    assert server.base_url == "http://localhost:9999"


def test_command_and_base_url():
    server, popen = _mk_server(model="numind/NuExtract3", port=9999, mem_fraction=0.5)
    cmd = popen.call_args[0][0]
    assert cmd[0] == sys.executable
    assert "vllm.entrypoints.openai.api_server" in cmd
    assert "--model" in cmd
    assert "numind/NuExtract3" in cmd
    assert "9999" in cmd
    assert "0.5" in cmd
    assert server.base_url == "http://localhost:9999"


def test_extra_args_appended(monkeypatch):
    monkeypatch.setattr(Settings.llm, "vllm_extra_args", "--max-model-len 32768")
    server, popen = _mk_server(model="org/m", port=9999)
    cmd = popen.call_args[0][0]
    assert "--max-model-len" in cmd
    assert "32768" in cmd


def test_explicit_mem_fraction_parameter_controls_vllm_allocation():
    _server, popen = _mk_server(model="org/m", port=9999, mem_fraction=0.42)
    cmd = popen.call_args[0][0]
    idx = cmd.index("--gpu-memory-utilization")
    assert cmd[idx + 1] == "0.42"


def test_launch_cmd_includes_registry_args_and_user_extra_last(monkeypatch):
    monkeypatch.setattr(Settings.llm, "vllm_extra_args", "--max-model-len 16384")
    server, popen = _mk_server(model="numind/NuExtract3", port=9999)
    cmd = popen.call_args[0][0]
    # MTP stays opt-in because it disables prefix caching and has a reported
    # accuracy interaction; the registry only supplies non-speculative args.
    assert "--speculative-config" not in cmd
    assert "--trust-remote-code" not in cmd
    assert "--revision" in cmd
    # User extra args appear AFTER the registry args (user wins on repeats).
    reg_idx = cmd.index("--max-model-len")
    user_idx = len(cmd) - 1 - cmd[::-1].index("--max-model-len")
    assert user_idx > reg_idx
    # And the last --max-model-len value is the user's override.
    assert cmd[user_idx + 1] == "16384"


def test_launch_cmd_no_registry_args_for_unknown_model():
    server, popen = _mk_server(model="org/unknown", port=9999)
    cmd = popen.call_args[0][0]
    assert "--speculative-config" not in cmd


def test_launch_cmd_pins_xgrammar_for_structured_outputs():
    _server, popen = _mk_server(model="org/unknown", port=9999)
    cmd = popen.call_args[0][0]
    option = cmd.index("--structured-outputs-config")
    assert cmd[option + 1] == '{"backend":"xgrammar"}'


def test_configure_llm_client(monkeypatch):
    server, _ = _mk_server(model="org/m", port=9999)
    monkeypatch.setattr(Settings.llm, "provider", "google")
    monkeypatch.setattr(Settings.llm, "base_url", Settings.llm.base_url)
    monkeypatch.setattr(Settings.llm, "api_key", Settings.llm.api_key)
    monkeypatch.setattr(Settings.llm, "model", Settings.llm.model)
    server.configure_llm_client()
    assert server._settings.llm.provider == "openai"
    assert server._settings.llm.base_url == "http://localhost:9999/v1"
    assert server._settings.llm.model == "org/m"


def test_defaults_from_settings(monkeypatch):
    monkeypatch.setattr(Settings.llm, "local_model", "org/from-settings")
    monkeypatch.setattr(Settings.llm, "vllm_port", 8123)
    server, popen = _mk_server()
    cmd = popen.call_args[0][0]
    assert "org/from-settings" in cmd
    assert server.base_url == "http://localhost:8123"


def _bare_server(process=None):
    from bibr.local.vllm_llm import VllmLlmServer

    server = VllmLlmServer.__new__(VllmLlmServer)
    server._model = "org/m"
    server._port = 9999
    server._process = process
    server._stderr_fh = None
    server._stderr_log = None
    return server


def test_health_poll_returns_when_server_is_ready(monkeypatch):
    from bibr.local import vllm_llm

    process = MagicMock()
    process.poll.return_value = None
    request = MagicMock(return_value=(200, "OK", b""))
    server = _bare_server(process)
    monkeypatch.setattr(Settings.llm, "vllm_startup_timeout", 10)
    monkeypatch.setattr(vllm_llm.time, "monotonic", MagicMock(side_effect=[0, 1]))
    monkeypatch.setattr(vllm_llm, "request_bytes", request)

    server._wait_until_healthy()

    process.poll.assert_called_once()
    request.assert_called_once_with("http://localhost:9999/health", timeout=5)


def test_health_poll_reports_early_process_exit(tmp_path, monkeypatch):
    process = MagicMock(returncode=7)
    process.poll.return_value = 7
    server = _bare_server(process)
    server._stderr_log = tmp_path / "vllm.log"
    server._stderr_log.write_text("fatal model load")
    server._stderr_fh = MagicMock()
    monkeypatch.setattr(Settings.llm, "vllm_startup_timeout", 10)

    with pytest.raises(RuntimeError, match="fatal model load"):
        server._wait_until_healthy()

    assert server._process is None
    assert server._stderr_fh is None


def test_health_timeout_attempts_cleanup(monkeypatch, caplog):
    from bibr.local import vllm_llm

    server = _bare_server(MagicMock())
    monkeypatch.setattr(Settings.llm, "vllm_startup_timeout", 0)
    monkeypatch.setattr(vllm_llm.time, "monotonic", MagicMock(return_value=0))
    cleanup = MagicMock(side_effect=RuntimeError("cleanup failed"))
    monkeypatch.setattr(server, "shutdown", cleanup)

    with pytest.raises(TimeoutError, match="within 0s"):
        server._wait_until_healthy()

    cleanup.assert_called_once()
    assert "cleanup failed" in caplog.text


def test_shutdown_terminates_process_group(monkeypatch):
    from bibr.local import vllm_llm

    process = MagicMock(pid=123)
    server = _bare_server(process)
    server._stderr_fh = MagicMock()
    killpg = MagicMock()
    monkeypatch.setattr(vllm_llm.os, "killpg", killpg, raising=False)

    server.shutdown()

    killpg.assert_called_once_with(123, signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=vllm_llm._TERM_GRACE_S)
    assert server._process is None
    assert server._stderr_fh is None


def test_shutdown_falls_back_and_escalates(monkeypatch):
    from bibr.local import vllm_llm

    process = MagicMock(pid=456)
    process.wait.side_effect = [subprocess.TimeoutExpired("vllm", 10), None]
    server = _bare_server(process)
    killpg = MagicMock(side_effect=[PermissionError(), ProcessLookupError()])
    monkeypatch.setattr(vllm_llm.os, "killpg", killpg, raising=False)

    server.shutdown()

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_args_list[-1].kwargs == {"timeout": 5}


def test_configure_llm_client_raises_default_rate_limit():
    """The 60 RPM default is a cloud-quota guard. Against a server bibr owns it
    caps the run near 8-12 papers/min (roughly 5-8 LLM calls per paper)
    however fast the GPU is, so the server's own queue should bound instead.
    """
    from bibr.local.http_runtime import MANAGED_LOCAL_LLM_RATE_LIMIT_RPM

    # The server snapshots settings at construction, so drive its own copy.
    server, _ = _mk_server(model="org/m", port=9999)
    server._settings.llm.rate_limit_rpm = 60
    server._settings.llm.model_fields_set.discard("rate_limit_rpm")

    server.configure_llm_client()

    assert server._settings.llm.rate_limit_rpm == MANAGED_LOCAL_LLM_RATE_LIMIT_RPM


def test_configure_llm_client_respects_explicit_rate_limit():
    """A user-set LLM_RATE_LIMIT_RPM must not be silently overridden."""
    server, _ = _mk_server(model="org/m", port=9999)
    server._settings.llm.rate_limit_rpm = 25
    server._settings.llm.model_fields_set.add("rate_limit_rpm")

    server.configure_llm_client()

    assert server._settings.llm.rate_limit_rpm == 25
