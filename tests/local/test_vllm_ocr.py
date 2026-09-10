"""Managed PaddleOCR-VL vLLM runtime and client."""

import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.config import Settings


def _mk_server(vllm_installed=True, uv_present=True, **kwargs):
    from bibr.local import vllm_ocr

    with (
        patch("bibr.ocr.registry._cuda_vram_gb", return_value=24.0),
        patch("importlib.util.find_spec", return_value=object() if vllm_installed else None),
        patch.object(vllm_ocr.shutil, "which", return_value="/usr/bin/uv" if uv_present else None),
        patch.object(vllm_ocr, "guard_managed_server_port", return_value=False),
        patch.object(vllm_ocr.subprocess, "Popen") as popen,
        patch.object(vllm_ocr.VllmOcrServer, "_wait_until_ready"),
    ):
        popen.return_value = MagicMock()
        return vllm_ocr.VllmOcrServer(**kwargs), popen


def test_uv_fallback_launches_pinned_paddle_vllm_command(monkeypatch):
    from bibr.local import vllm_ocr

    # Pin the interpreter so the expected command does not depend on the CI
    # matrix: on 3.14 the bootstrap adds ``--python 3.13`` (covered below).
    monkeypatch.setattr(vllm_ocr.sys, "version_info", (3, 13, 0, "final", 0))
    _server, popen = _mk_server(vllm_installed=False, model="PaddlePaddle/PaddleOCR-VL-1.6")

    cmd = popen.call_args.args[0]
    assert cmd[:9] == [
        "uv",
        "tool",
        "run",
        "--from",
        "vllm==0.27.0",
        "--with",
        "openai>=2.54.0,<3",
        "vllm",
        "serve",
    ]
    assert cmd[9] == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert "--model" not in cmd


def test_launch_command_uses_audited_paddle_runtime_settings(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "paddle_vllm_extra_args", "--max-model-len 8192")
    _server, popen = _mk_server(model="PaddlePaddle/PaddleOCR-VL-1.6", port=9123)

    cmd = popen.call_args.args[0]
    expected = {
        "--revision": "66317acc4c9fc17bd154591ce650735cd2855f3e",
        "--served-model-name": "paddle-ocr-vl-1.6",
        "--gpu-memory-utilization": "0.92",
        "--max-model-len": "16384",
        "--max-num-seqs": "12",
        "--max-num-batched-tokens": "16384",
        "--mm-processor-cache-gb": "0",
    }
    for option, value in expected.items():
        assert cmd[cmd.index(option) + 1] == value
    assert "--no-enable-prefix-caching" in cmd
    assert not any("flashinfer" in part.lower() or "attention" in part.lower() for part in cmd)
    # Explicit operator flags are appended last and override defaults.
    assert cmd[-2:] == ["--max-model-len", "8192"]


def test_readiness_requires_the_exact_served_alias(monkeypatch):
    from bibr.local import vllm_ocr

    process = MagicMock()
    process.poll.return_value = None
    server = vllm_ocr.VllmOcrServer.__new__(vllm_ocr.VllmOcrServer)
    server._settings = Settings
    server._model = "PaddlePaddle/PaddleOCR-VL-1.6"
    server._served_model = "paddle-ocr-vl-1.6"
    server._port = 9123
    server._process = process
    server._stderr_fh = None
    server._stderr_log = None
    server._reused = False
    monkeypatch.setattr(Settings.ocr, "paddle_vllm_startup_timeout", 10)
    monkeypatch.setattr(vllm_ocr.time, "monotonic", MagicMock(side_effect=[0, 1, 2]))
    monkeypatch.setattr(
        vllm_ocr,
        "request_bytes",
        MagicMock(
            side_effect=[
                (200, "OK", b'{"data":[{"id":"wrong-alias"}]}'),
                (200, "OK", b'{"data":[{"id":"paddle-ocr-vl-1.6"}]}'),
            ]
        ),
    )
    monkeypatch.setattr(vllm_ocr.time, "sleep", MagicMock())

    server._wait_until_ready()

    assert vllm_ocr.request_bytes.call_count == 2
    vllm_ocr.request_bytes.assert_called_with("http://localhost:9123/v1/models", timeout=5)


def test_shutdown_terminates_the_vllm_process_group(monkeypatch):
    from bibr.local import vllm_ocr

    process = MagicMock(pid=123)
    server = vllm_ocr.VllmOcrServer.__new__(vllm_ocr.VllmOcrServer)
    server._process = process
    server._stderr_fh = MagicMock()
    server._stderr_log = None
    server._reused = False
    killpg = MagicMock()
    monkeypatch.setattr(vllm_ocr.os, "killpg", killpg, raising=False)

    server.shutdown()

    killpg.assert_called_once_with(123, signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=vllm_ocr._TERM_GRACE_S)
    assert server._process is None


@pytest.mark.asyncio
async def test_client_shuts_down_server_when_http_client_construction_fails(monkeypatch):
    from bibr.local import vllm_ocr

    server = MagicMock()
    monkeypatch.setattr(vllm_ocr, "VllmOcrServer", MagicMock(return_value=server))
    monkeypatch.setattr(
        vllm_ocr, "PaddleHttpOcrClient", MagicMock(side_effect=RuntimeError("bad client"))
    )

    with pytest.raises(RuntimeError, match="bad client"):
        vllm_ocr.PaddleVllmOcrClient(settings=Settings)

    server.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_client_delegates_the_ocr_protocol(monkeypatch):
    from bibr.local import vllm_ocr

    server = MagicMock(loaded=True)
    http_client = MagicMock()
    http_client.recognize = AsyncMock(return_value="text")
    http_client.wait_for_server = AsyncMock()
    http_client.shutdown = AsyncMock()
    monkeypatch.setattr(vllm_ocr, "VllmOcrServer", MagicMock(return_value=server))
    monkeypatch.setattr(vllm_ocr, "PaddleHttpOcrClient", MagicMock(return_value=http_client))
    client = vllm_ocr.PaddleVllmOcrClient(settings=Settings)

    assert client.name == "paddle-vllm"
    assert client.loaded is True
    assert await client.recognize("image", "OCR:") == "text"
    await client.wait_for_server()
    await client.shutdown()

    http_client.recognize.assert_called_once_with("image", "OCR:")
    http_client.wait_for_server.assert_awaited_once()
    http_client.shutdown.assert_awaited_once()
    server.shutdown.assert_called_once()


def test_launch_refuses_without_a_suitable_gpu():
    """No GPU means vLLM cannot run at all: fail fast before any download."""
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import vllm_ocr

    with (
        patch("bibr.ocr.registry._cuda_vram_gb", return_value=None),
        patch("importlib.util.find_spec", return_value=None),
        patch.object(vllm_ocr.shutil, "which", return_value="/usr/bin/uv"),
        patch.object(vllm_ocr, "guard_managed_server_port", return_value=False),
        patch.object(vllm_ocr.subprocess, "Popen") as popen,
    ):
        with pytest.raises(UpstreamServiceError, match="no NVIDIA GPU"):
            vllm_ocr.VllmOcrServer(model="PaddlePaddle/PaddleOCR-VL-1.6")
    popen.assert_not_called()


def test_uv_bootstrap_warns_with_the_install_remedy(caplog):

    with caplog.at_level("WARNING", logger="bibr.local.vllm_ocr"):
        _mk_server(vllm_installed=False, model="PaddlePaddle/PaddleOCR-VL-1.6")

    messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("uv sync --extra vllm" in m and "several GB" in m for m in messages)


def test_uv_bootstrap_pins_a_supported_interpreter_on_python_314(monkeypatch):
    from bibr.local import vllm_ocr

    monkeypatch.setattr(vllm_ocr.sys, "version_info", (3, 14, 0, "final", 0))
    _server, popen = _mk_server(vllm_installed=False, model="PaddlePaddle/PaddleOCR-VL-1.6")

    cmd = popen.call_args.args[0]
    assert cmd[:11] == [
        "uv",
        "tool",
        "run",
        "--python",
        "3.13",
        "--from",
        "vllm==0.27.0",
        "--with",
        "openai>=2.54.0,<3",
        "vllm",
        "serve",
    ]
