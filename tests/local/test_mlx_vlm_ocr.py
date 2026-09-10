"""Managed Apple Silicon PaddleOCR-VL runtime tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def _mk_server(monkeypatch, **kwargs):
    from bibr.local import mlx_vlm_ocr as mod

    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(mod, "guard_managed_server_port", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mod.subprocess, "Popen", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(mod.MlxVlmOcrServer, "_wait_until_ready", lambda self: None)
    return mod, mod.MlxVlmOcrServer(**kwargs)


def test_uv_fallback_launches_exact_pinned_mlx_vlm_tool(monkeypatch):
    mod, _server = _mk_server(monkeypatch, model="olragon/PaddleOCR-VL-1.6-8bit")

    cmd = mod.subprocess.Popen.call_args.args[0]
    assert cmd[:8] == [
        "uv",
        "tool",
        "run",
        "--from",
        "mlx-vlm==0.6.3",
        "mlx_vlm.server",
        "--model",
        "olragon/PaddleOCR-VL-1.6-8bit",
    ]
    assert cmd[-4:] == ["--host", "127.0.0.1", "--port", "8775"]


def test_mlx_vlm_server_smoke_uses_paddle_client(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local import mlx_vlm_ocr as mod

    settings = GlobalSettings()
    server = mod.MlxVlmOcrServer.__new__(mod.MlxVlmOcrServer)
    server._settings = settings
    server._model = "olragon/PaddleOCR-VL-1.6-8bit"
    server._port = 8775
    client = MagicMock()
    client.recognize = AsyncMock(return_value="OCR OK")
    client.shutdown = AsyncMock()
    monkeypatch.setattr(mod, "PaddleHttpOcrClient", MagicMock(return_value=client))

    server._run_smoke()

    client.recognize.assert_awaited_once()
    assert client.recognize.call_args.args[1] == "OCR:"
    client.shutdown.assert_awaited_once()


def test_mlx_vlm_smoke_tears_the_client_down_in_its_own_loop(monkeypatch):
    """The httpx pool is bound to the loop that created it.

    Running the request in one ``asyncio.run`` and the teardown in a second
    raised ``RuntimeError: Event loop is closed`` out of the ``finally``,
    replacing a successful smoke result — so the backend could never start.
    """
    import asyncio

    from bibr.config import GlobalSettings
    from bibr.local import mlx_vlm_ocr as mod

    class LoopBoundClient:
        def __init__(self, **_kwargs):
            self.loop = None

        async def recognize(self, _image, _prompt):
            self.loop = asyncio.get_running_loop()
            return "OCR OK"

        async def shutdown(self):
            if asyncio.get_running_loop() is not self.loop:
                raise RuntimeError("Event loop is closed")

    settings = GlobalSettings()
    server = mod.MlxVlmOcrServer.__new__(mod.MlxVlmOcrServer)
    server._settings = settings
    server._model = "olragon/PaddleOCR-VL-1.6-8bit"
    server._port = 8775
    monkeypatch.setattr(mod, "PaddleHttpOcrClient", LoopBoundClient)

    server._run_smoke()


def test_mlx_vlm_reused_listener_must_pass_strict_smoke(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local import mlx_vlm_ocr as mod

    popen = MagicMock()
    killpg = MagicMock()
    client = MagicMock()
    client.recognize = AsyncMock(return_value="OCR OK")
    client.shutdown = AsyncMock()
    monkeypatch.setattr(mod, "guard_managed_server_port", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(mod.subprocess, "Popen", popen)
    monkeypatch.setattr(mod.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(mod, "PaddleHttpOcrClient", MagicMock(return_value=client))

    server = mod.MlxVlmOcrServer(settings=GlobalSettings())

    client.recognize.assert_awaited_once()
    assert server.loaded is True
    assert server._process is None
    popen.assert_not_called()
    killpg.assert_not_called()


def test_mlx_vlm_reuse_rejects_failed_smoke_without_shutdown(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import mlx_vlm_ocr as mod

    popen = MagicMock()
    killpg = MagicMock()
    client = MagicMock()
    client.recognize = AsyncMock(return_value="")
    client.shutdown = AsyncMock()
    monkeypatch.setattr(mod, "guard_managed_server_port", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(mod.subprocess, "Popen", popen)
    monkeypatch.setattr(mod.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(mod, "PaddleHttpOcrClient", MagicMock(return_value=client))

    with pytest.raises(UpstreamServiceError, match="smoke"):
        mod.MlxVlmOcrServer(settings=GlobalSettings())

    popen.assert_not_called()
    killpg.assert_not_called()


@pytest.mark.parametrize("result", ["", "OCR missing", "OK missing"])
def test_mlx_vlm_server_rejects_nonmatching_smoke(monkeypatch, result):
    from bibr.config import GlobalSettings
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import mlx_vlm_ocr as mod

    server = mod.MlxVlmOcrServer.__new__(mod.MlxVlmOcrServer)
    server._settings = GlobalSettings()
    server._model = "olragon/PaddleOCR-VL-1.6-8bit"
    server._port = 8775
    client = MagicMock()
    client.recognize = AsyncMock(return_value=result)
    client.shutdown = AsyncMock()
    monkeypatch.setattr(mod, "PaddleHttpOcrClient", MagicMock(return_value=client))

    with pytest.raises(UpstreamServiceError, match="smoke"):
        server._run_smoke()


@pytest.mark.asyncio
async def test_client_delegates_to_paddle_http_client(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local import mlx_vlm_ocr as mod

    server = MagicMock(loaded=True)
    http = MagicMock()
    http.recognize = AsyncMock(return_value="text")
    http.shutdown = AsyncMock()
    monkeypatch.setattr(mod, "MlxVlmOcrServer", MagicMock(return_value=server))
    monkeypatch.setattr(mod, "PaddleHttpOcrClient", MagicMock(return_value=http))

    client = mod.PaddleMlxVlmOcrClient(settings=GlobalSettings())
    assert client.name == "paddle-mlx-vlm"
    assert await client.recognize("image", "OCR:") == "text"
    await client.shutdown()
    http.recognize.assert_awaited_once_with("image", "OCR:")
    http.shutdown.assert_awaited_once()
    server.shutdown.assert_called_once()


def test_model_only_client_constructor_uses_one_model_for_server_and_http(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local import mlx_vlm_ocr as mod

    server = MagicMock(base_url="http://localhost:8775", loaded=True)
    server_factory = MagicMock(return_value=server)
    http_factory = MagicMock()
    monkeypatch.setattr(mod, "MlxVlmOcrServer", server_factory)
    monkeypatch.setattr(mod, "PaddleHttpOcrClient", http_factory)

    mod.PaddleMlxVlmOcrClient(model="custom/PaddleOCR-VL", settings=GlobalSettings())

    assert server_factory.call_args.kwargs["model"] == "custom/PaddleOCR-VL"
    assert http_factory.call_args.kwargs["model"] == "custom/PaddleOCR-VL"
