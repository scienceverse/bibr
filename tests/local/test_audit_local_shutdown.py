"""OCR clients stop the server they own even when closing their HTTP pool fails."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from bibr.config import GlobalSettings


class _FailingHttp:
    """An HTTP client whose pool cannot be closed, e.g. one bound to a loop already closed."""

    async def shutdown(self) -> None:
        raise RuntimeError("Event loop is closed")


async def test_llama_cpp_ocr_shutdown_stops_the_server_when_the_http_close_fails(monkeypatch):
    from bibr.local import llama_cpp
    from bibr.local import ocr as ocr_mod

    server = MagicMock(base_url="http://127.0.0.1:8771", loaded=True, api_key="k")
    monkeypatch.setattr(llama_cpp, "LlamaCppServer", MagicMock(return_value=server))
    client = ocr_mod.LlamaCppOcrClient(settings=GlobalSettings())
    await client._http_client.shutdown()
    client._http_client = _FailingHttp()

    with pytest.raises(RuntimeError, match="Event loop is closed"):
        await client.shutdown()

    server.shutdown.assert_called_once()


async def test_rapid_mlx_ocr_shutdown_stops_the_server_when_the_http_close_fails(monkeypatch):
    from bibr.local import rapid_mlx as mod

    server = MagicMock(loaded=True)
    monkeypatch.setattr(
        mod._ManagedRapidMlxOcrClient,
        "_start_generation",
        lambda self, stop_event=None: (server, _FailingHttp()),
    )
    client = mod.RapidMlxOcrClient(settings=GlobalSettings())

    with pytest.raises(RuntimeError, match="Event loop is closed"):
        await client.shutdown()

    server.shutdown.assert_called_once()


async def test_cloud_ocr_shutdown_closes_the_sdk_client(monkeypatch):
    pytest.importorskip("instructor")
    from bibr.local.ocr_cloud import CloudOcrClient

    closed = []

    class _Sdk:
        async def close(self) -> None:
            closed.append(True)

    instructor_client = MagicMock()
    instructor_client.client = _Sdk()
    client = CloudOcrClient(provider="openai", settings=GlobalSettings())
    client._client = instructor_client

    await client.shutdown()

    assert closed == [True]
    assert client._client is None


async def test_cloud_ocr_encodes_the_region_off_the_event_loop(monkeypatch):
    pytest.importorskip("instructor")
    from PIL import Image

    import bibr.ocr.image_utils as image_utils
    from bibr.local.ocr_cloud import CloudOcrClient

    threads = []
    real_encode = image_utils.pil_to_base64_glmocr

    def recording_encode(img, fmt="JPEG"):
        threads.append(threading.current_thread())
        return real_encode(img, fmt)

    monkeypatch.setattr(image_utils, "pil_to_base64_glmocr", recording_encode)

    class _Result:
        text = "read"

    class _Instructor:
        async def create(self, **_kw):
            return _Result()

    client = CloudOcrClient(provider="openai", settings=GlobalSettings())
    client._client = _Instructor()
    try:
        assert await client.recognize(Image.new("RGB", (8, 8)), "Text Recognition:") == "read"
    finally:
        client._client = None
        await client.shutdown()

    assert threads and threads[0] is not threading.main_thread()
