"""A consensus or escalation recognizer that does not start is closed, not dropped open."""

from __future__ import annotations

import asyncio

import pytest


def _ocr_settings(**ocr):
    from bibr.config import snapshot_settings

    settings = snapshot_settings()
    return settings.model_copy(update={"ocr": settings.ocr.model_copy(update=ocr)})


def _track_http_clients(monkeypatch):
    from bibr.local.ocr import HttpOcrClient, PaddleHttpOcrClient

    created = []
    real_init = HttpOcrClient.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        created.append(self)

    for cls in (HttpOcrClient, PaddleHttpOcrClient):
        monkeypatch.setattr(cls, "__init__", init)
    return created


def test_a_consensus_server_that_never_gets_ready_has_its_client_closed(monkeypatch):
    from bibr.local.ocr import HttpOcrClient
    from bibr.scan.consensus import start_recognizers

    created = _track_http_clients(monkeypatch)

    async def down(self):
        raise ConnectionError("refused")

    monkeypatch.setattr(HttpOcrClient, "wait_for_server", down)
    settings = _ocr_settings(consensus_backend="glm-http", consensus_url="http://127.0.0.1:9")

    assert asyncio.run(start_recognizers(settings)) is None
    assert [client._client.is_closed for client in created] == [True]


def test_an_escalation_server_that_never_gets_ready_has_its_client_closed(monkeypatch):
    from bibr.local.ocr import HttpOcrClient, PaddleHttpOcrClient
    from bibr.scan.consensus import start_recognizers

    created = _track_http_clients(monkeypatch)

    async def ready(self):
        return None

    async def down(self):
        raise ConnectionError("refused")

    monkeypatch.setattr(HttpOcrClient, "wait_for_server", ready)
    monkeypatch.setattr(PaddleHttpOcrClient, "wait_for_server", down)
    settings = _ocr_settings(
        consensus_backend="glm-http",
        consensus_url="http://127.0.0.1:9",
        escalation_backend="paddle-http",
        escalation_url="http://127.0.0.1:10",
    )

    async def start():
        recognizers = await start_recognizers(settings)
        states = [client._client.is_closed for client in created]
        await recognizers[0].shutdown()
        return recognizers, states

    (second, escalation), states = asyncio.run(start())
    assert escalation is None
    # The consensus client stays open for the run; the escalation one is closed.
    assert states == [False, True]


def test_a_cancelled_start_closes_every_client_it_opened(monkeypatch):
    from bibr.local.ocr import HttpOcrClient, PaddleHttpOcrClient
    from bibr.scan.consensus import start_recognizers

    created = _track_http_clients(monkeypatch)

    async def ready(self):
        return None

    async def hang(self):
        await asyncio.sleep(3600)

    monkeypatch.setattr(HttpOcrClient, "wait_for_server", ready)
    monkeypatch.setattr(PaddleHttpOcrClient, "wait_for_server", hang)
    settings = _ocr_settings(
        consensus_backend="glm-http",
        consensus_url="http://127.0.0.1:9",
        escalation_backend="paddle-http",
        escalation_url="http://127.0.0.1:10",
    )

    async def start_then_cancel():
        task = asyncio.create_task(start_recognizers(settings))
        while len(created) < 2:
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(start_then_cancel())
    assert [client._client.is_closed for client in created] == [True, True]
