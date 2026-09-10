import asyncio

import pytest


async def test_get_client_serializes_concurrent_init(monkeypatch):
    """Two concurrent first-time recognize calls must build only one client."""
    pytest.importorskip("instructor")

    from bibr.local.ocr_cloud import CloudOcrClient

    builds = {"n": 0}

    def fake_from_provider(*_a, **_kw):
        builds["n"] += 1

        class Fake:
            pass

        return Fake()

    monkeypatch.setattr("instructor.from_provider", fake_from_provider, raising=False)

    c = CloudOcrClient.__new__(CloudOcrClient)
    c._client = None
    c._provider = "google"
    c._loaded = True
    c._init_lock = asyncio.Lock()  # this attr should exist after the fix

    async def go():
        return await c._aget_client()

    results = await asyncio.gather(go(), go(), go())
    assert builds["n"] == 1, f"expected 1 build, got {builds['n']}"
    assert all(r is results[0] for r in results)
