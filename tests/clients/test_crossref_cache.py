"""Crossref response cache: repeat lookups within a process must not re-hit the API."""

import httpx
import pytest

from bibr.clients.crossref import CrossrefClient
from bibr.config import GlobalSettings


def _client_with_counting_request(
    monkeypatch, fail: bool = False, *, cache_size: int | None = None
):
    calls = {"n": 0}

    async def fake_request(self, path, params=None):  # noqa: ARG001
        calls["n"] += 1
        if fail:
            raise httpx.ConnectError("boom")
        return {"message": {"path": path, "params": params}}

    monkeypatch.setattr(CrossrefClient, "_request", fake_request)
    settings = GlobalSettings()
    if cache_size is not None:
        settings.crossref.cache_size = cache_size
    client = CrossrefClient(settings=settings)
    return client, calls


def test_get_client_separates_distinct_runtime_cache_policies():
    from bibr.clients import crossref

    first = GlobalSettings()
    second = GlobalSettings()
    first.crossref.cache_size = 16
    first.crossref.redis_cache = False
    second.crossref.cache_size = 64
    second.crossref.redis_cache = True
    second.crossref.cache_redis_url = "redis://second"

    crossref._clients_by_settings.clear()
    try:
        assert crossref.get_client(settings=first) is not crossref.get_client(settings=second)
    finally:
        crossref._clients_by_settings.clear()


async def test_works_caches_repeat_doi(monkeypatch):
    client, calls = _client_with_counting_request(monkeypatch)

    first = await client.works("10.1093/bioinformatics/btac123")
    second = await client.works("10.1093/bioinformatics/btac123")

    assert calls["n"] == 1
    assert first == second


async def test_works_distinct_dois_not_conflated(monkeypatch):
    client, calls = _client_with_counting_request(monkeypatch)

    a = await client.works("10.1000/a")
    b = await client.works("10.1000/b")

    assert calls["n"] == 2
    assert a != b


async def test_search_caches_repeat_query(monkeypatch):
    client, calls = _client_with_counting_request(monkeypatch)

    first = await client.search("smith 2020 my paper", limit=3)
    second = await client.search("smith 2020 my paper", limit=3)

    assert calls["n"] == 1
    assert first == second


async def test_search_limit_is_part_of_key(monkeypatch):
    client, calls = _client_with_counting_request(monkeypatch)

    await client.search("smith 2020 my paper", limit=3)
    await client.search("smith 2020 my paper", limit=5)

    assert calls["n"] == 2


async def test_failures_are_not_cached(monkeypatch):
    client, calls = _client_with_counting_request(monkeypatch, fail=True)

    with pytest.raises(httpx.ConnectError):
        await client.works("10.1000/a")
    with pytest.raises(httpx.ConnectError):
        await client.works("10.1000/a")

    assert calls["n"] == 2


async def test_cache_disabled_via_setting(monkeypatch):
    client, calls = _client_with_counting_request(monkeypatch, cache_size=0)

    await client.works("10.1000/a")
    await client.works("10.1000/a")

    assert calls["n"] == 2


async def test_cache_evicts_least_recently_used(monkeypatch):
    client, calls = _client_with_counting_request(monkeypatch, cache_size=1)

    await client.works("10.1000/a")
    await client.works("10.1000/b")  # evicts a
    await client.works("10.1000/a")  # miss again

    assert calls["n"] == 3
