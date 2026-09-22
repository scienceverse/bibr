"""Tests for the two-tier cache wiring in CrossrefClient._cached."""

from unittest.mock import AsyncMock

from bibr.clients.crossref import CrossrefClient
from bibr.config import Settings


def _client():
    # __init__ does not touch Redis (limiter is lazy); construct directly.
    return CrossrefClient(mailto="test@example.com")


async def test_cached_skips_redis_when_disabled(monkeypatch):
    monkeypatch.setattr(Settings.crossref, "redis_cache", False)
    client = _client()

    fetch = AsyncMock(return_value={"message": "fresh"})
    result = await client._cached("works:10.1/x", fetch)

    assert result == {"message": "fresh"}
    fetch.assert_awaited_once()
    # Backend is never created when disabled.
    assert client._response_cache_backend is None


async def test_cached_returns_redis_hit_without_fetch(monkeypatch):
    monkeypatch.setattr(Settings.crossref, "redis_cache", True)
    monkeypatch.setattr(Settings.crossref, "cache_size", 0)  # isolate tier 2
    client = _client()
    # Inject a fake tier-2 backend and mark init done so _ensure skips creation.
    client._response_cache_init = True
    backend = AsyncMock()
    backend.get = AsyncMock(return_value={"message": "cached"})
    client._response_cache_backend = backend

    fetch = AsyncMock(return_value={"message": "fresh"})
    result = await client._cached("works:10.1/x", fetch)

    assert result == {"message": "cached"}
    fetch.assert_not_awaited()
    backend.get.assert_awaited_once_with("works:10.1/x")


async def test_cached_populates_redis_on_miss(monkeypatch):
    monkeypatch.setattr(Settings.crossref, "redis_cache", True)
    monkeypatch.setattr(Settings.crossref, "cache_size", 0)  # isolate tier 2
    client = _client()
    client._response_cache_init = True
    backend = AsyncMock()
    backend.get = AsyncMock(return_value=None)
    client._response_cache_backend = backend

    fetch = AsyncMock(return_value={"message": "fresh"})
    result = await client._cached("works:10.1/x", fetch)

    assert result == {"message": "fresh"}
    fetch.assert_awaited_once()
    backend.set.assert_awaited_once_with("works:10.1/x", {"message": "fresh"})


async def test_cached_redis_hit_populates_lru(monkeypatch):
    # tier-2 hit with tier-1 enabled must back-fill the LRU, so a second call
    # serves from tier 1 without re-hitting the backend.
    monkeypatch.setattr(Settings.crossref, "redis_cache", True)
    monkeypatch.setattr(Settings.crossref, "cache_size", 2)
    client = _client()
    client._response_cache_init = True
    backend = AsyncMock()
    backend.get = AsyncMock(return_value={"message": "cached"})
    client._response_cache_backend = backend

    fetch = AsyncMock(return_value={"message": "fresh"})
    result = await client._cached("works:10.1/x", fetch)

    assert result == {"message": "cached"}
    fetch.assert_not_awaited()
    assert "works:10.1/x" in client.__dict__["_response_cache"]

    backend.get.reset_mock()
    result2 = await client._cached("works:10.1/x", fetch)
    assert result2 == {"message": "cached"}
    backend.get.assert_not_awaited()  # served from tier 1


async def test_ensure_response_cache_disabled_without_url(monkeypatch):
    # redis_cache enabled but no URL anywhere -> degrade to disabled (None),
    # init guard set so the warning fires once, not per request.
    monkeypatch.setattr(Settings.crossref, "redis_cache", True)
    monkeypatch.setattr(Settings.crossref, "cache_redis_url", None)
    monkeypatch.setattr(Settings.redis, "url", None)
    client = _client()

    backend = await client._ensure_response_cache()

    assert backend is None
    assert client._response_cache_backend is None
    assert client._response_cache_init is True


async def test_cached_stores_a_doi_404_in_redis_with_its_own_ttl(monkeypatch):
    import httpx
    import pytest

    monkeypatch.setattr(Settings.crossref, "redis_cache", True)
    monkeypatch.setattr(Settings.crossref, "cache_size", 0)  # isolate tier 2
    monkeypatch.setattr(Settings.crossref, "not_found_ttl_seconds", 3600)
    client = _client()
    client._response_cache_init = True
    backend = AsyncMock()
    backend.get = AsyncMock(return_value=None)
    client._response_cache_backend = backend
    request = httpx.Request("GET", "https://api.crossref.org/works/10.1/x")
    fetch = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "404", request=request, response=httpx.Response(404, request=request)
        )
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client._cached("works:10.1/x", fetch, not_found_path="/works/10.1/x")

    backend.set.assert_awaited_once()
    (key, entry), kwargs = backend.set.await_args
    assert key == "works:10.1/x"
    assert kwargs == {"ttl_seconds": 3600}

    # A later process reads the entry back from Redis and re-raises the 404.
    backend.get = AsyncMock(return_value=entry)
    fetch.reset_mock()
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await client._cached("works:10.1/x", fetch, not_found_path="/works/10.1/x")
    assert excinfo.value.response.status_code == 404
    fetch.assert_not_awaited()
