"""Crossref's advertised limits: requests in flight per pool, rate headers,
the search field set and its cache key, and the per-batch request tally."""

import asyncio
import logging

import httpx
import pytest

import bibr.clients.crossref as mod
from bibr.config import GlobalSettings


class _NoLimiter:
    """These tests are about requests in flight, not request spacing."""

    window_seconds = 0.0

    async def acquire(self):
        return None

    async def close(self):
        return None


def _client(**crossref):
    settings = GlobalSettings(crossref={"cache_size": 0, **crossref})
    client = mod.CrossrefClient(settings=settings)
    client.limiter = _NoLimiter()
    return client


class _BlockingHttp:
    """Answers each GET once the test releases it, counting how many are open."""

    def __init__(self, headers=None):
        self.headers = headers or {}
        self.open = 0
        self.max_open = 0
        self.release = asyncio.Event()

    async def get(self, path, params=None):  # noqa: ARG002
        self.open += 1
        self.max_open = max(self.max_open, self.open)
        try:
            await self.release.wait()
        finally:
            self.open -= 1
        request = httpx.Request("GET", mod._CROSSREF_API_BASE + path)
        return httpx.Response(200, json={"message": {}}, headers=self.headers, request=request)


async def _run_concurrent(client, http, n):
    async def fake_get_http_client():
        return http

    client._get_http_client = fake_get_http_client
    tasks = [asyncio.create_task(client.works(ids=f"10.1/{i}")) for i in range(n)]
    for _ in range(20):
        await asyncio.sleep(0)
    observed = http.max_open
    http.release.set()
    await asyncio.gather(*tasks)
    return observed


def test_default_in_flight_limits_follow_the_pool():
    assert _client(api_email="t@x.org")._pool_concurrency == {"single": 3, "list": 3}
    assert _client()._pool_concurrency == {"single": 1, "list": 1}


@pytest.mark.asyncio
async def test_polite_pool_keeps_three_requests_in_flight():
    """enrich_concurrency (12) references at once no longer means 12 requests at
    once: Crossref's polite pool accepts 3, and the excess earned 429s."""
    client = _client(api_email="t@x.org", enrich_concurrency=12)
    observed = await _run_concurrent(client, _BlockingHttp(), 8)
    assert observed == 3


@pytest.mark.asyncio
async def test_anonymous_pool_keeps_one_request_in_flight():
    client = _client()
    observed = await _run_concurrent(client, _BlockingHttp(), 4)
    assert observed == 1


@pytest.mark.asyncio
async def test_advertised_concurrency_limit_is_adopted_per_pool():
    client = _client(api_email="t@x.org")
    http = _BlockingHttp(headers={"x-concurrency-limit": "2", "x-api-pool": "polite-single"})
    http.release.set()

    async def fake_get_http_client():
        return http

    client._get_http_client = fake_get_http_client
    await client.works(ids="10.1/a")

    assert client._pool_concurrency == {"single": 2, "list": 3}

    http.release = asyncio.Event()
    assert await _run_concurrent(client, http, 6) == 2


def test_advertised_concurrency_limit_is_bounded():
    client = _client(api_email="t@x.org")
    headers = httpx.Headers({"x-concurrency-limit": "0"})

    async def adapt(value):
        client._adapt_concurrency(httpx.Headers({"x-concurrency-limit": value}), "list")

    asyncio.run(adapt("0"))
    assert client._pool_concurrency["list"] == 1
    asyncio.run(adapt("1000"))
    assert client._pool_concurrency["list"] == mod._MAX_CONCURRENCY
    asyncio.run(adapt("garbage"))
    assert client._pool_concurrency["list"] == mod._MAX_CONCURRENCY
    assert headers  # the header object itself is untouched


@pytest.mark.asyncio
async def test_raised_limit_wakes_waiting_requests():
    gate_limit = {"n": 1}
    gate = mod._InFlightGate(lambda: gate_limit["n"])
    await gate.acquire()
    waiter = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)
    assert not waiter.done()

    gate_limit["n"] = 2
    gate.wake()
    await asyncio.sleep(0)
    assert waiter.done()
    assert gate.active == 2


@pytest.mark.asyncio
async def test_cancelled_waiter_gives_no_slot_away():
    gate = mod._InFlightGate(lambda: 1)
    await gate.acquire()
    waiter = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    gate.release()
    assert gate.active == 0
    await gate.acquire()
    assert gate.active == 1


def test_pool_of_request_paths():
    assert mod._pool_of("/works") == "list"
    assert mod._pool_of("/works/10.1/abc") == "single"


@pytest.mark.asyncio
async def test_search_selects_update_notices(monkeypatch):
    """updated-by carries retraction notices; without it in select= a search
    hit would never show that the work was retracted."""
    client = _client(api_email="t@x.org")
    seen = {}

    async def fake_request(path, params=None):
        seen["path"], seen["params"] = path, params
        return {"message": {"items": []}}

    monkeypatch.setattr(client, "_request", fake_request)
    await client.search("A title", limit=3)

    assert seen["path"] == "/works"
    assert "updated-by" in seen["params"]["select"].split(",")


@pytest.mark.asyncio
async def test_search_cache_key_names_the_field_set(monkeypatch):
    """Entries cached for 30 days under the old field set (no updated-by) must
    not be served for the new one."""
    client = _client(api_email="t@x.org")
    keys = []

    async def fake_cached(key, fetch, **_kw):
        keys.append(key)
        return {}

    monkeypatch.setattr(client, "_cached", fake_cached)
    await client.search("A title", limit=3)

    (key,) = keys
    assert key.startswith(mod._SEARCH_CACHE_PREFIX + ":")
    assert not key.startswith("search:3:")
    assert key == f"{mod._SEARCH_CACHE_PREFIX}:3:A title"


@pytest.mark.asyncio
async def test_request_tally_counts_throttling_and_retries(monkeypatch):
    client = _client(api_email="t@x.org")
    request = httpx.Request("GET", mod._CROSSREF_API_BASE + "/works/10.1/a")
    responses = [
        httpx.Response(429, headers={"retry-after": "0"}, request=request),
        httpx.Response(200, json={"message": {"DOI": "10.1/a"}}, request=request),
    ]

    class _Http:
        async def get(self, path, params=None):  # noqa: ARG002
            return responses.pop(0)

    async def fake_get_http_client():
        return _Http()

    async def no_sleep(_delay):
        return None

    client._get_http_client = fake_get_http_client
    monkeypatch.setattr(mod.asyncio, "sleep", no_sleep)

    with mod.count_requests() as tally:
        await client.works(ids="10.1/a")

    assert (tally.requests, tally.throttled, tally.retries) == (2, 1, 1)
    # Outside the block nothing is counted.
    assert mod._REQUEST_TALLY.get() is None


def test_contact_email_is_not_logged(caplog):
    with caplog.at_level(logging.DEBUG, logger="bibr.clients.crossref"):
        _client(api_email="someone.private@example.org")
    assert "someone.private@example.org" not in caplog.text
    assert "contact email" in caplog.text
