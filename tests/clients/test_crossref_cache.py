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


# --- remembered 404s (CROSSREF_NOT_FOUND_TTL_SECONDS) -------------------------


def _client_with_status(monkeypatch, status: int, *, not_found_ttl: int | None = None):
    calls = {"n": 0}

    async def fake_request(self, path, params=None):  # noqa: ARG001
        calls["n"] += 1
        request = httpx.Request("GET", f"https://api.crossref.org{path}")
        response = httpx.Response(status, request=request)
        raise httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)

    monkeypatch.setattr(CrossrefClient, "_request", fake_request)
    settings = GlobalSettings()
    if not_found_ttl is not None:
        settings.crossref.not_found_ttl_seconds = not_found_ttl
    return CrossrefClient(settings=settings), calls


async def test_doi_404_is_remembered_and_re_raised(monkeypatch):
    client, calls = _client_with_status(monkeypatch, 404)

    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await client.works("10.1000/missing")
        assert excinfo.value.response.status_code == 404

    assert calls["n"] == 1
    assert excinfo.value.request.url.path == "/works/10.1000/missing"


async def test_remembered_404_expires(monkeypatch):
    from bibr.clients import crossref

    client, calls = _client_with_status(monkeypatch, 404, not_found_ttl=60)
    now = 1_000_000.0
    monkeypatch.setattr(crossref.time, "time", lambda: now)
    with pytest.raises(httpx.HTTPStatusError):
        await client.works("10.1000/missing")

    now += 61
    with pytest.raises(httpx.HTTPStatusError):
        await client.works("10.1000/missing")

    assert calls["n"] == 2


async def test_remembered_404_disabled_by_zero_ttl(monkeypatch):
    client, calls = _client_with_status(monkeypatch, 404, not_found_ttl=0)

    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            await client.works("10.1000/missing")

    assert calls["n"] == 2


async def test_other_http_errors_are_not_remembered(monkeypatch):
    client, calls = _client_with_status(monkeypatch, 500)

    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            await client.works("10.1000/flaky")

    assert calls["n"] == 2


async def test_remembered_404_keeps_the_doi_miss_semantics(monkeypatch):
    # A DOI 404 ends the reference's Crossref lookup without a bibliographic
    # search; the remembered 404 must do the same, not look like an empty
    # response (which would fall through to the search).
    from unittest.mock import AsyncMock

    from bibr.enrich.references import _fetch_crossref_item
    from bibr.paper import PaperReference

    client, calls = _client_with_status(monkeypatch, 404)
    search = AsyncMock(return_value={"message": {"items": []}})
    monkeypatch.setattr(client, "search", search)
    ref = PaperReference(
        bib_id=1,
        title="A sufficiently long printed title",
        doi="10.1000/missing",
        year=2020,
        first_page=None,
        volume=None,
        authors=None,
        container=None,
    )

    assert await _fetch_crossref_item(ref, client) is None
    assert await _fetch_crossref_item(ref, client) is None

    assert calls["n"] == 1
    search.assert_not_awaited()


async def test_bulk_prefetch_skips_a_remembered_404(monkeypatch):
    client, calls = _client_with_status(monkeypatch, 404)
    with pytest.raises(httpx.HTTPStatusError):
        await client.works("10.1000/missing")

    assert await client.prefetch_works_by_doi(["10.1000/MISSING"]) == 0
    assert calls["n"] == 1
