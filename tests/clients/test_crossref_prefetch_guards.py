"""Bulk DOI prefetch guards (clients-external-enrich-13).

Prefetch exists only to warm the response cache: with both cache tiers
disabled it must send no request and seed nothing, and DOIs containing a
comma must stay out of the comma-joined bulk filter (they keep their
individual lookup). Fake transport throughout — no Crossref calls.
"""

from bibr.clients.crossref import CrossrefClient
from bibr.config import GlobalSettings


def _client(**crossref):
    settings = GlobalSettings(crossref={"cache_size": 0, "redis_cache": False, **crossref})
    return CrossrefClient(settings=settings)


async def test_prefetch_sends_nothing_when_both_caches_off():
    client = _client()
    calls = []

    async def fake_request(path, params=None):
        calls.append((path, params))
        return {"message": {"items": []}}

    client._request = fake_request
    seeded = await client.prefetch_works_by_doi([f"10.1000/xyz{i}" for i in range(5)])
    assert seeded == 0
    assert calls == []


async def test_comma_doi_skipped_in_bulk_filter():
    client = _client(cache_size=100)
    calls = []

    async def fake_request(path, params=None):
        calls.append(params["filter"])
        return {"message": {"items": []}}

    client._request = fake_request
    seeded = await client.prefetch_works_by_doi(["10.1000/clean", "10.1037/a0012345,Smith"])
    assert seeded == 0
    assert len(calls) == 1
    assert calls[0] == "doi:10.1000/clean"


async def test_prefetch_still_warms_when_cache_on():
    client = _client(cache_size=100)
    calls = []

    async def fake_request(path, params=None):
        calls.append(params["filter"])
        return {"message": {"items": [{"DOI": "10.1000/clean", "title": ["T"]}]}}

    client._request = fake_request
    seeded = await client.prefetch_works_by_doi(["10.1000/clean"])
    assert seeded == 1
    assert calls == ["doi:10.1000/clean"]
    assert client._cache_peek("works:10.1000/clean") is not None
