"""Bulk DOI prefetch: one filter query where there were N /works/{doi} calls."""

import pytest

import bibr.clients.crossref as mod
from bibr.config import GlobalSettings


def _client(**crossref_overrides):
    settings = GlobalSettings(
        crossref={"api_email": "t@x.com", **crossref_overrides},
    )
    return mod.CrossrefClient(settings=settings)


def _item(doi, title="A paper"):
    return {"DOI": doi, "title": [title], "type": "journal-article"}


def _bulk_response(items):
    return {"message": {"items": items, "total-results": len(items)}}


@pytest.mark.asyncio
async def test_one_request_serves_every_doi(monkeypatch):
    client = _client()
    calls = []

    async def fake_request(path, params=None):
        calls.append((path, params))
        return _bulk_response([_item("10.1/a"), _item("10.1/b"), _item("10.1/c")])

    monkeypatch.setattr(client, "_request", fake_request)

    seeded = await client.prefetch_works_by_doi(["10.1/a", "10.1/b", "10.1/c"])

    assert seeded == 3
    assert len(calls) == 1
    path, params = calls[0]
    assert path == "/works"
    assert params["filter"] == "doi:10.1/a,doi:10.1/b,doi:10.1/c"
    # Without an explicit rows, Crossref's default of 20 would truncate a chunk.
    assert params["rows"] == 3


@pytest.mark.asyncio
async def test_seeded_entries_are_served_to_the_per_reference_path(monkeypatch):
    """The prefetch must be invisible to _resolve_reference: works() has to
    return the same single-work shape /works/{doi} would have."""
    client = _client()

    async def fake_request(path, params=None):  # noqa: ARG001
        return _bulk_response([_item("10.1/a", "Cached title")])

    monkeypatch.setattr(client, "_request", fake_request)
    await client.prefetch_works_by_doi(["10.1/a"])

    async def explode(*_a, **_kw):
        raise AssertionError("works() must not hit the network after a prefetch hit")

    monkeypatch.setattr(client, "_request", explode)
    result = await client.works(ids="10.1/a")

    assert result["message"]["DOI"] == "10.1/a"
    assert result["message"]["title"] == ["Cached title"]


@pytest.mark.asyncio
async def test_doi_lookup_is_case_insensitive_across_the_prefetch(monkeypatch):
    """Crossref lowercases DOIs in responses; a reference may carry any casing.
    Both sides fold, so the cache must still hit."""
    client = _client()

    async def fake_request(path, params=None):  # noqa: ARG001
        return _bulk_response([_item("10.1/abc")])

    monkeypatch.setattr(client, "_request", fake_request)
    await client.prefetch_works_by_doi(["10.1/ABC"])

    async def explode(*_a, **_kw):
        raise AssertionError("expected a cache hit")

    monkeypatch.setattr(client, "_request", explode)
    assert (await client.works(ids="10.1/ABC"))["message"]["DOI"] == "10.1/abc"


@pytest.mark.asyncio
async def test_missing_dois_are_left_for_their_own_lookup(monkeypatch):
    """A DOI absent from the bulk response must NOT be seeded as a miss: its
    own lookup still needs to 404, which stops enrichment for that reference
    rather than falling through to a bibliographic search."""
    client = _client()

    async def fake_request(path, params=None):  # noqa: ARG001
        return _bulk_response([_item("10.1/found")])

    monkeypatch.setattr(client, "_request", fake_request)
    seeded = await client.prefetch_works_by_doi(["10.1/found", "10.1/missing"])

    assert seeded == 1
    assert client._cache_peek("works:10.1/missing") is None
    assert client._cache_peek("works:10.1/found") is not None


@pytest.mark.asyncio
async def test_chunks_large_reference_lists(monkeypatch):
    client = _client()
    calls = []

    async def fake_request(path, params=None):  # noqa: ARG001
        dois = [term.removeprefix("doi:") for term in params["filter"].split(",")]
        calls.append(dois)
        return _bulk_response([_item(d) for d in dois])

    monkeypatch.setattr(client, "_request", fake_request)
    dois = [f"10.1/{i}" for i in range(mod._BULK_DOI_CHUNK + 5)]

    seeded = await client.prefetch_works_by_doi(dois)

    assert seeded == len(dois)
    assert len(calls) == 2
    assert len(calls[0]) == mod._BULK_DOI_CHUNK
    assert len(calls[1]) == 5


@pytest.mark.asyncio
async def test_duplicate_and_already_cached_dois_are_not_re_requested(monkeypatch):
    client = _client()
    calls = []

    async def fake_request(path, params=None):  # noqa: ARG001
        calls.append(params["filter"])
        return _bulk_response([_item("10.1/a"), _item("10.1/b")])

    monkeypatch.setattr(client, "_request", fake_request)
    await client.prefetch_works_by_doi(["10.1/a", "10.1/A", "10.1/b"])
    assert calls[0] == "doi:10.1/a,doi:10.1/b"

    # A second pass has nothing left to fetch.
    assert await client.prefetch_works_by_doi(["10.1/a", "10.1/b"]) == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_unsafe_dois_are_skipped(monkeypatch):
    client = _client()
    calls = []

    async def fake_request(path, params=None):  # noqa: ARG001
        calls.append(params["filter"])
        return _bulk_response([_item("10.1/ok")])

    monkeypatch.setattr(client, "_request", fake_request)
    await client.prefetch_works_by_doi(["10.1/ok", "../etc/passwd", ""])

    assert calls == ["doi:10.1/ok"]


@pytest.mark.asyncio
async def test_request_failure_degrades_to_the_per_reference_path(monkeypatch):
    """Enrichment must never fail because a prefetch did."""
    client = _client()

    async def boom(*_a, **_kw):
        raise RuntimeError("crossref down")

    monkeypatch.setattr(client, "_request", boom)

    assert await client.prefetch_works_by_doi(["10.1/a"]) == 0
    assert client._cache_peek("works:10.1/a") is None


@pytest.mark.asyncio
async def test_empty_input_makes_no_request(monkeypatch):
    client = _client()

    async def explode(*_a, **_kw):
        raise AssertionError("no request expected")

    monkeypatch.setattr(client, "_request", explode)

    assert await client.prefetch_works_by_doi([]) == 0


@pytest.mark.asyncio
async def test_prefetch_is_a_noop_when_the_lru_is_disabled(monkeypatch):
    """cache_size=0 means nothing can be seeded in-process; the prefetch still
    must not crash or poison the per-ref path."""
    client = _client(cache_size=0)

    async def fake_request(path, params=None):  # noqa: ARG001
        return _bulk_response([_item("10.1/a")])

    monkeypatch.setattr(client, "_request", fake_request)

    assert await client.prefetch_works_by_doi(["10.1/a"]) == 1
    assert client._cache_peek("works:10.1/a") is None


@pytest.mark.asyncio
async def test_end_to_end_request_count_drops_to_one(monkeypatch):
    """The point of the change: a real client driven through enrich_references
    spends one request for every DOI-bearing reference, not one each."""
    from bibr.enrich.references import enrich_references
    from bibr.paper import PaperReference

    client = _client()
    paths = []

    def _work(doi):
        return {
            "DOI": doi,
            "title": ["Test Paper"],
            "author": [{"given": "J.", "family": "Smith"}],
            "issued": {"date-parts": [[2020]]},
            "type": "journal-article",
        }

    async def fake_request(path, params=None):
        paths.append(path)
        if path == "/works":
            dois = [t.removeprefix("doi:") for t in params["filter"].split(",")]
            return _bulk_response([_work(d) for d in dois])
        raise AssertionError(f"unexpected per-DOI request: {path}")

    monkeypatch.setattr(client, "_request", fake_request)

    refs = [
        PaperReference(
            bib_id=i,
            title="Test Paper",
            first_page=None,
            volume=None,
            authors="Smith, J.",
            year=2020,
            container=None,
            doi=f"10.1000/{i}",
        )
        for i in range(12)
    ]

    await enrich_references(refs, crossref_client=client)

    assert paths == ["/works"], f"expected a single bulk request, got {paths}"
    assert all(r.match for r in refs), "every reference should still be enriched"
