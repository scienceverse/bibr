import httpx
import pytest

from bibr.clients.resolver import ResolverClient


def _client(handler):
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(base_url="http://resolver", transport=transport)
    return ResolverClient("http://resolver", client=http)


async def test_search_returns_candidates():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"candidates": [{"title": "X", "source": "openalex", "doi": "10.1/x"}]}
        )

    rc = _client(handler)
    out = await rc.search("X", 2020, 5)
    assert captured["path"] == "/search"
    assert captured["body"] == {"title": "X", "year": 2020, "limit": 5}
    assert out == [{"title": "X", "source": "openalex", "doi": "10.1/x"}]
    await rc.close()


async def test_search_includes_sources_when_provided():
    """A ``sources`` list is forwarded in the POST body so the resolver queries the
    named corpora (crossref+openalex) instead of its crossref-only default."""
    captured = {}

    def handler(request):
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": []})

    rc = _client(handler)
    await rc.search("X", 2020, 5, sources=["crossref", "openalex"])
    assert captured["body"] == {
        "title": "X",
        "year": 2020,
        "limit": 5,
        "sources": ["crossref", "openalex"],
    }
    await rc.close()


async def test_search_omits_sources_key_when_absent():
    """No ``sources`` argument → no ``sources`` key in the body, so the resolver keeps
    its default tier. Preserves the pre-existing request shape."""
    captured = {}

    def handler(request):
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": []})

    rc = _client(handler)
    await rc.search("X", 2020, 5)
    assert "sources" not in captured["body"]
    await rc.close()


async def test_search_many_forwards_sources():
    bodies = []

    def handler(request):
        import json

        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": []})

    rc = _client(handler)
    await rc.search_many(
        [{"title": "a", "year": None, "limit": 5}, {"title": "b", "year": None, "limit": 5}],
        sources=["crossref", "openalex"],
    )
    assert all(b.get("sources") == ["crossref", "openalex"] for b in bodies)
    await rc.close()


async def test_search_error_returns_empty():
    def handler(request):
        return httpx.Response(500)

    rc = _client(handler)
    assert await rc.search("X", None, 5) == []
    await rc.close()


async def test_search_many_posts_one_search_per_query_aligned_results():
    captured = {"paths": [], "bodies": []}

    def handler(request):
        import json

        captured["paths"].append(request.url.path)
        body = json.loads(request.content)
        captured["bodies"].append(body)
        if body["title"] == "alpha":
            return httpx.Response(200, json={"candidates": [{"title": "A", "source": "openalex"}]})
        return httpx.Response(200, json={"candidates": []})

    rc = _client(handler)
    queries = [
        {"title": "alpha", "year": 2020, "limit": 5},
        {"title": "beta", "year": None, "limit": 5},
    ]
    out = await rc.search_many(queries)
    assert captured["paths"] == ["/search", "/search"]
    # results align 1:1 with queries, by order, even though requests may complete out of order
    assert out == [[{"title": "A", "source": "openalex"}], []]
    await rc.close()


async def test_search_many_per_query_error_isolated_to_that_slot():
    def handler(request):
        import json

        body = json.loads(request.content)
        if body["title"] == "bad":
            return httpx.Response(500)
        return httpx.Response(200, json={"candidates": [{"title": "ok"}]})

    rc = _client(handler)
    out = await rc.search_many(
        [
            {"title": "good1", "year": None, "limit": 5},
            {"title": "bad", "year": None, "limit": 5},
            {"title": "good2", "year": None, "limit": 5},
        ]
    )
    assert out == [[{"title": "ok"}], [], [{"title": "ok"}]]
    await rc.close()


async def test_search_raise_on_error_reraises():
    """With raise_on_error, a transport error propagates instead of degrading to []
    so an authoritative caller can tell a real error apart from an empty result."""

    def handler(request):
        return httpx.Response(500)

    rc = _client(handler)
    with pytest.raises(httpx.HTTPError):
        await rc.search("X", None, 5, raise_on_error=True)
    await rc.close()


async def test_search_many_raise_on_error_isolates_exception_in_slot():
    """A per-query error under raise_on_error is captured as an Exception in that
    slot (not raised out of the batch), keeping the other slots' results intact."""

    def handler(request):
        import json

        body = json.loads(request.content)
        if body["title"] == "bad":
            return httpx.Response(500)
        return httpx.Response(200, json={"candidates": [{"title": "ok"}]})

    rc = _client(handler)
    out = await rc.search_many(
        [
            {"title": "good1", "year": None, "limit": 5},
            {"title": "bad", "year": None, "limit": 5},
            {"title": "good2", "year": None, "limit": 5},
        ],
        raise_on_error=True,
    )
    assert out[0] == [{"title": "ok"}]
    assert isinstance(out[1], Exception)
    assert out[2] == [{"title": "ok"}]
    await rc.close()


async def test_search_many_empty_queries_makes_no_request():
    def handler(request):
        raise AssertionError("must not call the resolver with an empty query list")

    rc = _client(handler)
    assert await rc.search_many([]) == []
    await rc.close()


async def test_search_many_respects_concurrency_bound():
    """concurrency must actually cap in-flight requests, not just be accepted and ignored —
    otherwise a large reference list floods the resolver with an unbounded fan-out."""
    import asyncio

    in_flight = {"current": 0, "max": 0}

    async def handler(request):
        in_flight["current"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["current"])
        await asyncio.sleep(0.02)
        in_flight["current"] -= 1
        return httpx.Response(200, json={"candidates": []})

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(base_url="http://resolver", transport=transport)
    rc = ResolverClient("http://resolver", client=http)
    queries = [{"title": f"t{i}", "year": None, "limit": 5} for i in range(10)]
    await rc.search_many(queries, concurrency=3)
    assert in_flight["max"] <= 3
    await rc.close()


async def test_lookup_doi_returns_candidate():
    def handler(request):
        assert request.url.path == "/works/10.1/abc"
        return httpx.Response(200, json={"title": "Y", "source": "openalex", "doi": "10.1/abc"})

    rc = _client(handler)
    out = await rc.lookup_doi("10.1/abc")
    assert out["doi"] == "10.1/abc"
    await rc.close()


async def test_lookup_doi_404_returns_none():
    def handler(request):
        return httpx.Response(404)

    rc = _client(handler)
    assert await rc.lookup_doi("10.1/missing") is None
    await rc.close()


async def test_lookup_doi_error_returns_none():
    def handler(request):
        return httpx.Response(503)

    rc = _client(handler)
    assert await rc.lookup_doi("10.1/x") is None
    await rc.close()


async def test_lookup_doi_raise_on_error_reraises_non_404():
    def handler(request):
        return httpx.Response(503)

    rc = _client(handler)
    with pytest.raises(httpx.HTTPError):
        await rc.lookup_doi("10.1/x", raise_on_error=True)
    await rc.close()


async def test_lookup_doi_raise_on_error_404_still_returns_none():
    """A 404 is a genuine not-found, not an error — it returns None even under
    raise_on_error so an authoritative caller treats it as a clean miss."""

    def handler(request):
        return httpx.Response(404)

    rc = _client(handler)
    assert await rc.lookup_doi("10.1/missing", raise_on_error=True) is None
    await rc.close()


async def test_health_ok_returns_true():
    def handler(request):
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "index": "works", "num_docs": 42})

    rc = _client(handler)
    assert await rc.healthy() is True
    await rc.close()


async def test_health_degraded_returns_false():
    def handler(request):
        return httpx.Response(200, json={"status": "degraded", "index": "works"})

    rc = _client(handler)
    assert await rc.healthy() is False
    await rc.close()


async def test_health_unreachable_returns_false():
    def handler(request):
        return httpx.Response(503)

    rc = _client(handler)
    assert await rc.healthy() is False
    await rc.close()


async def test_health_cached_probes_once():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"status": "ok", "index": "works"})

    rc = _client(handler)
    assert await rc.healthy() is True
    assert await rc.healthy() is True
    assert calls["n"] == 1
    await rc.close()


# --- SSRF / path-traversal hardening (audit M1) -------------------------------


async def test_lookup_doi_rejects_dotdot_traversal_without_network():
    """A DOI with a '..' path segment (e.g. extracted from a malicious PDF)
    must never reach the resolver host — it would escape ``/works/*`` since
    ``quote(doi, safe='/')`` preserves '/' and '.' and httpx does not
    normalize '..'. It degrades to a clean miss (None)."""
    called = {"hit": False}

    def handler(request):
        called["hit"] = True
        return httpx.Response(200, json={"secret": "internal"})

    rc = _client(handler)
    assert await rc.lookup_doi("10.1234/../../jobs/running") is None
    assert called["hit"] is False, "traversal DOI reached the network"
    await rc.close()


async def test_lookup_doi_rejects_non_doi_absolute_path():
    """The report's literal example ``../jobs/running`` — no '10.' prefix — is
    refused before any request."""
    called = {"hit": False}

    def handler(request):
        called["hit"] = True
        return httpx.Response(200, json={})

    rc = _client(handler)
    assert await rc.lookup_doi("../jobs/running") is None
    assert called["hit"] is False
    await rc.close()


async def test_lookup_doi_unsafe_returns_none_even_under_raise_on_error():
    """An unsafe DOI is a bad input, not a transient transport error, so it is a
    clean miss (None) rather than a raise — the caller falls through to CrossRef."""
    called = {"hit": False}

    def handler(request):
        called["hit"] = True
        return httpx.Response(200, json={})

    rc = _client(handler)
    assert await rc.lookup_doi("10.5/a/../../etc", raise_on_error=True) is None
    assert called["hit"] is False
    await rc.close()


async def test_lookup_doi_allows_legit_doi_with_dots_in_suffix():
    """Regression guard: '.' inside a path *segment* (normal in real DOIs) is
    fine — only a standalone '.'/'..' segment traverses."""

    def handler(request):
        assert request.url.path == "/works/10.1016/j.foo.2024"
        return httpx.Response(200, json={"doi": "10.1016/j.foo.2024"})

    rc = _client(handler)
    out = await rc.lookup_doi("10.1016/j.foo.2024")
    assert out == {"doi": "10.1016/j.foo.2024"}
    await rc.close()


# --- Response shapes ----------------------------------------------------------
# The client promises the resolver never fails enrichment. A proxy or another
# service answering with other JSON used to raise AttributeError out of
# healthy(), which failed the paper's whole reference enrichment.


@pytest.mark.parametrize("body", [["ok"], "ok", 1, None])
async def test_health_non_object_body_is_unhealthy(body):
    rc = _client(lambda request: httpx.Response(200, json=body))
    assert await rc.healthy() is False
    await rc.close()


async def test_lookup_doi_non_object_body_is_a_miss():
    rc = _client(lambda request: httpx.Response(200, json=["not", "a", "work"]))
    assert await rc.lookup_doi("10.1/x") is None
    await rc.close()


async def test_lookup_doi_non_object_body_is_an_error_when_asked():
    rc = _client(lambda request: httpx.Response(200, json=["not", "a", "work"]))
    with pytest.raises(ValueError, match="not a work"):
        await rc.lookup_doi("10.1/x", raise_on_error=True)
    await rc.close()


async def test_search_drops_candidates_that_are_not_objects():
    body = {"candidates": [{"title": "Kept"}, "stray", None, 3]}
    rc = _client(lambda request: httpx.Response(200, json=body))
    assert await rc.search("A title", 2020, 5) == [{"title": "Kept"}]
    await rc.close()
