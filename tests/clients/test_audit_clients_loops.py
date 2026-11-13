"""Process-wide clients reused from a second event loop."""

from __future__ import annotations

import asyncio
import threading

import pytest


@pytest.fixture
def redis_url():
    from fakeredis import TcpFakeServer

    server = TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"redis://127.0.0.1:{server.server_address[1]}/0"
    finally:
        server.shutdown()
        server.server_close()


def test_crossref_redis_cache_answers_from_every_asyncio_run(redis_url):
    # bibr.chew() runs each call in its own asyncio.run(), while the CrossRef
    # client (and so its tier-2 cache) is shared by the whole process.
    from bibr.clients.response_cache import RedisResponseCache

    cache = RedisResponseCache(redis_url, connect_timeout=2, socket_timeout=2)
    asyncio.run(cache.set("works:10.1/x", {"message": {"DOI": "10.1/x"}}))

    answers = [asyncio.run(cache.get("works:10.1/x")) for _ in range(3)]

    assert answers == [{"message": {"DOI": "10.1/x"}}] * 3
    assert cache.misses == 0


def test_crossref_redis_cache_serves_threads_with_their_own_loops(redis_url):
    from bibr.clients.response_cache import RedisResponseCache

    cache = RedisResponseCache(redis_url, connect_timeout=2, socket_timeout=2)
    asyncio.run(cache.set("k", {"v": 1}))
    results: list = []

    def worker() -> None:
        for _ in range(3):
            results.append(asyncio.run(cache.get("k")))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)

    assert results == [{"v": 1}] * 6


def test_ror_client_keeps_one_http_pool_per_live_loop(monkeypatch):
    # Two bibr.Chewer sessions in one thread each drive a private loop that
    # stays open between calls, and share the process-wide ROR client.
    import httpx

    from bibr.clients import ror
    from bibr.config import GlobalSettings

    made: list[httpx.AsyncClient] = []
    real = httpx.AsyncClient

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    def build(**kwargs):
        client = real(transport=httpx.MockTransport(answer), **kwargs)
        made.append(client)
        return client

    monkeypatch.setattr(ror.httpx, "AsyncClient", build)
    client = ror.RorClient(settings=GlobalSettings())
    first, second = asyncio.new_event_loop(), asyncio.new_event_loop()
    try:
        for loop, text in ((first, "Uni A"), (second, "Uni B"), (first, "Uni C")):
            assert loop.run_until_complete(client.lookup(text)) == (None, None)

        assert len(made) == 2
        assert not any(c.is_closed for c in made)
        first.run_until_complete(client.aclose())
        second.run_until_complete(client.aclose())
        assert all(c.is_closed for c in made)
    finally:
        first.close()
        second.close()
