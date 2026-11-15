"""Process-wide clients reused from a second event loop."""

from __future__ import annotations

import asyncio
import threading

import pytest


def _bounded(fn, seconds: float = 60.0):
    """Run *fn* in a thread: a hang on real sockets fails the test, not the suite."""
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the test's thread
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"still running after {seconds:.0f} s"
    if "error" in box:
        raise box["error"]
    return box.get("value")


@pytest.fixture
def redis_url():
    from fakeredis import TcpFakeServer

    server = TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    # A connection whose event loop is gone is closed when it is collected, not
    # before: server_close() must not wait for the handler thread serving it
    # (on Windows it waited for the rest of the run).
    server.daemon_threads = True
    server.block_on_close = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"redis://127.0.0.1:{server.server_address[1]}/0"
    finally:
        closer = threading.Thread(
            target=lambda: (server.shutdown(), server.server_close()), daemon=True
        )
        closer.start()
        closer.join(10)


def test_crossref_redis_cache_answers_from_every_asyncio_run(redis_url):
    # bibr.chew() runs each call in its own asyncio.run(), while the CrossRef
    # client (and so its tier-2 cache) is shared by the whole process.
    from bibr.clients.response_cache import RedisResponseCache

    cache = RedisResponseCache(redis_url, connect_timeout=2, socket_timeout=2)

    def calls() -> list:
        asyncio.run(cache.set("works:10.1/x", {"message": {"DOI": "10.1/x"}}))
        return [asyncio.run(cache.get("works:10.1/x")) for _ in range(3)]

    answers = _bounded(calls)

    assert answers == [{"message": {"DOI": "10.1/x"}}] * 3
    assert cache.misses == 0


def test_crossref_redis_cache_serves_threads_with_their_own_loops(redis_url):
    from bibr.clients.response_cache import RedisResponseCache

    cache = RedisResponseCache(redis_url, connect_timeout=2, socket_timeout=2)
    _bounded(lambda: asyncio.run(cache.set("k", {"v": 1})))
    results: list = []

    def worker() -> None:
        for _ in range(3):
            results.append(asyncio.run(cache.get("k")))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not any(t.is_alive() for t in threads)
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
