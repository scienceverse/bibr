"""One Crossref client shared by threads that each run their own event loop.

``get_client(settings)`` hands one instance to every caller in the process:
``bibr.chew()`` called from a web threadpool, or two ``Chewer`` objects. Each
loop switch used to replace "the" HTTP client and close the other thread's
under its in-flight requests, and to rebuild the rate limiter, handing every
thread a fresh rate budget.
"""

import asyncio
import threading
from unittest.mock import patch

import httpx
import pytest

from bibr.clients.crossref import _CROSSREF_API_BASE, CrossrefClient
from bibr.config import snapshot_settings
from bibr.utils.rate_limiter import AsyncLocalRateLimiter


def _client(**crossref) -> CrossrefClient:
    settings = snapshot_settings().model_copy(deep=True)
    settings.redis.url = None
    settings.crossref.cache_size = 0
    settings.crossref.redis_cache = False
    settings.crossref.rate_limit_rpm = 100_000
    for name, value in crossref.items():
        setattr(settings.crossref, name, value)
    client = CrossrefClient(mailto="test@example.com", settings=settings)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"DOI": request.url.path}})

    client._make_http_client = lambda: httpx.AsyncClient(
        base_url=_CROSSREF_API_BASE, transport=httpx.MockTransport(handler)
    )
    return client


def _run_threads(*coro_fns, timeout: float = 10.0) -> list[BaseException]:
    errors: list[BaseException] = []

    def target(fn):
        try:
            asyncio.run(fn())
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            errors.append(exc)

    threads = [threading.Thread(target=target, args=(fn,), daemon=True) for fn in coro_fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    assert not any(t.is_alive() for t in threads), "a thread hung"
    return errors


def test_other_threads_loop_does_not_close_this_loops_http_client():
    """Reproduced: 2 of 60 requests failed with 'the client has been closed'."""
    client = _client()
    a_has_client, b_requested = threading.Event(), threading.Event()
    seen: dict[str, object] = {}

    async def loop_a():
        http = await client._get_http_client()
        a_has_client.set()
        while not b_requested.is_set():
            await asyncio.sleep(0.005)
        # A request already holding its client when the other thread got in.
        seen["a"] = (await http.get("/works/10.1/a")).json()
        seen["a_client"] = http

    async def loop_b():
        while not a_has_client.is_set():
            await asyncio.sleep(0.005)
        seen["b"] = await client.works("10.1/b")
        seen["b_client"] = await client._get_http_client()
        b_requested.set()

    assert _run_threads(loop_a, loop_b) == []
    assert seen["a"] == {"message": {"DOI": "/works/10.1/a"}}
    assert seen["b"] == {"message": {"DOI": "/works/10.1/b"}}
    assert seen["a_client"] is not seen["b_client"]


def test_threads_share_one_rate_limiter():
    """The rate budget is per process: a second loop must not get a fresh limiter."""
    client = _client()
    limiters: list[object] = []
    a_done = threading.Event()

    async def loop_a():
        await client.works("10.1/a")
        limiters.append(client.limiter)
        a_done.set()

    async def loop_b():
        while not a_done.is_set():
            await asyncio.sleep(0.005)
        await client.works("10.1/b")
        limiters.append(client.limiter)

    assert _run_threads(loop_a, loop_b) == []
    assert isinstance(limiters[0], AsyncLocalRateLimiter)
    assert limiters[0] is limiters[1]


def test_enrich_semaphore_bounds_each_loop_despite_other_threads():
    """A loop switch used to swap in a fresh semaphore, letting a second holder in."""
    client = _client(enrich_concurrency=1)
    a_holds, b_held = threading.Event(), threading.Event()
    outcome: dict[str, bool] = {}

    async def loop_a():
        async with client.enrich_semaphore:
            a_holds.set()
            while not b_held.is_set():
                await asyncio.sleep(0.005)
            try:
                async with asyncio.timeout(0.1):
                    async with client.enrich_semaphore:
                        outcome["second_holder_admitted"] = True
            except TimeoutError:
                outcome["second_holder_admitted"] = False

    async def loop_b():
        while not a_holds.is_set():
            await asyncio.sleep(0.005)
        async with client.enrich_semaphore:  # its own loop's bound
            b_held.set()

    assert _run_threads(loop_a, loop_b) == []
    assert outcome == {"second_holder_admitted": False}


def test_closed_loops_http_clients_are_closed_not_leaked():
    """Successive asyncio.run() calls: a dead loop's client is closed and dropped."""
    client = _client()
    clients: list[httpx.AsyncClient] = []

    async def run_once():
        await client.works("10.1/x")
        clients.append(await client._get_http_client())

    for _ in range(3):
        asyncio.run(run_once())

    assert len({id(c) for c in clients}) == 3
    assert clients[0].is_closed and clients[1].is_closed
    assert not clients[2].is_closed
    assert list(client._http_clients.values()) == [clients[2]]


def test_close_leaves_other_running_loops_client_open():
    client = _client()
    a_has_client, b_closed = threading.Event(), threading.Event()
    seen: dict[str, bool] = {}

    async def loop_a():
        http = await client._get_http_client()
        a_has_client.set()
        while not b_closed.is_set():
            await asyncio.sleep(0.005)
        seen["a_closed"] = http.is_closed

    async def loop_b():
        while not a_has_client.is_set():
            await asyncio.sleep(0.005)
        await client._get_http_client()
        await client.close()
        b_closed.set()

    assert _run_threads(loop_a, loop_b) == []
    assert seen == {"a_closed": False}


async def test_limiter_redis_probe_is_bounded():
    """A Redis that accepts the connection but never answers held the first request."""

    class WedgedRedis:
        async def ping(self):
            await asyncio.Event().wait()

        async def aclose(self):
            pass

    client = _client()
    client._settings.redis.url = "redis://127.0.0.1:6379/0"
    with patch("redis.asyncio.Redis") as redis_cls:
        redis_cls.from_url.return_value = WedgedRedis()
        try:
            await asyncio.wait_for(client._ensure_limiter(), timeout=5)
        except TimeoutError:
            pytest.fail("the Redis probe was not bounded")
    assert isinstance(client.limiter, AsyncLocalRateLimiter)
