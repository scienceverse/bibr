"""The LLM rate limiter's Redis probe must not block the event loop.

``bibr serve`` runs one LitServe worker with ``enable_async=True``, so a single
event loop serves every concurrent request. Any synchronous network call inside
a coroutine head-of-line-blocks every in-flight paper, not just its own caller.
"""

import asyncio
import socket
import threading
import time

import pytest

# The whole point of this module is what the *async Redis probe* does to the
# loop. `redis` ships in the `cache` extra, and the core CI environments sync
# without it — there `_ensure_limiter` fails on the import before it ever
# opens a socket, so the stall being measured never happens.
pytest.importorskip("redis.asyncio")

from bibr.clients.llm import LLMClient  # noqa: E402
from bibr.config import GlobalSettings  # noqa: E402
from bibr.utils.rate_limiter import AsyncLocalRateLimiter  # noqa: E402

# How long the stand-in Redis stays silent after accepting. Long enough that a
# blocking probe is unmistakable against the heartbeat below, short enough to
# stay well inside the probe timeout.
_STALL_SECONDS = 0.5
_HEARTBEAT_INTERVAL = 0.01


class _StallingRedis:
    """Accepts the connection, answers nothing, then hangs up.

    A Redis that is *reachable but wedged* — the case the probe's
    ``socket_connect_timeout`` does not cover, because the connect succeeds and
    it is the command round-trip that never completes. Runs on its own thread
    so it can still serve a client whose event loop is frozen.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        host, port = self._sock.getsockname()
        self.url = f"redis://{host}:{port}/0"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self._sock.settimeout(0.05)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            threading.Thread(target=self._stall, args=(conn,), daemon=True).start()

    @staticmethod
    def _stall(conn: socket.socket) -> None:
        try:
            conn.recv(4096)
            time.sleep(_STALL_SECONDS)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close()


@pytest.fixture
def stalling_redis():
    server = _StallingRedis()
    try:
        yield server
    finally:
        server.close()


def _client(redis_url: str) -> LLMClient:
    # ``password`` is pinned empty because a set password makes GlobalSettings
    # synthesize ``redis://:<pw>@redis:6379/0`` for an empty url — the suite's
    # conftest exports REDIS_PASSWORD, so "" would otherwise mean "an
    # unresolvable host", not "no Redis configured".
    settings = GlobalSettings(redis={"url": redis_url, "password": ""})
    client = LLMClient.__new__(LLMClient)  # skip provider init entirely
    client._settings = settings
    client._limiter = None
    client._limiter_init_lock = None
    client._limiter_init_loop = None
    return client


async def _count_ticks(stop: asyncio.Event) -> int:
    """Tick until told to stop — the loop's own liveness, measured."""
    ticks = 0
    while not stop.is_set():
        await asyncio.sleep(_HEARTBEAT_INTERVAL)
        ticks += 1
    return ticks


async def test_a_wedged_redis_does_not_freeze_the_event_loop(stalling_redis):
    client = _client(stalling_redis.url)
    stop = asyncio.Event()
    heartbeat = asyncio.create_task(_count_ticks(stop))
    await asyncio.sleep(0)  # let the heartbeat reach its first await

    await client._ensure_limiter()

    stop.set()
    ticks = await heartbeat

    # A synchronous probe pins the loop for the whole stall: the heartbeat gets
    # no chance to run and comes back with 0-1 ticks. Off-loop, it keeps its
    # 10 ms cadence throughout.
    assert ticks >= _STALL_SECONDS / _HEARTBEAT_INTERVAL / 2
    assert client._limiter is not None


async def test_an_unreachable_redis_falls_back_to_the_local_limiter():
    client = _client("redis://127.0.0.1:1/0")
    await client._ensure_limiter()
    assert isinstance(client._limiter, AsyncLocalRateLimiter)


async def test_no_redis_url_configured_uses_the_local_limiter():
    client = _client("")
    await client._ensure_limiter()
    assert isinstance(client._limiter, AsyncLocalRateLimiter)


async def test_concurrent_first_callers_build_exactly_one_limiter():
    """Without the init lock both callers probe and one limiter is orphaned."""
    client = _client("")
    built: list[object] = []

    original = LLMClient._ensure_limiter

    async def tracking(self):
        await original(self)
        built.append(self._limiter)

    await asyncio.gather(*(tracking(client) for _ in range(4)))
    assert len({id(limiter) for limiter in built}) == 1


async def test_the_limiter_property_does_not_build_anything():
    """Construction is _ensure_limiter's job; the property must not block."""
    client = _client("")
    assert client.limiter is None
    await client._ensure_limiter()
    assert client.limiter is client._limiter
