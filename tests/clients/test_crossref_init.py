def test_adapt_rate_limit_does_not_use_threading_lock():
    """Verify the module no longer uses a module-level rate-limit lock.

    Regression guard for H6: a `threading.Lock` acquired from inside the
    asyncio event loop blocks the loop. The single attribute write
    `self.limiter.window_seconds = ...` is atomic in CPython, so no lock
    is needed.
    """
    import bibr.clients.crossref as mod

    assert not hasattr(mod, "_rate_limit_lock"), (
        "threading.Lock for rate-limit adaptation should be removed"
    )

    # Smoke: _adapt_rate_limit runs cleanly and updates the limiter.
    client = mod.CrossrefClient.__new__(mod.CrossrefClient)
    client.limiter = type("L", (), {"window_seconds": 0.1})()

    fake_headers = {"x-rate-limit-limit": "5", "x-rate-limit-interval": "1s"}
    client._adapt_rate_limit(type("H", (), {"get": fake_headers.get})())

    # 1s / 5 req = 0.2s server-suggested interval > 0.1s configured -> adopted
    assert client.limiter.window_seconds == 0.2
    assert client._recovery_floor == 0.2


def test_construct_does_not_block_on_redis_ping(monkeypatch):
    """Constructing CrossrefClient should not perform a sync Redis ping."""
    import sys
    import time
    import types

    import bibr.clients.crossref as mod

    pings = []

    class FakeSyncRedis:
        @classmethod
        def from_url(cls, *_a, **_kw):
            pings.append("from_url")
            return cls()

        def ping(self):
            pings.append("ping")
            time.sleep(1.0)

        def close(self):
            pass

    fake_redis_module = types.ModuleType("redis")
    fake_redis_module.Redis = FakeSyncRedis
    monkeypatch.setitem(sys.modules, "redis", fake_redis_module)

    t0 = time.monotonic()
    client = mod.CrossrefClient(mailto="t@x.com")
    elapsed = time.monotonic() - t0
    assert elapsed < 0.2, f"Construction took {elapsed:.2f}s — must not ping Redis"
    assert pings == [], f"Constructor must not perform any Redis call (was: {pings!r})"

    # Limiter is unset until first request triggers _ensure_limiter
    assert client.limiter is None


def test_httpx_timeout_uses_request_timeout(monkeypatch):
    """The per-HTTP-request timeout must be driven by ``request_timeout``, NOT
    by the per-paper ``enrich_timeout`` budget — otherwise a single hung call
    can monopolize a concurrency slot for the whole 120s paper budget."""
    import bibr.clients.crossref as mod
    from bibr.config import Settings

    monkeypatch.setattr(Settings.crossref, "request_timeout", 15.0)
    monkeypatch.setattr(Settings.crossref, "enrich_timeout", 120.0)
    client = mod.CrossrefClient(mailto="t@x.com")
    timeout = client._make_http_client().timeout
    # httpx.Timeout exposes per-aspect attrs all set to the same value when
    # constructed from a single float.
    assert timeout.connect == 15.0


def test_adapt_rate_limit_runs_on_429(monkeypatch):
    """Crossref returns the new limit in headers on 429 — we must adopt it
    on the failure path, not only on successful responses."""
    import bibr.clients.crossref as mod

    client = mod.CrossrefClient.__new__(mod.CrossrefClient)
    client.limiter = type("L", (), {"window_seconds": 0.1})()

    fake_headers = {"x-rate-limit-limit": "2", "x-rate-limit-interval": "1s"}
    headers_obj = type("H", (), {"get": fake_headers.get})()
    client._adapt_rate_limit(headers_obj)
    # 1s / 2 req = 0.5s server-suggested interval > 0.1s configured -> adopted
    assert client.limiter.window_seconds == 0.5
    assert client._recovery_floor == 0.5


def _headers(mapping):
    return type("H", (), {"get": mapping.get})()


def test_throttle_on_429_without_headers_widens_window():
    """Crossref stopped sending X-Rate-Limit headers — a header-less 429 must
    still tighten the local limiter (multiplicative), or a storm of resolver
    misses hammers the search endpoint at the configured RPM forever."""
    import bibr.clients.crossref as mod

    client = mod.CrossrefClient.__new__(mod.CrossrefClient)
    client.limiter = type("L", (), {"window_seconds": 0.1})()

    client._throttle_on_429(_headers({}))
    assert client.limiter.window_seconds == 0.2
    client._throttle_on_429(_headers({}))
    assert client.limiter.window_seconds == 0.4


def test_throttle_on_429_is_capped():
    import bibr.clients.crossref as mod

    client = mod.CrossrefClient.__new__(mod.CrossrefClient)
    client.limiter = type("L", (), {"window_seconds": 1.8})()

    client._throttle_on_429(_headers({}))
    assert client.limiter.window_seconds == mod._MAX_THROTTLE_WINDOW_SECONDS
    client._throttle_on_429(_headers({}))
    assert client.limiter.window_seconds == mod._MAX_THROTTLE_WINDOW_SECONDS


def test_throttle_on_429_defers_to_rate_limit_headers():
    """When Crossref DOES advertise a limit, the header-based adaptation owns
    the window — the blind multiplicative widening must not double-apply."""
    import bibr.clients.crossref as mod

    client = mod.CrossrefClient.__new__(mod.CrossrefClient)
    client.limiter = type("L", (), {"window_seconds": 0.1})()

    client._throttle_on_429(_headers({"x-rate-limit-limit": "5", "x-rate-limit-interval": "1s"}))
    # Untouched: _adapt_rate_limit (called separately on every response)
    # handles the header path.
    assert client.limiter.window_seconds == 0.1


def test_enrich_semaphore_rebinds_across_event_loops():
    """M6: a second asyncio.run() must not hit a semaphore bound to the
    first (dead) loop — that RuntimeError silently stopped enrichment.

    The uncontended fast path never binds the loop, so the test parks a
    waiter to force binding."""
    import asyncio

    import bibr.clients.crossref as mod

    client = mod.CrossrefClient(mailto="t@x.com")
    client._enrich_concurrency = 1

    async def contended_use():
        sem = client.enrich_semaphore
        async with sem:
            waiter = asyncio.create_task(sem.acquire())
            await asyncio.sleep(0)  # waiter parks on a loop-bound future
        await waiter
        sem.release()

    asyncio.run(contended_use())
    asyncio.run(contended_use())  # must not raise


def test_http_client_rebinds_across_event_loops(monkeypatch):
    """Repeated library calls must not retain a client bound to a dead loop."""
    import asyncio

    import bibr.clients.crossref as mod

    client = mod.CrossrefClient(mailto="t@x.com")
    created = []

    class FakeHttp:
        def __init__(self):
            self.is_closed = False
            created.append(self)

        async def aclose(self):
            self.is_closed = True

    monkeypatch.setattr(client, "_make_http_client", FakeHttp)

    first = asyncio.run(client._get_http_client())
    second = asyncio.run(client._get_http_client())

    assert second is not first
    assert first.is_closed is True
    assert len(created) == 2


def test_anonymous_pool_clamp_names_the_configured_rate(caplog):
    """Without a mailto the polite pool is unavailable, so a configured 200 RPM
    silently becomes 60. Saying so beats leaving the operator to infer a 3.3x
    enrichment slowdown from wall-clock."""
    import logging

    import bibr.clients.crossref as mod
    from bibr.config import GlobalSettings

    settings = GlobalSettings(crossref={"api_email": None, "rate_limit_rpm": 200})
    with caplog.at_level(logging.WARNING, logger="bibr.clients.crossref"):
        client = mod.CrossrefClient(settings=settings)

    assert client.interval == 1.0  # 60 RPM, not the configured 200
    assert client._anonymous_clamped_from == 200
    assert "60 RPM, not the configured 200" in caplog.text


def test_no_clamp_warning_when_configured_at_or_below_the_anonymous_cap(caplog):
    """Nothing was taken away, so the specific warning must not fire."""
    import logging

    import bibr.clients.crossref as mod
    from bibr.config import GlobalSettings

    settings = GlobalSettings(crossref={"api_email": None, "rate_limit_rpm": 30})
    with caplog.at_level(logging.WARNING, logger="bibr.clients.crossref"):
        client = mod.CrossrefClient(settings=settings)

    assert client._anonymous_clamped_from is None
    assert client.interval == 2.0
    assert "not the configured" not in caplog.text


def test_polite_pool_keeps_the_configured_rate(caplog):
    import logging

    import pytest

    import bibr.clients.crossref as mod
    from bibr.config import GlobalSettings

    settings = GlobalSettings(crossref={"api_email": "someone@example.org", "rate_limit_rpm": 200})
    with caplog.at_level(logging.WARNING, logger="bibr.clients.crossref"):
        client = mod.CrossrefClient(settings=settings)

    assert client._anonymous_clamped_from is None
    assert client.interval == pytest.approx(0.3)
    assert "WITHOUT mailto" not in caplog.text
