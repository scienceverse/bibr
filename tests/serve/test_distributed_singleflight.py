"""Distributed cache single-flight remains bounded and fail-open."""

import asyncio

import pytest

from bibr.pipeline.context import RunConfig
from bibr.serve.deployments.pipeline import BibrPipelineAPI


class _Lease:
    def __init__(self, cache):
        self.cache = cache
        self.released = False
        self.renew_calls = 0

    async def renew(self):
        self.renew_calls += 1
        return True

    async def release(self):
        self.released = True
        self.cache.lease_held = False
        return True


class _SharedCache:
    def __init__(self):
        self.values = {}
        self.lease_held = False
        self.leases = []

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value):
        self.values[key] = value

    async def lease_alive(self, key):  # noqa: ARG002
        return self.lease_held

    async def try_acquire_lease(self, key, *, ttl_seconds):  # noqa: ARG002
        if self.lease_held:
            return None
        self.lease_held = True
        lease = _Lease(self)
        self.leases.append(lease)
        return lease


class _Pipeline:
    def __init__(self, delay=0.04):
        self._config = RunConfig()
        self.delay = delay
        self.calls = 0

    async def process_file(self, filename, paper_id, content, config):  # noqa: ARG002
        self.calls += 1
        await asyncio.sleep(self.delay)
        return {"paper_id": paper_id}


def _api(tmp_path, cache, pipeline=None, *, settings=None):
    api = BibrPipelineAPI(upload_root=tmp_path, settings=settings)
    api._cache = cache
    api._cache_inited = True
    api._pipeline = pipeline or _Pipeline()
    api._inflight_sem = None
    return api


def _inputs():
    return {
        "filename": "paper.pdf",
        "content": b"same-pdf",
        "start_page": None,
        "end_page": None,
        "include_figures": False,
        "include_regions": False,
        "consolidate": None,
    }


async def test_separate_workers_share_one_extraction(monkeypatch, tmp_path):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.cache.distributed_singleflight = True
    settings.cache.singleflight_poll_interval_ms = 2
    settings.cache.singleflight_wait_seconds = 0.2
    cache = _SharedCache()
    first, second = (
        _api(tmp_path, cache, settings=settings),
        _api(tmp_path, cache, settings=settings),
    )

    results = await asyncio.gather(first.predict(_inputs()), second.predict(_inputs()))

    assert first._pipeline.calls + second._pipeline.calls == 1
    assert all(result["success"] for result in results)
    assert cache.leases[0].released is True


async def test_redis_lease_error_fails_open(monkeypatch, tmp_path):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.cache.distributed_singleflight = True
    cache = _SharedCache()

    async def fail(*args, **kwargs):
        raise ConnectionError("redis unavailable")

    cache.try_acquire_lease = fail
    api = _api(tmp_path, cache, settings=settings)

    result = await api.predict(_inputs())

    assert result["success"] is True
    assert api._pipeline.calls == 1


async def test_wait_timeout_falls_back_to_extraction(monkeypatch, tmp_path):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.cache.distributed_singleflight = True
    settings.cache.singleflight_poll_interval_ms = 1
    settings.cache.singleflight_wait_seconds = 0.005
    cache = _SharedCache()
    cache.lease_held = True
    api = _api(tmp_path, cache, _Pipeline(delay=0), settings=settings)

    result = await api.predict(_inputs())

    assert result["success"] is True
    assert api._pipeline.calls == 1


async def test_opt_out_does_not_touch_redis_lease(monkeypatch, tmp_path):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.cache.distributed_singleflight = False
    cache = _SharedCache()

    async def unexpected(*args, **kwargs):
        raise AssertionError("lease should not be attempted")

    cache.try_acquire_lease = unexpected
    api = _api(tmp_path, cache, _Pipeline(delay=0), settings=settings)

    assert (await api.predict(_inputs()))["success"] is True


@pytest.mark.slow
async def test_waiter_outlasts_the_default_wait_window(tmp_path):
    """A waiter coalesces an extraction slower than the old 10 s flat wait.

    No explicit ``singleflight_wait_seconds``: the budget follows
    ``PIPELINE_TIMEOUT``, so a ~10.5 s owner is still coalesced instead of
    extracted twice. Fails on the pre-fix default (wait 10 s, duplicate run).
    """
    import time

    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    assert "singleflight_wait_seconds" not in settings.cache.model_fields_set
    settings.cache.singleflight_poll_interval_ms = 50
    cache = _SharedCache()
    owner = _api(tmp_path, cache, _Pipeline(delay=10.5), settings=settings)
    waiter = _api(tmp_path, cache, _Pipeline(delay=10.5), settings=settings)

    owner_task = asyncio.create_task(owner.predict(_inputs()))
    await asyncio.sleep(0.2)  # let the owner take the lease first
    started = time.monotonic()
    waiter_result = await waiter.predict(_inputs())
    waiter_latency = time.monotonic() - started
    owner_result = await owner_task

    assert owner_result["success"] is True
    assert waiter_result["success"] is True
    assert owner._pipeline.calls + waiter._pipeline.calls == 1
    assert waiter_latency >= 10.0


async def test_waiter_takes_over_when_the_owner_dies(tmp_path):
    """A vanished lease ends the wait at once instead of burning the budget.

    The owner holds the lease for two polls then dies without publishing (its
    TTL expires); the waiter must fall back to its own extraction quickly,
    far inside the pipeline-timeout budget.
    """
    import time

    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.cache.singleflight_poll_interval_ms = 5
    cache = _SharedCache()
    cache.lease_held = True
    polls = {"n": 0}

    async def lease_alive(key):  # noqa: ARG001
        polls["n"] += 1
        alive = polls["n"] < 3
        if not alive:
            # A dead owner's lease key is gone (expired), so the takeover
            # acquisition succeeds exactly as against real Redis.
            cache.lease_held = False
        return alive

    cache.lease_alive = lease_alive
    api = _api(tmp_path, cache, _Pipeline(delay=0), settings=settings)

    started = time.monotonic()
    result = await api.predict(_inputs())
    elapsed = time.monotonic() - started

    assert result["success"] is True
    assert api._pipeline.calls == 1
    assert elapsed < 5.0


async def test_lease_check_error_fails_open(tmp_path):
    """A failing lease liveness check extracts normally (fail-open)."""
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.cache.singleflight_poll_interval_ms = 1
    cache = _SharedCache()
    cache.lease_held = True

    async def lease_alive(key):  # noqa: ARG001
        raise ConnectionError("redis unavailable")

    cache.lease_alive = lease_alive
    api = _api(tmp_path, cache, _Pipeline(delay=0), settings=settings)

    result = await api.predict(_inputs())

    assert result["success"] is True
    assert api._pipeline.calls == 1


async def test_waiter_re_reads_cache_when_the_lease_vanishes(tmp_path):
    """No duplicate extraction when the owner's publish lands between the
    waiter's cache read and its lease check.

    The owner finishes right after the waiter's first in-loop cache miss; the
    waiter must see the published result on its re-read instead of extracting.
    Fails without the re-read (waiter calls == 1).
    """
    import pathlib
    import tempfile

    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.cache.singleflight_poll_interval_ms = 5
    cache = _SharedCache()
    tmp = pathlib.Path(tempfile.mkdtemp())
    owner = _api(tmp, cache, _Pipeline(delay=0.2), settings=settings)
    waiter = _api(tmp, cache, _Pipeline(delay=0.2), settings=settings)

    owner_task = asyncio.create_task(owner.predict(_inputs()))
    await asyncio.sleep(0.05)
    real_get = cache.get
    state = {"n": 0}

    async def get(key):
        value = await real_get(key)
        state["n"] += 1
        if value is None and state["n"] == 3:
            # First in-loop poll (after the two pre-lease lookups): the owner
            # publishes and releases right after this miss.
            await owner_task
        return value

    cache.get = get
    await waiter.predict(_inputs())

    assert owner._pipeline.calls == 1
    assert waiter._pipeline.calls == 0


async def test_only_one_waiter_takes_over_when_the_owner_dies(tmp_path):
    """Two waiters on a dead owner's key: exactly one extracts, the other
    consumes the winner's published result.

    Fails without the takeover acquisition (both waiters extract).
    """
    import pathlib
    import tempfile

    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.cache.singleflight_poll_interval_ms = 5
    cache = _SharedCache()
    cache.lease_held = True
    tmp = pathlib.Path(tempfile.mkdtemp())

    async def lease_alive(key):  # noqa: ARG001
        return cache.lease_held

    cache.lease_alive = lease_alive

    async def die_soon():
        await asyncio.sleep(0.05)
        cache.lease_held = False

    first = _api(tmp, cache, _Pipeline(delay=0.05), settings=settings)
    second = _api(tmp, cache, _Pipeline(delay=0.05), settings=settings)
    results = await asyncio.gather(
        first.predict(_inputs()),
        second.predict(_inputs()),
        die_soon(),
    )

    assert all(result["success"] for result in results[:2])
    assert first._pipeline.calls + second._pipeline.calls == 1
