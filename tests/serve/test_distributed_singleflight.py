"""Distributed cache single-flight remains bounded and fail-open."""

import asyncio

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
