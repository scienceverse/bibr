"""A stalled Redis degrades to a cache miss; it never wedges the request.

Redis that accepts the TCP connection but never answers (fsync stall,
blackholed route) used to hang every extract while it held its admission
slot, because the cache read, the single-flight lease and the cache write all
ran outside any timeout. Each touch is now bounded by
``CACHE_OPERATION_TIMEOUT_SECONDS``.
"""

import asyncio

from bibr.config import GlobalSettings
from bibr.pipeline.context import RunConfig
from bibr.serve.deployments.pipeline import BibrPipelineAPI


class _HangingCache:
    """Accepts every call and never answers."""

    def __init__(self):
        self.calls: list[str] = []

    async def get(self, key):  # noqa: ARG002
        self.calls.append("get")
        await asyncio.sleep(3600)

    async def set(self, key, value):  # noqa: ARG002
        self.calls.append("set")
        await asyncio.sleep(3600)

    async def delete(self, key):  # noqa: ARG002
        self.calls.append("delete")
        await asyncio.sleep(3600)

    async def try_acquire_lease(self, key, *, ttl_seconds):  # noqa: ARG002
        self.calls.append("lease")
        await asyncio.sleep(3600)


class _Pipeline:
    def __init__(self):
        self._config = RunConfig()
        self.calls = 0

    async def process_file(self, filename, paper_id, content, config):  # noqa: ARG002
        self.calls += 1
        return {"paper_id": paper_id}


def _api(tmp_path, cache, settings):
    api = BibrPipelineAPI(upload_root=tmp_path, settings=settings)
    api._cache = cache
    api._cache_inited = True
    api._pipeline = _Pipeline()
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


async def test_hanging_cache_degrades_to_a_normal_extraction(tmp_path):
    settings = GlobalSettings()
    settings.cache.distributed_singleflight = True
    settings.cache.operation_timeout_seconds = 0.05
    cache = _HangingCache()
    api = _api(tmp_path, cache, settings)

    result = await asyncio.wait_for(api.predict(_inputs()), timeout=5)

    assert result["success"] is True
    assert api._pipeline.calls == 1
    # Every touch was attempted, bounded, and skipped: two reads (before and
    # inside the per-key flight), the lease, then the write.
    assert cache.calls == ["get", "get", "lease", "set"]


def test_cache_operation_timeout_has_a_sane_default():
    assert GlobalSettings().cache.operation_timeout_seconds == 5.0
