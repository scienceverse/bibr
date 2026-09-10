"""Response-cache misses are coalesced per key inside a serve worker."""

import asyncio

from bibr.extract.ref_extractor import _resolve_ref_strategies
from bibr.pipeline.context import RunConfig
from bibr.serve.deployments.pipeline import BibrPipelineAPI


class _MemoryCache:
    def __init__(self):
        self.values: dict[str, bytes] = {}
        self.set_calls = 0
        self.delete_calls: list[str] = []

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes) -> None:
        self.set_calls += 1
        self.values[key] = value

    async def delete(self, key: str) -> None:
        self.delete_calls.append(key)
        self.values.pop(key, None)


class _CountingPipeline:
    def __init__(self):
        self._config = RunConfig()
        self.calls = 0
        self.inflight = 0
        self.max_inflight = 0

    async def process_file(self, filename, paper_id, content, config):  # noqa: ARG002
        self.calls += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.05)
        self.inflight -= 1
        return {"paper_id": paper_id, "info": {"file_name": filename}}


def _api(tmp_path) -> tuple[BibrPipelineAPI, _CountingPipeline, _MemoryCache]:
    api = BibrPipelineAPI(upload_root=tmp_path)
    pipeline = _CountingPipeline()
    cache = _MemoryCache()
    api._pipeline = pipeline
    api._cache = cache
    api._cache_inited = True
    api._inflight_sem = None
    return api, pipeline, cache


def _inputs(content: bytes = b"same-pdf") -> dict:
    return {
        "filename": "paper.pdf",
        "content": content,
        "start_page": None,
        "end_page": None,
        "include_figures": False,
        "include_regions": False,
        "consolidate": None,
    }


async def test_identical_cache_misses_run_pipeline_once(tmp_path):
    api, pipeline, cache = _api(tmp_path)

    results = await asyncio.gather(*(api.predict(_inputs()) for _ in range(6)))

    assert pipeline.calls == 1
    assert cache.set_calls == 1
    assert all(result["success"] for result in results)
    assert len({result["paper_json"]["paper_id"] for result in results}) == 1


async def test_different_cache_keys_are_not_serialized(tmp_path):
    api, pipeline, _ = _api(tmp_path)

    await asyncio.gather(api.predict(_inputs(b"first")), api.predict(_inputs(b"second")))

    assert pipeline.calls == 2
    assert pipeline.max_inflight == 2


async def test_cache_hit_rebinds_filename_to_current_request(tmp_path):
    api, pipeline, _ = _api(tmp_path)
    first = _inputs()
    first["filename"] = "alice-secret.pdf"
    second = _inputs()
    second["filename"] = "paper.pdf"

    assert (await api.predict(first))["paper_json"]["info"]["file_name"] == "alice-secret.pdf"
    assert (await api.predict(second))["paper_json"]["info"]["file_name"] == "paper.pdf"
    assert pipeline.calls == 1


async def test_corrupt_cached_json_is_deleted_and_recomputed(tmp_path):
    api, pipeline, cache = _api(tmp_path)
    inputs = _inputs()
    digest = __import__("hashlib").sha256(inputs["content"]).hexdigest()[:16]
    ref_seg, refs = _resolve_ref_strategies(None, None)
    key = api._cache_key(digest, None, None, False, False, None, refs=refs, ref_seg=ref_seg)
    cache.values[key] = b"{broken-json"

    result = await api.predict(inputs)

    assert result["success"] is True
    assert pipeline.calls == 1
    assert cache.delete_calls == [key]
