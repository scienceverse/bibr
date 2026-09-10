"""BibrPipelineAPI bounds how many requests run the pipeline concurrently.

The LitServe async loop dispatches an unbounded number of concurrent
``predict`` coroutines per worker; without a cap, a flood of uploads would
all render page images at once and exhaust host RAM. A per-worker semaphore
gates entry into the expensive pipeline run.
"""

import asyncio

from bibr.pipeline.context import RunConfig
from bibr.serve.deployments.pipeline import BibrPipelineAPI


class _FakePipeline:
    def __init__(self):
        self._config = RunConfig()
        self.inflight = 0
        self.max_inflight = 0

    async def process_file(self, filename, paper_id, content, config):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.05)  # hold the pipeline "busy" so overlap shows
        self.inflight -= 1
        return {"ok": True, "id": paper_id}


def _inputs(i: int) -> dict:
    return {
        "filename": f"f{i}.pdf",
        "content": str(i).encode(),
        "start_page": None,
        "end_page": None,
        "include_figures": False,
        "include_regions": False,
        "consolidate": None,
    }


async def test_predict_caps_concurrent_pipeline_runs(tmp_path):
    api = BibrPipelineAPI(upload_root=tmp_path)
    api._cache = None
    api._cache_inited = True
    pipe = _FakePipeline()
    api._pipeline = pipe
    api._inflight_sem = asyncio.Semaphore(2)

    results = await asyncio.gather(*(api.predict(_inputs(i)) for i in range(6)))

    assert pipe.max_inflight == 2, f"ran {pipe.max_inflight} pipelines at once (cap was 2)"
    assert all(r["success"] for r in results)


async def test_predict_unbounded_when_no_semaphore(tmp_path):
    """A None gate (operator set the cap to 0 = unlimited) must not block."""
    api = BibrPipelineAPI(upload_root=tmp_path)
    api._cache = None
    api._cache_inited = True
    pipe = _FakePipeline()
    api._pipeline = pipe
    api._inflight_sem = None

    await asyncio.gather(*(api.predict(_inputs(i)) for i in range(4)))

    assert pipe.max_inflight == 4
