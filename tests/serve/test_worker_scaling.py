"""`bibr serve` runs exactly one inference worker, and it is not configurable.

One worker already serves many requests concurrently on its async loop, with the layout/segmenter `GpuBatcher` coalescing their GPU work into
single forward passes. A second worker would duplicate the models (~1.5 GB RSS
each), give each its own batcher so batches shrink as workers rise, and
parallelize only GIL-bound Python — torch/ONNX intra-op threads already use
every core from one process.
"""

from unittest import mock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from bibr.config import Settings  # noqa: E402


def test_build_server_pins_exactly_one_inference_worker(monkeypatch):
    import asyncio

    from bibr.serve import app as app_mod
    from bibr.serve import worker as worker_mod

    built = []
    real_worker = worker_mod.InferenceWorker

    def counting_worker(api, **kwargs):
        built.append(kwargs)
        return real_worker(api, **kwargs)

    monkeypatch.setattr(worker_mod, "InferenceWorker", counting_worker)
    server = app_mod.build_server()
    try:
        assert built == [
            {"timeout": Settings.pipeline.timeout, "restart": Settings.pipeline.restart_workers}
        ]
        assert server.app.state.inference_worker is server.worker
    finally:
        asyncio.run(server.app.state.inference_tracker.close())
        asyncio.run(server.app.state.upload_store.close())


def test_worker_count_is_not_a_setting():
    """A knob whose only correct value is 1 is not a knob. Left in an existing
    .env, PIPELINE_WORKERS_PER_DEVICE is ignored (settings use extra="ignore")
    rather than failing startup."""
    assert not hasattr(Settings.pipeline, "workers_per_device")
    assert "workers_per_device" not in Settings.pipeline.model_fields


def test_ocr_semaphore_is_the_configured_server_wide_cap(tmp_path, monkeypatch):
    """With one worker the OCR cap needs no per-worker division: this semaphore
    is the only gate between the process and a shared OCR endpoint."""
    from bibr.serve.deployments.pipeline import BibrPipelineAPI

    monkeypatch.setattr(Settings.ocr, "max_concurrent_regions", 16)
    api = BibrPipelineAPI(upload_root=tmp_path, settings=Settings)
    with (
        mock.patch("bibr.serve.deployments.layout.LayoutDetector"),
        mock.patch("bibr.serve.deployments.segmenter.SentenceSegmenter"),
        mock.patch("bibr.serve.pipeline.ServePipeline"),
    ):
        api.setup(device="cpu")

    assert api._ocr_sem._value == 16


def test_torch_thread_capping_is_gone():
    """cap_inference_threads existed only to divide cores among co-located
    workers; with one worker it set torch's own default and did nothing."""
    import bibr.utils.device as device_mod

    assert not hasattr(device_mod, "cap_inference_threads")
