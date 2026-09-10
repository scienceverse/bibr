import asyncio


async def test_recorder_keeps_concurrent_request_metrics_separate():
    from bibr.metrics.performance import PerformanceRecorder

    recorder = PerformanceRecorder(enabled=True)

    async def observe(request_id: str, batch_size: int):
        async with recorder.request(request_id):
            with recorder.stage("validate"):
                await asyncio.sleep(0)
            recorder.record_model_call("paper_classifier", batch_size)

    await asyncio.gather(observe("a", 2), observe("b", 3))

    a = recorder.snapshot("a")
    b = recorder.snapshot("b")
    assert a.stages["validate"].calls == 1
    assert b.stages["validate"].calls == 1
    assert a.model_calls["paper_classifier"].calls == 1
    assert a.model_calls["paper_classifier"].items == 2
    assert b.model_calls["paper_classifier"].items == 3


async def test_recorder_accumulates_nested_stage_and_model_observations():
    from bibr.metrics.performance import PerformanceRecorder

    recorder = PerformanceRecorder(enabled=True)
    async with recorder.request("req"):
        with recorder.stage("post_parse"):
            with recorder.stage("classification"):
                recorder.record_model_call("section_classifier", 4)
                recorder.record_model_call("section_classifier", 2)

    snapshot = recorder.snapshot("req")
    assert snapshot.stages["post_parse"].calls == 1
    assert snapshot.stages["classification"].calls == 1
    assert snapshot.model_calls["section_classifier"].calls == 2
    assert snapshot.model_calls["section_classifier"].items == 6
    assert snapshot.stages["post_parse"].total_seconds >= 0


async def test_disabled_recorder_is_a_noop():
    from bibr.metrics.performance import PerformanceRecorder

    recorder = PerformanceRecorder(enabled=False)
    async with recorder.request("req"):
        with recorder.stage("validate"):
            pass
        recorder.record_model_call("paper_classifier", 7)

    snapshot = recorder.snapshot("req")
    assert snapshot.stages == {}
    assert snapshot.model_calls == {}
    assert snapshot.peak_rss_bytes is None
    assert snapshot.peak_vram_bytes is None
