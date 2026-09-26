"""ExportStage's throttled gc.collect() must run off the event loop.

gc.collect() holds the GIL for the whole sweep; on the event loop that stalls
every concurrent serve request. It must run in a thread executor instead, while
keeping the existing once-per-window throttle.
"""

import gc
import threading
import time

from bibr.pipeline.stages.export import ExportStage


async def test_maybe_gc_collect_runs_off_event_loop(monkeypatch):
    stage = ExportStage()
    stage._last_gc_time = -1e9  # far in the past → not throttled

    main_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def fake_collect(*_a, **_k):
        seen["tid"] = threading.get_ident()
        return 0

    monkeypatch.setattr(gc, "collect", fake_collect)
    await stage._maybe_gc_collect()

    assert "tid" in seen, "gc.collect should have run when not throttled"
    assert seen["tid"] != main_thread, "gc.collect must run off the event loop thread"


async def test_maybe_gc_collect_respects_throttle(monkeypatch):
    stage = ExportStage()
    stage._last_gc_time = time.monotonic()  # just collected → inside the window

    calls = {"n": 0}
    monkeypatch.setattr(gc, "collect", lambda *_a, **_k: calls.__setitem__("n", calls["n"] + 1))

    await stage._maybe_gc_collect()
    assert calls["n"] == 0, "must not collect again within the throttle window"
