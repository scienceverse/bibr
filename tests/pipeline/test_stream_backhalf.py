"""StreamingRenderOcrStage — cloud-LLM back-half streaming.

With a remote (cloud) LLM there is no OCR→LLM VRAM handoff, so each OCR
window's back half (parse → extract → enrich → export) can start as soon as
that window's OCR completes instead of waiting for the whole chunk to clear
each stage barrier. These tests use fake front-half/back-half stages and
event-based synchronization (no wall-clock sleeps) so ordering assertions
are deterministic under load.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.config import Settings
from bibr.local.pipeline import LocalPipeline
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.render_ocr import StreamingRenderOcrStage
from bibr.pipeline.state import FileState

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeLayout:
    name = "layout"

    async def run(self, ctx):
        for fs in ctx.alive():
            fs.page_images = [object()]
            fs.page_indices = [0]
            fs.layout_results = [[]]


class _FakeNative:
    name = "native_text"

    async def run(self, ctx):
        return


class _EventOcr:
    """Fake OcrStage: records events, optionally gating named files on an
    ``asyncio.Event`` so tests control cross-window interleaving."""

    name = "ocr"

    def __init__(self, events, gates=None):
        self._events = events
        self._gates = gates or {}

    async def run(self, ctx):
        for fs in ctx.alive():
            self._events.append(f"ocr_start:{fs.path.name}")
            gate = self._gates.get(fs.path.name)
            if gate is not None:
                await gate.wait()
            fs.ocr_regions = [[]]
            self._events.append(f"ocr_end:{fs.path.name}")


class _FakeParse:
    name = "parse"

    def __init__(self, events=None, hold=None):
        self._events = events
        self._hold = hold  # asyncio.Event: every run blocks on it (cap test)

    async def run(self, ctx):
        ctx.progress.stage_start(self.name)
        if self._events is not None:
            for fs in ctx.file_states:
                self._events.append(f"parse:{fs.path.name}")
        if self._hold is not None:
            await self._hold.wait()
        for fs in ctx.alive():
            if fs.contents is None:
                fs.contents = f"contents:{fs.path.name}"
        ctx.progress.stage_end(self.name)


class _FakeExtract:
    name = "extract"

    def __init__(self, events=None, raise_for=()):
        self._events = events
        self._raise_for = set(raise_for)

    async def run(self, ctx):
        ctx.progress.stage_start(self.name)
        for fs in ctx.alive():
            if fs.path.name in self._raise_for:
                raise RuntimeError(f"extract blew up on {fs.path.name}")
            fs.paper = f"paper:{fs.path.name}"
            if self._events is not None:
                self._events.append(f"extract:{fs.path.name}")
        ctx.progress.stage_end(self.name)


class _FakeEnrich:
    name = "enrich"

    async def run(self, ctx):
        ctx.progress.stage_start(self.name)
        ctx.progress.stage_end(self.name)


class _FakeCheckpoint:
    name = "core_checkpoint"

    async def run(self, ctx):
        ctx.progress.stage_start(self.name)
        ctx.progress.stage_end(self.name)


class _FakeExport:
    name = "export"

    def __init__(self, events=None, on_export=None):
        self._events = events
        self._on_export = on_export

    async def run(self, ctx):
        ctx.progress.stage_start(self.name)
        for fs in ctx.alive():
            fs.result_json = {"paper_id": fs.path.name}
            if self._events is not None:
                self._events.append(f"export:{fs.path.name}")
            if self._on_export is not None:
                self._on_export(fs)
            fs.free_all()
        ctx.progress.stage_end(self.name)


class _RecordingProgress:
    """ProgressTracker that records which stage_start/stage_end fired."""

    def __init__(self):
        self.starts: list[str] = []
        self.ends: list[str] = []

    def stage_start(self, name, detail=""):
        self.starts.append(name)

    def stage_end(self, name):
        self.ends.append(name)

    def ocr_start(self, total_regions):
        pass

    def ocr_region_done(self):
        pass

    def ocr_end(self):
        pass


def _mk(name: str) -> FileState:
    fs = FileState(path=Path(name))
    fs.pdf_bytes = b"%PDF-1.4"
    return fs


def _cfg(**overrides) -> RunConfig:
    defaults = {"ocr_backend": "glm-mlx", "llm_backend": "cloud", "memory_mode": "balanced"}
    defaults.update(overrides)
    return RunConfig(**defaults)


def _ctx(files, config, progress=None) -> PipelineContext:
    rm = MagicMock()
    rm.shutdown_ocr = AsyncMock(return_value=None)
    return PipelineContext(
        file_states=files,
        progress=progress if progress is not None else NullProgress(),
        resources=rm,
        config=config,
    )


def _stage(
    events=None,
    *,
    ocr=None,
    parse=None,
    post_parse=None,
    checkpoint=None,
    enrich=None,
    export=None,
):
    return StreamingRenderOcrStage(
        layout=_FakeLayout(),
        native_text=_FakeNative(),
        ocr=ocr or _EventOcr(events if events is not None else []),
        parse=parse or _FakeParse(events),
        post_parse=post_parse or _FakeExtract(events),
        checkpoint=checkpoint or _FakeCheckpoint(),
        enrich=enrich or _FakeEnrich(),
        export=export or _FakeExport(events),
    )


# ---------------------------------------------------------------------------
# Gating: which stage list does LocalPipeline build?
# ---------------------------------------------------------------------------


def _uses_streaming(pipe) -> bool:
    return any(isinstance(s, StreamingRenderOcrStage) for s in pipe._stages)


class TestGating:
    def test_cloud_non_aggressive_streams(self):
        pipe = LocalPipeline(llm_backend="cloud", memory_mode="balanced")
        assert isinstance(pipe._stages[-1], StreamingRenderOcrStage)
        names = [type(s).__name__ for s in pipe._stages]
        assert names.index("ClassifierStage") < names.index("StreamingRenderOcrStage")
        # The back-half stages moved inside the composite; no barrier stages
        # (and no LlmServerStage — cloud never starts a managed server).
        for barrier_stage in (
            "LlmServerStage",
            "ParseSegmentStage",
            "PostParseStage",
            "EnrichmentStage",
            "ExportStage",
        ):
            assert barrier_stage not in names

    def test_keep_all_streams_too(self):
        assert _uses_streaming(LocalPipeline(llm_backend="cloud", memory_mode="keep_all"))

    def test_managed_local_llm_keeps_barrier(self):
        pipe = LocalPipeline(llm_backend="vllm", memory_mode="balanced")
        assert not _uses_streaming(pipe)
        names = [type(s).__name__ for s in pipe._stages]
        assert "LlmServerStage" in names
        assert names[-1] == "ExportStage"

    def test_aggressive_memory_keeps_barrier(self):
        pipe = LocalPipeline(llm_backend="cloud", memory_mode="aggressive")
        assert not _uses_streaming(pipe)
        assert type(pipe._stages[-1]).__name__ == "ExportStage"

    def test_setting_off_keeps_barrier(self, monkeypatch):
        monkeypatch.setattr(Settings.pipeline, "stream_backhalf", False)
        pipe = LocalPipeline(llm_backend="cloud", memory_mode="balanced")
        assert not _uses_streaming(pipe)
        assert type(pipe._stages[-1]).__name__ == "ExportStage"

    def test_env_var_disables_streaming(self, monkeypatch):
        from bibr.config import GlobalSettings

        monkeypatch.setenv("PIPELINE_STREAM_BACKHALF", "false")
        custom = GlobalSettings()
        pipe = LocalPipeline(llm_backend="cloud", memory_mode="balanced", settings=custom)
        assert not _uses_streaming(pipe)

    def test_streaming_list_passes_contract_validation(self, monkeypatch):
        # LocalPipeline already runs validate_stage_contracts at construction;
        # spy to assert it saw the streaming shape (composite declares the
        # union of its inner stages' produces).
        import bibr.pipeline.pipeline as pipeline_mod

        seen = {}
        real = pipeline_mod.validate_stage_contracts

        def _spy(stages):
            seen["names"] = [s.name for s in stages]
            real(stages)

        monkeypatch.setattr(pipeline_mod, "validate_stage_contracts", _spy)
        LocalPipeline(llm_backend="cloud", memory_mode="balanced")
        assert seen["names"][0] == "validate"
        assert seen["names"][-1] == "render_ocr_stream"


# ---------------------------------------------------------------------------
# Streaming behavior
# ---------------------------------------------------------------------------


class TestStreamingOverlap:
    async def test_window1_exports_before_window2_ocr_finishes(self):
        # Window 2's OCR blocks on a gate that ONLY window 1's export opens:
        # if the back half streamed correctly the run completes, and window
        # 1's export event precedes window 2's ocr_end. Under the old barrier
        # behavior this would deadlock (caught by wait_for).
        events: list[str] = []
        gate = asyncio.Event()
        a, b = _mk("a.pdf"), _mk("b.pdf")
        stage = _stage(
            events,
            ocr=_EventOcr(events, gates={"b.pdf": gate}),
            export=_FakeExport(events, on_export=lambda fs: gate.set()),
        )

        await asyncio.wait_for(stage.run(_ctx([a, b], _cfg())), timeout=30)

        assert events.index("export:a.pdf") < events.index("ocr_end:b.pdf")
        assert a.result_json == {"paper_id": "a.pdf"}
        assert b.result_json == {"paper_id": "b.pdf"}

    async def test_memory_freed_per_window(self):
        events: list[str] = []
        a, b = _mk("a.pdf"), _mk("b.pdf")
        ocr_regions_at_extract: dict[str, object] = {}

        class _CapturingExtract(_FakeExtract):
            async def run(self, ctx):
                # Entry here is right after the parse-stage boundary free
                # (free_after_stage("parse") → free_pre_parse). Capture what
                # it left so the assertion below pins THIS boundary — the
                # final-state checks alone are satisfied by export's free_all.
                for fs in ctx.alive():
                    ocr_regions_at_extract[fs.path.name] = fs.ocr_regions
                await super().run(ctx)

        await _stage(events, post_parse=_CapturingExtract(events)).run(_ctx([a, b], _cfg()))

        assert ocr_regions_at_extract == {"a.pdf": None, "b.pdf": None}
        for fs in (a, b):
            # Front-half frees (free_pre_ocr) and back-half frees
            # (free_pre_parse after parse, free_all at export).
            assert fs.pdf_bytes is None
            assert fs.page_images is None
            assert fs.ocr_regions is None
            assert fs.contents is None
            assert fs.paper is None
            assert fs.result_json is not None

    async def test_error_in_one_window_isolated_from_others(self):
        events: list[str] = []
        a, b = _mk("a.pdf"), _mk("b.pdf")
        stage = _stage(events, post_parse=_FakeExtract(events, raise_for={"a.pdf"}))

        await stage.run(_ctx([a, b], _cfg()))

        assert a.error is not None
        assert "extract blew up" in a.error
        assert a.result_json is None
        # Errored file with no result is fully freed (free_after_stage semantics).
        assert a.contents is None and a.ocr_regions is None
        # B is a separate FileState in a separate window: fully unaffected.
        assert b.error is None
        assert b.result_json == {"paper_id": "b.pdf"}

    async def test_native_contents_bypass_streams_without_ocr(self):
        # DOCX/JATS/HTML-style file: fs.contents pre-set by its handling
        # stage. It must be back-half processed immediately (window 0) and
        # never rendered or OCR'd.
        fs = FileState(path=Path("doc.docx"))
        fs.contents = "native-contents"
        layout = MagicMock(run=AsyncMock())
        native = MagicMock(run=AsyncMock())
        ocr = MagicMock(run=AsyncMock())
        stage = StreamingRenderOcrStage(
            layout=layout,
            native_text=native,
            ocr=ocr,
            parse=_FakeParse(),
            post_parse=_FakeExtract(),
            enrich=_FakeEnrich(),
            export=_FakeExport(),
        )

        await stage.run(_ctx([fs], _cfg()))

        layout.run.assert_not_awaited()
        native.run.assert_not_awaited()
        ocr.run.assert_not_awaited()
        assert fs.error is None
        assert fs.result_json == {"paper_id": "doc.docx"}

    async def test_backhalf_windows_capped_at_two(self):
        # Three windows, back halves all blocked on one event: no more than
        # two may be inside the back half at once (stage-level pressure cap).
        # Pure event/yield synchronization — no wall-clock waits.
        events: list[str] = []
        hold = asyncio.Event()
        files = [_mk(f"{i}.pdf") for i in range(3)]
        stage = _stage(events, parse=_FakeParse(events, hold=hold))

        task = asyncio.create_task(stage.run(_ctx(files, _cfg())))
        # Let the loop drain: OCR for all three windows is instant, so every
        # back-half task is spawned; only two can enter parse.
        for _ in range(50):
            await asyncio.sleep(0)
        parses = [e for e in events if e.startswith("parse:")]
        assert len(parses) == 2, f"expected 2 concurrent back-half windows, saw {parses}"

        hold.set()
        await asyncio.wait_for(task, timeout=30)
        parses = [e for e in events if e.startswith("parse:")]
        assert len(parses) == 3
        assert all(fs.result_json is not None for fs in files)

    async def test_backhalf_task_never_outlives_run_on_front_half_crash(self):
        # A front-half crash mid-loop must still settle already-spawned
        # back-half tasks before run() re-raises.
        events: list[str] = []
        started = asyncio.Event()

        class _SignalingParse(_FakeParse):
            async def run(self, ctx):
                started.set()
                await super().run(ctx)

        class _CrashingOcr:
            name = "ocr"

            async def run(self, ctx):
                for fs in ctx.alive():
                    if fs.path.name == "b.pdf":
                        # Suspend until window 1's back half has actually
                        # entered parse, then crash: the settle path below is
                        # exercised with a genuinely in-flight back-half task.
                        await started.wait()
                        raise RuntimeError("engine died")
                    fs.ocr_regions = [[]]

        a, b = _mk("a.pdf"), _mk("b.pdf")
        stage = _stage(
            events,
            ocr=_CrashingOcr(),
            parse=_SignalingParse(events),
            export=_FakeExport(events, on_export=lambda fs: None),
        )

        with pytest.raises(RuntimeError, match="engine died"):
            await asyncio.wait_for(stage.run(_ctx([a, b], _cfg())), timeout=30)

        # Window 1's back half completed (its task was gathered, not leaked).
        assert a.result_json == {"paper_id": "a.pdf"}

    async def test_outer_cancellation_cancels_and_settles_hung_backhalf(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class _HungParse(_FakeParse):
            async def run(self, ctx):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        fs = _mk("hung.pdf")
        task = asyncio.create_task(_stage([], parse=_HungParse()).run(_ctx([fs], _cfg())))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()

        for _ in range(10):
            await asyncio.sleep(0)
            if cancelled.is_set():
                break
        assert cancelled.is_set(), "child cancellation must not wait for the outer timeout"

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)

    async def test_front_half_failure_cancels_hung_backhalf_without_leak(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class _HungParse(_FakeParse):
            async def run(self, ctx):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        class _CrashAfterBackhalfStarts:
            name = "ocr"

            async def run(self, ctx):
                for fs in ctx.alive():
                    if fs.path.name == "b.pdf":
                        await started.wait()
                        raise RuntimeError("front failed")
                    fs.ocr_regions = [[]]

        a, b = _mk("a.pdf"), _mk("b.pdf")
        stage = _stage(ocr=_CrashAfterBackhalfStarts(), parse=_HungParse())

        with pytest.raises(RuntimeError, match="front failed"):
            await asyncio.wait_for(stage.run(_ctx([a, b], _cfg())), timeout=1)
        assert cancelled.is_set()


# ---------------------------------------------------------------------------
# Back-half progress reporting (single vs multi window)
# ---------------------------------------------------------------------------


class TestBackhalfProgress:
    async def test_single_window_reports_real_stage_progress(self):
        # One pending file → one back-half group → nothing to interleave with,
        # so the back half reports real per-stage progress. This is the
        # dominant single-file ``bibr chew paper.pdf`` case, which must not go
        # silent through the LLM+Crossref tail.
        prog = _RecordingProgress()
        a = _mk("a.pdf")

        await _stage([]).run(_ctx([a], _cfg(), progress=prog))

        assert prog.starts == ["parse", "extract", "core_checkpoint", "enrich", "export"]
        assert prog.ends == ["parse", "extract", "core_checkpoint", "enrich", "export"]
        assert a.result_json == {"paper_id": "a.pdf"}

    async def test_single_bypass_group_reports_real_progress(self):
        # A lone native/bypass file (DOCX/JATS: contents pre-set) is also a
        # single group — real progress even though nothing is rendered/OCR'd.
        prog = _RecordingProgress()
        fs = FileState(path=Path("doc.docx"))
        fs.contents = "native-contents"

        await _stage([]).run(_ctx([fs], _cfg(), progress=prog))

        assert prog.starts == ["parse", "extract", "core_checkpoint", "enrich", "export"]
        assert fs.result_json == {"paper_id": "doc.docx"}

    async def test_multi_window_suppresses_backhalf_progress(self):
        # Two files → two overlapping back-half groups → NullProgress, so no
        # per-stage back-half lines can interleave in the terminal (the real
        # ctx.progress records nothing from the back half).
        prog = _RecordingProgress()
        a, b = _mk("a.pdf"), _mk("b.pdf")

        await _stage([]).run(_ctx([a, b], _cfg(), progress=prog))

        assert prog.starts == []
        assert prog.ends == []
        assert a.result_json == {"paper_id": "a.pdf"}
        assert b.result_json == {"paper_id": "b.pdf"}
