"""Opt-in per-stage memory telemetry (``PIPELINE_MEMORY_PROFILE``).

The fold-and-reset logic is driven by a scripted fake process (exact numbers);
the /proc readings and the pipeline wiring run for real on Linux.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.config import GlobalSettings
from bibr.pipeline import memory_profile
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.memory_profile import StageMemory, measure_stage, memory_export
from bibr.pipeline.pipeline import Pipeline
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState
from bibr.utils import process_memory
from bibr.utils.process_memory import MIB, ProcessMemory

linux_only = pytest.mark.skipif(
    not Path("/proc/self/status").exists(), reason="reads /proc/self/status"
)


def _settings(*, on: bool = True, tracemalloc_on: bool = False) -> GlobalSettings:
    return GlobalSettings(
        pipeline={"memory_profile": on, "memory_profile_tracemalloc": tracemalloc_on}
    )


class _FakeProcess:
    """A process whose RSS the test sets; VmHWM follows it and resets like Linux."""

    def __init__(self, rss: int = 100, *, resettable: bool = True) -> None:
        self.rss = rss
        self.hwm = rss
        self.swap = 7
        self.resettable = resettable
        self.resets = 0

    def set(self, rss: int) -> None:
        self.rss = rss
        self.hwm = max(self.hwm, rss)

    def read(self) -> ProcessMemory:
        return ProcessMemory(
            rss=self.rss, peak=self.hwm, swap=self.swap, peak_resettable=self.resettable
        )

    def reset(self) -> bool:
        if not self.resettable:
            return False
        self.resets += 1
        self.hwm = self.rss
        return True


@pytest.fixture
def proc(monkeypatch):
    fake = _FakeProcess()
    monkeypatch.setattr(memory_profile, "read_process_memory", fake.read)
    monkeypatch.setattr(memory_profile, "reset_peak_rss", fake.reset)
    return fake


def _ctx(files, settings=None):
    return SimpleNamespace(
        file_states=files,
        alive=lambda: [fs for fs in files if fs.error is None],
        settings=settings if settings is not None else _settings(),
        scratch={},
    )


def _files(n: int = 1) -> list[FileState]:
    return [FileState(path=Path(f"{i}.pdf")) for i in range(n)]


# --- the measurement ----------------------------------------------------------


def test_off_by_default_reads_and_records_nothing(monkeypatch):
    def boom():
        raise AssertionError("read while profiling is off")

    monkeypatch.setattr(memory_profile, "read_process_memory", boom)
    files = _files()
    ctx = _ctx(files, settings=GlobalSettings())

    with measure_stage(ctx, "layout"):
        pass

    assert GlobalSettings().pipeline.memory_profile is False
    assert files[0].stage_memory == {}
    assert ctx.scratch == {}


def test_a_magicmock_settings_object_does_not_switch_profiling_on():
    """Only a real ``True`` turns it on; a test double's truthy attribute does not."""
    assert memory_profile.profiling_enabled(MagicMock()) is False


def test_stage_records_peak_end_rss_delta_and_swap(proc):
    files = _files(2)
    ctx = _ctx(files)

    with measure_stage(ctx, "layout"):
        proc.set(300)
        proc.set(150)

    record = files[0].stage_memory["layout"]
    assert record == StageMemory(
        peak_rss=300,
        rss=150,
        rss_delta=50,
        swap=7,
        files=2,
        overlapped=False,
        peak_reset=True,
    )
    assert files[1].stage_memory["layout"] is record
    assert ctx.scratch["stage_memory"]["layout"] is record
    assert memory_profile._active == {}


def test_peak_is_the_stage_own_not_an_earlier_stage(proc):
    files = _files()
    ctx = _ctx(files)

    with measure_stage(ctx, "layout"):
        proc.set(900)
        proc.set(200)
    with measure_stage(ctx, "parse"):
        proc.set(250)

    assert files[0].stage_memory["layout"].peak_rss == 900
    assert files[0].stage_memory["parse"].peak_rss == 250


def test_nested_stage_resets_without_losing_the_outer_peak(proc):
    """The fused render/OCR stage measures its layout and OCR passes inside its
    own record: the inner reset folds the outer peak first."""
    files = _files()
    ctx = _ctx(files)

    with measure_stage(ctx, "render_ocr"):
        proc.set(500)
        proc.set(200)
        with measure_stage(ctx, "layout"):
            proc.set(260)
            proc.set(180)
        with measure_stage(ctx, "ocr"):
            proc.set(240)

    memory = files[0].stage_memory
    assert memory["layout"].peak_rss == 260
    assert memory["ocr"].peak_rss == 240
    assert memory["render_ocr"].peak_rss == 500
    assert not any(record.overlapped for record in memory.values())


async def test_concurrent_stages_are_flagged_and_keep_both_peaks(proc):
    """Two stages running side by side (streamed back halves, serve requests)
    share the process: both are flagged, and neither peak is lost to the
    other's reset."""
    first, second = _files(), _files()
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_first():
        with measure_stage(_ctx(first), "parse"):
            proc.set(800)
            proc.set(300)
            started.set()
            await release.wait()

    async def run_second():
        await started.wait()
        with measure_stage(_ctx(second), "extract"):
            proc.set(400)
            release.set()
            await asyncio.sleep(0)

    await asyncio.gather(run_first(), run_second())

    parse = first[0].stage_memory["parse"]
    extract = second[0].stage_memory["extract"]
    assert parse.overlapped and extract.overlapped
    assert parse.peak_rss == 800
    assert extract.peak_rss == 400


async def test_tasks_spawned_inside_a_stage_are_part_of_it(proc):
    """A back half spawned inside the streaming stage copies its context: the
    outer stage is its parent, not a concurrent neighbour."""
    files = _files()
    ctx = _ctx(files)

    async def back_half():
        with measure_stage(ctx, "parse"):
            proc.set(350)

    with measure_stage(ctx, "render_ocr_stream"):
        await asyncio.create_task(back_half())

    assert files[0].stage_memory["parse"].overlapped is False
    assert files[0].stage_memory["render_ocr_stream"].overlapped is False
    assert files[0].stage_memory["render_ocr_stream"].peak_rss == 350


def test_unresettable_peak_is_reported_as_lifetime(proc):
    proc.resettable = False
    proc.set(900)
    proc.set(100)
    files = _files()

    with measure_stage(_ctx(files), "layout"):
        proc.set(200)

    record = files[0].stage_memory["layout"]
    assert record.peak_reset is False
    assert record.peak_rss == 900  # the process's peak so far, not the stage's


def test_a_failing_stage_is_still_recorded_and_closed(proc):
    files = _files()

    with pytest.raises(RuntimeError), measure_stage(_ctx(files), "extract"):
        proc.set(400)
        raise RuntimeError("boom")

    assert files[0].stage_memory["extract"].peak_rss == 400
    assert memory_profile._active == {}


def test_only_files_alive_at_the_start_get_the_record(proc):
    files = _files(3)
    files[2].error = "failed earlier"

    with measure_stage(_ctx(files), "parse"):
        pass

    assert files[0].stage_memory["parse"].files == 2
    assert files[2].stage_memory == {}


def test_a_stage_recorded_twice_merges():
    first = StageMemory(100, 80, 10, 1, 4, False, True)
    later = StageMemory(90, 85, 5, 2, 2, True, False, python_peak=7)

    merged = first.merge(later)

    assert merged == StageMemory(
        peak_rss=100,
        rss=85,
        rss_delta=15,
        swap=2,
        files=4,
        overlapped=True,
        peak_reset=False,
        python_peak=7,
    )


def test_python_peak_needs_its_own_switch(proc):
    was_tracing = tracemalloc.is_tracing()
    files = _files()
    try:
        with measure_stage(_ctx(files, settings=_settings(tracemalloc_on=True)), "extract"):
            buffer = bytearray(4 * MIB)
            del buffer
        with measure_stage(_ctx(files), "export"):
            pass
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert files[0].stage_memory["extract"].python_peak >= 4 * MIB
    assert files[0].stage_memory["export"].python_peak is None


# --- the export block ---------------------------------------------------------


def test_memory_export_is_in_mib_and_names_the_highest_peak():
    records = {
        "layout": StageMemory(3 * MIB, 2 * MIB, MIB // 2, 0, 4, False, True),
        "parse": StageMemory(5 * MIB, MIB, -MIB, None, 4, True, True, python_peak=MIB),
    }

    block = memory_export(records)

    assert block == {
        "peak_rss_mib": 5.0,
        "stages": {
            "layout": {
                "peak_rss_mib": 3.0,
                "rss_mib": 2.0,
                "rss_delta_mib": 0.5,
                "swap_mib": 0.0,
                "files": 4,
                "overlapped": False,
                "peak_reset": True,
            },
            "parse": {
                "peak_rss_mib": 5.0,
                "rss_mib": 1.0,
                "rss_delta_mib": -1.0,
                "swap_mib": None,
                "files": 4,
                "overlapped": True,
                "peak_reset": True,
                "python_peak_mib": 1.0,
            },
        },
    }
    assert memory_export({}) is None
    assert memory_export(None) is None


# --- the pipeline -------------------------------------------------------------


class _Stage:
    def __init__(self, name: str, grow: int = 0):
        self.name = name
        self._grow = grow

    async def run(self, ctx):
        if self._grow:
            buffer = b"x" * self._grow
            del buffer


def _pipeline(stages, settings) -> Pipeline:
    return Pipeline(stages=stages, resources=MagicMock(), config=RunConfig(), settings=settings)


@linux_only
async def test_pipeline_records_every_stage_per_file_when_on():
    files = _files(2)

    await _pipeline(
        [_Stage("validate"), _Stage("layout", grow=32 * MIB)], _settings()
    ).process_chunk(files)

    for fs in files:
        assert set(fs.stage_memory) == {"validate", "layout"}
        layout = fs.stage_memory["layout"]
        assert layout.files == 2
        assert layout.rss is not None and layout.rss > 0
        assert layout.peak_rss >= layout.rss
        if layout.peak_reset:
            # The 32 MiB buffer lived inside the stage only.
            assert layout.peak_rss - layout.rss >= 16 * MIB


async def test_pipeline_records_nothing_when_off():
    files = _files()

    await _pipeline([_Stage("validate")], GlobalSettings()).process_chunk(files)

    assert files[0].stage_memory == {}


async def test_fused_render_ocr_measures_its_inner_passes(proc):
    from bibr.pipeline.stages.render_ocr import InterleavedRenderOcrStage

    files = _files(2)
    for fs in files:
        fs.pdf_bytes = b"%PDF-1.4"

    class _Pass:
        def __init__(self, name, rss):
            self.name = name
            self._rss = rss

        async def run(self, ctx):
            proc.set(self._rss)
            if self.name == "ocr":
                for fs in ctx.alive():
                    fs.ocr_regions = [[]]
            proc.set(100)

    resources = MagicMock()
    resources.shutdown_ocr = AsyncMock(return_value=None)
    ctx = PipelineContext(
        file_states=files,
        progress=NullProgress(),
        resources=resources,
        config=RunConfig(ocr_backend="glm-mlx", memory_mode="balanced", llm_backend="cloud"),
        settings=_settings(),
    )
    stage = InterleavedRenderOcrStage(
        layout=_Pass("layout", 700), native_text=_Pass("native_text", 150), ocr=_Pass("ocr", 400)
    )

    from bibr.pipeline.pipeline import run_stage

    await run_stage(ctx, stage)

    for fs in files:
        memory = fs.stage_memory
        assert set(memory) == {"render_ocr", "layout", "native_text", "ocr"}
        assert memory["layout"].peak_rss == 700
        assert memory["ocr"].peak_rss == 400
        assert memory["layout"].files == 1  # local OCR: one file per window
        assert memory["render_ocr"].peak_rss == 700
        assert memory["render_ocr"].files == 2


# --- process readings ---------------------------------------------------------

_STATUS = """Name:\tpython
VmPeak:\t 2000000 kB
VmHWM:\t  812345 kB
VmRSS:\t  700000 kB
VmSwap:\t    1024 kB
Threads:\t8
"""


def test_parse_status_reads_rss_peak_and_swap_in_bytes():
    reading = process_memory.parse_status(_STATUS)

    assert reading == ProcessMemory(
        rss=700000 * 1024, peak=812345 * 1024, swap=1024 * 1024, peak_resettable=True
    )


def test_parse_status_without_the_fields_reports_none():
    assert process_memory.parse_status("Name:\tx\n") == ProcessMemory(None, None, None, False)


def test_without_proc_only_the_lifetime_peak_is_known(monkeypatch, tmp_path):
    monkeypatch.setattr(process_memory, "_STATUS_PATH", tmp_path / "missing")

    reading = process_memory.read_process_memory()

    assert reading.rss is None and reading.swap is None
    assert reading.peak_resettable is False
    if sys.platform != "win32":
        assert reading.peak > 0


def test_reset_reports_failure_instead_of_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(process_memory, "_CLEAR_REFS_PATH", tmp_path / "no" / "clear_refs")

    assert process_memory.reset_peak_rss() is False


@linux_only
def test_reset_brings_the_peak_back_down_to_the_current_rss():
    buffer = b"x" * (64 * MIB)
    del buffer
    before = process_memory.read_process_memory()
    if not process_memory.reset_peak_rss():
        pytest.skip("this kernel or sandbox refuses /proc/self/clear_refs")

    after = process_memory.read_process_memory()

    assert before.peak - before.rss >= 32 * MIB
    assert after.peak - after.rss < 16 * MIB


def test_to_mib_rounds_to_one_decimal():
    assert process_memory.to_mib(None) is None
    assert process_memory.to_mib(3 * MIB + MIB // 4) == 3.2


def test_model_load_delta_is_logged(monkeypatch, caplog):
    readings = iter([ProcessMemory(100 * MIB, None, None), ProcessMemory(400 * MIB, None, None)])
    monkeypatch.setattr(process_memory, "read_process_memory", lambda: next(readings))

    with caplog.at_level(logging.INFO, logger="bibr.utils.process_memory"):
        with process_memory.log_rss_delta("NER parser", verbose=True):
            pass

    assert "NER parser loaded: RSS +300 MiB (now 400 MiB" in caplog.text


def test_model_load_delta_is_debug_unless_profiling(monkeypatch, caplog):
    readings = iter([ProcessMemory(100 * MIB, None, None), ProcessMemory(150 * MIB, None, None)])
    monkeypatch.setattr(process_memory, "read_process_memory", lambda: next(readings))

    with caplog.at_level(logging.INFO, logger="bibr.utils.process_memory"):
        with process_memory.log_rss_delta("layout model"):
            pass

    assert caplog.text == ""


def test_a_failed_model_load_logs_nothing(monkeypatch, caplog):
    readings = iter([ProcessMemory(100 * MIB, None, None)])
    monkeypatch.setattr(process_memory, "read_process_memory", lambda: next(readings))

    with caplog.at_level(logging.DEBUG, logger="bibr.utils.process_memory"):
        with pytest.raises(OSError), process_memory.log_rss_delta("layout model", verbose=True):
            raise OSError("no bundle")

    assert "loaded" not in caplog.text


# --- model loads --------------------------------------------------------------


@linux_only
def test_layout_and_segmenter_loads_log_the_rss_they_add(caplog):
    from bibr.pipeline.resources import ResourceManager

    def heavy_model(**_kwargs):
        return SimpleNamespace(loaded=True, weights=b"x" * (8 * MIB))

    resources = ResourceManager(
        settings=_settings(), layout_factory=heavy_model, segmenter_factory=heavy_model
    )
    with caplog.at_level(logging.INFO, logger="bibr.utils.process_memory"):
        resources.ensure_layout()
        resources.ensure_segmenter()
        resources.ensure_layout()  # already loaded: nothing to log

    assert caplog.text.count("layout model loaded: RSS +") == 1
    assert "sentence segmenter loaded: RSS +" in caplog.text


@linux_only
def test_classifier_load_logs_under_the_inference_lock(caplog):
    from bibr.pipeline.classifier_resources import _locked_load

    with caplog.at_level(logging.INFO, logger="bibr.utils.process_memory"):
        model = _locked_load(
            lambda model_id, revision, device: (model_id, revision, device),
            "org/paper",
            "sha",
            "cpu",
            label="paper classifier",
            verbose=True,
        )

    assert model == ("org/paper", "sha", "cpu")
    assert "paper classifier loaded: RSS" in caplog.text


@linux_only
def test_ner_parser_load_logs_its_rss(monkeypatch, caplog):
    import bibr.extract.ref_extractor as ex
    import bibr.ner.runtime as runtime

    monkeypatch.setattr(ex, "_NER_PARSER", None)
    monkeypatch.setattr(ex, "_NER_PARSER_KEY", None)
    monkeypatch.setattr(runtime, "load_ref_parser", lambda *args, **kwargs: object())

    with caplog.at_level(logging.INFO, logger="bibr.utils.process_memory"):
        ex._get_ner_parser(_settings())

    assert "NER parser loaded: RSS" in caplog.text
