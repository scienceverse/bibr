"""``extraction.timings.memory`` (schema 12.3): opt-in, additive, omitted when off."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from bibr.export import PaperExport, PaperExportReader
from bibr.export.json_export import export_paper_to_json, validate_export
from bibr.pipeline.memory_profile import StageMemory
from bibr.pipeline.state import FileState
from bibr.utils.process_memory import MIB
from tests.export.conftest import extraction_block

_MEMORY = {
    "peak_rss_mib": 2210.5,
    "stages": {
        "render_ocr": {
            "peak_rss_mib": 2210.5,
            "rss_mib": 1650.0,
            "rss_delta_mib": 420.3,
            "swap_mib": 0.0,
            "files": 8,
            "overlapped": False,
            "peak_reset": True,
        },
        "extract": {
            "peak_rss_mib": 1702.1,
            "rss_mib": 1690.4,
            "rss_delta_mib": -12.0,
            "swap_mib": None,
            "files": 8,
            "overlapped": True,
            "peak_reset": True,
            "python_peak_mib": 88.2,
        },
    },
}


def _timings(memory=None) -> dict:
    timings = {"stages": {"render_ocr": 30.0, "extract": 12.5}, "total_seconds": 42.5}
    if memory is not None:
        timings["memory"] = memory
    return timings


def test_memory_block_validates_and_round_trips(demo_paper):
    demo_paper.extraction = extraction_block(timings=_timings(_MEMORY))

    payload = export_paper_to_json(demo_paper)

    assert payload["schema_version"] == "12.3"
    assert payload["extraction"]["timings"]["memory"] == _MEMORY
    assert validate_export(payload) == []


def test_memory_is_omitted_when_profiling_was_off(demo_paper):
    """Off is the default, and the timings block stays exactly what 12.2 wrote."""
    demo_paper.extraction = extraction_block(timings=_timings())

    timings = export_paper_to_json(demo_paper)["extraction"]["timings"]

    assert set(timings) == {"stages", "total_seconds"}


def test_python_peak_is_omitted_without_tracemalloc(demo_paper):
    demo_paper.extraction = extraction_block(timings=_timings(_MEMORY))

    stages = export_paper_to_json(demo_paper)["extraction"]["timings"]["memory"]["stages"]

    assert "python_peak_mib" not in stages["render_ocr"]
    assert stages["extract"]["python_peak_mib"] == 88.2


def test_strict_model_rejects_unknown_memory_keys_and_the_reader_keeps_them(demo_paper):
    demo_paper.extraction = extraction_block(timings=_timings(_MEMORY))
    payload = export_paper_to_json(demo_paper)
    payload["extraction"]["timings"]["memory"]["stages"]["extract"]["vram_mib"] = 512.0

    with pytest.raises(ValidationError):
        PaperExport.model_validate(payload)
    reader = PaperExportReader.model_validate(payload)
    assert reader.extraction.timings.memory.stages["extract"].model_extra == {"vram_mib": 512.0}


def test_a_12_2_export_is_still_reader_input(demo_paper):
    demo_paper.extraction = extraction_block(timings=_timings())
    payload = export_paper_to_json(demo_paper)
    payload["schema_version"] = "12.2"

    assert PaperExportReader.model_validate(payload).schema_version == "12.2"


def _ctx(settings=None):
    from bibr.config import snapshot_settings
    from bibr.pipeline.context import RunConfig

    return types.SimpleNamespace(
        config=RunConfig(),
        settings=settings if settings is not None else snapshot_settings(),
        scratch={},
    )


def test_build_extraction_exports_the_file_own_records(demo_paper):
    from bibr.pipeline.stages.export import _build_extraction

    fs = FileState(path=Path("paper.pdf"))
    fs.stage_times = {"layout": 3.0, "parse": 1.0}
    fs.stage_memory = {
        "layout": StageMemory(900 * MIB, 600 * MIB, 200 * MIB, 0, 4, False, True),
        "parse": StageMemory(650 * MIB, 610 * MIB, 10 * MIB, 0, 4, False, True),
    }

    timings = _build_extraction(_ctx(), demo_paper, fs)["timings"]

    assert timings["memory"]["peak_rss_mib"] == 900.0
    assert timings["memory"]["stages"]["parse"]["rss_mib"] == 610.0
    demo_paper.extraction = extraction_block(timings=timings)
    assert validate_export(export_paper_to_json(demo_paper)) == []


def test_build_extraction_has_no_memory_key_when_nothing_was_recorded(demo_paper):
    from bibr.pipeline.stages.export import _build_extraction

    fs = FileState(path=Path("paper.pdf"))
    fs.stage_times = {"layout": 3.0}

    assert "memory" not in _build_extraction(_ctx(), demo_paper, fs)["timings"]
