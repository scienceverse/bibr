"""Tests for centralized FileState freeing policy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.state import FileState


def _make_full_fs() -> FileState:
    fs = FileState(path=Path("/tmp/test.pdf"))
    fs.pdf_bytes = b"x"
    fs.page_images = [MagicMock()]
    fs.page_indices = [0]
    fs.layout_results = [[{"label": "text"}]]
    fs.ocr_regions = [[{"label": "text"}]]
    fs._native_parser = MagicMock()
    return fs


def _make_ctx(files):
    return PipelineContext(
        file_states=files,
        progress=MagicMock(),
        resources=MagicMock(),
        config=RunConfig(),
    )


def test_free_after_ocr_clears_pre_ocr_data():
    fs = _make_full_fs()
    ctx = _make_ctx([fs])
    ctx.free_after_stage("ocr")  # frees pre-OCR data after OCR consumes it
    assert fs.pdf_bytes is None
    assert fs.page_images is None
    assert fs.layout_results is None
    assert fs.ocr_regions is not None  # still needed for parse


def test_free_after_parse_clears_pre_parse_data():
    fs = _make_full_fs()
    ctx = _make_ctx([fs])
    ctx.free_after_stage("parse")
    assert fs.ocr_regions is None
    assert fs.page_indices is None
    assert fs._native_parser is None


def test_free_after_unknown_stage_is_noop():
    fs = _make_full_fs()
    ctx = _make_ctx([fs])
    ctx.free_after_stage("layout")  # nothing to free yet
    assert fs.pdf_bytes is not None  # untouched


def test_free_after_stage_includes_errored_files():
    """Errored files retain memory until processed; freeing must still
    apply so a mid-batch failure doesn't pin all subsequent file memory."""
    good = _make_full_fs()
    bad = _make_full_fs()
    bad.set_error("dummy")
    ctx = _make_ctx([good, bad])
    ctx.free_after_stage("ocr")
    assert good.pdf_bytes is None
    assert bad.pdf_bytes is None


def test_errored_file_fully_freed_at_any_stage_boundary():
    """A file that errors mid-chunk must not pin its large buffers until GC —
    free_all applies at every boundary, even ones with no matching free rule."""
    fs = _make_full_fs()
    fs.contents = MagicMock()
    fs.paper = MagicMock()
    fs.set_error("boom")
    ctx = _make_ctx([fs])
    ctx.free_after_stage("layout")
    assert fs.pdf_bytes is None
    assert fs.page_images is None
    assert fs.layout_results is None
    assert fs.ocr_regions is None
    assert fs.contents is None
    assert fs.paper is None


def test_errored_file_with_exported_result_is_not_force_freed():
    fs = _make_full_fs()
    fs.result_json = {"ok": True}
    fs.set_error("post-export issue")
    ctx = _make_ctx([fs])
    ctx.free_after_stage("layout")
    assert fs.pdf_bytes is not None


def test_alive_file_untouched_at_unmatched_boundary():
    fs = _make_full_fs()
    ctx = _make_ctx([fs])
    ctx.free_after_stage("layout")
    assert fs.pdf_bytes is not None
    assert fs.ocr_regions is not None
