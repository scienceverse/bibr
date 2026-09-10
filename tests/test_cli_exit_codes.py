"""``bibr chew`` must exit non-zero when any file fails.

Regression tests for two silent-failure bugs in ``bibr.local.cli``:

1. ``_run_process`` counted per-file failures into ``errors`` but never
   turned that into a process exit code — ``bibr chew`` exited 0 even when
   every file failed.
2. ``_collect_files`` silently dropped a missing/unmatched path when mixed
   with at least one valid file — no message, no error accounting.

The pipeline itself (OCR/LLM/layout models) is mocked out — these tests
exercise the CLI's own error-accounting and exit-code wiring, not the real
extraction pipeline.
"""

from __future__ import annotations

import pytest


class _FakePipeline:
    """Stand-in for ``LocalPipeline`` — construction is a no-op, and
    ``process_chunk`` is driven per-test to simulate success or failure."""

    def __init__(self, **_kwargs):
        pass

    async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
        raise NotImplementedError  # overridden per test

    async def aclose(self):
        pass

    def llm_usage_snapshot(self):
        return {}


def _disable_ocr_runtime_preflight(monkeypatch):
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)


async def test_run_process_single_corrupt_file_exits_1(tmp_path, monkeypatch):
    """A single failing file (e.g. corrupt/empty input) must exit 1, not 0."""
    from bibr.local.cli import _build_parser, _run_process

    class _FailingPipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):
            for fs in file_states:
                fs.error = "Unsupported format or corrupt file"

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FailingPipeline)

    bad_file = tmp_path / "corrupt.pdf"
    bad_file.write_bytes(b"")  # empty — not a real PDF

    args = _build_parser().parse_args(["chew", str(bad_file), "--no-llm"])

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 1


async def test_run_process_all_succeed_exits_cleanly(tmp_path, monkeypatch):
    """Sanity check: the happy path must NOT raise SystemExit (no false positive)."""
    from bibr.local.cli import _build_parser, _run_process

    class _SucceedingPipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):
            for fs in file_states:
                fs.result_json = {"info": {"title": fs.path.name}}

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _SucceedingPipeline)

    good_file = tmp_path / "paper.pdf"
    good_file.write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(["chew", str(good_file), "--no-llm"])

    # Must return normally — no SystemExit at all.
    await _run_process(args)


async def test_run_process_missing_input_among_valid_still_exits_1(tmp_path, monkeypatch):
    """A missing/unmatched input mixed in with valid files must still fail
    the overall exit code, even though every *resolved* file processes fine.

    Exercises the ``_collect_files`` -> ``missing_count`` -> ``errors`` wiring
    directly, since driving a real missing-path CLI invocation end-to-end
    would require a live pipeline.
    """
    from bibr.local.cli import _build_parser, _run_process

    class _SucceedingPipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):
            for fs in file_states:
                fs.result_json = {"info": {"title": fs.path.name}}

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _SucceedingPipeline)

    good_file = tmp_path / "paper.pdf"
    good_file.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("bibr.local.cli._collect_files", lambda _inputs: ([good_file], 1))

    args = _build_parser().parse_args(["chew", str(good_file), "--no-llm"])

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 1
