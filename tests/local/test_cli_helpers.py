"""Unit tests for the extracted CLI helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---- _apply_runtime_settings -----------------------------------------------


def test_apply_runtime_settings_applies_llm_provider():
    from bibr.local.cli import _apply_runtime_settings

    args = Namespace(
        preset=None,
        llm_provider="anthropic",
        llm_model=None,
        no_equations=False,
        refs=None,
        ref_seg=None,
    )
    with patch("bibr.config.Settings") as mock_settings:
        mock_settings.llm = MagicMock()
        _apply_runtime_settings(args)
        assert mock_settings.llm.provider == "anthropic"


def test_apply_runtime_settings_disables_equations():
    from bibr.local.cli import _apply_runtime_settings

    args = Namespace(
        preset=None,
        llm_provider=None,
        llm_model=None,
        no_equations=True,
        refs=None,
        ref_seg=None,
    )
    with patch("bibr.config.Settings") as mock_settings:
        _apply_runtime_settings(args)
        assert mock_settings.EQUATION_EXTRACTION is False


def test_apply_runtime_settings_unknown_preset_exits():
    from bibr.local.cli import _apply_runtime_settings

    args = Namespace(
        preset="nonexistent",
        llm_provider=None,
        llm_model=None,
        no_equations=False,
        refs=None,
        ref_seg=None,
    )
    with pytest.raises(SystemExit):
        _apply_runtime_settings(args)


# ---- _write_chunk_results -------------------------------------------------


def test_format_run_summary_shows_backend_model_and_profile():
    from bibr.local.cli.process import _format_run_summary
    from bibr.local.cli.run_config import ResolvedRunConfig

    summary = _format_run_summary(
        ResolvedRunConfig(
            ocr_backend="paddle-http",
            ocr_model="paddle-ocr-vl-1.6",
            ocr_profile="paddle",
            memory_mode="balanced",
            llm_backend="cloud",
        )
    )

    assert "OCR backend:[/dim] paddle-http" in summary
    assert "OCR model:[/dim] paddle-ocr-vl-1.6" in summary
    assert "OCR profile:[/dim] paddle" in summary


@pytest.mark.asyncio
async def test_run_process_reports_actual_ocr_identity_and_fallback(tmp_path, monkeypatch, capsys):
    """Runtime identity is emitted from the selected candidate, not the planned selector."""
    from bibr.local.cli import _build_parser, _run_process

    class _Pipeline:
        def __init__(self, **_kwargs):
            self._resources = SimpleNamespace(
                ocr_runtime_identity=SimpleNamespace(
                    backend="glm-rapid-mlx",
                    model="mlx-community/GLM-OCR-8bit",
                    profile="glm",
                ),
                ocr_fallback_reason="paddle-rapid-mlx: startup smoke failed",
            )

        async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
            file_states[0].result_json = {"info": {"title": "fixture"}}

        def llm_usage_snapshot(self):
            return {}

        async def aclose(self):
            return None

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _Pipeline)
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(["chew", str(pdf), "--no-llm"])
    await _run_process(args)

    err = capsys.readouterr().err
    assert "OCR backend: glm-rapid-mlx" in err
    assert "OCR model: mlx-community/GLM-OCR-8bit" in err
    assert "OCR profile: glm" in err
    assert "Fallback reason: paddle-rapid-mlx: startup smoke failed" in err


def test_write_chunk_results_writes_per_file_in_batch_mode(tmp_path):
    from bibr.local.cli import _write_chunk_results
    from bibr.pipeline.state import FileState

    fs = FileState(path=tmp_path / "sample.pdf")
    fs.result_json = {"info": {"title": "x"}}
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    console = MagicMock()

    processed, errors = _write_chunk_results(
        [fs],
        output_path=out_dir,
        json_kwargs={"indent": 2},
        console=console,
        is_batch=True,
        total_files=1,
        total_t0=0,
    )
    assert processed == 1
    assert errors == 0
    assert (out_dir / "sample.json").exists()
    assert json.loads((out_dir / "sample.json").read_text())["info"]["title"] == "x"


def test_write_chunk_results_reports_errors_with_hints():
    from bibr.local.cli import _write_chunk_results
    from bibr.pipeline.state import FileState

    fs = FileState(path=Path("/tmp/x.pdf"))
    fs.error = "OCR failed: backend down"
    console = MagicMock()

    processed, errors = _write_chunk_results(
        [fs],
        output_path=None,
        json_kwargs={"indent": 2},
        console=console,
        is_batch=False,
        total_files=1,
        total_t0=0,
    )
    assert processed == 0
    assert errors == 1
    printed = " ".join(str(c) for c in console.print.call_args_list)
    assert "ocr failed" in printed.lower()
    assert "bibr doctor" in printed.lower()


def test_write_chunk_results_writes_to_stdout_when_no_output(capsys):
    from bibr.local.cli import _write_chunk_results
    from bibr.pipeline.state import FileState

    fs = FileState(path=Path("/tmp/x.pdf"))
    fs.result_json = {"info": {"title": "stdout-test"}}
    console = MagicMock()

    processed, errors = _write_chunk_results(
        [fs],
        output_path=None,
        json_kwargs={"indent": 2},
        console=console,
        is_batch=False,
        total_files=1,
        total_t0=0,
    )
    captured = capsys.readouterr()
    assert "stdout-test" in captured.out
    assert processed == 1
    assert errors == 0


# ---- ChunkProcessor -------------------------------------------------------


async def test_chunk_processor_runs_pipeline_and_returns_states():
    from bibr.local.cli import ChunkProcessor

    pipeline = MagicMock()

    async def fake_chunk(states, **kw):
        for s in states:
            s.result_json = {"info": {"title": s.path.name}}

    pipeline.process_chunk = fake_chunk

    files = [Path("/tmp/a.pdf"), Path("/tmp/b.pdf")]
    cp = ChunkProcessor(pipeline=pipeline, paper_id=None, is_batch=True, active_stages=[])
    states = await cp.run(
        files,
        chunk_index=1,
        total_chunks=1,
        console=MagicMock(),
    )

    assert len(states) == 2
    assert all(s.result_json is not None for s in states)


async def test_chunk_processor_filters_mixed_batch_progress_for_each_chunk():
    from bibr.local.cli import ChunkProcessor, ResolvedRunConfig

    files = [Path("/tmp/a.pdf"), Path("/tmp/b.docx")]
    config = ResolvedRunConfig(ocr_backend="paddle", memory_mode="balanced", llm_backend="cloud")
    observed_stages = []

    async def fake_chunk(states, *, progress):
        observed_stages.append(progress._stages)
        for state in states:
            state.result_json = {"info": {"title": state.path.name}}

    pipeline = MagicMock()
    pipeline.process_chunk = fake_chunk
    processor = ChunkProcessor(
        pipeline=pipeline, paper_id=None, is_batch=True, active_stages=config.active_stages(files)
    )
    for index, path in enumerate(files, 1):
        await processor.run([path], chunk_index=index, total_chunks=2, console=MagicMock())

    assert observed_stages == [
        ["validate", "layout", "ocr", "parse", "extract", "enrich", "export"],
        ["validate", "docx", "parse", "extract", "enrich", "export"],
    ]


# ---- _parse_pages ------------------------------------------------------------


class TestParsePages:
    """Pages are 1-based on the CLI; 0/negative must fail loudly, not wrap
    to the last page via Python negative indexing."""

    def test_single_page(self):
        from bibr.local.cli import _parse_pages

        assert _parse_pages("3") == (2, 2)

    def test_range(self):
        from bibr.local.cli import _parse_pages

        assert _parse_pages("1-5") == (0, 4)

    def test_page_zero_rejected(self):
        from bibr.local.cli import _parse_pages

        with pytest.raises(ValueError, match="1-based"):
            _parse_pages("0")

    def test_range_starting_at_zero_rejected(self):
        from bibr.local.cli import _parse_pages

        with pytest.raises(ValueError, match="1-based"):
            _parse_pages("0-3")

    def test_reversed_range_rejected(self):
        from bibr.local.cli import _parse_pages

        with pytest.raises(ValueError, match="start"):
            _parse_pages("5-2")

    def test_junk_rejected(self):
        from bibr.local.cli import _parse_pages

        with pytest.raises(ValueError):
            _parse_pages("abc")
