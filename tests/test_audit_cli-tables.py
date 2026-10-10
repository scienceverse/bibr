"""Audit fixes: ``bibr tables`` on ``bibr chew -o`` output, ``chew`` batch mode,
output checks and Rich markup, and ``chew()`` argument checks."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from rich.console import Console


def _export(paper_id: str) -> dict:
    from bibr.export.json_export import _export_paper_payload
    from tests.export.conftest import _demo_paper

    data = _export_paper_payload(_demo_paper(with_refs=False))
    data["paper_id"] = paper_id
    return data


class _FakePipeline:
    """``LocalPipeline`` stand-in: every file succeeds unless its name says fail."""

    def __init__(self, **_kwargs):
        pass

    async def process_chunk(self, file_states, progress=None):
        for fs in file_states:
            if "fail" in fs.path.name:
                fs.set_error("boom [/oops] [type=missing, input_value={}]", code="x")
            else:
                fs.result_json = {"info": {"title": fs.path.name}}

    async def aclose(self):
        pass

    def llm_usage_snapshot(self):
        return {}


class _ForbiddenPipeline:
    """For runs a guard must stop before any pipeline is built."""

    def __init__(self, **_kwargs):
        raise AssertionError("the guard must stop the run before LocalPipeline")


def _xml(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<article/>", encoding="utf-8")
    return path


# --- bibr tables on bibr chew -o output ---------------------------------------


def _chew_output(results: Path, paper_id: str, *, disposition=None) -> None:
    """What ``bibr chew <paper> -o results/`` leaves on disk for *paper_id*."""
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import EnrichmentSidecar, RunState

    sink = LocalArtifactSink(results / f"{paper_id}.json", json_kwargs={"indent": 2})
    fs = SimpleNamespace(
        artifact_disposition=disposition, path=Path(f"{paper_id}.pdf"), core_sha256=None
    )
    sink.record(fs, RunState.STARTED)
    payload = _export(paper_id)
    fs.core_sha256 = sink.write_core(fs, payload)
    sink.write_enrichment(
        fs,
        EnrichmentSidecar(
            schema_version="1",
            core_sha256=fs.core_sha256,
            settings_digest="d",
            completeness="complete",
        ),
    )
    sink.materialize(fs, payload)
    sink.record(fs, RunState.ENRICHMENT_COMPLETE)


def test_tables_reads_a_chew_output_directory(tmp_path, capsys):
    from bibr.local.cli.tables import run_tables
    from bibr.pipeline.artifacts import ArtifactDisposition

    results = tmp_path / "results"
    _chew_output(results, "a")
    _chew_output(results, "b")
    # An earlier run of a quarantined it; the quarantined export repeats its id.
    _chew_output(results, "a", disposition=ArtifactDisposition.BLOCKED)
    assert (results / "_quarantine" / "blocked" / "a.json").is_file()
    assert {p.name for p in results.iterdir()} >= {
        "a.json",
        "a.core.json",
        "a.json.receipt.json",
        "a.json.enrichment.json",
    }

    code = run_tables(SimpleNamespace(inputs=[str(results)], out=str(tmp_path / "tables")))

    assert code == 0
    papers = pq.read_table(tmp_path / "tables" / "paper.parquet").to_pylist()
    assert sorted(row["paper_id"] for row in papers) == ["a", "b"]
    assert "skipped" not in capsys.readouterr().err


def test_tables_reads_shell_expanded_chew_output(tmp_path):
    """``bibr tables results/*.json`` names every sidecar beside its export."""
    from bibr.local.cli.tables import run_tables

    results = tmp_path / "results"
    _chew_output(results, "a")
    _chew_output(results, "b")

    code = run_tables(
        SimpleNamespace(
            inputs=[str(p) for p in sorted(results.glob("*.json"))], out=str(tmp_path / "tables")
        )
    )

    assert code == 0
    papers = pq.read_table(tmp_path / "tables" / "paper.parquet").to_pylist()
    assert sorted(row["paper_id"] for row in papers) == ["a", "b"]


def test_only_chew_sidecars_are_left_out(tmp_path):
    """bibr batch writes <paper_id>.json and no receipts: its papers x and
    x.core, or one named like a sidecar, are all exports."""
    from bibr.local.cli.tables import export_files

    for name in ("x.json", "x.core.json", "foo.receipt.json", "bar.enrichment.json"):
        (tmp_path / name).write_text("{}")
    # chew output: y.json's receipt marks y.core.json as its core.
    for name in ("y.json", "y.core.json", "y.json.receipt.json", "y.json.enrichment.json"):
        (tmp_path / name).write_text("{}")

    files, missing = export_files([str(tmp_path)])

    names = ["bar.enrichment.json", "foo.receipt.json", "x.core.json", "x.json", "y.json"]
    assert [p.name for p in files] == names
    assert missing == []
    # A sidecar named on its own is read; write_tables then skips what is no export.
    files, _ = export_files([str(tmp_path / "y.core.json"), str(tmp_path / "y.json.receipt.json")])
    assert [p.name for p in files] == ["y.core.json", "y.json.receipt.json"]


def test_json_that_is_not_a_current_export_is_skipped_not_fatal(tmp_path):
    from bibr.export.tables import write_tables

    paths = {
        "a.json": _export("a"),
        "a.json.receipt.json": {"schema_version": "1", "events": []},
        "old.json": {**_export("old"), "schema_version": "11.0"},
        "no_id.json": {k: v for k, v in _export("x").items() if k != "paper_id"},
        "null_id.json": {**_export("x"), "paper_id": None},
        "numeric.json": {"schema_version": 12},
    }
    for name, data in paths.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")

    report = write_tables(sorted(tmp_path.glob("*.json")), tmp_path / "tables")

    assert report.papers == 1
    reasons = {Path(source).name: reason for source, reason in report.skipped}
    assert set(reasons) == {
        "a.json.receipt.json",
        "old.json",
        "no_id.json",
        "null_id.json",
        "numeric.json",
    }
    assert "schema_version '1'" in reasons["a.json.receipt.json"]
    assert "schema_version '11.0'" in reasons["old.json"]
    assert "no paper_id" in reasons["no_id.json"]
    assert "no paper_id" in reasons["null_id.json"]


def test_tables_report_keeps_bracketed_names():
    from bibr.local.cli.tables import report_tables

    report = SimpleNamespace(
        skipped=(("Smith [draft].json", "not a bibr export [/x]"),),
        rows={"paper": 0},
        papers=0,
        files={},
        out_dir=Path("out [b]"),
    )
    console = Console(file=io.StringIO(), width=300)

    report_tables(console, report)

    out = console.file.getvalue()
    assert "skipped Smith [draft].json: not a bibr export [/x]" in out
    assert "out [b]/" in out


# --- chew batch output names --------------------------------------------------


def test_core_sidecar_of_one_input_is_the_export_of_another(tmp_path):
    from bibr.local.cli.inputs import _find_stem_collisions

    x, x_core = tmp_path / "x.pdf", tmp_path / "x.core.pdf"
    a, b = tmp_path / "a" / "y.pdf", tmp_path / "b" / "Y.xml"

    assert _find_stem_collisions([x, x_core], sidecars=True) == {"x.core.json": [x, x_core]}
    # A shared stem shares every sidecar name too; the export names the group.
    assert _find_stem_collisions([a, b], sidecars=True) == {"y.json": [a, b]}
    assert _find_stem_collisions([x, a], sidecars=True) == {}
    # Without -o no sidecar is written: only the export names count.
    assert _find_stem_collisions([x, x_core]) == {}
    assert _find_stem_collisions([a, b]) == {"y.json": [a, b]}


async def test_chew_refuses_inputs_whose_outputs_collide(tmp_path, monkeypatch, capsys):
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _ForbiddenPipeline)
    x = _xml(tmp_path / "x.xml")
    x_core = _xml(tmp_path / "x.core.xml")
    args = _build_parser().parse_args(
        ["chew", str(x), str(x_core), "-o", str(tmp_path / "out"), "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "x.core.json would be written by:" in err
    assert str(x) in err and str(x_core) in err


async def test_stdout_batch_of_x_and_x_core_is_not_a_collision(tmp_path, monkeypatch, capsys):
    """Without -o nothing is written to disk, so no sidecar can collide."""
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)
    papers = tmp_path / "papers"
    _xml(papers / "x.xml")
    _xml(papers / "x.core.xml")

    try:
        await _run_process(_build_parser().parse_args(["chew", str(papers), "--no-llm"]))
    except SystemExit as exc:
        assert exc.code in (0, None)

    captured = capsys.readouterr()
    assert "collision" not in captured.err
    assert '"x.core.xml"' in captured.out and '"x.xml"' in captured.out


# --- chew batch mode follows the kind of input ---------------------------------


async def test_a_directory_with_one_paper_writes_a_directory(tmp_path, monkeypatch):
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)
    papers = tmp_path / "papers"
    _xml(papers / "only.xml")
    results = tmp_path / "results"

    await _run_process(
        _build_parser().parse_args(["chew", str(papers), "-o", str(results), "--no-llm"])
    )

    assert results.is_dir()
    assert json.loads((results / "only.json").read_text()) == {"info": {"title": "only.xml"}}


async def test_a_glob_matching_one_paper_is_a_batch(tmp_path, monkeypatch, capsys):
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _ForbiddenPipeline)
    _xml(tmp_path / "only.xml")
    args = _build_parser().parse_args(
        ["chew", str(tmp_path / "*.xml"), "--paper-id", "p", "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)

    assert exc_info.value.code == 2
    assert "--paper-id is only valid for single-file input" in capsys.readouterr().err


def test_one_named_file_or_stdin_is_not_a_batch(tmp_path):
    from bibr.local.cli.process import _names_a_batch

    paper = _xml(tmp_path / "p.xml")

    assert not _names_a_batch([str(paper)])
    assert not _names_a_batch(["-"])
    assert _names_a_batch([str(tmp_path)])
    assert _names_a_batch([str(tmp_path / "*.xml")])
    assert _names_a_batch([str(paper), str(paper)])


# --- --dry-run checks -o without creating it ----------------------------------


async def test_dry_run_blocks_a_batch_output_that_is_a_file(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    _xml(tmp_path / "papers" / "a.xml")
    _xml(tmp_path / "papers" / "b.xml")
    blocked = tmp_path / "results [b]"
    blocked.write_text("not a directory")
    args = _build_parser().parse_args(
        ["chew", str(tmp_path / "papers"), "-o", str(blocked), "--no-llm", "--dry-run"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)

    assert exc_info.value.code == 1
    out = " ".join(capsys.readouterr().out.split())
    assert f"{blocked} is not a directory" in out
    assert blocked.read_text() == "not a directory"


def test_output_problem_checks_without_creating(tmp_path):
    from bibr.local.cli.inputs import _output_path_problem

    blocker = tmp_path / "file"
    blocker.write_text("")
    new_dir = tmp_path / "new" / "dir"

    assert _output_path_problem(None, is_batch=True) is None
    assert _output_path_problem(str(new_dir), is_batch=True) is None
    assert _output_path_problem(str(new_dir / "out.json"), is_batch=False) is None
    assert _output_path_problem(str(blocker / "out.json"), is_batch=False) is not None
    assert _output_path_problem(str(blocker), is_batch=True) is not None
    assert _output_path_problem(str(blocker), is_batch=False) is None  # overwritten
    assert not (tmp_path / "new").exists()


# --- Rich markup in chew status lines ------------------------------------------


async def test_dry_run_prints_a_bracketed_ocr_model_literally(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    paper = _xml(tmp_path / "x.xml")
    args = _build_parser().parse_args(
        [
            "chew",
            str(paper),
            "--no-llm",
            "--dry-run",
            "--ocr",
            "glm-llama",
            "--ocr-model",
            "m[/x]",
            "--ocr-profile",
            "glm",
        ]
    )

    await _run_process(args)

    assert "OCR model: m[/x]" in capsys.readouterr().out


def test_unknown_preset_settings_are_printed_literally(monkeypatch, capsys):
    from argparse import Namespace
    from unittest.mock import patch

    from bibr.local.cli.run_config import _apply_runtime_settings

    monkeypatch.setattr(
        "bibr.presets.PresetManager.apply_to_settings", lambda self, name, settings: {"a[/x]"}
    )
    args = Namespace(
        preset="p", llm_provider=None, llm_model=None, no_equations=False, refs=None, ref_seg=None
    )

    with patch("bibr.config.Settings"):
        _apply_runtime_settings(args)

    assert "(skipped): a[/x]" in capsys.readouterr().err


def test_chunk_results_print_names_and_errors_literally(tmp_path):
    from bibr.local.cli.process import _write_chunk_results
    from bibr.pipeline.state import FileState

    failed = FileState(path=tmp_path / "Smith [draft].pdf")
    failed.set_error(
        "1 validation error\n  Field required [type=missing, input_value={}, input_type=dict]"
    )
    closing = FileState(path=tmp_path / "b.pdf")
    closing.set_error("bad tag [/x] in reply")
    done = FileState(path=tmp_path / "Jones [b].pdf")
    done.result_json = {"info": {}}
    console = Console(file=io.StringIO(), width=300)

    processed, errors = _write_chunk_results(
        [failed, closing, done],
        output_path=tmp_path / "out",
        json_kwargs={},
        console=console,
        is_batch=True,
        total_files=3,
        total_t0=0,
    )

    assert (processed, errors) == (1, 2)
    out = console.file.getvalue()
    assert "✗ Smith [draft].pdf:" in out
    assert "[type=missing, input_value={}, input_type=dict]" in out
    assert "bad tag [/x] in reply" in out
    assert f"✓ Jones [b].pdf → {tmp_path / 'out' / 'Jones [b].json'}" in out


def test_ocr_fallback_reason_is_printed_literally():
    from bibr.local.cli.process import _print_actual_ocr_summary

    resources = SimpleNamespace(
        ocr_runtime_identity=SimpleNamespace(backend="b", model="m [q4]", profile="p"),
        ocr_fallback_reason="RuntimeError: [/gpu] not found",
    )
    console = Console(file=io.StringIO(), width=300)

    _print_actual_ocr_summary(console, resources)

    out = console.file.getvalue()
    assert "OCR model: m [q4]" in out
    assert "Fallback reason: RuntimeError: [/gpu] not found" in out


async def test_a_bracketed_error_does_not_abort_the_run(tmp_path, monkeypatch, capsys):
    """A ``[/...]`` in one file's error used to raise MarkupError out of the
    result loop, before later chunks ran."""
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)
    papers = tmp_path / "papers"
    for name in ("a fail.xml", "b [draft].xml", "c.xml"):
        _xml(papers / name)
    results = tmp_path / "results"
    args = _build_parser().parse_args(
        ["chew", str(papers), "-o", str(results), "--no-llm", "--batch-size", "1"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)

    assert exc_info.value.code == 1  # the one failed file
    assert (results / "b [draft].json").is_file()
    assert (results / "c.json").is_file()
    err = capsys.readouterr().err
    assert "b [draft].xml" in err
    assert "boom [/oops] [type=missing, input_value={}]" in err


# --- chew() argument checks ------------------------------------------------------


@pytest.mark.parametrize("bound", ["start_page", "end_page"])
def test_pages_and_a_page_index_are_not_both_accepted(bound):
    import bibr
    from bibr.api import _pipeline_kwargs

    with pytest.raises(TypeError, match="not both"):
        _pipeline_kwargs({"pages": "1-3", bound: 0})
    with pytest.raises(TypeError, match="not both"):
        bibr.Chewer(pages="1-3", **{bound: 0})
    # An unset index neither conflicts with pages nor overrides it.
    expected = {"start_page": 1, "end_page": 2}
    assert _pipeline_kwargs({"pages": "2-3", bound: None}) == expected
    assert _pipeline_kwargs({bound: None, "pages": "2-3"}) == expected


def test_empty_directory_error_lists_every_supported_format(tmp_path):
    from bibr.api import _collect_batch
    from bibr.input.supported_files import SupportedFileType

    (tmp_path / "notes.txt").write_text("x")

    with pytest.raises(ValueError, match="no supported files") as exc_info:
        _collect_batch(tmp_path)

    for file_type in SupportedFileType:
        assert file_type.value in str(exc_info.value)
