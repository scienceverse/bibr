"""The command-line experience: what ``bibr`` prints, in what order, and what it refuses.

These pin the output redesign (usage errors, run preamble, output routing, interrupts,
preset prompts without a terminal) at the seams the rest of the CLI tests do not reach.
The pipeline is always a stand-in: no model loads and nothing touches the network.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import signal
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

posix_only = pytest.mark.skipif(
    os.name != "posix", reason="POSIX file modes, signals and exit codes"
)


class _Pipeline:
    """Stand-in for ``LocalPipeline``: records that it was built and what it was given."""

    built: list[dict] = []
    seen: list[list] = []

    def __init__(self, **kwargs):
        type(self).built.append(kwargs)

    async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
        type(self).seen.append([fs.path for fs in file_states])
        for fs in file_states:
            fs.result_json = {"info": {"title": fs.path.name}, "paper_id": fs.path.stem}

    async def aclose(self):
        pass

    def llm_usage_snapshot(self):
        return {}


@pytest.fixture
def pipeline(monkeypatch):
    """A chew run with no models: the pipeline, OCR preflight and OpenCV check are stubbed."""
    _Pipeline.built = []
    _Pipeline.seen = []
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    monkeypatch.setattr("bibr.local.cli.process._preflight_ocr_runtime", lambda config: None)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _Pipeline)
    return _Pipeline


def _pdf(directory: Path, name: str = "paper.pdf") -> Path:
    path = directory / name
    path.write_bytes(b"%PDF-1.4\n")
    return path


async def _chew(*argv: str) -> None:
    from bibr.local.cli import _build_parser, _run_process

    await _run_process(_build_parser().parse_args(["chew", *argv]))


def _main(monkeypatch, *argv: str) -> None:
    from bibr.local.cli import main

    monkeypatch.setattr(sys, "argv", ["bibr", *argv])
    main()


# --- usage errors: one styled line, a pointer to --help, exit 2 -----------------------


@pytest.mark.parametrize(
    ("argv", "line", "help_command"),
    [
        (["chw"], "Unknown command 'chw'. Did you mean 'chew'?", "bibr --help"),
        (
            ["chew", "paper.pdf", "--ouput", "x.json"],
            "Unknown option '--ouput'. Did you mean '--output'?",
            "bibr chew --help",
        ),
        (
            ["chew", "paper.pdf", "--refs", "lm"],
            "--refs: 'lm' is not one of its values. Did you mean 'llm'?",
            "bibr chew --help",
        ),
        (
            ["chew", "paper.pdf", "--refs", "zzzzzz"],
            "--refs: 'zzzzzz' is not one of its values. Choose from: llm, ner,",
            "bibr chew --help",
        ),
        (["preset", "rn", "x"], "Unknown command 'rn'.", "bibr preset --help"),
    ],
)
def test_a_usage_error_is_one_line_and_a_pointer_to_help(
    monkeypatch, capsys, argv, line, help_command
):
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, *argv)

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert line in err
    assert f"Run {help_command} to see what it accepts." in err
    # No argparse synopsis, no traceback.
    assert "usage:" not in err.lower()
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 2


def test_an_unknown_command_with_nothing_close_gets_no_suggestion(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "zzzzzzzz")

    assert exc.value.code == 2
    assert "Unknown command 'zzzzzzzz'." in capsys.readouterr().err


def test_an_unknown_option_far_from_every_real_one_names_it_without_a_suggestion(
    monkeypatch, capsys
):
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "chew", "paper.pdf", "--qqqqqqqq")

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Unknown option '--qqqqqqqq'." in err
    assert "Did you mean" not in err


def test_describe_usage_error_capitalises_any_other_message():
    from bibr.local.cli import _build_parser
    from bibr.local.cli.parser import _describe_usage_error

    parser = _build_parser()

    text, help_prog = _describe_usage_error(parser, "the following arguments are required: x")

    assert text == "The following arguments are required: x"
    assert help_prog == parser.prog


# --- no input / conflicting inputs ------------------------------------------------------


def test_chew_with_no_input_says_what_to_name(monkeypatch, capsys, pipeline):
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "chew")

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert (
        "no input: name a paper or a folder of papers, e.g. bibr chew paper.pdf -o result.json"
        in err
    )
    assert pipeline.built == []


def test_inputs_and_a_manifest_together_are_refused(tmp_path, monkeypatch, capsys, pipeline):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "chew", str(_pdf(tmp_path)), "--manifest", str(manifest))

    assert exc.value.code == 2
    assert "pass input paths or --manifest, not both" in capsys.readouterr().err
    assert pipeline.built == []


def test_resolve_cli_source_mode_names_each_mistake_and_the_valid_modes():
    from bibr.local.manifest import ManifestError, resolve_cli_source_mode

    assert resolve_cli_source_mode(["a.pdf"], None) == "inputs"
    assert resolve_cli_source_mode([], "m.jsonl") == "manifest"
    with pytest.raises(ManifestError, match="no input: name a paper or a folder of papers"):
        resolve_cli_source_mode([], None)
    with pytest.raises(ManifestError, match="not both"):
        resolve_cli_source_mode(["a.pdf"], "m.jsonl")


# --- interrupts and broken pipes ----------------------------------------------------------


def _raise_from_run_process(monkeypatch, exc: BaseException) -> None:
    async def _boom(args):  # noqa: ARG001
        raise exc

    monkeypatch.setattr("bibr.local.cli._run_process", _boom)


def test_ctrl_c_during_chew_prints_interrupted_and_exits_130(monkeypatch, capsys):
    _raise_from_run_process(monkeypatch, KeyboardInterrupt())
    monkeypatch.setattr("bibr.local.cli._terminating_signal", lambda: None)

    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "chew", "paper.pdf")

    assert exc.value.code == 130
    err = capsys.readouterr().err
    assert "Interrupted." in err
    assert "Traceback" not in err


@pytest.mark.parametrize("signum", [signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)])
@posix_only
def test_a_terminating_signal_during_chew_exits_128_plus_the_signal(monkeypatch, capsys, signum):
    _raise_from_run_process(monkeypatch, KeyboardInterrupt())
    monkeypatch.setattr("bibr.local.cli._terminating_signal", lambda: int(signum))

    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "chew", "paper.pdf")

    assert exc.value.code == 128 + int(signum)
    assert "Interrupted." in capsys.readouterr().err


@posix_only
def test_a_closed_pipe_exits_141_without_a_traceback(monkeypatch, capsys):
    """`bibr chew paper.pdf | head`: the reader leaves, the way a SIGPIPE-killed program does."""
    _raise_from_run_process(monkeypatch, BrokenPipeError(32, "Broken pipe"))

    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "chew", "paper.pdf")

    assert exc.value.code == 141
    assert "Traceback" not in capsys.readouterr().err


# --- the run preamble -------------------------------------------------------------------


async def test_a_missing_path_is_reported_alone_before_any_mode_line(tmp_path, capsys, pipeline):
    with pytest.raises(SystemExit) as exc:
        await _chew(str(tmp_path / "nope.pdf"), "--no-llm")

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "nope.pdf" in err
    assert "mode " not in err
    assert pipeline.built == []


async def test_an_unsupported_extension_is_reported_alone_before_any_mode_line(
    tmp_path, capsys, pipeline
):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a paper", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        await _chew(str(notes), "--no-llm")

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Unsupported file format: notes.txt" in err
    assert "bibr accepts .pdf, .docx, .xml (JATS), .html/.htm, and .epub files." in err
    assert "mode " not in err
    assert pipeline.built == []


async def test_an_unsupported_file_beside_a_supported_one_is_left_to_the_pipeline(
    tmp_path, capsys, pipeline
):
    """Only a run with nothing readable stops early; the pipeline refuses the odd one out."""
    notes = tmp_path / "notes.txt"
    notes.write_text("x", encoding="utf-8")

    await _chew(str(_pdf(tmp_path)), str(notes), "--no-llm", "-o", str(tmp_path / "out"))

    assert pipeline.built


async def test_the_preamble_is_a_single_mode_line_on_the_text_layer_route(
    tmp_path, capsys, pipeline
):
    await _chew(
        str(_pdf(tmp_path)), "--fast", "--ocr", "text-layer", "-o", str(tmp_path / "o.json")
    )

    lines = [line.strip() for line in capsys.readouterr().err.splitlines()]
    assert lines[0] == (
        "mode fast · local models only, no LLM call · "
        "OCR text-layer (no OCR model; reads the PDF's own text)"
    )
    # No OCR model runs, so no model/profile line follows the mode line.
    assert not any(line.startswith("OCR model") for line in lines)


async def test_the_preamble_adds_a_model_line_when_an_ocr_model_runs(tmp_path, capsys, pipeline):
    await _chew(
        str(_pdf(tmp_path)),
        "--no-llm",
        "--ocr",
        "paddle-http",
        "--ocr-url",
        "http://127.0.0.1:9",
        "-o",
        str(tmp_path / "o.json"),
    )

    lines = [line.strip() for line in capsys.readouterr().err.splitlines()]
    assert lines[0].startswith("mode structure only (--no-llm) · skips authors, references")
    assert "OCR paddle-http at http://127.0.0.1:9" in lines[0]
    assert lines[1].startswith("OCR model ")
    assert " · profile " in lines[1]


async def test_a_dry_run_prints_the_plan_but_no_preamble(tmp_path, capsys, pipeline):
    await _chew(str(_pdf(tmp_path)), "--fast", "--dry-run")

    captured = capsys.readouterr()
    # The plan carries the mode and the OCR route; the one-line preamble would repeat it.
    assert "Plan" in captured.out
    assert "local models only, no LLM call" not in captured.out + captured.err
    assert pipeline.built == []


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        (
            {"ocr_backend": "paddle", "ocr_summary_backend": "paddle-vllm"},
            "paddle-vllm (automatic)",
        ),
        ({"ocr_backend": "paddle"}, "paddle (automatic)"),
        (
            {"ocr_backend": "text-layer"},
            "text-layer (no OCR model; reads the PDF's own text)",
        ),
        (
            {"ocr_backend": "text-layer", "ocr_text_layer_fallback": True},
            "text-layer (no OCR backend can start here; reads the PDF's own text)",
        ),
        ({"ocr_backend": "gemini"}, "gemini"),
    ],
)
def test_ocr_route_names_the_route_the_run_will_take(route, expected):
    from bibr.local.cli.process import _ocr_route
    from bibr.local.cli.run_config import ResolvedRunConfig

    config = ResolvedRunConfig(memory_mode="balanced", llm_backend="cloud", **route)

    assert _ocr_route(config) == expected


# --- the validation summary -------------------------------------------------------------


def test_llm_skipped_is_not_counted_as_a_processing_note():
    from bibr.local.cli import _format_validation_line

    payload = {
        "validation": {"errors": 0, "warnings": 1, "issues": []},
        "extraction": {
            "warnings": [
                {"code": "LLM_SKIPPED", "message": "fast mode"},
                {"code": "REF_FALLBACK", "message": "fallback parser"},
            ]
        },
    }

    assert _format_validation_line(payload) == "1 warning to review · 1 processing note"

    payload["extraction"]["warnings"] = payload["extraction"]["warnings"][:1]
    assert _format_validation_line(payload) == "1 warning to review"


def test_processing_notes_keep_every_entry_unless_routine_ones_are_left_out():
    from bibr.local.cli.process import processing_notes

    payload = {
        "extraction": {
            "warnings": [{"code": "LLM_SKIPPED"}, {"code": "X"}, "not-a-dict"],
        }
    }

    assert [n["code"] for n in processing_notes(payload)] == ["LLM_SKIPPED", "X"]
    assert [n["code"] for n in processing_notes(payload, routine=False)] == ["X"]
    assert processing_notes(None) == []
    assert processing_notes({"extraction": {"warnings": "oops"}}) == []


@pytest.mark.parametrize(
    ("code", "subject"),
    [
        ("VAL_AUTHOR_EMPTY", "authors"),
        ("VAL_AFFILIATION_MISSING", "affiliations"),
        ("VAL_TITLE_GENERIC", "title"),
        ("VAL_DANGLING_REF", "references"),
        ("VAL_XREF_ZERO", "in-text citations"),
        ("VAL_PLACEHOLDER", None),
        ("NOT_A_VALIDATION_CODE", None),
        (None, None),
    ],
)
def test_validation_subject_names_the_part_of_the_paper_a_code_concerns(code, subject):
    from bibr.local.cli.process import validation_subject

    assert validation_subject(code) == subject


def test_validation_issues_lists_errors_first_and_skips_malformed_entries():
    from bibr.local.cli.process import validation_issues

    payload = {
        "validation": {
            "issues": [
                {"code": "W", "severity": "warning"},
                "junk",
                {"code": "E", "severity": "error"},
                {"code": "U"},
            ]
        }
    }

    assert [i["code"] for i in validation_issues(payload)] == ["E", "W", "U"]
    assert validation_issues(None) == []
    assert validation_issues({"validation": {"issues": "nope"}}) == []


def test_the_review_line_has_a_yellow_bang_only_when_there_are_errors(tmp_path):
    from rich.console import Console

    from bibr.local.cli.process import _print_validation_line

    def render(payload) -> str:
        console = Console(file=io.StringIO(), width=300, force_terminal=False)
        _print_validation_line(console, payload, tmp_path / "p.json")
        return console.file.getvalue()

    errors = render({"validation": {"errors": 1, "warnings": 0, "issues": []}})
    warnings_only = render({"validation": {"errors": 0, "warnings": 2, "issues": []}})

    assert errors == f"    ! 1 error to review · details: bibr inspect {tmp_path / 'p.json'}\n"
    assert warnings_only == (
        f"    2 warnings to review · details: bibr inspect {tmp_path / 'p.json'}\n"
    )
    assert render({"validation": {"errors": 0, "warnings": 0}}) == ""


# --- where the result goes ----------------------------------------------------------------


async def test_an_unwritable_output_stops_before_any_model_loads(tmp_path, capsys, pipeline):
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    target = blocker / "out.json"

    with pytest.raises(SystemExit) as exc:
        await _chew(str(_pdf(tmp_path)), "--no-llm", "-o", str(target))

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Cannot write output" in err
    assert "Traceback" not in err
    assert pipeline.built == []


def test_an_output_under_a_regular_file_is_the_blocker_the_dry_run_reports(tmp_path):
    from bibr.local.cli.inputs import _output_path_problem

    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")

    problem = _output_path_problem(str(blocker / "out.json"), is_batch=False)

    assert problem is not None
    assert problem.startswith("Cannot write output")


async def test_a_json_file_for_several_papers_is_refused(tmp_path, capsys, pipeline):
    _pdf(tmp_path, "a.pdf")
    _pdf(tmp_path, "b.pdf")

    with pytest.raises(SystemExit) as exc:
        await _chew(str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf"), "--no-llm", "-o", "x.json")

    assert exc.value.code == 2
    assert (
        "several papers write one JSON file each, so -o must name a directory"
        in capsys.readouterr().err
    )
    assert pipeline.built == []
    assert not Path("x.json").exists()


async def test_a_json_file_for_a_directory_input_is_refused(tmp_path, capsys, pipeline):
    papers = tmp_path / "papers"
    papers.mkdir()
    _pdf(papers)

    with pytest.raises(SystemExit) as exc:
        await _chew(str(papers), "--no-llm", "-o", str(tmp_path / "result.json"))

    assert exc.value.code == 2
    assert "so -o must name a directory" in capsys.readouterr().err
    assert not (tmp_path / "result.json").exists()
    assert pipeline.built == []


def test_output_problem_for_a_batch_only_objects_to_the_json_suffix(tmp_path):
    from bibr.local.cli.inputs import _output_path_problem

    assert _output_path_problem(None, is_batch=True) is None
    assert _output_path_problem(str(tmp_path / "Qwen3.5-4B"), is_batch=True) is None
    assert _output_path_problem(str(tmp_path / "results"), is_batch=True) is None
    assert "must name a directory" in _output_path_problem(str(tmp_path / "r.JSON"), is_batch=True)
    # A directory that already has a .json name is a directory: nothing to refuse.
    existing = tmp_path / "exports.json"
    existing.mkdir()
    assert _output_path_problem(str(existing), is_batch=True) is None
    # A single paper may of course name a file.
    assert _output_path_problem(str(tmp_path / "r.json"), is_batch=False) is None


async def test_several_papers_without_dash_o_write_json_lines_to_stdout(tmp_path, capsys, pipeline):
    await _chew(str(_pdf(tmp_path, "a.pdf")), str(_pdf(tmp_path, "b.pdf")), "--no-llm")

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert {p["paper_id"] for p in parsed} == {"a", "b"}
    # Compact: one document per line, no indentation.
    assert all("\n" not in line and ": " not in line for line in lines)
    assert "writing JSON Lines to stdout, one paper per line (-o DIR writes a file each)" in (
        captured.err
    )


async def test_json_lines_with_compact_prints_no_note(tmp_path, capsys, pipeline):
    await _chew(str(_pdf(tmp_path, "a.pdf")), str(_pdf(tmp_path, "b.pdf")), "--no-llm", "--compact")

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 2
    assert "JSON Lines" not in captured.err


async def test_one_paper_on_stdout_stays_indented_without_a_note(tmp_path, capsys, pipeline):
    await _chew(str(_pdf(tmp_path)), "--no-llm")

    captured = capsys.readouterr()
    assert captured.out.startswith("{\n  ")
    assert json.loads(captured.out)["paper_id"] == "paper"
    assert "JSON Lines" not in captured.err


@posix_only
def test_exports_are_created_honouring_the_umask(tmp_path):
    from bibr.local.artifacts import atomic_write_json

    previous = os.umask(0o022)
    try:
        atomic_write_json(tmp_path / "out.json", {"a": 1})
        os.umask(0o077)
        atomic_write_json(tmp_path / "private.json", {"a": 1})
    finally:
        os.umask(previous)

    assert (tmp_path / "out.json").stat().st_mode & 0o777 == 0o644
    assert (tmp_path / "private.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == {"a": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json", "private.json"]


# --- stdin and the bundled sample ---------------------------------------------------------


def test_stdin_input_is_named_stdin_with_the_payloads_extension(tmp_path, monkeypatch):
    from bibr.local.cli.inputs import _collect_files

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"%PDF-1.4\nrest")))

    files, missing = _collect_files(["-"])

    assert missing == 0
    (path,) = files
    assert path.name == "stdin.pdf"
    assert path.stem == "stdin"
    assert path.read_bytes() == b"%PDF-1.4\nrest"


async def test_a_piped_paper_gets_the_paper_id_stdin(tmp_path, monkeypatch, capsys, pipeline):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"%PDF-1.4\n")))

    await _chew("-", "--no-llm")

    assert json.loads(capsys.readouterr().out)["paper_id"] == "stdin"
    assert [p.name for p in pipeline.seen[0]] == ["stdin.pdf"]


def test_the_bundled_sample_paper_exists():
    from bibr.local.cli.inputs import sample_paper_path

    path = sample_paper_path()

    assert path.name == "sample_paper.pdf"
    assert path.read_bytes().startswith(b"%PDF")


async def test_sample_processes_the_bundled_paper(capsys, pipeline):
    from bibr.local.cli.inputs import sample_paper_path

    await _chew("--sample", "--no-llm")

    assert pipeline.seen == [[sample_paper_path()]]
    assert json.loads(capsys.readouterr().out)["paper_id"] == "sample_paper"


async def test_sample_refuses_an_input_beside_it(tmp_path, capsys, pipeline):
    with pytest.raises(SystemExit) as exc:
        await _chew("--sample", str(_pdf(tmp_path)), "--no-llm")

    assert exc.value.code == 2
    assert "--sample processes the paper that ships with bibr; drop the input." in (
        capsys.readouterr().err
    )
    assert pipeline.built == []


@pytest.mark.parametrize(
    "argv",
    [
        ["chew", "p.pdf"],
        ["batch", "papers/"],
        ["tables", "results/"],
        ["references", "results/"],
        ["ref-model", "download"],
    ],
)
@pytest.mark.parametrize("flag", ["-o", "--out", "--output"])
def test_every_output_spelling_is_accepted_by_every_command_that_writes(argv, flag):
    from bibr.local.cli import _build_parser

    args = _build_parser().parse_args([*argv, flag, "dest"])

    # The destination lives under `output` or `out` depending on the command.
    assert "dest" in {str(value) for value in vars(args).values()}


# --- the text-layer stage label -----------------------------------------------------------


def _stage_lines(labels):
    from bibr.pipeline.progress import RichProgress

    progress = RichProgress(stages=["ocr", "structure"], labels=labels)
    progress._console = __import__("rich.console", fromlist=["Console"]).Console(
        file=io.StringIO(), width=120, force_terminal=False
    )
    progress.stage_start("ocr")
    return progress._console.file.getvalue()


def test_the_text_layer_route_reads_the_pdf_text_layer_not_ocr():
    assert "Reading the PDF text layer..." in _stage_lines({"ocr": "Reading the PDF text layer"})
    assert "Reading text (OCR)" in _stage_lines(None)


@pytest.mark.parametrize(
    ("argv", "labels"),
    [
        (["--fast", "--ocr", "text-layer"], {"ocr": "Reading the PDF text layer"}),
        (["--no-llm", "--ocr", "paddle-http", "--ocr-url", "http://127.0.0.1:9"], None),
    ],
)
async def test_chew_labels_the_ocr_stage_by_the_route_it_takes(
    tmp_path, monkeypatch, pipeline, argv, labels
):
    captured: list[dict] = []
    import bibr.pipeline.progress as progress_module

    real = progress_module.RichProgress

    def spy(*args, **kwargs):
        captured.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(progress_module, "RichProgress", spy)

    await _chew(str(_pdf(tmp_path)), *argv, "-o", str(tmp_path / "o.json"))

    assert captured and captured[0]["labels"] == labels


# --- page ranges --------------------------------------------------------------------------


def test_a_page_range_that_is_not_numbers_says_what_is_accepted():
    from bibr.utils.pages import parse_pages

    for bad in ("abc", "1-x", "a-3", ""):
        with pytest.raises(
            ValueError,
            match=r"pages must be one page or a range, counted from 1 \(like '3' or '1-5'\), got",
        ) as exc:
            parse_pages(bad)
        assert repr(bad)[1:-1] in str(exc.value)
        assert exc.value.__cause__ is None

    # Counted from 1 by the user, 0-based inside.
    assert parse_pages("3") == (2, 2)
    assert parse_pages("1-5") == (0, 4)
    with pytest.raises(ValueError, match="pages are 1-based, got '0'"):
        parse_pages("0")


def test_pages_abc_is_refused_on_the_command_line_in_one_line(
    monkeypatch, tmp_path, capsys, pipeline
):
    paper = _pdf(tmp_path)

    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "chew", str(paper), "--no-llm", "--pages", "abc")

    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "pages must be one page or a range, counted from 1 (like '3' or '1-5'), got 'abc'" in err
    assert "Traceback" not in err
    assert pipeline.built == []


# --- presets without a terminal -------------------------------------------------------------


@pytest.fixture
def preset_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".bibr").mkdir(parents=True)
    (home / ".bibr" / ".env").write_text("LLM_PROVIDER=google\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("BIBR_ENV_FILE", raising=False)
    monkeypatch.setenv("BIBR_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # not a TTY; reading it raises EOFError
    return tmp_path / "presets"


def _preset(**kwargs) -> int:
    from bibr.local.cli.presets import _run_preset

    code = 0
    try:
        _run_preset(argparse.Namespace(**kwargs))
    except SystemExit as exc:
        code = exc.code
    return code


def test_preset_rm_without_a_terminal_names_yes_instead_of_a_traceback(preset_env, capsys):
    from bibr.presets import PresetManager

    assert _preset(preset_command="save", name="p", force=True) == 0
    assert PresetManager(presets_dir=preset_env).exists("p")
    capsys.readouterr()

    code = _preset(preset_command="rm", name="p", yes=False)

    assert code == 1
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "no terminal to ask on" in text
    assert "--yes" in text
    assert PresetManager(presets_dir=preset_env).exists("p")


def test_preset_rm_with_yes_still_works_without_a_terminal(preset_env, capsys):
    from bibr.presets import PresetManager

    _preset(preset_command="save", name="p", force=True)

    assert _preset(preset_command="rm", name="p", yes=True) == 0
    assert not PresetManager(presets_dir=preset_env).exists("p")


def test_preset_save_over_an_existing_one_without_a_terminal_names_force(preset_env, capsys):
    from bibr.presets import PresetManager

    _preset(preset_command="save", name="p", force=True)
    before = PresetManager(presets_dir=preset_env).load("p")
    capsys.readouterr()

    code = _preset(preset_command="save", name="p", force=False)

    assert code == 1
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "no terminal to ask on" in text
    assert "--force" in text
    assert PresetManager(presets_dir=preset_env).load("p") == before


# --- logging of reference-extraction notes -----------------------------------------------


def test_a_reference_extraction_note_is_logged_at_info_not_warning(caplog):
    from bibr.extract.ref_extractor import ReferenceExtractor
    from bibr.processing_warnings import WarningCode

    extractor = ReferenceExtractor.__new__(ReferenceExtractor)
    extractor.contents = SimpleNamespace(processing_warnings=[])
    code = WarningCode.REF_PARSE_FINALIZE_FAILED

    with caplog.at_level(logging.INFO, logger="bibr.extract.ref_extractor"):
        extractor._record_warning(code, "reference 3 kept as parsed")

    records = [r for r in caplog.records if r.name == "bibr.extract.ref_extractor"]
    assert [r.levelno for r in records] == [logging.INFO]
    assert records[0].getMessage() == (
        f"Reference extraction note {code}: reference 3 kept as parsed"
    )
    # The note still reaches the export.
    (note,) = extractor.contents.processing_warnings
    assert (note.code, note.message) == (code, "reference 3 kept as parsed")
