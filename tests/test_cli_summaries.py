"""Tests for validation-review summaries in ``bibr chew`` run output.

The export JSON carries a top-level ``validation = {"errors": int,
"warnings": int, "issues": [...]}`` block (absent when the output-validation
gate was skipped) plus ``extraction.warnings`` ``{code, message}`` objects
(``processing_warnings`` strings before v11). After each file's
✓ line, ``_write_chunk_results`` prints an indented dim one-liner ("2 errors
and 3 warnings to review (title, authors)") — and nothing extra for a clean
payload. The codes and messages are ``bibr inspect``'s job, not this line's. The batch summary
additionally gains an aggregate "X files clean, Y with warnings, Z with
errors" line, computed in ``_run_process`` (not ``_write_chunk_results``,
whose ``(processed, errors)`` return signature other tests rely on).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from bibr.pipeline.state import FileState


def _printed(console: MagicMock) -> str:
    return "\n".join(str(c) for c in console.print.call_args_list)


# --- per-file review line: absence on clean payloads ------------------------------


def test_no_validation_key_prints_nothing_extra(tmp_path):
    """A payload with no ``validation`` key and no ``processing_warnings``
    must print nothing extra — clean-run output stays byte-identical."""
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "clean.pdf")
    fs.result_json = {"info": {"title": "x"}}
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

    assert processed == 1
    assert errors == 0
    printed = _printed(console)
    assert "to review" not in printed
    assert "Done" in printed  # the existing ✓ line is still there


def test_validation_present_but_zero_counts_prints_nothing_extra(tmp_path):
    """``validation`` block present but errors == warnings == 0 (gate ran,
    found nothing) must also print nothing extra."""
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "clean.pdf")
    fs.result_json = {
        "info": {"title": "x"},
        "validation": {"errors": 0, "warnings": 0, "issues": []},
        "processing_warnings": [],
    }
    console = MagicMock()

    _write_chunk_results(
        [fs],
        output_path=None,
        json_kwargs={"indent": 2},
        console=console,
        is_batch=False,
        total_files=1,
        total_t0=0,
    )

    assert "to review" not in _printed(console)


# --- per-file review line: presence + formatting ----------------------------------


def test_validation_errors_and_warnings_prints_summary_line(tmp_path):
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "dirty.pdf")
    fs.result_json = {
        "info": {"title": "x"},
        "validation": {
            "errors": 2,
            "warnings": 3,
            "issues": [
                {
                    "code": "VAL_PLACEHOLDER",
                    "severity": "error",
                    "message": "2 placeholder token(s) leaked",
                    "count": 2,
                },
                {
                    "code": "VAL_TITLE_GENERIC",
                    "severity": "warning",
                    "message": "generic or empty title",
                    "count": 1,
                },
            ],
        },
    }
    console = MagicMock()

    _write_chunk_results(
        [fs],
        output_path=None,
        json_kwargs={"indent": 2},
        console=console,
        is_batch=False,
        total_files=1,
        total_t0=0,
    )

    printed = _printed(console)
    assert "2 errors and 3 warnings to review (title)" in printed
    # Codes and messages moved to `bibr inspect`; the summary line is counts and subjects.
    assert "VAL_PLACEHOLDER" not in printed
    assert "placeholder token" not in printed
    # Errors earn a yellow ! under the ✓ line; the result went to stdout, so no file to inspect.
    assert "[yellow]![/yellow]" in printed
    assert "details: bibr inspect" not in printed


def test_subjects_follow_severity_order_and_stop_at_three(tmp_path):
    """The parenthesised subjects list errors' subjects first, even when the
    issues list orders warnings before errors, and names at most three."""
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "dirty.pdf")
    fs.result_json = {
        "validation": {
            "errors": 1,
            "warnings": 4,
            "issues": [
                {"code": "VAL_TITLE_GENERIC", "severity": "warning", "message": "warn one"},
                {"code": "VAL_XREF_ZERO", "severity": "warning", "message": "warn two"},
                {"code": "VAL_DOI_BAD", "severity": "warning", "message": "warn three"},
                {"code": "VAL_ABSTRACT_SHORT", "severity": "warning", "message": "warn four"},
                {"code": "VAL_AUTHOR_EMPTY", "severity": "error", "message": "err one"},
            ],
        }
    }
    console = MagicMock()

    _write_chunk_results(
        [fs],
        output_path=None,
        json_kwargs={"indent": 2},
        console=console,
        is_batch=False,
        total_files=1,
        total_t0=0,
    )

    printed = _printed(console)
    # The error's subject leads, three subjects are named, the rest are elided.
    assert "1 error and 4 warnings to review (authors, title, in-text citations, ...)" in printed
    assert "abstract" not in printed


def test_processing_warnings_become_a_processing_notes_suffix(tmp_path):
    """``extraction.warnings`` entries add a "· K processing notes" suffix.
    v11 stopped mirroring gate findings there, so every entry is a genuine
    processing note and nothing is double-counted."""
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "dirty.pdf")
    fs.result_json = {
        "validation": {
            "errors": 0,
            "warnings": 1,
            "issues": [
                {"code": "VAL_TITLE_GENERIC", "severity": "warning", "message": "generic title"},
            ],
        },
        "extraction": {
            "warnings": [
                {
                    "code": "STATEMENT_LEXICAL_FALLBACK",
                    "message": "funding_statement filled by lexical anchor matching",
                }
            ]
        },
    }
    console = MagicMock()

    _write_chunk_results(
        [fs],
        output_path=None,
        json_kwargs={"indent": 2},
        console=console,
        is_batch=False,
        total_files=1,
        total_t0=0,
    )

    printed = _printed(console)
    assert "1 warning to review (title) · 1 processing note" in printed
    assert "STATEMENT_LEXICAL_FALLBACK" not in printed


def test_processing_warnings_absent_no_notes_suffix(tmp_path):
    """No ``extraction.warnings`` -> no '· K processing notes' suffix."""
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "dirty.pdf")
    fs.result_json = {
        "validation": {
            "errors": 1,
            "warnings": 0,
            "issues": [{"code": "VAL_EMPTY_EQ", "severity": "error", "message": "bad eq"}],
        },
        "extraction": {"warnings": []},
    }
    console = MagicMock()

    _write_chunk_results(
        [fs],
        output_path=None,
        json_kwargs={"indent": 2},
        console=console,
        is_batch=False,
        total_files=1,
        total_t0=0,
    )

    printed = _printed(console)
    assert "processing note" not in printed
    assert "1 error to review (equations)" in printed
    assert "warnings to review" not in printed


# --- per-file review line appears across the write paths -------------------


def test_batch_write_path_prints_review_line_with_inspect_pointer(tmp_path):
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "dirty.pdf")
    fs.result_json = {"validation": {"errors": 1, "warnings": 0, "issues": []}}
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    console = MagicMock()

    _write_chunk_results(
        [fs],
        output_path=out_dir,
        json_kwargs={"indent": 2},
        console=console,
        is_batch=True,
        total_files=1,
        total_t0=0,
    )

    printed = _printed(console)
    assert "dirty.pdf" in printed
    assert "1 error to review" in printed
    # A result written to a file ends with the command that lists every finding.
    assert f"details: bibr inspect {out_dir / 'dirty.json'}" in printed
    assert printed.index("dirty.pdf") < printed.index("1 error to review")


def test_single_output_file_write_path_prints_review_line_with_inspect_pointer(tmp_path):
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "dirty.pdf")
    fs.result_json = {"validation": {"errors": 1, "warnings": 0, "issues": []}}
    out_file = tmp_path / "out.json"
    console = MagicMock()

    _write_chunk_results(
        [fs],
        output_path=out_file,
        json_kwargs={"indent": 2},
        console=console,
        is_batch=False,
        total_files=1,
        total_t0=0,
    )

    printed = _printed(console)
    assert "Wrote" in printed
    assert "1 error to review" in printed
    assert f"details: bibr inspect {out_file}" in printed


def test_processing_error_file_never_prints_review_line(tmp_path):
    """A file that failed processing entirely (``fs.error`` set) has no
    ``result_json`` — must not blow up and must not print a review line."""
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "corrupt.pdf")
    fs.error = "Unsupported format or corrupt file"
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
    assert "to review" not in _printed(console)


# --- batch summary aggregate -------------------------------------------------


class _FakePipeline:
    """Stand-in for ``LocalPipeline``; ``process_chunk`` is set per-test."""

    def __init__(self, **_kwargs):
        pass

    async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
        raise NotImplementedError

    async def aclose(self):
        pass

    def llm_usage_snapshot(self):
        return {}


def _disable_ocr_runtime_preflight(monkeypatch):
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    # The OCR runtime preflight fails where no automatic runtime can start
    # (Windows CI: llama.cpp only, no llama-server); the pipeline is a fake.
    monkeypatch.setattr("bibr.local.cli.process._preflight_ocr_runtime", lambda config: None)


async def test_batch_summary_reports_clean_warning_error_breakdown(tmp_path, monkeypatch, capsys):
    """Batch summary gains an aggregate 'X files clean, Y with warnings, Z
    with errors' line reflecting each successfully-processed file's
    validation status."""
    from bibr.local.cli import _build_parser, _run_process

    results = {
        "clean.pdf": {"info": {"title": "clean"}},
        "warn.pdf": {
            "info": {"title": "warn"},
            "validation": {"errors": 0, "warnings": 1, "issues": []},
        },
        "err.pdf": {
            "info": {"title": "err"},
            "validation": {"errors": 1, "warnings": 0, "issues": []},
        },
    }

    class _Pipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
            for fs in file_states:
                fs.result_json = results[fs.path.name]

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _Pipeline)

    for name in results:
        (tmp_path / name).write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(
        [
            "chew",
            str(tmp_path / "clean.pdf"),
            str(tmp_path / "warn.pdf"),
            str(tmp_path / "err.pdf"),
            "--no-llm",
        ]
    )
    await _run_process(args)

    err = capsys.readouterr().err
    assert "1 files clean, 1 with warnings, 1 with errors" in err


async def test_batch_summary_all_clean_still_shows_zero_breakdown(tmp_path, monkeypatch, capsys):
    from bibr.local.cli import _build_parser, _run_process

    class _Pipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
            for fs in file_states:
                fs.result_json = {"info": {"title": fs.path.name}}

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _Pipeline)

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(
        ["chew", str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf"), "--no-llm"]
    )
    await _run_process(args)

    err = capsys.readouterr().err
    assert "2 files clean, 0 with warnings, 0 with errors" in err


# --- malformed payloads must never crash a run whose output was written -------


def test_malformed_count_types_treated_as_zero():
    """Non-int errors/warnings (a corrupt or hand-edited payload) must coerce
    to 0, not TypeError — in both the per-file line and the batch accumulator
    seam (`_validation_counts` feeds both)."""
    from bibr.local.cli import _format_validation_line, _validation_counts

    payload = {"validation": {"errors": "bad", "warnings": 0, "issues": []}}
    assert _validation_counts(payload) == (0, 0)
    assert _format_validation_line(payload) is None

    # bool is an int subclass — must not sneak through as a count
    assert _validation_counts({"validation": {"errors": True, "warnings": None}}) == (0, 0)
    assert _validation_counts({"validation": {"errors": -3, "warnings": 2}}) == (0, 2)


def test_non_dict_issue_entries_skipped_not_crashed():
    from bibr.local.cli import _format_validation_line

    payload = {
        "validation": {
            "errors": 1,
            "warnings": 0,
            "issues": ["not-a-dict", {"code": "C1", "severity": "error", "message": "m"}],
        }
    }
    line = _format_validation_line(payload)
    assert line is not None
    assert line.startswith("1 error to review")
    assert "not-a-dict" not in line
    assert "?" not in line


def test_issue_dicts_missing_keys_do_not_render_placeholders():
    from bibr.local.cli import _format_validation_line

    payload = {"validation": {"errors": 1, "warnings": 0, "issues": [{}]}}
    line = _format_validation_line(payload)
    assert line is not None
    # No code means no known subject: the counts stand alone, with no placeholder.
    assert line == "1 error to review"
    assert "None" not in line


def test_non_list_processing_warnings_ignored():
    """A string here would otherwise be iterated char-by-char (silent wrong
    count); an int would raise. Both must be treated as no extra warnings."""
    from bibr.local.cli import _format_validation_line

    for bad in ("oops", 7, {"k": "v"}):
        payload = {
            "validation": {"errors": 0, "warnings": 1, "issues": []},
            "processing_warnings": bad,
        }
        line = _format_validation_line(payload)
        assert line is not None
        assert "processing note" not in line


def test_malformed_validation_block_does_not_crash_write_path(tmp_path):
    """End-to-end through _write_chunk_results: a malformed block must not
    take down a run whose output was already produced."""
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "weird.pdf")
    fs.result_json = {
        "info": {"title": "x"},
        "validation": {"errors": "bad", "warnings": [], "issues": "nope"},
        "processing_warnings": 3,
    }
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

    assert processed == 1
    assert errors == 0
    assert "to review" not in _printed(console)
