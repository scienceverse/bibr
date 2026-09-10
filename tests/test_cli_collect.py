"""Tests for ``bibr.local.cli._collect_files`` — focuses on stdin handling."""

import io
import sys
from pathlib import Path

import pytest

from bibr.local.cli import (
    _collect_files,
    _resolve_single_output_path,
    _suffix_for_stdin_payload,
)


def test_suffix_detection_for_pdf():
    assert _suffix_for_stdin_payload(b"%PDF-1.4\n...rest") == ".pdf"


def test_suffix_detection_for_docx():
    # ZIP local-file-header magic — DOCX is a ZIP under the hood.
    assert _suffix_for_stdin_payload(b"PK\x03\x04...") == ".docx"


def test_suffix_detection_default():
    assert _suffix_for_stdin_payload(b"random bytes") == ".pdf"


def test_collect_files_stdin_pdf_uses_pdf_suffix(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _StubStdin(b"%PDF-1.4\nhello"))
    files, missing_count = _collect_files(["-"])
    assert len(files) == 1
    assert files[0].suffix == ".pdf"
    assert missing_count == 0
    files[0].unlink(missing_ok=True)


def test_collect_files_stdin_docx_uses_docx_suffix(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _StubStdin(b"PK\x03\x04zip-stuff"))
    files, missing_count = _collect_files(["-"])
    assert len(files) == 1
    assert files[0].suffix == ".docx"
    assert missing_count == 0
    files[0].unlink(missing_ok=True)


def test_collect_files_stdin_registers_cleanup(monkeypatch):
    """The stdin temp file must be scheduled for atexit removal so each
    ``bibr chew -`` invocation doesn't leak a PDF-sized file in /tmp."""
    registered: list = []

    def fake_register(func, *args, **kwargs):
        registered.append((func, args, kwargs))

    monkeypatch.setattr("atexit.register", fake_register)
    monkeypatch.setattr(sys, "stdin", _StubStdin(b"%PDF-1.4\n"))

    files, _missing_count = _collect_files(["-"])
    assert len(registered) == 1
    # Run the cleanup callback manually and confirm the file is gone.
    func, _args, _kwargs = registered[0]
    assert files[0].exists()
    func()
    assert not files[0].exists()


class _StubStdin:
    def __init__(self, data: bytes):
        self.buffer = io.BytesIO(data)


def test_resolve_single_output_path_to_existing_dir(tmp_path):
    """``-o some_dir/`` (existing directory) must write ``<stem>.json`` inside."""
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    paper = Path("/abs/data/0956797617707270.pdf")

    resolved = _resolve_single_output_path(out_dir, paper)
    assert resolved == out_dir / "0956797617707270.json"


def test_resolve_single_output_path_to_explicit_file(tmp_path):
    """An explicit file path is used as-is, even with a ``.json`` suffix."""
    out_file = tmp_path / "result.json"
    paper = Path("/abs/data/paper.pdf")

    resolved = _resolve_single_output_path(out_file, paper)
    assert resolved == out_file


def test_resolve_single_output_path_to_missing_path(tmp_path):
    """Non-existent path is treated as a file target (parent assumed to exist)."""
    out_file = tmp_path / "subdir" / "result.json"
    paper = Path("/abs/data/paper.pdf")

    resolved = _resolve_single_output_path(out_file, paper)
    assert resolved == out_file


# ---- _collect_files — missing/mixed input reporting ------------------------


def test_collect_files_all_missing_exits_and_reports(tmp_path, capsys):
    """No valid files at all: exit 1, no misleading zero-error return."""
    missing = str(tmp_path / "does-not-exist.pdf")
    with pytest.raises(SystemExit) as exc_info:
        _collect_files([missing])
    assert exc_info.value.code == 1


def test_collect_files_mixed_valid_and_missing_reports_missing_and_counts(tmp_path, capsys):
    """One valid file + one missing path: the valid file must still be
    returned (not silently dropped) and the missing path must be surfaced
    both as a printed message and as a nonzero ``missing_count`` — previously
    this was dropped with no message and no error accounting at all."""
    valid = tmp_path / "paper.pdf"
    valid.write_bytes(b"%PDF-1.4\n")
    missing = str(tmp_path / "does-not-exist.pdf")

    files, missing_count = _collect_files([str(valid), missing])

    assert files == [valid]
    assert missing_count == 1
    captured = capsys.readouterr()
    assert "does-not-exist.pdf" in captured.err
    assert "not found" in captured.err.lower()
