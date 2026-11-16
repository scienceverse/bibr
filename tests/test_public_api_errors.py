"""Errors a library caller meets: specific, early, re-exported, and quiet on stderr.

``bibr.chew()`` on a missing file, an unsupported file or a misspelt OCR
backend raises a bibr exception before any model loads, and expected user
errors log no traceback. ``import bibr`` installs a NullHandler, so internal
WARNING records do not print on stderr in a script that configured no logging.
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import bibr
from bibr import exceptions
from bibr.api import Records, _library_wording
from bibr.ocr import registry


def _no_pipeline(monkeypatch):
    def _refuse(*_args, **_kwargs):
        raise AssertionError("the pipeline was built for a call that should fail first")

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _refuse)


# --- re-exports -------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "BibrError",
        "ProcessingError",
        "InputValidationError",
        "InputFileNotFoundError",
        "InvalidOptionError",
        "ConfigurationError",
        "UpstreamServiceError",
    ],
)
def test_exceptions_are_reexported_lazily(name):
    assert name in bibr.__all__
    assert getattr(bibr, name) is getattr(exceptions, name)


def test_dir_lists_only_the_public_names():
    assert dir(bibr) == sorted(bibr.__all__)
    for helper in ("Any", "TYPE_CHECKING", "PackageNotFoundError", "version", "logging"):
        assert helper not in dir(bibr)


# --- missing file ---------------------------------------------------------------------


def test_missing_file_raises_before_any_model_loads(tmp_path, monkeypatch):
    _no_pipeline(monkeypatch)
    missing = tmp_path / "nope.pdf"
    with pytest.raises(bibr.InputFileNotFoundError) as raised:
        bibr.chew(missing)
    error = raised.value
    assert isinstance(error, FileNotFoundError)
    assert isinstance(error, bibr.InputValidationError)
    assert isinstance(error, bibr.BibrError)
    assert error.error_code == "file_not_found"
    assert error.filename == str(missing)
    assert str(error) == f"input file not found: {missing}"


def test_missing_file_error_pickles():
    error = pickle.loads(pickle.dumps(exceptions.InputFileNotFoundError("a.pdf")))
    assert isinstance(error, exceptions.InputFileNotFoundError)
    assert str(error) == "input file not found: a.pdf"


def test_a_cold_chewer_refuses_a_missing_file_without_loading(tmp_path, monkeypatch):
    _no_pipeline(monkeypatch)
    chewer = bibr.Chewer(mode="fast")
    with pytest.raises(FileNotFoundError):
        chewer.chew(tmp_path / "nope.pdf")
    assert chewer._pipeline is None
    chewer.close()


@pytest.mark.asyncio
async def test_validate_stage_records_a_missing_file_without_a_traceback(caplog):
    from unittest.mock import MagicMock

    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.validate import ValidateStage
    from bibr.pipeline.state import FileState

    fs = FileState(path=Path("/nonexistent/paper.pdf"))
    ctx = PipelineContext(
        file_states=[fs], progress=NullProgress(), resources=MagicMock(), config=RunConfig()
    )
    with caplog.at_level(logging.DEBUG, logger="bibr"):
        await ValidateStage().run(ctx)

    assert fs.error_code == "file_not_found"
    assert fs.failed_stage == "validate"
    assert isinstance(fs.original_error, exceptions.InputFileNotFoundError)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING or r.exc_info]


# --- unsupported format ---------------------------------------------------------------


def test_unsupported_format_names_the_extension_and_the_supported_ones(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("plain text")
    with pytest.raises(bibr.ProcessingError) as raised:
        bibr.chew(notes, mode="fast")
    assert raised.value.error_code == "unsupported_format"
    message = str(raised.value)
    assert message.startswith("Unsupported file format")
    assert "'.txt'" in message and ".pdf" in message and ".docx" in message


def test_a_refused_extension_says_what_to_do():
    from bibr.input.validate import validate_input_file

    with pytest.raises(bibr.InputValidationError, match=r"explicitly unsupported.*\.docx"):
        validate_input_file("old.doc", b"binary")


# --- OCR backend names ----------------------------------------------------------------


def test_unknown_ocr_backend_fails_at_once_with_a_suggestion(monkeypatch):
    _no_pipeline(monkeypatch)
    with pytest.raises(bibr.InvalidOptionError) as raised:
        bibr.chew("paper.pdf", ocr="padle")
    error = raised.value
    assert isinstance(error, ValueError) and isinstance(error, bibr.ConfigurationError)
    assert "ocr='padle'" in str(error)
    assert "Did you mean 'paddle'?" in str(error)
    assert "text-layer" in str(error)


def test_chewer_checks_the_backend_at_construction():
    with pytest.raises(bibr.InvalidOptionError, match="ocr_backend='nope'"):
        bibr.Chewer(ocr_backend="nope")


@pytest.mark.parametrize(
    "name", [*registry.OCR_BACKEND_NAMES, "http", "glm-local", "glm-mlx", "serve-http"]
)
def test_every_documented_name_and_alias_passes(name):
    registry.check_backend_request(name)


def test_the_name_list_covers_every_registered_backend_and_cli_choice():
    import importlib

    for module in ("bibr.local.ocr", "bibr.local.rapid_mlx", "bibr.local.ocr_cloud"):
        importlib.import_module(module)
    importlib.import_module("bibr.ocr.text_layer")
    from bibr.local.cli.parser import OCR_BACKEND_CHOICES

    for name in [*registry.known_backends(), *OCR_BACKEND_CHOICES]:
        registry.check_backend_request(name)
    assert set(OCR_BACKEND_CHOICES) == set(registry.OCR_BACKEND_NAMES)


def test_other_option_errors_are_invalid_option_errors_and_value_errors():
    for options in ({"refs": "bad"}, {"mode": "fastest"}, {"ref_seg": "x"}, {"consolidate": "x"}):
        with pytest.raises(bibr.InvalidOptionError):
            bibr.Chewer(**options)
        with pytest.raises(ValueError):
            bibr.Chewer(**options)


# --- scanned PDF wording ---------------------------------------------------------------


def test_scan_refusal_uses_the_library_option_names():
    message = f"This looks like a scanned PDF, and needs {registry.SCAN_NEEDS_OCR_ROUTES}"
    worded = _library_wording(message, "scan_needs_ocr")
    assert "--ocr" not in worded
    assert 'ocr="anthropic"' in worded and 'ocr="paddle"' in worded and "ocr_url=" in worded
    # Other failures are left as they are.
    assert _library_wording(message, "ocr_failed") == message


@pytest.mark.asyncio
async def test_chew_on_rewords_a_raised_scan_refusal(tmp_path):
    from bibr.api import _chew_on

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF")
    message = f"no usable text layer; needs {registry.SCAN_NEEDS_OCR_ROUTES}"

    class _Pipeline:
        async def process_file(self, *_args, **_kwargs):
            raise bibr.ProcessingError(message, error_code="scan_needs_ocr")

    with pytest.raises(bibr.ProcessingError) as raised:
        await _chew_on(_Pipeline(), pdf)
    assert raised.value.error_code == "scan_needs_ocr"
    assert 'ocr="anthropic"' in str(raised.value)

    # bibr batch prints its failures on the command line: the flags stay.
    with pytest.raises(bibr.ProcessingError) as raised:
        await _chew_on(_Pipeline(), pdf, library_wording=False)
    assert "--ocr anthropic" in str(raised.value)


def test_bibr_batch_keeps_the_cli_wording(monkeypatch):
    from bibr.batch.runner import LocalOptions, WarmChewMany

    monkeypatch.setattr("bibr.config.snapshot_settings", lambda settings=None: settings)
    monkeypatch.setattr("bibr.api._preflight_llm", lambda settings, kwargs: "fast")
    warm = WarmChewMany(LocalOptions(chew_options={"mode": "fast"}))
    assert warm._chewer._library_wording is False
    warm.close()


# --- Records ------------------------------------------------------------------------


def test_records_repr_is_a_short_summary():
    rows = Records([{"a": i, "b": str(i), "c": None} for i in range(100)])
    text = repr(rows)
    assert text == "<bibr.Records: 100 rows × 3 columns: a, b, c>"
    assert repr(Records([])) == "<bibr.Records: 0 rows>"
    assert repr(Records([{"x": 1}])) == "<bibr.Records: 1 row × 1 columns: x>"
    wide = Records([{f"k{i}": i for i in range(12)}])
    assert repr(wide).endswith("k7, … (+4)>")


def test_records_stay_a_list_and_render_as_a_table():
    rows = Records([{"a": 1}, {"a": 2, "b": 3}])
    assert rows == [{"a": 1}, {"a": 2, "b": 3}] and rows[1]["b"] == 3
    html = rows._repr_html_()
    assert html is not None and "<table" in html
    assert Records.df.__doc__


# --- logging --------------------------------------------------------------------------


def _python(code: str) -> subprocess.CompletedProcess:
    repo_root = str(Path(bibr.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": repo_root}
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )


def test_bibr_warnings_stay_off_stderr_without_logging_configured():
    code = "import bibr, logging; logging.getLogger('bibr.extract.x').warning('internal note')"
    assert "internal note" not in _python(code).stderr


def test_bibr_warnings_reach_a_configured_root_handler():
    code = (
        "import logging; logging.basicConfig(); import bibr; "
        "logging.getLogger('bibr.extract.x').warning('internal note')"
    )
    assert "internal note" in _python(code).stderr
