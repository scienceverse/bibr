"""CLI logging vs Rich Live displays.

In streaming mode a window's back half (which logs ``bibr.*`` at INFO) runs
while a later window's OCR bar — a Rich ``Live`` display — is active. Rich
already redirects ``sys.stderr`` through a ``FileProxy`` that reprints whole
lines above the bar, but a plain ``StreamHandler`` binds the original stream
object at ``basicConfig`` time and bypasses the proxy, garbling the bar. The
CLI therefore logs through a handler that resolves ``sys.stderr`` per record.
"""

from __future__ import annotations

import logging
import sys
import warnings
from io import StringIO

import pytest
from rich.console import Console
from rich.file_proxy import FileProxy

from bibr.local.cli import _LiveStderrHandler
from bibr.pipeline.progress import RichProgress


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("bibr.pipeline", logging.INFO, __file__, 1, msg, None, None)


def _handler() -> _LiveStderrHandler:
    handler = _LiveStderrHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


class TestLiveStderrHandler:
    def test_resolves_stderr_at_emit_time(self, monkeypatch):
        # The stream a record lands on is whatever sys.stderr is AT EMIT,
        # not the object bound when the handler was constructed.
        handler = _handler()
        buf = StringIO()
        monkeypatch.setattr(sys, "stderr", buf)

        handler.emit(_record("resolved late"))

        assert "resolved late" in buf.getvalue()

    def test_routes_through_live_console_while_ocr_bar_active(self, monkeypatch):
        # While the OCR bar's Live display is up, Rich proxies sys.stderr;
        # a record emitted mid-bar must land in the bar's console (printed
        # above the live region), not on the real stderr underneath it.
        real_stderr = StringIO()
        monkeypatch.setattr(sys, "stderr", real_stderr)
        rp = RichProgress()
        rp._console = Console(file=StringIO(), force_terminal=True, width=80)
        handler = _handler()

        rp.ocr_start(3)
        try:
            assert isinstance(sys.stderr, FileProxy)
            handler.emit(_record("mid-bar log line"))
        finally:
            rp.ocr_end()

        assert "mid-bar log line" in rp._console.file.getvalue()
        assert "mid-bar log line" not in real_stderr.getvalue()
        # Live stopped: the proxy is gone and plain emits reach stderr again.
        assert not isinstance(sys.stderr, FileProxy)
        handler.emit(_record("after the bar"))
        assert "after the bar" in real_stderr.getvalue()


@pytest.fixture
def cli_logging():
    """Let ``main()`` configure the root logger afresh, then put every logger back.

    Returns the function that clears the root's handlers: pytest adds its own after fixture
    setup, and ``basicConfig`` does nothing while the root has any.
    """
    names = ("bibr.local", "bibr.pipeline", "bibr.structure", "bibr.extract", "httpcore", "httpx")
    root = logging.getLogger()
    saved = (root.level, list(root.handlers), {n: logging.getLogger(n).level for n in names})

    def clear_root_handlers():
        for handler in list(root.handlers):
            root.removeHandler(handler)

    yield clear_root_handlers
    clear_root_handlers()
    for handler in saved[1]:
        root.addHandler(handler)
    root.setLevel(saved[0])
    for name, level in saved[2].items():
        logging.getLogger(name).setLevel(level)


class TestShowOnceFilter:
    """The Hugging Face "unauthenticated requests" advice is printed once, not per client."""

    _ADVICE = "You are sending unauthenticated requests to the HF Hub. Set HF_TOKEN."

    def _passes(self, flt, msg: str) -> bool:
        return flt.filter(_record(msg))

    def test_the_first_record_with_the_marker_passes_and_repeats_do_not(self):
        from bibr.local.cli import _ShowOnceFilter

        flt = _ShowOnceFilter("unauthenticated requests")

        assert self._passes(flt, self._ADVICE) is True
        assert self._passes(flt, self._ADVICE) is False
        assert self._passes(flt, "Warning: " + self._ADVICE + " (from another client)") is False

    def test_other_records_always_pass(self):
        from bibr.local.cli import _ShowOnceFilter

        flt = _ShowOnceFilter("unauthenticated requests")

        for _ in range(3):
            assert self._passes(flt, "downloading model.onnx") is True

    def test_each_marker_is_tracked_on_its_own(self):
        from bibr.local.cli import _ShowOnceFilter

        flt = _ShowOnceFilter("first advice", "second advice")

        assert self._passes(flt, "first advice") is True
        assert self._passes(flt, "second advice") is True
        assert self._passes(flt, "first advice") is False
        assert self._passes(flt, "second advice") is False

    def test_the_filter_matches_the_formatted_message_not_the_template(self):
        from bibr.local.cli import _ShowOnceFilter

        flt = _ShowOnceFilter("unauthenticated requests")
        record = logging.LogRecord(
            "huggingface_hub",
            logging.WARNING,
            __file__,
            1,
            "You are sending %s",
            ("unauthenticated requests",),
            None,
        )

        assert flt.filter(record) is True
        assert flt.filter(record) is False

    def test_the_cli_installs_it_on_its_handlers_and_shows_the_advice_once(
        self, tmp_path, monkeypatch, capsys, cli_logging
    ):
        from bibr.local.cli import _ShowOnceFilter, main

        monkeypatch.setattr(sys, "argv", ["bibr", "batch", "report", str(tmp_path)])
        cli_logging()
        with pytest.raises(SystemExit):
            main()
        capsys.readouterr()
        handlers = logging.getLogger().handlers
        assert handlers
        assert all(any(isinstance(f, _ShowOnceFilter) for f in h.filters) for h in handlers)

        for client in ("huggingface_hub.a", "transformers.b", "wtpsplit.c"):
            logging.getLogger(client).warning(self._ADVICE)
        logging.getLogger("bibr.x").warning("a different warning")
        err = capsys.readouterr().err

        assert err.count("unauthenticated requests") == 1
        assert "a different warning" in err

    def test_the_same_advice_raised_as_a_python_warning_shows_once(
        self, tmp_path, monkeypatch, cli_logging
    ):
        from bibr.local.cli import main

        message = "unauthenticated requests: this text is unique to the warnings test"
        monkeypatch.setattr(sys, "argv", ["bibr", "batch", "report", str(tmp_path)])
        cli_logging()
        with warnings.catch_warnings(record=True) as seen:
            with pytest.raises(SystemExit):
                main()
            warnings.warn(message, UserWarning, stacklevel=1)
            warnings.warn(message, UserWarning, stacklevel=1)

        assert [str(w.message) for w in seen].count(message) == 1
