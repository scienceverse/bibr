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

from bibr.local.cli import _LIFECYCLE_LOGGERS, _LiveStderrHandler
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
    names = (
        "bibr",
        "bibr.local",
        "bibr.pipeline",
        "bibr.structure",
        "bibr.extract",
        "httpcore",
        "httpx",
        *_LIFECYCLE_LOGGERS,
    )
    root = logging.getLogger()
    saved = (root.level, list(root.handlers), {n: logging.getLogger(n).level for n in names})
    saved_filters = list(warnings.filters)

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
    warnings.filters[:] = saved_filters


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

    def test_the_cli_installs_it_on_its_handlers_and_shows_the_advice_once_under_v(
        self, tmp_path, monkeypatch, capsys, cli_logging
    ):
        from bibr.local.cli import _ShowOnceFilter, main

        monkeypatch.setattr(sys, "argv", ["bibr", "batch", "-v", "report", str(tmp_path)])
        cli_logging()
        with pytest.raises(SystemExit):
            main()
        capsys.readouterr()
        handlers = logging.getLogger().handlers
        assert handlers
        assert all(any(isinstance(f, _ShowOnceFilter) for f in h.filters) for h in handlers)

        # Not huggingface_hub.*: that package attaches a stream handler of its own once imported.
        for client in ("hub_client.a", "transformers.b", "wtpsplit.c"):
            logging.getLogger(client).warning(self._ADVICE)
        logging.getLogger("bibr.x").warning("a different warning")
        err = capsys.readouterr().err

        assert err.count("unauthenticated requests") == 1
        assert "a different warning" in err

    def test_the_same_advice_raised_as_a_python_warning_shows_once_under_v(
        self, tmp_path, monkeypatch, cli_logging
    ):
        from bibr.local.cli import main

        message = "unauthenticated requests: this text is unique to the warnings test"
        monkeypatch.setattr(sys, "argv", ["bibr", "batch", "-v", "report", str(tmp_path)])
        cli_logging()
        with warnings.catch_warnings(record=True) as seen:
            with pytest.raises(SystemExit):
                main()
            warnings.warn(message, UserWarning, stacklevel=1)
            warnings.warn(message, UserWarning, stacklevel=1)

        assert [str(w.message) for w in seen].count(message) == 1

    def test_the_advice_is_silent_by_default_as_a_log_record_and_as_a_python_warning(
        self, tmp_path, monkeypatch, capsys, cli_logging
    ):
        from bibr.local.cli import main

        message = "unauthenticated requests: this text is unique to the silence test"
        monkeypatch.setattr(sys, "argv", ["bibr", "batch", "report", str(tmp_path)])
        cli_logging()
        with warnings.catch_warnings(record=True) as seen:
            with pytest.raises(SystemExit):
                main()
            capsys.readouterr()
            logging.getLogger("hub_client.a").warning(self._ADVICE)
            logging.getLogger("bibr.x").warning("a different warning")
            warnings.warn(message, UserWarning, stacklevel=1)
        err = capsys.readouterr().err

        assert "unauthenticated requests" not in err
        assert "a different warning" in err
        assert [str(w.message) for w in seen].count(message) == 0


class TestConfigureLogging:
    """``_configure_logging``: nothing but problems by default, bibr's own log with -v."""

    def _lines(self, capsys) -> list[str]:
        return capsys.readouterr().err.splitlines()

    def test_default_shows_warnings_and_errors_as_glyph_lines_without_log_decoration(
        self, capsys, cli_logging
    ):
        from bibr.local.cli import _configure_logging

        cli_logging()
        _configure_logging(0)
        logging.getLogger("bibr.pipeline").info("stage decision")
        logging.getLogger("bibr.pipeline").warning("slow disk")
        try:
            raise ValueError("inner detail")
        except ValueError:
            logging.getLogger("bibr.pipeline").error("it broke", exc_info=True)

        lines = self._lines(capsys)

        assert lines == ["  ! slow disk", "  ✗ it broke"]

    def test_default_keeps_the_managed_server_loggers_at_info(self, capsys, cli_logging):
        from bibr.local.cli import _configure_logging

        cli_logging()
        _configure_logging(0)
        for name in _LIFECYCLE_LOGGERS:
            logging.getLogger(name).info("starting server")
            assert logging.getLogger(name).isEnabledFor(logging.INFO)
        logging.getLogger("bibr.pipeline").info("not a server")

        lines = self._lines(capsys)

        assert lines == ["  · starting server"] * len(_LIFECYCLE_LOGGERS)

    def test_default_hides_the_cuda_initialization_warning(self, cli_logging):
        from bibr.local.cli import _configure_logging

        cli_logging()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            _configure_logging(0)
            warnings.warn("CUDA initialization: driver too old", UserWarning, stacklevel=1)
            warnings.warn("something else", UserWarning, stacklevel=1)

        assert [str(w.message) for w in seen] == ["something else"]

    def test_v_shows_bibr_info_in_the_timestamped_logger_level_format(self, capsys, cli_logging):
        import re

        from bibr.local.cli import _configure_logging

        cli_logging()
        _configure_logging(1)
        logging.getLogger("bibr.pipeline").info("stage decision")
        logging.getLogger("bibr.pipeline").debug("frame detail")
        logging.getLogger("elsewhere").info("another library")

        (line,) = self._lines(capsys)

        assert re.fullmatch(r"\d\d:\d\d:\d\d bibr\.pipeline INFO stage decision", line)

    def test_vv_adds_debug_but_not_the_transport_libraries(self, capsys, cli_logging):
        from bibr.local.cli import _configure_logging

        cli_logging()
        _configure_logging(2)
        logging.getLogger("bibr.pipeline").debug("frame detail")
        logging.getLogger("httpcore.http11").debug("send_request_headers")

        (line,) = self._lines(capsys)

        assert line.endswith("bibr.pipeline DEBUG frame detail")

    def test_v_still_shows_a_traceback_logged_with_an_error(self, capsys, cli_logging):
        from bibr.local.cli import _configure_logging

        cli_logging()
        _configure_logging(1)
        try:
            raise ValueError("inner detail")
        except ValueError:
            logging.getLogger("bibr.pipeline").error("it broke", exc_info=True)

        err = capsys.readouterr().err

        assert "ERROR it broke" in err
        assert "ValueError: inner detail" in err

    def test_verbose_counts_on_chew_batch_and_mcp(self):
        from bibr.local.cli import _build_parser

        parser = _build_parser()

        assert parser.parse_args(["chew", "p.pdf"]).verbose == 0
        assert parser.parse_args(["chew", "p.pdf", "-v"]).verbose == 1
        assert parser.parse_args(["chew", "p.pdf", "-vv"]).verbose == 2
        assert parser.parse_args(["chew", "p.pdf", "--verbose", "-v"]).verbose == 2
        assert parser.parse_args(["batch", "papers/", "-vv"]).verbose == 2
        assert parser.parse_args(["mcp", "-v"]).verbose == 1


def test_the_quiet_formatter_drops_a_traceback_and_names_no_logger():
    from bibr.local.cli import _QuietFormatter

    formatter = _QuietFormatter()
    try:
        raise ValueError("x")
    except ValueError:
        import sys as _sys

        record = logging.LogRecord(
            "bibr.pipeline",
            logging.ERROR,
            __file__,
            1,
            "failed: %s",
            ("paper.pdf",),
            _sys.exc_info(),
        )

    assert formatter.format(record) == "  ✗ failed: paper.pdf"


def test_the_drop_filter_drops_matching_records_and_passes_the_rest():
    from bibr.local.cli import _DropFilter

    flt = _DropFilter("unauthenticated requests", "other marker")

    assert flt.filter(_record("You are sending unauthenticated requests")) is False
    assert flt.filter(_record("an other marker here")) is False
    assert flt.filter(_record("downloading model.onnx")) is True
