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
from io import StringIO

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
