import io
import sys

from rich.console import Console

from bibr.local.cli import ui


def test_cli_status_does_not_crash_on_a_legacy_output_encoding(monkeypatch):
    output = io.BytesIO()
    stream = io.TextIOWrapper(output, encoding="cp1252")
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", stream)
        ui.configure_output_streams()
        ui.ok(Console(file=stream), "Saved preset")
        stream.flush()
    assert b"Saved preset" in output.getvalue()
    assert b"\\u2713" in output.getvalue()
