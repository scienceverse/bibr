import io

from rich.console import Console
from rich.markup import escape

from bibr.local.cli import ui


def _render(emit) -> str:
    buffer = io.StringIO()
    emit(Console(file=buffer, width=200))
    return buffer.getvalue()


def test_hint_keeps_a_package_extra_that_looks_like_a_markup_tag():
    out = _render(
        lambda c: ui.warn(
            c, "Rapid-MLX missing", hint="Install with: pip install 'rapid-mlx[guided]'"
        )
    )
    assert "pip install 'rapid-mlx[guided]'" in out


def test_status_message_keeps_a_package_extra():
    out = _render(lambda c: ui.fail(c, "launcher not available; pip install 'rapid-mlx[guided]'"))
    assert "pip install 'rapid-mlx[guided]'" in out


def test_hint_markup_still_renders():
    out = _render(
        lambda c: ui.error(c, "No .env file found.", hint="Run [cyan]bibr setup[/cyan] first.")
    )
    assert "Run bibr setup first." in out
    assert "[cyan]" not in out


def test_hint_already_escaped_by_the_caller_shows_no_backslash():
    out = _render(lambda c: ui.fail(c, "missing", hint=escape("Reinstall: pip install 'bibr[ml]'")))
    assert "Reinstall: pip install 'bibr[ml]'" in out
    assert "\\" not in out


def test_hint_ending_in_a_backslash_does_not_escape_its_styling():
    out = _render(lambda c: ui.warn(c, "missing", hint="Put it in C:\\llama\\"))
    assert "Put it in C:\\llama\\\n" in out
    assert "[/dim]" not in out
