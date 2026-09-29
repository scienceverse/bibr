import io

import pytest
from rich.console import Console
from rich.markup import escape

from bibr.local.cli import ui


def _render(emit, **console_options) -> str:
    buffer = io.StringIO()
    emit(Console(file=buffer, width=200, **console_options))
    return buffer.getvalue()


def _render_terminal(emit) -> str:
    # legacy_windows=False: on a Windows host rich otherwise renders for the
    # legacy console API, which drops OSC 8 hyperlinks.
    return _render(
        emit, force_terminal=True, color_system="standard", no_color=False, legacy_windows=False
    )


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


def test_ok_line_keeps_a_package_extra():
    out = _render(lambda c: ui.ok(c, "installed with pip install 'bibr[ml]'"))
    assert out == "  ✓ installed with pip install 'bibr[ml]'\n"


def test_extra_inside_deliberate_markup_is_kept_and_styled():
    hint = "Repair with: [cyan]pip install onnxruntime-gpu[cuda,cudnn][/cyan]"
    out = _render(lambda c: ui.warn(c, "CUDA provider missing", hint=hint))
    assert "Repair with: pip install onnxruntime-gpu[cuda,cudnn]\n" in out
    assert "[cyan]" not in out
    styled = _render_terminal(lambda c: ui.warn(c, "CUDA provider missing", hint=hint))
    assert "\x1b[2;36mpip install onnxruntime-gpu[cuda,cudnn]\x1b[0m" in styled


def test_multi_line_hint_prints_each_line_indented_with_its_brackets():
    hint = "Install with:\npip install 'rapid-mlx[guided]'\nthen run [cyan]bibr doctor[/cyan]"
    out = _render(lambda c: ui.fail(c, "Rapid-MLX missing", hint=hint))
    assert out == (
        "  ✗ Rapid-MLX missing\n"
        "    Install with:\n"
        "    pip install 'rapid-mlx[guided]'\n"
        "    then run bibr doctor\n"
    )


def test_multi_word_and_link_markup_still_render():
    def emit(c):
        ui.fail(
            c,
            "Run [bold red]bibr setup[/bold red] first",
            hint="Read [link=https://example.org/guide]the guide[/link].",
        )

    out = _render(emit)
    assert "Run bibr setup first\n" in out
    assert "Read the guide.\n" in out
    assert "[bold red]" not in out and "[link=" not in out
    styled = _render_terminal(emit)
    assert "\x1b[1;31mbibr setup\x1b[0m" in styled
    assert "\x1b]8;id=" in styled and ";https://example.org/guide\x1b\\" in styled


def test_stray_closing_tag_in_a_message_is_printed_as_written():
    out = _render(lambda c: ui.error(c, "cannot write [/tmp/x]: permission denied"))
    assert "cannot write [/tmp/x]: permission denied\n" in out


def test_stray_closing_tag_in_a_hint_is_printed_as_written():
    out = _render(lambda c: ui.warn(c, "cache unusable", hint="Remove [/tmp/x] and retry."))
    assert "    Remove [/tmp/x] and retry.\n" in out


@pytest.mark.parametrize(
    ("markup", "printed"),
    [
        ("Preset [cyan]a[/cyan]b[/cyan] not found.", "Preset ab[/cyan] not found."),
        ("[bold]x[/bold][/bold]", "x[/bold]"),
        ("[bold]done[/bold] then [/]", "done then [/]"),
        ("nothing open [/]", "nothing open [/]"),
    ],
)
@pytest.mark.parametrize("where", ["message", "hint"])
def test_surplus_closing_tag_is_printed_not_raised(markup, printed, where):
    if where == "message":
        out = _render(lambda c: ui.fail(c, markup))
    else:
        out = _render(lambda c: ui.fail(c, "failed", hint=markup))
    assert f"{printed}\n" in out


def test_closing_tag_matches_an_opening_tag_by_its_normalized_style():
    # Rich closes ``[b]`` with ``[/bold]``: both normalize to "bold".
    out = _render(lambda c: ui.fail(c, "[b]x[/bold] done", hint="[bold red]y[/red bold] done"))
    assert out == "  ✗ x done\n    y done\n"


@pytest.mark.parametrize(
    "text",
    ["use a [@property] here", "bad handler [@click=app.bell(]here[/@click]"],
)
def test_handler_tags_are_printed_as_written(text):
    assert f"{text}\n" in _render(lambda c: ui.fail(c, text))
    assert f"    {text}\n" in _render(lambda c: ui.fail(c, "failed", hint=text))


def test_escaped_hint_ending_in_a_backslash_shows_one_backslash():
    out = _render(lambda c: ui.warn(c, "missing", hint=escape("Put it in C:\\llama\\")))
    assert out.endswith("    Put it in C:\\llama\\\n")


def test_escaped_message_ending_in_a_backslash_shows_one_backslash():
    out = _render(lambda c: ui.fail(c, escape("Cannot read C:\\llama\\")))
    assert out == "  ✗ Cannot read C:\\llama\\\n"
