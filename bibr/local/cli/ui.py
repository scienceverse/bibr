"""Shared terminal-UI primitives for the bibr CLI.

One design system across every ``bibr`` command:

* a single 🦫 brand accent (no emoji walls),
* semantic status glyphs — ``✓`` ok, ``✗`` error, ``!`` warning,
* dim text for scaffolding (labels, hints), default text for content,
* thin rules to separate phases in long flows,
* borderless tables with a single header rule.

Pure rich — imports nothing from ``bibr`` — so any module (the CLI package,
the setup wizard, ``config_cli``) can use it without circular-import risk.
"""

from __future__ import annotations

import io
import re
import shutil
import sys

from rich import box
from rich.console import Console
from rich.errors import StyleError
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

BRAND = "🦫"
OK = "✓"
FAIL = "✗"
WARN = "!"
ARROW = "→"
SEP = "·"
INDENT = "  "

# Preferred left-to-right / top-to-bottom command order for the main help
# screen: the daily-driver commands first, servers last. Names not listed
# here (future subcommands) append in parser order.
_COMMAND_ORDER = (
    "chew",
    "batch",
    "setup",
    "doctor",
    "inspect",
    "tables",
    "preset",
    "config",
    "serve",
    "demo",
    "mcp",
)


def configure_output_streams() -> None:
    """Keep status glyphs from aborting commands on legacy output encodings."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")


def brand_header(console: Console, title: str, *, subtitle: str | None = None) -> None:
    """Print the brand line used by full-screen commands (doctor, setup, help).

    Renders a blank line, ``🦫 <title>`` in bold, and an optional dim
    subtitle — the only place the 🦫 appears. Prints no trailing blank line;
    callers control spacing with :func:`section` or their own prints.
    """
    console.print()
    console.print(f"{BRAND} [bold]{title}[/bold]")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")


# Status lines routinely carry file paths, URLs and model IDs — single tokens
# longer than the console width. Rich's default word wrap *folds* those
# mid-token, inserting a hard newline into the byte stream, so a path in an
# error message can no longer be copied or grepped once the output is piped or
# captured (a non-tty console defaults to 80 columns). ``soft_wrap`` leaves the
# line intact and lets the terminal do any visual wrapping.
def ok(console: Console, msg: str) -> None:
    """One indented green-check status line."""
    console.print(f"{INDENT}[green]{OK}[/green] {_markup(console, msg)}", soft_wrap=True)


def warn(console: Console, msg: str, *, hint: str = "") -> None:
    """One indented yellow status line, plus an optional dim hint below it."""
    console.print(f"{INDENT}[yellow]{WARN}[/yellow] {_markup(console, msg)}", soft_wrap=True)
    _hint(console, hint)


def fail(console: Console, msg: str, *, hint: str = "") -> None:
    """One indented red-cross status line, plus an optional dim hint below it."""
    console.print(f"{INDENT}[red]{FAIL}[/red] {_markup(console, msg)}", soft_wrap=True)
    _hint(console, hint)


def _hint(console: Console, hint: str) -> None:
    # Dim via the Text style, not a wrapping ``[dim]…[/dim]``: a hint ending in
    # a backslash (a Windows path) would escape the closing tag.
    for line in hint.splitlines():
        text = Text.from_markup(_markup(console, line), style="dim")
        console.print(Text(f"{INDENT}  ") + text, soft_wrap=True)


def _markup(console: Console, text: str) -> str:
    """Caller text made safe to print as the end of a markup line."""
    # ``rich.markup.escape`` doubles a lone trailing backslash so that it cannot
    # escape a tag the caller appends. Nothing follows the text here, so the
    # second backslash would be printed; drop it. Unescaped text that ends in
    # exactly two backslashes therefore shows one.
    if text.endswith("\\\\") and not text.endswith("\\\\\\"):
        text = text[:-1]
    return _literal_brackets(console, text)


# Rich's own tag pattern (``rich.markup.RE_TAGS``): a run of backslashes, then
# ``[`` and a tag body starting with a lowercase letter, ``#``, ``/`` or ``@``.
_MARKUP_TAG = re.compile(r"(\\*)\[([a-z#/@][^[]*?)]")


def _literal_brackets(console: Console, text: str) -> str:
    """Escape bracketed text that Rich would swallow or reject as a tag.

    Status lines mix deliberate markup (``[cyan]bibr setup[/cyan]``) with
    literal text that looks like a tag: ``pip install 'rapid-mlx[guided]'``
    parses as a ``guided`` style that does not exist, and Rich drops it without
    a word, while a stray ``[/tmp/x]`` makes it raise ``MarkupError``. Escaped
    here are opening tags that name no style this console knows, ``@`` handler
    tags (no status line uses them, and bad parameters raise), and closing tags
    with no open tag left to close. Open tags are tracked as a stack and
    popped the way Rich pops them, so ``[bold]x[/bold][/bold]`` keeps the first
    ``[/bold]`` and escapes the second. Real markup still renders, and text a
    caller already passed through ``rich.markup.escape`` is left as it is
    instead of gaining backslashes.

    This is a heuristic with a known limit: literal brackets that happen to
    name a real style or theme entry (``[b]``, ``[s]``, ``[red]``,
    ``[default]``) are still read as markup and vanish from the line. Callers
    that interpolate paths, exception text or other outside strings should
    pass them through ``rich.markup.escape`` at the call site. ``section``,
    ``phase``, ``kv`` and ``brand_header`` do not go through this at all.
    """
    open_tags: list[str] = []

    def replace(match: re.Match[str]) -> str:
        backslashes, tag = match.groups()
        if len(backslashes) % 2:
            return match.group(0)  # already escaped
        name, _, parameters = tag.partition("=")
        if name.startswith("/"):
            keep = _pop_open_tag(open_tags, name[1:].strip())
        else:
            keep = not name.startswith("@") and _is_style(console, name, parameters)
            if keep:
                open_tags.append(Style.normalize(name))
        return match.group(0) if keep else f"{backslashes}\\[{tag}]"

    return _MARKUP_TAG.sub(replace, text)


def _pop_open_tag(open_tags: list[str], name: str) -> bool:
    """Close a tag as ``rich.markup.render`` does; False if nothing matches."""
    if not name:  # ``[/]`` closes the innermost open tag
        if not open_tags:
            return False
        open_tags.pop()
        return True
    name = Style.normalize(name)
    for index in range(len(open_tags) - 1, -1, -1):
        if open_tags[index] == name:
            del open_tags[index]
            return True
    return False


def _is_style(console: Console, name: str, parameters: str) -> bool:
    try:
        console.get_style(f"{name} {parameters}" if parameters else name)
    except StyleError:
        return False
    return True


def error(console: Console, msg: str, *, hint: str = "") -> None:
    """Alias for :func:`fail` used on fatal-error paths (reads better there)."""
    fail(console, msg, hint=hint)


def section(console: Console, title: str) -> None:
    """A blank line followed by a bold section title (content goes below)."""
    console.print()
    console.print(f"[bold]{title}[/bold]")


def phase(console: Console, label: str) -> None:
    """A dim rule + bold label — phase separator for long interactive flows."""
    console.print()
    console.print(Rule(style="dim"))
    console.print(f"[bold]{label}[/bold]")


def step(console: Console, n: int, total: int, title: str) -> None:
    """Wizard step header: a dim rule, then ``n/total · title`` in bold."""
    phase(console, step_label(n, total, title))


def step_label(n: int, total: int, title: str) -> str:
    """The plain-text step label (kept separate so callers can pass it around)."""
    return f"{n}/{total} {SEP} {title}"


def kv(label: str, value: str, *, width: int = 10) -> str:
    """An indented key/value row: dim padded label, default-color value."""
    return f"{INDENT}[dim]{label:<{width}}[/dim] {value}"


def rule(console: Console) -> None:
    """A full-width dim horizontal rule."""
    console.print(Rule(style="dim"))


def minimal_table(*columns: str, **kwargs) -> Table:
    """A borderless table with a single thin header rule and bold headers.

    Pass column names positionally; extra ``rich.table.Table`` kwargs
    (e.g. ``title=``) are forwarded.
    """
    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style="bold",
        show_header=bool(columns),
        **kwargs,
    )
    for name in columns:
        table.add_column(name, overflow="fold")
    return table


def render_main_help(version: str, commands: list[tuple[str, str]]) -> str:
    """Render the designed top-level help screen as a string.

    ``commands`` is ``[(name, short_help), ...]`` collected from the main
    parser's subparsers action. Colors are included only when stdout is a
    terminal, so piped ``bibr --help`` stays clean.
    """
    buffer = io.StringIO()
    width = shutil.get_terminal_size(fallback=(88, 24)).columns
    console = Console(
        file=buffer,
        force_terminal=sys.stdout.isatty(),
        width=max(min(width, 100), 60),
    )

    brand_header(console, f"bibr v{version}", subtitle="scientific paper metadata extraction")

    section(console, "Usage")
    console.print(f"{INDENT}bibr <command> \\[options]")

    section(console, "Commands")
    ordered = sorted(
        commands,
        key=lambda c: _COMMAND_ORDER.index(c[0]) if c[0] in _COMMAND_ORDER else len(_COMMAND_ORDER),
    )
    name_w = max((len(name) for name, _ in ordered), default=0)
    for name, help_text in ordered:
        console.print(f"{INDENT}[cyan]{name:<{name_w}}[/cyan]  {help_text}")

    section(console, "Quick start")
    console.print(f"{INDENT}bibr chew paper.pdf -o result.json")
    console.print(f"{INDENT}bibr setup")

    console.print()
    console.print(
        f"[dim]bibr <command> --help shows command details {SEP} "
        f"bibr --version prints the version[/dim]"
    )
    console.print()

    return buffer.getvalue()
