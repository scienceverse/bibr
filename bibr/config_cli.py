"""``bibr config`` subcommand suite: show / path / set / example.

Config-safety centerpiece for the CLI: lets a user see what bibr will
actually read, where each value comes from, and edit ``.env`` without
hand-rolling ``sed``. Redaction is non-negotiable — every render path funnels
through :func:`format_value`, which masks secret fields, and there is no flag
anywhere in this module to bypass that.

Secrecy is decided by ``SettingDoc.is_secret`` alone (the end-anchored
``(_KEY|_TOKEN|_PASSWORD|_SECRET)$`` regex in ``bibr/config_introspect.py``)
— the single source of truth, so ``LLM_MAX_TOKENS`` (plural, no match) is
never redacted while ``LLM_API_KEY`` always is.
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from dotenv import set_key
from pydantic import TypeAdapter, ValidationError

from bibr.config_introspect import SettingDoc, iter_setting_docs
from bibr.env_utils import read_dotenv
from bibr.local.cli import ui
from bibr.utils.redact import redact_url_secrets

# The 4 settings a new user actually has to touch to get bibr running:
# an LLM provider (defaults to Google/Gemini), an LLM key, an OCR backend,
# and a Crossref contact email (unlocks the "polite pool" rate limit for
# reference enrichment).
_MINIMAL_EXAMPLE_KEYS: tuple[str, ...] = (
    "LLM_PROVIDER",
    "GOOGLE_API_KEY",
    "CROSSREF_API_EMAIL",
    "OCR_BACKEND",
)


class Provenance(NamedTuple):
    """Where a setting's current value comes from, and what it is."""

    value: str | None
    tier: str  # "env" | "dotenv" | "default"
    path: str | None  # populated only for tier == "dotenv"


def _env_chain() -> tuple[Path, ...]:
    """The runtime's configured dotenv files, in highest-first precedence."""
    from bibr.config import _default_env_files

    return tuple(path.absolute() for path in reversed(_default_env_files()))


def resolve_provenance(doc: SettingDoc) -> Provenance:
    """Resolve where *doc*'s current value comes from.

    Process env beats the configured dotenv chain. Files merge key-by-key
    before aliases resolve, matching pydantic-settings. Keys are case-insensitive.
    """
    names = tuple(name.lower() for name in (doc.env_name, *doc.aliases))
    environment = {name.lower(): value for name, value in os.environ.items()}
    for name in names:
        if name in environment:
            return Provenance(environment[name], "env", None)
    merged: dict[str, tuple[str | None, Path]] = {}
    for path in reversed(_env_chain()):
        if not path.exists():
            continue
        merged.update((name.lower(), (value, path)) for name, value in read_dotenv(path).items())
    for name in names:
        if name in merged:
            value, path = merged[name]
            if value is not None:
                return Provenance(value, "dotenv", str(path))
    return Provenance(None, "default", None)


def format_value(doc: SettingDoc, value: str) -> str:
    """Render *value* for display, redacting it if *doc* is a secret field.

    The mask shape (first 4 / last 4 characters, full mask at <= 8 chars) is
    adapted from ``bibr.presets.redact_value`` for this CLI-display context
    rather than reused verbatim: that helper's own secret check is a broad
    substring match that disagrees with the end-anchored rule here (it would
    also flag ``LLM_MAX_TOKENS``). There is no flag anywhere to skip this.
    """
    if not value:
        return value
    if not doc.is_secret:
        # REDIS_URL and friends may carry a password in their user-info.
        return redact_url_secrets(value) if doc.env_name.endswith("_URL") else value
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


def suggest_key(key: str, valid_names: Iterable[str]) -> str | None:
    """Closest known setting name to *key*, or ``None`` if nothing is close."""
    matches = difflib.get_close_matches(key, list(valid_names), n=1)
    return matches[0] if matches else None


def _known_names(docs: list[SettingDoc]) -> dict[str, SettingDoc]:
    lookup: dict[str, SettingDoc] = {}
    for doc in docs:
        lookup[doc.env_name] = doc
        for alias in doc.aliases:
            lookup[alias] = doc
    return lookup


def _default_display(doc: SettingDoc) -> str:
    """Render *doc*'s default as a bare, paste-ready ``.env`` value.

    Secret fields always default to ``None`` (never a real placeholder) —
    rendered as an empty string, same as any other unset default. Booleans
    are lowercased to match ``.env`` convention; computed (factory) defaults
    have no static value worth printing.
    """
    raw = doc.default_repr
    if raw in ("None", "(computed)"):
        return ""
    if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw == "True":
        return "true"
    if raw == "False":
        return "false"
    return raw


def validate_value(doc: SettingDoc, raw: str):
    """Validate *raw* (a plain string) against *doc*'s real pydantic annotation.

    Returns the coerced Python value on success. Raises ``ValueError`` with a
    pydantic-produced message on failure — for ``Literal`` fields this lists
    the allowed choices. Only ``err["msg"]`` from the pydantic error is used
    (never ``str(exc)``, which embeds the raw input value) so an invalid
    value for a secret field can never leak through the error text.
    """
    adapter = TypeAdapter(doc.annotation)
    try:
        if doc.annotation is str:
            return adapter.validate_python(raw)
        return adapter.validate_strings(raw)
    except ValidationError as exc:
        message = "; ".join(err["msg"] for err in exc.errors())
        raise ValueError(message) from exc


def render_env_example(full: bool = False) -> str:
    """Render a ``.env`` template.

    ``full=False`` (default): the 4 settings a new user actually needs to
    touch (see ``_MINIMAL_EXAMPLE_KEYS``), uncommented, with descriptions.
    ``full=True``: every known setting, grouped by section, commented out as
    ``# ENV_NAME=<default>`` with its description above — secret fields
    always default to ``None`` and render bare (``# LLM_API_KEY=``), never a
    fake placeholder value.
    """
    docs = iter_setting_docs()
    if not full:
        by_name = {d.env_name: d for d in docs}
        lines = [
            "# bibr .env — minimal settings to get started.",
            "# Run `bibr config example --full` to see every setting.",
            "",
        ]
        for name in _MINIMAL_EXAMPLE_KEYS:
            doc = by_name.get(name)
            if doc is None:
                continue
            if doc.description:
                lines.append(f"# {doc.description}")
            lines.append(f"{name}={_default_display(doc)}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    by_section: dict[str, list[SettingDoc]] = {}
    for doc in docs:
        by_section.setdefault(doc.section, []).append(doc)
    ordered = [s for s in by_section if s] + [s for s in by_section if not s]
    lines = [
        "# bibr .env — every setting, commented out with its default.",
        "# Uncomment and edit the ones you need; unknown keys are ignored.",
        "",
    ]
    for section in ordered:
        title = section.rstrip("_") or "Top-level"
        lines.append(f"# ==== {title} ====")
        lines.append("")
        for doc in by_section[section]:
            if doc.description:
                lines.append(f"# {doc.description}")
            lines.append(f"# {doc.env_name}={_default_display(doc)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _make_console():
    """A rich Console wide enough not to silently ellipsis-truncate values.

    Absolute ``.env`` paths and long (though still-redacted) values are
    exactly the content this command prints. Rich's default table-cell
    overflow is ``ellipsis``, which under a non-tty stdout (piped output,
    ``capsys`` in tests — both report a conservative 80-column fallback)
    would cut a value off with no visual indication. Real terminals keep
    their real width via ``shutil.get_terminal_size``; only the non-tty
    fallback is widened.
    """
    from rich.console import Console

    width = shutil.get_terminal_size(fallback=(200, 24)).columns
    return Console(width=max(width, 200))


def _format_source(prov: Provenance) -> str:
    if prov.tier == "env":
        return "env"
    if prov.tier == "dotenv":
        return f".env ({prov.path})"
    return "default"


def _cmd_show(args: argparse.Namespace, console) -> int:
    from rich.markup import escape

    show_sources = getattr(args, "sources", False)
    show_all = getattr(args, "all", False)

    rows: list[tuple[SettingDoc, str, Provenance]] = []
    for doc in iter_setting_docs():
        prov = resolve_provenance(doc)
        if not show_all and prov.tier == "default":
            continue
        value = _default_display(doc) if prov.tier == "default" else format_value(doc, prov.value)
        rows.append((doc, value, prov))

    if not rows:
        console.print(
            "[dim]No non-default settings found. Run "
            "`bibr config show --all` to see every setting.[/dim]"
        )
        return 0

    rows.sort(key=lambda r: r[0].env_name)
    table = ui.minimal_table("Key", "Value", *(["Source"] if show_sources else []))
    for doc, value, prov in rows:
        cells = [doc.env_name, escape(value)]
        if show_sources:
            cells.append(escape(_format_source(prov)))
        table.add_row(*cells)
    console.print(table)
    if any(doc.is_secret for doc, _, _ in rows):
        console.print("[dim]Secret values are always redacted.[/dim]")
    return 0


def _cmd_path(console) -> int:
    from rich.markup import escape

    paths = _env_chain()
    if not paths:
        console.print("[dim]Dotenv loading is disabled by BIBR_ENV_FILE.[/dim]")
        return 0
    any_exists = False
    for path in paths:
        exists = path.exists()
        any_exists = any_exists or exists
        marker = "[green]exists[/green]" if exists else "[dim]missing[/dim]"
        console.print(f"  {escape(str(path))} [dim]—[/dim] {marker}")
    if not any_exists:
        console.print(
            f"[dim]none found — `bibr config set` will create {escape(str(paths[0]))}[/dim]"
        )
    return 0


def _cmd_set(args: argparse.Namespace, console) -> int:
    from rich.markup import escape

    key = args.key.strip().upper()
    docs = iter_setting_docs()
    lookup = _known_names(docs)
    doc = lookup.get(key)
    if doc is None:
        suggestion = suggest_key(key, lookup.keys())
        hint = f"Did you mean [cyan]{escape(suggestion)}[/cyan]?" if suggestion else ""
        ui.error(console, f"Unknown setting [cyan]{escape(key)}[/cyan].", hint=hint)
        return 2

    try:
        validate_value(doc, args.value)
    except ValueError as exc:
        ui.error(console, f"Invalid value for [cyan]{escape(key)}[/cyan]: {escape(str(exc))}")
        return 2

    paths = _env_chain()
    if not paths:
        ui.error(
            console,
            "Dotenv loading is disabled by BIBR_ENV_FILE.",
            hint="Set BIBR_ENV_FILE to a file path or unset it before editing configuration.",
        )
        return 2
    target = next((path for path in paths if path.exists()), paths[0])
    set_key(str(target), key, args.value)

    display_value = format_value(doc, args.value)
    ui.ok(
        console,
        f"Set [cyan]{escape(key)}[/cyan]={escape(display_value)} in {escape(str(target))}",
    )
    return 0


def _cmd_example(args: argparse.Namespace, console) -> int:
    full = getattr(args, "full", False)
    console.print(render_env_example(full=full), markup=False, highlight=False)
    return 0


def run_config_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None = None
) -> int:
    """Handle ``bibr config`` subcommands. Returns the process exit code."""
    console = _make_console()
    cmd = getattr(args, "config_command", None)

    if cmd is None:
        # Render the argparse help instead of a hand-rolled usage line so the
        # output matches every other ``bibr X --help`` page (mirrors
        # ``bibr preset`` with no subcommand).
        if parser is not None:
            parser.print_help()
        return 1

    if cmd == "show":
        return _cmd_show(args, console)
    if cmd == "path":
        return _cmd_path(console)
    if cmd == "set":
        return _cmd_set(args, console)
    if cmd == "example":
        return _cmd_example(args, console)

    if parser is not None:
        parser.print_help()
    return 1
