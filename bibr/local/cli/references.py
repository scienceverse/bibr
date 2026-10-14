"""``bibr references``: write the reference lists of bibr JSON exports as
CSL-JSON, BibTeX or RIS."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from rich.markup import escape

from bibr.local.cli import ui

_HINT = "bibr references results/ -o refs.bib"


def run_references(args: Any) -> int:
    """Entry point for ``bibr references``; returns the exit code."""
    from rich.console import Console

    from bibr.export.citations import format_for_path, render_references, write_references
    from bibr.export.tables import _payloads
    from bibr.local.cli.tables import export_files

    console = Console(stderr=True)
    out = args.out if args.out not in (None, "-") else None
    fmt = args.format or (format_for_path(out) if out else None)
    if fmt is None:
        if out:
            ui.error(
                console,
                f"Cannot tell the format from {escape(Path(out).name)}.",
                hint="pass --format csl-json|bibtex|ris, or use a .json, .bib or .ris file",
            )
        else:
            ui.error(console, "Writing to standard output needs --format.", hint=_HINT)
        return 2
    files, missing = export_files(list(args.inputs or []))
    for raw in missing:
        ui.warn(console, f"not found: {escape(raw)}")
    if out:
        # A CSL-JSON file written into a directory of exports is not an input.
        resolved_out = Path(out).resolve()
        files = [f for f in files if f.resolve() != resolved_out]
    if not files:
        ui.error(console, "No JSON exports found.", hint=_HINT)
        return 2
    check_note = not args.no_check_note
    try:
        if out:
            report = write_references(
                files, out, format=fmt, prefer=args.prefer, check_note=check_note
            )
            skipped, count, papers = report.skipped, report.references, report.papers
        else:
            exports, skipped_list = [], []
            for label, payload, reason in _payloads(files):
                if payload is None:
                    skipped_list.append((label, reason or "skipped"))
                else:
                    exports.append(payload)
            text, count = render_references(exports, fmt, prefer=args.prefer, check_note=check_note)
            skipped, papers = tuple(skipped_list), len(exports)
            # Bytes, so RIS's CRLF line endings pass through unchanged.
            sys.stdout.buffer.write(text.encode("utf-8"))
            sys.stdout.flush()
    except (OSError, ValueError) as exc:
        ui.error(console, escape(str(exc).splitlines()[0]))
        return 1
    for source, reason in skipped:
        ui.warn(console, f"skipped {escape(str(source))}: {escape(str(reason))}")
    target = escape(out) if out else "standard output"
    ui.ok(console, f"{count} references from {papers} papers {ui.ARROW} {target} ({fmt})")
    return 0
