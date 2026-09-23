"""``bibr tables``: write bibr JSON exports as Parquet tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bibr.local.cli import ui


def export_files(inputs: list[str]) -> tuple[list[Path], list[str]]:
    """JSON files named in *inputs* (directories are searched recursively),
    and the inputs that do not exist."""
    files: list[Path] = []
    missing: list[str] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.json") if not p.name.startswith(".")))
        elif path.is_file():
            files.append(path)
        else:
            missing.append(raw)
    return list(dict.fromkeys(files)), missing


def report_tables(console: Any, report: Any) -> None:
    """One summary line per written run, plus the skipped inputs."""
    for source, reason in report.skipped:
        ui.warn(console, f"skipped {source}: {reason}")
    table_rows = {name: n for name, n in report.rows.items() if name != "paper"}
    non_empty = sum(1 for n in table_rows.values() if n)
    ui.ok(
        console,
        f"{report.papers} papers {ui.ARROW} {report.out_dir}/ "
        f"({len(report.files)} Parquet files, {non_empty} with rows)",
    )


def run_tables(args: Any) -> int:
    """Entry point for ``bibr tables``; returns the exit code."""
    from rich.console import Console

    from bibr.export.tables import write_tables

    console = Console(stderr=True)
    files, missing = export_files(list(args.inputs or []))
    for raw in missing:
        ui.warn(console, f"not found: {raw}")
    if not files:
        ui.error(console, "No JSON exports found.", hint="bibr tables results/ --out tables/")
        return 2
    out = Path(args.out)
    resolved_out = out.resolve()
    files = [f for f in files if resolved_out not in f.resolve().parents]
    try:
        report = write_tables(files, out)
    except ValueError as exc:
        ui.error(console, str(exc).splitlines()[0])
        return 1
    report_tables(console, report)
    return 0
