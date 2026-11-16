"""``bibr tables``: write bibr JSON exports as Parquet tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.markup import escape

from bibr.local.cli import ui

# What ``bibr chew -o`` writes beside each export x.json (``bibr.local.artifacts``):
# its receipt, its enrichment sidecar, its core x.core.json and the quarantined
# exports.
_SIDECAR_SUFFIXES = (".json.receipt.json", ".json.enrichment.json")
_QUARANTINE_DIR = "_quarantine"
_CORE_SUFFIX = ".core.json"
# JSON that bibr writes into a batch output directory beside the exports:
# ``bibr batch``'s run_info.json and the evaluation's validation_report.json.
# Skipped without a warning when they are not exports.
_OWN_NON_EXPORTS = frozenset({"run_info.json", "validation_report.json"})


def _chew_export_of(path: Path) -> Path | None:
    """The export ``bibr chew -o`` wrote *path* beside as one of its sidecars,
    or None when *path* is not such a sidecar."""
    name = path.name
    for suffix in _SIDECAR_SUFFIXES:
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)] + ".json")
    # x.core.json is the core of x.json and repeats its paper_id. The receipt
    # beside x.json tells it from bibr batch's export of a paper named x.core.
    if name.endswith(_CORE_SUFFIX):
        export = path.with_name(name[: -len(_CORE_SUFFIX)] + ".json")
        if export.is_file() and export.with_name(export.name + ".receipt.json").is_file():
            return export
    return None


def export_files(inputs: list[str]) -> tuple[list[Path], list[str]]:
    """JSON files named in *inputs*, and the inputs that do not exist.

    Directories are searched recursively, leaving out what ``bibr chew -o``
    writes beside each export. A file named explicitly is read unless it is
    such a sidecar of an export also read (``bibr tables results/*.json``).
    """
    files: list[Path] = []
    missing: list[str] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(
                sorted(
                    p
                    for p in path.rglob("*.json")
                    if not p.name.startswith(".")
                    and _QUARANTINE_DIR not in p.relative_to(path).parts[:-1]
                    and _chew_export_of(p) is None
                )
            )
        elif path.is_file():
            files.append(path)
        else:
            missing.append(raw)
    files = list(dict.fromkeys(files))
    found = set(files)
    return [f for f in files if _chew_export_of(f) not in found], missing


def papers_given(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """*files* split into the papers among them (a PDF, a DOCX, ...) and the rest."""
    from bibr.input.supported_files import SUPPORTED_EXTENSIONS

    papers = [f for f in files if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    return papers, [f for f in files if f not in papers]


def warn_papers(console: Any, papers: list[Path]) -> None:
    """Say that each of *papers* is a paper, not an export, and how to get one."""
    for paper in papers:
        ui.warn(
            console,
            f"{escape(paper.name)} is a paper, not a bibr export: run "
            f"bibr chew {escape(str(paper))} -o {escape(paper.stem)}.json first "
            f"and pass {escape(paper.stem)}.json",
        )


def warn_skipped(console: Any, skipped: Any) -> None:
    """Warn about each skipped input, except bibr's own files beside the exports."""
    for source, reason in skipped:
        if Path(str(source)).name in _OWN_NON_EXPORTS and str(reason).startswith(
            "not a bibr export"
        ):
            continue
        ui.warn(console, f"skipped {escape(str(source))}: {escape(str(reason))}")


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def report_tables(console: Any, report: Any) -> None:
    """One summary line per written run, plus the skipped inputs."""
    warn_skipped(console, report.skipped)
    table_rows = {name: n for name, n in report.rows.items() if name != "paper"}
    non_empty = sum(1 for n in table_rows.values() if n)
    ui.ok(
        console,
        f"{plural(report.papers, 'paper')} {ui.ARROW} {escape(str(report.out_dir))}/ "
        f"({len(report.files)} Parquet files, {non_empty} with rows)",
    )


def run_tables(args: Any) -> int:
    """Entry point for ``bibr tables``; returns the exit code."""
    from rich.console import Console

    from bibr.export.tables import NoExportsError, write_tables

    console = Console(stderr=True)
    files, missing = export_files(list(args.inputs or []))
    for raw in missing:
        ui.warn(console, f"not found: {escape(raw)}")
    papers, files = papers_given(files)
    warn_papers(console, papers)
    if not files:
        if papers:
            ui.error(console, "No bibr exports to read; nothing written.")
            return 1
        ui.error(console, "No JSON exports found.", hint="bibr tables results/ --out tables/")
        return 2
    out = Path(args.out)
    resolved_out = out.resolve()
    files = [f for f in files if resolved_out not in f.resolve().parents]
    try:
        report = write_tables(files, out, require_exports=True)
    except NoExportsError as exc:
        warn_skipped(console, exc.skipped)
        ui.error(console, "No bibr exports among the inputs; nothing written.")
        return 1
    except ValueError as exc:
        ui.error(console, escape(str(exc).splitlines()[0]))
        return 1
    report_tables(console, report)
    return 0
