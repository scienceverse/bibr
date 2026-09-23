"""``bibr batch`` — thin CLI over :mod:`bibr.batch`.

Turns parsed arguments into :class:`bibr.batch.runner.BatchOptions`: the
local executor reuses ``bibr chew``'s option resolution
(:func:`resolve_run_config`, :func:`_apply_runtime_settings`) so both commands
build the same pipeline from the same flags; the remote executor maps the
subset the serve's job API accepts onto multipart form fields.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bibr.local.cli import ui

# ``bibr chew`` flags that configure a *local* pipeline and mean nothing to a
# remote serve (its own settings decide). Named so the runner can say so.
_LOCAL_ONLY_FLAGS = (
    "ocr",
    "ocr_url",
    "ocr_model",
    "ocr_profile",
    "llm",
    "llm_provider",
    "llm_model",
    "memory",
    "device",
    "no_llm",
    "no_equations",
    "batch_size",
    "preset",
    "region_meta",
)


def chew_options_from_config(config: Any) -> dict[str, Any]:
    """``ResolvedRunConfig`` → :class:`bibr.api.Chewer` kwargs.

    Mirrors the ``LocalPipeline(...)`` construction in ``bibr chew`` field for
    field, through the public option names ``bibr.api`` accepts.
    """
    options: dict[str, Any] = {
        "ocr_backend": config.ocr_backend,
        "llm_backend": config.llm_backend,
        "memory_mode": config.memory_mode,
        "ocr_url": config.ocr_url,
        "ocr_model": config.ocr_model,
        "ocr_profile": config.ocr_profile,
        "device": config.device,
        "crossref": config.crossref,
        "equations": config.equations,
        "no_llm": config.no_llm,
        "figure_images": config.figure_images,
        "include_regions": config.include_regions,
        "include_region_meta": config.include_region_meta,
        "start_page": config.start_page,
        "end_page": config.end_page,
    }
    if config.consolidate:
        options["consolidate"] = config.consolidate
    if config.ref_seg:
        options["ref_seg"] = config.ref_seg
    if config.refs:
        options["refs"] = config.refs
    return options


def _local_summary(config: Any) -> list[str]:
    """Dry-run lines describing the resolved local pipeline (no model loads)."""
    from bibr.extract.ref_extractor import _resolve_ref_strategies
    from bibr.local.cli.dry_run import _dry_run_enrichment_lines, _dry_run_llm_label

    seg, parse = _resolve_ref_strategies(config.ref_seg, config.refs)
    model = config.ocr_model or config.ocr_summary_model or "(resolved at startup)"
    lines = [
        f"ocr {config.ocr_backend} {ui.SEP} {model}",
        f"llm {_dry_run_llm_label(config)}",
        f"refs {'disabled (--no-llm)' if config.no_llm else f'seg {seg} {ui.SEP} parse {parse}'}",
    ]
    enrich = _dry_run_enrichment_lines(config, parse)
    lines.append(f"crossref {enrich[0].split(': ', 1)[-1]}")
    lines.append(f"memory {config.memory_mode}")
    return lines


def _cli_options(args: Any) -> dict[str, Any]:
    """The invocation, minus secrets and positionals, for ``run_info.json``."""
    skip = {"command", "inputs", "token"}
    return {k: v for k, v in sorted(vars(args).items()) if k not in skip}


def _form_fields(args: Any, console: Any) -> dict[str, str] | None:
    """Job-API form fields from the flags the serve understands."""
    form: dict[str, str] = {}
    if args.refs:
        form["refs"] = args.refs
    if args.ref_seg:
        form["ref_seg"] = args.ref_seg
    if args.consolidate:
        form["consolidate"] = args.consolidate
    # The serve honours a per-request enrichment switch; forward an explicit
    # --crossref / --no-crossref and let the serve's CROSSREF_ENRICH decide otherwise.
    if getattr(args, "crossref", False):
        form["crossref"] = "true"
    elif getattr(args, "no_crossref", False):
        form["crossref"] = "false"
    if args.pages:
        from bibr.utils.pages import parse_pages

        try:
            start, end = parse_pages(args.pages)
        except ValueError as exc:
            ui.error(console, f"Invalid --pages value: {exc}")
            return None
        if start is not None:
            form["start_page"] = str(start)
        if end is not None:
            form["end_page"] = str(end)
    if args.figure_images:
        form["include_figures"] = "true"
    if args.regions:
        form["include_regions"] = "true"
    for raw in args.form:
        key, sep, value = raw.partition("=")
        if not sep or not key:
            ui.error(console, f"--form expects K=V, got {raw!r}")
            return None
        form[key.strip()] = value
    return form


def _remote_options(args: Any, console: Any) -> Any:
    from bibr.batch.remote import RemoteOptions, resolve_token

    form = _form_fields(args, console)
    if form is None:
        return None
    ignored = [
        f"--{name.replace('_', '-')}"
        for name in _LOCAL_ONLY_FLAGS
        if getattr(args, name, None) not in (None, False, 0)
    ]
    if ignored:
        ui.warn(
            console,
            "ignored by the remote executor (the serve's own settings apply): "
            + ", ".join(ignored),
        )
    token = resolve_token(args.token)
    if not token:
        ui.warn(
            console,
            "no bearer token — set AUTH_API_KEY / BIBR_SERVE_TOKEN or pass --token "
            "(fine only if the serve runs without auth)",
        )
    return RemoteOptions(
        serve_url=args.serve_url,
        token=token,
        concurrency=args.concurrency,
        min_concurrency=args.min_concurrency,
        max_concurrency=args.max_concurrency,
        poll_timeout=args.poll_timeout,
        poll_interval=args.poll_interval,
        retries=args.retries,
        ready_timeout=args.ready_timeout,
        form=form,
    )


def _local_options(args: Any, console: Any) -> Any:
    """Resolve the local pipeline exactly as ``bibr chew`` does; ``None`` on error."""
    from bibr.batch.runner import LocalOptions
    from bibr.local.cli.run_config import (
        _apply_runtime_settings,
        _preflight_ocr_runtime,
        _preflight_opencv,
        resolve_run_config,
    )

    _apply_runtime_settings(args)
    try:
        config = resolve_run_config(args)
    except ValueError as exc:
        ui.error(console, f"Invalid option: {exc}")
        return None

    def preflight(files: Sequence[Path]) -> str | None:
        if not any(f.suffix.lower() == ".pdf" for f in files):
            return None
        opencv_problem = _preflight_opencv()
        if opencv_problem is not None:
            message, repair = opencv_problem
            return f"{message} (repair with: {repair})"
        return _preflight_ocr_runtime(config)

    return LocalOptions(
        chew_options=chew_options_from_config(config),
        batch_size=config.chunk_size,
        stages=config.active_stages(),
        summary=_local_summary(config),
        preflight=preflight,
    )


def _print_report(out: Path, *, json_output: bool, console: Any) -> int:
    from bibr.batch.ledger import LEDGER_FILENAME, Ledger
    from bibr.batch.report import compute_report, render_report

    ledger = Ledger(out / LEDGER_FILENAME)
    if not ledger.path.is_file():
        ui.error(console, f"No ledger found at {ledger.path}")
        return 2
    report = compute_report(ledger.read())
    if json_output:
        print(json.dumps(report, indent=2))
    else:
        print(render_report(report, title=f"bibr batch report {ui.SEP} {out}"))
    return 0


def _run_batch(args: Any) -> int:
    """Entry point for ``bibr batch``; returns the exit code."""
    from rich.console import Console

    console = Console(stderr=True)
    inputs = list(args.inputs or [])

    report_mode = bool(args.report) or (bool(inputs) and inputs[0] == "report")
    if report_mode:
        if inputs and inputs[0] == "report":
            inputs = inputs[1:]
        out = Path(args.out) if args.out else (Path(inputs[0]) if inputs else None)
        if out is None:
            ui.error(console, "bibr batch report needs the run directory: bibr batch report <out>")
            return 2
        return _print_report(out, json_output=args.json, console=console)

    if not inputs:
        ui.error(
            console,
            "No inputs given.",
            hint="bibr batch <manifest.txt|dir|file>... --out DIR",
        )
        return 2
    if not args.out:
        ui.error(console, "--out DIR is required.")
        return 2

    from bibr.batch.runner import BatchOptions, parse_deadline, run_batch

    deadline = None
    if args.deadline:
        try:
            deadline = parse_deadline(args.deadline)
        except ValueError as exc:
            ui.error(console, str(exc))
            return 2

    options = BatchOptions(
        inputs=inputs,
        out=Path(args.out),
        limit=args.limit,
        shuffle=bool(args.shuffle or args.seed is not None),
        seed=args.seed,
        deadline=deadline,
        retry_failed=args.retry_failed,
        force=args.force,
        dry_run=args.dry_run,
        cli_options=_cli_options(args),
        report_json=args.json,
        tables=not args.no_tables,
    )
    if args.serve_url:
        options.remote = _remote_options(args, console)
        if options.remote is None:
            return 2
    else:
        options.local = _local_options(args, console)
        if options.local is None:
            return 2
    return run_batch(options, console=console)
