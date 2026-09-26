"""Chunked file processing loop for ``bibr chew``."""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bibr.local.cli import ui
from bibr.local.cli.dry_run import _dry_run_blockers, _print_dry_run_plan
from bibr.local.cli.inputs import (
    _find_stem_collisions,
    _prepare_output_path,
    _resolve_single_output_path,
)
from bibr.local.cli.run_config import (
    ResolvedRunConfig,
    _apply_runtime_settings,
    _preflight_local_backend,
    _preflight_ocr_runtime,
    _preflight_opencv,
    resolve_run_config,
)
from bibr.validation import payload_validation

if TYPE_CHECKING:
    from bibr.local.pipeline import LocalPipeline

logger = logging.getLogger(__name__)


def _as_count(value: object) -> int:
    """Coerce a count field to a non-negative int, treating anything
    malformed as 0 — the summary printer must never crash a run whose
    output was already written."""
    if isinstance(value, int) and not isinstance(value, bool):
        return max(value, 0)
    return 0


def _validation_counts(result_json: dict | None) -> tuple[int, int]:
    """Return ``(errors, warnings)`` from *result_json*'s top-level
    ``validation`` block.

    ``(0, 0)`` when the block is absent (the output-validation gate was
    skipped) or malformed — a clean/degraded payload must never be treated
    as having validation findings.
    """
    validation = payload_validation(result_json)
    if validation is None:
        return 0, 0
    return _as_count(validation.get("errors")), _as_count(validation.get("warnings"))


def _format_validation_line(result_json: dict | None) -> str | None:
    """Build the dim validation-summary line for one file's export payload.

    ``None`` when there's nothing to report: the ``validation`` block is
    absent or its errors+warnings count is zero, so a clean payload prints
    nothing extra. Otherwise: the top (by severity) 1-2 issues plus a count of
    the processing warnings in ``extraction.warnings`` (e.g.
    STATEMENT_LEXICAL_FALLBACK). Gate findings are not mirrored there, so every
    entry is a genuine processing warning.
    """
    n_errors, n_warnings = _validation_counts(result_json)
    if n_errors + n_warnings <= 0:
        return None

    validation = payload_validation(result_json) or {}
    raw_issues = validation.get("issues")
    issues = [i for i in raw_issues if isinstance(i, dict)] if isinstance(raw_issues, list) else []
    severity_rank = {"error": 0, "warning": 1}
    top = sorted(issues, key=lambda i: severity_rank.get(i.get("severity"), 2))[:2]
    top_str = "; ".join(f"{i.get('code') or '?'}: {i.get('message') or '?'}" for i in top)

    processing_warnings = (result_json.get("extraction") or {}).get("warnings")
    if not isinstance(processing_warnings, list):
        processing_warnings = []
    extra = len(processing_warnings)

    warnings_part = f"{n_warnings} warnings"
    if extra > 0:
        warnings_part += f" + {extra} processing warnings"

    return f"  ⚠ {n_errors} validation errors, {warnings_part} — top: {top_str}"


def _print_validation_line(console, result_json: dict | None) -> None:
    """Print the dim validation-summary line for *result_json*, if any."""
    line = _format_validation_line(result_json)
    if line:
        console.print(f"[dim]{line}[/dim]")


def _format_run_summary(config: ResolvedRunConfig) -> str:
    """Render the user-facing OCR identity without constructing a runtime."""
    model = config.ocr_model or config.ocr_summary_model or "(resolved at startup)"
    profile = config.ocr_profile or "(resolved at startup)"
    backend = config.ocr_backend
    if backend == "paddle":
        backend = "paddle (automatic; use --dry-run for ordered candidates)"
    return (
        f"  [dim]OCR backend:[/dim] {backend}\n"
        f"  [dim]OCR model:[/dim] {model}\n"
        f"  [dim]OCR profile:[/dim] {profile}\n"
        f"  [dim]llm[/dim] {config.llm_backend} [dim]{ui.SEP}[/dim] "
        f"[dim]memory[/dim] {config.memory_mode}"
    )


def _print_actual_ocr_summary(console, resources) -> None:
    """Report the concrete OCR candidate selected during managed startup."""
    identity = getattr(resources, "ocr_runtime_identity", None)
    if identity is None:
        return
    console.print(f"  [dim]OCR backend:[/dim] {identity.backend}")
    console.print(f"  [dim]OCR model:[/dim] {identity.model}")
    console.print(f"  [dim]OCR profile:[/dim] {identity.profile}")
    fallback_reason = getattr(resources, "ocr_fallback_reason", None)
    if fallback_reason:
        console.print(f"  [dim]Fallback reason:[/dim] {fallback_reason}")


def _write_stdout_json(json_str: str) -> None:
    """Write the export payload to stdout as UTF-8 bytes.

    ``main()`` configures stdout with ``errors=\"backslashreplace\"`` so
    status glyphs survive legacy consoles — but that turns unencodable
    characters (e.g. math italic U+1D465) into ``\\U0001d465``/``\\x81``
    escapes that are not legal JSON. Bypass the text wrapper and emit
    UTF-8 (the JSON interchange encoding) directly; fall back to
    ``print()`` when stdout has no buffer (captured StringIO in tests).
    """
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(json_str.encode("utf-8") + b"\n")
        buffer.flush()
    else:
        print(json_str)


def _hint_for_file_error(fs) -> str:
    """Hint for a failed file, keyed by its structured ``error_code``.

    The pipeline sets ``fs.error_code`` (``unsupported_format``,
    ``encrypted_file``, ``ocr_failed``, ``llm_server_failed``, …) while
    the human message varies — substring matching on it misfires
    (``'api'`` matches ``'rapid-mlx'`` and any path containing ``'api'``)
    and misses (``'Unsupported file format'`` contains no
    ``'unsupported format'``). A state without a code falls back to the
    pipeline's real message wordings, never to a bare ``'api'`` substring.
    """
    code = getattr(fs, "error_code", None)
    if code == "unsupported_format":
        from bibr.input.supported_files import SupportedFileType

        formats = ", ".join(file_type.value for file_type in SupportedFileType)
        return f"\n         [dim]bibr accepts {formats} files[/dim]"
    if code == "encrypted_file":
        return "\n         [dim]Remove the password and try again[/dim]"
    if code in ("ocr_failed", "ocr_mostly_failed"):
        return "\n         [dim]Check your OCR backend with: bibr doctor[/dim]"
    if code == "llm_server_failed":
        return "\n         [dim]Check your LLM backend with: bibr doctor[/dim]"
    if code is not None:
        return ""
    err_lower = str(fs.error or "").lower()
    if "unsupported file format" in err_lower:
        from bibr.input.supported_files import SupportedFileType

        formats = ", ".join(file_type.value for file_type in SupportedFileType)
        return f"\n         [dim]bibr accepts {formats} files[/dim]"
    if "password-protected" in err_lower or "password" in err_lower:
        return "\n         [dim]Remove the password and try again[/dim]"
    if "ocr backend init failed" in err_lower or "ocr failed" in err_lower:
        return "\n         [dim]Check your OCR backend with: bibr doctor[/dim]"
    if "llm server start failed" in err_lower:
        return "\n         [dim]Check your LLM backend with: bibr doctor[/dim]"
    if "api key" in err_lower:
        return "\n         [dim]Check your API keys with: bibr doctor[/dim]"
    return ""


def _write_chunk_results(
    file_states: list,
    *,
    output_path: Path | None,
    json_kwargs: dict,
    console,
    is_batch: bool,
    total_files: int,  # noqa: ARG001 — kept for symmetry with caller's accounting
    total_t0: float,
) -> tuple[int, int]:
    """Write JSON output and report per-file status.

    Returns ``(processed, errors)`` counts for the chunk.
    """
    processed = 0
    errors = 0

    from bibr.local.artifacts import atomic_write_json

    for fs in file_states:
        if fs.error:
            err_msg = str(fs.error)
            hint = _hint_for_file_error(fs)
            console.print(f"  [red]✗ {fs.path.name}:[/red] {err_msg}{hint}")
            errors += 1
            continue

        processed += 1
        json_str = json.dumps(fs.result_json, **json_kwargs)

        sink = fs.artifact_sink
        sink_path = sink.destination_path(fs) if sink is not None else None

        if sink_path is not None and fs.core_sha256 is not None:
            # The sink owns checkpoint/final materialization and receipts.
            # Rewriting here would allow unverified in-memory state to bypass
            # that protocol, so the CLI is reporting-only for sink-bound files.
            if is_batch or fs.manifest_output_path is not None:
                console.print(f"  [green]✓ {fs.path.name}[/green] → {sink_path}")
            else:
                total_elapsed = time.monotonic() - total_t0
                console.print(f"  [green]✓[/green] Wrote {sink_path} ({total_elapsed:.1f}s)")
            _print_validation_line(console, fs.result_json)
            continue

        if fs.manifest_output_path is not None:
            out_file = sink_path or fs.manifest_output_path
            atomic_write_json(out_file, fs.result_json, **json_kwargs)
            console.print(f"  [green]✓ {fs.path.name}[/green] → {out_file}")
            _print_validation_line(console, fs.result_json)
        elif output_path is None:
            _write_stdout_json(json_str)
            if not is_batch:
                total_elapsed = time.monotonic() - total_t0
                console.print(f"  [green]✓[/green] Done ({total_elapsed:.1f}s)")
                _print_validation_line(console, fs.result_json)
        elif is_batch:
            out_file = sink_path or output_path / f"{fs.path.stem}.json"
            atomic_write_json(out_file, fs.result_json, **json_kwargs)
            console.print(f"  [green]✓ {fs.path.name}[/green] → {out_file}")
            _print_validation_line(console, fs.result_json)
        else:
            out_file = sink_path or _resolve_single_output_path(output_path, fs.path)
            atomic_write_json(out_file, fs.result_json, **json_kwargs)
            total_elapsed = time.monotonic() - total_t0
            console.print(f"  [green]✓[/green] Wrote {out_file} ({total_elapsed:.1f}s)")
            _print_validation_line(console, fs.result_json)

    return processed, errors


@dataclass
class ChunkProcessor:
    """Run a single chunk of files through the pipeline.

    Holds the shared pipeline + paper_id binding so the chunk loop can
    iterate slices of the file list without re-passing those args.
    """

    pipeline: "LocalPipeline"
    paper_id: str | None
    is_batch: bool
    active_stages: list[str]
    output_path: Path | None = None
    json_kwargs: dict | None = None

    async def run(
        self,
        chunk_files: list,
        *,
        chunk_index: int,  # noqa: ARG002 — present for caller-side logging symmetry
        total_chunks: int,
        console,
    ) -> list:
        """Build FileStates, run the pipeline, return the resulting states."""
        from bibr.local.artifacts import LocalArtifactSink
        from bibr.local.manifest import ManifestRecord
        from bibr.pipeline.artifacts import RunState
        from bibr.pipeline.progress import RichProgress, stages_for_files
        from bibr.pipeline.state import FileState

        file_states = []
        for item in chunk_files:
            if isinstance(item, ManifestRecord):
                manifest_record = item
                fp = item.input_path
            else:
                manifest_record = None
                fp = item
            state = FileState(
                path=fp,
                paper_id=self.paper_id if not self.is_batch else None,
                expected_identity=(
                    manifest_record.expected_identity if manifest_record is not None else None
                ),
                manifest_output_path=(
                    manifest_record.output_path if manifest_record is not None else None
                ),
                content_sha256=(
                    manifest_record.content_sha256 if manifest_record is not None else None
                ),
            )
            requested_output = None
            if manifest_record is not None:
                requested_output = manifest_record.output_path
            elif self.output_path is not None:
                requested_output = (
                    self.output_path / f"{fp.stem}.json"
                    if self.is_batch
                    else _resolve_single_output_path(self.output_path, fp)
                )
            if requested_output is not None:
                state.artifact_sink = LocalArtifactSink(
                    requested_output,
                    json_kwargs=self.json_kwargs,
                )
                state.artifact_sink.record(state, RunState.STARTED)
                state.artifact_started = True
            file_states.append(state)

        chunk_t0 = time.monotonic()
        progress = RichProgress(
            stages=stages_for_files(self.active_stages, (state.path for state in file_states))
        )
        try:
            await self.pipeline.process_chunk(file_states, progress=progress)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit) as exc:
            for fs in file_states:
                if fs.artifact_sink is not None:
                    try:
                        fs.artifact_sink.record(
                            fs, RunState.CUTOFF_INTERRUPTED, detail=type(exc).__name__
                        )
                    except Exception:  # noqa: BLE001 - preserve interruption
                        logger.warning("Failed to persist cutoff receipt", exc_info=True)
            raise
        except BaseException as exc:
            for fs in file_states:
                if fs.artifact_sink is not None:
                    try:
                        fs.artifact_sink.record(fs, RunState.FAILED, detail=str(exc))
                    except Exception:  # noqa: BLE001 - preserve originating failure
                        logger.warning("Failed to persist run failure receipt", exc_info=True)
            raise
        for fs in file_states:
            if fs.error and fs.artifact_sink is not None:
                try:
                    fs.artifact_sink.record(fs, RunState.FAILED, detail=str(fs.error))
                except Exception:  # noqa: BLE001 - processing error remains authoritative
                    logger.warning("Failed to persist file failure receipt", exc_info=True)
        chunk_elapsed = time.monotonic() - chunk_t0

        if self.is_batch and total_chunks > 1:
            console.print(f"  [dim]Chunk done in {chunk_elapsed:.1f}s[/dim]")

        return file_states


async def _run_process(args) -> None:
    """Run the process command."""
    from rich.console import Console

    # Imported from the package root (not from the ``inputs`` submodule
    # directly) so that ``monkeypatch.setattr("bibr.local.cli._collect_files",
    # ...)`` (used by existing tests) still takes effect — the patch sets an
    # attribute on the package, and only a lookup through the package at call
    # time observes it. ``_preflight_opencv`` resolves
    # ``_opencv_unavailable_reason`` the same way.
    from bibr.local.cli import _collect_files

    console = Console(stderr=True)

    from bibr.local.manifest import ManifestError, load_manifest, resolve_cli_source_mode

    try:
        source_mode = resolve_cli_source_mode(args.input, args.manifest)
        if source_mode == "manifest" and args.output is not None:
            raise ManifestError(
                "--manifest cannot be combined with -o/--output; "
                "use each manifest row's output_path"
            )
        manifest_records = load_manifest(args.manifest) if source_mode == "manifest" else []
    except ManifestError as exc:
        ui.error(console, str(exc))
        sys.exit(2)

    _apply_runtime_settings(args)

    try:
        config = resolve_run_config(args)
    except ValueError as e:
        ui.error(console, f"Invalid option: {e}")
        sys.exit(2)

    console.print(_format_run_summary(config))

    # Fail fast on missing LLM credentials — the provider would otherwise
    # raise only on the first LLM call, after the user already sat through OCR.
    # Skipped under --dry-run: it constructs a real provider client (no
    # network call, but --dry-run previews the plan without touching any
    # client/model machinery at all — see ``_print_dry_run_plan``).
    if not args.dry_run and not config.no_llm and config.llm_backend == "cloud":
        try:
            from bibr.clients.llm import preflight_credentials

            preflight_credentials()
        except ValueError as e:
            ui.error(
                console,
                str(e),
                hint="Run [cyan]bibr setup[/cyan] to configure an LLM provider, "
                "or pass [cyan]--no-llm[/cyan] for structural output only.",
            )
            sys.exit(1)

    # Fail fast on a managed local backend that can't run here (unsupported
    # hardware or a missing launcher) — otherwise the run crashes deep in vLLM
    # after OCR. Skipped under --no-llm (no server is started) and --dry-run
    # (nothing is started).
    from bibr.local.pipeline import LOCAL_LLM_BACKENDS

    if not args.dry_run and not config.no_llm and config.llm_backend in LOCAL_LLM_BACKENDS:
        err = _preflight_local_backend(config.llm_backend)
        if err:
            ui.error(console, err)
            sys.exit(1)

    # Collect files
    if source_mode == "manifest":
        files = [record.input_path for record in manifest_records]
        work_items = list(manifest_records)
        missing_count = 0
    else:
        files, missing_count = _collect_files(args.input)
        work_items = list(files)
    is_batch = len(files) > 1

    # --paper-id only makes sense when writing a single result — silently
    # discarding it for batch input used to hide the mistake entirely.
    if is_batch and args.paper_id:
        ui.error(
            console, f"--paper-id is only valid for single-file input (got {len(files)} files)."
        )
        sys.exit(2)

    # Batch output writes <dir>/<stem>.json — files from different
    # directories sharing a stem would silently overwrite each other.
    if is_batch and source_mode != "manifest":
        collisions = _find_stem_collisions(files)
        if collisions:
            ui.error(
                console,
                "Output filename collision: multiple input files "
                "would write to the same <stem>.json:",
            )
            # Plain print (not console.print) for the paths themselves —
            # Rich hard-wraps long lines at the terminal width with no
            # regard for word boundaries, which can split a long absolute
            # path mid-character.
            for stem, paths in sorted(collisions.items()):
                print(f"  {stem}.json would be written by:", file=sys.stderr)
                for p in paths:
                    print(f"    {p}", file=sys.stderr)
            console.print(
                "[dim]Rename the colliding files, or process them in separate "
                "runs with different -o targets.[/dim]"
            )
            sys.exit(2)

    if not args.dry_run and any(p.suffix.lower() == ".pdf" for p in files):
        opencv_problem = _preflight_opencv()
        if opencv_problem is not None:
            message, repair = opencv_problem
            ui.error(console, message, hint=f"Repair with: [cyan]{repair}[/cyan]")
            sys.exit(1)
        # Fail fast when no local OCR runtime can start here (no suitable GPU
        # for paddle-vllm, no llama-server on PATH) — otherwise the run loads
        # the layout model and, for paddle-vllm, bootstraps vLLM before the
        # transactional chain reports the same thing.
        ocr_reason = _preflight_ocr_runtime(config)
        if ocr_reason is not None:
            ui.error(console, ocr_reason)
            sys.exit(1)

    # --dry-run: print the fully-resolved plan and exit — no pipeline is
    # constructed, no model/backend/HTTP client is touched. Placed after the
    # guards above so a colliding batch still hard-errors under --dry-run
    # (the same collision output — that's the intended reuse) instead of
    # previewing a plan for files that would never write successfully.
    # The cheap preflights run here too and print as a Blockers section:
    # without them the preview exits 0 for runs that fail immediately.
    if args.dry_run:
        blockers = _dry_run_blockers(config, files, missing_count)
        _print_dry_run_plan(
            args,
            config,
            files,
            is_batch=is_batch,
            manifest_outputs=[record.output_path for record in manifest_records]
            if source_mode == "manifest"
            else None,
            blockers=blockers or None,
        )
        if blockers:
            sys.exit(1)
        return

    # Resolve -o before the pipeline loads: a path blocked by an existing
    # file used to crash after model loads, outside the aclose() cleanup.
    if source_mode == "manifest":
        output_path = None
    else:
        try:
            output_path = _prepare_output_path(args.output, is_batch=is_batch)
        except OSError as exc:
            ui.error(console, f"Cannot write output {args.output!r}: {exc}")
            sys.exit(2)

    # Create pipeline
    from bibr.local.pipeline import LocalPipeline

    pipeline = LocalPipeline(
        memory_mode=config.memory_mode,
        ocr_backend=config.ocr_backend,
        ocr_url=config.ocr_url,
        ocr_model=config.ocr_model,
        ocr_profile=config.ocr_profile,
        device=config.device,
        crossref=config.crossref,
        equations=config.equations,
        start_page=config.start_page,
        end_page=config.end_page,
        llm_backend=config.llm_backend,
        no_llm=config.no_llm,
        figure_images=config.figure_images,
        include_regions=config.include_regions,
        include_region_meta=config.include_region_meta,
        consolidate=config.consolidate,
        ref_seg_strategy=config.ref_seg,
        ref_parse_strategy=config.refs,
    )

    chunk_size = min(config.chunk_size, len(files))

    if is_batch:
        console.print(f"  [dim]{len(files)} files {ui.SEP} batch size {chunk_size}[/dim]")

    active_stages = config.active_stages(files)

    if args.compact:
        json_kwargs = {"separators": (",", ":"), "ensure_ascii": False}
    else:
        json_kwargs = {"indent": 2, "ensure_ascii": False}

    total_t0 = time.monotonic()
    processed = 0
    errors = missing_count
    # Batch-summary validation breakdown, tallied per successfully-processed
    # file (files that failed processing entirely have no result_json and
    # are excluded — they're already reflected in ``errors`` above).
    validation_clean = 0
    validation_with_warnings = 0
    validation_with_errors = 0

    chunk_processor = ChunkProcessor(
        pipeline=pipeline,
        paper_id=args.paper_id,
        is_batch=is_batch,
        active_stages=active_stages,
        output_path=output_path,
        json_kwargs=json_kwargs,
    )

    llm_usage: dict[str, dict[str, int]] = {}
    ocr_resources = None
    try:
        for chunk_start in range(0, len(work_items), chunk_size):
            chunk_items = work_items[chunk_start : chunk_start + chunk_size]
            chunk_num = chunk_start // chunk_size + 1
            total_chunks = (len(work_items) + chunk_size - 1) // chunk_size

            if is_batch and total_chunks > 1:
                console.print(f"\n[bold]Chunk {chunk_num}/{total_chunks}[/bold]")

            for i, item in enumerate(chunk_items):
                fp = item.input_path if source_mode == "manifest" else item
                file_num = chunk_start + i + 1
                console.print(f"  [dim]\\[{file_num}/{len(files)}][/dim] {fp.name}")

            file_states = await chunk_processor.run(
                chunk_items,
                chunk_index=chunk_num,
                total_chunks=total_chunks,
                console=console,
            )

            ch_processed, ch_errors = _write_chunk_results(
                file_states,
                output_path=output_path,
                json_kwargs=json_kwargs,
                console=console,
                is_batch=is_batch,
                total_files=len(files),
                total_t0=total_t0,
            )
            processed += ch_processed
            errors += ch_errors

            for fs in file_states:
                if fs.error:
                    continue
                n_val_errors, n_val_warnings = _validation_counts(fs.result_json)
                if n_val_errors > 0:
                    validation_with_errors += 1
                elif n_val_warnings > 0:
                    validation_with_warnings += 1
                else:
                    validation_clean += 1

            # OOM safeguard: explicit GC between chunks
            import gc

            gc.collect()
    finally:
        # Snapshot LLM token usage before teardown — close() clears it.
        llm_usage = pipeline.llm_usage_snapshot()
        ocr_resources = getattr(pipeline, "_resources", None)
        # Pipeline-lifetime teardown: release the local LLM server / OCR
        # engine exactly once after all chunks are done (or on error).
        await pipeline.aclose()

    _print_actual_ocr_summary(console, ocr_resources)

    for model, counts in llm_usage.items():
        if counts.get("total_tokens", 0) <= 0:
            continue
        console.print(
            f"  [dim]LLM tokens · {model}: "
            f"in={counts.get('input_tokens', 0):,} "
            f"out={counts.get('output_tokens', 0):,} "
            f"total={counts.get('total_tokens', 0):,}[/dim]"
        )

    total_elapsed = time.monotonic() - total_t0
    if is_batch:
        if errors == 0:
            console.print(
                f"\n  [green]✓[/green] [bold]{processed}/{len(files)} files[/bold]"
                f" processed in {total_elapsed:.1f}s"
            )
        else:
            console.print(
                f"\n  [yellow]![/yellow] [bold]{processed}/{len(files)} files[/bold]"
                f" processed, [red]{errors} failed[/red],"
                f" in {total_elapsed:.1f}s"
            )
        console.print(
            f"  [dim]{validation_clean} files clean, {validation_with_warnings} with "
            f"warnings, {validation_with_errors} with errors[/dim]"
        )

    if errors:
        sys.exit(1)
