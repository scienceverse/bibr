"""Chunked file processing loop for ``bibr chew``."""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markup import escape

from bibr.local.cli import ui
from bibr.local.cli.dry_run import _dry_run_blockers, _print_dry_run_plan, announce_downloads
from bibr.local.cli.inputs import (
    _find_stem_collisions,
    _output_path_problem,
    _prepare_output_path,
    _resolve_single_output_path,
    sample_paper_path,
)
from bibr.local.cli.run_config import (
    ResolvedRunConfig,
    _apply_runtime_settings,
    _preflight_front_tagger,
    _preflight_local_backend,
    _preflight_model_overrides,
    _preflight_ocr_runtime,
    _preflight_opencv,
    apply_text_layer_fallback,
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


# What a validation code is about, in the reader's words, by the code's leading
# words: a new code in a known family needs no entry here.
_VALIDATION_SUBJECTS = (
    ("VAL_AUTHOR", "authors"),
    ("VAL_AFFILIATION", "affiliations"),
    ("VAL_TITLE", "title"),
    ("VAL_ABSTRACT", "abstract"),
    ("VAL_DOI", "DOI"),
    ("VAL_PUBLICATION_DATE", "publication date"),
    ("VAL_CORE_METADATA", "metadata"),
    ("VAL_METADATA", "metadata"),
    ("VAL_REF", "references"),
    ("VAL_DANGLING_REF", "references"),
    ("VAL_XREF", "in-text citations"),
    ("VAL_CAPTION", "captions"),
    ("VAL_PANEL_CAPTION", "captions"),
    ("VAL_STATEMENT", "statements"),
    ("VAL_EMPTY_EQ", "equations"),
    ("VAL_URL", "links"),
)

# Processing notes every run of a mode carries: saying so on each paper is noise.
_ROUTINE_NOTES = frozenset({"LLM_SKIPPED"})


def validation_subject(code: object) -> str | None:
    """The part of the paper a validation *code* is about, or ``None`` if unknown."""
    text = str(code or "")
    for prefix, subject in _VALIDATION_SUBJECTS:
        if text.startswith(prefix):
            return subject
    return None


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def validation_issues(result_json: dict | None) -> list[dict]:
    """The export's validation issues, errors first, tolerant of malformed payloads."""
    validation = payload_validation(result_json) or {}
    raw_issues = validation.get("issues")
    issues = [i for i in raw_issues if isinstance(i, dict)] if isinstance(raw_issues, list) else []
    severity_rank = {"error": 0, "warning": 1}
    return sorted(issues, key=lambda i: severity_rank.get(i.get("severity"), 2))


def processing_notes(result_json: dict | None, *, routine: bool = True) -> list[dict]:
    """``extraction.warnings`` entries; *routine* False leaves out the every-run ones."""
    extraction = (result_json or {}).get("extraction")
    raw = extraction.get("warnings") if isinstance(extraction, dict) else None
    notes = [n for n in raw if isinstance(n, dict)] if isinstance(raw, list) else []
    if routine:
        return notes
    return [n for n in notes if n.get("code") not in _ROUTINE_NOTES]


def _format_validation_line(result_json: dict | None) -> str | None:
    """One line saying how much of *result_json* is worth checking, and where.

    ``None`` when there's nothing to report: the ``validation`` block is
    absent or its errors+warnings count is zero, so a clean payload prints
    nothing extra. Otherwise the counts, the parts of the paper they concern
    (``authors``, ``references``, ...) and how many processing notes
    ``extraction.warnings`` holds beyond the ones every run of the mode carries.
    The codes and full messages are ``bibr inspect``'s job.
    """
    n_errors, n_warnings = _validation_counts(result_json)
    if n_errors + n_warnings <= 0:
        return None

    subjects: list[str] = []
    for issue in validation_issues(result_json):
        subject = validation_subject(issue.get("code"))
        if subject and subject not in subjects:
            subjects.append(subject)

    counts = []
    if n_errors:
        counts.append(_plural(n_errors, "error"))
    if n_warnings:
        counts.append(_plural(n_warnings, "warning"))
    line = f"{' and '.join(counts)} to review"
    if subjects:
        line += f" ({', '.join(subjects[:3])}{', ...' if len(subjects) > 3 else ''})"

    extra = len(processing_notes(result_json, routine=False))
    if extra > 0:
        line += f" {ui.SEP} {_plural(extra, 'processing note')}"
    return line


def _print_validation_line(console, result_json: dict | None, export_path=None) -> None:
    """Print the review line for *result_json*, if any, under its file's status line.

    With *export_path* (a result written to a file) it ends with the command
    that lists every finding.
    """
    line = _format_validation_line(result_json)
    if not line:
        return
    if export_path is not None:
        line += f" {ui.SEP} details: bibr inspect {export_path}"
    n_errors, _ = _validation_counts(result_json)
    glyph = f"[yellow]{ui.WARN}[/yellow] " if n_errors else ""
    console.print(f"    {glyph}[dim]{escape(line)}[/dim]", soft_wrap=True)


def _format_run_summary(config: ResolvedRunConfig) -> str:
    """The line(s) a run opens with: its mode and OCR route, without building a runtime."""
    lines = [_format_mode_line(config)]
    if config.ocr_backend != "text-layer":
        model = config.ocr_model or config.ocr_summary_model or "resolved at startup"
        profile = config.ocr_profile or "resolved at startup"
        lines.append(
            f"  [dim]OCR model[/dim] {escape(str(model))} [dim]{ui.SEP} profile[/dim] "
            f"{escape(str(profile))}"
        )
    return "\n".join(lines)


def _ocr_route(config: ResolvedRunConfig) -> str:
    """The OCR route in one phrase: the backend, or the first of the automatic chain."""
    if config.ocr_url:
        return f"{config.ocr_backend} at {config.ocr_url}"
    if config.ocr_backend == "paddle" and config.ocr_summary_backend:
        return f"{config.ocr_summary_backend} (automatic)"
    if config.ocr_backend == "paddle":
        return "paddle (automatic)"
    if config.ocr_backend == "text-layer":
        if config.ocr_text_layer_fallback:
            return "text-layer (no OCR backend can start here; reads the PDF's own text)"
        return "text-layer (no OCR model; reads the PDF's own text)"
    return str(config.ocr_backend)


def _format_mode_line(config: ResolvedRunConfig) -> str:
    """One line: the run mode, the OCR route and what the mode leaves out."""
    sep = f" [dim]{ui.SEP}[/dim] "
    route = f"[dim]OCR[/dim] {escape(_ocr_route(config))}"
    if config.no_llm:
        head = "[dim]mode[/dim] structure only (--no-llm)"
        tail = "skips authors, references, equations and Crossref"
    elif config.mode == "fast":
        head = "[dim]mode[/dim] fast"
        tail = "local models only, no LLM call"
    else:
        return (
            f"  [dim]mode[/dim] full{sep}{route}{sep}[dim]llm[/dim] "
            f"{escape(str(config.llm_backend))}{sep}[dim]memory[/dim] "
            f"{escape(str(config.memory_mode))}"
        )
    return f"  {head}{sep}{escape(tail)}{sep}{route}"


def _print_actual_ocr_summary(console, resources) -> None:
    """Report the concrete OCR candidate selected during managed startup."""
    identity = getattr(resources, "ocr_runtime_identity", None)
    if identity is None:
        return
    console.print(
        f"  [dim]OCR used[/dim] {escape(str(identity.backend))} [dim]{ui.SEP} model[/dim] "
        f"{escape(str(identity.model))} [dim]{ui.SEP} profile[/dim] "
        f"{escape(str(identity.profile or 'none'))}",
        soft_wrap=True,
    )
    fallback_reason = getattr(resources, "ocr_fallback_reason", None)
    if fallback_reason:
        console.print(
            f"  [dim]Fallback reason:[/dim] {escape(str(fallback_reason))}", soft_wrap=True
        )


def _cloud_credential_problem(config: ResolvedRunConfig) -> str | None:
    """Why the cloud LLM this run needs cannot be called, or ``None``.

    The provider would otherwise raise only on the first LLM call, after the
    user already sat through OCR. A fast run calls no LLM and needs no key (it
    is what a run with none configured takes without a flag), so only a full
    run is asked. This builds a real provider client but makes no network call.
    """
    if not (config.uses_llm and config.llm_backend == "cloud"):
        return None
    try:
        from bibr.clients.llm import preflight_credentials

        preflight_credentials()
    except ValueError as e:
        return str(e)
    return None


def _report_missing_llm(console, problem: str) -> None:
    ui.error(
        console,
        escape(problem),
        hint="Run [cyan]bibr setup[/cyan] to configure an LLM provider, "
        "or pass [cyan]--fast[/cyan] to run on the local models alone "
        "([cyan]--no-llm[/cyan] for structural output only).",
    )


def _names_a_batch(inputs: list[str]) -> bool:
    """Do the ``chew`` inputs name a batch whatever they match: several inputs,
    or a directory or glob rather than one file (``-`` is one file)?"""
    return len(inputs) > 1 or any(raw != "-" and not Path(raw).is_file() for raw in inputs)


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
    if code == "scan_needs_ocr":
        return (
            "\n         [dim]A scanned PDF needs an OCR backend; see which can run here "
            "with: bibr doctor[/dim]"
        )
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
            console.print(
                f"  [red]✗ {escape(fs.path.name)}:[/red] {escape(err_msg)}{hint}", soft_wrap=True
            )
            errors += 1
            continue

        processed += 1

        sink = fs.artifact_sink
        sink_path = sink.destination_path(fs) if sink is not None else None

        if sink_path is not None and fs.core_sha256 is not None:
            # The sink owns checkpoint/final materialization and receipts.
            # Rewriting here would allow unverified in-memory state to bypass
            # that protocol, so the CLI is reporting-only for sink-bound files.
            if is_batch or fs.manifest_output_path is not None:
                console.print(
                    f"  [green]✓ {escape(fs.path.name)}[/green] → {escape(str(sink_path))}"
                )
            else:
                total_elapsed = time.monotonic() - total_t0
                console.print(
                    f"  [green]✓[/green] Wrote {escape(str(sink_path))} ({total_elapsed:.1f}s)"
                )
            _print_validation_line(console, fs.result_json, sink_path)
            continue

        if fs.manifest_output_path is not None:
            out_file = sink_path or fs.manifest_output_path
            atomic_write_json(out_file, fs.result_json, **json_kwargs)
            console.print(f"  [green]✓ {escape(fs.path.name)}[/green] → {escape(str(out_file))}")
            _print_validation_line(console, fs.result_json, out_file)
        elif output_path is None:
            _write_stdout_json(json.dumps(fs.result_json, **json_kwargs))
            if not is_batch:
                total_elapsed = time.monotonic() - total_t0
                console.print(f"  [green]✓[/green] Done ({total_elapsed:.1f}s)")
                _print_validation_line(console, fs.result_json)
        elif is_batch:
            out_file = sink_path or output_path / f"{fs.path.stem}.json"
            atomic_write_json(out_file, fs.result_json, **json_kwargs)
            console.print(f"  [green]✓ {escape(fs.path.name)}[/green] → {escape(str(out_file))}")
            _print_validation_line(console, fs.result_json, out_file)
        else:
            out_file = sink_path or _resolve_single_output_path(output_path, fs.path)
            atomic_write_json(out_file, fs.result_json, **json_kwargs)
            total_elapsed = time.monotonic() - total_t0
            console.print(
                f"  [green]✓[/green] Wrote {escape(str(out_file))} ({total_elapsed:.1f}s)"
            )
            _print_validation_line(console, fs.result_json, out_file)

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
    stage_labels: dict[str, str] | None = None

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
            stages=stages_for_files(self.active_stages, (state.path for state in file_states)),
            labels=self.stage_labels,
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
        if self.is_batch:
            # Hand the chunk's freed heap back before the next one renders
            # (PIPELINE_HEAP_TRIM, see bibr.utils.heap).
            from bibr.utils.heap import trim_after_chunk

            await trim_after_chunk(getattr(self.pipeline, "settings", None))

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

    if getattr(args, "sample", False):
        if args.input or args.manifest:
            ui.error(console, "--sample processes the paper that ships with bibr; drop the input.")
            sys.exit(2)
        args.input = [str(sample_paper_path())]

    try:
        source_mode = resolve_cli_source_mode(args.input, args.manifest)
        if source_mode == "manifest" and args.output is not None:
            raise ManifestError(
                "--manifest cannot be combined with -o/--output; "
                "use each manifest row's output_path"
            )
        manifest_records = load_manifest(args.manifest) if source_mode == "manifest" else []
    except ManifestError as exc:
        ui.error(console, escape(str(exc)))
        sys.exit(2)

    _apply_runtime_settings(args)

    try:
        config = resolve_run_config(args)
    except ValueError as e:
        ui.error(console, f"Invalid option: {escape(str(e))}")
        sys.exit(2)

    # OCR_FALLBACK_TO_TEXT_LAYER=auto: an automatic OCR backend that cannot
    # start here becomes the text layer before the summary names the backend;
    # the warning waits until the inputs show there are PDFs to read.
    config, fallback_warning = apply_text_layer_fallback(config, _preflight_ocr_runtime)

    # Collect files before anything is said about the run: a mistyped path is
    # the whole story, not a footnote under the mode line.
    if source_mode == "manifest":
        files = [record.input_path for record in manifest_records]
        work_items = list(manifest_records)
        missing_count = 0
    else:
        files, missing_count = _collect_files(args.input)
        work_items = list(files)

    # The dry-run plan carries the same facts in full.
    if not args.dry_run:
        console.print(_format_run_summary(config), soft_wrap=True)
    # Batch mode follows the kind of input, not how many files it matched: a
    # directory holding one paper still writes <dir>/<stem>.json, as it does
    # with two, instead of a file named after the directory.
    is_batch = source_mode == "manifest" or len(files) > 1 or _names_a_batch(args.input)
    if is_batch:
        # PIPELINE_MALLOC_ARENAS, before the pipeline starts any thread.
        from bibr.utils.heap import limit_arenas_for_batch

        limit_arenas_for_batch()

    # --paper-id only makes sense when writing a single result — silently
    # discarding it for batch input used to hide the mistake entirely.
    if is_batch and args.paper_id:
        ui.error(
            console,
            "--paper-id is only valid for single-file input, not for a directory, "
            "glob, manifest or several inputs.",
        )
        sys.exit(2)

    # Batch output writes <dir>/<stem>.json (and, with -o, its sidecars) — files
    # from different directories sharing a stem would overwrite each other.
    if is_batch and source_mode != "manifest":
        collisions = _find_stem_collisions(files, sidecars=args.output is not None)
        if collisions:
            ui.error(
                console,
                "Output filename collision: multiple input files would write the same output file:",
            )
            # Plain print (not console.print) for the paths themselves —
            # Rich hard-wraps long lines at the terminal width with no
            # regard for word boundaries, which can split a long absolute
            # path mid-character.
            for name, paths in sorted(collisions.items()):
                print(f"  {name} would be written by:", file=sys.stderr)
                for p in paths:
                    print(f"    {p}", file=sys.stderr)
            console.print(
                "[dim]Rename the colliding files, or process them in separate "
                "runs with different -o targets.[/dim]"
            )
            sys.exit(2)

    # Fail fast on a managed local backend that can't run here (unsupported
    # hardware or a missing launcher) — otherwise the run crashes deep in vLLM
    # after OCR. Skipped under --no-llm (no server is started) and --dry-run
    # (nothing is started).
    from bibr.local.pipeline import LOCAL_LLM_BACKENDS

    if not args.dry_run and config.uses_llm and config.llm_backend in LOCAL_LLM_BACKENDS:
        err = _preflight_local_backend(config.llm_backend)
        if err:
            ui.error(console, escape(err))
            sys.exit(1)

    if fallback_warning is not None and any(p.suffix.lower() == ".pdf" for p in files):
        ui.warn(console, escape(fallback_warning))

    downloads_announced = False
    if not args.dry_run and any(p.suffix.lower() == ".pdf" for p in files):
        opencv_problem = _preflight_opencv()
        if opencv_problem is not None:
            message, repair = opencv_problem
            ui.error(console, escape(message), hint=f"Repair with: [cyan]{repair}[/cyan]")
            sys.exit(1)
        # Fail fast when no local OCR runtime can start here (no suitable GPU
        # for paddle-vllm, no llama-server on PATH) — otherwise the run loads
        # the layout model and, for paddle-vllm, bootstraps vLLM before the
        # transactional chain reports the same thing.
        ocr_reason = _preflight_ocr_runtime(config)
        if ocr_reason is not None:
            # A full run with no LLM set up is missing that first: say so
            # before the OCR runtime, which this user may never need.
            credential_problem = _cloud_credential_problem(config)
            if credential_problem is not None:
                _report_missing_llm(console, credential_problem)
            ui.error(console, escape(ocr_reason))
            sys.exit(1)
        # A fast run reads the front matter with the tagger alone; without its
        # bundle every paper would come out with no title, authors or abstract.
        # The check fetches the bundle on a first run, so the download notice
        # comes first: it used to follow, with the largest model already
        # downloaded in silence and missing from its list.
        announce_downloads(config, files, console)
        downloads_announced = True
        tagger_reason = _preflight_front_tagger(config)
        if tagger_reason is not None:
            ui.error(console, escape(tagger_reason))
            sys.exit(1)
    if not args.dry_run:
        # Every input kind: a DOCX, XML or HTML paper loads the reference,
        # classifier and statement models too.
        override_reason = _preflight_model_overrides(
            config, warn=lambda m: ui.warn(console, escape(m))
        )
        if override_reason is not None:
            ui.error(console, escape(override_reason))
            sys.exit(1)

    # Fail fast on missing LLM credentials, after the inputs resolved so a
    # mistyped path is reported as such — the provider would otherwise
    # raise only on the first LLM call, after the user already sat through OCR.
    # It comes after the OCR check above: a machine that cannot OCR at all
    # should hear that before it is asked for a key. A fast run calls no LLM and
    # needs no key (it is what a run with none configured takes without a flag),
    # so only a full run is asked. Skipped under --dry-run: it constructs a real
    # provider client (no network call, but --dry-run previews the plan without
    # touching any client/model machinery at all — see ``_print_dry_run_plan``).
    if not args.dry_run:
        credential_problem = _cloud_credential_problem(config)
        if credential_problem is not None:
            _report_missing_llm(console, credential_problem)
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
        if source_mode != "manifest":
            output_problem = _output_path_problem(args.output, is_batch=is_batch)
            if output_problem is not None:
                blockers.append(output_problem)
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
        # The same check --dry-run reports as a blocker: without it an
        # unwritable -o surfaced as a traceback from the first write.
        output_problem = _output_path_problem(args.output, is_batch=is_batch)
        if output_problem is not None:
            ui.error(console, escape(output_problem))
            sys.exit(2)
        try:
            output_path = _prepare_output_path(args.output, is_batch=is_batch)
        except OSError as exc:
            ui.error(console, escape(f"Cannot write output {args.output!r}: {exc}"))
            sys.exit(2)

    if not downloads_announced:
        announce_downloads(config, files, console)

    # Create pipeline
    from bibr.local.pipeline import LocalPipeline

    pipeline = LocalPipeline(
        memory_mode=config.memory_mode,
        ocr_backend=config.pipeline_ocr_backend,
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
        mode=config.mode,
        ocr_requested=config.ocr_requested,
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

    # Several papers on stdout are JSON Lines, one paper per line: indented
    # documents back to back are not something a JSON parser can read.
    json_lines = is_batch and output_path is None and source_mode != "manifest"
    if args.compact or json_lines:
        json_kwargs = {"separators": (",", ":"), "ensure_ascii": False}
    else:
        json_kwargs = {"indent": 2, "ensure_ascii": False}
    if json_lines and not args.compact:
        console.print(
            "  [dim]several papers and no -o: writing JSON Lines to stdout, one paper "
            "per line (-o DIR writes a file each)[/dim]",
            soft_wrap=True,
        )

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
        # No OCR model runs on the text-layer route; the stage only reads the PDF.
        stage_labels=(
            {"ocr": "Reading the PDF text layer"} if config.ocr_backend == "text-layer" else None
        ),
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
                console.print(f"  [dim]\\[{file_num}/{len(files)}][/dim] {escape(fp.name)}")

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
            f"  [dim]LLM tokens · {escape(str(model))}: "
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
