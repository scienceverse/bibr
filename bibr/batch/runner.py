"""Orchestration for ``bibr batch``: plan → ``run_info.json`` → execute → ledger → report.

The runner is executor-agnostic. The **local** executor feeds chunks of
``--batch-size`` files to one warm :class:`bibr.api.Chewer` (stage-major
inside a chunk, models load once) and writes ledger lines as each chunk
finishes; the **remote** executor (:mod:`bibr.batch.remote`) submits papers
to a ``bibr serve`` job API with adaptive concurrency. Both report through
the same :func:`Ledger.record` path, so ``outcomes.jsonl`` has one shape.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bibr.batch.ledger import (
    CHUNK_ERROR,
    INTERRUPTED,
    LEDGER_FILENAME,
    UPSTREAM_UNAVAILABLE,
    Ledger,
    LedgerContext,
    Outcome,
    ResumePlan,
    summarize_export,
    utc_now_iso,
)
from bibr.batch.manifest import BatchItem, Discovery, assign_paper_ids, discover_inputs
from bibr.batch.remote import RemoteAuthError, RemoteExecutor, RemoteOptions, configured_value
from bibr.batch.report import compute_report, render_report
from bibr.utils.transient import is_service_outage

logger = logging.getLogger(__name__)

RUN_INFO_FILENAME = "run_info.json"
RUN_HISTORY_FILENAME = "runs.jsonl"
EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

ChewMany = Callable[[list[Path], int], list[Any]]


@dataclass
class LocalOptions:
    """What the local executor needs: warm-pipeline kwargs and chunking."""

    chew_options: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 0
    stages: list[str] | None = None
    summary: list[str] = field(default_factory=list)
    # Runs after discovery, before any model loads; returns a blocking reason.
    preflight: Callable[[Sequence[Path]], str | None] | None = None


@dataclass
class BatchOptions:
    inputs: list[str]
    out: Path
    limit: int = 0
    shuffle: bool = False
    seed: int | None = None
    deadline: float | None = None  # epoch seconds; stop submitting after
    retry_failed: bool = False
    force: bool = False
    dry_run: bool = False
    local: LocalOptions | None = None
    remote: RemoteOptions | None = None
    cli_options: dict[str, Any] = field(default_factory=dict)
    report_json: bool = False
    tables: bool = True  # write <out>/tables/*.parquet after the run

    @property
    def executor(self) -> str:
        return "remote" if self.remote is not None else "local"


@dataclass
class BatchPlan:
    discovery: Discovery
    items: list[BatchItem]
    resume: ResumePlan
    to_run: list[BatchItem]
    seed: int | None

    @property
    def collisions(self) -> list[BatchItem]:
        return [item for item in self.items if item.disambiguated]


def parse_deadline(text: str) -> float:
    """``--deadline`` as epoch seconds: a number, or ISO-8601 (naive = local time)."""
    value = text.strip()
    try:
        return float(value)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            f"--deadline must be ISO-8601 (2026-09-03T06:00:00Z) or epoch seconds, got {text!r}"
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.timestamp()


def build_plan(options: BatchOptions, ledger: Ledger) -> BatchPlan:
    """Discover inputs, assign ids, apply resume rules, shuffle and limit."""
    discovery = discover_inputs(options.inputs)
    entries = ledger.read()
    items = assign_paper_ids(discovery.files, recorded=entries)
    resume = ledger.plan(
        items, force=options.force, retry_failed=options.retry_failed, entries=entries
    )
    to_run = list(resume.to_run)
    seed = options.seed
    if options.shuffle:
        if seed is None:
            seed = random.randrange(2**31)  # noqa: S311 — recorded for reproducibility
        random.Random(seed).shuffle(to_run)  # noqa: S311 — order only
    if options.limit and options.limit > 0:
        to_run = to_run[: options.limit]
    return BatchPlan(discovery=discovery, items=items, resume=resume, to_run=to_run, seed=seed)


def redacted_settings_snapshot() -> dict[str, str]:
    """Non-default settings as ``bibr config show`` renders them (secrets masked)."""
    from bibr.config_cli import format_value, resolve_provenance
    from bibr.config_introspect import iter_setting_docs

    snapshot: dict[str, str] = {}
    for doc in iter_setting_docs():
        provenance = resolve_provenance(doc)
        if provenance.tier == "default" or provenance.value is None:
            continue
        snapshot[doc.env_name] = format_value(doc, provenance.value)
    return dict(sorted(snapshot.items()))


def bibr_version() -> str:
    import bibr

    return getattr(bibr, "__version__", "unknown")


def local_build_sha() -> str | None:
    """``BIBR_BUILD_SHA`` when stamped (images), else the checkout's git head."""
    try:
        stamped = configured_value("BIBR_BUILD_SHA")
    except Exception:  # noqa: BLE001 — settings problems are reported by the CLI itself
        stamped = None
    if stamped:
        return stamped
    import bibr

    repo = Path(bibr.__file__).resolve().parent.parent
    if not (repo / ".git").exists():
        return None
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip() if result.returncode == 0 else ""
    return sha or None


def write_run_info(out_dir: Path, info: dict[str, Any], *, append_history: bool) -> Path:
    from bibr.local.artifacts import atomic_write_json

    path = out_dir / RUN_INFO_FILENAME
    atomic_write_json(path, info, indent=2)
    if append_history:
        with (out_dir / RUN_HISTORY_FILENAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(info, ensure_ascii=False, default=str) + "\n")
    return path


def export_path(out_dir: Path, item: BatchItem) -> Path:
    return out_dir / f"{item.paper_id}.json"


TABLES_DIRNAME = "tables"


def write_batch_tables(
    out_dir: Path, ledger: Ledger, entries: list[dict[str, Any]], *, console: Any
) -> None:
    """Rebuild ``<out>/tables/`` from every paper whose latest attempt is ok.

    Best-effort: the JSON exports are the run's result, so a table failure is
    a warning, never a failed run. Exports of another schema major — left by
    an older bibr in a resumed out dir — are left out and counted, since one
    would otherwise fail the whole rebuild on every run.
    """
    from bibr.export.schema_artifact import SCHEMA_MAJOR
    from bibr.export.tables import write_tables
    from bibr.local.cli import ui
    from bibr.local.cli.tables import report_tables

    files = [
        path
        for paper_id, entry in sorted(ledger.latest(entries).items())
        if entry.get("status") == "ok" and (path := out_dir / f"{paper_id}.json").is_file()
    ]
    if not files:
        return
    other_major: list[tuple[str, str]] = []

    def current_major() -> Iterator[Any]:
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = None
            version = payload.get("schema_version") if isinstance(payload, dict) else None
            if not isinstance(version, str):
                yield path  # not an export: write_tables reports it as skipped
            elif version.split(".")[0] != SCHEMA_MAJOR:
                other_major.append((path.name, version))
            else:
                yield payload

    exports = current_major()
    first = next(exports, None)
    report = None
    if first is not None:  # else keep whatever tables an older bibr wrote
        try:
            report = write_tables(itertools.chain([first], exports), out_dir / TABLES_DIRNAME)
        except Exception as exc:  # noqa: BLE001 - tables are a derived convenience
            logger.warning("writing Parquet tables failed", exc_info=True)
            ui.warn(console, f"Parquet tables not written: {exc}")
            return
    if other_major:
        name, version = other_major[0]
        ui.warn(
            console,
            f"{len(other_major)} export(s) of another schema major left out of the tables "
            f"({name}: {version}); re-run them with --force to include them",
        )
    if report is not None:
        report_tables(console, report)


# --- local executor -----------------------------------------------------------


class WarmChewMany:
    """Default local ``chew_many``: one :class:`bibr.api.Chewer` for the whole run."""

    def __init__(self, local: LocalOptions):
        from bibr.api import Chewer

        self._chewer = Chewer(**local.chew_options)
        self._stages = local.stages

    def __call__(self, paths: list[Path], batch_size: int) -> list[Any]:
        progress = None
        if self._stages:
            from bibr.pipeline.progress import RichProgress

            progress = RichProgress(stages=self._stages)
        result = self._chewer.chew(list(paths), batch_size=batch_size, progress=progress)
        return list(result) if isinstance(result, list) else [result]

    def close(self) -> None:
        self._chewer.close()


def open_chew_many(local: LocalOptions) -> WarmChewMany:
    """Factory for the local executor's ``chew_many`` (patched in tests)."""
    return WarmChewMany(local)


class LocalExecutor:
    """Chunked, stage-major local processing with per-chunk ledger streaming."""

    def __init__(
        self,
        chew_many: ChewMany,
        *,
        batch_size: int,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ):
        self._chew_many = chew_many
        self.batch_size = max(1, batch_size)
        self._clock = clock
        self._wall = wall

    def run(
        self,
        items: Sequence[BatchItem],
        *,
        on_outcome: Callable[[BatchItem, Outcome], None],
        deadline: float | None = None,
        stop: threading.Event | None = None,
        on_chunk: Callable[[int, int, Sequence[BatchItem]], None] | None = None,
    ) -> str:
        """Returns ``completed`` | ``deadline`` | ``stopped`` | ``interrupted``."""
        total_chunks = (len(items) + self.batch_size - 1) // self.batch_size
        for index, start in enumerate(range(0, len(items), self.batch_size), start=1):
            if stop is not None and stop.is_set():
                return "stopped"
            if deadline is not None and self._wall() >= deadline:
                return "deadline"
            chunk = list(items[start : start + self.batch_size])
            if on_chunk is not None:
                on_chunk(index, total_chunks, chunk)
            started_at = utc_now_iso()
            t0 = self._clock()
            try:
                results = self._chew_many([item.path for item in chunk], len(chunk))
            except KeyboardInterrupt:
                elapsed = self._clock() - t0
                finished_at = utc_now_iso()
                for item in chunk:
                    on_outcome(
                        item,
                        Outcome(
                            "failed",
                            error_code=INTERRUPTED,
                            error="interrupted by user",
                            started_at=started_at,
                            finished_at=finished_at,
                            duration_s=elapsed,
                        ),
                    )
                return "interrupted"
            except Exception as exc:  # noqa: BLE001 — a crashed chunk is N failed papers
                logger.exception("chunk %d/%d crashed", index, total_chunks)
                elapsed = self._clock() - t0
                finished_at = utc_now_iso()
                code = UPSTREAM_UNAVAILABLE if is_service_outage(exc) else CHUNK_ERROR
                for item in chunk:
                    on_outcome(
                        item,
                        Outcome(
                            "failed",
                            error_code=code,
                            error=f"{type(exc).__name__}: {exc}",
                            started_at=started_at,
                            finished_at=finished_at,
                            duration_s=elapsed,
                        ),
                    )
                continue
            elapsed = self._clock() - t0
            finished_at = utc_now_iso()
            if len(results) != len(chunk):
                raise RuntimeError(
                    f"chew_many returned {len(results)} results for {len(chunk)} paths"
                )
            for item, result in zip(chunk, results, strict=True):
                on_outcome(item, _local_outcome(result, started_at, finished_at, elapsed))
        return "completed"


def _local_outcome(result: Any, started_at: str, finished_at: str, elapsed: float) -> Outcome:
    if getattr(result, "ok", False):
        export = getattr(result, "data", None)
        if export is None and isinstance(result, dict):
            export = result
        pipeline_seconds = summarize_export(export).get("total_seconds")
        return Outcome(
            "ok",
            export=export,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=pipeline_seconds if pipeline_seconds is not None else elapsed,
        )
    # A service outage says nothing about the paper: record it under the code
    # the remote executor uses for the same failure, which resume re-runs.
    code = UPSTREAM_UNAVAILABLE if getattr(result, "outage", False) is True else None
    return Outcome(
        "failed",
        error_code=code or getattr(result, "error_code", None) or "processing_error",
        failed_stage=getattr(result, "failed_stage", None),
        error=getattr(result, "error", None) or "processing failed",
        started_at=started_at,
        finished_at=finished_at,
        duration_s=elapsed,
    )


# --- the run ------------------------------------------------------------------


def _print(console: Any, message: str, **kwargs: Any) -> None:
    if console is not None:
        console.print(message, **kwargs)


def _print_plan(options: BatchOptions, plan: BatchPlan, ledger: Ledger) -> None:
    """The ``--dry-run`` payload, on stdout like ``bibr chew --dry-run``."""
    from rich.console import Console

    from bibr.local.cli import ui

    console = Console(soft_wrap=True)

    def kv(label: str, value: str) -> str:
        return ui.kv(label, value, width=12)

    discovery = plan.discovery
    ui.brand_header(
        console, f"bibr batch {ui.SEP} dry run", subtitle="plan preview — nothing will be processed"
    )
    ui.section(console, f"Input ({len(discovery.files)} files)")
    for path in discovery.files[:5]:
        console.print(f"  {path}", soft_wrap=True)
    if len(discovery.files) > 5:
        console.print(f"  … and {len(discovery.files) - 5} more")
    console.print(
        kv(
            "sources",
            f"{len(discovery.manifests)} manifests {ui.SEP} "
            f"{len(discovery.directories)} directories {ui.SEP} "
            f"{len(discovery.missing)} missing {ui.SEP} "
            f"{len(discovery.unsupported)} unsupported",
        )
    )

    ui.section(console, "Resume")
    n_lines = len(ledger.read()) if ledger.path.is_file() else 0
    console.print(kv("ledger", f"{ledger.path} ({n_lines} lines)"), soft_wrap=True)
    console.print(kv("skip ok", str(len(plan.resume.skipped_ok))))
    hint = " (use --retry-failed)" if plan.resume.skipped_failed else ""
    console.print(kv("skip failed", f"{len(plan.resume.skipped_failed)}{hint}"))
    detail = []
    if options.limit:
        detail.append(f"limit {options.limit}")
    if options.shuffle:
        detail.append(f"shuffled, seed {plan.seed}")
    suffix = f" ({', '.join(detail)})" if detail else ""
    console.print(kv("to run", f"{len(plan.to_run)} of {len(plan.items)}{suffix}"))
    if plan.collisions:
        console.print(kv("ids", f"{len(plan.collisions)} stem collisions disambiguated:"))
        for item in plan.collisions[:5]:
            console.print(f"    {item.path} {ui.ARROW} {item.paper_id}", soft_wrap=True)
    if options.deadline is not None:
        stamp = datetime.fromtimestamp(options.deadline, tz=UTC).isoformat(timespec="seconds")
        console.print(kv("deadline", f"{stamp} (stop submitting after)"))

    ui.section(console, "Executor")
    if options.remote is not None:
        remote = options.remote
        console.print(kv("remote", remote.serve_url))
        console.print(
            kv(
                "in-flight",
                f"{remote.concurrency} (min {remote.min_concurrency}, "
                f"max {remote.max_concurrency})",
            )
        )
        console.print(
            kv(
                "retries",
                f"{remote.retries} {ui.SEP} poll timeout {remote.poll_timeout:.0f}s "
                f"{ui.SEP} poll every {remote.poll_interval:.0f}s",
            )
        )
        form = " ".join(f"{k}={v}" for k, v in remote.form.items()) or "(none)"
        console.print(kv("form", form))
        console.print(kv("token", "set" if remote.token else "none (auth disabled?)"))
    else:
        local = options.local or LocalOptions()
        console.print(kv("local", f"batch size {local.batch_size or 'auto'}"))
        for line in local.summary:
            console.print(f"  {line}", soft_wrap=True)

    ui.section(console, "Output")
    console.print(f"  {options.out}/<paper_id>.json", soft_wrap=True)
    console.print(f"  {options.out}/{LEDGER_FILENAME}", soft_wrap=True)
    console.print(f"  {options.out}/{RUN_INFO_FILENAME}", soft_wrap=True)
    for item in plan.to_run[:3]:
        console.print(
            f"  {item.path.name} {ui.ARROW} {export_path(options.out, item)}", soft_wrap=True
        )
    console.print("\n[dim]Dry run — nothing was processed.[/dim]")


def _run_remote(
    executor: RemoteExecutor,
    items: Sequence[BatchItem],
    *,
    on_outcome: Callable[[BatchItem, Outcome], None],
    on_ready: Callable[[dict[str, Any]], None],
    deadline: float | None,
    stop: asyncio.Event | None,
) -> str:
    async def main() -> str:
        loop = asyncio.get_running_loop()
        stop_event = stop or asyncio.Event()
        installed = False

        def on_sigint() -> None:
            # First Ctrl-C: stop submitting, drain in-flight. A second one
            # falls back to the default handler (KeyboardInterrupt).
            stop_event.set()
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

        try:
            loop.add_signal_handler(signal.SIGINT, on_sigint)
            installed = True
        except (NotImplementedError, RuntimeError, ValueError):
            installed = False
        try:
            return await executor.run(
                items,
                on_outcome=on_outcome,
                stop=stop_event,
                deadline=deadline,
                on_ready=on_ready,
            )
        finally:
            if installed:
                try:
                    loop.remove_signal_handler(signal.SIGINT)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass

    return asyncio.run(main())


def run_batch(
    options: BatchOptions,
    *,
    console: Any = None,
    transport: Any = None,
    stop: Any = None,
) -> int:
    """Execute one ``bibr batch`` run end to end; returns the process exit code."""
    from bibr.local.artifacts import atomic_write_json
    from bibr.local.cli import ui

    if console is None:
        from rich.console import Console

        console = Console(stderr=True)

    ledger = Ledger(options.out / LEDGER_FILENAME)
    plan = build_plan(options, ledger)
    discovery = plan.discovery

    for missing in discovery.missing:
        ui.warn(console, f"not found: {missing}")
    for unsupported in discovery.unsupported:
        ui.warn(console, f"unsupported file type, skipped: {unsupported}")
    for empty in discovery.empty_dirs:
        ui.warn(console, f"no supported files in: {empty}")
    if not discovery.files:
        ui.error(
            console,
            "No input files found.",
            hint="Pass a manifest (one path per line), a directory, or files.",
        )
        return EXIT_USAGE

    if options.dry_run:
        _print_plan(options, plan, ledger)
        return EXIT_OK

    if options.local is not None and options.local.preflight is not None and plan.to_run:
        problem = options.local.preflight([item.path for item in plan.to_run])
        if problem:
            ui.error(console, problem)
            return EXIT_FAILURES

    run_id = uuid.uuid4().hex[:12]
    started_at = utc_now_iso()
    options.out.mkdir(parents=True, exist_ok=True)
    context_holder: dict[str, LedgerContext] = {
        "context": LedgerContext(
            run_id=run_id,
            executor=options.executor,
            bibr_version=bibr_version(),
            build_sha=local_build_sha() if options.remote is None else None,
        )
    }
    info: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "bibr_version": context_holder["context"].bibr_version,
        "build_sha": context_holder["context"].build_sha,
        "executor": options.executor,
        "options": dict(options.cli_options),
        "inputs": list(options.inputs),
        "out": str(options.out),
        "ledger": str(ledger.path),
        "n_inputs": len(plan.items),
        "n_planned": len(plan.to_run),
        "n_skipped_ok": len(plan.resume.skipped_ok),
        "n_skipped_failed": len(plan.resume.skipped_failed),
        "shuffle_seed": plan.seed if options.shuffle else None,
        "deadline": (
            datetime.fromtimestamp(options.deadline, tz=UTC).isoformat(timespec="seconds")
            if options.deadline is not None
            else None
        ),
        "collisions": {item.paper_id: str(item.path) for item in plan.collisions},
        "settings": redacted_settings_snapshot(),
    }
    if options.remote is not None:
        info["serve"] = {
            "url": options.remote.serve_url,
            "concurrency": [
                options.remote.concurrency,
                options.remote.min_concurrency,
                options.remote.max_concurrency,
            ],
            "form": dict(options.remote.form),
            "ready": None,
        }
    write_run_info(options.out, info, append_history=True)

    n_ok = 0
    n_failed = 0
    done = 0
    total = len(plan.to_run)

    def record(item: BatchItem, outcome: Outcome) -> None:
        nonlocal n_ok, n_failed, done
        if outcome.ok and outcome.export is not None:
            # The batch's own id is the corpus key: unique by construction and
            # the name of the JSON file, where the export's default (the DOI,
            # else the file name) can collide across a corpus.
            export = {**outcome.export, "paper_id": item.paper_id}
            atomic_write_json(export_path(options.out, item), export, indent=2)
        entry = ledger.record(item, outcome, context=context_holder["context"])
        done += 1
        if outcome.ok:
            n_ok += 1
            refs = entry.get("n_refs")
            duration = entry.get("duration_s")
            detail = f"{duration:.1f}s" if isinstance(duration, (int, float)) else ""
            if isinstance(refs, int):
                detail += f" {ui.SEP} {refs} refs"
            console.print(
                f"  [green]{ui.OK}[/green] [dim]\\[{done}/{total}][/dim] {item.paper_id}"
                f"  [dim]{detail}[/dim]",
                soft_wrap=True,
            )
        else:
            n_failed += 1
            console.print(
                f"  [red]{ui.FAIL}[/red] [dim]\\[{done}/{total}][/dim] {item.paper_id}: "
                f"{outcome.error_code} — {outcome.error or ''}",
                soft_wrap=True,
            )

    console.print(
        f"  [dim]{total} to run {ui.SEP} {len(plan.resume.skipped_ok)} already ok "
        f"{ui.SEP} {len(plan.resume.skipped_failed)} skipped failed {ui.SEP} "
        f"{options.executor} {ui.SEP} run {run_id}[/dim]"
    )

    reason = "completed"
    interrupted = False
    fatal: str | None = None
    if total == 0:
        console.print("  [dim]nothing to do[/dim]")
    elif options.remote is not None:
        executor = RemoteExecutor(options.remote, transport=transport)

        def on_ready(body: dict[str, Any]) -> None:
            build_sha = body.get("build_sha")
            context_holder["context"] = LedgerContext(
                run_id=run_id,
                executor="remote",
                bibr_version=context_holder["context"].bibr_version,
                build_sha=str(build_sha) if build_sha else None,
            )
            info["build_sha"] = context_holder["context"].build_sha
            info["serve"]["ready"] = body
            write_run_info(options.out, info, append_history=False)
            console.print(
                f"  [dim]serve ready {ui.SEP} build {build_sha or '?'} {ui.SEP} "
                f"in-flight {executor.gate.size}[/dim]"
            )

        try:
            reason = _run_remote(
                executor,
                plan.to_run,
                on_outcome=record,
                on_ready=on_ready,
                deadline=options.deadline,
                stop=stop,
            )
        except KeyboardInterrupt:
            interrupted = True
            reason = "interrupted"
        except (RemoteAuthError, TimeoutError) as exc:
            fatal = str(exc)
        if executor.fatal:
            fatal = executor.fatal
        if executor.stats["submit_429"] or executor.stats["transient_retries"]:
            console.print(
                f"  [dim]429s {executor.stats['submit_429']} {ui.SEP} "
                f"transient retries {executor.stats['transient_retries']} {ui.SEP} "
                f"final in-flight {executor.gate.size}[/dim]"
            )
    else:
        local = options.local or LocalOptions()
        chew_many = open_chew_many(local)
        batch_size = local.batch_size
        if batch_size <= 0:
            from bibr.local.pipeline import _auto_batch_size

            memory_mode = local.chew_options.get("memory_mode") or "balanced"
            batch_size = _auto_batch_size(memory_mode)

        def on_chunk(index: int, total_chunks: int, chunk: Sequence[BatchItem]) -> None:
            if total_chunks > 1:
                console.print(f"\n[bold]Chunk {index}/{total_chunks}[/bold]")
            for item in chunk:
                console.print(f"  [dim]{item.path.name}[/dim]", soft_wrap=True)

        try:
            reason = LocalExecutor(chew_many, batch_size=batch_size).run(
                plan.to_run,
                on_outcome=record,
                deadline=options.deadline,
                stop=stop,
                on_chunk=on_chunk,
            )
        finally:
            close = getattr(chew_many, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 — teardown must not mask the run's result
                    logger.warning("pipeline teardown failed", exc_info=True)
        interrupted = reason == "interrupted"

    finished_at = utc_now_iso()
    info.update(
        {
            "finished_at": finished_at,
            "reason": reason,
            "n_ok": n_ok,
            "n_failed": n_failed,
            "fatal": fatal,
        }
    )
    write_run_info(options.out, info, append_history=False)

    entries = ledger.read()
    if options.tables:
        write_batch_tables(options.out, ledger, entries, console=console)
    report = compute_report(entries, run_id=run_id)
    if options.report_json:
        print(json.dumps(report, indent=2))
    else:
        print(render_report(report, title=f"bibr batch {ui.SEP} run {run_id} ({reason})"))
        if reason == "deadline":
            console.print("  [dim]deadline reached — re-run the same command to continue[/dim]")
        elif reason in ("stopped", "interrupted"):
            console.print(
                "  [dim]interrupted — re-run the same command to continue "
                "(interrupted papers are retried by default)[/dim]"
            )

    if fatal:
        ui.error(console, fatal)
        return EXIT_USAGE
    if interrupted or reason == "stopped":
        return EXIT_INTERRUPTED
    return EXIT_FAILURES if n_failed else EXIT_OK
