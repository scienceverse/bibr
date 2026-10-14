"""``bibr batch --llm-batch front-matter``: a corpus run around Anthropic Message Batches.

Three phases, each resumable by running the same command again:

1. **Round 1** runs the corpus as ``bibr batch`` does, with the per-paper
   front-matter LLM calls deferred (:mod:`bibr.clients.batch_defer`): a call
   that misses the LLM response cache is spooled to
   ``<out>/llm_batch/spool.jsonl`` and its paper fails softly with
   ``llm_deferred``, a ledger code resume always runs again.
2. **Fill**: the spooled requests, one per cache key and without the ones the
   cache already holds, go out as Message Batches (:mod:`bibr.clients.batch`)
   of at most 100,000 requests or 256 MB each. ``plan.json`` and one state
   file per batch let an interrupted run collect the same batches instead of
   paying for new ones. Every answer that validates is stored in the cache
   under its spooled key, exactly as the live call would have stored it; an
   errored, expired or invalid row stores nothing, so that call goes live in
   the replay.
3. **Round 2** replays the deferred papers. Their deferred calls should now
   hit the cache; the run reports the hit rate and stops loudly under 95 %,
   because then the replay's requests no longer match what the batch answered
   and every miss would be paid for again at the full price.

Message Batches cost half the token price. For bibr's front matter that is a
small saving per paper (under a cent on Haiku 4.5); it matters for hosted runs
of 100,000 papers or more, not for a lab's corpus.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bibr.batch.ledger import LEDGER_FILENAME, LLM_DEFERRED, Ledger, utc_now_iso
from bibr.batch.runner import (
    EXIT_FAILURES,
    EXIT_INTERRUPTED,
    EXIT_USAGE,
    BatchOptions,
    OutDirBusy,
    out_dir_lock,
    run_batch,
)
from bibr.clients.batch_defer import (
    FRONT_MATTER_LABELS,
    SPOOL_DIRNAME,
    SPOOL_FILENAME,
    ReplayStats,
    collecting,
    observing,
    read_spool,
    request_for,
)

logger = logging.getLogger(__name__)

SCOPES: dict[str, frozenset[str]] = {"front-matter": FRONT_MATTER_LABELS}

PLAN_FILENAME = "plan.json"
SUMMARY_FILENAME = "summary.json"
PLAN_VERSION = 1

# The replay must find the batch's answers; below this share of hits the
# requests no longer match what the batch answered.
MIN_REPLAY_HIT_RATE = 0.95
# Deferred-call lookups the replay sees before it may stop early.
EARLY_STOP_LOOKUPS = 20
# A batch ends within 24 hours (rows not done by then expire); wait a little
# longer than that before giving up on one.
BATCH_TIMEOUT_S = 25 * 3600


def llm_batch_blocker(options: BatchOptions, settings: Any) -> str | None:
    """Why ``--llm-batch`` cannot run with these options and settings, or ``None``."""
    if options.remote is not None or options.local is None:
        return "--llm-batch needs the local executor; drop --serve-url."
    chew = options.local.chew_options
    if chew.get("no_llm"):
        return "--llm-batch defers LLM calls; it cannot run with --no-llm."
    llm = settings.llm
    if (llm.provider or "").strip().lower() != "anthropic":
        return f"--llm-batch needs LLM_PROVIDER=anthropic (the provider is {llm.provider!r})."
    backend = chew.get("llm_backend") or llm.backend
    if backend != "cloud":
        return f"--llm-batch needs the cloud LLM backend (the backend is {backend!r})."
    if not settings.cache.llm:
        return (
            "--llm-batch needs CACHE_LLM=true: the batch answers reach the replay "
            "through the LLM response cache."
        )
    if not settings.cache.ocr:
        return (
            "--llm-batch needs CACHE_OCR=true, so the replay does not OCR the "
            "deferred papers a second time."
        )
    if not llm.track_usage:
        return (
            "--llm-batch needs LLM_TRACK_USAGE=true (the default): the deferred calls "
            "are recognised by their usage labels."
        )
    if llm.merged_core_metadata:
        return "--llm-batch defers the three front-matter calls; unset LLM_MERGED_CORE_METADATA."
    if llm.structured_backend == "nuextract-native":
        return "--llm-batch needs the Instructor backend; unset LLM_STRUCTURED_BACKEND."
    if not (llm.api_key or settings.ANTHROPIC_API_KEY):
        return "--llm-batch needs an Anthropic API key (LLM_API_KEY or ANTHROPIC_API_KEY)."
    if importlib.util.find_spec("anthropic") is None:
        return 'The anthropic SDK is not installed: pip install "bibr[batch]".'
    return None


@dataclass
class FillResult:
    """What the fill phase did, for the summary."""

    requests: int = 0  # unique spooled requests
    cached: int = 0  # already in the cache before this fill
    stale: int = 0  # spooled under another provider or model
    submitted: int = 0  # requests in the batches this fill collected
    batches: int = 0
    filled: int = 0  # answers stored in the cache
    errors: Counter = field(default_factory=Counter)  # unanswered rows by error type
    unanswered: set[str] = field(default_factory=set)


def _read_plan(path: Path) -> dict:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": PLAN_VERSION, "parts": []}
    if not isinstance(plan, dict) or plan.get("version") != PLAN_VERSION:
        raise ValueError(
            f"{path} is not a batch plan this bibr can resume; delete it to start over"
        )
    return plan


def _write_plan(path: Path, plan: dict) -> None:
    from bibr.local.artifacts import atomic_write_json

    atomic_write_json(path, plan, indent=1)


def fill_cache(
    work: Path,
    settings: Any,
    *,
    batch_client: Any = None,
    poll_interval: float | None = None,
    timeout_s: float = BATCH_TIMEOUT_S,
    progress: Any = None,
) -> FillResult:
    """Submit the spooled requests the cache lacks and store the answers.

    Requests already in a planned batch are never planned again: an errored
    row goes live in the replay rather than into another batch.
    """
    from bibr.clients.batch import AnthropicBatchAdapter, BatchError, BatchRequest
    from bibr.clients.llm_cache import LlmResponseCache, response_body

    result = FillResult()
    rows = {row["key"]: row for row in read_spool(work / SPOOL_FILENAME)}
    result.requests = len(rows)
    cache = LlmResponseCache(settings=settings)
    model = settings.llm.model
    plan_path = work / PLAN_FILENAME
    plan = _read_plan(plan_path)
    planned = {key for part in plan["parts"] for key in part["keys"]}

    requests_by_key: dict[str, BatchRequest] = {}
    for key, row in rows.items():
        try:
            requests_by_key[key] = request_for(row)
        except (KeyError, TypeError, ValueError, ImportError, AttributeError) as exc:
            logger.warning("Skipping unusable spool row %s: %s", key, exc)
            result.stale += 1

    fresh = []
    for key, row in rows.items():
        if key in planned or key not in requests_by_key:
            continue
        if row.get("provider") != "anthropic" or row.get("model") != model:
            result.stale += 1
        elif cache.get(key) is not None:
            result.cached += 1
        else:
            fresh.append(requests_by_key[key])

    def adapter_for(part_model: str) -> AnthropicBatchAdapter:
        return AnthropicBatchAdapter(model=part_model, settings=settings, client=batch_client)

    if fresh:
        for chunk in adapter_for(model).split(fresh):
            plan["parts"].append(
                {"model": model, "keys": [r.custom_id for r in chunk], "done": False}
            )
        _write_plan(plan_path, plan)

    poll = settings.llm.batch_poll_interval_s if poll_interval is None else poll_interval
    for number, part in enumerate(plan["parts"], start=1):
        if part.get("done"):
            result.unanswered.update(part.get("unanswered", ()))
            continue
        requests = [requests_by_key[key] for key in part["keys"] if key in requests_by_key]
        unanswered: set[str] = {key for key in part["keys"] if key not in requests_by_key}
        if requests:
            answers = adapter_for(part["model"]).run(
                requests,
                state_path=work / f"part-{number:04d}.state.json",
                poll_interval=poll,
                timeout_s=timeout_s,
                progress=(lambda text, n=number: progress(n, text)) if progress else None,
            )
            result.batches += 1
            result.submitted += len(requests)
            for request in requests:
                answer = answers.get(request.custom_id)
                row = rows[request.custom_id]
                if answer is None or isinstance(answer, BatchError):
                    result.errors[answer.error_type if answer else "missing"] += 1
                    unanswered.add(request.custom_id)
                elif cache.put(
                    request.custom_id,
                    response_body(answer),
                    model=row["model"],
                    schema_name=row["schema"],
                    label=row.get("label"),
                    source="batch",
                ):
                    result.filled += 1
                else:
                    result.errors["cache_write_failed"] += 1
                    unanswered.add(request.custom_id)
        part["done"] = True
        part["unanswered"] = sorted(unanswered)
        result.unanswered.update(unanswered)
        _write_plan(plan_path, plan)
    return result


def _deferred_papers(out: Path) -> set[str]:
    """Papers whose latest verdict is ``llm_deferred``."""
    latest: dict[str, dict] = {}
    for entry in Ledger(out / LEDGER_FILENAME).read():
        latest[str(entry.get("paper_id"))] = entry
    return {pid for pid, entry in latest.items() if entry.get("error_code") == LLM_DEFERRED}


def _deferred_calls(spool: Path) -> set[tuple[str, str]]:
    """``(file hash, label)`` of every call round 1 deferred."""
    return {
        (row["file_hash"], row["label"])
        for row in read_spool(spool, unique=False)
        if isinstance(row.get("file_hash"), str) and isinstance(row.get("label"), str)
    }


def _rate(stats: ReplayStats) -> str:
    rate = stats.hit_rate
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def run_llm_batch(
    options: BatchOptions,
    *,
    scope: str = "front-matter",
    console: Any = None,
    settings: Any = None,
    batch_client: Any = None,
) -> int:
    """Run ``bibr batch --llm-batch``; returns the process exit code."""
    from bibr.local.cli import ui

    if console is None:
        from rich.console import Console

        console = Console(stderr=True)
    if settings is None:
        from bibr.config import snapshot_settings

        settings = snapshot_settings()
    labels = SCOPES[scope]
    problem = llm_batch_blocker(options, settings)
    if problem:
        ui.error(console, problem)
        return EXIT_USAGE
    if options.dry_run:
        return run_batch(options, console=console)

    out = Path(options.out)
    work = out / SPOOL_DIRNAME
    spool = work / SPOOL_FILENAME
    summary: dict[str, Any] = {"scope": scope, "started_at": utc_now_iso()}

    ui.section(console, "LLM batch · round 1: run, deferring the front-matter LLM calls")
    # With --json, round 2 prints the one JSON report.
    first = dataclasses.replace(options, print_report=not options.report_json)
    with collecting(spool, labels) as collect:
        code = run_batch(first, console=console)
    summary["round1"] = {"exit_code": code, "spooled": collect.spooled}
    if collect.write_failures:
        ui.warn(console, f"{collect.write_failures} calls could not be spooled and went live")
    if code in (EXIT_USAGE, EXIT_INTERRUPTED):
        return code
    if options.deadline is not None and time.time() >= options.deadline:
        console.print("  [dim]deadline reached — re-run the same command to continue[/dim]")
        return code
    deferred = _deferred_papers(out)
    summary["deferred_papers"] = len(deferred)
    if not deferred:
        console.print("  [dim]no paper is waiting for the LLM batch[/dim]")
        _write_summary(work, summary)
        return code

    ui.section(console, f"LLM batch · {len(deferred)} papers wait for a Message Batch")

    def show(part: int, text: str) -> None:
        console.print(f"  [dim]batch {part} {ui.SEP} {text}[/dim]")

    try:
        with out_dir_lock(out):
            fill = fill_cache(work, settings, batch_client=batch_client, progress=show)
    except OutDirBusy as exc:
        ui.error(console, f"{exc}.", hint="Wait for it to finish, or pass another --out.")
        return EXIT_USAGE
    except KeyboardInterrupt:
        ui.error(
            console,
            "Interrupted while waiting for the Message Batch.",
            hint="It keeps running at Anthropic; re-run the same command to collect it.",
        )
        return EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 — any API failure ends this invocation
        ui.error(
            console,
            f"The Message Batch step failed: {type(exc).__name__}: {exc}",
            hint=(
                "Re-run the same command to resume it. Running without --llm-batch "
                "processes the deferred papers with live calls instead."
            ),
        )
        return EXIT_FAILURES
    summary["fill"] = {
        "requests": fill.requests,
        "already_cached": fill.cached,
        "stale": fill.stale,
        "batches": fill.batches,
        "submitted": fill.submitted,
        "filled": fill.filled,
        "unanswered": len(fill.unanswered),
        "errors": dict(fill.errors),
    }
    errors = ", ".join(f"{kind} ×{n}" for kind, n in fill.errors.most_common())
    console.print(
        f"  {fill.requests} deferred requests {ui.SEP} {fill.submitted} sent in "
        f"{fill.batches} batches {ui.SEP} {fill.filled} answers cached"
        + (
            f" {ui.SEP} {len(fill.unanswered)} unanswered ({errors}), these go live"
            if errors
            else ""
        )
    )

    ui.section(console, f"LLM batch · round 2: replay {len(deferred)} papers")
    stop = threading.Event()

    def watch(stats: ReplayStats) -> None:
        if (
            stats.lookups >= EARLY_STOP_LOOKUPS
            and stats.hit_rate is not None
            and stats.hit_rate < MIN_REPLAY_HIT_RATE
        ):
            stop.set()

    replay = dataclasses.replace(
        options,
        force=False,
        retry_failed=False,
        limit=0,
        shuffle=False,
        seed=None,
        only_paper_ids=frozenset(deferred),
    )
    started = time.monotonic()
    with observing(
        _deferred_calls(spool), labels=labels, unanswered_keys=fill.unanswered, on_update=watch
    ) as stats:
        code = run_batch(replay, console=console, stop=stop)
    summary["round2"] = {
        "exit_code": code,
        "seconds": round(time.monotonic() - started, 1),
        "hits": stats.hits,
        "misses": stats.misses,
        "unanswered": stats.unanswered,
        "hit_rate": stats.hit_rate,
        "stopped_early": stop.is_set(),
    }
    summary["finished_at"] = utc_now_iso()
    _write_summary(work, summary)

    console.print(
        f"  replay {ui.SEP} {stats.hits}/{stats.lookups} deferred calls hit the cache "
        f"({_rate(stats)})"
        + (f" {ui.SEP} {stats.unanswered} unanswered went live" if stats.unanswered else "")
    )
    if stats.hit_rate is not None and stats.hit_rate < MIN_REPLAY_HIT_RATE:
        ui.error(
            console,
            f"The replay found only {_rate(stats)} of the batch answers "
            f"(at least {MIN_REPLAY_HIT_RATE:.0%} expected)"
            + ("; it stopped early." if stop.is_set() else "."),
            hint=(
                "The replay's LLM requests differ from the ones the batch answered "
                "(changed settings, model or prompts?). Papers not replayed stay "
                "'llm_deferred'; a run without --llm-batch processes them live."
            ),
        )
        return EXIT_FAILURES
    return code


def _write_summary(work: Path, summary: dict) -> None:
    from bibr.local.artifacts import atomic_write_json

    try:
        atomic_write_json(work / SUMMARY_FILENAME, summary, indent=2)
    except OSError:
        pass  # the console carries the same numbers
