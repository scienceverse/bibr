"""``bibr batch --llm-batch front-matter``: a corpus run around Anthropic Message Batches.

Three phases, each resumable by running the same command again:

1. **Round 1** runs the corpus as ``bibr batch`` does, with the per-paper
   front-matter LLM calls deferred (:mod:`bibr.clients.batch_defer`): a call
   that misses the LLM response cache is spooled to
   ``<out>/llm_batch/spool.jsonl`` and its paper fails softly with
   ``llm_deferred``, a ledger code resume always runs again.
2. **Fill**: the requests round 1 spooled, one per cache key and without the
   ones the cache already holds, go out as Message Batches
   (:mod:`bibr.clients.batch`) of at most 100,000 requests or 256 MB each, all
   submitted before the wait. ``plan.json`` and one state file per batch let
   an interrupted run (or one stopped by ``--deadline``) collect the same
   batches instead of paying for new ones. Every answer that validates is
   stored in the cache under its spooled key, exactly as the live call would
   have stored it; an errored, expired or invalid row stores nothing, so that
   call goes live in the replay. When more than 5 % of the requests (and at
   least 20) have no answer, the run stops before the replay instead.
3. **Round 2** replays the papers round 1 deferred. Their deferred calls
   should now hit the cache; the run reports the hit rate and stops loudly
   under 95 %, because then the replay's requests no longer match what the
   batch answered and every miss would be paid for again at the full price.

The exit code and the ``--json`` report cover both rounds.

Message Batches cost half the token price. For bibr's front matter that is a
small saving per paper (under a cent on Haiku 4.5); it matters for hosted runs
of 100,000 papers or more, not for a lab's corpus.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import json
import logging
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bibr.batch.ledger import LEDGER_FILENAME, LLM_DEFERRED, Ledger, utc_now_iso
from bibr.batch.report import compute_report
from bibr.batch.runner import (
    EXIT_FAILURES,
    EXIT_INTERRUPTED,
    EXIT_OK,
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
    iter_spool,
    observing,
    read_row,
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
# A batch that leaves more than this share of its requests unanswered (and at
# least MIN_UNANSWERED_TO_STOP of them) stops the run before the replay, which
# would send them all live at the full price.
MAX_UNANSWERED_SHARE = 0.05
MIN_UNANSWERED_TO_STOP = 20
# A batch ends within 24 hours (rows not done by then expire); wait a little
# longer than that before giving up on one.
BATCH_TIMEOUT_S = 25 * 3600

# Where a spooled request stands in the plan (its latest part decides).
_PENDING = "pending"  # in a batch not collected yet
_ANSWERED = "answered"  # a collected batch answered it
_UNANSWERED = "unanswered"  # a collected batch did not: it goes live in the replay
# A spool row that cannot become a batch request any more.
_UNUSABLE = (KeyError, TypeError, ValueError, ImportError, AttributeError, OSError)


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
    from bibr.clients.batch import transport_problem

    return transport_problem()


@dataclass
class FillResult:
    """What the fill phase did, for the summary."""

    requests: int = 0  # unique requests spooled since the fill's offset
    cached: int = 0  # in the cache already (an earlier fill, or a live call)
    stale: int = 0  # spooled under another provider or model, or unusable
    lost: int = 0  # answered by an earlier batch but gone from the cache: batched again
    submitted: int = 0  # requests in the batches this fill collected
    batches: int = 0
    filled: int = 0  # answers stored in the cache
    unanswered: set[str] = field(default_factory=set)  # requests no batch answered
    go_live: set[str] = field(default_factory=set)  # unanswered and not cached since
    errors: Counter = field(default_factory=Counter)  # go_live by error type


@dataclass(frozen=True)
class _Spooled:
    """What the fill keeps of a spooled request: the row is read again by offset."""

    offset: int
    provider: Any
    model: Any
    schema: Any
    label: Any


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


def _part_state(part: dict) -> dict[str, str]:
    """``key → where it stands`` for the keys of one plan part."""
    if not part.get("done"):
        return dict.fromkeys(part["keys"], _PENDING)
    unanswered = part.get("unanswered") or {}
    return {key: _UNANSWERED if key in unanswered else _ANSWERED for key in part["keys"]}


def fill_cache(
    work: Path,
    settings: Any,
    *,
    since: int = 0,
    batch_client: Any = None,
    poll_interval: float | None = None,
    timeout_s: float = BATCH_TIMEOUT_S,
    progress: Any = None,
) -> FillResult:
    """Submit the spooled requests the cache lacks and store the answers.

    Only requests spooled from byte *since* on are planned (round 1 of this
    run, whose papers the replay runs again); batches planned earlier are
    collected whatever they hold. A request a collected batch did not answer
    is not planned again: it goes live in the replay rather than into another
    batch. One whose answer is no longer in the cache (a cleared cache,
    another ``CACHE_LLM_DIR``) is. New batches are all submitted before the
    wait, so they run side by side, and the wait ends after *timeout_s* in all
    (``TimeoutError``; the next call collects the same batches).

    The spool is read one line at a time and only one batch's requests are
    built at once: a corpus-scale spool does not have to fit in memory.
    """
    from bibr.clients.batch import AnthropicBatchAdapter, BatchError, partition
    from bibr.clients.llm_cache import LlmResponseCache, response_body

    result = FillResult()
    spool = work / SPOOL_FILENAME
    index: dict[str, _Spooled] = {}
    current: set[str] = set()
    for offset, row in iter_spool(spool):
        key = row["key"]
        if key not in index:
            index[key] = _Spooled(
                offset, row.get("provider"), row.get("model"), row.get("schema"), row.get("label")
            )
        if offset >= since:
            current.add(key)
    result.requests = len(current)
    cache = LlmResponseCache(settings=settings)
    model = settings.llm.model
    plan_path = work / PLAN_FILENAME
    plan = _read_plan(plan_path)
    state: dict[str, str] = {}
    for part in plan["parts"]:
        state.update(_part_state(part))

    adapters: dict[str, AnthropicBatchAdapter] = {}

    def adapter_for(part_model: str) -> AnthropicBatchAdapter:
        if part_model not in adapters:
            adapters[part_model] = AnthropicBatchAdapter(
                model=part_model, settings=settings, client=batch_client
            )
        return adapters[part_model]

    with contextlib.ExitStack() as stack:
        handle = stack.enter_context(spool.open("rb")) if index else None

        def load(key: str):
            if handle is None:
                raise KeyError(key)
            return request_for(read_row(handle, index[key].offset))

        def fresh() -> Iterator[tuple[str, int]]:
            sizer = adapter_for(model)
            for key, spooled in index.items():
                if key not in current:
                    continue
                where = state.get(key)
                if where in (_PENDING, _UNANSWERED):
                    continue  # being collected, or goes live in the replay
                if spooled.provider != "anthropic" or spooled.model != model:
                    result.stale += 1
                    continue
                if cache.get(key) is not None:
                    result.cached += 1
                    continue
                try:
                    request = load(key)
                except _UNUSABLE as exc:
                    logger.warning("Skipping unusable spool row %s: %s", key, exc)
                    result.stale += 1
                    continue
                if where == _ANSWERED:
                    result.lost += 1
                yield key, sizer.request_bytes(request)

        new_parts = list(partition(fresh()))
        for keys in new_parts:
            number = len(plan["parts"]) + 1
            plan["parts"].append(
                {
                    "model": model,
                    "keys": keys,
                    "done": False,
                    # Unique, so a plan started over never resumes an old batch.
                    "state": f"part-{number:04d}-{uuid.uuid4().hex[:8]}.state.json",
                }
            )
            state.update(dict.fromkeys(keys, _PENDING))
        if new_parts:
            _write_plan(plan_path, plan)

        def batch_of(part: dict) -> tuple[list, dict[str, str]]:
            requests, unusable = [], {}
            for key in part["keys"]:
                try:
                    requests.append(load(key))
                except _UNUSABLE:
                    unusable[key] = "unusable_spool_row"
            return requests, unusable

        def state_path(number: int, part: dict) -> Path:
            return work / (part.get("state") or f"part-{number:04d}.state.json")

        waiting = [(n, part) for n, part in enumerate(plan["parts"], start=1) if not part["done"]]
        for number, part in waiting:
            requests, _ = batch_of(part)
            if requests:
                adapter_for(part["model"]).ensure_submitted(
                    requests, state_path=state_path(number, part)
                )
            del requests

        poll = settings.llm.batch_poll_interval_s if poll_interval is None else poll_interval
        wait_until = time.monotonic() + timeout_s
        for number, part in waiting:
            requests, unanswered = batch_of(part)
            if requests:
                answers = adapter_for(part["model"]).run(
                    requests,
                    state_path=state_path(number, part),
                    poll_interval=poll,
                    timeout_s=max(0.0, wait_until - time.monotonic()),
                    progress=(lambda text, n=number: progress(n, text)) if progress else None,
                )
                result.batches += 1
                result.submitted += len(requests)
                for request in requests:
                    key = request.custom_id
                    answer = answers.get(key)
                    spooled = index[key]
                    if answer is None or isinstance(answer, BatchError):
                        unanswered[key] = answer.error_type if answer else "missing"
                    elif cache.put(
                        key,
                        response_body(answer),
                        model=spooled.model,
                        schema_name=spooled.schema,
                        label=spooled.label,
                        source="batch",
                    ):
                        result.filled += 1
                    else:
                        unanswered[key] = "cache_write_failed"
                del answers
            del requests
            part["done"] = True
            part["unanswered"] = dict(sorted(unanswered.items()))
            state.update(_part_state(part))
            _write_plan(plan_path, plan)

    reasons: dict[str, str] = {}
    for part in plan["parts"]:
        unanswered = (part.get("unanswered") or {}) if part.get("done") else {}
        if isinstance(unanswered, list):  # keys alone, no reasons
            unanswered = dict.fromkeys(unanswered, "missing")
        reasons.update(unanswered)
    result.unanswered = {key for key in current if state.get(key) == _UNANSWERED}
    result.go_live = {key for key in result.unanswered if cache.get(key) is None}
    result.errors = Counter(reasons.get(key, "missing") for key in result.go_live)
    return result


def _deferred_papers(out: Path, run_id: str) -> set[str]:
    """Papers whose latest verdict is round *run_id*'s ``llm_deferred``.

    A paper deferred by an earlier run and not reached by this round 1 (a
    ``--limit``, say) waits for the next run, whose round 1 spools it again.
    """
    latest: dict[str, dict] = {}
    for entry in Ledger(out / LEDGER_FILENAME).read():
        latest[str(entry.get("paper_id"))] = entry
    return {
        pid
        for pid, entry in latest.items()
        if entry.get("error_code") == LLM_DEFERRED and entry.get("run_id") == run_id
    }


def _deferred_calls(spool: Path, since: int) -> set[tuple[str, str]]:
    """``(file hash, label)`` of every call round 1 deferred (spooled from byte *since*)."""
    return {
        (row["file_hash"], row["label"])
        for offset, row in iter_spool(spool)
        if offset >= since
        and isinstance(row.get("file_hash"), str)
        and isinstance(row.get("label"), str)
    }


def _spool_size(spool: Path) -> int:
    try:
        return spool.stat().st_size
    except OSError:
        return 0


def _rate(stats: ReplayStats) -> str:
    rate = stats.hit_rate
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _own_entries(out: Path, run_ids: list[str]) -> list[dict]:
    """The ledger verdicts of this invocation's rounds."""
    if not run_ids:
        return []
    ids = set(run_ids)
    return [e for e in Ledger(out / LEDGER_FILENAME).read() if e.get("run_id") in ids]


def _failed_papers(entries: list[dict]) -> list[str]:
    """Papers whose latest verdict is a failure of their own (not a deferral)."""
    latest: dict[str, dict] = {}
    for entry in entries:
        latest[str(entry.get("paper_id"))] = entry
    return sorted(
        pid
        for pid, entry in latest.items()
        if entry.get("status") != "ok" and entry.get("error_code") != LLM_DEFERRED
    )


def run_llm_batch(
    options: BatchOptions,
    *,
    scope: str = "front-matter",
    console: Any = None,
    settings: Any = None,
    batch_client: Any = None,
) -> int:
    """Run ``bibr batch --llm-batch``; returns the process exit code.

    The exit code covers both rounds: 1 when a paper failed in either (a
    paper still waiting for its replay at a deadline is not a failure), 130
    after an interruption, 2 on a usage error. With ``--json`` one report
    over both rounds goes to stdout whenever round 1 ran.
    """
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
    run_ids: list[str] = []

    def conclude(code: int | None) -> int:
        """Write the summary, print the --json report, and return the exit code.

        *code* forces an exit code; ``None`` derives it from the ledger.
        """
        summary["finished_at"] = utc_now_iso()
        _write_summary(work, summary)
        entries = _own_entries(out, run_ids)
        if options.report_json:
            report = compute_report(entries)
            report["run_ids"] = list(run_ids)
            report["llm_batch"] = summary
            print(json.dumps(report, indent=2))
        if code is not None:
            return code
        return EXIT_FAILURES if _failed_papers(entries) else EXIT_OK

    def past_deadline() -> bool:
        return options.deadline is not None and time.time() >= options.deadline

    ui.section(console, "LLM batch · round 1: run, deferring the front-matter LLM calls")
    # Each round's own report covers only that round: with --json the one
    # report over both rounds is printed at the end instead.
    first = dataclasses.replace(options, print_report=not options.report_json)
    since = _spool_size(spool)
    with collecting(spool, labels) as collect:
        code = run_batch(first, console=console, on_run_id=run_ids.append)
    summary["round1"] = {"exit_code": code, "spooled": collect.spooled}
    if not run_ids:
        return code  # it never started (no inputs, a busy --out, a failed preflight)
    if collect.write_failures:
        ui.warn(console, f"{collect.write_failures} calls could not be spooled and went live")
    if code in (EXIT_USAGE, EXIT_INTERRUPTED):
        return conclude(code)
    if past_deadline():
        console.print("  [dim]deadline reached — re-run the same command to continue[/dim]")
        return conclude(None)
    deferred = _deferred_papers(out, run_ids[0])
    summary["deferred_papers"] = len(deferred)
    if not deferred:
        console.print("  [dim]no paper is waiting for the LLM batch[/dim]")
        return conclude(None)

    ui.section(console, f"LLM batch · {len(deferred)} papers wait for a Message Batch")

    def show(part: int, text: str) -> None:
        console.print(f"  [dim]batch {part} {ui.SEP} {text}[/dim]")

    timeout_s: float = BATCH_TIMEOUT_S
    if options.deadline is not None:
        timeout_s = max(0.0, min(timeout_s, options.deadline - time.time()))
    try:
        with out_dir_lock(out):
            fill = fill_cache(
                work,
                settings,
                since=since,
                batch_client=batch_client,
                timeout_s=timeout_s,
                progress=show,
            )
    except OutDirBusy as exc:
        ui.error(console, f"{exc}.", hint="Wait for it to finish, or pass another --out.")
        return conclude(EXIT_USAGE)
    except KeyboardInterrupt:
        ui.error(
            console,
            "Interrupted while waiting for the Message Batch.",
            hint="It keeps running at Anthropic; re-run the same command to collect it.",
        )
        return conclude(EXIT_INTERRUPTED)
    except TimeoutError as exc:
        summary["fill"] = {"timed_out": True}
        if options.deadline is not None and timeout_s < BATCH_TIMEOUT_S:
            console.print(
                "  [dim]deadline reached while the Message Batch runs at Anthropic — "
                "re-run the same command to collect it and continue[/dim]"
            )
            return conclude(None)
        ui.error(
            console,
            f"The Message Batch did not end in time: {exc}",
            hint="Re-run the same command to keep waiting for it.",
        )
        return conclude(EXIT_FAILURES)
    except Exception as exc:  # noqa: BLE001 — any API failure ends this invocation
        ui.error(
            console,
            f"The Message Batch step failed: {type(exc).__name__}: {exc}",
            hint=(
                "Re-run the same command to resume it. Running without --llm-batch "
                "processes the deferred papers with live calls instead."
            ),
        )
        return conclude(EXIT_FAILURES)
    summary["fill"] = {
        "requests": fill.requests,
        "already_cached": fill.cached,
        "stale": fill.stale,
        "batched_again": fill.lost,
        "batches": fill.batches,
        "submitted": fill.submitted,
        "filled": fill.filled,
        "unanswered": len(fill.go_live),
        "errors": dict(fill.errors),
    }
    errors = ", ".join(f"{kind} ×{n}" for kind, n in fill.errors.most_common())
    console.print(
        f"  {fill.requests} deferred requests {ui.SEP} {fill.submitted} sent in "
        f"{fill.batches} batches {ui.SEP} {fill.filled} answers cached"
        + (f" {ui.SEP} {len(fill.go_live)} unanswered ({errors})" if fill.go_live else "")
    )
    if fill.lost:
        console.print(
            f"  [dim]{fill.lost} answers of an earlier batch were no longer in the LLM "
            "cache and were batched again[/dim]"
        )
    unanswered = len(fill.go_live)
    if unanswered >= MIN_UNANSWERED_TO_STOP and unanswered > MAX_UNANSWERED_SHARE * fill.requests:
        summary["stopped"] = "unanswered"
        ui.error(
            console,
            f"The Message Batch left {unanswered} of {fill.requests} requests unanswered "
            f"({errors}); round 2 would send them live at the full price, so it did not run.",
            hint=(
                "Run without --llm-batch to process the deferred papers with live calls. "
                f"To batch them again instead, fix the cause, delete {work / PLAN_FILENAME} "
                "and re-run this command."
            ),
        )
        return conclude(EXIT_FAILURES)
    if fill.go_live:
        console.print(f"  [dim]the {unanswered} unanswered requests go live in round 2[/dim]")

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
        print_report=not options.report_json,
    )
    started = time.monotonic()
    with observing(
        _deferred_calls(spool, since),
        labels=labels,
        unanswered_keys=fill.unanswered,
        on_update=watch,
    ) as stats:
        code = run_batch(replay, console=console, stop=stop, on_run_id=run_ids.append)
    summary["round2"] = {
        "exit_code": code,
        "seconds": round(time.monotonic() - started, 1),
        "hits": stats.hits,
        "misses": stats.misses,
        "unanswered": stats.unanswered,
        "hit_rate": stats.hit_rate,
        "stopped_early": stop.is_set(),
    }

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
                "(changed settings, model or prompts?), or the batch answers are not in "
                "this LLM cache (another CACHE_LLM_DIR?). Papers not replayed stay "
                "'llm_deferred'; a run without --llm-batch processes them live."
            ),
        )
        return conclude(EXIT_FAILURES)
    if code in (EXIT_USAGE, EXIT_INTERRUPTED):
        return conclude(code)
    return conclude(None)


def _write_summary(work: Path, summary: dict) -> None:
    from bibr.local.artifacts import atomic_write_json

    try:
        # Private like the spool beside it (it holds prompts from the papers).
        work.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_write_json(work / SUMMARY_FILENAME, summary, indent=2)
    except (OSError, ValueError):
        pass  # the console carries the same numbers
