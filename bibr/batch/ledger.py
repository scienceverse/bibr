"""The ``outcomes.jsonl`` ledger: one JSON object per attempt, append-only.

Every paper a run touches gets exactly one verdict line per attempt, success
or failure, so the run is auditable even for papers that produced no export.
The local executor also writes a ``started`` line before each paper runs;
one with no verdict after it is a run killed outright (out of memory, a
native crash). :meth:`Ledger.read` returns verdict lines unless asked for
both. Resume semantics read the *latest* verdict per ``paper_id``:

* ``ok`` → skipped (unless ``--force``, or the input is now other bytes);
* ``failed`` → skipped unless ``--retry-failed`` (or ``--force``), except
  (:func:`reruns_by_default`) a paper that never got a verdict — the run was
  interrupted, or the serve refused the token — which is always picked up
  again, and a crash or a service outage (a killed run included), which is
  picked up again until the paper has failed that way
  :data:`MAX_UNSETTLED_FAILURES` times.

Schema (see ``docs/guides/batch.md`` for the table):

``paper_id, stem, path, sha256, bytes, status, error_code, failed_stage,
error, started_at, finished_at, duration_s, stage_times, llm_tokens,
llm_input_tokens, llm_output_tokens, n_refs, n_matched, n_sentences,
warnings, bibr_version, build_sha, executor, attempt, run_id`` plus, for the
remote executor, ``job_id`` and ``retries``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bibr.batch.manifest import BatchItem, resolved_path, sha256_file
from bibr.validation import payload_validation

logger = logging.getLogger(__name__)

LEDGER_FILENAME = "outcomes.jsonl"
ERROR_TEXT_LIMIT = 800
WARNING_SAMPLE = 3
WARNING_TEXT_LIMIT = 200
STARTED = "started"  # the status of a line written before the paper runs
INTERRUPTED = "interrupted"
UPSTREAM_UNAVAILABLE = "upstream_unavailable"
CHUNK_ERROR = "chunk_error"
# The export could not be written: usually the machine's (a full disk, a
# permission), so it counts as a crash below.
OUTPUT_WRITE_FAILED = "output_write_failed"
# Failed lines that say nothing about the paper, which never got a verdict:
# the run was stopped, or the serve refused the token. Resume always runs the
# paper again.
NEVER_RAN_CODES = frozenset({INTERRUPTED, "http_401", "http_403"})
# A crash or a service outage (``chunk_error``, a run killed mid-paper, an
# export that could not be written, ``upstream_unavailable``, or a remote line
# whose transient retries ran out) is usually the machine's or the service's,
# but it can be the paper's: a bug its content triggers, a prompt that brings an
# LLM server down, a model reply the serve reports as a 502. Resume runs such a
# paper again until it has failed this way this many times since its last
# success; then it waits for ``--retry-failed``, so a batch still converges.
# Timeouts are not here at all: a paper can be too slow on its own.
MAX_UNSETTLED_FAILURES = 3


def is_unsettled_failure(entry: Mapping[str, Any]) -> bool:
    """Is this ledger line a crash or a service outage (see ``MAX_UNSETTLED_FAILURES``)?"""
    return (
        entry.get("error_code") in (UPSTREAM_UNAVAILABLE, CHUNK_ERROR, OUTPUT_WRITE_FAILED)
        or entry.get("transient_exhausted") is True
    )


def reruns_by_default(entry: Mapping[str, Any], *, unsettled: int = 1) -> bool:
    """Does a resumed run pick up this failed ledger line without ``--retry-failed``?

    *unsettled* is how many crash or outage lines (and killed attempts) the
    paper has had since its last ``ok`` line, this one included.
    """
    if entry.get("error_code") in NEVER_RAN_CODES:
        return True
    return is_unsettled_failure(entry) and unsettled < MAX_UNSETTLED_FAILURES


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def bounded_text(text: object, limit: int = ERROR_TEXT_LIMIT) -> str | None:
    """Clip *text* to *limit* characters (``None`` stays ``None``)."""
    if text is None:
        return None
    value = str(text)
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def warning_key(warning: str) -> str:
    """Collapse a pre-12.0 prose warning to a frequency key.

    ``VALIDATION:<severity>:<CODE>: message`` keeps its first three segments;
    anything else keeps the text before the first ``:`` (or its first 60
    characters), so the report can count warning *kinds* across a corpus.
    """
    text = warning.strip()
    if text.startswith("VALIDATION:"):
        parts = text.split(":")
        return ":".join(p.strip() for p in parts[:3])
    head, sep, _ = text.partition(":")
    key = head.strip() if sep else text
    return key[:60]


def _warning_entry(warning: object) -> tuple[str, str] | None:
    """``(frequency key, sample text)`` of one export warning; None when malformed.

    A 12.x warning is a ``{code, message}`` object counted by its code; an older
    export's prose string is counted by :func:`warning_key`.
    """
    if isinstance(warning, str):
        return warning_key(warning), warning
    if isinstance(warning, Mapping):
        code, message = warning.get("code"), warning.get("message")
        if isinstance(code, str) and code:
            return code, f"{code}: {message}" if isinstance(message, str) and message else code
    return None


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


def _llm_tokens(usage: object) -> tuple[int, int, int]:
    """``(input, output, total)`` summed over the export's ``llm_usage`` block."""
    if not isinstance(usage, Mapping):
        return 0, 0, 0
    total_in = total_out = total = 0
    for counts in usage.values():
        if not isinstance(counts, Mapping):
            continue
        n_in = _as_int(counts.get("input_tokens", counts.get("prompt_tokens")))
        n_out = _as_int(counts.get("output_tokens", counts.get("completion_tokens")))
        n_total = _as_int(counts.get("total_tokens")) or (n_in + n_out)
        total_in += n_in
        total_out += n_out
        total += n_total
    return total_in, total_out, total


def summarize_export(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per-paper facts for the ledger — never the export itself."""
    if not isinstance(data, Mapping):
        return {}
    extraction = data.get("extraction")
    timings = extraction.get("timings") if isinstance(extraction, Mapping) else None
    stage_values = timings.get("stages", timings) if isinstance(timings, Mapping) else None
    stage_times = (
        {str(k): float(v) for k, v in stage_values.items() if isinstance(v, (int, float))}
        if isinstance(stage_values, Mapping)
        else None
    )
    total_seconds = extraction.get("total_seconds") if isinstance(extraction, Mapping) else None

    # v11 and v12 share the ``extraction.usage``/``timings`` shape; earlier
    # exports carry root ``llm_usage`` and no ``schema_version``.
    if str(data.get("schema_version") or "").split(".")[0] in ("11", "12"):
        extraction = extraction if isinstance(extraction, Mapping) else {}
        total_seconds = timings.get("total_seconds") if isinstance(timings, Mapping) else None
        usage = extraction.get("usage") or {}
        n_in, n_out, n_total = _llm_tokens({"totals": usage.get("totals")})
    else:
        n_in, n_out, n_total = _llm_tokens(data.get("llm_usage"))

    bib = data.get("bib")
    bib_match = data.get("bib_match")
    enrichment = (extraction or {}).get("enrichment", data.get("enrichment"))
    n_refs = len(bib) if isinstance(bib, list) else 0
    if isinstance(bib_match, list):
        n_matched = len(bib_match)
    elif isinstance(enrichment, Mapping):
        n_matched = _as_int(enrichment.get("refs_enriched"))
    else:
        n_matched = 0

    text = data.get("text")
    raw_warnings = (extraction or {}).get("warnings", data.get("processing_warnings"))
    warnings = (
        [entry for entry in map(_warning_entry, raw_warnings) if entry is not None]
        if isinstance(raw_warnings, list)
        else []
    )
    codes: dict[str, int] = {}
    for key, _ in warnings:
        codes[key] = codes.get(key, 0) + 1

    validation = payload_validation(data)
    n_val_errors = n_val_warnings = 0
    if isinstance(validation, Mapping):
        n_val_errors = _as_int(validation.get("errors", validation.get("error_count")))
        n_val_warnings = _as_int(validation.get("warnings", validation.get("warning_count")))

    return {
        "stage_times": stage_times,
        "total_seconds": float(total_seconds)
        if isinstance(total_seconds, (int, float)) and not isinstance(total_seconds, bool)
        else None,
        "llm_tokens": n_total,
        "llm_input_tokens": n_in,
        "llm_output_tokens": n_out,
        "n_refs": n_refs,
        "n_matched": n_matched,
        "n_sentences": len(text) if isinstance(text, list) else 0,
        "n_validation_errors": n_val_errors,
        "n_validation_warnings": n_val_warnings,
        "warnings": {
            "count": len(warnings),
            "first": [
                bounded_text(text, WARNING_TEXT_LIMIT) for _, text in warnings[:WARNING_SAMPLE]
            ],
            "codes": codes,
        },
    }


@dataclass
class Outcome:
    """What happened to one paper in one attempt."""

    status: str  # "ok" | "failed"
    error_code: str | None = None
    failed_stage: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    export: Mapping[str, Any] | None = None
    sha256: str | None = None
    size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class LedgerContext:
    """Run-level fields stamped on every line."""

    run_id: str
    executor: str
    bibr_version: str
    build_sha: str | None


@dataclass
class ResumePlan:
    to_run: list[BatchItem]
    skipped_ok: list[BatchItem]
    skipped_failed: list[BatchItem]
    # Papers whose last attempt never finished: the local executor runs each
    # alone, so one that kills the run takes no chunk neighbours with it.
    unfinished: set[str] = field(default_factory=set)


def _same_input(entry: Mapping[str, Any], item: BatchItem) -> bool:
    """Is *item*'s file the one *entry* recorded? A line without a sha256 cannot tell: yes.

    The file at the recorded path is compared by size, so a resume does not
    read every input again; any other file by its sha256.
    """
    recorded = entry.get("sha256")
    if not isinstance(recorded, str) or not recorded:
        return True
    sha = item.sha256
    if sha is None:
        path = resolved_path(item.path)
        if path is not None and path == resolved_path(entry.get("path")):
            size = entry.get("bytes")
            try:
                return not isinstance(size, int) or item.path.stat().st_size == size
            except OSError:
                return True
        try:
            sha = sha256_file(item.path)
        except OSError:
            return True  # unreadable now: keep the verdict it has
    return sha == recorded


class Ledger:
    """Append-only JSONL ledger at ``<out>/outcomes.jsonl``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._attempts: dict[str, int] | None = None
        # paper_id → (attempt, sha256, bytes) of a started line awaiting its verdict
        self._started: dict[str, tuple[int, str | None, int | None]] = {}
        self._tail_checked = False

    # -- reading ---------------------------------------------------------

    def read(self, *, started: bool = False) -> list[dict[str, Any]]:
        """All well-formed lines, in file order. Malformed lines are skipped.

        ``started`` lines are left out unless *started*: every reader that
        looks for a verdict (the report, the tables) sees only verdicts.
        """
        if not self.path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        bad = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if isinstance(entry, dict) and entry.get("paper_id"):
                    if started or entry.get("status") != STARTED:
                        entries.append(entry)
                else:
                    bad += 1
        if bad:
            logger.warning("%s: skipped %d malformed ledger line(s)", self.path, bad)
        return entries

    def latest(self, entries: Iterable[Mapping[str, Any]] | None = None) -> dict[str, dict]:
        """Latest verdict line per ``paper_id`` (file order wins ties)."""
        out: dict[str, dict] = {}
        for entry in self.read() if entries is None else entries:
            if entry.get("status") != STARTED:
                out[str(entry["paper_id"])] = dict(entry)
        return out

    def attempts(self, entries: Iterable[Mapping[str, Any]] | None = None) -> dict[str, int]:
        """Attempts per ``paper_id``: a verdict closes the attempt its started line opened."""
        counts: dict[str, int] = {}
        open_attempts: set[str] = set()
        for entry in self.read(started=True) if entries is None else entries:
            pid = str(entry["paper_id"])
            if entry.get("status") == STARTED:
                open_attempts.add(pid)
            elif pid in open_attempts:
                open_attempts.discard(pid)
                continue
            counts[pid] = counts.get(pid, 0) + 1
        return counts

    # -- planning --------------------------------------------------------

    def plan(
        self,
        items: Sequence[BatchItem],
        *,
        force: bool = False,
        retry_failed: bool = False,
        entries: Sequence[Mapping[str, Any]] | None = None,
    ) -> ResumePlan:
        """Split *items* into run / skip buckets from the latest verdict per paper.

        *entries* are the ledger lines (``started`` lines included) when the
        caller has read them already. An ``ok`` line counts only for the bytes
        it recorded: a paper whose input is now another file runs again. A
        ``started`` line with no verdict after it is an attempt the run never
        finished — killed by the OOM killer or a native crash — and counts as
        a crash.
        """
        entries = self.read(started=True) if entries is None else entries
        latest = self.latest(entries)
        self._attempts = self.attempts(entries)
        unsettled: dict[str, int] = {}
        unfinished: set[str] = set()
        for entry in entries:
            pid = str(entry["paper_id"])
            status = entry.get("status")
            if status == STARTED:
                if pid in unfinished:  # the attempt before never finished either
                    unsettled[pid] = unsettled.get(pid, 0) + 1
                unfinished.add(pid)
                continue
            unfinished.discard(pid)
            if status == "ok":
                unsettled.pop(pid, None)
            elif is_unsettled_failure(entry):
                unsettled[pid] = unsettled.get(pid, 0) + 1
        for pid in unfinished:
            unsettled[pid] = unsettled.get(pid, 0) + 1
        plan = ResumePlan(to_run=[], skipped_ok=[], skipped_failed=[])
        for item in items:
            pid = item.paper_id
            last = latest.get(pid)
            n_unsettled = unsettled.get(pid, 0)
            if last is not None and last.get("status") == "ok" and not force:
                # The export of that ok line is intact: a later killed attempt
                # (a --force re-run) never replaced it.
                if _same_input(last, item):
                    plan.skipped_ok.append(item)
                else:
                    plan.to_run.append(item)
                continue
            if pid in unfinished:
                plan.unfinished.add(pid)
                runs = retry_failed or n_unsettled < MAX_UNSETTLED_FAILURES
            else:
                runs = (
                    last is None or retry_failed or reruns_by_default(last, unsettled=n_unsettled)
                )
            if force or runs:
                plan.to_run.append(item)
            else:
                plan.skipped_failed.append(item)
        return plan

    # -- writing ---------------------------------------------------------

    def append(self, entry: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(entry), ensure_ascii=False, default=str) + "\n"
        if not self._tail_checked:
            # A run killed mid-append (OOM, SIGKILL, a full disk) leaves a
            # last line with no newline; appending straight after it would
            # glue this record onto the fragment and lose both.
            self._tail_checked = True
            if self._ends_mid_line():
                line = "\n" + line
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()

    def _ends_mid_line(self) -> bool:
        try:
            with self.path.open("rb") as handle:
                if handle.seek(0, os.SEEK_END) == 0:
                    return False
                handle.seek(-1, os.SEEK_END)
                return handle.read(1) != b"\n"
        except OSError:
            return False

    def next_attempt(self, paper_id: str) -> int:
        if self._attempts is None:
            self._attempts = self.attempts()
        count = self._attempts.get(paper_id, 0) + 1
        self._attempts[paper_id] = count
        return count

    def start(self, item: BatchItem, *, context: LedgerContext) -> dict[str, Any]:
        """Append a ``started`` line for *item* just before it runs, and return it.

        Its verdict (:meth:`record`) repeats the attempt number. A run killed
        before the verdict leaves the line alone, which :meth:`plan` counts
        as a crash.
        """
        sha256, size = item.sha256, None
        try:
            sha256 = sha256 or sha256_file(item.path)
            size = item.path.stat().st_size
        except OSError:
            pass
        attempt = self.next_attempt(item.paper_id)
        self._started[item.paper_id] = (attempt, sha256, size)
        entry: dict[str, Any] = {
            "paper_id": item.paper_id,
            "stem": item.stem,
            "path": str(item.path),
            "sha256": sha256,
            "bytes": size,
            "status": STARTED,
            "started_at": utc_now_iso(),
            "bibr_version": context.bibr_version,
            "build_sha": context.build_sha,
            "executor": context.executor,
            "attempt": attempt,
            "run_id": context.run_id,
        }
        self.append(entry)
        return entry

    def record(
        self,
        item: BatchItem,
        outcome: Outcome,
        *,
        context: LedgerContext,
    ) -> dict[str, Any]:
        """Append one line for *item* and return it.

        ``sha256``/``bytes`` come from the outcome when the executor already
        read the file (remote uploads), or from the :meth:`start` line of
        this attempt, else are computed here.
        """
        sha256 = outcome.sha256
        size = outcome.size
        attempt: int | None = None
        if (started := self._started.pop(item.paper_id, None)) is not None:
            attempt, started_sha256, started_size = started
            sha256 = sha256 or started_sha256
            size = size if size is not None else started_size
        if sha256 is None or size is None:
            try:
                sha256 = sha256 or sha256_file(item.path)
                size = size if size is not None else item.path.stat().st_size
            except OSError:
                pass
        summary = summarize_export(outcome.export) if outcome.ok else {}
        duration = outcome.duration_s
        if duration is None and summary.get("total_seconds") is not None:
            duration = summary["total_seconds"]
        entry: dict[str, Any] = {
            "paper_id": item.paper_id,
            "stem": item.stem,
            "path": str(item.path),
            "sha256": sha256,
            "bytes": size,
            "status": outcome.status,
            "error_code": outcome.error_code,
            "failed_stage": outcome.failed_stage,
            "error": bounded_text(outcome.error),
            "started_at": outcome.started_at,
            "finished_at": outcome.finished_at or utc_now_iso(),
            "duration_s": round(duration, 3) if isinstance(duration, (int, float)) else None,
            "pipeline_seconds": summary.get("total_seconds"),
            "stage_times": summary.get("stage_times"),
            "llm_tokens": summary.get("llm_tokens"),
            "llm_input_tokens": summary.get("llm_input_tokens"),
            "llm_output_tokens": summary.get("llm_output_tokens"),
            "n_refs": summary.get("n_refs"),
            "n_matched": summary.get("n_matched"),
            "n_sentences": summary.get("n_sentences"),
            "n_validation_errors": summary.get("n_validation_errors"),
            "n_validation_warnings": summary.get("n_validation_warnings"),
            "warnings": summary.get("warnings"),
            "bibr_version": context.bibr_version,
            "build_sha": context.build_sha,
            "executor": context.executor,
            "attempt": attempt if attempt is not None else self.next_attempt(item.paper_id),
            "run_id": context.run_id,
        }
        entry.update(outcome.extra)
        self.append(entry)
        return entry
