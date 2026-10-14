"""Sample a bibr run's memory from outside the process, and join it to the ledger.

``sample`` reads ``/proc`` twice a second and writes one JSON line per tick:
for each group of processes, the summed resident memory (RSS), proportional
set size (PSS, which splits shared pages between the processes that map
them), swap, and the highest peak RSS (``VmHWM``) of any process in it, plus
the host's available memory and swap use. A group is a process tree: a root
pid (``--pid`` for the bibr run, ``--server NAME=PID`` for a model server)
or the processes whose command line matches a pattern (``--match
NAME=REGEX``, looked up again every tick, so it follows a process that
restarts), with all of their descendants.

``join`` lines the samples up with a ``bibr batch`` ledger's ``started_at`` /
``finished_at`` and reports each group's peak per chunk and per paper. The
local executor runs a chunk of papers at once and stamps them all with the
chunk's times, so a local run's peaks are per chunk.

    python scripts/mem_sampler.py sample --pid "$(pgrep -f 'bibr batch')" \\
        --server llm=12345 --out mem.jsonl
    python scripts/mem_sampler.py sample --match run='bibr batch' \\
        --match llm='vllm serve' --out mem.jsonl --duration 3600
    python scripts/mem_sampler.py join mem.jsonl results/outcomes.jsonl

Linux only; standard library only. Ledger times have one-second resolution.
With ``PIPELINE_MEMORY_PROFILE`` on, bibr resets its own ``VmHWM`` at every
stage, so the sampled peak then covers the current stage only.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import signal
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

MIB = 1024 * 1024
DEFAULT_INTERVAL = 0.5
# Ledger times are truncated to the second, so a paper can finish up to a
# second after its ``finished_at``.
LEDGER_SLACK_S = 1.0
STARTED = "started"


# --- /proc ------------------------------------------------------------------


def parse_stat(text: str) -> tuple[int, str, int]:
    """``(pid, comm, ppid)`` from a ``/proc/<pid>/stat`` line.

    ``comm`` sits in parentheses and may itself hold spaces and parentheses,
    so the fields after it are split from the last ``)``.
    """
    head, _, tail = text.rpartition(")")
    pid_text, _, comm = head.partition(" (")
    fields = tail.split()
    return int(pid_text), comm, int(fields[1])


def parse_kib_fields(text: str) -> dict[str, int]:
    """``{"VmRSS": bytes, ...}`` from ``Key:   123 kB`` lines (status, smaps, meminfo)."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        parts = rest.split()
        if len(parts) == 2 and parts[1] == "kB" and parts[0].isdigit():
            values[key.strip()] = int(parts[0]) * 1024
    return values


@dataclass(frozen=True)
class ProcessUsage:
    """One process's memory, in bytes (``None`` where the kernel would not say)."""

    rss: int
    hwm: int | None
    swap: int | None
    pss: int | None


class ProcFS:
    """The few ``/proc`` files the sampler reads, under a configurable root."""

    def __init__(self, root: Path | str = "/proc") -> None:
        self.root = Path(root)

    def _read(self, *parts: str | int) -> str | None:
        try:
            return self.root.joinpath(*map(str, parts)).read_text(errors="replace")
        except OSError:
            return None

    def parents(self) -> dict[int, int]:
        """``{pid: ppid}`` for every process now running."""
        parents: dict[int, int] = {}
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return parents
        for entry in entries:
            if not entry.name.isdigit():
                continue
            text = self._read(entry.name, "stat")
            if text is None:
                continue  # it exited during the scan
            try:
                pid, _comm, ppid = parse_stat(text)
            except (ValueError, IndexError):
                continue
            parents[pid] = ppid
        return parents

    def alive(self, pid: int) -> bool:
        return self.root.joinpath(str(pid), "stat").exists()

    def cmdline(self, pid: int) -> str:
        text = self._read(pid, "cmdline")
        return "" if text is None else text.replace("\0", " ").strip()

    def usage(self, pid: int, *, pss: bool = True) -> ProcessUsage | None:
        """The process's memory; ``None`` when it is gone or has no memory (a kernel thread)."""
        status = self._read(pid, "status")
        if status is None:
            return None
        fields = parse_kib_fields(status)
        if "VmRSS" not in fields:
            return None
        pss_bytes = None
        if pss:
            rollup = self._read(pid, "smaps_rollup")
            if rollup is not None:
                pss_bytes = parse_kib_fields(rollup).get("Pss")
        return ProcessUsage(
            rss=fields["VmRSS"],
            hwm=fields.get("VmHWM"),
            swap=fields.get("VmSwap"),
            pss=pss_bytes,
        )

    def meminfo(self) -> dict[str, int]:
        text = self._read("meminfo")
        return {} if text is None else parse_kib_fields(text)


def descendants(parents: Mapping[int, int], roots: Iterable[int]) -> set[int]:
    """*roots* that are running, and every process below them."""
    children: dict[int, list[int]] = {}
    for pid, ppid in parents.items():
        children.setdefault(ppid, []).append(pid)
    tree: set[int] = set()
    stack = [pid for pid in roots if pid in parents]
    while stack:
        pid = stack.pop()
        if pid in tree:
            continue
        tree.add(pid)
        stack.extend(children.get(pid, ()))
    return tree


def ancestors(parents: Mapping[int, int], pid: int) -> set[int]:
    """Every process above *pid*."""
    seen: set[int] = set()
    while (pid := parents.get(pid, 0)) > 0 and pid not in seen:
        seen.add(pid)
    return seen


@dataclass(frozen=True)
class Group:
    """A named process tree: fixed root pids, or roots found by command-line pattern."""

    name: str
    pids: tuple[int, ...] = ()
    pattern: re.Pattern[str] | None = None

    def roots(self, proc: ProcFS, parents: Mapping[int, int], exclude: set[int]) -> set[int]:
        if self.pattern is None:
            return {pid for pid in self.pids if pid in parents}
        matched = {
            pid for pid in parents if pid not in exclude and self.pattern.search(proc.cmdline(pid))
        }
        # The topmost matches: a matching child is already in its parent's tree.
        return {pid for pid in matched if not (ancestors(parents, pid) & matched)}


def _mib(value: int | None) -> float | None:
    return None if value is None else round(value / MIB, 1)


def sample_once(
    proc: ProcFS,
    groups: Sequence[Group],
    *,
    pss: bool = True,
    exclude: set[int] | None = None,
    wall: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """One sample line: each group's summed memory and the host's.

    Processes in *exclude* (the sampler itself) are never counted, even when
    the sampler runs inside a sampled tree.
    """
    parents = proc.parents()
    exclude = exclude or set()
    rows: dict[str, Any] = {}
    for group in groups:
        rss = swap = pss_total = 0
        hwm: int | None = None
        pss_complete = pss
        count = 0
        tree = descendants(parents, group.roots(proc, parents, exclude)) - exclude
        for pid in sorted(tree):
            usage = proc.usage(pid, pss=pss)
            if usage is None:
                continue
            count += 1
            rss += usage.rss
            swap += usage.swap or 0
            if usage.hwm is not None:
                hwm = usage.hwm if hwm is None else max(hwm, usage.hwm)
            if usage.pss is None:
                pss_complete = False
            else:
                pss_total += usage.pss
        # No process (not started yet, or gone): no reading, rather than zeros.
        rows[group.name] = {
            "pids": count,
            "rss_mib": _mib(rss) if count else None,
            # A process whose smaps the kernel would not show (another user's)
            # would make the sum an undercount, so the group gets none.
            "pss_mib": _mib(pss_total) if pss_complete and count else None,
            "swap_mib": _mib(swap) if count else None,
            "hwm_mib": _mib(hwm),
        }
    meminfo = proc.meminfo()
    swap_used = (
        meminfo["SwapTotal"] - meminfo["SwapFree"]
        if "SwapTotal" in meminfo and "SwapFree" in meminfo
        else None
    )
    now = wall()
    return {
        "t": round(now, 3),
        "time": datetime.fromtimestamp(now, tz=UTC).isoformat(timespec="milliseconds"),
        "groups": rows,
        "mem_available_mib": _mib(meminfo.get("MemAvailable")),
        "swap_used_mib": _mib(swap_used),
    }


def run_sampler(
    proc: ProcFS,
    groups: Sequence[Group],
    out: TextIO,
    *,
    interval: float = DEFAULT_INTERVAL,
    duration: float | None = None,
    watch: Sequence[int] = (),
    pss: bool = True,
    exclude: set[int] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wall: Callable[[], float] = time.time,
) -> int:
    """Write samples to *out* every *interval* seconds; return how many were written.

    Stops after *duration* seconds, when *should_stop* says so, or once none
    of the *watch* pids is running any more (after one last sample).
    """
    started = clock()
    next_tick = started
    written = 0
    while not should_stop():
        line = sample_once(proc, groups, pss=pss, exclude=exclude, wall=wall)
        out.write(json.dumps(line, separators=(",", ":")) + "\n")
        out.flush()
        written += 1
        if watch and not any(proc.alive(pid) for pid in watch):
            break
        next_tick += interval
        now = clock()
        if duration is not None and next_tick - started > duration:
            break
        if next_tick > now:
            sleep(next_tick - now)
        else:
            next_tick = now  # a slow tick: carry on from here, do not burst
    return written


# --- join -------------------------------------------------------------------


def _epoch(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The object lines of *path*; a torn last line (a killed writer) is skipped."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples = [row for row in read_jsonl(path) if isinstance(row.get("t"), (int, float))]
    return sorted(samples, key=lambda row: row["t"])


@dataclass
class Interval:
    """Ledger lines that share one ``started_at``/``finished_at``: a local chunk."""

    start: float
    end: float
    rows: list[dict[str, Any]]


def ledger_intervals(rows: Iterable[Mapping[str, Any]]) -> list[Interval]:
    intervals: dict[tuple[float, float], Interval] = {}
    for row in rows:
        if row.get("status") == STARTED:
            continue
        start, end = _epoch(row.get("started_at")), _epoch(row.get("finished_at"))
        if start is None or end is None or end < start:
            continue
        key = (start, end)
        if key not in intervals:
            intervals[key] = Interval(start, end, [])
        intervals[key].rows.append(dict(row))
    return sorted(intervals.values(), key=lambda interval: (interval.start, interval.end))


def _max(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _min(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _percentile(values: Sequence[float], pct: float) -> float | None:
    """Linear-interpolated percentile, as ``bibr batch report`` computes it."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _spread(values: Sequence[float]) -> dict[str, Any]:
    def rounded(value: float | None) -> float | None:
        return None if value is None else round(value, 1)

    return {
        "n": len(values),
        "p50": rounded(_percentile(values, 0.5)),
        "p90": rounded(_percentile(values, 0.9)),
        "max": rounded(max(values)) if values else None,
    }


def _window_peaks(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted({name for sample in samples for name in (sample.get("groups") or {})})
    groups = {}
    for name in names:
        rows = [(sample.get("groups") or {}).get(name) or {} for sample in samples]
        groups[name] = {
            "peak_rss_mib": _max(row.get("rss_mib") for row in rows),
            "peak_pss_mib": _max(row.get("pss_mib") for row in rows),
            "max_swap_mib": _max(row.get("swap_mib") for row in rows),
            "peak_hwm_mib": _max(row.get("hwm_mib") for row in rows),
        }
    return {
        "samples": len(samples),
        "groups": groups,
        "min_mem_available_mib": _min(sample.get("mem_available_mib") for sample in samples),
        "max_swap_used_mib": _max(sample.get("swap_used_mib") for sample in samples),
    }


def join(
    samples: Sequence[Mapping[str, Any]],
    ledger_rows: Iterable[Mapping[str, Any]],
    *,
    slack: float = LEDGER_SLACK_S,
) -> dict[str, Any]:
    """Each ledger interval's peaks, each paper's, and their spread per group."""
    intervals = ledger_intervals(ledger_rows)
    samples = sorted(samples, key=lambda sample: float(sample["t"]))
    times = [float(sample["t"]) for sample in samples]
    interval_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    overlapping = 0
    previous_end: float | None = None
    for interval in intervals:
        if previous_end is not None and interval.start < previous_end:
            overlapping += 1
        previous_end = interval.end if previous_end is None else max(previous_end, interval.end)
        lo = bisect.bisect_left(times, interval.start)
        hi = bisect.bisect_left(times, interval.end + slack)
        window = samples[lo:hi]
        peaks = _window_peaks(window)
        interval_rows.append(
            {
                "started_at": interval.rows[0].get("started_at"),
                "finished_at": interval.rows[0].get("finished_at"),
                "papers": len(interval.rows),
                **peaks,
            }
        )
        for row in interval.rows:
            paper_rows.append(
                {
                    "paper_id": row.get("paper_id"),
                    "stem": row.get("stem"),
                    "status": row.get("status"),
                    "attempt": row.get("attempt"),
                    "started_at": row.get("started_at"),
                    "finished_at": row.get("finished_at"),
                    "chunk_papers": len(interval.rows),
                    **peaks,
                }
            )

    sampled = [row for row in interval_rows if row["samples"]]
    names = sorted({name for row in sampled for name in row["groups"]})
    summary = {}
    for name in names:
        summary[name] = {
            key: _spread(
                [
                    value
                    for row in sampled
                    if (value := row["groups"].get(name, {}).get(key)) is not None
                ]
            )
            for key in ("peak_rss_mib", "peak_pss_mib", "max_swap_mib")
        }
    return {
        "samples": len(samples),
        "sampled_from": samples[0].get("time") if samples else None,
        "sampled_to": samples[-1].get("time") if samples else None,
        "ledger_lines": sum(len(interval.rows) for interval in intervals),
        "intervals": len(intervals),
        "unsampled_intervals": len(intervals) - len(sampled),
        "overlapping_intervals": overlapping,
        "per_interval": summary,
        "min_mem_available_mib": _min(row["min_mem_available_mib"] for row in sampled),
        "max_swap_used_mib": _max(row["max_swap_used_mib"] for row in sampled),
        "chunks": interval_rows,
        "papers": paper_rows,
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}"


def _fmt_spread(spread: Mapping[str, Any]) -> str:
    if not spread.get("n"):
        return "—"
    return f"{_fmt(spread['p50'])} / {_fmt(spread['p90'])} / {_fmt(spread['max'])}"


def render(report: Mapping[str, Any], *, chunks: bool = False) -> str:
    lines = [
        f"{report['samples']} samples ({report['sampled_from']} → {report['sampled_to']}), "
        f"{report['ledger_lines']} ledger lines in {report['intervals']} chunks"
    ]
    if report["unsampled_intervals"]:
        lines.append(f"  {report['unsampled_intervals']} chunks had no sample in their window")
    if report["overlapping_intervals"]:
        lines.append(
            f"  {report['overlapping_intervals']} chunks overlapped another, "
            "so their peaks are shared"
        )
    per_interval = report["per_interval"]
    if per_interval:
        table = [("group", "rss", "pss", "swap")]
        table += [
            (
                name,
                _fmt_spread(spreads["peak_rss_mib"]),
                _fmt_spread(spreads["peak_pss_mib"]),
                _fmt_spread(spreads["max_swap_mib"]),
            )
            for name, spreads in per_interval.items()
        ]
        widths = [max(len(row[column]) for row in table) for column in range(3)]
        lines.append("  peak per chunk (MiB: p50 / p90 / max)")
        lines.extend(
            "    "
            + "  ".join(cell.ljust(width) for cell, width in zip(row[:3], widths, strict=True))
            + "  "
            + row[3]
            for row in table
        )
    lines.append(
        f"  host: lowest MemAvailable {_fmt(report['min_mem_available_mib'])} MiB, "
        f"most swap used {_fmt(report['max_swap_used_mib'])} MiB"
    )
    if chunks:
        lines.append("  chunks (started_at, papers, samples, peak rss per group in MiB)")
        for row in report["chunks"]:
            peaks = ", ".join(
                f"{name} {_fmt(group['peak_rss_mib'])}" for name, group in row["groups"].items()
            )
            lines.append(
                f"    {row['started_at']}  {row['papers']:>3}  {row['samples']:>5}  {peaks or '—'}"
            )
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------


def _named(value: str, kind: str) -> tuple[str, str]:
    name, sep, rest = value.partition("=")
    if not sep or not name or not rest:
        raise argparse.ArgumentTypeError(f"expected NAME={kind}, got {value!r}")
    return name, rest


def _server(value: str) -> tuple[str, int]:
    name, pid = _named(value, "PID")
    if not pid.isdigit():
        raise argparse.ArgumentTypeError(f"expected NAME=PID, got {value!r}")
    return name, int(pid)


def _match(value: str) -> tuple[str, re.Pattern[str]]:
    name, pattern = _named(value, "REGEX")
    try:
        return name, re.compile(pattern)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"bad pattern in {value!r}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mem_sampler.py",
        description="Sample process-tree memory from /proc and join it to a bibr batch ledger.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="write one JSON line of memory per tick")
    sample.add_argument("--pid", type=int, help="the run's root pid (group 'run')")
    sample.add_argument(
        "--server",
        type=_server,
        action="append",
        default=[],
        metavar="NAME=PID",
        help="a model server's root pid; repeatable",
    )
    sample.add_argument(
        "--match",
        type=_match,
        action="append",
        default=[],
        metavar="NAME=REGEX",
        help="processes whose command line matches REGEX, found again every tick; repeatable",
    )
    sample.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="seconds")
    sample.add_argument("--duration", type=float, help="stop after this many seconds")
    sample.add_argument(
        "--keep-going",
        action="store_true",
        help="keep sampling after the --pid process exits",
    )
    sample.add_argument("--no-pss", action="store_true", help="skip PSS (smaps_rollup)")
    sample.add_argument("--out", type=Path, help="append samples here (default: stdout)")
    sample.add_argument("--proc-root", type=Path, default=Path("/proc"), help=argparse.SUPPRESS)

    joined = sub.add_parser("join", help="peaks per ledger chunk and per paper")
    joined.add_argument("samples", type=Path)
    joined.add_argument("ledger", type=Path, help="a bibr batch outcomes.jsonl")
    joined.add_argument("--json", action="store_true", help="print the full join as JSON")
    joined.add_argument("--chunks", action="store_true", help="also list every chunk")
    joined.add_argument(
        "--slack", type=float, default=LEDGER_SLACK_S, help="seconds added to each finished_at"
    )
    return parser


def _cmd_sample(args: argparse.Namespace) -> int:
    proc = ProcFS(args.proc_root)
    if not proc.root.joinpath("meminfo").exists():
        print(f"mem_sampler: {proc.root} is not a Linux /proc", file=sys.stderr)
        return 2
    groups: list[Group] = []
    if args.pid is not None:
        groups.append(Group("run", pids=(args.pid,)))
    groups.extend(Group(name, pids=(pid,)) for name, pid in args.server)
    groups.extend(Group(name, pattern=pattern) for name, pattern in args.match)
    if not groups:
        print("mem_sampler: give --pid, --server or --match", file=sys.stderr)
        return 2
    names = [group.name for group in groups]
    if len(set(names)) != len(names):
        print(f"mem_sampler: group names repeat: {names}", file=sys.stderr)
        return 2
    if args.interval <= 0:
        print("mem_sampler: --interval must be positive", file=sys.stderr)
        return 2

    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    # The sampler's own command line (and that of a shell or `timeout` that
    # started it) holds the --match patterns; never count those processes.
    me = os.getpid()
    exclude = {me} | {
        pid for pid in ancestors(proc.parents(), me) if "mem_sampler" in proc.cmdline(pid)
    }
    watch = (args.pid,) if args.pid is not None and not args.keep_going else ()
    out = args.out.open("a", encoding="utf-8") if args.out else sys.stdout
    try:
        run_sampler(
            proc,
            groups,
            out,
            interval=args.interval,
            duration=args.duration,
            watch=watch,
            pss=not args.no_pss,
            exclude=exclude,
            should_stop=lambda: stopping,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def _cmd_join(args: argparse.Namespace) -> int:
    report = join(load_samples(args.samples), read_jsonl(args.ledger), slack=args.slack)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report, chunks=args.chunks))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sample":
        return _cmd_sample(args)
    return _cmd_join(args)


if __name__ == "__main__":
    raise SystemExit(main())
