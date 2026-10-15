"""scripts/mem_sampler.py: /proc sampling of process trees, and the ledger join."""

from __future__ import annotations

import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.mem_sampler import (
    DEFAULT_PSS_EVERY,
    Group,
    ProcFS,
    build_parser,
    descendants,
    join,
    ledger_intervals,
    load_samples,
    main,
    parse_kib_fields,
    parse_stat,
    render,
    run_sampler,
    sample_once,
)

MIB_KB = 1024


class FakeProc:
    """A /proc tree on disk: stat, status, smaps_rollup, cmdline and meminfo."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "meminfo").write_text(
            "MemTotal:       32000000 kB\n"
            "MemAvailable:    4096000 kB\n"
            "SwapTotal:       8192000 kB\n"
            "SwapFree:        7168000 kB\n"
        )
        (root / "self").mkdir()  # not a pid: skipped

    def add(
        self,
        pid: int,
        ppid: int,
        cmdline: str,
        *,
        rss_mib: int,
        hwm_mib: int | None = None,
        swap_mib: int = 0,
        pss_mib: int | None = None,
        comm: str = "python3",
    ) -> None:
        d = self.root / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1 4194304 0 0\n")
        (d / "cmdline").write_text("\0".join(cmdline.split()) + "\0")
        (d / "status").write_text(
            f"Name:\t{comm}\nPid:\t{pid}\nPPid:\t{ppid}\n"
            f"VmHWM:\t{(hwm_mib or rss_mib) * MIB_KB} kB\n"
            f"VmRSS:\t{rss_mib * MIB_KB} kB\n"
            f"VmSwap:\t{swap_mib * MIB_KB} kB\n"
        )
        if pss_mib is not None:
            (d / "smaps_rollup").write_text(
                "55b3740d8000-7ffe9ac36000 ---p 00000000 00:00 0    [rollup]\n"
                f"Rss:  {rss_mib * MIB_KB} kB\nPss:  {pss_mib * MIB_KB} kB\n"
                f"Pss_Anon:  {pss_mib * MIB_KB} kB\nSwap:  0 kB\n"
            )

    def kernel_thread(self, pid: int, ppid: int = 2) -> None:
        d = self.root / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} (kworker/0:1) I {ppid} 0 0 0 -1\n")
        (d / "cmdline").write_text("")
        (d / "status").write_text(f"Name:\tkworker/0:1\nPid:\t{pid}\nPPid:\t{ppid}\n")

    def remove(self, pid: int) -> None:
        for child in (self.root / str(pid)).iterdir():
            child.unlink()
        (self.root / str(pid)).rmdir()


@pytest.fixture
def proc(tmp_path) -> FakeProc:
    fake = FakeProc(tmp_path / "proc")
    fake.add(1, 0, "/sbin/init", rss_mib=10, pss_mib=5, comm="systemd")
    fake.add(50, 1, "/bin/bash full-run.sh", rss_mib=4, pss_mib=2, comm="bash")
    # The run: bibr batch, a worker it spawned, and that worker's helper.
    fake.add(
        100,
        50,
        "/venv/bin/python /venv/bin/bibr batch corpus/",
        rss_mib=2000,
        hwm_mib=2600,
        swap_mib=100,
        pss_mib=1900,
    )
    fake.add(
        101,
        100,
        "/venv/bin/python -c from multiprocessing.spawn import spawn_main",
        rss_mib=300,
        hwm_mib=320,
        pss_mib=250,
    )
    fake.add(102, 101, "/usr/bin/pdftoppm -r 150", rss_mib=50, pss_mib=40, comm="pdftoppm")
    # A model server with an engine child, and an unrelated process.
    fake.add(
        200,
        1,
        "/venv/bin/python -m vllm.entrypoints.openai.api_server --port 8800",
        rss_mib=1500,
        pss_mib=1400,
    )
    fake.add(201, 200, "VLLM::EngineCore", rss_mib=3000, pss_mib=2900)
    fake.add(300, 1, "/usr/bin/firefox", rss_mib=900, pss_mib=800, comm="firefox")
    fake.kernel_thread(400)
    return fake


def _wall() -> float:
    return datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC).timestamp()


def test_parse_stat_handles_spaces_and_parentheses_in_the_command_name():
    assert parse_stat("123 (Web Content (x)) S 45 123 123 0 -1 0") == (123, "Web Content (x)", 45)


def test_parse_kib_fields_reads_kib_lines_as_bytes_and_skips_the_rest():
    text = "Name:\tpython3\nVmRSS:\t   2048 kB\nThreads:\t12\nPss:  10 kB\n"

    assert parse_kib_fields(text) == {"VmRSS": 2048 * 1024, "Pss": 10 * 1024}


def test_descendants_walks_the_whole_tree_and_ignores_gone_roots():
    parents = {1: 0, 100: 1, 101: 100, 102: 101, 200: 1}

    assert descendants(parents, [100, 999]) == {100, 101, 102}


def test_a_pid_group_sums_its_tree(proc):
    sample = sample_once(ProcFS(proc.root), [Group("run", pids=(100,))], wall=_wall)

    assert sample["groups"]["run"] == {
        "pids": 3,
        "rss_mib": 2350.0,
        "pss_mib": 2190.0,
        "swap_mib": 100.0,
        "hwm_mib": 2600.0,
    }
    assert sample["mem_available_mib"] == 4000.0
    assert sample["swap_used_mib"] == 1000.0
    assert sample["time"] == "2026-10-14T12:00:00.000+00:00"


def test_server_groups_include_their_children(proc):
    sample = sample_once(
        ProcFS(proc.root), [Group("run", pids=(100,)), Group("llm", pids=(200,))], wall=_wall
    )

    assert sample["groups"]["llm"]["pids"] == 2
    assert sample["groups"]["llm"]["rss_mib"] == 4500.0


def test_a_match_group_takes_the_topmost_matching_process(proc):
    group = Group("run", pattern=re.compile(r"bibr batch|multiprocessing"))

    sample = sample_once(ProcFS(proc.root), [group], wall=_wall)

    # 101 matches too, but it is inside 100's tree: counted once.
    assert sample["groups"]["run"]["pids"] == 3
    assert sample["groups"]["run"]["rss_mib"] == 2350.0


def test_the_sampler_never_counts_itself(proc):
    proc.add(
        103,
        100,
        "python3 scripts/mem_sampler.py sample --match run=bibr batch",
        rss_mib=20,
        pss_mib=15,
    )
    group = Group("run", pattern=re.compile("bibr batch"))

    sample = sample_once(ProcFS(proc.root), [group], exclude={103}, wall=_wall)

    assert sample["groups"]["run"]["rss_mib"] == 2350.0


def test_pss_is_none_when_any_process_hides_its_smaps(proc):
    (proc.root / "102" / "smaps_rollup").unlink()

    row = sample_once(ProcFS(proc.root), [Group("run", pids=(100,))], wall=_wall)["groups"]["run"]

    assert row["pss_mib"] is None
    assert row["rss_mib"] == 2350.0


def test_no_pss_skips_smaps(proc):
    row = sample_once(ProcFS(proc.root), [Group("run", pids=(100,))], pss=False, wall=_wall)

    assert row["groups"]["run"]["pss_mib"] is None
    assert row["pss_read"] is False


class _CountingProcFS(ProcFS):
    """Counts smaps_rollup reads: each one walks a target's page tables under its mmap lock."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.smaps_reads = 0

    def _read(self, *parts: str | int) -> str | None:
        if parts and parts[-1] == "smaps_rollup":
            self.smaps_reads += 1
        return super()._read(*parts)


def _run_ticks(proc_fs: ProcFS, ticks: int, **kwargs) -> list[dict]:
    clock = _Clock()
    out = io.StringIO()
    run_sampler(
        proc_fs,
        [Group("run", pids=(100,))],
        out,
        interval=0.5,
        duration=0.5 * (ticks - 1),
        clock=clock,
        sleep=clock.sleep,
        wall=_wall,
        **kwargs,
    )
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(lines) == ticks
    return lines


def test_pss_is_read_on_every_nth_tick_only(proc):
    proc_fs = _CountingProcFS(proc.root)

    lines = _run_ticks(proc_fs, 7, pss_every=3)

    assert [line["pss_read"] for line in lines] == [True, False, False, True, False, False, True]
    assert [line["groups"]["run"]["pss_mib"] for line in lines] == [
        2190.0,
        None,
        None,
        2190.0,
        None,
        None,
        2190.0,
    ]
    # RSS is read on every tick all the same.
    assert {line["groups"]["run"]["rss_mib"] for line in lines} == {2350.0}
    assert proc_fs.smaps_reads == 3 * 3  # three ticks, three processes


def test_the_default_reads_pss_far_less_often_than_rss(proc):
    proc_fs = _CountingProcFS(proc.root)

    lines = _run_ticks(proc_fs, 2 * DEFAULT_PSS_EVERY)

    assert DEFAULT_PSS_EVERY > 1
    assert sum(line["pss_read"] for line in lines) == 2
    assert proc_fs.smaps_reads == 2 * 3


def test_pss_every_zero_never_reads_smaps(proc):
    proc_fs = _CountingProcFS(proc.root)

    lines = _run_ticks(proc_fs, 4, pss_every=0)

    assert not any(line["pss_read"] for line in lines)
    assert proc_fs.smaps_reads == 0


def test_a_group_with_no_process_has_no_reading(proc):
    row = sample_once(ProcFS(proc.root), [Group("gone", pids=(999,))], wall=_wall)["groups"]

    assert row["gone"] == {
        "pids": 0,
        "rss_mib": None,
        "pss_mib": None,
        "swap_mib": None,
        "hwm_mib": None,
    }


def test_a_process_that_exits_mid_scan_is_skipped(proc):
    (proc.root / "102" / "status").unlink()

    row = sample_once(ProcFS(proc.root), [Group("run", pids=(100,))], wall=_wall)["groups"]["run"]

    assert row["pids"] == 2


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 3))
        self.now += seconds


def test_run_sampler_ticks_on_schedule_and_stops_after_its_duration(proc):
    clock = _Clock()
    out = io.StringIO()

    written = run_sampler(
        ProcFS(proc.root),
        [Group("run", pids=(100,))],
        out,
        interval=0.5,
        duration=2.0,
        clock=clock,
        sleep=clock.sleep,
        wall=_wall,
    )

    assert written == 5
    assert clock.slept == [0.5, 0.5, 0.5, 0.5]
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [line["groups"]["run"]["pids"] for line in lines] == [3] * 5


def test_run_sampler_stops_once_the_watched_process_exits(proc):
    clock = _Clock()
    out = io.StringIO()

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if len(clock.slept) == 2:
            for pid in (102, 101, 100):
                proc.remove(pid)

    written = run_sampler(
        ProcFS(proc.root),
        [Group("run", pids=(100,))],
        out,
        watch=(100,),
        clock=clock,
        sleep=sleep,
        wall=_wall,
    )

    assert written == 3
    last = json.loads(out.getvalue().splitlines()[-1])
    assert last["groups"]["run"]["pids"] == 0


def test_run_sampler_does_not_burst_after_a_slow_tick(proc):
    clock = _Clock()
    costs = iter([2.0])  # the first sample takes four intervals

    def slow_wall() -> float:
        clock.now += next(costs, 0.0)
        return _wall()

    written = run_sampler(
        ProcFS(proc.root),
        [Group("run", pids=(100,))],
        io.StringIO(),
        interval=0.5,
        duration=3.0,
        clock=clock,
        sleep=clock.sleep,
        wall=slow_wall,
    )

    # Ticks at 0 (slow), 2.0, 2.5 and 3.0: no catch-up burst for the missed ones.
    assert written == 4
    assert clock.slept == [0.5, 0.5]


def test_run_sampler_honours_a_stop_request(proc):
    calls = iter([False, True])

    written = run_sampler(
        ProcFS(proc.root),
        [Group("run", pids=(100,))],
        io.StringIO(),
        should_stop=lambda: next(calls),
        sleep=lambda _s: None,
        wall=_wall,
    )

    assert written == 1


# --- join -------------------------------------------------------------------

T0 = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC).timestamp()


def _iso(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, tz=UTC).isoformat(timespec="seconds")


def _sample(offset: float, run_rss: float, llm_rss: float = 4000.0, avail: float = 3000.0):
    return {
        "t": T0 + offset,
        "time": _iso(offset),
        "groups": {
            "run": {"pids": 3, "rss_mib": run_rss, "pss_mib": run_rss - 100, "swap_mib": 0.0},
            "llm": {"pids": 2, "rss_mib": llm_rss, "pss_mib": None, "swap_mib": 10.0},
        },
        "mem_available_mib": avail,
        "swap_used_mib": 500.0,
    }


def _row(paper: str, start: float, end: float, status: str = "ok") -> dict:
    return {
        "paper_id": paper,
        "stem": paper,
        "status": status,
        "attempt": 1,
        "started_at": _iso(start),
        "finished_at": _iso(end),
    }


@pytest.fixture
def ledger_rows():
    return [
        _row("a", 0, 10, status="started"),  # a start line: no interval
        _row("a", 0, 10),
        _row("b", 0, 10),
        _row("c", 0, 10, status="failed"),
        _row("d", 10, 20),
        _row("e", 20, 30),  # nothing sampled here
        {"paper_id": "f", "status": "ok", "started_at": None, "finished_at": None},
    ]


@pytest.fixture
def samples():
    return [
        _sample(1, 1500),
        _sample(5, 2600, avail=900),
        # 10.4 s: after chunk 1's printed finish, inside the 1 s slack.
        _sample(10.4, 2700),
        _sample(12, 1800, llm_rss=4200),
        _sample(19, 1900),
    ]


def test_ledger_lines_sharing_times_form_one_chunk(ledger_rows):
    intervals = ledger_intervals(ledger_rows)

    assert [len(interval.rows) for interval in intervals] == [3, 1, 1]
    assert [row["paper_id"] for row in intervals[0].rows] == ["a", "b", "c"]


def test_join_gives_each_chunk_and_paper_its_peaks(samples, ledger_rows):
    report = join(samples, ledger_rows)

    first, second, third = report["chunks"]
    assert first["papers"] == 3
    assert first["samples"] == 3  # 1, 5 and the slack sample at 10.4
    assert first["groups"]["run"]["peak_rss_mib"] == 2700
    assert first["groups"]["run"]["peak_pss_mib"] == 2600
    assert first["groups"]["llm"]["peak_pss_mib"] is None
    assert first["min_mem_available_mib"] == 900
    # The slack sample belongs to the next chunk too: ledger times are whole seconds.
    assert second["samples"] == 3
    assert second["groups"]["llm"]["peak_rss_mib"] == 4200
    assert third["samples"] == 0
    papers = {row["paper_id"]: row for row in report["papers"]}
    assert set(papers) == {"a", "b", "c", "d", "e"}
    assert papers["c"]["status"] == "failed"
    assert papers["b"]["chunk_papers"] == 3
    assert papers["d"]["groups"]["run"]["peak_rss_mib"] == 2700


def test_join_summarises_the_spread_over_sampled_chunks(samples, ledger_rows):
    report = join(samples, ledger_rows)

    assert report["intervals"] == 3
    assert report["unsampled_intervals"] == 1
    assert report["ledger_lines"] == 5
    assert report["per_interval"]["run"]["peak_rss_mib"] == {
        "n": 2,
        "p50": 2700.0,
        "p90": 2700.0,
        "max": 2700.0,
    }
    assert report["per_interval"]["llm"]["peak_pss_mib"]["n"] == 0
    assert report["min_mem_available_mib"] == 900


def test_join_counts_overlapping_chunks(samples):
    rows = [_row("a", 0, 10), _row("b", 5, 15), _row("c", 15, 20)]

    assert join(samples, rows)["overlapping_intervals"] == 1


def test_render_prints_the_spread_and_optional_chunks(samples, ledger_rows):
    text = render(join(samples, ledger_rows), chunks=True)

    assert "5 samples" in text
    assert "5 ledger lines in 3 chunks" in text
    assert "1 chunks had no sample in their window" in text
    assert re.search(r"run\s+2,700 / 2,700 / 2,700", text)
    assert "lowest MemAvailable 900 MiB" in text
    assert f"{_iso(20)}    1      0  —" in text


def test_load_samples_sorts_and_skips_headers_and_torn_lines(tmp_path, samples):
    path = tmp_path / "mem.jsonl"
    lines = [json.dumps(samples[2]), json.dumps({"header": {"interval": 0.5}}), "[1, 2]"]
    lines += [json.dumps(samples[0]), '{"t": 1791']
    path.write_text("\n".join(lines) + "\n")

    assert [row["t"] for row in load_samples(path)] == [samples[0]["t"], samples[2]["t"]]


# --- CLI --------------------------------------------------------------------


def test_cli_join_prints_text_and_json(tmp_path, samples, ledger_rows, capsys):
    mem = tmp_path / "mem.jsonl"
    mem.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    ledger = tmp_path / "outcomes.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in ledger_rows))

    assert main(["join", str(mem), str(ledger)]) == 0
    assert "peak per chunk" in capsys.readouterr().out
    assert main(["join", str(mem), str(ledger), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["intervals"] == 3


def test_cli_sample_writes_jsonl_until_the_run_exits(tmp_path, proc, monkeypatch):
    monkeypatch.setattr("scripts.mem_sampler.signal.signal", lambda *_a: None)
    out = tmp_path / "mem.jsonl"
    proc.remove(100)
    proc.remove(101)
    proc.remove(102)

    code = main(
        [
            "sample",
            "--pid",
            "100",
            "--server",
            "llm=200",
            "--match",
            "browser=firefox",
            "--out",
            str(out),
            "--proc-root",
            str(proc.root),
        ]
    )

    assert code == 0
    (line,) = [json.loads(text) for text in out.read_text().splitlines()]
    assert set(line["groups"]) == {"run", "llm", "browser"}
    assert line["groups"]["browser"]["rss_mib"] == 900.0
    # The first tick reads PSS; later ones only every --pss-every-th.
    assert line["pss_read"] is True
    assert line["groups"]["browser"]["pss_mib"] == 800.0


@pytest.mark.parametrize(
    ("flags", "pss_every"),
    [([], DEFAULT_PSS_EVERY), (["--pss-every", "1"], 1), (["--no-pss"], 0)],
)
def test_cli_sample_passes_the_pss_cadence(tmp_path, proc, monkeypatch, flags, pss_every):
    monkeypatch.setattr("scripts.mem_sampler.signal.signal", lambda *_a: None)
    seen = {}
    monkeypatch.setattr(
        "scripts.mem_sampler.run_sampler", lambda *_args, **kwargs: seen.update(kwargs) or 0
    )

    code = main(["sample", "--pid", "100", "--proc-root", str(proc.root), *flags])

    assert code == 0
    assert seen["pss_every"] == pss_every


def test_cli_sample_rejects_a_pss_cadence_below_one(proc, capsys):
    assert main(["sample", "--pid", "100", "--pss-every", "0", "--proc-root", str(proc.root)]) == 2
    assert "--no-pss" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["sample", "--server", "llm"],
        ["sample", "--server", "llm=abc"],
        ["sample", "--match", "=x"],
        ["sample", "--match", "run=("],
    ],
)
def test_cli_rejects_malformed_groups(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_cli_sample_needs_a_group_and_distinct_names(tmp_path, proc, capsys):
    root = ["--proc-root", str(proc.root)]

    assert main(["sample", *root]) == 2
    assert main(["sample", "--server", "a=1", "--server", "a=200", *root]) == 2
    assert main(["sample", "--pid", "1", "--proc-root", str(tmp_path / "nowhere")]) == 2
    assert "not a Linux /proc" in capsys.readouterr().err
