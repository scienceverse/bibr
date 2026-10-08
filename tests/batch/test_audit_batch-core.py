"""Audit batch-core: paper ids across runs, cwds and filesystems; an export
that cannot be written; the out-dir lock; ``started`` lines for runs killed
outright; and table rebuilds only when an export changed."""

from __future__ import annotations

import errno
import io
import json
import unicodedata
from pathlib import Path

import pytest
from rich.console import Console

from bibr.batch.ledger import (
    LEDGER_FILENAME,
    MAX_UNSETTLED_FAILURES,
    OUTPUT_WRITE_FAILED,
    STARTED,
    Ledger,
    LedgerContext,
    Outcome,
)
from bibr.batch.manifest import (
    MAX_ID_BYTES,
    NAME_MAX,
    BatchItem,
    assign_paper_ids,
    sha256_file,
)
from bibr.batch.report import compute_report
from bibr.batch.runner import (
    LOCK_FILENAME,
    RUN_INFO_FILENAME,
    BatchOptions,
    LocalOptions,
    out_dir_lock,
    run_batch,
)
from tests.batch.test_local import _FakeChewMany, _RealExports

CTX = LedgerContext(run_id="r1", executor="local", bibr_version="9.9.9", build_sha=None)


def _pdf(path: Path, body: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n" + body)
    return path


def _sha8(path: Path) -> str:
    return sha256_file(path)[:8]


def _ids(files: list[Path], recorded: list[dict] | None = None) -> list[str]:
    return [item.paper_id for item in assign_paper_ids(files, recorded=recorded or [])]


@pytest.fixture(autouse=True)
def _offline_run(monkeypatch):
    monkeypatch.setattr("bibr.batch.runner.local_build_sha", lambda: "deadbeef")
    monkeypatch.setattr("bibr.batch.runner.redacted_settings_snapshot", dict)


def _run(monkeypatch, inputs: list[Path], out: Path, fake=None, **overrides) -> tuple[int, str]:
    fake = _FakeChewMany() if fake is None else fake
    monkeypatch.setattr("bibr.batch.runner.open_chew_many", lambda local: fake)
    options = {
        "inputs": [str(path) for path in inputs],
        "out": out,
        "local": LocalOptions(batch_size=2),
        "tables": False,
        **overrides,
    }
    console = Console(file=io.StringIO(), width=400)
    code = run_batch(BatchOptions(**options), console=console)
    return code, " ".join(console.file.getvalue().split())


# --- 1. long names, and an export that cannot be written ------------------------


def test_atomic_write_json_takes_any_destination_name_up_to_name_max(tmp_path):
    from bibr.local.artifacts import atomic_write_json

    destination = tmp_path / ("x" * (NAME_MAX - len(".json")) + ".json")
    atomic_write_json(destination, {"ok": True})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"ok": True}
    assert [path.name for path in tmp_path.iterdir()] == [destination.name]


def test_an_over_long_stem_gets_a_clipped_id_that_fits_with_its_sidecars(tmp_path, monkeypatch):
    stem = "論文" * 40 + "x" * 10  # 250 bytes: a legal file name, too long for an id
    long_pdf = _pdf(tmp_path / "papers" / f"{stem}.pdf", b"long")
    _pdf(tmp_path / "papers" / "after.pdf", b"after")
    out = tmp_path / "out"

    assert _run(monkeypatch, [tmp_path / "papers"], out)[0] == 0

    latest = Ledger(out / LEDGER_FILENAME).latest()
    (long_id,) = (pid for pid in latest if pid != "after")
    assert long_id.endswith(f"-{_sha8(long_pdf)}")
    assert stem.startswith(long_id[: -len("-12345678")])
    assert len(long_id.encode()) <= MAX_ID_BYTES
    assert len(f"{long_id}.json.enrichment.json".encode()) <= NAME_MAX
    assert {row["status"] for row in latest.values()} == {"ok"}
    assert (out / f"{long_id}.json").is_file()
    assert json.loads((out / RUN_INFO_FILENAME).read_text(encoding="utf-8"))["finished_at"]


def test_an_export_that_cannot_be_written_fails_only_its_paper(tmp_path, monkeypatch):
    import bibr.local.artifacts as artifacts

    papers = tmp_path / "papers"
    for name in ("a_full", "b_ok", "c_ok"):
        _pdf(papers / f"{name}.pdf", name.encode())
    real_write = artifacts.atomic_write_json

    def disk_full_for_a(path, payload, **kwargs):
        if Path(path).name == "a_full.json":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(path, payload, **kwargs)

    monkeypatch.setattr(artifacts, "atomic_write_json", disk_full_for_a)
    out = tmp_path / "out"

    code, printed = _run(monkeypatch, [papers], out)

    assert code == 1
    latest = Ledger(out / LEDGER_FILENAME).latest()
    assert latest["a_full"]["status"] == "failed"
    assert latest["a_full"]["error_code"] == OUTPUT_WRITE_FAILED
    assert "No space left on device" in latest["a_full"]["error"]
    # Its chunk neighbour and the next chunk are recorded, and the run finished.
    assert latest["b_ok"]["status"] == latest["c_ok"]["status"] == "ok"
    info = json.loads((out / RUN_INFO_FILENAME).read_text(encoding="utf-8"))
    assert info["finished_at"] and info["n_ok"] == 2 and info["n_failed"] == 1
    assert "a_full: output_write_failed" in printed

    # A full disk is the machine's: the next run writes it, no --retry-failed.
    monkeypatch.setattr(artifacts, "atomic_write_json", real_write)
    again = _FakeChewMany()
    assert _run(monkeypatch, [papers], out, again)[0] == 0
    assert [[path.stem for path in paths] for paths, _ in again.calls] == [["a_full"]]
    assert (out / "a_full.json").is_file()


def test_an_export_json_rejects_is_a_failed_paper_not_a_crashed_run(tmp_path, monkeypatch):
    class NaNExports(_FakeChewMany):
        def __call__(self, paths, batch_size):
            results = super().__call__(paths, batch_size)
            for path, result in zip(paths, results, strict=True):
                if path.stem == "a_nan":
                    result.data["score"] = float("nan")
            return results

    papers = tmp_path / "papers"
    for name in ("a_nan", "b_ok"):
        _pdf(papers / f"{name}.pdf", name.encode())
    out = tmp_path / "out"

    assert _run(monkeypatch, [papers], out, NaNExports())[0] == 1

    latest = Ledger(out / LEDGER_FILENAME).latest()
    assert latest["a_nan"]["error_code"] == OUTPUT_WRITE_FAILED
    assert latest["b_ok"]["status"] == "ok"
    assert not list(out.glob(".*.tmp"))


# --- 2./3. recorded ids: other files, other cwds, other spellings -------------------


def test_a_same_named_file_of_another_run_is_a_new_paper(tmp_path, monkeypatch):
    """journalA/ then journalB/ into one --out: B's paper.pdf is not A's."""
    for journal, body in (("journalA", b"A"), ("journalB", b"B, longer")):
        _pdf(tmp_path / journal / "paper.pdf", body)
        _pdf(tmp_path / journal / "1.pdf", body + b"1")
    out = tmp_path / "out"
    assert _run(monkeypatch, [tmp_path / "journalA"], out)[0] == 0

    second = _FakeChewMany()
    assert _run(monkeypatch, [tmp_path / "journalB"], out, second)[0] == 0

    assert [[p.parent.name for p in paths] for paths, _ in second.calls] == [["journalB"] * 2]
    latest = Ledger(out / LEDGER_FILENAME).latest()
    by_path = {
        Path(row["path"]).relative_to(tmp_path).as_posix(): pid for pid, row in latest.items()
    }
    b_paper = tmp_path / "journalB" / "paper.pdf"
    assert by_path == {
        "journalA/1.pdf": "1",
        "journalA/paper.pdf": "paper",
        "journalB/1.pdf": f"1-{_sha8(tmp_path / 'journalB' / '1.pdf')}",
        "journalB/paper.pdf": f"paper-{_sha8(b_paper)}",
    }
    a_export = json.loads((out / "paper.json").read_text(encoding="utf-8"))
    assert a_export["paper_id"] == "paper"


def test_a_recorded_id_is_reserved_case_insensitively_across_runs(tmp_path):
    first = _pdf(tmp_path / "a" / "Paper.pdf", b"A")
    newcomer = _pdf(tmp_path / "b" / "paper.pdf", b"B")
    recorded = [
        {"paper_id": "Paper", "path": str(first), "sha256": sha256_file(first), "status": "ok"}
    ]
    assert _ids([newcomer], recorded) == [f"paper-{_sha8(newcomer)}"]
    # The same bytes, though, are the recorded paper wherever they now are.
    moved = _pdf(tmp_path / "moved" / "paper.pdf", b"A")
    assert _ids([moved], recorded) == ["Paper"]


def test_an_ok_line_counts_only_for_the_bytes_it_recorded(tmp_path):
    ledger = Ledger(tmp_path / LEDGER_FILENAME)
    a = _pdf(tmp_path / "a" / "paper.pdf", b"A")
    b = _pdf(tmp_path / "b" / "paper.pdf", b"B")
    ledger.record(BatchItem(a, "paper", "paper"), Outcome("ok", export={}), context=CTX)

    assert ledger.plan([BatchItem(a, "paper", "paper")]).skipped_ok
    plan = ledger.plan([BatchItem(b, "paper", "paper")])
    assert [item.path for item in plan.to_run] == [b]
    assert not plan.skipped_ok


def test_a_recorded_paper_keeps_its_id_however_its_path_is_spelled(tmp_path, monkeypatch):
    a = _pdf(tmp_path / "corpus" / "a" / "paper.pdf", b"A")
    b = _pdf(tmp_path / "corpus" / "b" / "paper.pdf", b"B")
    monkeypatch.chdir(tmp_path)
    recorded = [
        {
            "paper_id": "paper",
            "path": "corpus/a/paper.pdf",  # relative, as the first run was given it
            "sha256": sha256_file(a),
            "bytes": a.stat().st_size,
            "status": "ok",
        }
    ]
    via_dotdot = tmp_path / "lists" / ".." / "corpus" / "a" / "paper.pdf"
    expected = ["paper", f"paper-{_sha8(b)}"]
    assert _ids([a, b], recorded) == expected  # absolute now
    assert _ids([via_dotdot, b], recorded) == expected  # a manifest's ../
    monkeypatch.chdir(tmp_path / "corpus")  # the relative path names nothing here
    assert _ids([a, b], recorded) == expected


def test_a_moved_file_keeps_its_suffixed_id_by_content(tmp_path):
    moved = _pdf(tmp_path / "new" / "paper.pdf", b"A")
    recorded = [
        {
            "paper_id": "paper-0badc0de",
            "path": "/old/place/paper.pdf",
            "sha256": sha256_file(moved),
            "bytes": moved.stat().st_size,
            "status": "ok",
        }
    ]
    items = assign_paper_ids([moved], recorded=recorded)
    assert [item.paper_id for item in items] == ["paper-0badc0de"]
    assert Ledger(tmp_path / "none.jsonl").plan(items, entries=recorded).skipped_ok == items


def test_a_file_replaced_at_its_recorded_path_is_a_new_paper(tmp_path):
    path = _pdf(tmp_path / "paper.pdf", b"first version")
    recorded = [
        {
            "paper_id": "paper",
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "status": "ok",
        }
    ]
    _pdf(path, b"the corrected, longer version")
    assert _ids([path], recorded) == [f"paper-{_sha8(path)}"]


# --- 7. names Windows or APFS cannot hold apart -----------------------------------


def test_nfc_and_nfd_spellings_of_a_stem_collide(tmp_path):
    nfc, nfd = (unicodedata.normalize(form, "café") for form in ("NFC", "NFD"))
    a = _pdf(tmp_path / "x" / f"{nfc}.pdf", b"one")
    b = _pdf(tmp_path / "y" / f"{nfd}.pdf", b"two")
    assert _ids([a, b]) == [f"{nfc}-{_sha8(a)}", f"{nfd}-{_sha8(b)}"]


@pytest.mark.parametrize(
    ("stem", "template"),
    [
        ("CON", "CON-{}"),
        ("aux", "aux-{}"),
        ("Com1", "Com1-{}"),
        ("lpt9", "lpt9-{}"),
        ("nul.tar", "nul-{}.tar"),  # NUL.tar.json still opens the device
        ("console", "console"),
        ("con-1", "con-1"),
    ],
)
def test_windows_device_names_are_reserved(tmp_path, stem, template):
    path = _pdf(tmp_path / f"{stem}.pdf", stem.encode())
    assert _ids([path]) == [template.format(_sha8(path))]


def test_an_unreadable_device_name_still_gets_a_usable_id(tmp_path):
    assert _ids([tmp_path / "CON.pdf"]) == ["CON-2"]


# --- 4. one run per out dir --------------------------------------------------------


def test_a_second_run_on_a_busy_out_dir_fails_fast(tmp_path, monkeypatch):
    from bibr.local.cli.tables import export_files

    _pdf(tmp_path / "papers" / "a.pdf", b"a")
    out = tmp_path / "out"
    fake = _FakeChewMany()

    with out_dir_lock(out):
        code, printed = _run(monkeypatch, [tmp_path / "papers"], out, fake)

    assert code == 2
    assert fake.calls == []
    assert f"another bibr batch run is using {out}" in printed
    assert not (out / LEDGER_FILENAME).exists()
    assert not (out / RUN_INFO_FILENAME).exists()

    assert _run(monkeypatch, [tmp_path / "papers"], out)[0] == 0  # released
    assert (out / LOCK_FILENAME).is_file()
    assert LOCK_FILENAME not in {path.name for path in export_files([str(out)])[0]}


# --- 5. a run killed outright ----------------------------------------------------


def test_started_lines_are_on_disk_but_not_in_verdict_readers(tmp_path, monkeypatch):
    papers = tmp_path / "papers"
    for name in ("good1", "bad1"):
        _pdf(papers / f"{name}.pdf", name.encode())
    out = tmp_path / "out"
    _run(monkeypatch, [papers], out)
    _run(monkeypatch, [papers], out, force=True)

    raw = [json.loads(line) for line in (out / LEDGER_FILENAME).read_text().splitlines()]
    assert [
        (r["paper_id"], r["status"], r["attempt"]) for r in raw if r["paper_id"] == "good1"
    ] == [
        ("good1", STARTED, 1),
        ("good1", "ok", 1),
        ("good1", STARTED, 2),
        ("good1", "ok", 2),
    ]
    started = next(r for r in raw if r["status"] == STARTED)
    assert started["sha256"] == sha256_file(papers / f"{started['paper_id']}.pdf")
    verdicts = Ledger(out / LEDGER_FILENAME).read()
    assert STARTED not in {row["status"] for row in verdicts}
    assert compute_report(verdicts)["papers"] == {"total": 2, "ok": 1, "failed": 1, "attempts": 4}


def test_papers_of_a_killed_run_run_again_alone_and_last(tmp_path, monkeypatch):
    papers = tmp_path / "papers"
    files = [_pdf(papers / f"{name}.pdf", name.encode()) for name in ("k1", "k2", "n1", "n2", "n3")]
    out = tmp_path / "out"
    killed = Ledger(out / LEDGER_FILENAME)
    for item in assign_paper_ids(files[:2]):  # the chunk the OOM killer ended
        killed.start(item, context=CTX)

    fake = _FakeChewMany()
    assert _run(monkeypatch, [papers], out, fake)[0] == 0

    assert [[p.stem for p in paths] for paths, _ in fake.calls] == [
        ["n1", "n2"],
        ["n3"],
        ["k1"],
        ["k2"],
    ]
    latest = Ledger(out / LEDGER_FILENAME).latest()
    assert latest["k1"]["status"] == "ok" and latest["k1"]["attempt"] == 2


def test_a_paper_that_keeps_killing_the_run_waits_for_retry_failed(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    item = BatchItem(_pdf(tmp_path / "p.pdf", b"poison"), "p", "p")
    for _ in range(MAX_UNSETTLED_FAILURES - 1):
        Ledger(path).start(item, context=CTX)  # a run that never came back
        plan = Ledger(path).plan([item])
        assert plan.to_run == [item] and plan.unfinished == {"p"}
    Ledger(path).start(item, context=CTX)

    plan = Ledger(path).plan([item])
    assert plan.to_run == [] and plan.skipped_failed == [item]
    assert Ledger(path).plan([item], retry_failed=True).to_run == [item]
    assert Ledger(path).attempts() == {"p": MAX_UNSETTLED_FAILURES}
    assert Ledger(path).read() == []  # no verdict yet


def test_a_killed_force_rerun_keeps_the_ok_verdict(tmp_path):
    ledger = Ledger(tmp_path / LEDGER_FILENAME)
    item = BatchItem(_pdf(tmp_path / "p.pdf", b"x"), "p", "p")
    ledger.record(item, Outcome("ok", export={}), context=CTX)
    ledger.start(item, context=CTX)

    plan = Ledger(ledger.path).plan([item])
    assert plan.skipped_ok == [item]
    assert Ledger(ledger.path).latest()["p"]["status"] == "ok"


# --- 6. tables rebuilt only when an export changed --------------------------------


def test_tables_are_rebuilt_only_when_the_ok_exports_changed(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    import bibr.export.tables as tables_module

    builds: list[int] = []
    real_write_tables = tables_module.write_tables

    def counting(sources, out_dir):
        builds.append(1)
        return real_write_tables(sources, out_dir)

    monkeypatch.setattr(tables_module, "write_tables", counting)
    papers = tmp_path / "papers"
    for name in ("good1", "good2"):
        _pdf(papers / f"{name}.pdf", name.encode())
    out = tmp_path / "out"

    def run() -> str:
        return _run(monkeypatch, [papers], out, _RealExports(), tables=True)[1]

    run()
    assert len(builds) == 1
    assert f"{out / 'tables'}/ up to date" in run()  # nothing new
    assert len(builds) == 1

    _pdf(papers / "good3.pdf", b"good3")
    run()
    assert len(builds) == 2
    paper_ids = pq.read_table(out / "tables" / "paper.parquet").column("paper_id").to_pylist()
    assert paper_ids == ["good1", "good2", "good3"]

    (out / "tables" / "bib.parquet").unlink()
    run()
    assert len(builds) == 3 and (out / "tables" / "bib.parquet").is_file()
    # `bibr tables <out>` never reads the record of what the tables came from.
    from bibr.local.cli.tables import export_files

    found = {path.relative_to(out).as_posix() for path in export_files([str(out)])[0]}
    assert found == {"good1.json", "good2.json", "good3.json", RUN_INFO_FILENAME}
