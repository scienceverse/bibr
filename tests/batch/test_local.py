"""Runner + local executor with a stubbed warm ``chew_many`` (no models, no network)."""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import pytest

from bibr.api import ChewFailure
from bibr.batch.ledger import INTERRUPTED, LEDGER_FILENAME, Ledger
from bibr.batch.manifest import BatchItem
from bibr.batch.runner import (
    RUN_HISTORY_FILENAME,
    RUN_INFO_FILENAME,
    BatchOptions,
    LocalExecutor,
    LocalOptions,
    parse_deadline,
    run_batch,
)


class _Result:
    ok = True

    def __init__(self, data: dict):
        self.data = data


def _export(stem: str) -> dict:
    return {
        "paper_id": stem,
        "info": {"title": stem.upper()},
        "text": [{"id": 1}],
        "bib": [{"id": "b1"}, {"id": "b2"}],
        "bib_match": [{"bib_id": "b1"}],
        "llm_usage": {"m": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
        "extraction": {"timings": {"ocr": 2.0, "extract": 1.0}, "total_seconds": 3.5},
        "processing_warnings": [],
    }


class _FakeChewMany:
    """Scripted ``chew_many``: fails stems starting with ``bad``; can raise on a call."""

    def __init__(self, *, raise_on_call: int | None = None, exc: BaseException | None = None):
        self.calls: list[tuple[list[Path], int]] = []
        self.closed = False
        self.raise_on_call = raise_on_call
        self.exc = exc

    def __call__(self, paths, batch_size):
        self.calls.append((list(paths), batch_size))
        if self.raise_on_call == len(self.calls) and self.exc is not None:
            raise self.exc
        results = []
        for path in paths:
            if path.stem.startswith("bad"):
                results.append(
                    ChewFailure(
                        path, "OCR failed: no text", error_code="OCR_FAILED", failed_stage="ocr"
                    )
                )
            else:
                results.append(_Result(_export(path.stem)))
        return results

    def close(self):
        self.closed = True


@pytest.fixture
def corpus(tmp_path) -> Path:
    papers = tmp_path / "papers"
    papers.mkdir()
    for name in ("good1", "good2", "good3", "bad1"):
        (papers / f"{name}.pdf").write_bytes(b"%PDF-1.4\n" + name.encode())
    return papers


@pytest.fixture(autouse=True)
def _stable_build_sha(monkeypatch):
    monkeypatch.setattr("bibr.batch.runner.local_build_sha", lambda: "deadbeef")


def _install(monkeypatch, fake: _FakeChewMany) -> None:
    monkeypatch.setattr("bibr.batch.runner.open_chew_many", lambda local: fake)


def _options(corpus: Path, out: Path, **overrides) -> BatchOptions:
    base = {
        "inputs": [str(corpus)],
        "out": out,
        "local": LocalOptions(chew_options={"memory_mode": "balanced"}, batch_size=2),
        "cli_options": {"batch_size": 2, "no_llm": True},
    }
    base.update(overrides)
    return BatchOptions(**base)


def _ledger(out: Path) -> list[dict]:
    return Ledger(out / LEDGER_FILENAME).read()


# --- the run --------------------------------------------------------------------


def test_local_run_writes_exports_ledger_run_info_and_report(corpus, tmp_path, monkeypatch, capsys):
    fake = _FakeChewMany()
    _install(monkeypatch, fake)
    out = tmp_path / "out"

    code = run_batch(_options(corpus, out))

    assert code == 1  # bad1 failed
    assert [len(paths) for paths, _ in fake.calls] == [2, 2]  # stage-major chunks of 2
    assert fake.closed
    assert sorted(p.name for p in out.glob("*.json") if p.name != RUN_INFO_FILENAME) == [
        "good1.json",
        "good2.json",
        "good3.json",
    ]
    assert json.loads((out / "good1.json").read_text(encoding="utf-8"))["info"]["title"] == "GOOD1"

    rows = _ledger(out)
    assert [r["paper_id"] for r in rows] == ["bad1", "good1", "good2", "good3"]
    ok = next(r for r in rows if r["paper_id"] == "good1")
    assert ok["status"] == "ok"
    assert ok["executor"] == "local"
    assert ok["build_sha"] == "deadbeef"
    assert ok["n_refs"] == 2 and ok["n_matched"] == 1
    assert ok["stage_times"] == {"ocr": 2.0, "extract": 1.0}
    assert ok["duration_s"] == 3.5  # the export's own pipeline time, not the chunk's
    assert ok["llm_tokens"] == 15
    assert ok["attempt"] == 1
    bad = next(r for r in rows if r["paper_id"] == "bad1")
    assert bad["status"] == "failed"
    assert bad["error_code"] == "OCR_FAILED"
    assert bad["failed_stage"] == "ocr"
    assert "OCR failed" in bad["error"]
    assert len({r["run_id"] for r in rows}) == 1

    info = json.loads((out / RUN_INFO_FILENAME).read_text(encoding="utf-8"))
    assert info["executor"] == "local"
    assert info["run_id"] == rows[0]["run_id"]
    assert info["n_inputs"] == 4 and info["n_planned"] == 4
    assert info["n_ok"] == 3 and info["n_failed"] == 1
    assert info["reason"] == "completed"
    assert info["options"] == {"batch_size": 2, "no_llm": True}
    assert info["bibr_version"]
    assert isinstance(info["settings"], dict)
    assert len((out / RUN_HISTORY_FILENAME).read_text(encoding="utf-8").splitlines()) == 1

    report = capsys.readouterr().out
    assert "bibr batch" in report
    assert "3 ok · 1 failed · 4 total" in report


def test_run_info_redacts_secrets_in_the_settings_snapshot(corpus, tmp_path, monkeypatch):
    _install(monkeypatch, _FakeChewMany())
    secret = "sk-1234567890abcdef"  # documented placeholder in .gitleaks.toml
    monkeypatch.setenv("LLM_API_KEY", secret)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    out = tmp_path / "out"

    run_batch(_options(corpus, out))

    text = (out / RUN_INFO_FILENAME).read_text(encoding="utf-8")
    assert secret not in text
    info = json.loads(text)
    assert info["settings"]["LLM_API_KEY"] == "sk-1…cdef"
    assert info["settings"]["LLM_PROVIDER"] == "openai"
    assert "test-google-key" not in text  # conftest's GOOGLE_API_KEY is masked too


def test_resume_skips_ok_and_reruns_failed_only_with_retry_failed(corpus, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _install(monkeypatch, _FakeChewMany())
    assert run_batch(_options(corpus, out)) == 1
    assert len(_ledger(out)) == 4

    # Plain re-run: nothing left to do (ok skipped, failed skipped).
    second = _FakeChewMany()
    _install(monkeypatch, second)
    assert run_batch(_options(corpus, out)) == 0
    assert second.calls == []
    assert len(_ledger(out)) == 4

    # --retry-failed: only bad1 runs again, attempt 2.
    third = _FakeChewMany()
    _install(monkeypatch, third)
    assert run_batch(_options(corpus, out, retry_failed=True)) == 1
    assert [[p.stem for p in paths] for paths, _ in third.calls] == [["bad1"]]
    rows = _ledger(out)
    assert len(rows) == 5
    assert rows[-1]["paper_id"] == "bad1" and rows[-1]["attempt"] == 2

    # --force: everything again.
    fourth = _FakeChewMany()
    _install(monkeypatch, fourth)
    run_batch(_options(corpus, out, force=True))
    assert sum(len(paths) for paths, _ in fourth.calls) == 4
    assert len(_ledger(out)) == 9


def test_keyboard_interrupt_records_interrupted_and_resume_reruns_them(
    corpus, tmp_path, monkeypatch
):
    out = tmp_path / "out"
    _install(monkeypatch, _FakeChewMany(raise_on_call=1, exc=KeyboardInterrupt()))

    assert run_batch(_options(corpus, out)) == 130

    rows = _ledger(out)
    assert len(rows) == 2  # only the interrupted chunk
    assert all(r["status"] == "failed" and r["error_code"] == INTERRUPTED for r in rows)
    assert not [p for p in out.glob("*.json") if p.name != RUN_INFO_FILENAME]
    info = json.loads((out / RUN_INFO_FILENAME).read_text(encoding="utf-8"))
    assert info["reason"] == "interrupted"

    resumed = _FakeChewMany()
    _install(monkeypatch, resumed)
    run_batch(_options(corpus, out))
    assert sum(len(paths) for paths, _ in resumed.calls) == 4  # interrupted + never-run
    latest = Ledger(out / LEDGER_FILENAME).latest()
    assert {pid: row["status"] for pid, row in latest.items()} == {
        "bad1": "failed",
        "good1": "ok",
        "good2": "ok",
        "good3": "ok",
    }


def test_chunk_crash_records_chunk_error_and_continues(corpus, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _install(monkeypatch, _FakeChewMany(raise_on_call=1, exc=RuntimeError("OCR engine died")))

    assert run_batch(_options(corpus, out)) == 1

    rows = _ledger(out)
    assert len(rows) == 4
    crashed = [r for r in rows if r["error_code"] == "chunk_error"]
    assert len(crashed) == 2
    assert "OCR engine died" in crashed[0]["error"]
    assert sum(1 for r in rows if r["status"] == "ok") == 2


def test_deadline_in_the_past_submits_nothing(corpus, tmp_path, monkeypatch):
    out = tmp_path / "out"
    fake = _FakeChewMany()
    _install(monkeypatch, fake)

    assert run_batch(_options(corpus, out, deadline=1.0)) == 0

    assert fake.calls == []
    assert _ledger(out) == []
    assert json.loads((out / RUN_INFO_FILENAME).read_text(encoding="utf-8"))["reason"] == "deadline"


def test_limit_and_seeded_shuffle_are_deterministic_and_recorded(corpus, tmp_path, monkeypatch):
    out = tmp_path / "out"
    fake = _FakeChewMany()
    _install(monkeypatch, fake)

    run_batch(_options(corpus, out, shuffle=True, seed=7, limit=2))

    expected = sorted(p.stem for p in corpus.iterdir())
    random.Random(7).shuffle(expected)  # noqa: S311
    ran = [p.stem for paths, _ in fake.calls for p in paths]
    assert ran == expected[:2]
    info = json.loads((out / RUN_INFO_FILENAME).read_text(encoding="utf-8"))
    assert info["shuffle_seed"] == 7 and info["n_planned"] == 2
    assert len(_ledger(out)) == 2


def test_stem_collisions_get_sha_suffixed_exports(tmp_path, monkeypatch):
    a = tmp_path / "x" / "paper.pdf"
    b = tmp_path / "y" / "paper.pdf"
    for path, body in ((a, b"one"), (b, b"two")):
        path.parent.mkdir()
        path.write_bytes(b"%PDF-1.4\n" + body)
    _install(monkeypatch, _FakeChewMany())
    out = tmp_path / "out"

    assert run_batch(_options(tmp_path / "x", out, inputs=[str(a), str(b)])) == 0

    rows = _ledger(out)
    assert {r["stem"] for r in rows} == {"paper"}
    ids = {r["paper_id"] for r in rows}
    assert len(ids) == 2 and all(pid.startswith("paper-") for pid in ids)
    for pid in ids:
        assert (out / f"{pid}.json").is_file()
    info = json.loads((out / RUN_INFO_FILENAME).read_text(encoding="utf-8"))
    assert set(info["collisions"]) == ids


def test_dry_run_creates_nothing_and_prints_the_plan(corpus, tmp_path, monkeypatch, capsys):
    def boom(local):
        raise AssertionError("dry-run must not open a pipeline")

    monkeypatch.setattr("bibr.batch.runner.open_chew_many", boom)
    out = tmp_path / "out"
    options = _options(corpus, out, dry_run=True)
    options.local.summary = ["ocr paddle · model-x"]

    assert run_batch(options) == 0

    assert not out.exists()
    printed = capsys.readouterr().out
    assert "Dry run — nothing was processed." in printed
    assert "Input (4 files)" in printed
    assert "to run" in printed and "4 of 4" in printed
    assert "ocr paddle · model-x" in printed


def test_no_input_files_is_a_usage_error(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeChewMany())
    code = run_batch(_options(tmp_path, tmp_path / "out", inputs=[str(tmp_path / "missing")]))
    assert code == 2
    assert not (tmp_path / "out").exists()


def test_preflight_blocks_the_run_before_any_model_loads(corpus, tmp_path, monkeypatch):
    def boom(local):
        raise AssertionError("must not open a pipeline when preflight fails")

    monkeypatch.setattr("bibr.batch.runner.open_chew_many", boom)
    options = _options(corpus, tmp_path / "out")
    options.local.preflight = lambda files: "No local OCR runtime can start here"

    assert run_batch(options) == 1
    assert not (tmp_path / "out" / RUN_INFO_FILENAME).exists()


# --- executor unit ------------------------------------------------------------------


def test_local_executor_chunks_and_reports_each_paper(corpus):
    fake = _FakeChewMany()
    items = [BatchItem(path=p, paper_id=p.stem, stem=p.stem) for p in sorted(corpus.iterdir())]
    seen = []

    reason = LocalExecutor(fake, batch_size=3).run(
        items, on_outcome=lambda item, outcome: seen.append((item.paper_id, outcome.status))
    )

    assert reason == "completed"
    assert [size for _, size in fake.calls] == [3, 1]
    assert seen == [("bad1", "failed"), ("good1", "ok"), ("good2", "ok"), ("good3", "ok")]


def test_local_executor_result_count_mismatch_is_loud(corpus):
    items = [BatchItem(path=p, paper_id=p.stem, stem=p.stem) for p in sorted(corpus.iterdir())]
    with pytest.raises(RuntimeError, match="returned 1 results for 4 paths"):
        LocalExecutor(lambda paths, n: [_Result({})], batch_size=10).run(
            items, on_outcome=lambda *_: None
        )


def test_parse_deadline_accepts_epoch_and_iso():
    assert parse_deadline("1700000000") == 1700000000.0
    assert parse_deadline("2026-09-03T06:00:00Z") == 1788415200.0
    assert parse_deadline("2026-09-03T06:00:00+02:00") == 1788408000.0
    with pytest.raises(ValueError, match="--deadline"):
        parse_deadline("tomorrow")


# --- Parquet tables ----------------------------------------------------------------


class _RealExports(_FakeChewMany):
    """Returns a real v12 export (the shared demo paper) for every good stem."""

    def __call__(self, paths, batch_size):
        from bibr.export.json_export import _export_paper_payload
        from tests.export.conftest import _demo_paper

        results = super().__call__(paths, batch_size)
        return [
            _Result(_export_paper_payload(_demo_paper(with_refs=True))) if r.ok else r
            for r in results
        ]


def test_tables_are_written_keyed_by_the_batch_paper_id(corpus, tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    _install(monkeypatch, _RealExports())
    out = tmp_path / "out"
    run_batch(_options(corpus, out))

    # Every export printed the same DOI; the batch id keeps them apart.
    exported = json.loads((out / "good1.json").read_text(encoding="utf-8"))
    assert exported["paper_id"] == "good1"
    papers = pq.read_table(out / "tables" / "paper.parquet").column("paper_id").to_pylist()
    assert papers == ["good1", "good2", "good3"]
    bib = pq.read_table(out / "tables" / "bib.parquet")
    assert set(bib.column("paper_id").to_pylist()) == {"good1", "good2", "good3"}


def test_no_tables_option_skips_them(corpus, tmp_path, monkeypatch):
    _install(monkeypatch, _RealExports())
    out = tmp_path / "out"
    run_batch(_options(corpus, out, tables=False))
    assert not (out / "tables").exists()


# --- resume after outages and crashes (real Chewer, stub pipeline) ----------------


class _StubLocalPipeline:
    """Stands in for ``LocalPipeline`` under the real ``WarmChewMany``/``Chewer``.

    ``script(stems)`` runs per ``process_chunk`` call: it may raise (a crash)
    or return ``{stem: exception}`` for files the OCR stage should fail.
    """

    script = staticmethod(lambda stems: {})
    calls: list[list[str]] = []

    def __init__(self, **kwargs):
        pass

    async def process_chunk(self, file_states, progress=None, config=None):
        from tests.test_api import _export_fixture

        stems = [fs.path.stem for fs in file_states]
        _StubLocalPipeline.calls.append(stems)
        failures = _StubLocalPipeline.script(stems)
        for fs in file_states:
            exc = failures.get(fs.path.stem)
            if exc is not None:
                fs.set_error(
                    f"OCR failed for all pages: {exc}", code="ocr_failed", stage="ocr", exc=exc
                )
            else:
                data = _export_fixture()
                data["paper_id"] = fs.paper_id or fs.path.stem
                fs.result_json = data

    async def aclose(self):
        pass


@pytest.fixture
def stub_pipeline(monkeypatch):
    _StubLocalPipeline.calls = []
    _StubLocalPipeline.script = staticmethod(lambda stems: {})
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _StubLocalPipeline)
    return _StubLocalPipeline


def _real_options(corpus: Path, out: Path, **overrides) -> BatchOptions:
    local = LocalOptions(chew_options={"memory_mode": "balanced", "no_llm": True}, batch_size=2)
    return _options(corpus, out, local=local, tables=False, **overrides)


def _latest_codes(out: Path) -> dict[str, tuple[str, str | None]]:
    latest = Ledger(out / LEDGER_FILENAME).latest()
    return {pid: (row["status"], row["error_code"]) for pid, row in sorted(latest.items())}


def test_an_ocr_outage_is_recorded_as_upstream_unavailable_and_resumed(
    corpus, tmp_path, stub_pipeline
):
    """The OCR server dies after the first chunk: those papers never got a
    verdict, so the next run picks them up without --retry-failed."""
    import httpx

    out = tmp_path / "out"
    refused = httpx.ConnectError("[Errno 111] Connection refused")
    stub_pipeline.script = staticmethod(
        lambda stems: dict.fromkeys(stems, refused) if len(stub_pipeline.calls) > 1 else {}
    )

    assert run_batch(_real_options(corpus, out)) == 1
    assert _latest_codes(out) == {
        "bad1": ("ok", None),
        "good1": ("ok", None),
        "good2": ("failed", "upstream_unavailable"),
        "good3": ("failed", "upstream_unavailable"),
    }
    failed = Ledger(out / LEDGER_FILENAME).latest()["good2"]
    assert failed["failed_stage"] == "ocr"
    assert failed["error"] == "OCR failed for all pages: [Errno 111] Connection refused"

    stub_pipeline.calls = []
    stub_pipeline.script = staticmethod(lambda stems: {})
    assert run_batch(_real_options(corpus, out)) == 0
    assert stub_pipeline.calls == [["good2", "good3"]]
    assert {row[0] for row in _latest_codes(out).values()} == {"ok"}


def test_a_crashed_chunk_runs_its_papers_alone_and_a_crash_is_retried_once(
    corpus, tmp_path, stub_pipeline
):
    """bad1 crashes the pipeline whenever it is in a chunk; its neighbour must
    not fail with it, and resume retries the crash once, not forever."""
    out = tmp_path / "out"

    def crash_on_bad1(stems):
        if "bad1" in stems:
            raise KeyError("level")
        return {}

    stub_pipeline.script = staticmethod(crash_on_bad1)

    assert run_batch(_real_options(corpus, out)) == 1
    assert stub_pipeline.calls == [["bad1", "good1"], ["bad1"], ["good1"], ["good2", "good3"]]
    assert _latest_codes(out) == {
        "bad1": ("failed", "chunk_error"),
        "good1": ("ok", None),
        "good2": ("ok", None),
        "good3": ("ok", None),
    }
    assert Ledger(out / LEDGER_FILENAME).latest()["bad1"]["error"] == "KeyError: 'level'"

    stub_pipeline.calls = []
    assert run_batch(_real_options(corpus, out)) == 1
    assert stub_pipeline.calls == [["bad1"]]

    stub_pipeline.calls = []
    assert run_batch(_real_options(corpus, out)) == 0
    assert stub_pipeline.calls == []


def test_a_pipeline_that_cannot_be_built_is_retried_on_the_next_run(corpus, tmp_path, monkeypatch):
    out = tmp_path / "out"

    class Broken:
        def __init__(self, **kwargs):
            raise RuntimeError("CUDA driver initialization failed")

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", Broken)
    assert run_batch(_real_options(corpus, out)) == 1
    assert {row[1] for row in _latest_codes(out).values()} == {"chunk_error"}

    fake = _FakeChewMany()
    _install(monkeypatch, fake)
    run_batch(_options(corpus, out))
    assert sum(len(paths) for paths, _ in fake.calls) == 4


def test_an_input_named_run_info_keeps_its_export(tmp_path, monkeypatch):
    papers = tmp_path / "papers"
    papers.mkdir()
    for name in ("run_info", "other"):
        (papers / f"{name}.pdf").write_bytes(b"%PDF-1.4\n" + name.encode())
    _install(monkeypatch, _FakeChewMany())
    out = tmp_path / "out"

    assert run_batch(_options(papers, out)) == 0

    ids = sorted(Ledger(out / LEDGER_FILENAME).latest())
    assert ids[0] == "other" and ids[1].startswith("run_info-") and len(ids) == 2
    assert json.loads((out / f"{ids[1]}.json").read_text(encoding="utf-8"))["paper_id"] == ids[1]
    assert json.loads((out / RUN_INFO_FILENAME).read_text(encoding="utf-8"))["n_ok"] == 2


def test_a_same_named_newcomer_does_not_rename_a_processed_paper(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "paper.pdf").write_bytes(b"%PDF-1.4\na")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("a/paper.pdf\n")
    out = tmp_path / "out"
    _install(monkeypatch, _FakeChewMany())
    run_batch(_options(tmp_path, out, inputs=[str(manifest)]))

    (tmp_path / "b" / "paper.pdf").write_bytes(b"%PDF-1.4\nb")
    manifest.write_text("a/paper.pdf\nb/paper.pdf\n")
    second = _FakeChewMany()
    _install(monkeypatch, second)
    run_batch(_options(tmp_path, out, inputs=[str(manifest)]))

    assert [[p.parent.name for p in paths] for paths, _ in second.calls] == [["b"]]
    exports = sorted(p.stem for p in out.glob("*.json") if p.name != RUN_INFO_FILENAME)
    assert exports == sorted(Ledger(out / LEDGER_FILENAME).latest())
    assert len(exports) == 2 and exports[0] == "paper"


def test_tables_leave_out_exports_of_another_schema_major(corpus, tmp_path, monkeypatch, capsys):
    """A resumed out dir that still holds a v11 export must not lose its tables."""
    import pyarrow.parquet as pq

    out = tmp_path / "out"
    out.mkdir()
    (out / "old.json").write_text(
        json.dumps({"paper_id": "old", "schema_version": "11.0", "bib": []}), encoding="utf-8"
    )
    Ledger(out / LEDGER_FILENAME).append({"paper_id": "old", "status": "ok"})
    _install(monkeypatch, _RealExports())

    run_batch(_options(corpus, out))

    papers = pq.read_table(out / "tables" / "paper.parquet").column("paper_id").to_pylist()
    assert papers == ["good1", "good2", "good3"]


def test_tables_of_an_older_major_stay_when_no_current_export_exists(tmp_path, capsys):
    from rich.console import Console

    from bibr.batch.runner import write_batch_tables

    out = tmp_path / "out"
    (out / "tables").mkdir(parents=True)
    (out / "tables" / "paper.parquet").write_bytes(b"written by bibr 0.5")
    (out / "old.json").write_text(json.dumps({"paper_id": "old", "schema_version": "11.0"}))
    ledger = Ledger(out / LEDGER_FILENAME)
    ledger.append({"paper_id": "old", "status": "ok"})
    console = Console(file=io.StringIO(), width=200)

    write_batch_tables(out, ledger, ledger.read(), console=console)

    assert (out / "tables" / "paper.parquet").read_bytes() == b"written by bibr 0.5"
    assert " ".join(console.file.getvalue().split()) == (
        "! 1 export(s) of another schema major left out of the tables (old.json: 11.0); "
        "re-run them with --force to include them"
    )
