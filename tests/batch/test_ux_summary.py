"""What ``bibr batch`` prints: nothing-to-do resumes, interrupts, warnings, stage shares."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from bibr.batch.report import compute_report, render_report
from bibr.batch.runner import BatchOptions, LocalOptions, run_batch


class _Result:
    ok = True

    def __init__(self, stem: str):
        self.data = {
            "paper_id": stem,
            "text": [{"id": 1}],
            "bib": [],
            "processing_warnings": [],
        }


class _FakeChewMany:
    def __init__(self, exc: BaseException | None = None):
        self.exc = exc

    def __call__(self, paths, batch_size):
        if self.exc is not None:
            raise self.exc
        return [_Result(path.stem) for path in paths]

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _stable_build_sha(monkeypatch):
    monkeypatch.setattr("bibr.batch.runner.local_build_sha", lambda: "deadbeef")


@pytest.fixture
def corpus(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    for name in ("p1", "p2"):
        (papers / f"{name}.pdf").write_bytes(b"%PDF-1.4\n" + name.encode())
    return papers


def _run(monkeypatch, corpus, out, fake) -> tuple[int, str, str]:
    monkeypatch.setattr("bibr.batch.runner.open_chew_many", lambda local: fake)
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, color_system=None)
    options = BatchOptions(
        inputs=[str(corpus)],
        out=out,
        local=LocalOptions(chew_options={"memory_mode": "balanced"}, batch_size=2),
        cli_options={},
        tables=False,
    )
    code = run_batch(options, console=console)
    return code, buffer.getvalue(), ""


def test_a_resume_with_nothing_to_do_says_so(monkeypatch, corpus, tmp_path, capsys):
    out = tmp_path / "out"
    assert _run(monkeypatch, corpus, out, _FakeChewMany())[0] == 0
    capsys.readouterr()

    code, console_text, _ = _run(monkeypatch, corpus, out, _FakeChewMany())
    printed = capsys.readouterr().out

    assert code == 0
    assert "nothing to do: 2 papers already ok in the ledger" in console_text
    # No table of zeros for a run that processed nothing.
    assert "0 ok · 0 failed · 0 total" not in printed


def test_an_interrupt_is_not_counted_as_failures(monkeypatch, corpus, tmp_path, capsys):
    out = tmp_path / "out"
    code, console_text, _ = _run(monkeypatch, corpus, out, _FakeChewMany(KeyboardInterrupt()))
    printed = capsys.readouterr().out

    assert code == 130
    assert "interrupted — interrupted" not in console_text
    assert "p1: interrupted" in console_text
    assert "0 ok · 0 failed · 2 interrupted · 2 total" in printed
    assert "2 papers interrupted or not started — re-run the same command" in console_text


def _row(paper_id, **fields):
    base = {
        "paper_id": paper_id,
        "status": "ok",
        "started_at": "2026-09-02T10:00:00+00:00",
        "finished_at": "2026-09-02T10:01:00+00:00",
        "run_id": "r1",
    }
    base.update(fields)
    return base


def test_a_fast_runs_llm_skipped_note_is_not_a_warning():
    rows = [
        _row("a", warnings={"count": 1, "codes": {"LLM_SKIPPED": 1}}),
        _row("b", warnings={"count": 2, "codes": {"LLM_SKIPPED": 1, "REF_SEG_GEOM_CASCADE": 1}}),
    ]
    report = compute_report(rows)
    assert report["warnings"]["papers_with_warnings"] == 1
    assert report["warnings"]["top"] == [{"warning": "REF_SEG_GEOM_CASCADE", "count": 1}]
    # The code stays in the JSON.
    assert report["warnings"]["expected"] == [{"warning": "LLM_SKIPPED", "count": 2}]

    text = render_report(report)
    assert "LLM_SKIPPED" not in text
    assert "REF_SEG_GEOM_CASCADE" not in text
    assert "1 paper · reference split: geometry declined ×1" in text


def test_an_unmapped_code_is_shown_as_it_is_and_no_warnings_is_plain():
    report = compute_report([_row("a", warnings={"count": 1, "codes": {"SOME_NEW_CODE": 1}})])
    assert "SOME_NEW_CODE ×1" in render_report(report)
    clean = render_report(compute_report([_row("a", warnings={"count": 0, "codes": {}})]))
    assert "warnings     0 papers" in clean


def test_stages_that_round_to_zero_are_left_out():
    rows = [_row("a", stage_times={"parse_segment": 99.8, "ocr": 0.2})]
    text = render_report(compute_report(rows))
    assert "parse_segment 100%" in text
    assert "ocr 0%" not in text
