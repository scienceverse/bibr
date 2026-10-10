"""Remote executor audit fixes: abandoned jobs are cancelled on the serve, a
result that vanished before the fetch is re-submitted, and a malformed export
never reaches ``<out>``.

Runs on the scripted serve and virtual clock of ``test_remote``.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from bibr.batch.ledger import INTERRUPTED, LEDGER_FILENAME, Ledger
from bibr.batch.remote import RemoteExecutor, RemoteOptions
from bibr.batch.runner import BatchOptions, run_batch
from tests.batch.test_remote import FakeServe, _Clock, _items, _NoJitter, _run, _Sleeper


class CancellableServe(FakeServe):
    """``FakeServe`` plus ``DELETE /papers/jobs/{id}`` as the serve answers it:
    a queued job is failed as cancelled (200), a running or finished one 409,
    an unknown one 404. Stems in *queued* never leave the queue on their own."""

    def __init__(self, *, queued=(), **kwargs):
        super().__init__(**kwargs)
        self.queued = set(queued)
        self.cancelled: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method != "DELETE":
            return super().handler(request)
        self.requests.append(("DELETE", request.url.path))
        job_id = request.url.path.rsplit("/", 1)[-1]
        job = self.jobs.get(job_id)
        if job is None:
            return httpx.Response(404, json={"detail": "job not found"})
        status = self._status(job)
        if status != "queued":
            return httpx.Response(409, json={"detail": "not queued", "status": status})
        job["cancelled"] = True
        job["failure"] = ({"detail": "cancelled", "error_code": "job_cancelled"}, 410)
        self.cancelled.append(job_id)
        return httpx.Response(200, json={"job_id": job_id, "status": "failed"})

    def _status(self, job: dict) -> str:
        if job.get("cancelled"):
            return "failed"
        if job["stem"] in self.queued:
            return "queued"
        return super()._status(job)

    def deletes(self) -> list[str]:
        return [path for method, path in self.requests if method == "DELETE"]


def _executor(handler, **overrides) -> RemoteExecutor:
    clock = _Clock()
    options = {"serve_url": "http://serve", "token": "tok", "poll_interval": 1.0}
    options.update(overrides)
    return RemoteExecutor(
        RemoteOptions(**options),
        transport=httpx.MockTransport(handler),
        sleep=_Sleeper(clock),
        clock=clock,
        rng=_NoJitter(),
    )


def _valid_export(paper_id: str) -> dict:
    """A real current-major export (the shared demo paper) under *paper_id*."""
    from bibr.export.json_export import _export_paper_payload
    from tests.export.conftest import _demo_paper

    data = _export_paper_payload(_demo_paper(with_refs=False))
    data["paper_id"] = paper_id
    return data


def _serving(serve: FakeServe, exports: dict[str, dict]):
    """*serve*'s handler, answering a finished job's result with ``exports[stem]``."""

    def handler(request: httpx.Request) -> httpx.Response:
        response = serve.handler(request)
        if request.method == "GET" and request.url.path.endswith("/result"):
            stem = serve.jobs[request.url.path.split("/")[3]]["stem"]
            if response.status_code == 200 and stem in exports:
                return httpx.Response(200, json=exports[stem])
        return response

    return handler


# --- abandoned jobs are cancelled on the serve ------------------------------------


async def test_poll_timeout_cancels_the_queued_job(tmp_path):
    serve = CancellableServe(queued={"slow"})
    executor = _executor(serve.handler, poll_timeout=5.0, poll_interval=2.0)

    _, [(_, outcome)] = await _run(executor, _items(tmp_path, "slow"))

    assert outcome.error_code == "poll_timeout"
    assert outcome.extra["job_id"] == "job1"
    assert serve.cancelled == ["job1"]  # no longer holds a JOBS_MAX_ACTIVE slot
    assert len(serve.submits) == 1


async def test_poll_errors_cancel_the_old_job_before_resubmitting(tmp_path):
    serve = CancellableServe()

    def flaky(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/papers/jobs/job1":
            return httpx.Response(503, json={"detail": "proxy down"})
        return serve.handler(request)

    executor = _executor(flaky, retries=1)

    _, [(_, outcome)] = await _run(executor, _items(tmp_path, "good"))

    assert outcome.ok and outcome.extra["job_id"] == "job2"
    assert outcome.extra["retries"] == 1
    assert serve.cancelled == ["job1"]  # job1 was still queued: it never runs
    posts = [i for i, (method, _) in enumerate(serve.requests) if method == "POST"]
    assert serve.requests.index(("DELETE", "/papers/jobs/job1")) < posts[1]


async def test_ctrl_c_cancels_the_jobs_still_in_flight_after_the_grace(tmp_path):
    serve = CancellableServe(queued={"slow"})
    executor = _executor(serve.handler, concurrency=2, grace=0.0)
    stop = asyncio.Event()
    seen = []

    def on_outcome(item, outcome):
        seen.append((item.paper_id, outcome.error_code))
        stop.set()

    reason = await executor.run(_items(tmp_path, "good", "slow"), on_outcome=on_outcome, stop=stop)

    assert reason == "stopped"
    assert seen == [("good", None), ("slow", INTERRUPTED)]
    slow_job = next(job_id for job_id, job in serve.jobs.items() if job["stem"] == "slow")
    assert serve.deletes() == [f"/papers/jobs/{slow_job}"]  # the finished one is left alone
    assert serve.cancelled == [slow_job]


async def test_a_persistently_failing_fetch_cancels_before_resubmitting(tmp_path):
    serve = CancellableServe()

    def down(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/result"):
            raise httpx.ReadTimeout("connection reset", request=request)
        return serve.handler(request)

    executor = _executor(down, retries=1)

    _, [(_, outcome)] = await _run(executor, _items(tmp_path, "good"))

    assert outcome.error_code == "connection_error"
    assert serve.deletes() == ["/papers/jobs/job1", "/papers/jobs/job2"]


async def test_jobs_that_ran_to_the_end_or_vanished_are_not_cancelled(tmp_path):
    serve = CancellableServe(fail={"bad": ({"detail": "no text", "error_code": "OCR_EMPTY"}, 422)})

    def forgetful(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/papers/jobs/job3":
            return httpx.Response(404, json={"detail": "job not found"})
        return serve.handler(request)

    executor = _executor(forgetful, concurrency=1, max_concurrency=1)

    _, seen = await _run(executor, _items(tmp_path, "good", "bad", "lost"))

    assert [(i.paper_id, o.error_code) for i, o in seen] == [
        ("good", None),
        ("bad", "OCR_EMPTY"),
        ("lost", None),  # job3 vanished, job4 ran
    ]
    assert serve.deletes() == []


@pytest.mark.parametrize("failure", ["unreachable", "http_500"])
async def test_a_failed_cancel_is_logged_and_keeps_the_papers_outcome(tmp_path, caplog, failure):
    serve = FakeServe(stuck={"slow"})

    def no_delete(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            if failure == "unreachable":
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(500, json={"detail": "boom"})
        return serve.handler(request)

    executor = _executor(no_delete, poll_timeout=5.0, poll_interval=2.0)

    with caplog.at_level(logging.WARNING, logger="bibr.batch.remote"):
        _, [(_, outcome)] = await _run(executor, _items(tmp_path, "slow"))

    assert outcome.error_code == "poll_timeout"
    assert "could not cancel job job1" in caplog.text


# --- a result that vanished before the fetch is re-submitted ----------------------


async def test_a_result_404_after_success_resubmits_the_paper(tmp_path):
    serve = FakeServe()

    def evicting(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/papers/jobs/job1/result":
            return httpx.Response(404, json={"detail": "job result no longer available"})
        return serve.handler(request)

    executor = _executor(evicting, retries=2)

    _, [(_, outcome)] = await _run(executor, _items(tmp_path, "good"))

    assert outcome.ok
    assert outcome.extra == {"job_id": "job2", "retries": 1}
    assert executor.gate.size == 3  # not backpressure: in-flight grew from 2


async def test_a_result_that_keeps_vanishing_is_left_for_resume(tmp_path):
    serve = FakeServe()

    def evicting(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/result"):
            return httpx.Response(404, json={"detail": "job not found"})
        return serve.handler(request)

    executor = _executor(evicting, retries=1)

    _, [(_, outcome)] = await _run(executor, _items(tmp_path, "good"))

    assert outcome.error_code == "job_lost"  # not a permanent http_404
    assert outcome.extra["transient_exhausted"] is True
    assert outcome.extra["job_id"] == "job2"


# --- a remote export is validated before it counts as ok --------------------------


@pytest.mark.parametrize(
    ("patch", "where"),
    [
        ({"text": "not a list"}, "text"),
        ({"schema_version": "12"}, "schema_version"),
        ({"schema_version": 12}, "schema_version"),
    ],
)
async def test_a_malformed_export_of_this_major_fails_the_paper(tmp_path, patch, where):
    serve = FakeServe()
    executor = _executor(_serving(serve, {"bad": _valid_export("x") | patch}))

    _, [(_, outcome)] = await _run(executor, _items(tmp_path, "bad"))

    assert not outcome.ok
    assert outcome.export is None
    assert outcome.error_code == "invalid_remote_export"
    assert f"{where}:" in outcome.error
    assert outcome.extra["job_id"] == "job1"
    assert len(serve.submits) == 1  # the serve would only produce it again


@pytest.mark.parametrize("kind", ["current", "older-major", "newer-major", "pre-11"])
async def test_valid_and_other_major_exports_are_kept(tmp_path, kind):
    """Another major (or a pre-11 export) is left out of the tables with a
    warning, as before; only this major's exports must pass the reader."""
    export = {
        # The batch id replaces the export's own paper_id, as the runner writes it.
        "current": {k: v for k, v in _valid_export("x").items() if k != "paper_id"},
        "older-major": {"paper_id": "old", "schema_version": "11.0", "bib": "a v11 export"},
        "newer-major": {"paper_id": "new", "schema_version": "13.0", "bib": "a v13 export"},
        "pre-11": {"info": {"title": "pre-11"}, "text": "no schema_version"},
    }[kind]
    serve = FakeServe()
    executor = _executor(_serving(serve, {"good": export}))

    _, [(_, outcome)] = await _run(executor, _items(tmp_path, "good"))

    assert outcome.ok
    assert outcome.export == export


def test_a_malformed_export_never_blocks_the_tables(tmp_path, monkeypatch):
    """One bad export used to be written and counted ok, after which every
    tables rebuild of ``<out>`` failed on it."""
    import pyarrow.parquet as pq

    monkeypatch.setattr("bibr.batch.runner.local_build_sha", lambda: None)
    papers = tmp_path / "papers"
    papers.mkdir()
    for stem in ("bad", "good"):
        (papers / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n" + stem.encode())
    out = tmp_path / "out"
    serve = FakeServe()
    exports = {
        "bad": _valid_export("bad") | {"bib": {"b1": "not a list"}},
        "good": _valid_export("good"),
    }
    remote = RemoteOptions(serve_url="http://serve", token="tok", poll_interval=0.001)  # noqa: S106
    options = BatchOptions(inputs=[str(papers)], out=out, remote=remote)

    code = run_batch(options, transport=httpx.MockTransport(_serving(serve, exports)))

    assert code == 1
    assert not (out / "bad.json").exists()
    assert json.loads((out / "good.json").read_text())["paper_id"] == "good"
    latest = Ledger(out / LEDGER_FILENAME).latest()
    assert latest["bad"]["error_code"] == "invalid_remote_export"
    assert latest["good"]["status"] == "ok"
    table = pq.read_table(out / "tables" / "paper.parquet")
    assert table.column("paper_id").to_pylist() == ["good"]
