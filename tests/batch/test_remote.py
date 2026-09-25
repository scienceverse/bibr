"""Remote executor against an ``httpx.MockTransport`` fake of the serve job API.

The fake scripts the protocol (202 → queued → running → succeeded/failed,
429 and 5xx on submit, a job that never finishes, an upstream outage) and
the executor runs on a virtual clock, so every backoff is asserted exactly
without sleeping.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest

from bibr.batch.ledger import INTERRUPTED, LEDGER_FILENAME, Ledger
from bibr.batch.manifest import BatchItem
from bibr.batch.remote import (
    AdaptiveGate,
    RemoteExecutor,
    RemoteOptions,
    looks_transient,
    resolve_token,
)
from bibr.batch.runner import RUN_INFO_FILENAME, BatchOptions, run_batch

_FILENAME_RE = re.compile(rb'filename="([^"]+)"')
_FIELD_RE = re.compile(rb'name="([A-Za-z_]+)"\r\n\r\n([^\r]*)\r\n')


def _export(stem: str) -> dict:
    return {
        "info": {"title": stem},
        "text": [{"id": 1}, {"id": 2}],
        "bib": [{"id": "b1"}],
        "bib_match": [],
        "extraction": {"timings": {"ocr": 1.0, "extract": 0.5}, "total_seconds": 1.5},
        "llm_usage": {"m": {"total_tokens": 7}},
        "processing_warnings": ["VALIDATION:warning:REF_YEAR: x"],
    }


class FakeServe:
    """Scripted job API state machine behind ``httpx.MockTransport``."""

    def __init__(
        self,
        *,
        ready_script=None,
        submit_script=None,
        polls_to_finish: int = 2,
        fail: dict | None = None,
        fail_once: dict | None = None,
        stuck=(),
        auth_token: str | None = None,
    ):
        self.ready_script = list(ready_script or [])
        self.submit_script = list(submit_script or [])
        self.polls_to_finish = polls_to_finish
        self.fail = dict(fail or {})
        self.fail_once = dict(fail_once or {})
        self.stuck = set(stuck)
        self.auth_token = auth_token
        self.jobs: dict[str, dict] = {}
        self.submits: list[tuple[str, dict]] = []
        self.ready_calls = 0
        self.requests: list[tuple[str, str]] = []
        self._n = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))
        if (
            self.auth_token
            and path != "/ready"
            and request.headers.get("authorization") != f"Bearer {self.auth_token}"
        ):
            return httpx.Response(401, json={"detail": "unauthorized"})
        if path == "/ready":
            self.ready_calls += 1
            if self.ready_script:
                code, body = self.ready_script.pop(0)
                return httpx.Response(code, json=body)
            return httpx.Response(
                200, json={"status": "ready", "checks": {"ocr": "ok"}, "build_sha": "serve-sha-1"}
            )
        if path == "/papers/jobs" and request.method == "POST":
            body = request.content
            match = _FILENAME_RE.search(body)
            filename = match.group(1).decode() if match else "?"
            form = {k.decode(): v.decode() for k, v in _FIELD_RE.findall(body) if k != b"file"}
            self.submits.append((filename, form))
            if self.submit_script:
                code = self.submit_script.pop(0)
                if code == 429:
                    return httpx.Response(
                        429, json={"detail": "active job cap reached"}, headers={"retry-after": "7"}
                    )
                return httpx.Response(code, json={"detail": f"scripted {code}"})
            self._n += 1
            job_id = f"job{self._n}"
            stem = Path(filename).stem
            failure = self.fail_once.pop(stem, None) or self.fail.get(stem)
            self.jobs[job_id] = {"stem": stem, "polls": 0, "failure": failure}
            return httpx.Response(
                202,
                json={"job_id": job_id, "status": "queued", "status_url": f"/papers/jobs/{job_id}"},
            )
        if path.startswith("/papers/jobs/"):
            parts = path.split("/")
            job = self.jobs.get(parts[3])
            if job is None:
                return httpx.Response(404, json={"detail": "job not found"})
            if len(parts) == 5 and parts[4] == "result":
                status = self._status(job)
                if status in ("queued", "running"):
                    return httpx.Response(409, json={"detail": "job not finished"})
                if status == "failed":
                    error, http_status = job["failure"]
                    return httpx.Response(http_status, json=error)
                return httpx.Response(200, json=_export(job["stem"]))
            job["polls"] += 1
            status = self._status(job)
            body = {"job_id": parts[3], "status": status, "filename": job["stem"] + ".pdf"}
            if status == "failed":
                body["error"] = job["failure"][0]
            if status == "succeeded":
                body["result_url"] = f"/papers/jobs/{parts[3]}/result"
            return httpx.Response(200, json=body)
        return httpx.Response(404, json={"detail": "no route"})

    def _status(self, job: dict) -> str:
        if job["stem"] in self.stuck:
            return "running"
        if job["polls"] < self.polls_to_finish:
            return "queued" if job["polls"] < 1 else "running"
        return "failed" if job["failure"] else "succeeded"


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Sleeper:
    def __init__(self, clock: _Clock):
        self.clock = clock
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.now += seconds
        await asyncio.sleep(0)


class _NoJitter:
    def uniform(self, a: float, b: float) -> float:
        return 0.0


def _items(tmp_path: Path, *stems: str) -> list[BatchItem]:
    items = []
    for stem in stems:
        path = tmp_path / f"{stem}.pdf"
        path.write_bytes(b"%PDF-1.4\n" + stem.encode())
        items.append(BatchItem(path=path, paper_id=stem, stem=stem))
    return items


def _executor(serve: FakeServe, **overrides) -> tuple[RemoteExecutor, _Sleeper]:
    clock = _Clock()
    sleeper = _Sleeper(clock)
    options = {"serve_url": "http://serve/", "token": "tok", "poll_interval": 1.0}
    options.update(overrides)
    executor = RemoteExecutor(
        RemoteOptions(**options),
        transport=httpx.MockTransport(serve.handler),
        sleep=sleeper,
        clock=clock,
        rng=_NoJitter(),
    )
    return executor, sleeper


async def _run(executor: RemoteExecutor, items, **kwargs):
    seen: list[tuple[BatchItem, object]] = []
    reason = await executor.run(items, on_outcome=lambda i, o: seen.append((i, o)), **kwargs)
    return reason, seen


# --- happy path -----------------------------------------------------------------------


async def test_success_path_queued_running_succeeded(tmp_path):
    serve = FakeServe()
    executor, sleeper = _executor(serve, form={"refs": "llm", "consolidate": "fill"})
    (item,) = _items(tmp_path, "good")

    reason, seen = await _run(executor, [item])

    assert reason == "completed"
    [(seen_item, outcome)] = seen
    assert seen_item is item
    assert outcome.ok
    assert outcome.export["info"]["title"] == "good"
    assert outcome.extra == {"job_id": "job1", "retries": 0}
    assert outcome.sha256 and outcome.size == item.path.stat().st_size
    assert outcome.duration_s == sum(sleeper.calls)  # virtual clock: the poll sleeps
    assert executor.ready["build_sha"] == "serve-sha-1"
    assert serve.submits == [("good.pdf", {"refs": "llm", "consolidate": "fill"})]
    assert executor.gate.size == 3  # grew from 2 toward max 4


async def test_failed_job_is_permanent_and_keeps_the_serve_error_code(tmp_path):
    serve = FakeServe(
        fail={"bad": ({"message": "ref parse failed", "error_code": "REF_PARSE_FAILED"}, 422)}
    )
    executor, _ = _executor(serve)
    (item,) = _items(tmp_path, "bad")

    _, [(_, outcome)] = await _run(executor, [item])

    assert not outcome.ok
    assert outcome.error_code == "REF_PARSE_FAILED"
    assert outcome.error == "ref parse failed"
    assert outcome.extra["http_status"] == 422
    assert outcome.extra["job_id"] == "job1"
    assert len(serve.submits) == 1  # no retry
    assert executor.gate.size == 2  # a paper's own failure is not backpressure


# --- backpressure and transient failures --------------------------------------------


async def test_429_shrinks_in_flight_to_min_and_waits_retry_after(tmp_path):
    serve = FakeServe(submit_script=[429])
    executor, sleeper = _executor(serve, concurrency=3, min_concurrency=1, max_concurrency=4)
    (item,) = _items(tmp_path, "good")

    _, [(_, outcome)] = await _run(executor, [item])

    assert outcome.ok
    assert len(serve.submits) == 2
    assert executor.stats["submit_429"] == 1
    assert executor.stats["transient_retries"] == 0
    assert 7.0 in sleeper.calls  # Retry-After honoured
    assert executor.gate.size == 2  # dropped to min 1, grew by one on success
    assert outcome.extra["retries"] == 0  # 429 is not a failure


async def test_503_on_submit_is_retried_with_backoff(tmp_path):
    serve = FakeServe(submit_script=[503])
    executor, sleeper = _executor(serve, retries=3)
    (item,) = _items(tmp_path, "good")

    _, [(_, outcome)] = await _run(executor, [item])

    assert outcome.ok
    assert outcome.extra["retries"] == 1
    assert executor.stats["transient_retries"] == 1
    assert sleeper.calls[0] == 5.0  # first backoff step
    assert len(serve.submits) == 2
    assert executor.gate.size == 2  # shrank to 1 on the 503, grew back on success


async def test_transient_failures_exhaust_retries_and_record_the_last_code(tmp_path):
    serve = FakeServe(submit_script=[503, 503])
    executor, _ = _executor(serve, retries=1)
    (item,) = _items(tmp_path, "good")

    _, [(_, outcome)] = await _run(executor, [item])

    assert not outcome.ok
    assert outcome.error_code == "http_503"
    assert outcome.extra["transient_exhausted"] is True
    assert outcome.extra["retries"] == 1
    assert len(serve.submits) == 2


async def test_connection_error_on_submit_is_transient(tmp_path):
    calls = 0

    def flaky(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.path == "/papers/jobs" and calls == 2:
            raise httpx.ConnectError("boom", request=request)
        return serve.handler(request)

    serve = FakeServe()
    clock = _Clock()
    sleeper = _Sleeper(clock)
    executor = RemoteExecutor(
        RemoteOptions(serve_url="http://serve", token="t", poll_interval=1.0, retries=2),  # noqa: S106
        transport=httpx.MockTransport(flaky),
        sleep=sleeper,
        clock=clock,
        rng=_NoJitter(),
    )
    (item,) = _items(tmp_path, "good")

    _, [(_, outcome)] = await _run(executor, [item])

    assert outcome.ok
    assert outcome.extra["retries"] == 1
    assert executor.stats["transient_retries"] == 1


async def test_upstream_outage_reported_by_a_failed_job_is_retried(tmp_path):
    serve = FakeServe(fail_once={"flaky": ({"detail": "Circuit breaker 'ocr' is OPEN"}, 502)})
    executor, sleeper = _executor(serve, retries=2)
    (item,) = _items(tmp_path, "flaky")

    _, [(_, outcome)] = await _run(executor, [item])

    assert outcome.ok
    assert len(serve.submits) == 2
    assert outcome.extra["retries"] == 1
    assert outcome.extra["job_id"] == "job2"


async def test_poll_timeout_fails_the_paper_without_retry(tmp_path):
    serve = FakeServe(stuck={"slow"})
    executor, sleeper = _executor(serve, poll_timeout=5.0, poll_interval=2.0, retries=3)
    (item,) = _items(tmp_path, "slow")

    _, [(_, outcome)] = await _run(executor, [item])

    assert not outcome.ok
    assert outcome.error_code == "poll_timeout"
    assert outcome.extra["job_id"] == "job1"
    assert len(serve.submits) == 1
    assert executor.stats["transient_retries"] == 0


async def test_serve_pipeline_timeout_is_retried_once_and_blames_the_paper(tmp_path):
    """A job the serve failed with 504 ran out of PIPELINE_TIMEOUT: the serve
    is healthy, the paper is slow. One more try, no in-flight shrink, and the
    ledger code says so instead of blaming the OCR/LLM services."""
    serve = FakeServe(fail={"thesis": ({"detail": "Pipeline processing timed out"}, 504)})
    executor, sleeper = _executor(serve, retries=3, concurrency=3, max_concurrency=4)
    (item,) = _items(tmp_path, "thesis")

    _, [(_, outcome)] = await _run(executor, [item])

    assert len(serve.submits) == 2
    assert outcome.error_code == "pipeline_timeout"
    assert outcome.error == "Pipeline processing timed out"
    assert outcome.extra == {"http_status": 504, "job_id": "job2", "retries": 1}
    assert executor.gate.size == 3
    assert executor.stats["transient_retries"] == 0
    assert [s for s in sleeper.calls if s >= 5] == [5.0]


async def test_a_timeout_that_clears_on_the_retry_succeeds(tmp_path):
    serve = FakeServe(fail_once={"busy": ({"detail": "Pipeline processing timed out"}, 504)})
    executor, _ = _executor(serve, retries=3)
    (item,) = _items(tmp_path, "busy")

    _, [(_, outcome)] = await _run(executor, [item])

    assert outcome.ok
    assert outcome.extra == {"job_id": "job2", "retries": 1}


async def test_lost_job_is_transient(tmp_path):
    serve = FakeServe()

    def forgetful(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/papers/jobs/job1":
            return httpx.Response(404, json={"detail": "job not found"})
        return serve.handler(request)

    clock = _Clock()
    executor = RemoteExecutor(
        RemoteOptions(serve_url="http://serve", token="t", poll_interval=1.0, retries=1),  # noqa: S106
        transport=httpx.MockTransport(forgetful),
        sleep=_Sleeper(clock),
        clock=clock,
        rng=_NoJitter(),
    )
    (item,) = _items(tmp_path, "good")

    _, [(_, outcome)] = await _run(executor, [item])

    assert outcome.ok
    assert outcome.extra["job_id"] == "job2"
    assert outcome.extra["retries"] == 1


# --- readiness, auth, stopping ------------------------------------------------------


async def test_wait_ready_polls_until_ready_and_reports_build_sha(tmp_path):
    serve = FakeServe(ready_script=[(503, {"status": "not_ready"}), (503, {"status": "not_ready"})])
    executor, sleeper = _executor(serve, ready_interval=15.0, ready_timeout=900.0)
    (item,) = _items(tmp_path, "good")
    ready_seen = []

    reason, seen = await _run(executor, [item], on_ready=ready_seen.append)

    assert reason == "completed"
    assert serve.ready_calls == 3
    assert ready_seen == [
        {"status": "ready", "checks": {"ocr": "ok"}, "build_sha": "serve-sha-1", "http_status": 200}
    ]
    assert sleeper.calls[:3] == [15.0, 15.0, 2.0]  # two waits, then the breaker cool-down


async def test_ready_timeout_raises(tmp_path):
    serve = FakeServe(ready_script=[(503, {"status": "not_ready"})] * 50)
    executor, _ = _executor(serve, ready_interval=15.0, ready_timeout=30.0)
    with pytest.raises(TimeoutError, match="never became ready"):
        await _run(executor, _items(tmp_path, "good"))


async def test_rejected_token_stops_the_run(tmp_path):
    serve = FakeServe(auth_token="right")  # noqa: S106
    executor, _ = _executor(serve, token="wrong", concurrency=1, max_concurrency=1)  # noqa: S106
    items = _items(tmp_path, "a", "b")

    reason, seen = await _run(executor, items)

    assert reason == "stopped"
    assert [(i.paper_id, o.error_code) for i, o in seen] == [("a", "http_401")]
    assert executor.fatal
    assert serve.requests.count(("POST", "/papers/jobs")) == 1


async def test_stop_drains_in_flight_with_grace_then_records_interrupted(tmp_path):
    serve = FakeServe(stuck={"slow"})
    executor, _ = _executor(serve, concurrency=2, grace=0.0)
    items = _items(tmp_path, "good", "slow", "later")
    stop = asyncio.Event()
    seen = []

    def on_outcome(item, outcome):
        seen.append((item.paper_id, outcome))
        stop.set()  # first Ctrl-C after the first paper lands

    reason = await executor.run(items, on_outcome=on_outcome, stop=stop)

    assert reason == "stopped"
    assert [pid for pid, _ in seen] == ["good", "slow"]
    assert seen[0][1].ok
    assert seen[1][1].error_code == INTERRUPTED
    assert [name for name, _ in serve.submits] == ["good.pdf", "slow.pdf"]  # "later" never sent


async def test_stop_set_before_the_run_submits_nothing(tmp_path):
    serve = FakeServe()
    executor, _ = _executor(serve)
    stop = asyncio.Event()
    stop.set()

    reason, seen = await _run(executor, _items(tmp_path, "a", "b"), stop=stop)

    assert reason == "stopped"
    assert seen == [] and serve.submits == []


async def test_deadline_stops_submission_but_lets_in_flight_finish(tmp_path):
    serve = FakeServe()
    clock = _Clock()
    executor = RemoteExecutor(
        RemoteOptions(serve_url="http://serve", token="t", poll_interval=1.0, concurrency=1),  # noqa: S106
        transport=httpx.MockTransport(serve.handler),
        sleep=_Sleeper(clock),
        clock=clock,
        wall=lambda: 100.0,
        rng=_NoJitter(),
    )
    items = _items(tmp_path, "a", "b")

    reason, seen = await _run(executor, items, deadline=50.0)

    assert reason == "deadline"
    assert seen == []  # wall clock already past the deadline: nothing submitted
    assert serve.submits == []


# --- runner integration -----------------------------------------------------------


def test_papers_failed_by_a_rejected_token_run_again_once_it_is_fixed(tmp_path, monkeypatch):
    """With two in flight, both papers get a 401 before the run stops; fixing
    the token and re-running the same command must pick them up."""
    monkeypatch.setattr("bibr.batch.runner.local_build_sha", lambda: None)
    papers = tmp_path / "papers"
    papers.mkdir()
    for stem in ("a", "b", "c"):
        (papers / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n" + stem.encode())
    out = tmp_path / "out"

    def options(token: str) -> BatchOptions:
        remote = RemoteOptions(
            serve_url="http://serve", token=token, poll_interval=0.001, concurrency=2
        )
        return BatchOptions(inputs=[str(papers)], out=out, remote=remote, tables=False)

    serve = FakeServe(auth_token="right")  # noqa: S106
    assert run_batch(options("wrong"), transport=httpx.MockTransport(serve.handler)) == 2
    rows = Ledger(out / LEDGER_FILENAME).read()
    assert sorted((r["paper_id"], r["error_code"]) for r in rows) == [
        ("a", "http_401"),
        ("b", "http_401"),
    ]

    assert run_batch(options("right"), transport=httpx.MockTransport(serve.handler)) == 0
    assert sorted(name for name, _ in serve.submits) == ["a.pdf", "b.pdf", "c.pdf"]
    latest = Ledger(out / LEDGER_FILENAME).latest()
    assert {pid: row["status"] for pid, row in latest.items()} == {
        "a": "ok",
        "b": "ok",
        "c": "ok",
    }


def test_run_batch_remote_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("bibr.batch.runner.local_build_sha", lambda: "unused-locally")
    serve = FakeServe(fail={"bad": ({"message": "no text", "error_code": "OCR_EMPTY"}, 422)})
    papers = tmp_path / "papers"
    papers.mkdir()
    for stem in ("good", "bad"):
        (papers / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n" + stem.encode())
    out = tmp_path / "out"
    options = BatchOptions(
        inputs=[str(papers)],
        out=out,
        remote=RemoteOptions(
            serve_url="http://serve",
            token="tok",  # noqa: S106
            poll_interval=0.001,
            concurrency=1,
            form={"refs": "ner"},
        ),
    )

    code = run_batch(options, transport=httpx.MockTransport(serve.handler))

    assert code == 1
    assert json.loads((out / "good.json").read_text())["info"]["title"] == "good"
    assert not (out / "bad.json").exists()
    rows = Ledger(out / LEDGER_FILENAME).read()
    by_id = {r["paper_id"]: r for r in rows}
    assert by_id["good"]["status"] == "ok"
    assert by_id["good"]["executor"] == "remote"
    assert by_id["good"]["build_sha"] == "serve-sha-1"
    assert by_id["bad"]["job_id"] == "job1"  # sorted input order, one in flight
    assert by_id["good"]["job_id"] == "job2"
    assert by_id["good"]["n_refs"] == 1
    assert by_id["good"]["warnings"]["codes"] == {"VALIDATION:warning:REF_YEAR": 1}
    assert by_id["bad"]["error_code"] == "OCR_EMPTY"
    info = json.loads((out / RUN_INFO_FILENAME).read_text())
    assert info["executor"] == "remote"
    assert info["build_sha"] == "serve-sha-1"
    assert info["serve"]["ready"]["build_sha"] == "serve-sha-1"
    assert info["serve"]["form"] == {"refs": "ner"}
    assert serve.submits[0][1] == {"refs": "ner"}


# --- small units ---------------------------------------------------------------------


def test_adaptive_gate_bounds():
    gate = AdaptiveGate(3, 1, 4)
    gate.grow()
    gate.grow()
    assert gate.size == 4
    gate.shrink()
    assert gate.size == 3
    gate.shrink_to_min()
    assert gate.size == 1
    gate.shrink()
    assert gate.size == 1
    assert AdaptiveGate(10, 2, 5).size == 5


def test_remote_options_clamp_concurrency():
    options = RemoteOptions(
        serve_url="http://x/", concurrency=9, min_concurrency=0, max_concurrency=3
    )
    assert (options.min_concurrency, options.concurrency, options.max_concurrency) == (1, 3, 3)
    assert options.serve_url == "http://x"


def test_looks_transient_markers():
    assert looks_transient("Circuit breaker 'ocr' is OPEN")
    assert looks_transient("Error in OCR: upstream unreachable")
    assert not looks_transient("reference parse produced no rows")
    # A timeout can be the paper's own slowness; never an outage by its text.
    assert not looks_transient("Post-parse failed: LLM call timed out after 600s")


def test_resolve_token_precedence(monkeypatch):
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    monkeypatch.delenv("BIBR_SERVE_TOKEN", raising=False)
    monkeypatch.setattr(
        "bibr.batch.remote.configured_value",
        lambda name: "from-dotenv" if name == "AUTH_API_KEY" else None,
    )
    assert resolve_token("explicit") == "explicit"
    assert resolve_token(None) == "from-dotenv"
    monkeypatch.setenv("BIBR_SERVE_TOKEN", "from-serve-env")
    assert resolve_token(None) == "from-serve-env"
    monkeypatch.setenv("AUTH_API_KEY", "from-auth-env")
    assert resolve_token(None) == "from-auth-env"


def test_configured_value_reads_env_and_honours_dotenv_kill_switch(monkeypatch):
    from bibr.batch.remote import configured_value

    monkeypatch.setenv("BIBR_BUILD_SHA", " abc123 ")
    assert configured_value("BIBR_BUILD_SHA") == "abc123"
    monkeypatch.setenv("BIBR_DISABLE_DOTENV", "1")
    assert configured_value("BIBR_BUILD_SHA") == "abc123"
    monkeypatch.delenv("BIBR_BUILD_SHA")
    assert configured_value("BIBR_BUILD_SHA") is None
    assert configured_value("NOT_A_BIBR_SETTING") is None
