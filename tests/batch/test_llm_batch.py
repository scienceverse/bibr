"""``bibr batch --llm-batch``: round 1 defers, the batch fills the cache, round 2 replays.

The pipeline is a stand-in that makes the real front-matter fan-out
(``LLMClient.extract_core_metadata``) through the real client, cache and
deferral hook; the live provider and the Message Batches API are fakes, so
the tests count exactly which calls went where.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import instructor
import pytest

from bibr.api import ChewFailure
from bibr.batch import llm_batch
from bibr.batch.ledger import LEDGER_FILENAME, LLM_DEFERRED, Ledger, reruns_by_default
from bibr.batch.llm_batch import FillResult, llm_batch_blocker, run_llm_batch
from bibr.batch.report import compute_report, render_report
from bibr.batch.runner import EXIT_USAGE, BatchOptions, LocalOptions, RemoteOptions
from bibr.clients.batch_defer import read_spool
from bibr.clients.llm import LLMClient, usage_file_context
from bibr.config import GlobalSettings
from bibr.exceptions import ProcessingError
from tests.clients.test_batch_request import _ANSWERS, _message


def _settings(tmp_path: Path, **llm) -> GlobalSettings:
    return GlobalSettings(
        llm={
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "api_key": "test",
            "track_usage": True,
            "batch_poll_interval_s": 0,
            **llm,
        },
        cache={
            "llm": True,
            "llm_dir": str(tmp_path / "llm-cache"),
            "ocr": True,
            "ocr_dir": str(tmp_path / "ocr-cache"),
        },
    )


@pytest.fixture
def corpus(tmp_path) -> Path:
    papers = tmp_path / "papers"
    papers.mkdir()
    for name in ("alpha", "beta", "gamma"):
        (papers / f"{name}.pdf").write_bytes(f"%PDF-1.4\n{name}".encode())
    return papers


@pytest.fixture(autouse=True)
def _stable_build_sha(monkeypatch):
    monkeypatch.setattr("bibr.batch.runner.local_build_sha", lambda: "deadbeef")


def _tool(body: str) -> str:
    return next(name for name in _ANSWERS if f'"name":"{name}"' in body.replace(" ", ""))


def _default_answer(tool: str, request_text: str) -> dict:
    return _ANSWERS[tool]


class Live:
    """The provider behind the live client: every request it sees is a paid call.

    *answers* maps a tool name and the request's text to the tool input.
    """

    def __init__(self, answers=_default_answer):
        self.calls: list[str] = []
        self.answers = answers

    def client(self, settings) -> LLMClient:
        def answer(request: httpx.Request) -> httpx.Response:
            body = request.read().decode()
            name = _tool(body)
            self.calls.append(name)
            return httpx.Response(200, json=_message(name, self.answers(name, body)))

        raw = anthropic.AsyncAnthropic(
            api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer))
        )
        client = LLMClient(settings=settings)
        live = instructor.from_anthropic(raw, model=settings.llm.model)
        client._get_client = lambda: live
        return client


class FakeChew:
    """``chew_many`` that runs only the front-matter fan-out, through the real client.

    A paper named in *fail* fails on its own (before any LLM call) with that code.
    """

    def __init__(self, settings, live: Live, fail: dict[str, str] | None = None):
        self.settings = settings
        self.live = live
        self.fail = fail or {}

    def text(self, name: str) -> str:
        """The front matter of the paper *name*."""
        return f"Front matter of {name}\nJane Doe\nAbstract."

    def export(self, meta) -> dict:
        """The export of a paper with core metadata *meta*."""
        return {"info": {"title": meta.title}, "bib": []}

    def __call__(self, paths, batch_size):
        client = self.live.client(self.settings)

        async def one(path: Path):
            if path.stem in self.fail:
                return ChewFailure(path, "failed on its own", error_code=self.fail[path.stem])
            data = path.read_bytes()
            file_hash = hashlib.sha256(data).hexdigest()[:16]
            text = self.text(data.decode().split()[-1])
            with usage_file_context(f"{file_hash}#1"):
                try:
                    meta = await client.extract_core_metadata(text, file_hash=file_hash)
                except ProcessingError as exc:
                    return ChewFailure(path, str(exc), error_code=exc.error_code)
            return SimpleNamespace(ok=True, data=self.export(meta))

        async def chunk():
            return [await one(path) for path in paths]

        return asyncio.run(chunk())

    def close(self):
        pass


class FakeBatches:
    """``client.messages.batches`` with server state shared across invocations."""

    def __init__(
        self,
        server: dict,
        *,
        fail_one: str | None = None,
        fail_all: str | None = None,
        interrupt_polls: int = 0,
        answers=_default_answer,
    ):
        self.server = server
        self.answers = answers
        self.fail_one = fail_one  # error the first request for this tool
        self.fail_all = fail_all  # error every request with this error type
        self.errored: set[str] = set()
        self.interrupt_polls = interrupt_polls
        self.created: list[list[dict]] = []
        self.log: list[str] = []

    def create(self, requests):
        self.log.append("create")
        self.created.append(list(requests))
        if self.fail_all:
            self.errored.update(r["custom_id"] for r in requests)
        if self.fail_one:
            self.errored.add(
                next(
                    r["custom_id"]
                    for r in requests
                    if r["params"]["tools"][0]["name"] == self.fail_one
                )
            )
        batch_id = f"msgbatch_{len(self.server) + 1}"
        self.server[batch_id] = list(requests)
        return SimpleNamespace(id=batch_id)

    def retrieve(self, batch_id):
        self.log.append("poll")
        if self.interrupt_polls:
            self.interrupt_polls -= 1
            raise KeyboardInterrupt
        n = len(self.server[batch_id])
        counts = SimpleNamespace(processing=0, succeeded=n, errored=0, canceled=0, expired=0)
        return SimpleNamespace(processing_status="ended", request_counts=counts)

    def results(self, batch_id):
        for request in self.server[batch_id]:
            custom_id = request["custom_id"]
            if custom_id in self.errored:
                kind = self.fail_all or "overloaded_error"
                error = SimpleNamespace(error=SimpleNamespace(type=kind, message="x"))
                result = SimpleNamespace(type="errored", error=error)
            else:
                name = request["params"]["tools"][0]["name"]
                answer = self.answers(name, json.dumps(request["params"]))
                message = anthropic.types.Message.model_validate(_message(name, answer))
                result = SimpleNamespace(type="succeeded", message=message)
            yield SimpleNamespace(custom_id=custom_id, result=result)


def _batch_client(batches: FakeBatches):
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def _options(corpus: Path, out: Path, **overrides) -> BatchOptions:
    base = {
        "inputs": [str(corpus)],
        "out": out,
        "local": LocalOptions(chew_options={"llm_backend": "cloud"}, batch_size=2),
        "tables": False,
    }
    base.update(overrides)
    return BatchOptions(**base)


def _install(
    monkeypatch, settings, live: Live, fail: dict[str, str] | None = None, *, chew=None
) -> None:
    chew = chew or FakeChew
    monkeypatch.setattr(
        "bibr.batch.runner.open_chew_many", lambda local: chew(settings, live, fail)
    )


def _latest(out: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for entry in Ledger(out / LEDGER_FILENAME).read():
        latest[entry["paper_id"]] = entry
    return latest


def test_front_matter_goes_through_one_batch_and_none_of_it_live(
    tmp_path, corpus, monkeypatch, capsys
):
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    batches = FakeBatches({})
    out = tmp_path / "out"

    code = run_llm_batch(
        _options(corpus, out), settings=settings, batch_client=_batch_client(batches)
    )

    assert code == 0
    assert live.calls == []  # 3 papers x 3 calls, all answered by the batch
    (submitted,) = batches.created
    assert len(submitted) == 9
    assert {r["params"]["model"] for r in submitted} == {"claude-haiku-4-5"}
    assert all(r["params"]["temperature"] == 0.0 for r in submitted)
    assert {e["status"] for e in _latest(out).values()} == {"ok"}
    attempts = [e["error_code"] for e in Ledger(out / LEDGER_FILENAME).read()]
    assert attempts.count(LLM_DEFERRED) == 3

    summary = json.loads((out / "llm_batch" / "summary.json").read_text())
    assert summary["deferred_papers"] == 3
    assert summary["fill"]["filled"] == 9
    assert summary["round2"]["hit_rate"] == 1.0
    assert summary["round2"]["hits"] == 9


def test_identical_requests_are_sent_once(tmp_path, corpus, monkeypatch):
    (corpus / "alpha-copy.pdf").write_bytes((corpus / "alpha.pdf").read_bytes())
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    batches = FakeBatches({})
    out = tmp_path / "out"

    assert (
        run_llm_batch(_options(corpus, out), settings=settings, batch_client=_batch_client(batches))
        == 0
    )

    (submitted,) = batches.created
    assert len(submitted) == 9  # alpha-copy's three requests are alpha's
    assert len(read_spool(out / "llm_batch" / "spool.jsonl", unique=False)) == 12
    assert live.calls == []


def test_an_unanswered_row_goes_live_in_the_replay(tmp_path, corpus, monkeypatch):
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    batches = FakeBatches({}, fail_one="AuthorsLLM")
    out = tmp_path / "out"

    code = run_llm_batch(
        _options(corpus, out), settings=settings, batch_client=_batch_client(batches)
    )

    assert code == 0
    assert live.calls == ["AuthorsLLM"]
    summary = json.loads((out / "llm_batch" / "summary.json").read_text())
    assert summary["fill"]["errors"] == {"overloaded_error": 1}
    assert summary["round2"]["unanswered"] == 1
    assert summary["round2"]["hit_rate"] == 1.0
    assert {e["status"] for e in _latest(out).values()} == {"ok"}


def test_a_replay_that_misses_stops_loudly(tmp_path, corpus, monkeypatch, capsys):
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    # The batch "answers" but nothing reaches the cache: every replay call misses.
    monkeypatch.setattr(llm_batch, "fill_cache", lambda *a, **k: FillResult())
    monkeypatch.setattr(llm_batch, "EARLY_STOP_LOOKUPS", 2)
    out = tmp_path / "out"

    code = run_llm_batch(
        _options(
            corpus, out, local=LocalOptions(chew_options={"llm_backend": "cloud"}, batch_size=1)
        ),
        settings=settings,
        batch_client=_batch_client(FakeBatches({})),
    )

    assert code == 1
    assert "of the batch answers" in capsys.readouterr().err
    # One paper went live before the replay stopped; the others still wait.
    assert len(live.calls) == 3
    codes = sorted(str(e.get("error_code")) for e in _latest(out).values())
    assert codes == ["None", LLM_DEFERRED, LLM_DEFERRED]
    summary = json.loads((out / "llm_batch" / "summary.json").read_text())
    assert summary["round2"]["stopped_early"] is True


def test_an_interrupted_wait_resumes_the_same_batch(tmp_path, corpus, monkeypatch):
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    server: dict = {}
    out = tmp_path / "out"

    first = FakeBatches(server, interrupt_polls=1)
    assert (
        run_llm_batch(_options(corpus, out), settings=settings, batch_client=_batch_client(first))
        == 130
    )
    assert len(first.created) == 1
    assert len(list((out / "llm_batch").glob("part-0001-*.state.json"))) == 1

    second = FakeBatches(server)
    code = run_llm_batch(
        _options(corpus, out), settings=settings, batch_client=_batch_client(second)
    )

    assert code == 0
    assert second.created == []  # collected, not paid for twice
    assert live.calls == []
    assert {e["status"] for e in _latest(out).values()} == {"ok"}


def test_cached_requests_are_not_batched_again(tmp_path, corpus, monkeypatch):
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    batches = FakeBatches({})
    out = tmp_path / "out"
    assert (
        run_llm_batch(_options(corpus, out), settings=settings, batch_client=_batch_client(batches))
        == 0
    )

    again = FakeBatches({})
    code = run_llm_batch(
        _options(corpus, tmp_path / "out2"), settings=settings, batch_client=_batch_client(again)
    )

    assert code == 0
    assert again.created == []
    assert live.calls == []


@pytest.mark.parametrize(
    "change, message",
    [
        ({"llm": {"provider": "openai"}}, "LLM_PROVIDER=anthropic"),
        ({"cache": {"llm": False}}, "CACHE_LLM=true"),
        ({"cache": {"ocr": False}}, "CACHE_OCR=true"),
        ({"llm": {"track_usage": False}}, "LLM_TRACK_USAGE=true"),
        ({"llm": {"merged_core_metadata": True}}, "LLM_MERGED_CORE_METADATA"),
        ({"llm": {"structured_backend": "nuextract-native"}}, "Instructor backend"),
        ({"llm": {"api_key": ""}, "ANTHROPIC_API_KEY": ""}, "API key"),
    ],
)
def test_refused_settings(tmp_path, corpus, change, message):
    settings = _settings(tmp_path)
    update = {}
    for section, values in change.items():
        if isinstance(values, dict):
            update[section] = getattr(settings, section).model_copy(update=values)
        else:
            update[section] = values
    settings = settings.model_copy(update=update)

    assert message in llm_batch_blocker(_options(corpus, tmp_path / "out"), settings)


def test_json_prints_one_report(tmp_path, corpus, monkeypatch, capsys):
    settings = _settings(tmp_path)
    _install(monkeypatch, settings, Live())
    out = tmp_path / "out"

    code = run_llm_batch(
        _options(corpus, out, report_json=True),
        settings=settings,
        batch_client=_batch_client(FakeBatches({})),
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)  # one document: round 2's
    assert report["papers"]["ok"] == 3


def test_refused_executors(tmp_path, corpus, capsys):
    settings = _settings(tmp_path)
    remote = _options(
        corpus, tmp_path / "out", local=None, remote=RemoteOptions(serve_url="http://127.0.0.1:1")
    )
    no_llm = _options(corpus, tmp_path / "out", local=LocalOptions(chew_options={"no_llm": True}))
    local_llm = _options(
        corpus, tmp_path / "out", local=LocalOptions(chew_options={"llm_backend": "vllm"})
    )

    assert "--serve-url" in llm_batch_blocker(remote, settings)
    assert "--no-llm" in llm_batch_blocker(no_llm, settings)
    assert "cloud" in llm_batch_blocker(local_llm, settings)
    assert run_llm_batch(remote, settings=settings) == EXIT_USAGE
    assert not (tmp_path / "out").exists()


def test_a_deferred_paper_runs_again_and_is_counted():
    entry = {"paper_id": "p", "status": "failed", "error_code": LLM_DEFERRED}
    assert reruns_by_default(entry)

    report = compute_report([entry, {"paper_id": "q", "status": "ok"}])
    assert report["papers"]["deferred"] == 1
    assert "1 deferred to an LLM batch" in render_report(report)


def test_cli_flag_refuses_before_any_work(tmp_path, monkeypatch, capsys):
    from bibr.local.cli import _build_parser
    from bibr.local.cli.batch import _run_batch

    def boom(local):
        raise AssertionError("a refused --llm-batch must not open a pipeline")

    monkeypatch.setattr("bibr.batch.runner.open_chew_many", boom)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "out"
    parser = _build_parser()

    remote = parser.parse_args(
        ["batch", str(pdf), "--out", str(out), "--llm-batch", "front-matter"]
        + ["--serve-url", "http://127.0.0.1:1"]
    )
    assert _run_batch(remote) == 2
    assert "--llm-batch runs locally" in capsys.readouterr().err

    no_llm = parser.parse_args(
        ["batch", str(pdf), "--out", str(out), "--llm-batch", "front-matter", "--no-llm"]
    )
    assert _run_batch(no_llm) == 2
    assert "--llm-batch" in capsys.readouterr().err
    assert not out.exists()

    with pytest.raises(SystemExit):
        parser.parse_args(["batch", str(pdf), "--out", str(out), "--llm-batch", "references"])


def test_a_round1_failure_fails_the_run_after_a_clean_replay(tmp_path, corpus, monkeypatch):
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live, fail={"gamma": "ocr_failed"})
    out = tmp_path / "out"

    code = run_llm_batch(
        _options(corpus, out), settings=settings, batch_client=_batch_client(FakeBatches({}))
    )

    latest = _latest(out)
    assert latest["alpha"]["status"] == latest["beta"]["status"] == "ok"
    assert latest["gamma"]["error_code"] == "ocr_failed"  # not deferred, so not replayed
    assert code == 1  # the replay's own exit code (0) must not hide gamma


def test_json_report_when_nothing_is_deferred(tmp_path, corpus, monkeypatch, capsys):
    settings = _settings(tmp_path)
    _install(monkeypatch, settings, Live())
    batches = _batch_client(FakeBatches({}))
    warm = run_llm_batch(
        _options(corpus, tmp_path / "warm"), settings=settings, batch_client=batches
    )
    assert warm == 0
    capsys.readouterr()

    code = run_llm_batch(
        _options(corpus, tmp_path / "out", report_json=True),
        settings=settings,
        batch_client=batches,
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)  # round 1 was the whole run
    assert report["papers"]["ok"] == report["papers"]["total"] == 3
    assert len(report["run_ids"]) == 1
    assert report["llm_batch"]["deferred_papers"] == 0


def test_json_report_covers_both_rounds(tmp_path, corpus, monkeypatch, capsys):
    settings = _settings(tmp_path)
    _install(monkeypatch, settings, Live())
    alpha_only = tmp_path / "alpha-only"
    alpha_only.mkdir()
    shutil.copy(corpus / "alpha.pdf", alpha_only / "alpha.pdf")
    batches = _batch_client(FakeBatches({}))
    assert (
        run_llm_batch(
            _options(alpha_only, tmp_path / "warm"), settings=settings, batch_client=batches
        )
        == 0
    )
    capsys.readouterr()

    code = run_llm_batch(
        _options(corpus, tmp_path / "out", report_json=True),
        settings=settings,
        batch_client=batches,
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    # alpha finished in round 1 (its answers were cached), beta and gamma in round 2.
    assert report["papers"] == {"total": 3, "ok": 3, "failed": 0, "deferred": 0, "attempts": 5}
    assert len(report["run_ids"]) == 2
    assert report["llm_batch"]["round2"]["hit_rate"] == 1.0


def test_an_answer_gone_from_the_cache_is_batched_again(tmp_path, corpus, monkeypatch):
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    server: dict = {}
    out = tmp_path / "out"
    assert (
        run_llm_batch(
            _options(corpus, out),
            settings=settings,
            batch_client=_batch_client(FakeBatches(server)),
        )
        == 0
    )
    shutil.rmtree(tmp_path / "llm-cache")
    (corpus / "alpha-copy.pdf").write_bytes((corpus / "alpha.pdf").read_bytes())

    again = FakeBatches(server)
    code = run_llm_batch(
        _options(corpus, out), settings=settings, batch_client=_batch_client(again)
    )

    assert code == 0
    (submitted,) = again.created  # alpha's three requests, answered once but no longer cached
    assert len(submitted) == 3
    assert live.calls == []
    summary = json.loads((out / "llm_batch" / "summary.json").read_text())
    assert summary["fill"]["batched_again"] == 3
    assert summary["round2"]["hit_rate"] == 1.0


def test_the_deadline_bounds_the_batch_wait(tmp_path, corpus, monkeypatch, capsys):
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    seen = {}

    def fill(work, settings, **kwargs):
        seen["timeout_s"] = kwargs["timeout_s"]
        raise TimeoutError("batch msgbatch_1 did not end in time")

    monkeypatch.setattr(llm_batch, "fill_cache", fill)
    out = tmp_path / "out"

    code = run_llm_batch(
        _options(corpus, out, deadline=time.time() + 3600),
        settings=settings,
        batch_client=_batch_client(FakeBatches({})),
    )

    assert 3000 < seen["timeout_s"] <= 3600  # not the 25 h batch wait
    assert code == 0  # a resumable stop, as a deadline is for `bibr batch`
    assert "deadline reached" in capsys.readouterr().err
    assert {e["error_code"] for e in _latest(out).values()} == {LLM_DEFERRED}
    assert live.calls == []


class EndlessBatches(FakeBatches):
    """A batch that is still processing whenever it is polled."""

    def retrieve(self, batch_id):
        self.log.append("poll")
        n = len(self.server[batch_id])
        counts = SimpleNamespace(processing=n, succeeded=0, errored=0, canceled=0, expired=0)
        return SimpleNamespace(processing_status="in_progress", request_counts=counts)


def test_a_timed_out_fill_collects_the_same_batch_later(tmp_path, corpus, monkeypatch):
    settings = _settings(tmp_path, batch_poll_interval_s=60)
    _install(monkeypatch, settings, Live())
    out = tmp_path / "out"
    from bibr.clients.batch_defer import collecting

    with collecting(out / "llm_batch" / "spool.jsonl") as collect:
        from bibr.batch.runner import run_batch

        run_batch(_options(corpus, out))
    assert collect.spooled == 9
    server: dict = {}

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        llm_batch.fill_cache(
            out / "llm_batch",
            settings,
            batch_client=_batch_client(EndlessBatches(server)),
            timeout_s=0.05,
        )
    assert time.monotonic() - started < 30  # the 60 s poll interval is cut to the time box

    later = FakeBatches(server)
    fill = llm_batch.fill_cache(out / "llm_batch", settings, batch_client=_batch_client(later))
    assert later.created == []
    assert fill.filled == 9


def test_every_batch_is_submitted_before_the_wait(tmp_path, corpus, monkeypatch):
    import bibr.clients.batch as batch

    monkeypatch.setattr(batch, "MAX_BATCH_REQUESTS", 4)
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    batches = FakeBatches({})

    code = run_llm_batch(
        _options(corpus, tmp_path / "out"), settings=settings, batch_client=_batch_client(batches)
    )

    assert code == 0
    assert [len(c) for c in batches.created] == [4, 4, 1]
    assert batches.log[:3] == ["create"] * 3  # side by side at Anthropic, not one after another
    assert live.calls == []


def test_a_batch_that_answers_too_little_stops_before_the_replay(
    tmp_path, corpus, monkeypatch, capsys
):
    monkeypatch.setattr(llm_batch, "MIN_UNANSWERED_TO_STOP", 2)
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    server: dict = {}
    out = tmp_path / "out"

    code = run_llm_batch(
        _options(corpus, out),
        settings=settings,
        batch_client=_batch_client(FakeBatches(server, fail_all="invalid_request_error")),
    )

    assert code == 1
    assert live.calls == []  # nothing paid at the full price
    assert {e["error_code"] for e in _latest(out).values()} == {LLM_DEFERRED}
    assert "unanswered" in capsys.readouterr().err
    summary = json.loads((out / "llm_batch" / "summary.json").read_text())
    assert summary["stopped"] == "unanswered"
    assert summary["fill"]["errors"] == {"invalid_request_error": 9}

    # The way out the error names: start the plan over and batch them again.
    (out / "llm_batch" / "plan.json").unlink()
    again = FakeBatches(server)
    code = run_llm_batch(
        _options(corpus, out), settings=settings, batch_client=_batch_client(again)
    )
    assert code == 0
    assert [len(c) for c in again.created] == [9]
    assert live.calls == []


def test_refuses_an_instructor_without_the_v2_request_builder(tmp_path, corpus, monkeypatch):
    import bibr.clients.batch as batch

    def missing():
        raise ModuleNotFoundError("No module named 'instructor.v2'")

    monkeypatch.setattr(batch, "_instructor_tools", missing)

    problem = llm_batch_blocker(_options(corpus, tmp_path / "out"), _settings(tmp_path))
    assert "upgrade instructor" in problem


def test_the_replay_runs_only_what_this_round_deferred(tmp_path, corpus, monkeypatch):
    settings = _settings(tmp_path)
    live = Live()
    _install(monkeypatch, settings, live)
    server: dict = {}
    out = tmp_path / "out"
    first = FakeBatches(server, interrupt_polls=1)
    assert (
        run_llm_batch(_options(corpus, out), settings=settings, batch_client=_batch_client(first))
        == 130
    )

    # --limit 1: round 1 reaches one of the three waiting papers.
    second = FakeBatches(server)
    code = run_llm_batch(
        _options(corpus, out, limit=1), settings=settings, batch_client=_batch_client(second)
    )

    assert code == 0  # the two papers not reached still wait; that is no failure
    assert second.created == []  # the first run's batch is collected, all 9 answers
    latest = _latest(out)
    assert sorted(e["status"] for e in latest.values()) == ["failed", "failed", "ok"]
    summary = json.loads((out / "llm_batch" / "summary.json").read_text())
    assert summary["deferred_papers"] == 1
    assert summary["fill"]["filled"] == 9

    third = FakeBatches(server)
    assert (
        run_llm_batch(_options(corpus, out), settings=settings, batch_client=_batch_client(third))
        == 0
    )
    assert third.created == []
    assert {e["status"] for e in _latest(out).values()} == {"ok"}
    assert live.calls == []


# ---------------------------------------------------------------------------
# LLM_POINTER_OUTPUTS: the batch answers the pointer calls, and round 2 copies
# the abstract and affiliations back exactly as a live answer is copied back.
# ---------------------------------------------------------------------------

_PAPERS = ("alpha", "beta", "gamma")


def _abstract(paper: str) -> str:
    return f"We studied {paper} in depth. It went well."


class PaperChew(FakeChew):
    """Front matter with an abstract and an affiliation, and an export of every
    core field, including what tells an omitted abstract from a missing one."""

    def text(self, name: str) -> str:
        return (
            f"A Study of {name}\nJane Doe\nUniversity of Somewhere\n\n"
            f"Abstract. {_abstract(name)}\n\nKeywords: {name}; things"
        )

    def export(self, meta) -> dict:
        return {
            "info": meta.model_dump(mode="json"),
            "abstract_given": "abstract" in meta.model_fields_set,
            "abstract_explicitly_absent": meta._abstract_explicitly_absent,
            "field_failures": meta._field_failures,
            "bib": [],
        }


def _paper_answer(tool: str, request_text: str) -> dict:
    """The same model answer for a request, live or batched.

    alpha's pointer answer is copied back; beta's omits the abstract; gamma's
    names words its text does not hold, so bibr asks for the verbatim answer.
    """
    paper = next(p for p in _PAPERS if f"A Study of {p}" in request_text)
    keywords = [paper, "things"]
    if tool == "TitleKeywordsPointerLLM":
        anchors = {
            "alpha": {
                "abstract_first_words": f"We studied {paper} in depth.",
                "abstract_last_words": "It went well.",
            },
            "beta": {},
            "gamma": {
                "abstract_first_words": "Nothing like this",
                "abstract_last_words": "appears in the text.",
            },
        }[paper]
        return {"title": f"A Study of {paper}", "keywords": keywords, **anchors}
    if tool == "TitleKeywordsLLM":
        abstract = {} if paper == "beta" else {"abstract": _abstract(paper)}
        return {"title": f"A Study of {paper}", "keywords": keywords, **abstract}
    if tool == "AuthorsPointerLLM":
        return {
            "affiliations": ["University of Somewhere"],
            "authors": [{"given": "Jane", "family": "Doe", "affiliation_ids": [1]}],
        }
    if tool == "AuthorsLLM":
        return {
            "authors": [
                {"given": "Jane", "family": "Doe", "affiliation": "University of Somewhere"}
            ]
        }
    return _ANSWERS[tool]


@pytest.mark.parametrize("pointers", [False, True], ids=["verbatim", "pointers"])
def test_the_replay_exports_what_a_live_run_exports(tmp_path, corpus, monkeypatch, pointers):
    """Byte for byte, with LLM_POINTER_OUTPUTS off and on: a live run, a run
    from the cache that live run filled, and an --llm-batch run whose cache
    the batch filled."""
    from bibr.batch.runner import run_batch
    from bibr.clients.llm_cache import LlmResponseCache

    # The live runs make 9 or 10 calls; LLM_RATE_LIMIT_RPM's default would pace them.
    llm = {"pointer_outputs": pointers, "rate_limit_rpm": 60_000}
    live_settings = _settings(tmp_path / "live", **llm)
    live = Live(_paper_answer)
    _install(monkeypatch, live_settings, live, chew=PaperChew)
    assert run_batch(_options(corpus, tmp_path / "out-live")) == 0
    # gamma's pointer answer cannot be placed: one verbatim re-ask.
    assert len(live.calls) == (10 if pointers else 9)

    cached = Live(_paper_answer)
    _install(monkeypatch, live_settings, cached, chew=PaperChew)
    assert run_batch(_options(corpus, tmp_path / "out-cached")) == 0
    assert cached.calls == []

    settings = _settings(tmp_path / "batch", **llm)
    replay = Live(_paper_answer)
    _install(monkeypatch, settings, replay, chew=PaperChew)
    batches = FakeBatches({}, answers=_paper_answer)
    out = tmp_path / "out-batch"
    assert (
        run_llm_batch(_options(corpus, out), settings=settings, batch_client=_batch_client(batches))
        == 0
    )

    (submitted,) = batches.created
    title, authors = (
        ("TitleKeywordsPointerLLM", "AuthorsPointerLLM")
        if pointers
        else ("TitleKeywordsLLM", "AuthorsLLM")
    )
    assert sorted(r["params"]["tools"][0]["name"] for r in submitted) == sorted(
        [title, authors, "PaperClassificationLLM"] * 3
    )
    # The verbatim re-ask was never batched: it goes live, as a fallback that
    # does not count against the replay's hit rate.
    assert replay.calls == (["TitleKeywordsLLM"] if pointers else [])
    round2 = json.loads((out / "llm_batch" / "summary.json").read_text())["round2"]
    assert (round2["hits"], round2["misses"], round2["fallbacks"]) == (9, 0, int(pointers))
    assert round2["hit_rate"] == 1.0

    for paper in _PAPERS:
        reference = (tmp_path / "out-live" / f"{paper}.json").read_bytes()
        assert (tmp_path / "out-cached" / f"{paper}.json").read_bytes() == reference
        assert (out / f"{paper}.json").read_bytes() == reference
    # The batch stored each answer as the live call stored the same answer.
    live_cache, batch_cache = (
        LlmResponseCache(settings=live_settings),
        LlmResponseCache(settings=settings),
    )
    for row in read_spool(out / "llm_batch" / "spool.jsonl"):
        assert batch_cache.get(row["key"]) == live_cache.get(row["key"]) is not None

    exports = {p: json.loads((out / f"{p}.json").read_text()) for p in _PAPERS}
    assert exports["alpha"]["info"]["abstract"] == _abstract("alpha")
    assert exports["alpha"]["info"]["authors"][0]["affiliation"] == "University of Somewhere"
    assert exports["beta"]["abstract_given"] is False
    assert exports["beta"]["abstract_explicitly_absent"] is False
    assert exports["gamma"]["info"]["abstract"] == _abstract("gamma")
