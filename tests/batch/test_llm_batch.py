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


class Live:
    """The provider behind the live client: every request it sees is a paid call."""

    def __init__(self):
        self.calls: list[str] = []

    def client(self, settings) -> LLMClient:
        def answer(request: httpx.Request) -> httpx.Response:
            name = _tool(request.read().decode())
            self.calls.append(name)
            return httpx.Response(200, json=_message(name, _ANSWERS[name]))

        raw = anthropic.AsyncAnthropic(
            api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer))
        )
        client = LLMClient(settings=settings)
        live = instructor.from_anthropic(raw, model=settings.llm.model)
        client._get_client = lambda: live
        return client


class FakeChew:
    """``chew_many`` that runs only the front-matter fan-out, through the real client."""

    def __init__(self, settings, live: Live):
        self.settings = settings
        self.live = live

    def __call__(self, paths, batch_size):
        client = self.live.client(self.settings)

        async def one(path: Path):
            data = path.read_bytes()
            file_hash = hashlib.sha256(data).hexdigest()[:16]
            text = f"Front matter of {data.decode().split()[-1]}\nJane Doe\nAbstract."
            with usage_file_context(f"{file_hash}#1"):
                try:
                    meta = await client.extract_core_metadata(text, file_hash=file_hash)
                except ProcessingError as exc:
                    return ChewFailure(path, str(exc), error_code=exc.error_code)
            return SimpleNamespace(ok=True, data={"info": {"title": meta.title}, "bib": []})

        async def chunk():
            return [await one(path) for path in paths]

        return asyncio.run(chunk())

    def close(self):
        pass


class FakeBatches:
    """``client.messages.batches`` with server state shared across invocations."""

    def __init__(self, server: dict, *, fail_one: str | None = None, interrupt_polls: int = 0):
        self.server = server
        self.fail_one = fail_one  # error the first request for this tool
        self.errored: set[str] = set()
        self.interrupt_polls = interrupt_polls
        self.created: list[list[dict]] = []

    def create(self, requests):
        self.created.append(list(requests))
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
                error = SimpleNamespace(error=SimpleNamespace(type="overloaded_error", message="x"))
                result = SimpleNamespace(type="errored", error=error)
            else:
                name = request["params"]["tools"][0]["name"]
                message = anthropic.types.Message.model_validate(_message(name, _ANSWERS[name]))
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


def _install(monkeypatch, settings, live: Live) -> None:
    monkeypatch.setattr("bibr.batch.runner.open_chew_many", lambda local: FakeChew(settings, live))


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
    assert (out / "llm_batch" / "part-0001.state.json").is_file()

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
