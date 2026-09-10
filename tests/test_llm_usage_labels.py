"""Per-label LLM usage attribution (tests/test_llm_usage_labels.py)."""

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from bibr.clients.llm import LLMClient, _usage_label, usage_file_context
from bibr.config import GlobalSettings
from bibr.exceptions import UpstreamServiceError
from bibr.export.json_export import export_paper_to_json
from bibr.schemas import TitleKeywordsLLM
from tests.test_llm_usage_export import _minimal_paper


def _fake_completion(inp=100, out=10):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out)
    )


class _Transient(Exception):
    status_code = 503


class _Fatal(Exception):
    status_code = 400


class _TelemetryBackend:
    def __init__(self, effects):
        self.effects = list(effects)

    async def create(self, **kwargs):
        effect = self.effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        completion = _fake_completion(100, 10)
        return effect, completion


class _BarrierBackend:
    def __init__(self):
        self.labels = []
        self.release = asyncio.Event()

    async def create(self, **kwargs):  # noqa: ARG002
        label = _usage_label.get()
        self.labels.append(label)
        if len(self.labels) == 2:
            self.release.set()
        await self.release.wait()
        output_tokens = 10 if label == "first" else 20
        result = TitleKeywordsLLM(title=label or "", keywords=[])
        return result, _fake_completion(100, output_tokens)


class _NotifyingSemaphore(asyncio.Semaphore):
    def __init__(self, value):
        super().__init__(value)
        self.acquire_started = asyncio.Event()

    async def acquire(self):
        self.acquire_started.set()
        return await super().acquire()


def _tracked_client(effects):
    settings = GlobalSettings(llm={"track_usage": True})
    client = LLMClient(settings=settings, backend=_TelemetryBackend(effects))
    client._limiter = mock.MagicMock()
    client._limiter.acquire = mock.AsyncMock(return_value=None)
    return client


def _lkey(client, label):
    """The (label, provider, model) triple a client's label buckets are keyed by."""
    return (label, client._settings.llm.provider, client._settings.llm.model)


def test_record_usage_attributes_label_per_file():
    client = LLMClient.__new__(LLMClient)  # skip provider init
    client._usage = {}
    client._usage_by_file = {}
    client._labels_by_file = {}
    client._track_usage = True
    client._settings = SimpleNamespace(llm=SimpleNamespace(provider="p", model="m"))

    with usage_file_context("hashA#1"):
        tok = _usage_label.set("extract_authors")
        try:
            client._record_usage(_fake_completion(100, 10))
            client._record_usage(_fake_completion(50, 5))
        finally:
            _usage_label.reset(tok)
        tok = _usage_label.set("extract_title_keywords")
        try:
            client._record_usage(_fake_completion(200, 20))
        finally:
            _usage_label.reset(tok)

    labels = client.usage_labels_pop_file("hashA#1")
    assert labels[("extract_authors", "p", "m")] == {
        "input_tokens": 150,
        "output_tokens": 15,
        "total_tokens": 165,
        "cached_input_tokens": 0,
        "calls": 2,
    }
    assert labels[("extract_title_keywords", "p", "m")]["calls"] == 1
    assert client.usage_labels_pop_file("hashA#1") == {}  # popped


def test_record_usage_without_label_uses_unlabeled_bucket():
    client = LLMClient.__new__(LLMClient)
    client._usage = {}
    client._usage_by_file = {}
    client._labels_by_file = {}
    client._track_usage = True
    client._settings = SimpleNamespace(llm=SimpleNamespace(provider="p", model="m"))
    with usage_file_context("hashB#1"):
        client._record_usage(_fake_completion())
    assert client.usage_labels_pop_file("hashB#1")[("unlabeled", "p", "m")]["calls"] == 1


async def test_label_bucket_records_success_waits_and_provider_time():
    result = TitleKeywordsLLM(title="T", keywords=[])
    client = _tracked_client([result])

    with usage_file_context("timed#1"):
        await client.extract_title_keywords("text")

    metrics = client.usage_labels_pop_file("timed#1")[_lkey(client, "extract_title_keywords")]
    assert metrics["calls"] == 1
    assert metrics["logical_calls"] == 1
    assert metrics["attempts"] == 1
    assert metrics["retries"] == 0
    assert metrics["failed_calls"] == 0
    for key in ("total_ms", "rate_limit_wait_ms", "concurrency_wait_ms", "provider_ms"):
        assert isinstance(metrics[key], int)
        assert metrics[key] >= 0


async def test_label_bucket_records_outer_retry(monkeypatch):
    result = TitleKeywordsLLM(title="T", keywords=[])
    client = _tracked_client([_Transient("busy"), result])
    monkeypatch.setattr(client, "_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr("bibr.clients.llm.asyncio.sleep", mock.AsyncMock())

    with usage_file_context("retry#1"):
        await client.extract_title_keywords("text")

    metrics = client.usage_labels_pop_file("retry#1")[_lkey(client, "extract_title_keywords")]
    assert metrics["logical_calls"] == 1
    assert metrics["attempts"] == 2
    assert metrics["retries"] == 1
    assert metrics["failed_calls"] == 0


async def test_label_bucket_records_terminal_failure_without_completion():
    client = _tracked_client([_Fatal("bad request")])

    with usage_file_context("failed#1"):
        with pytest.raises(UpstreamServiceError):
            await client.extract_title_keywords("text")

    metrics = client.usage_labels_pop_file("failed#1")[_lkey(client, "extract_title_keywords")]
    assert metrics["calls"] == 0
    assert metrics["logical_calls"] == 1
    assert metrics["attempts"] == 1
    assert metrics["failed_calls"] == 1


async def test_invoke_structured_acquires_limiter_under_explicit_label():
    result = TitleKeywordsLLM(title="T", keywords=[])
    client = _tracked_client([result])

    with usage_file_context("external#1"):
        await client.invoke_structured(
            TitleKeywordsLLM,
            [{"role": "user", "content": "x"}],
            "system",
            label="section_classifier",
        )

    client._limiter.acquire.assert_awaited_once()
    metrics = client.usage_labels_pop_file("external#1")[_lkey(client, "section_classifier")]
    assert metrics["logical_calls"] == 1
    assert metrics["rate_limit_wait_ms"] >= 0


async def test_cancelled_semaphore_wait_records_zero_attempt_and_restores_label():
    result = TitleKeywordsLLM(title="T", keywords=[])
    client = _tracked_client([result])
    gate = _NotifyingSemaphore(1)
    await gate.acquire()
    gate.acquire_started.clear()
    client._concurrency_sem = gate
    client._settings.llm.max_concurrency = 1
    restored_labels = []

    async def invoke_and_observe_cleanup():
        try:
            await client.extract_title_keywords("text")
        except asyncio.CancelledError:
            restored_labels.append(_usage_label.get())
            raise

    outer_token = _usage_label.set("outer")
    try:
        with usage_file_context("cancelled-wait#1"):
            task = asyncio.create_task(invoke_and_observe_cleanup())
            await gate.acquire_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        gate.release()
        _usage_label.reset(outer_token)

    metrics = client.usage_labels_pop_file("cancelled-wait#1")[
        _lkey(client, "extract_title_keywords")
    ]
    assert metrics["logical_calls"] == 1
    assert metrics["attempts"] == 0
    assert metrics["native_attempts"] == 0
    assert metrics["instructor_attempts"] == 0
    assert metrics["retries"] == 0
    assert metrics["failed_calls"] == 1
    assert isinstance(metrics["concurrency_wait_ms"], int)
    assert metrics["concurrency_wait_ms"] >= 0
    assert isinstance(metrics["total_ms"], int)
    assert metrics["total_ms"] >= 0
    assert restored_labels == ["outer"]


async def test_cancelled_retry_backoff_does_not_count_unstarted_retry(monkeypatch):
    client = _tracked_client([_Transient("busy")])
    backoff_started = asyncio.Event()

    async def held_backoff(delay):  # noqa: ARG001
        backoff_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("bibr.clients.llm.asyncio.sleep", held_backoff)

    with usage_file_context("cancelled-backoff#1"):
        task = asyncio.create_task(client.extract_title_keywords("text"))
        await backoff_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    metrics = client.usage_labels_pop_file("cancelled-backoff#1")[
        _lkey(client, "extract_title_keywords")
    ]
    assert metrics["logical_calls"] == 1
    assert metrics["attempts"] == 1
    assert metrics["native_attempts"] + metrics["instructor_attempts"] == 1
    assert metrics["retries"] == 0
    assert metrics["failed_calls"] == 1


async def test_concurrent_explicit_labels_are_isolated_within_file_context():
    settings = GlobalSettings(llm={"track_usage": True, "max_concurrency": 0})
    backend = _BarrierBackend()
    client = LLMClient(settings=settings, backend=backend)
    client._limiter = mock.MagicMock()
    client._limiter.acquire = mock.AsyncMock(return_value=None)

    with usage_file_context("concurrent#1"):
        first, second = await asyncio.gather(
            client.invoke_structured(
                TitleKeywordsLLM,
                [{"role": "user", "content": "one"}],
                "system",
                label="first",
            ),
            client.invoke_structured(
                TitleKeywordsLLM,
                [{"role": "user", "content": "two"}],
                "system",
                label="second",
            ),
        )

    assert {first.title, second.title} == {"first", "second"}
    assert set(backend.labels) == {"first", "second"}
    metrics = client.usage_labels_pop_file("concurrent#1")
    first_metrics = metrics[_lkey(client, "first")]
    second_metrics = metrics[_lkey(client, "second")]
    assert first_metrics["calls"] == 1
    assert first_metrics["output_tokens"] == 10
    assert first_metrics["logical_calls"] == 1
    assert first_metrics["attempts"] == 1
    assert second_metrics["calls"] == 1
    assert second_metrics["output_tokens"] == 20
    assert second_metrics["logical_calls"] == 1
    assert second_metrics["attempts"] == 1


async def test_concurrent_file_protocol_counters_do_not_leak():
    client = _tracked_client([])
    both_started = asyncio.Event()
    started = 0

    async def run_one(key, label, category):
        nonlocal started

        async def record():
            nonlocal started
            client._record_label_metric("attempts", 2)
            client._record_label_metric("native_attempts", 1)
            client._record_label_metric("instructor_attempts", 1)
            client._record_label_metric("protocol_fallbacks", 1)
            client._record_label_metric("native_invalid_outputs", 1)
            client._record_label_metric(f"native_invalid_{category}", 1)
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()

        with usage_file_context(key):
            await client._run_labeled_call(label, record)

    await asyncio.gather(
        run_one("file-a#1", "extract_title_keywords", "non_json"),
        run_one("file-b#1", "extract_authors", "truncated"),
    )

    first = client.usage_labels_pop_file("file-a#1")[_lkey(client, "extract_title_keywords")]
    second = client.usage_labels_pop_file("file-b#1")[_lkey(client, "extract_authors")]
    assert first["attempts"] == first["native_attempts"] + first["instructor_attempts"] == 2
    assert second["attempts"] == second["native_attempts"] + second["instructor_attempts"] == 2
    assert first["native_invalid_non_json"] == 1
    assert first["native_invalid_truncated"] == 0
    assert second["native_invalid_non_json"] == 0
    assert second["native_invalid_truncated"] == 1


def test_export_breaks_usage_down_by_label_provider_and_model():
    from bibr.export.usage import build_usage_export
    from tests.export.conftest import extraction_block as _extraction_block

    paper = _minimal_paper(
        llm_usage_labels={
            ("extract_authors", "google", "m"): {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "cached_input_tokens": 0,
                "calls": 1,
            }
        },
    )
    paper.extraction = _extraction_block(usage=build_usage_export(paper.llm_usage_labels))
    data = export_paper_to_json(paper)
    row = data["extraction"]["usage"]["breakdown"][0]
    assert (row["label"], row["provider"], row["model"], row["calls"]) == (
        "extract_authors",
        "google",
        "m",
        1,
    )
    assert data["schema_version"] == "11.0"


def test_export_omits_usage_when_no_labels_were_tracked():
    from tests.export.conftest import extraction_block as _extraction_block

    paper = _minimal_paper()
    paper.extraction = _extraction_block()
    data = export_paper_to_json(paper)
    assert "usage" not in data["extraction"]
