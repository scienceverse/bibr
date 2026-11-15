"""bibr.Chewer driven from several threads."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch

import pytest

import bibr
from tests.test_api import _export_fixture, _StubPipeline, _touch_pdfs


@pytest.fixture
def stub_pipeline():
    _StubPipeline.instances = []
    with patch("bibr.local.pipeline.LocalPipeline", _StubPipeline):
        yield _StubPipeline


def test_chewer_runs_calls_from_two_threads_one_at_a_time(stub_pipeline, tmp_path, monkeypatch):
    """A second thread's chew() used to raise "This event loop is already
    running" after scheduling its file on the loop the first thread was
    running: the file was processed anyway, concurrently with the other, and
    its result thrown away."""
    events: list[str] = []
    first_started = threading.Event()

    async def slow_file(self, path, paper_id=None, progress=None):
        events.append(f"start {path.name}")
        first_started.set()
        await asyncio.sleep(0.3)
        events.append(f"end {path.name}")
        return _export_fixture()

    monkeypatch.setattr(stub_pipeline, "process_file", slow_file)
    a, b = _touch_pdfs(tmp_path, "a.pdf", "b.pdf")
    chewer = bibr.Chewer()
    results: dict[str, object] = {}

    def run(path):
        try:
            results[path.name] = chewer.chew(path)
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            results[path.name] = exc

    first = threading.Thread(target=run, args=(a,))
    first.start()
    assert first_started.wait(5)
    second = threading.Thread(target=run, args=(b,))
    second.start()
    first.join(10)
    second.join(10)
    chewer.close()

    assert all(isinstance(r, bibr.Result) for r in results.values()), results
    assert events == ["start a.pdf", "end a.pdf", "start b.pdf", "end b.pdf"]


def test_chewer_close_from_another_thread_waits_for_the_running_chew(
    stub_pipeline, tmp_path, monkeypatch
):
    events: list[str] = []
    started = threading.Event()

    async def slow_file(self, path, paper_id=None, progress=None):
        started.set()
        await asyncio.sleep(0.3)
        events.append("chew done")
        return _export_fixture()

    async def aclose(self):
        events.append("aclose")

    monkeypatch.setattr(stub_pipeline, "process_file", slow_file)
    monkeypatch.setattr(stub_pipeline, "aclose", aclose)
    (paper,) = _touch_pdfs(tmp_path, "a.pdf")
    chewer = bibr.Chewer()
    results: list[object] = []
    worker = threading.Thread(target=lambda: results.append(chewer.chew(paper)))
    worker.start()
    assert started.wait(5)
    chewer.close()
    worker.join(10)

    assert events == ["chew done", "aclose"]
    assert isinstance(results[0], bibr.Result)
