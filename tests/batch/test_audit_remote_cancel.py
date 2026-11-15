"""A cancelled remote run keeps the outcome of every job that finished before it."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from bibr.batch.ledger import INTERRUPTED
from bibr.batch.manifest import BatchItem
from bibr.batch.remote import RemoteExecutor, RemoteOptions


def _export(stem: str) -> dict:
    return {"info": {"title": stem}, "text": [], "bib": []}


def _items(tmp_path: Path, *stems: str) -> list[BatchItem]:
    items = []
    for stem in stems:
        path = tmp_path / f"{stem}.pdf"
        path.write_bytes(b"%PDF-1.4\n" + stem.encode())
        items.append(BatchItem(path=path, paper_id=stem, stem=stem))
    return items


async def test_a_second_ctrl_c_during_the_grace_keeps_the_jobs_that_finished(tmp_path):
    """First Ctrl-C drains in-flight jobs for ``grace``; a second one cancels the run.

    A job that finished during the drain already has its export: it must be
    recorded as ok, not as interrupted.
    """
    jobs: dict[str, str] = {}
    fast_done = {"now": False}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if path == "/papers/jobs" and request.method == "POST":
            stem = "fast" if b'filename="fast.pdf"' in request.content else "slow"
            job_id = f"job-{stem}"
            jobs[job_id] = stem
            return httpx.Response(202, json={"job_id": job_id})
        parts = path.split("/")
        job_id = parts[3]
        if request.method == "DELETE":
            return httpx.Response(409, json={"detail": "running"})
        if len(parts) == 5:
            return httpx.Response(200, json=_export(jobs[job_id]))
        status = "succeeded" if jobs[job_id] == "fast" and fast_done["now"] else "running"
        return httpx.Response(200, json={"job_id": job_id, "status": status})

    async def sleep(_seconds: float) -> None:
        await asyncio.sleep(0)

    executor = RemoteExecutor(
        RemoteOptions(serve_url="http://serve", concurrency=2, poll_interval=1.0, grace=3600.0),
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    stop = asyncio.Event()
    seen: list[tuple[str, object]] = []
    fast, slow = _items(tmp_path, "fast", "slow")

    run = asyncio.create_task(
        executor.run([fast, slow], on_outcome=lambda i, o: seen.append((i.paper_id, o)), stop=stop)
    )
    while len(jobs) < 2:
        await asyncio.sleep(0)
    stop.set()  # the first Ctrl-C
    for _ in range(50):
        await asyncio.sleep(0)
    assert seen == []  # draining, nothing finished yet
    # The fast job finishes inside the drain; then the second Ctrl-C.
    fast_done["now"] = True
    for _ in range(200):
        await asyncio.sleep(0)
    assert seen == []  # the drain waits for every job, up to the grace
    run.cancel()
    try:
        await run
    except asyncio.CancelledError:
        pass

    outcomes = dict(seen)
    assert sorted(outcomes) == ["fast", "slow"]
    assert outcomes["fast"].ok
    assert outcomes["fast"].export["info"]["title"] == "fast"
    assert outcomes["slow"].error_code == INTERRUPTED
