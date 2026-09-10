import pytest

from bibr.serve.admission import UploadAdmissionGate


def test_gate_rejects_above_limit_and_recovers():
    gate = UploadAdmissionGate(2)
    assert gate.try_acquire() is True
    assert gate.try_acquire() is True
    assert gate.try_acquire() is False
    assert gate.active == 2

    gate.release()
    assert gate.try_acquire() is True
    assert gate.active == 2


def test_gate_rejects_unbalanced_release():
    import pytest

    gate = UploadAdmissionGate(1)
    with pytest.raises(RuntimeError, match="without an acquisition"):
        gate.release()


async def test_middleware_rejects_before_second_upload_is_processed():
    import asyncio

    import httpx
    from fastapi import FastAPI, Request

    from bibr.serve.admission import add_upload_admission

    app = FastAPI()
    entered = asyncio.Event()
    release = asyncio.Event()
    processed = 0

    @app.post("/papers/extract")
    async def extract(request: Request):
        nonlocal processed
        processed += 1
        await request.body()
        entered.set()
        await release.wait()
        return {"ok": True}

    add_upload_admission(app, 1)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.post("/papers/extract", content=b"first"))
        await entered.wait()
        rejected = await client.post("/papers/extract", content=b"second")
        release.set()
        accepted = await first

    assert accepted.status_code == 200
    assert rejected.status_code == 429
    assert rejected.headers["retry-after"] == "1"
    assert processed == 1


async def test_spooled_upload_frees_its_slot_without_dropping_the_inflight_cap():
    """The gate bounds spooling, not the pipeline run that follows.

    A synchronous ``/papers/extract`` used to hold its slot for the whole
    30-120s pipeline, so a saturated worker rejected async job submissions with
    429 while the upload path was completely idle. It now hands the slot back
    once the body is on disk — but still holds an inflight slot, so a flood of
    synchronous extracts is told to back off rather than queueing unboundedly.
    """
    import asyncio

    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from bibr.serve.admission import add_upload_admission, release_spool_slot

    app = FastAPI()
    spooled = asyncio.Event()
    finish = asyncio.Event()

    @app.post("/papers/extract")
    async def extract(request: Request):
        await request.body()
        release_spool_slot(request)  # upload persisted; pipeline starts here
        spooled.set()
        await finish.wait()
        return {"ok": True}

    @app.post("/papers/jobs")
    async def submit_job(request: Request):
        await request.body()
        release_spool_slot(request)
        return JSONResponse({"queued": True}, status_code=202)

    gate = add_upload_admission(app, 1)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        running = asyncio.create_task(client.post("/papers/extract", content=b"first"))
        await spooled.wait()

        # The spool slot is back even though the extract is still running.
        assert gate.spool.active == 0
        assert gate.inflight.active == 1

        queued = await client.post("/papers/jobs", content=b"job")
        # The whole point: this was a 429 before the split.
        assert queued.status_code == 202

        # The synchronous route is still capped while one is in flight.
        rejected = await client.post("/papers/extract", content=b"second")
        assert rejected.status_code == 429
        assert rejected.headers["retry-after"] == "1"

        finish.set()
        accepted = await running

    assert accepted.status_code == 200
    assert gate.spool.active == 0
    assert gate.inflight.active == 0


async def test_middleware_releases_the_spool_slot_when_a_route_never_does():
    """A route that forgets the early release must not leak a slot."""
    import httpx
    from fastapi import FastAPI

    from bibr.serve.admission import add_upload_admission

    app = FastAPI()

    @app.post("/papers/jobs")
    async def submit_job():
        return {"queued": True}

    gate = add_upload_admission(app, 1)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            assert (await client.post("/papers/jobs", content=b"x")).status_code == 200
    assert gate.spool.active == 0


def test_release_spool_slot_is_a_no_op_off_route():
    from starlette.requests import Request

    from bibr.serve.admission import release_spool_slot

    request = Request({"type": "http", "method": "GET", "path": "/info", "headers": []})
    release_spool_slot(request)  # no slot was ever taken


@pytest.mark.parametrize("path", ["/mcp", "/mcp/", "/mcp/nested"])
async def test_mcp_rejected_before_reading_json_when_rest_upload_fills_spool(path):
    import httpx
    from fastapi import FastAPI

    from bibr.serve.admission import add_upload_admission

    app = FastAPI()
    gate = add_upload_admission(app, 1)
    body_read = False

    async def body():
        nonlocal body_read
        body_read = True
        yield b'{"large": "payload"}'

    with gate.admit(inflight=False):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(path, content=body())
    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert not body_read
    assert gate.spool.active == 0
