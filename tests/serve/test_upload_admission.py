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


def test_base64_envelope_matches_the_encoder():
    import base64

    from bibr.serve.admission import base64_envelope

    for size in (0, 1, 2, 3, 4, 5, 6, 100, 1000, 65_537):
        assert base64_envelope(size) == len(base64.b64encode(b"x" * size))


def _body_gated_app(limit: int, *, threshold: int = 16, max_body: int | None = 1000):
    from fastapi import FastAPI, Request

    from bibr.serve.admission import add_upload_admission

    app = FastAPI()
    handled: list[int] = []

    @app.post("/mcp/")
    async def mcp(request: Request):
        body = await request.body()
        handled.append(len(body))
        return {"received": len(body)}

    gate = add_upload_admission(
        app,
        limit,
        body_gated_paths=("/mcp", "/mcp/"),
        body_threshold=threshold,
        max_body=max_body,
        max_file_size=600,
    )
    return app, gate, handled


async def test_body_gate_holds_a_slot_only_while_a_large_body_is_received():
    import asyncio

    import httpx

    app, gate, handled = _body_gated_app(1)
    first_chunk_sent = asyncio.Event()
    finish_first = asyncio.Event()

    async def slow_body():
        yield b"x" * 32
        first_chunk_sent.set()
        await finish_first.wait()
        yield b"y" * 32

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(
            client.post("/mcp/", content=slow_body(), headers={"content-length": "64"})
        )
        await first_chunk_sent.wait()
        # The slot is held while the first body is still arriving...
        assert gate.active == 1
        rejected = await client.post("/mcp/", content=b"z" * 64)
        # ...but a body under the threshold is not an upload and passes.
        small = await client.post("/mcp/", content=b"s" * 8)
        finish_first.set()
        accepted = await first
        # Once a body is fully received the slot is free again, before the
        # handler has finished with it.
        after = await client.post("/mcp/", content=b"w" * 64)

    assert accepted.status_code == 200 and accepted.json() == {"received": 64}
    assert rejected.status_code == 429 and rejected.headers["retry-after"] == "1"
    assert small.status_code == 200
    assert after.status_code == 200
    assert gate.active == 0
    assert sorted(handled) == [8, 64, 64]


async def test_body_gate_refuses_a_declared_oversize_body_before_reading_it():
    import httpx

    app, gate, handled = _body_gated_app(4, max_body=1000)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/", content=b"x" * 2000)

    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert "exceeds the 0.0 MiB limit" in detail
    assert "chew_paper accepts files up to 0 MiB" in detail
    assert "chew_url or POST /papers/extract" in detail
    assert handled == []
    assert gate.active == 0


async def test_body_gate_ignores_other_paths_and_methods():
    import httpx
    from fastapi import FastAPI

    from bibr.serve.admission import add_upload_admission

    app = FastAPI()

    @app.post("/other")
    async def other():
        return {"ok": True}

    @app.get("/mcp/")
    async def mcp_get():
        return {"ok": True}

    gate = add_upload_admission(app, 1, body_gated_paths=("/mcp/",), body_threshold=0)
    gate.try_acquire()  # the gate is full
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/other", content=b"x" * 100)).status_code == 200
        assert (await client.get("/mcp/")).status_code == 200


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
