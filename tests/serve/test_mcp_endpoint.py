"""Tests for the serve-mounted MCP endpoint (``bibr/serve/mcp.py``).

Tool logic runs against the MCP SDK's in-memory transport with a fake
inference tracker (the real dispatch path is covered by the ingress and jobs
tests). The mount itself is exercised over the actual streamable-HTTP
protocol: SDK client → httpx ASGI transport → FastAPI app with the bearer
gate → mounted transport handler, including the bare-``/mcp`` (no trailing
slash) case that a router mount alone would 307.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("fastapi")

import httpx  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402
from mcp.shared.memory import (  # noqa: E402
    create_connected_server_and_client_session as memory_session,
)

from bibr.mcp_server import _PaperStore  # noqa: E402
from bibr.serve.admission import UploadAdmissionGate, add_upload_admission  # noqa: E402
from bibr.serve.ingress import UploadStore  # noqa: E402
from bibr.serve.mcp import _SessionStores, build_serve_mcp, mount_mcp  # noqa: E402

FIXTURE = Path(__file__).parent.parent / "fixtures" / "inspect_full_export.json"
FIXTURE_ID = "10.1234/example.5678"
PDF_BYTES = b"%PDF-1.4 stub content"
PDF_B64 = base64.b64encode(PDF_BYTES).decode()


def _fixture_data() -> dict:
    return json.loads(FIXTURE.read_text())


class _FakeTracker:
    """Dispatch double; the real tracker is covered in ingress/jobs tests."""

    def __init__(self, result: dict | None = None, error: HTTPException | None = None):
        self.result = result if result is not None else _fixture_data()
        self.error = error
        self.descriptors: list[dict] = []

    async def submit(self, descriptor, request_state=None, *, admission=None):
        self.descriptors.append(descriptor)
        if self.error is not None:
            raise self.error
        return self.result


def _payload(result):
    assert not result.isError, [c.text for c in result.content]
    payload = result.structuredContent
    assert payload is not None
    return payload["result"] if set(payload) == {"result"} else payload


def _error_text(result) -> str:
    assert result.isError
    return result.content[0].text


@asynccontextmanager
async def _tool_session(
    tracker: _FakeTracker,
    *,
    max_size: int = 1_000_000,
    max_papers: int = 8,
    admission_gate: UploadAdmissionGate | None = None,
):
    store = UploadStore.create(max_size=max_size, spool_memory_bytes=1024, stale_after_seconds=60)
    try:
        server = build_serve_mcp(
            upload_store=store,
            tracker=tracker,
            max_papers_per_session=max_papers,
            admission_gate=admission_gate or UploadAdmissionGate(8),
        )
        async with memory_session(server) as client:
            yield client
    finally:
        await store.close()


async def test_serve_surface_has_no_filesystem_tools():
    async with _tool_session(_FakeTracker()) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert "chew_paper" in tools
        assert "chew_url" in tools
        assert "load_paper" not in tools
        assert "save_paper" not in tools


async def test_chew_url_can_be_disabled():
    store = UploadStore.create(max_size=1_000_000, spool_memory_bytes=1024, stale_after_seconds=60)
    try:
        server = build_serve_mcp(
            upload_store=store,
            tracker=_FakeTracker(),
            admission_gate=UploadAdmissionGate(8),
            max_papers_per_session=4,
            chew_url_enabled=False,
        )
        async with memory_session(server) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert "chew_url" not in tools
            assert "chew_paper" in tools
    finally:
        await store.close()


async def test_chew_url_fetches_and_dispatches(monkeypatch):
    import bibr.serve.mcp as serve_mcp
    from bibr.utils.safe_fetch import FetchedFile

    seen: dict[str, object] = {}

    async def fake_fetch(url, *, max_size, allowed_hosts=None, **kwargs):
        seen["url"] = url
        seen["max_size"] = max_size
        seen["allowed_hosts"] = allowed_hosts
        return FetchedFile(
            content=PDF_BYTES,
            filename="fetched.pdf",
            content_type="application/pdf",
            final_url=url,
        )

    monkeypatch.setattr(serve_mcp, "fetch_url_safely", fake_fetch)
    tracker = _FakeTracker()
    async with _tool_session(tracker) as client:
        summary = _payload(
            await client.call_tool(
                "chew_url", {"url": "https://arxiv.org/pdf/1234.pdf", "refs": "off"}
            )
        )
        # The fetch is capped by the same limit as uploads, and the summary's
        # source is the URL the caller asked for.
        assert seen["max_size"] == 1_000_000
        assert summary["source"] == "https://arxiv.org/pdf/1234.pdf"
        (descriptor,) = tracker.descriptors
        assert descriptor["filename"] == "fetched.pdf"
        assert descriptor["refs"] == "off"
        assert descriptor["sha256"] == hashlib.sha256(PDF_BYTES).hexdigest()


async def test_chew_url_option_validation_precedes_fetch(monkeypatch):
    import bibr.serve.mcp as serve_mcp

    async def must_not_run(*a, **k):
        raise AssertionError("fetch must not run for invalid options")

    monkeypatch.setattr(serve_mcp, "fetch_url_safely", must_not_run)
    async with _tool_session(_FakeTracker()) as client:
        assert "refs must be one of" in _error_text(
            await client.call_tool(
                "chew_url", {"url": "https://arxiv.org/pdf/1.pdf", "refs": "bogus"}
            )
        )


async def test_chew_url_enforces_allowlist_and_policy():
    store = UploadStore.create(max_size=1_000_000, spool_memory_bytes=1024, stale_after_seconds=60)
    try:
        server = build_serve_mcp(
            upload_store=store,
            tracker=_FakeTracker(),
            admission_gate=UploadAdmissionGate(8),
            max_papers_per_session=4,
            url_allowed_hosts=["arxiv.org"],
        )
        async with memory_session(server) as client:
            assert "allowlist" in _error_text(
                await client.call_tool("chew_url", {"url": "https://evil.org/p.pdf"})
            )
            assert "only https" in _error_text(
                await client.call_tool("chew_url", {"url": "http://arxiv.org/p.pdf"})
            )
    finally:
        await store.close()


async def test_chew_paper_uploads_and_dispatches():
    tracker = _FakeTracker()
    async with _tool_session(tracker) as client:
        summary = _payload(
            await client.call_tool(
                "chew_paper",
                {
                    "filename": "paper.pdf",
                    "content_base64": PDF_B64,
                    "start_page": 1,
                    "end_page": 5,
                    "refs": "ner",
                },
            )
        )
        assert summary["paper_id"] == FIXTURE_ID
        assert summary["source"] == "paper.pdf"
        assert summary["counts"]["references"] == 2
        assert isinstance(summary["seconds"], float)

        # The descriptor is the same shape /papers/extract submits: content
        # identity plus the validated pass-through options.
        (descriptor,) = tracker.descriptors
        assert descriptor["filename"] == "paper.pdf"
        assert descriptor["size"] == len(PDF_BYTES)
        assert descriptor["sha256"] == hashlib.sha256(PDF_BYTES).hexdigest()
        assert descriptor["start_page"] == "0"
        assert descriptor["end_page"] == "4"
        assert descriptor["refs"] == "ner"
        assert "consolidate" not in descriptor

        # The chewed paper is registered for the query tools.
        refs = _payload(
            await client.call_tool("get_references", {"paper_id": FIXTURE_ID, "limit": 1})
        )
        assert refs["total"] == 2


async def test_chew_paper_input_errors():
    tracker = _FakeTracker()
    async with _tool_session(tracker) as client:
        assert "not valid base64" in _error_text(
            await client.call_tool(
                "chew_paper", {"filename": "x.pdf", "content_base64": "!!not-base64!!"}
            )
        )
        assert "refs must be one of" in _error_text(
            await client.call_tool(
                "chew_paper",
                {"filename": "x.pdf", "content_base64": PDF_B64, "refs": "bogus"},
            )
        )
        assert "start_page (9) must be <= end_page (2)" in _error_text(
            await client.call_tool(
                "chew_paper",
                {"filename": "x.pdf", "content_base64": PDF_B64, "start_page": 9, "end_page": 2},
            )
        )
        assert "invalid upload" in _error_text(
            await client.call_tool("chew_paper", {"filename": "x.pdf", "content_base64": ""})
        )
        # Nothing was ever dispatched for a rejected request.
        assert tracker.descriptors == []


async def test_chew_paper_too_large():
    async with _tool_session(_FakeTracker(), max_size=8) as client:
        assert "file too large" in _error_text(
            await client.call_tool("chew_paper", {"filename": "x.pdf", "content_base64": PDF_B64})
        )


async def test_chew_paper_maps_pipeline_errors():
    tracker = _FakeTracker(error=HTTPException(status_code=422, detail="pipeline exploded"))
    async with _tool_session(tracker) as client:
        text = _error_text(
            await client.call_tool("chew_paper", {"filename": "p.pdf", "content_base64": PDF_B64})
        )
        assert "extraction failed for p.pdf" in text
        assert "pipeline exploded" in text


@pytest.mark.parametrize("tool_name", ["chew_paper", "chew_url"])
@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        ({}, {}),
        ({"start_page": 1}, {"start_page": "0"}),
        ({"end_page": 3}, {"end_page": "2"}),
        ({"start_page": 1, "end_page": 1}, {"start_page": "0", "end_page": "0"}),
    ],
)
async def test_mcp_page_numbers_are_converted_for_the_worker(
    monkeypatch, tool_name, pages, expected
):
    from bibr.utils.safe_fetch import FetchedFile

    async def fetch(*args, **kwargs):
        return FetchedFile(PDF_BYTES, "p.pdf", "application/pdf", "https://example.org/p.pdf")

    monkeypatch.setattr("bibr.serve.mcp.fetch_url_safely", fetch)
    tracker = _FakeTracker()
    arguments = (
        {"filename": "p.pdf", "content_base64": PDF_B64}
        if tool_name == "chew_paper"
        else {"url": "https://example.org/p.pdf"}
    )
    async with _tool_session(tracker) as client:
        _payload(await client.call_tool(tool_name, {**arguments, **pages}))
    actual = {k: v for k, v in tracker.descriptors[0].items() if k in ("start_page", "end_page")}
    assert actual == expected


@pytest.mark.parametrize("tool_name", ["chew_paper", "chew_url"])
@pytest.mark.parametrize("name", ["start_page", "end_page"])
@pytest.mark.parametrize("page", [0, -1])
async def test_mcp_rejects_nonpositive_page_numbers(tool_name, name, page):
    tracker = _FakeTracker()
    arguments = (
        {"filename": "p.pdf", "content_base64": PDF_B64}
        if tool_name == "chew_paper"
        else {"url": "https://example.org/p.pdf"}
    )
    async with _tool_session(tracker) as client:
        assert "pages are 1-based" in _error_text(
            await client.call_tool(tool_name, {**arguments, name: page})
        )
    assert not tracker.descriptors


async def test_tool_errors_release_slots_for_the_next_extraction():
    gate = UploadAdmissionGate(1)
    tracker = _FakeTracker(error=HTTPException(status_code=422, detail="pipeline exploded"))
    async with _tool_session(tracker, max_size=32, admission_gate=gate) as client:
        cases = [
            ("chew_paper", {"filename": "p.pdf", "content_base64": "!"}, "not valid base64"),
            ("chew_paper", {"filename": "p.pdf", "content_base64": "A" * 100}, "file too large"),
            ("chew_url", {"url": "http://example.org/p.pdf"}, "only https"),
            ("chew_paper", {"filename": "p.pdf", "content_base64": PDF_B64}, "pipeline exploded"),
        ]
        for tool, arguments, error in cases:
            assert error in _error_text(await client.call_tool(tool, arguments))
            assert gate.spool.active == gate.inflight.active == 0


def test_session_stores_isolate_and_release():
    class _Session:
        """Weakref-able stand-in for an MCP ServerSession."""

    class _Ctx:
        def __init__(self, session):
            self.session = session

    stores = _SessionStores(max_papers=4)
    a, b = _Session(), _Session()
    store_a = stores.resolve(_Ctx(a))
    store_b = stores.resolve(_Ctx(b))
    assert store_a is not store_b
    assert stores.resolve(_Ctx(a)) is store_a

    store_a.add(_fixture_data(), source="a.pdf")
    with pytest.raises(Exception, match="unknown paper_id"):
        store_b.get(FIXTURE_ID)

    del a
    gc.collect()
    assert len(stores._stores) == 1


def test_paper_store_evicts_oldest_beyond_cap():
    store = _PaperStore(max_papers=2)
    store.add({"paper_id": "one"}, source="one.pdf")
    store.add({"paper_id": "two"}, source="two.pdf")
    store.add({"paper_id": "three"}, source="three.pdf")
    assert [pid for pid, _ in store.items()] == ["two", "three"]
    # Overwriting an existing id is not an eviction.
    store.add({"paper_id": "three"}, source="three.pdf")
    assert [pid for pid, _ in store.items()] == ["two", "three"]


# ---------------------------------------------------------------------------
# Streamable-HTTP integration: mount + auth gate + protocol round-trip
# ---------------------------------------------------------------------------

_TEST_KEY = "sk_test_0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def _restore_api_key():
    from bibr.config import Settings

    original = Settings.auth.api_key
    yield
    Settings.auth.api_key = original


@asynccontextmanager
async def _mounted_app(tracker: _FakeTracker, *, limit: int = 8):
    """A FastAPI app shaped like serve's: bearer gate middleware + /mcp mount."""
    from bibr.config import Settings
    from bibr.serve.auth import check_bearer

    Settings.auth.api_key = _TEST_KEY
    app = FastAPI()
    gate = add_upload_admission(app, limit)
    app.state.admission_gate = gate

    @app.middleware("http")
    async def _auth_gate(request, call_next):
        detail = check_bearer(request.headers.get("authorization"))
        if detail is not None:
            return JSONResponse(
                {"detail": detail}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
            )
        return await call_next(request)

    store = UploadStore.create(max_size=1_000_000, spool_memory_bytes=1024, stale_after_seconds=60)
    try:
        mount_mcp(app, Settings, upload_store=store, tracker=tracker, admission_gate=gate)
        async with app.router.lifespan_context(app):
            yield app
    finally:
        await store.close()


def _asgi_factory(app):
    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    return factory


async def test_http_mount_requires_bearer():
    async with _mounted_app(_FakeTracker()) as app:
        # The MCP client surfaces the 401 as a connection failure; assert the
        # HTTP layer directly so the contract is visible.
        async with _asgi_factory(app)() as client:
            resp = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                headers={"Accept": "application/json, text/event-stream"},
            )
            assert resp.status_code == 401


async def test_http_mount_serves_bare_path_without_redirect():
    async with _mounted_app(_FakeTracker()) as app:
        async with _asgi_factory(app)() as client:
            resp = await client.post(
                "/mcp",
                headers={
                    "Authorization": f"Bearer {_TEST_KEY}",
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "t", "version": "0"},
                    },
                },
            )
            # Not a 307: the bare path is served directly.
            assert resp.status_code == 200
            assert resp.headers.get("mcp-session-id")


async def test_http_round_trip_chew_and_query():
    tracker = _FakeTracker()
    async with _mounted_app(tracker) as app:
        headers = {"Authorization": f"Bearer {_TEST_KEY}"}
        async with streamablehttp_client(
            "http://testserver/mcp", headers=headers, httpx_client_factory=_asgi_factory(app)
        ) as (read, write, get_session_id):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                assert init.serverInfo.name == "bibr"
                assert get_session_id()

                tools = {t.name for t in (await session.list_tools()).tools}
                assert "chew_paper" in tools
                assert "load_paper" not in tools

                res = await session.call_tool(
                    "chew_paper", {"filename": "p.pdf", "content_base64": PDF_B64}
                )
                assert not res.isError
                assert res.structuredContent["paper_id"] == FIXTURE_ID

                papers = await session.call_tool("list_papers", {})
                assert len(papers.structuredContent["result"]) == 1


@pytest.mark.parametrize("first_source", ["mcp", "rest"])
async def test_http_mcp_and_rest_share_extraction_capacity(first_source, monkeypatch):
    from bibr.serve.admission import release_spool_slot

    entered = asyncio.Event()
    finish = asyncio.Event()

    class BlockingTracker(_FakeTracker):
        async def submit(self, descriptor, request_state=None, *, admission=None):
            self.descriptors.append(descriptor)
            entered.set()
            await finish.wait()
            return self.result

    async def forbidden_fetch(*args, **kwargs):
        raise AssertionError("busy URL extraction must not start a download")

    monkeypatch.setattr("bibr.serve.mcp.fetch_url_safely", forbidden_fetch)
    tracker = BlockingTracker()
    async with _mounted_app(tracker, limit=1) as app:

        @app.post("/papers/extract")
        async def extract(request: Request):
            await request.body()
            release_spool_slot(request)
            entered.set()
            await finish.wait()
            return {"ok": True}

        headers = {"Authorization": f"Bearer {_TEST_KEY}"}
        async with _asgi_factory(app)(headers=headers) as rest:
            async with streamablehttp_client(
                "http://testserver/mcp", headers=headers, httpx_client_factory=_asgi_factory(app)
            ) as (read, write, _):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    # Drain the initialization notification's HTTP upload
                    # before competing with a one-slot REST upload.
                    await client.list_tools()
                    first = asyncio.create_task(
                        client.call_tool(
                            "chew_paper", {"filename": "p.pdf", "content_base64": PDF_B64}
                        )
                        if first_source == "mcp"
                        else rest.post("/papers/extract", content=b"paper")
                    )
                    try:
                        await asyncio.wait_for(entered.wait(), timeout=5)
                        gate = app.state.admission_gate
                        assert gate.spool.active == 0
                        assert gate.inflight.active == 1
                        rejected = await rest.post("/papers/extract", content=b"second")
                        assert rejected.status_code == 429
                        for tool, args in [
                            ("chew_paper", {"filename": "p.pdf", "content_base64": "invalid!"}),
                            ("chew_url", {"url": "https://example.org/p.pdf"}),
                        ]:
                            assert "Too many requests in flight" in _error_text(
                                await client.call_tool(tool, args)
                            )
                        # Queries remain usable while inference consumes the cap.
                        _payload(await client.call_tool("list_papers", {}))
                    finally:
                        finish.set()
                        await first
                    assert gate.spool.active == gate.inflight.active == 0
                    _payload(
                        await client.call_tool(
                            "chew_paper", {"filename": "p.pdf", "content_base64": PDF_B64}
                        )
                    )
