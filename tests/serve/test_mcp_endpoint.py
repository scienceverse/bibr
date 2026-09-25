"""Tests for the serve-mounted MCP endpoint (``bibr/serve/mcp.py``).

Tool logic runs against the MCP SDK's in-memory transport with a fake
inference tracker (the real dispatch path is covered by the ingress and jobs
tests). The mount itself is exercised over the actual streamable-HTTP
protocol: SDK client → httpx ASGI transport → FastAPI app with the bearer
gate → mounted transport handler, including the bare-``/mcp`` (no trailing
slash) case that a router mount alone would 307.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("fastapi")

import httpx2 as httpx  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.client._memory import InMemoryTransport  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

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
    assert not result.is_error, [c.text for c in result.content]
    payload = result.structured_content
    assert payload is not None
    return payload["result"] if set(payload) == {"result"} else payload


def _error_text(result) -> str:
    assert result.is_error
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
            admission_gate=admission_gate,
        )
        async with Client(InMemoryTransport(server), mode="legacy") as client:
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
            max_papers_per_session=4,
            chew_url_enabled=False,
        )
        async with Client(InMemoryTransport(server), mode="legacy") as client:
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


@pytest.mark.parametrize(("crossref", "expected"), [(True, "true"), (False, "false")])
async def test_chew_paper_passes_crossref_switch_through(crossref, expected):
    """The tool's ``crossref`` knob rides the descriptor exactly like the form field."""
    tracker = _FakeTracker()
    async with _tool_session(tracker) as client:
        await client.call_tool(
            "chew_paper",
            {"filename": "paper.pdf", "content_base64": PDF_B64, "crossref": crossref},
        )
        (descriptor,) = tracker.descriptors
        assert descriptor["crossref"] == expected


async def test_chew_paper_omits_crossref_when_not_given():
    tracker = _FakeTracker()
    async with _tool_session(tracker) as client:
        await client.call_tool("chew_paper", {"filename": "paper.pdf", "content_base64": PDF_B64})
        (descriptor,) = tracker.descriptors
        assert "crossref" not in descriptor


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
            max_papers_per_session=4,
            url_allowed_hosts=["arxiv.org"],
        )
        async with Client(InMemoryTransport(server), mode="legacy") as client:
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
        assert descriptor["start_page"] == "1"
        assert descriptor["end_page"] == "5"
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


async def test_chew_tools_hold_an_upload_admission_slot():
    """Both chew tools take the serve upload slot and give it back, like /papers/extract."""
    gate = UploadAdmissionGate(1)
    tracker = _FakeTracker()
    async with _tool_session(tracker, admission_gate=gate) as client:
        gate.try_acquire()  # another upload is in flight
        busy = await client.call_tool(
            "chew_paper", {"filename": "x.pdf", "content_base64": PDF_B64}
        )
        assert "server busy" in _error_text(busy)
        assert tracker.descriptors == []
        gate.release()

        summary = _payload(
            await client.call_tool("chew_paper", {"filename": "x.pdf", "content_base64": PDF_B64})
        )
        assert summary["paper_id"]
        assert len(tracker.descriptors) == 1
        assert gate.active == 0

        # A failing extraction releases the slot too.
        failing = _FakeTracker(error=HTTPException(status_code=422, detail="bad"))
        async with _tool_session(failing, admission_gate=gate) as client2:
            result = await client2.call_tool(
                "chew_paper", {"filename": "x.pdf", "content_base64": PDF_B64}
            )
            assert result.is_error
            assert gate.active == 0


async def test_chew_url_holds_an_upload_admission_slot(monkeypatch):
    from bibr.serve import mcp as serve_mcp

    async def fake_fetch(url, *, max_size, allowed_hosts=None):
        raise AssertionError("fetch must not run without an admission slot")

    monkeypatch.setattr(serve_mcp, "fetch_url_safely", fake_fetch)
    gate = UploadAdmissionGate(1)
    gate.try_acquire()
    async with _tool_session(_FakeTracker(), admission_gate=gate) as client:
        busy = await client.call_tool("chew_url", {"url": "https://example.org/p.pdf"})
        assert "server busy" in _error_text(busy)
    assert gate.active == 1


async def test_chew_tools_free_the_spool_slot_before_the_pipeline_runs():
    """An extraction runs for 30-120s and holds an *inflight* slot for it, not a
    spool slot — otherwise a busy pipeline rejects uploads on every route with
    429 while the spooling phase itself sits idle."""
    gate = UploadAdmissionGate(1)
    seen: dict[str, int] = {}

    class _ObservingTracker(_FakeTracker):
        async def submit(self, descriptor, request_state=None, *, admission=None):
            seen["spool"] = gate.spool.active
            seen["inflight"] = gate.inflight.active
            return await super().submit(descriptor, request_state)

    async with _tool_session(_ObservingTracker(), admission_gate=gate) as client:
        summary = _payload(
            await client.call_tool("chew_paper", {"filename": "x.pdf", "content_base64": PDF_B64})
        )
        assert summary["paper_id"]

    assert seen == {"spool": 0, "inflight": 1}
    assert gate.spool.active == 0
    assert gate.inflight.active == 0


async def test_chew_paper_refused_when_only_inflight_is_exhausted():
    """A saturated pipeline backs the tool off even with spool capacity free,
    and hands the spool slot it took back on the way out."""
    gate = UploadAdmissionGate(4, inflight_limit=1)
    gate.inflight.try_acquire()  # a pipeline run is already going
    tracker = _FakeTracker()
    async with _tool_session(tracker, admission_gate=gate) as client:
        busy = await client.call_tool(
            "chew_paper", {"filename": "x.pdf", "content_base64": PDF_B64}
        )
        assert "too many requests in flight" in _error_text(busy)
        assert tracker.descriptors == []
        assert gate.spool.active == 0


async def test_chew_url_frees_the_spool_slot_before_the_pipeline_runs(monkeypatch):
    from bibr.serve import mcp as serve_mcp
    from bibr.utils.safe_fetch import FetchedFile

    async def fake_fetch(url, *, max_size, allowed_hosts=None, **kwargs):
        # The fetch is the memory-heavy phase, so it runs under the spool slot.
        assert gate.spool.active == 1
        return FetchedFile(
            content=PDF_BYTES,
            filename="fetched.pdf",
            content_type="application/pdf",
            final_url=url,
        )

    monkeypatch.setattr(serve_mcp, "fetch_url_safely", fake_fetch)
    gate = UploadAdmissionGate(1)
    seen: dict[str, int] = {}

    class _ObservingTracker(_FakeTracker):
        async def submit(self, descriptor, request_state=None, *, admission=None):
            seen["spool"] = gate.spool.active
            seen["inflight"] = gate.inflight.active
            return await super().submit(descriptor, request_state)

    async with _tool_session(_ObservingTracker(), admission_gate=gate) as client:
        assert _payload(await client.call_tool("chew_url", {"url": "https://example.org/p.pdf"}))

    assert seen == {"spool": 0, "inflight": 1}
    assert gate.spool.active == 0
    assert gate.inflight.active == 0


async def test_chew_paper_maps_pipeline_errors():
    tracker = _FakeTracker(error=HTTPException(status_code=422, detail="pipeline exploded"))
    async with _tool_session(tracker) as client:
        text = _error_text(
            await client.call_tool("chew_paper", {"filename": "p.pdf", "content_base64": PDF_B64})
        )
        assert "extraction failed for p.pdf" in text
        assert "pipeline exploded" in text


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
            self.session = type("SessionProxy", (), {"client_params": session})()

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
async def _mounted_app(tracker: _FakeTracker, *, max_size: int = 1_000_000, max_body: int = 4096):
    """A FastAPI app shaped like serve's: bearer gate middleware + /mcp mount."""
    from bibr.config import Settings
    from bibr.serve.auth import check_bearer

    Settings.auth.api_key = _TEST_KEY
    app = FastAPI()
    # Same order as bibr.serve.app: admission inside, auth outside.
    gate = add_upload_admission(
        app,
        1,
        body_gated_paths=("/mcp", "/mcp/"),
        body_threshold=1024,
        max_body=max_body,
        max_file_size=max(3 * 1024 * 1024, max_size),
    )
    app.state.upload_admission_gate = gate

    @app.middleware("http")
    async def _auth_gate(request, call_next):
        detail = check_bearer(request.headers.get("authorization"))
        if detail is not None:
            return JSONResponse(
                {"detail": detail}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
            )
        return await call_next(request)

    store = UploadStore.create(max_size=max_size, spool_memory_bytes=1024, stale_after_seconds=60)
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


async def test_http_mount_expires_idle_sessions(monkeypatch):
    """A client that vanishes without DELETE must not pin its papers forever."""
    from bibr.config import Settings

    store = UploadStore.create(max_size=1_000_000, spool_memory_bytes=1024, stale_after_seconds=60)
    try:
        monkeypatch.setattr(Settings.mcp, "session_idle_timeout_seconds", 123.0)
        server = mount_mcp(FastAPI(), Settings, upload_store=store, tracker=_FakeTracker())
        assert server.session_manager.session_idle_timeout == 123.0

        monkeypatch.setattr(Settings.mcp, "session_idle_timeout_seconds", 0)
        server = mount_mcp(FastAPI(), Settings, upload_store=store, tracker=_FakeTracker())
        assert server.session_manager.session_idle_timeout is None
    finally:
        await store.close()


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


async def test_http_mount_gates_large_bodies_before_the_transport_buffers_them():
    async with _mounted_app(_FakeTracker()) as app:
        headers = {
            "Authorization": f"Bearer {_TEST_KEY}",
            "Accept": "application/json, text/event-stream",
        }
        async with _asgi_factory(app)() as client:
            # Declared above the body cap: a clean 413 that explains the base64 math.
            resp = await client.post("/mcp", content=b"x" * 5000, headers=headers)
            assert resp.status_code == 413
            assert "chew_paper accepts files up to 3 MiB" in resp.json()["detail"]

            # Under the cap but above the threshold: takes the (only) slot while
            # the body is received — so with the gate full it is refused.
            gate = app.state.upload_admission_gate
            gate.try_acquire()
            try:
                resp = await client.post("/mcp", content=b"x" * 2000, headers=headers)
                assert resp.status_code == 429
            finally:
                gate.release()

            # A small JSON-RPC body is not an upload and reaches the transport.
            resp = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                headers=headers,
            )
            assert resp.status_code not in (413, 429)
            assert gate.active == 0


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


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_http_round_trip_chew_and_query(mode):
    tracker = _FakeTracker()
    async with _mounted_app(tracker) as app:
        headers = {"Authorization": f"Bearer {_TEST_KEY}"}
        async with (
            _asgi_factory(app)(headers=headers) as http_client,
            streamable_http_client("http://testserver/mcp", http_client=http_client) as streams,
        ):

            @asynccontextmanager
            async def transport():
                yield streams

            async with Client(transport(), mode=mode) as session:
                assert session.server_info.name == "bibr"
                assert session.protocol_version == "2025-11-25"

                tools = {t.name for t in (await session.list_tools()).tools}
                assert "chew_paper" in tools
                assert "load_paper" not in tools

                res = await session.call_tool(
                    "chew_paper", {"filename": "p.pdf", "content_base64": PDF_B64}
                )
                assert not res.is_error
                assert res.structured_content["paper_id"] == FIXTURE_ID

                papers = await session.call_tool("list_papers", {})
                assert len(papers.structured_content["result"]) == 1


@asynccontextmanager
async def _http_client(app):
    headers = {"Authorization": f"Bearer {_TEST_KEY}"}
    async with _asgi_factory(app)(headers=headers) as http_client:
        async with Client(
            streamable_http_client("http://testserver/mcp", http_client=http_client)
        ) as client:
            yield client


async def test_http_clients_sharing_bearer_key_have_separate_paper_stores():
    async with _mounted_app(_FakeTracker()) as app:
        async with _http_client(app) as first, _http_client(app) as second:
            _payload(
                await first.call_tool(
                    "chew_paper", {"filename": "p.pdf", "content_base64": PDF_B64}
                )
            )
            assert len(_payload(await first.call_tool("list_papers", {}))) == 1
            assert _payload(await second.call_tool("list_papers", {})) == []
            assert "unknown paper_id" in _error_text(
                await second.call_tool("get_metadata", {"paper_id": FIXTURE_ID})
            )
        async with _http_client(app) as reconnected:
            assert _payload(await reconnected.call_tool("list_papers", {})) == []


async def test_http_upload_above_sdk_default_body_limit():
    # The SDK defaults to 4 MiB of JSON; bibr's file limit must still apply.
    content = PDF_BYTES + b"x" * (4 * 1024 * 1024)
    tracker = _FakeTracker()
    async with _mounted_app(tracker, max_size=5 * 1024 * 1024, max_body=8 * 1024 * 1024) as app:
        async with _http_client(app) as client:
            result = _payload(
                await client.call_tool(
                    "chew_paper",
                    {"filename": "large.pdf", "content_base64": base64.b64encode(content).decode()},
                )
            )
            assert result["paper_id"] == FIXTURE_ID
            assert len(tracker.descriptors) == 1


# ---------------------------------------------------------------------------
# Keyless (loopback-only) serve: DNS-rebinding protection (x-security-7)
# ---------------------------------------------------------------------------

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "0"},
    },
}


@asynccontextmanager
async def _keyless_mounted_app(tracker: _FakeTracker):
    from bibr.config import Settings

    Settings.auth.api_key = None
    app = FastAPI()
    store = UploadStore.create(max_size=1_000_000, spool_memory_bytes=1024, stale_after_seconds=60)
    try:
        mount_mcp(app, Settings, upload_store=store, tracker=tracker)
        async with app.router.lifespan_context(app):
            yield app
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("base_url", "origin", "status"),
    [
        ("http://evil.example:8000", None, 421),  # a rebinding page's own host name
        ("http://127.0.0.1:8000", "https://evil.example", 403),
        ("http://127.0.0.1:8000", None, 200),
        ("http://localhost:8000", "http://localhost:8000", 200),
    ],
)
async def test_keyless_mount_admits_only_loopback_hosts_and_origins(base_url, origin, status):
    async with _keyless_mounted_app(_FakeTracker()) as app:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=base_url
        ) as client:
            headers = {"Accept": "application/json, text/event-stream"}
            if origin:
                headers["Origin"] = origin
            resp = await client.post("/mcp", json=_INITIALIZE, headers=headers)
            assert resp.status_code == status


async def test_keyless_mount_serves_a_local_mcp_client_end_to_end():
    async with _keyless_mounted_app(_FakeTracker()) as app:
        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
            ) as http_client,
            streamable_http_client("http://127.0.0.1:8000/mcp", http_client=http_client) as streams,
        ):

            @asynccontextmanager
            async def transport():
                yield streams

            async with Client(transport()) as session:
                res = await session.call_tool(
                    "chew_paper", {"filename": "p.pdf", "content_base64": PDF_B64}
                )
                assert res.structured_content["paper_id"] == FIXTURE_ID
