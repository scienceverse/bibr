"""Serve HTTP layer: event-loop work, readiness probes, path gates, bearer
bytes, multipart field budget and batch failure isolation."""

import asyncio
import gzip
import json
import threading

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from bibr.serve.batching import GpuBatcher

# ---------------------------------------------------------------- extract route


def _extract_app(result, *, gzip_middleware=False):
    from starlette.middleware.gzip import GZipMiddleware

    from bibr.serve.ingress import InferenceDispatchTracker, UploadStore, register_extract_route

    received = []

    async def dispatch(descriptor):
        received.append(descriptor)
        return result() if callable(result) else result

    app = FastAPI()
    if gzip_middleware:
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    store = UploadStore.create(max_size=1000, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    return app, store, tracker, received


def _close(store, tracker):
    asyncio.run(tracker.close())
    asyncio.run(store.close())


def _export(rows=200):
    return {
        "paper_id": "p",
        "metadata": {"title": "Über café ünïcode — ✓", "score": 0.25, "n": None},
        "text": [{"text_id": i, "text": f"Sentence {i} of the paper."} for i in range(rows)],
    }


def test_extract_renders_the_result_off_the_event_loop():
    """FastAPI encodes a returned dict on the loop thread, stalling every
    other request for as long as a multi-MB export takes to encode."""
    seen = {}

    class _ThreadProbe(dict):
        def items(self):
            seen["encoded_on"] = threading.get_ident()
            return super().items()

    def result():
        seen["loop"] = threading.get_ident()
        return _ThreadProbe(_export())

    app, store, tracker, _ = _extract_app(result)
    try:
        response = TestClient(app).post(
            "/papers/extract", files={"file": ("paper.pdf", b"%PDF", "application/pdf")}
        )
        assert response.status_code == 200
        assert response.json() == _export()
        assert seen["encoded_on"] != seen["loop"]
    finally:
        _close(store, tracker)


def test_extract_body_and_headers_match_fastapis_rendering():
    payload = _export()
    expected = JSONResponse(jsonable_encoder(payload)).body
    app, store, tracker, _ = _extract_app(payload)
    try:
        client = TestClient(app)
        plain = client.post(
            "/papers/extract",
            files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
            headers={"Accept-Encoding": "identity"},
        )
        assert plain.content == expected
        assert plain.headers["content-type"] == "application/json"
        assert plain.headers["content-length"] == str(len(expected))
        assert "content-encoding" not in plain.headers
    finally:
        _close(store, tracker)


def test_extract_gzips_a_large_body_itself_off_the_event_loop(monkeypatch):
    """GZipMiddleware compresses on the loop (level 9); the route now hands it
    a body that is already gzipped, compressed in the rendering thread."""
    payload = _export(rows=2000)
    expected = JSONResponse(jsonable_encoder(payload)).body
    threads = {}
    real_compress = gzip.compress

    def spy_compress(data, *args, **kwargs):
        threads["compress"] = threading.get_ident()
        return real_compress(data, *args, **kwargs)

    def result():
        threads["loop"] = threading.get_ident()
        return payload

    monkeypatch.setattr(gzip, "compress", spy_compress)
    # No GZipMiddleware here: a gzipped response can only come from the route.
    app, store, tracker, _ = _extract_app(result)
    try:
        response = TestClient(app).post(
            "/papers/extract",
            files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
            headers={"Accept-Encoding": "gzip"},
        )
        assert response.status_code == 200
        assert response.headers["content-encoding"] == "gzip"
        assert response.headers["vary"] == "Accept-Encoding"
        assert response.content == expected  # httpx decoded the gzip body
        assert threads["compress"] != threads["loop"]
    finally:
        _close(store, tracker)


def test_extract_leaves_a_small_body_to_the_middleware():
    payload = {"paper_id": "p"}
    app, store, tracker, _ = _extract_app(payload, gzip_middleware=True)
    try:
        response = TestClient(app).post(
            "/papers/extract",
            files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
            headers={"Accept-Encoding": "gzip"},
        )
        assert response.status_code == 200
        assert "content-encoding" not in response.headers
        assert response.content == JSONResponse(payload).body
    finally:
        _close(store, tracker)


@pytest.fixture
def server(monkeypatch):
    pytest.importorskip("litserve")
    from bibr.config import Settings
    from bibr.serve.app import build_server

    monkeypatch.setattr(Settings.auth, "api_key", None)
    server = build_server()
    yield server
    asyncio.run(server.app.state.inference_tracker.close())
    asyncio.run(server.app.state.upload_store.close())


def test_build_server_lowers_the_gzip_level(server):
    from starlette.middleware.gzip import GZipMiddleware

    from bibr.serve.ingress import GZIP_COMPRESSLEVEL

    (gzip_layer,) = [m for m in server.app.user_middleware if m.cls is GZipMiddleware]
    assert gzip_layer.kwargs["compresslevel"] == GZIP_COMPRESSLEVEL < 9
    assert gzip_layer.kwargs["minimum_size"] == 1000


def test_full_stack_passes_the_pregzipped_extract_body_through_once(server):
    payload = _export(rows=2000)

    async def dispatch(_descriptor):
        return payload

    server.app.state.inference_tracker._dispatch = dispatch
    response = TestClient(server.app, base_url="http://127.0.0.1:8000").post(
        "/papers/extract",
        files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
        headers={"Accept-Encoding": "gzip"},
    )
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert response.json() == payload  # compressed once, not twice
    assert "x-request-id" in response.headers


# ---------------------------------------------------------------- /ready


def _ready_app(monkeypatch, handler, ttl=60.0):
    import httpx

    import bibr.serve.app as app_mod
    from bibr.config import GlobalSettings

    # A long reuse window keeps these tests independent of machine speed.
    monkeypatch.setattr(app_mod, "_READINESS_PROBE_TTL_SECONDS", ttl, raising=False)

    class _PinnedClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _PinnedClient)

    class _Server:
        app = FastAPI()

    settings = GlobalSettings()
    app_mod._register_readiness_route(_Server(), settings)
    return _Server.app


def _ocr_handler(calls, gate=None):
    import httpx

    async def handler(request):
        calls.append(request.url.path)
        if gate is not None:
            await gate.wait()
        if request.url.path == "/health":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"data": [{"id": "glm-ocr"}]})

    return handler


def test_ready_reuses_one_probe_round_across_callers(monkeypatch):
    """Each anonymous /ready cost two OCR requests and the store pings."""
    calls = []
    client = TestClient(_ready_app(monkeypatch, _ocr_handler(calls)))
    for _ in range(5):
        assert client.get("/ready").status_code == 200
    assert calls == ["/health", "/v1/models"]


def test_concurrent_ready_calls_share_the_round_in_flight(monkeypatch):
    import httpx

    calls = []
    real_client = httpx.AsyncClient  # _ready_app pins AsyncClient to the OCR mock

    async def scenario():
        gate = asyncio.Event()
        app = _ready_app(monkeypatch, _ocr_handler(calls, gate))
        async with real_client(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            probes = [asyncio.create_task(client.get("/ready")) for _ in range(8)]
            await asyncio.sleep(0.2)
            in_flight = list(calls)
            gate.set()
            responses = await asyncio.gather(*probes)
        return in_flight, responses

    in_flight, responses = asyncio.run(scenario())
    assert in_flight.count("/health") <= 1
    assert [r.status_code for r in responses] == [200] * 8
    assert calls == ["/health", "/v1/models"]


def test_ready_probes_again_once_the_round_expires(monkeypatch):
    calls = []
    client = TestClient(_ready_app(monkeypatch, _ocr_handler(calls), ttl=0.0))
    client.get("/ready")
    client.get("/ready")
    assert calls == ["/health", "/v1/models"] * 2


def test_cached_round_still_answers_each_view_on_its_own(monkeypatch):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.auth, "api_key", "k" * 32)
    calls = []
    client = TestClient(_ready_app(monkeypatch, _ocr_handler(calls)))
    anonymous = client.get("/ready")
    authenticated = client.get("/ready", headers={"Authorization": f"Bearer {'k' * 32}"})
    again = client.get("/ready")
    assert anonymous.json() == again.json() == {"status": "ready"}
    assert list(authenticated.json()["checks"]) == ["ocr", "classifiers"]
    assert authenticated.json()["checks"]["ocr"] == "ok"
    assert calls == ["/health", "/v1/models"]


# ---------------------------------------------------------------- path gates


@pytest.mark.parametrize(
    ("path", "root_path"),
    [
        ("/papers/extract", ""),
        ("/api/papers/extract", "/api"),
        ("/api", "/api"),
        ("/apix/health", "/api"),
        ("/other/health", "/api"),
    ],
)
def test_route_path_matches_starlette_routing(path, root_path):
    from starlette._utils import get_route_path

    from bibr.serve.admission import route_path

    scope = {"type": "http", "path": path, "root_path": root_path}
    assert route_path(scope) == get_route_path(scope)


def test_auth_gate_ignores_a_path_forged_through_the_host_header(server, monkeypatch):
    """Starlette before 1.x built request.url from the raw Host header, so a
    Host of ``x/health?`` made any path read as the public /health."""
    from starlette.datastructures import URL
    from starlette.requests import HTTPConnection

    from bibr.config import Settings

    def url_from_raw_host(self):
        host = dict(self.scope["headers"]).get(b"host", b"").decode("latin-1")
        return URL(f"{self.scope['scheme']}://{host}{self.scope['path']}")

    monkeypatch.setattr(HTTPConnection, "url", property(url_from_raw_host))
    monkeypatch.setattr(Settings.auth, "api_key", "k" * 32)
    client = TestClient(server.app, raise_server_exceptions=False)
    resp = client.get("/papers/jobs/abc", headers={"Host": "x/health?"})
    assert resp.status_code == 401


def test_gates_use_the_routed_path_under_a_root_path(server, monkeypatch):
    from bibr.config import Settings

    async def dispatch(_descriptor):
        return {"ok": True}

    server.app.state.inference_tracker._dispatch = dispatch
    monkeypatch.setattr(Settings.auth, "api_key", "k" * 32)
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8000",
        root_path="/api",
        raise_server_exceptions=False,
    )
    # The probe stays public when the app is mounted under a root path.
    assert client.get("/api/health").status_code != 401

    # Upload admission covers the routed /papers/extract too.
    gate = server.app.state.upload_admission_gate
    monkeypatch.setattr(gate.spool, "limit", 1)
    monkeypatch.setattr(gate.spool, "active", 1)
    resp = client.post(
        "/api/papers/extract",
        files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
        headers={"Authorization": f"Bearer {'k' * 32}"},
    )
    assert resp.status_code == 429


# ---------------------------------------------------------------- bearer bytes


@pytest.fixture
def _restore_api_key():
    from bibr.config import Settings

    original = Settings.auth.api_key
    yield Settings
    Settings.auth.api_key = original


def _bearer_status(settings, key, header_bytes):
    """Status for a request whose Authorization header carries exactly these
    bytes (TestClient would re-encode a non-UTF-8 header as UTF-8)."""
    from fastapi import Depends

    from bibr.serve.auth import require_api_key

    settings.auth.api_key = key
    app = FastAPI()

    @app.get("/protected")
    async def protected(_=Depends(require_api_key)):
        return {"ok": True}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/protected",
        "raw_path": b"/protected",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1"), (b"authorization", b"Bearer " + header_bytes)],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8000),
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    body = b"".join(m.get("body", b"") for m in messages[1:])
    return messages[0]["status"], json.loads(body)


@pytest.mark.parametrize("key", ["ä" * 40, "ключ" * 10, "k€" * 20])
def test_non_ascii_key_sent_as_utf8_is_accepted(_restore_api_key, key):
    """Starlette hands the header over latin-1-decoded; re-encoding it as
    UTF-8 double-encoded every non-ASCII byte a curl client sent."""
    assert _bearer_status(_restore_api_key, key, key.encode("utf-8")) == (200, {"ok": True})


def test_non_ascii_key_sent_as_latin1_still_matches(_restore_api_key):
    key = "ä" * 40
    assert _bearer_status(_restore_api_key, key, key.encode("latin-1")) == (200, {"ok": True})


@pytest.mark.parametrize(
    "sent",
    [("ä" * 39 + "a").encode("utf-8"), b"\xff" * 40, ("ä" * 40).encode("utf-16-le")],
)
def test_wrong_non_ascii_token_is_a_401(_restore_api_key, sent):
    status, body = _bearer_status(_restore_api_key, "ä" * 40, sent)
    assert (status, body) == (401, {"detail": "Invalid bearer token"})


def test_in_process_token_outside_latin1_is_compared_not_raised(_restore_api_key):
    from bibr.serve.auth import check_bearer

    _restore_api_key.auth.api_key = "k€" * 20
    assert check_bearer("Bearer " + "k€" * 20) is None
    assert check_bearer("Bearer \ud800" + "k" * 39) == "Invalid bearer token"


# ---------------------------------------------------------------- multipart fields


_EVERY_OPTION = {
    "start_page": "0",
    "end_page": "1",
    "include_figures": "true",
    "include_regions": "false",
    "include_region_meta": "true",
    "crossref": "false",
    "consolidate": "fill",
    "refs": "ner",
    "ref_seg": "geom",
}


@pytest.mark.parametrize("unknown", [2, 16])
def test_every_option_fits_beside_unknown_fields(unknown):
    """Unknown names are ignored but were counted against a cap of exactly the
    known options, so a newer client's extra field turned a full request into
    a 400."""
    from bibr.serve.ingress import _FORM_OPTION_NAMES

    assert set(_EVERY_OPTION) == set(_FORM_OPTION_NAMES)
    app, store, tracker, received = _extract_app({"ok": True})
    try:
        response = TestClient(app).post(
            "/papers/extract",
            files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
            data={**_EVERY_OPTION, **{f"future_{i}": "x" for i in range(unknown)}},
        )
        assert response.status_code == 200, response.text
        (descriptor,) = received
        assert {name: descriptor[name] for name in _EVERY_OPTION} == _EVERY_OPTION
        assert not any(name.startswith("future_") for name in descriptor)
    finally:
        _close(store, tracker)


def test_job_route_takes_every_option_beside_unknown_fields():
    from bibr.serve.ingress import UploadStore
    from bibr.serve.jobs import MemoryJobStore, register_job_routes

    class _Tracker:
        async def submit(self, descriptor, request_state=None, **_kwargs):
            return {"ok": True}

        async def discard(self, descriptor):
            pass

    app = FastAPI()
    store = UploadStore.create(max_size=1000, spool_memory_bytes=4, stale_after_seconds=120)
    register_job_routes(app, store=MemoryJobStore(), upload_store=store, tracker=_Tracker())
    try:
        response = TestClient(app).post(
            "/papers/jobs",
            files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
            data={**_EVERY_OPTION, "future_a": "x", "future_b": "y"},
        )
        assert response.status_code == 202, response.text
    finally:
        asyncio.run(store.close())


# ---------------------------------------------------------------- GpuBatcher


async def test_one_poison_item_fails_only_its_own_caller():
    """A batch mixes concurrent requests' pages; one bad page used to fail
    every request that shared the batch."""
    calls: list[list[str]] = []

    def fn(items):
        calls.append(list(items))
        if "poison" in items:
            raise ValueError("bad page")
        return [item.upper() for item in items]

    batcher = GpuBatcher(fn, max_batch_size=8, batch_timeout=0.05)
    try:
        results = await asyncio.gather(
            batcher.submit("a"),
            batcher.submit("poison"),
            batcher.submit("b"),
            return_exceptions=True,
        )
    finally:
        await batcher.close()
    assert results[0] == "A"
    assert isinstance(results[1], ValueError)
    assert results[2] == "B"
    # One batch call, then each item once on its own: bounded by the batch.
    assert calls == [["a", "poison", "b"], ["a"], ["poison"], ["b"]]


async def test_a_failed_single_item_batch_is_not_retried():
    calls = []

    def fn(items):
        calls.append(list(items))
        raise ValueError("bad page")

    batcher = GpuBatcher(fn, max_batch_size=8, batch_timeout=0.0)
    try:
        with pytest.raises(ValueError, match="bad page"):
            await batcher.submit("only")
    finally:
        await batcher.close()
    assert calls == [["only"]]


async def test_retry_skips_a_caller_that_already_gave_up():
    calls: list[list[str]] = []
    started = threading.Event()
    release = threading.Event()

    def fn(items):
        calls.append(list(items))
        if len(items) > 1:
            started.set()
            release.wait(5)
            raise ValueError("batch failed")
        return list(items)

    batcher = GpuBatcher(fn, max_batch_size=8, batch_timeout=0.05)
    try:
        keep = asyncio.ensure_future(batcher.submit("keep"))
        gone = asyncio.ensure_future(batcher.submit("gone"))
        await asyncio.to_thread(started.wait, 5)
        gone.cancel()
        release.set()
        assert await keep == "keep"
        with pytest.raises(asyncio.CancelledError):
            await gone
    finally:
        await batcher.close()
    assert calls == [["keep", "gone"], ["keep"]]


def test_render_json_response_is_a_plain_function():
    """The route runs it with asyncio.to_thread, so it must not need a loop."""
    from bibr.serve.ingress import render_json_response

    response = render_json_response({"a": [1, 2.5, None, "ü"]}, "")
    assert json.loads(response.body) == {"a": [1, 2.5, None, "ü"]}
