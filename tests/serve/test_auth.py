import pytest

pytest.importorskip("fastapi")
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _restore_api_key():
    from bibr.config import Settings

    original = Settings.auth.api_key
    yield
    Settings.auth.api_key = original


def _build_app(api_key: str | None):
    from bibr.config import Settings
    from bibr.serve.auth import require_api_key

    Settings.auth.api_key = api_key  # mutated for test only
    app = FastAPI()

    @app.get("/protected")
    async def protected(_=Depends(require_api_key)):
        return {"ok": True}

    return app


def test_no_key_configured_allows_unauthenticated():
    app = _build_app(api_key=None)
    client = TestClient(app)
    resp = client.get("/protected")
    assert resp.status_code == 200


def test_key_configured_rejects_missing_header():
    app = _build_app(api_key="sk_test")
    client = TestClient(app)
    resp = client.get("/protected")
    assert resp.status_code == 401


def test_key_configured_rejects_wrong_token():
    app = _build_app(api_key="sk_test")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_key_configured_accepts_correct_token():
    app = _build_app(api_key="sk_test")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer sk_test"})
    assert resp.status_code == 200


def test_constant_time_compare_used(monkeypatch):
    """Confirms we don't use plain ==, which would be timing-vulnerable."""
    import bibr.serve.auth as mod

    calls = []
    real = mod.hmac.compare_digest

    def spy(a, b):
        calls.append(True)
        return real(a, b)

    monkeypatch.setattr(mod.hmac, "compare_digest", spy)
    app = _build_app(api_key="sk_test")
    client = TestClient(app)
    client.get("/protected", headers={"Authorization": "Bearer sk_test"})
    assert calls, "hmac.compare_digest must be used for token comparison"


def test_non_ascii_token_is_rejected_not_a_server_error():
    """A high byte in the header must be a 401, not a TypeError-driven 500."""
    from bibr.serve.auth import check_bearer

    app = _build_app(api_key="sk_test")
    assert check_bearer("Bearer sk_t\u00e9st") == "Invalid bearer token"

    client = TestClient(app)
    # Raw bytes bypass httpx's ASCII header validation, as a hostile or
    # misconfigured client would.
    r = client.get("/protected", headers={b"authorization": "Bearer sk_t\u00e9st".encode()})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid bearer token"


def test_empty_string_key_treated_as_disabled():
    app = _build_app(api_key="")
    client = TestClient(app)
    resp = client.get("/protected")
    assert resp.status_code == 200


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "[::1]", "localhost"])
def test_loopback_bind_allows_auth_disabled(host):
    from bibr.serve.auth import validate_bind_auth

    validate_bind_auth(host, None)


@pytest.mark.parametrize(  # noqa: S104 - intentionally exercise unsafe bind addresses
    "host",
    # 127.0.0.2 is loopback, but a keyless server answers only the names its
    # clients send for 127.0.0.1, ::1 and localhost (check_keyless_request).
    ["0.0.0.0", "::", "api.example.com", "192.168.1.10", "127.0.0.2"],  # noqa: S104
)
def test_network_bind_requires_auth(host):
    from bibr.serve.auth import validate_bind_auth

    with pytest.raises(ValueError, match="Refusing to bind"):
        validate_bind_auth(host, None)


def test_network_bind_allows_configured_auth():
    from bibr.serve.auth import validate_bind_auth

    validate_bind_auth("0.0.0.0", "a" * 32)  # noqa: S104 - authenticated bind is allowed


def test_network_bind_rejects_weak_key():
    from bibr.serve.auth import validate_bind_auth

    with pytest.raises(ValueError, match="at least 32"):
        validate_bind_auth("0.0.0.0", "too-short")  # noqa: S104


def test_papers_extract_requires_auth_when_configured(monkeypatch):
    pytest.importorskip("fastapi")

    from bibr.config import Settings

    monkeypatch.setattr(Settings.auth, "api_key", "sk_int")

    from bibr.serve.app import build_server

    server = build_server()
    client = TestClient(server.app)

    # Multipart with no auth header → 401
    resp = client.post(
        "/papers/extract",
        files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 401


def test_private_inference_route_is_never_http_accessible(monkeypatch):
    pytest.importorskip("fastapi")

    import asyncio

    from bibr.config import Settings
    from bibr.serve.app import build_server

    monkeypatch.setattr(Settings.auth, "api_key", None)
    server = build_server()
    try:
        # The old private dispatch path is no route at all now; a keyless
        # server answers loopback Hosts only, hence the base URL.
        response = TestClient(
            server.app, base_url="http://127.0.0.1:8000", raise_server_exceptions=False
        ).post(
            "/_bibr/inference",
            json={"upload_id": "0" * 32},
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}
    finally:
        asyncio.run(server.app.state.inference_tracker.close())
        asyncio.run(server.app.state.upload_store.close())


@pytest.mark.parametrize("failure_point", ["api", "worker", "metering"])
def test_build_server_construction_failure_removes_owned_upload_root(
    monkeypatch,
    failure_point,
):
    """Catches startup retries leaking a private temporary root each time."""
    import asyncio

    from bibr.serve import app as app_mod
    from bibr.serve import ingress as ingress_mod
    from bibr.serve import worker as worker_mod
    from bibr.serve.deployments import pipeline as pipeline_mod

    stores = []
    real_create = ingress_mod.UploadStore.create

    def create_store(**kwargs):
        store = real_create(**kwargs)
        stores.append(store)
        return store

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"{failure_point} construction failed")

    monkeypatch.setattr(ingress_mod.UploadStore, "create", create_store)
    if failure_point == "api":
        monkeypatch.setattr(pipeline_mod, "BibrPipelineAPI", fail)
    elif failure_point == "worker":
        monkeypatch.setattr(worker_mod, "InferenceWorker", fail)
    else:
        monkeypatch.setattr(app_mod, "_configure_metering_logging", fail)

    try:
        with pytest.raises(RuntimeError, match=f"{failure_point} construction failed"):
            app_mod.build_server()

        assert len(stores) == 1
        assert not stores[0].root.exists()
    finally:
        for store in stores:
            if store.root.exists():
                asyncio.run(store.close())


def _body_cap(app) -> int:
    from bibr.serve.ingress import BodySizeLimit

    (layer,) = [m for m in app.user_middleware if m.cls is BodySizeLimit]
    return layer.kwargs["max_size"]


def test_build_server_raises_the_body_cap_to_fit_base64_uploads_when_mcp_is_on(monkeypatch):
    pytest.importorskip("mcp")

    import asyncio

    from bibr.config import Settings
    from bibr.serve.admission import base64_envelope
    from bibr.serve.app import build_server

    monkeypatch.setattr(Settings.mcp, "enabled", True)
    server = build_server()
    try:
        expected = (
            base64_envelope(Settings.pipeline.max_file_size)
            + Settings.pipeline.multipart_overhead_bytes
        )
        assert _body_cap(server.app) == expected
        assert expected > Settings.pipeline.max_file_size * 4 // 3
        assert server.app.state.upload_admission_gate.limit == Settings.pipeline.max_active_uploads
    finally:
        asyncio.run(server.app.state.inference_tracker.close())


def test_build_server_preserves_payload_and_worker_runtime_options(monkeypatch):
    import asyncio

    from bibr.config import Settings
    from bibr.serve.app import build_server

    monkeypatch.setattr(Settings.pipeline, "restart_workers", True)
    server = build_server()
    try:
        assert _body_cap(server.app) == (
            Settings.pipeline.max_file_size + Settings.pipeline.multipart_overhead_bytes
        )
        assert server.worker._restart is True
        assert server.worker._timeout == Settings.pipeline.timeout
    finally:
        asyncio.run(server.app.state.inference_tracker.close())
        asyncio.run(server.app.state.upload_store.close())


@pytest.mark.parametrize("jobs_enabled", [False, True])
def test_main_runs_one_api_server_with_uvicorn_logging_off(monkeypatch, jobs_enabled):
    """Catches main handing uvicorn its own log config (records would skip the scrubber)."""
    import sys

    from bibr.config import Settings
    from bibr.serve import app as app_mod
    from bibr.serve import auth as auth_mod

    captured = {}

    class _FakeServer:
        def run(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(Settings.jobs, "enabled", jobs_enabled)
    monkeypatch.setattr(app_mod, "build_server", lambda: _FakeServer())
    monkeypatch.setattr(auth_mod, "validate_bind_auth", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["bibr", "--host", "127.0.0.1", "--port", "8123"])

    app_mod.main()

    assert captured == {
        "host": "127.0.0.1",
        "port": 8123,
        "log_level": Settings.SERVE_LOG_LEVEL,
        "log_config": None,
    }


def test_build_server_asgi_lifespan_runs_all_cleanup_in_order(monkeypatch):
    """Catches router shutdown handlers being bypassed by bibr's lifespan wrapper."""
    import asyncio
    import io

    from starlette.datastructures import UploadFile

    from bibr.config import Settings
    from bibr.serve import jobs as jobs_mod
    from bibr.serve.app import build_server

    events = []

    class FakeJobDispatcher:
        async def close(self):
            events.append("job_dispatcher")

    monkeypatch.setattr(Settings.jobs, "enabled", True)
    monkeypatch.setattr(jobs_mod, "JobDispatcher", FakeJobDispatcher)

    server = build_server()
    tracker = server.app.state.inference_tracker
    upload_store = server.app.state.upload_store
    upload_root = upload_store.root
    asyncio.run(upload_store.persist(UploadFile(file=io.BytesIO(b"payload"), filename="paper.pdf")))
    original_tracker_close = tracker.close
    original_store_close = upload_store.close

    async def tracked_tracker_close():
        events.append("tracker")
        await original_tracker_close()

    async def tracked_store_close():
        events.append("store")
        await original_store_close()

    tracker.close = tracked_tracker_close
    upload_store.close = tracked_store_close
    server.app.state.job_dispatcher = FakeJobDispatcher()

    try:
        with TestClient(server.app):
            events.append("serving")

        observed_events = list(events)
        root_exists = upload_root.exists()
    finally:
        if upload_root.exists():
            asyncio.run(tracker.close())
            asyncio.run(upload_store.close())

    assert observed_events == [
        "serving",
        "job_dispatcher",
        "tracker",
        "store",
    ]
    assert root_exists is False


class TestNonInferenceRoutesGated:
    """/openapi.json and /docs leak deployment details — they must be gated by
    the same bearer token as the inference route."""

    @staticmethod
    def _server_client(monkeypatch, api_key):
        pytest.importorskip("fastapi")
        from bibr.config import Settings

        monkeypatch.setattr(Settings.auth, "api_key", api_key)
        from bibr.serve.app import build_server

        # raise_server_exceptions=False: these tests only care about the auth
        # gate in front of each route. The loopback base URL is what a keyless
        # (loopback-only) server's clients send as Host.
        return TestClient(
            build_server().app, base_url="http://127.0.0.1:8000", raise_server_exceptions=False
        )

    def test_openapi_requires_auth_when_configured(self, monkeypatch):
        client = self._server_client(monkeypatch, "sk_int")
        assert client.get("/openapi.json").status_code == 401

    def test_docs_requires_auth_when_configured(self, monkeypatch):
        client = self._server_client(monkeypatch, "sk_int")
        assert client.get("/docs").status_code == 401

    def test_unknown_routes_are_gated_too(self, monkeypatch):
        client = self._server_client(monkeypatch, "sk_int")
        # LitServe's old /info route is gone; any path but the probes is gated.
        assert client.get("/info").status_code == 401

    def test_openapi_with_token_ok(self, monkeypatch):
        client = self._server_client(monkeypatch, "sk_int")
        resp = client.get("/openapi.json", headers={"Authorization": "Bearer sk_int"})
        assert resp.status_code == 200

    def test_health_stays_public(self, monkeypatch):
        client = self._server_client(monkeypatch, "sk_int")
        # Liveness probes carry no credentials — must never be 401.
        assert client.get("/health").status_code != 401

    def test_openapi_open_when_auth_disabled(self, monkeypatch):
        client = self._server_client(monkeypatch, None)
        assert client.get("/openapi.json").status_code == 200


class TestAuth401CorsHeaders:
    """A browser client on an allowed origin must be able to *read* the 401 —
    without Access-Control-Allow-Origin on the response, fetch() surfaces a
    network error instead of the auth failure."""

    @staticmethod
    def _server_client(monkeypatch, origins, allow_credentials=False):
        pytest.importorskip("fastapi")
        from bibr.config import Settings

        monkeypatch.setattr(Settings.auth, "api_key", "sk_int")
        monkeypatch.setattr(Settings.cors, "origins", origins)
        monkeypatch.setattr(Settings.cors, "allow_credentials", allow_credentials)
        from bibr.serve.app import build_server

        return TestClient(build_server().app, raise_server_exceptions=False)

    def test_401_carries_www_authenticate(self, monkeypatch):
        client = self._server_client(monkeypatch, origins=["https://app.example.com"])
        resp = client.get("/openapi.json", headers={"Origin": "https://app.example.com"})
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_401_echoes_allowed_origin(self, monkeypatch):
        client = self._server_client(
            monkeypatch, origins=["https://app.example.com"], allow_credentials=True
        )
        resp = client.get("/openapi.json", headers={"Origin": "https://app.example.com"})
        assert resp.status_code == 401
        assert resp.headers.get("access-control-allow-origin") == "https://app.example.com"

    def test_401_does_not_echo_disallowed_origin(self, monkeypatch):
        client = self._server_client(monkeypatch, origins=["https://app.example.com"])
        resp = client.get("/openapi.json", headers={"Origin": "https://evil.example.com"})
        assert resp.status_code == 401
        assert resp.headers.get("access-control-allow-origin") is None

    def test_401_without_cors_configured_has_no_cors_headers(self, monkeypatch):
        client = self._server_client(monkeypatch, origins=[])
        resp = client.get("/openapi.json", headers={"Origin": "https://app.example.com"})
        assert resp.status_code == 401
        assert resp.headers.get("access-control-allow-origin") is None

    def test_preflight_not_blocked_by_auth(self, monkeypatch):
        # Preflights carry no credentials; they must short-circuit in the
        # CORS layer rather than 401 in the auth gate.
        client = self._server_client(monkeypatch, origins=["https://app.example.com"])
        resp = client.options(
            "/papers/extract",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://app.example.com"


def test_short_loopback_key_warns_but_does_not_raise(caplog):
    """A short key on a loopback bind is allowed (dev) but warned (audit L4)."""
    import logging

    from bibr.serve.auth import validate_bind_auth

    with caplog.at_level(logging.WARNING):
        validate_bind_auth("127.0.0.1", "short-key")  # must not raise
    assert any("32 char" in r.message or "strong key" in r.message for r in caplog.records)


class TestKeylessLoopbackGate:
    """serve-7 / x-security-7: without AUTH_API_KEY, loopback is the only
    boundary, and a web page in the operator's browser can cross it — a
    cross-site form POST needs no preflight, a DNS-rebinding page arrives under
    its own Host. Legitimate local clients must keep working."""

    _CROSS_SITE = {"detail": "Cross-site request refused: bibr serve has no AUTH_API_KEY"}
    _BAD_HOST = {
        "detail": "Host not allowed: without AUTH_API_KEY bibr serve answers only loopback names"
    }

    @pytest.fixture
    def server(self, monkeypatch):
        pytest.importorskip("fastapi")
        import asyncio

        from bibr.config import Settings
        from bibr.serve.app import build_server

        monkeypatch.setattr(Settings.auth, "api_key", None)
        monkeypatch.setattr(Settings.cors, "origins", ["https://ui.example.org"])
        server = build_server()
        yield server
        asyncio.run(server.app.state.inference_tracker.close())
        asyncio.run(server.app.state.upload_store.close())

    @staticmethod
    def _client(server, base_url="http://127.0.0.1:8000"):
        return TestClient(server.app, base_url=base_url, raise_server_exceptions=False)

    @pytest.mark.parametrize(
        "headers",
        [
            {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
            {"Origin": "null"},
            {"Sec-Fetch-Site": "cross-site"},
        ],
    )
    def test_cross_site_form_post_is_refused(self, server, headers):
        resp = self._client(server).post("/papers/extract", data={"refs": "llm"}, headers=headers)
        assert (resp.status_code, resp.json()) == (403, self._CROSS_SITE)

    @pytest.mark.parametrize("path", ["/openapi.json", "/health", "/papers/jobs/abc"])
    def test_rebound_host_is_refused_on_every_path(self, server, path):
        resp = self._client(server, "http://evil.example:8000").get(path)
        assert (resp.status_code, resp.json()) == (421, self._BAD_HOST)

    @pytest.mark.parametrize("host", ["127.0.0.2:8000", "bibr.localhost:8000", "localhost."])
    def test_only_the_names_the_mcp_transport_admits_pass(self, server, host):
        """One host policy for REST and /mcp (the SDK's allowlist is exact)."""
        resp = self._client(server).get("/openapi.json", headers={"Host": host})
        assert (resp.status_code, resp.json()) == (421, self._BAD_HOST)

    @pytest.mark.parametrize("origin", ["https://evil.example", "ftp://127.0.0.1"])
    def test_a_wildcard_cors_setting_admits_no_origin_without_a_key(
        self, server, monkeypatch, origin
    ):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.cors, "origins", ["*"])
        resp = self._client(server).post(
            "/papers/extract", data={"refs": "llm"}, headers={"Origin": origin}
        )
        assert (resp.status_code, resp.json()) == (403, self._CROSS_SITE)

    @pytest.mark.parametrize(
        ("host", "origin"),
        [
            ("127.0.0.1:8000", "http://127.0.0.1:8000"),  # /docs "Try it out"
            ("localhost:8000", "http://localhost:8000"),
            ("[::1]:8000", "http://[::1]:8000"),
            ("127.0.0.1:8000", "https://ui.example.org"),  # listed in CORS_ORIGINS
            ("localhost", None),  # curl, scripts
        ],
    )
    def test_the_operators_own_clients_pass_the_gate(self, server, host, origin):
        headers = {"Host": host, **({"Origin": origin} if origin else {})}
        resp = self._client(server).post("/papers/extract", data={"refs": "llm"}, headers=headers)
        # Past the gate: the ingress itself rejects the empty upload.
        assert (resp.status_code, resp.json()) == (400, {"detail": "Invalid multipart body"})

    def test_bibr_batch_client_passes_the_gate(self, server):
        import asyncio

        import httpx

        from bibr.batch.remote import RemoteExecutor, RemoteOptions

        async def poll_unknown_job():
            executor = RemoteExecutor(
                RemoteOptions(serve_url="http://127.0.0.1:8000"),
                transport=httpx.ASGITransport(app=server.app),
            )
            async with executor.client() as client:
                return await client.get("/papers/jobs/0123456789abcdef")

        resp = asyncio.run(poll_unknown_job())
        assert (resp.status_code, resp.json()) == (404, {"detail": "job not found"})

    def test_a_key_makes_the_bearer_token_the_boundary(self, server, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.auth, "api_key", "k" * 32)
        resp = self._client(server, "https://bibr.example.org").get(
            "/openapi.json",
            headers={"Authorization": f"Bearer {'k' * 32}", "Origin": "https://evil.example"},
        )
        assert resp.status_code == 200
