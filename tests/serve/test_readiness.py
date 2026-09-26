"""Readiness payload info-disclosure gating (audit L1)."""

import pytest

pytest.importorskip("fastapi")


def _ocr_transport(health=200, models_status=200, model_ids=("glm-ocr",)):
    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(health, json={})
        if request.url.path == "/v1/models":
            return httpx.Response(models_status, json={"data": [{"id": mid} for mid in model_ids]})
        return httpx.Response(404, json={})

    return httpx.MockTransport(handler)


def _ready_client(monkeypatch, settings, transport):
    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import bibr.serve.app as app_mod

    class _Server:
        def __init__(self):
            self.app = FastAPI()

    class _PinnedClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(transport=transport, timeout=5.0)

    monkeypatch.setattr(httpx, "AsyncClient", _PinnedClient)
    server = _Server()
    app_mod._register_readiness_route(server, settings)
    return TestClient(server.app)


def test_readiness_payload_hides_detail_when_not_authenticated():
    from bibr.config import Settings
    from bibr.serve.app import readiness_payload

    checks = {"ocr": "ok", "redis": "ok"}
    full = readiness_payload("ready", checks, Settings, include_detail=True)
    assert full["checks"] == checks
    assert "build_sha" in full

    minimal = readiness_payload("not_ready", checks, Settings, include_detail=False)
    assert minimal == {"status": "not_ready"}
    assert "build_sha" not in minimal
    assert "checks" not in minimal


def test_wildcard_cors_disables_credentials():
    """`*` origins + credentials would reflect any Origin — credentials off (L3)."""
    from bibr.serve.app import _safe_cors_credentials

    assert _safe_cors_credentials(["*"], True) is False
    assert _safe_cors_credentials(["https://app.example"], True) is True
    assert _safe_cors_credentials(["*"], False) is False


def test_ready_is_not_ready_when_the_alias_is_not_served(monkeypatch):
    """/health 200 while /v1/models lacks the alias: every PDF would block
    then 502, so /ready must fail with the missing alias named."""
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    client = _ready_client(
        monkeypatch, settings, _ocr_transport(models_status=200, model_ids=("paddle-ocr-vl-1.6",))
    )
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["checks"]["ocr"] == "model_missing (glm-ocr)"


def test_ready_reports_unauthorized_when_models_needs_a_key(monkeypatch):
    """/health answers without auth; a 401 on /v1/models means OCR_API_KEY
    is wrong and must be reported as such, not as a model problem."""
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    client = _ready_client(monkeypatch, settings, _ocr_transport(models_status=401))
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["checks"]["ocr"] == "unauthorized (401; check OCR_API_KEY)"


def test_ready_stays_ready_when_the_alias_is_served(monkeypatch):
    """Guard: a healthy server listing the expected alias still reads ready."""
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    client = _ready_client(monkeypatch, settings, _ocr_transport())
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["checks"]["ocr"] == "ok"
