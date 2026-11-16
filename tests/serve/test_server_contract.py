"""HTTP behaviour of the built ``bibr serve`` app that does not need a worker.

Pins what the server framework adds around bibr's own routes: the request
body cap (413 ``Payload too large``, with or without a declared length), gzip
from 1000 bytes, the FastAPI metadata routes behind the auth gate, and 404
for any other path, the old private dispatch path included.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")

from fastapi import Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def server(monkeypatch):
    from bibr.config import Settings
    from bibr.serve.app import build_server

    monkeypatch.setattr(Settings.auth, "api_key", None)
    monkeypatch.setattr(Settings.pipeline, "max_file_size", 1000)
    monkeypatch.setattr(Settings.pipeline, "multipart_overhead_bytes", 1000)
    server = build_server()
    yield server
    asyncio.run(server.app.state.inference_tracker.close())
    asyncio.run(server.app.state.upload_store.close())


def _body_route(server):
    @server.app.post("/contract-body")
    async def body(request: Request):
        return {"size": len(await request.body())}


def test_a_body_over_the_cap_is_a_413(server):
    _body_route(server)
    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    assert client.post("/contract-body", content=b"x" * 2000).json() == {"size": 2000}
    response = client.post("/contract-body", content=b"x" * 2001)
    assert response.status_code == 413
    assert response.json() == {"detail": "Payload too large"}


def test_a_body_over_the_cap_without_a_length_is_a_413(server):
    _body_route(server)
    client = TestClient(server.app, base_url="http://127.0.0.1:8000")

    def chunks():
        for _ in range(3):
            yield b"x" * 1000

    response = client.post("/contract-body", content=chunks())
    assert response.status_code == 413
    assert response.json() == {"detail": "Payload too large"}


def test_responses_from_1000_bytes_are_gzipped(server):
    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    response = client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert "/papers/extract" in response.json()["paths"]

    small = client.get("/nothing-here", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in small.headers


def test_other_paths_are_404(server):
    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    for path in ("/nothing-here", "/_bibr/inference"):
        response = client.post(path, json={})
        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}


def test_metadata_routes_need_the_bearer_when_auth_is_on(server, monkeypatch):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.auth, "api_key", "k" * 32)
    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    for path in ("/openapi.json", "/docs"):
        assert client.get(path).status_code == 401
    assert client.get("/openapi.json", headers={"Authorization": f"Bearer {'k' * 32}"}).is_success
