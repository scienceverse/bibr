"""``GET /`` points at the docs, the ready line names the URLs, SIGTERM shuts down."""

from __future__ import annotations

import os
import signal

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from bibr.serve.app import INDEX_PAYLOAD, _display_base_url, _ready_line


def _client(monkeypatch, api_key):
    from bibr.config import Settings
    from bibr.serve.app import build_server

    monkeypatch.setattr(Settings.auth, "api_key", api_key)
    return TestClient(
        build_server().app, base_url="http://127.0.0.1:8000", raise_server_exceptions=False
    )


def test_root_points_at_the_docs_and_probes(monkeypatch):
    response = _client(monkeypatch, None).get("/")
    assert response.status_code == 200
    assert response.json() == INDEX_PAYLOAD
    assert response.json()["docs"] == "/docs"


def test_root_stays_behind_the_token_when_one_is_set(monkeypatch):
    client = _client(monkeypatch, "sk_int")
    assert client.get("/").status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer sk_int"}).json() == INDEX_PAYLOAD


def test_only_one_root_route(monkeypatch):
    app = _client(monkeypatch, None).app
    assert [getattr(r, "path", None) for r in app.router.routes].count("/") == 1


@pytest.mark.parametrize(
    ("host", "url"),
    [
        ("127.0.0.1", "http://127.0.0.1:8000"),
        ("0.0.0.0", "http://127.0.0.1:8000"),  # noqa: S104 — a display name
        ("::", "http://[::1]:8000"),
    ],
)
def test_display_base_url(host, url):
    assert _display_base_url(host, 8000) == url


def test_ready_line_names_the_docs_and_health():
    line = _ready_line("http://127.0.0.1:8012")
    assert line.startswith("bibr serve is ready: http://127.0.0.1:8012")
    assert "http://127.0.0.1:8012/docs" in line
    assert "http://127.0.0.1:8012/health" in line


@pytest.mark.skipif(os.name == "nt", reason="sends SIGTERM to a POSIX process")
def test_sigterm_stops_the_server_and_the_ready_line_came_first():
    from tests.serve.test_worker_contract import _stub_server_process

    with _stub_server_process() as (process, _base):
        process.send_signal(signal.SIGTERM)
        # uvicorn re-raises the signal once it has shut down.
        assert process.wait(timeout=30) == -signal.SIGTERM
        output = process.stdout.read()
    assert "bibr serve is ready: http://127.0.0.1:" in output
