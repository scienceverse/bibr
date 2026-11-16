"""``GET /`` points at the docs, the ready line names the URLs, SIGTERM shuts down.

The SIGTERM test stands in for LitServe's run loop: the handler must turn the
signal into the ``KeyboardInterrupt`` LitServe already shuts its workers down on,
and put the previous handler back afterwards.
"""

from __future__ import annotations

import os
import signal
import time

import pytest
from fastapi.testclient import TestClient

from bibr.serve.app import (
    INDEX_PAYLOAD,
    _announce_when_listening,
    _display_base_url,
    _sigterm_stops_gracefully,
)


def _client(monkeypatch, api_key):
    pytest.importorskip("litserve")
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


def test_ready_line_follows_the_api_server_start(capsys):
    class _Server:
        def _start_server(self, *args, **kwargs):
            print("started", flush=True)
            return {0: "worker"}

    server = _Server()
    _announce_when_listening(server, "http://127.0.0.1:8012")
    assert server._start_server(8012) == {0: "worker"}
    captured = capsys.readouterr()
    assert captured.out == "started\n"
    assert captured.err.startswith("bibr serve is ready: http://127.0.0.1:8012")
    assert "http://127.0.0.1:8012/docs" in captured.err


def test_a_server_without_the_hook_is_left_alone():
    class _Server:
        pass

    _announce_when_listening(_Server(), "http://127.0.0.1:8012")  # no error


def test_sigterm_becomes_a_keyboard_interrupt_and_the_handler_is_restored():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt), _sigterm_stops_gracefully():
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)  # the handler interrupts this
        pytest.fail("SIGTERM was not turned into KeyboardInterrupt")
    assert signal.getsignal(signal.SIGTERM) == before
