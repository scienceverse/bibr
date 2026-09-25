"""Managed-server tests must pass with bibr's default ports occupied (x-tests-4).

A leftover ``bibr chew --llm local`` server on :8773/:8772 used to fail 18
tests and stall the run for 2 minutes, because the tests stubbed the HTTP
layer but never the guard's real TCP probe. The autouse fixture in
``tests/local/conftest.py`` forces ``_port_is_held`` off for these modules
(the guard's own behavior tests opt out and bind ephemeral ports). These
tests hold the production ports for real: on the pre-fix tree the guard
sees the held port and raises; with the fixture the suite spawns normally.
A final test pins the ``request_bytes`` stub itself against a real
responding server, covering the leftover-server-answering case the
held-port tests cannot.
"""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bibr.local import http_runtime
from bibr.local.http_runtime import guard_managed_server_port

PRODUCTION_PORTS = (8766, 8767, 8769, 8770, 8771, 8772, 8773, 8774, 8775)


@pytest.fixture
def occupied_production_ports():
    """Hold the production ports the way a leftover developer server would.

    Skips (rather than errors) when another process already holds one: the
    point is demonstrating the held-port behavior, which is impossible then.
    """
    held = []
    try:
        for port in PRODUCTION_PORTS:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                sock.close()
                pytest.skip(f"production port {port} already held by another process")
            sock.listen(5)
            held.append(sock)
    except BaseException:
        for sock in held:
            sock.close()
        raise
    yield held
    for sock in held:
        sock.close()


def test_port_probe_is_forced_off_for_local_tests(occupied_production_ports):
    # Read through the module namespace, the way guard_managed_server_port
    # does — a direct from-import would keep the unpatched reference.
    assert http_runtime._port_is_held("http://127.0.0.1:8772") is False


def test_guard_spawns_normally_with_production_ports_held(occupied_production_ports):
    def not_found(url, **_kwargs):
        return 404, "Not Found", b""

    assert (
        guard_managed_server_port(
            "ocr",
            base_url="http://127.0.0.1:8772",
            model="m",
            server_label="Rapid-MLX",
            request_fn=not_found,
        )
        is False
    )


class _RespondingHandler(BaseHTTPRequestHandler):
    """A leftover server that answers like a live model endpoint."""

    def do_GET(self):
        body = b'{"models": [{"id": "other-model"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def responding_loopback_server():
    """A real HTTP server answering 200 on loopback (kept quiet on teardown)."""
    server = HTTPServer(("127.0.0.1", 0), _RespondingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    thread.join(timeout=10)
    server.server_close()


def test_managed_request_bytes_stub_holds_against_responding_server(
    responding_loopback_server,
):
    """Pin the ``request_bytes`` stub: with a live-looking server answering,
    every managed module must still raise ``LocalHttpError`` (the free-port
    behavior) instead of reading its response. Without the stub this passes
    only when no server answers — and stalls when one accepts silently."""
    import importlib

    from bibr.local.http_runtime import LocalHttpError
    from tests.local.conftest import _MANAGED_MODULES

    url = f"http://127.0.0.1:{responding_loopback_server}/v1/models"
    for module_name in _MANAGED_MODULES:
        module = importlib.import_module(module_name)
        with pytest.raises(LocalHttpError):
            module.request_bytes(url, timeout=5)
