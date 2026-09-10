from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bibr.local.http_runtime import LocalHttpError, request_bytes


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, format, *args):  # noqa: A002, ARG002
        pass


def test_request_bytes_reads_local_http_response():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, reason, body = request_bytes(f"http://127.0.0.1:{server.server_port}/health")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert reason == "OK"
    assert body == b'{"ok": true}'


def test_request_bytes_rejects_non_http_schemes():
    with pytest.raises(LocalHttpError):
        request_bytes("file:///etc/passwd")
