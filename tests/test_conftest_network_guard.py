"""The non-loopback socket guard in tests/conftest.py (x-tests-2).

Unit tests must never open real network sockets: three classifier tests used
to POST header text to the live LLM API and pass only because the call
failed. The autouse ``_block_non_loopback_sockets`` fixture fails any test
that tries a non-loopback connect or external DNS lookup, while loopback
(serve TestClients, the wedged-Redis probe, spawned LitServe workers) keeps
working. These tests fail on the pre-fix tree (no guard: real DNS/connect
attempts) and pass with it.
"""

from __future__ import annotations

import socket

import pytest

from tests.conftest import _is_loopback_host


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("::1", True),
        ("localhost", True),
        (None, True),
        ("8.8.8.8", False),
        ("192.0.2.1", False),
        ("generativelanguage.googleapis.com", False),
    ],
)
def test_loopback_detection(host, expected):
    assert _is_loopback_host(host) is expected


def test_external_dns_lookup_is_blocked():
    with pytest.raises(pytest.fail.Exception):
        socket.getaddrinfo("nonexistent.invalid", 443)


def test_external_connect_is_blocked():
    with pytest.raises(pytest.fail.Exception):
        socket.create_connection(("192.0.2.1", 443), timeout=5)


def test_loopback_connect_still_works():
    """Guard: loopback sockets the suite relies on must keep working."""
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass
