"""The non-loopback socket guard in tests/conftest.py (x-tests-2).

Unit tests must never open real network sockets: three classifier tests used
to POST header text to the live LLM API and pass only because the call
failed. The autouse ``_block_non_loopback_sockets`` fixture fails any test
that tries a non-loopback connect or external DNS lookup, while loopback
(serve TestClients, the wedged-Redis probe, spawned LitServe workers) keeps
working. These tests fail on the pre-fix tree (no guard: real DNS/connect
attempts) and pass with it.

Tests marked ``network`` opt out of the guard: the live/API tests exist
precisely to reach the network. The last two tests pin the opt-out without
touching the network (a marked test sees the real socket functions, an
unmarked one sees the wrappers).
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


def test_external_connect_ex_is_blocked():
    with socket.socket() as sock:
        with pytest.raises(pytest.fail.Exception):
            sock.connect_ex(("192.0.2.1", 443))


def test_external_gethostbyname_is_blocked():
    with pytest.raises(pytest.fail.Exception):
        socket.gethostbyname("nonexistent.invalid")


def test_loopback_connect_still_works():
    """Guard: loopback sockets the suite relies on must keep working."""
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass


def test_unmarked_tests_see_the_guarded_sockets():
    assert "guarded" in socket.getaddrinfo.__qualname__
    assert "guarded" in socket.gethostbyname.__qualname__
    assert "guarded" in socket.create_connection.__qualname__
    assert "guarded" in socket.socket.connect.__qualname__
    assert "guarded" in socket.socket.connect_ex.__qualname__


@pytest.mark.network
def test_network_mark_sees_the_real_sockets():
    """Opt-out pin: a ``network``-marked test skips the guard entirely."""
    assert "guarded" not in socket.getaddrinfo.__qualname__
    assert "guarded" not in socket.gethostbyname.__qualname__
    assert "guarded" not in socket.create_connection.__qualname__
    assert "guarded" not in socket.socket.connect.__qualname__
    assert "guarded" not in socket.socket.connect_ex.__qualname__


def _mark_names(obj) -> set[str]:
    """Mark names from a test function or module's ``pytestmark``."""
    marks = getattr(obj, "pytestmark", [])
    if not isinstance(marks, (list, tuple)):
        marks = [marks]
    return {mark.name for mark in marks}


def test_weight_downloading_slow_tests_are_marked_network():
    """Slow tests that download weights must opt out of the socket guard.

    On a cold Hub cache the guard's ``pytest.fail`` (a BaseException)
    escapes the ``except Exception`` fallbacks around the download, so an
    unmarked slow download test fails where it should download and pass.
    """
    import tests.local.test_vllm_llm_integration as vllm_integration
    import tests.test_setup_wizard as setup_wizard

    assert {"slow", "network"} <= _mark_names(setup_wizard.test_smoke_test_real_extraction_no_llm)
    assert "network" in _mark_names(
        vllm_integration.test_vllm_server_launch_and_shutdown
    ) | _mark_names(vllm_integration)
