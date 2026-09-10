"""Shared transient-failure classification.

Each network client used to hand-roll its own notion of "retryable" and each
missed a different exception shape. One predicate now backs the Crossref retry
loop, the LLM retry loop, and the geom-segmenter loader.
"""

import httpx
import pytest

from bibr.utils.transient import is_transient_network_error


class APIConnectionError(Exception):
    """Stand-in for the openai/anthropic SDK class, matched by name."""


class LocalEntryNotFoundError(Exception):
    """Stand-in for huggingface_hub's offline-and-uncached error."""


TRANSIENT = [
    TimeoutError("slow"),
    ConnectionError("reset"),
    ConnectionResetError("peer reset"),
    BrokenPipeError("gone"),
    httpx.ConnectTimeout("connect"),
    httpx.ReadTimeout("read"),
    httpx.PoolTimeout("pool"),
    httpx.ReadError("read"),
    httpx.RemoteProtocolError("protocol"),
    httpx.ConnectError("connect"),
    httpx.ProxyError("proxy"),
    APIConnectionError("connection"),
    LocalEntryNotFoundError("offline"),
]

NOT_TRANSIENT = [
    ValueError("bad value"),
    KeyError("missing"),
    ImportError("no module named vllm"),
    FileNotFoundError("checkpoint missing"),
    RuntimeError("model produced garbage"),
]


@pytest.mark.parametrize("exc", TRANSIENT, ids=lambda e: type(e).__name__)
def test_connection_shaped_failures_are_transient(exc):
    assert is_transient_network_error(exc)


@pytest.mark.parametrize("exc", NOT_TRANSIENT, ids=lambda e: type(e).__name__)
def test_definitive_failures_are_not_transient(exc):
    assert not is_transient_network_error(exc)


def test_decoding_error_is_transient():
    """A garbled body is a re-fetch, not a permanent verdict."""
    request = httpx.Request("GET", "https://api.crossref.org/works/10.1/x")
    assert is_transient_network_error(httpx.DecodingError("bad json", request=request))


def test_status_errors_are_left_to_the_call_site():
    """Folding status errors in here would silently retry a 404."""
    request = httpx.Request("GET", "https://api.crossref.org/works/10.1/x")
    response = httpx.Response(404, request=request)
    exc = httpx.HTTPStatusError("not found", request=request, response=response)

    assert not is_transient_network_error(exc)


def test_a_wrapped_connection_reset_is_still_transient():
    """SDKs routinely wrap the real cause in their own error type."""
    try:
        try:
            raise ConnectionResetError("peer reset")
        except ConnectionResetError as cause:
            raise RuntimeError("provider call failed") from cause
    except RuntimeError as exc:
        assert is_transient_network_error(exc)


def test_a_long_unrelated_cause_chain_does_not_leak_a_retry():
    # The connection error sits 8 hops down — past the bounded walk, so an
    # unrelated deep ancestor cannot turn a permanent failure into a retry.
    exc: BaseException = ConnectionError("reset")
    for _ in range(8):
        outer = ValueError("wrapper")
        outer.__cause__ = exc
        exc = outer

    assert not is_transient_network_error(exc)


def test_self_referential_cause_chain_terminates():
    exc = ValueError("loop")
    exc.__cause__ = exc

    assert not is_transient_network_error(exc)
