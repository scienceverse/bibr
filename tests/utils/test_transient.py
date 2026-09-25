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


# --- is_service_outage: does a failure say nothing about the input? ----------


class APITimeoutError(APIConnectionError):
    """Stand-in for the openai SDK class: a timeout, though it subclasses the
    connection error."""


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://ocr:8080/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def _wrapped(cause: BaseException) -> BaseException:
    """What the LLM client does: a catch-all UpstreamServiceError around the cause."""
    from bibr.exceptions import UpstreamServiceError

    try:
        raise UpstreamServiceError("LLM", "Failed to extract authors", cause) from cause
    except UpstreamServiceError as exc:
        return exc


def _raised_from(cause: BaseException) -> BaseException:
    """A plain wrapper with only ``__cause__`` (no ``original_error``)."""
    try:
        raise RuntimeError("OCR page failed") from cause
    except RuntimeError as exc:
        return exc


def _raised_while_handling(context: BaseException, *, suppress: bool) -> BaseException:
    try:
        try:
            raise context
        except type(context):
            if suppress:
                raise ValueError("no JSON object in the completion") from None
            raise ValueError("no JSON object in the completion")  # noqa: B904
    except ValueError as exc:
        return exc


class _GenaiError(Exception):
    """Stand-in for google-genai's APIError, which keeps the int status in ``code``."""

    def __init__(self, code: int):
        super().__init__(f"{code} error")
        self.code = code


def _circuit_open() -> BaseException:
    from bibr.exceptions import UpstreamServiceError
    from bibr.utils.circuit_breaker import CircuitOpenError

    opened = CircuitOpenError("ocr", 30.0)
    return UpstreamServiceError("ocr", str(opened), original_error=opened)


OUTAGES = [
    ConnectionRefusedError("refused"),
    httpx.ConnectError("[Errno 111] Connection refused"),
    httpx.ConnectTimeout("never connected"),
    httpx.ReadError("server went away"),
    httpx.RemoteProtocolError("Server disconnected without sending a response."),
    APIConnectionError("connection"),
    LocalEntryNotFoundError("offline"),
    _status_error(503),
    _status_error(502),
    _status_error(429),
    _wrapped(httpx.ConnectError("refused")),
    _raised_from(httpx.ConnectError("[Errno 111] Connection refused")),
    _raised_while_handling(httpx.ConnectError("refused"), suppress=False),
    _circuit_open(),
    _GenaiError(503),
]

NOT_OUTAGES = [
    None,
    TimeoutError("LLM call timed out after 600s"),
    httpx.ReadTimeout("read"),
    APITimeoutError("timed out"),
    _status_error(504),
    _status_error(500),
    _status_error(400),
    _wrapped(ValueError("no JSON object in the completion")),
    _wrapped(TimeoutError("slow")),
    _raised_while_handling(httpx.ConnectError("refused"), suppress=True),
    _GenaiError(404),
    ValueError("bad value"),
]


@pytest.mark.parametrize("exc", OUTAGES, ids=lambda e: type(e).__name__)
def test_service_outages_are_recognised_through_wrappers(exc):
    from bibr.utils.transient import is_service_outage

    assert is_service_outage(exc) is True


@pytest.mark.parametrize("exc", NOT_OUTAGES, ids=lambda e: type(e).__name__)
def test_timeouts_and_paper_failures_are_not_outages(exc):
    from bibr.utils.transient import is_service_outage

    assert is_service_outage(exc) is False


def test_an_upstream_error_alone_is_not_an_outage():
    """The LLM client wraps any failure, bad model output included, in an
    UpstreamServiceError; only a service-shaped cause makes it an outage."""
    from bibr.exceptions import UpstreamServiceError
    from bibr.utils.transient import is_service_outage

    assert (
        is_service_outage(UpstreamServiceError("LLM", "All 3 reference parse batches failed"))
        is False
    )
