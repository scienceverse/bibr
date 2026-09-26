"""Shared classification of "worth retrying" network failures.

Every network client here hand-rolled its own notion of transient, and each
one missed a different exception shape: the Crossref retry loop named exactly
three httpx timeout/connect classes (so a ``RemoteProtocolError``, ``ReadError``
or ``PoolTimeout`` escaped with zero retries and permanently lost that
reference's enrichment), the LLM retry loop keyed purely on HTTP status (so
``APIConnectionError`` / ``APITimeoutError`` / a bare connection reset — none
of which carry a status — were re-raised immediately despite
``_RETRY_MAX_ATTEMPTS=3``), and the geom-segmenter loader treated any load
failure as permanent for the process lifetime.

One predicate now backs all three. It matches on class *names* walked up the
MRO rather than importing httpx / the provider SDKs / huggingface_hub, so it
works regardless of which optional extras are installed and never drags a
heavy import into a hot path.

Deliberately absent: ``HTTPStatusError`` and friends. A status-carrying error
is classified by its status code at the call site (429 and 5xx retry, 4xx does
not); folding it in here would silently retry a 404.

:func:`is_service_outage` asks a narrower question for ``bibr batch`` resume:
did the failure come from a service being down, so that it says nothing about
the paper? It leaves timeouts out, and it does read HTTP status codes.
"""

from __future__ import annotations

# Matched against every class name in the exception's MRO, so a base class
# entry covers its whole subtree.
_TRANSIENT_EXC_NAMES: frozenset[str] = frozenset(
    {
        # Builtins. ConnectionError covers reset/abort/refused/broken-pipe.
        "TimeoutError",
        "ConnectionError",
        # httpx. TransportError is the base of every connect/read/write/pool
        # timeout, ConnectError, ReadError, ProxyError and both protocol
        # errors. DecodingError sits under HTTPError instead, so name it too:
        # a truncated/garbled body is a re-fetch, not a permanent verdict.
        "TransportError",
        "DecodingError",
        # requests / urllib3.
        "RequestException",
        "ProtocolError",
        "ChunkedEncodingError",
        # openai + anthropic SDKs (APITimeoutError subclasses APIConnectionError).
        "APIConnectionError",
        # google-genai / google-api-core.
        "ServiceUnavailable",
        "DeadlineExceeded",
        "ServerError",
        # aiohttp.
        "ClientConnectionError",
        "ClientOSError",
        "ServerTimeoutError",
        # TLS handshake failures are near-always retryable in practice.
        "SSLError",
        # huggingface_hub: "the Hub was unreachable AND the file is not in the
        # local cache" — the one Hub error that a later attempt can fix.
        "LocalEntryNotFoundError",
    }
)

# How far up ``__cause__`` / ``__context__`` to look. SDKs routinely wrap a
# connection reset in their own error type; one or two hops covers that
# without turning an unrelated chained exception into a retry.
_MAX_CAUSE_DEPTH = 4


def _names(exc: BaseException) -> set[str]:
    return {klass.__name__ for klass in type(exc).__mro__}


def is_transient_network_error(exc: BaseException) -> bool:
    """Is ``exc`` a connection/timeout-shaped failure worth retrying?

    Walks the cause chain, so an SDK error wrapping a connection reset is
    still recognised. Status-carrying HTTP errors are *not* handled here —
    classify those by status code at the call site.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_MAX_CAUSE_DEPTH):
        if current is None or id(current) in seen:
            return False
        seen.add(id(current))
        if _names(current) & _TRANSIENT_EXC_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


# A service that could not be reached, dropped the connection, or said it is
# down. Narrower than "worth retrying" above: it decides whether a failed
# paper is re-run by default when a batch is resumed, so it must say nothing
# about the paper. Timeouts are left out on purpose: a request the service
# accepted can be slow because of the input (a very long paper), and re-running
# that by default would never finish.
_OUTAGE_EXC_NAMES: frozenset[str] = frozenset(
    {
        "ConnectionError",  # builtin: refused, reset, aborted, broken pipe
        "NetworkError",  # httpx ConnectError / ReadError / WriteError / CloseError
        "RemoteProtocolError",  # httpx: the server hung up without an answer
        "ProxyError",  # httpx
        "APIConnectionError",  # openai / anthropic SDKs
        "ServiceUnavailable",  # google-api-core 503
        "ClientConnectionError",  # aiohttp
        "CircuitOpenError",  # bibr's breaker, open after repeated service failures
        "LocalEntryNotFoundError",  # huggingface_hub: Hub unreachable, file not cached
    }
)
_TIMEOUT_EXC_NAMES: frozenset[str] = frozenset(
    {"TimeoutError", "TimeoutException", "APITimeoutError", "DeadlineExceeded"}
)
# 429: rate limit or spent quota. 502/503: a gateway or the service says it is
# unavailable. 504 is a timeout, left out for the reason above.
_OUTAGE_HTTP_STATUSES = frozenset({429, 502, 503})
_MAX_OUTAGE_DEPTH = 6


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status an httpx / provider-SDK error carries, if any."""
    response = getattr(exc, "response", None)
    for holder, attr in ((response, "status_code"), (response, "status"), (exc, "status_code")):
        value = getattr(holder, attr, None) if holder is not None else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    code = getattr(exc, "code", None)  # google-genai keeps the int status here
    if isinstance(code, int) and not isinstance(code, bool) and 100 <= code <= 599:
        return code
    return None


def _is_outage_link(exc: BaseException, *, http_status: bool) -> bool:
    names = _names(exc)
    if "ConnectTimeout" in names:  # the host never accepted the connection
        return True
    if names & _TIMEOUT_EXC_NAMES:
        return False
    if names & _OUTAGE_EXC_NAMES:
        return True
    return http_status and _http_status(exc) in _OUTAGE_HTTP_STATUSES


def is_service_outage(exc: BaseException | None, *, http_status: bool = True) -> bool:
    """Did *exc*, or an error it wraps, come from a service being down?

    True for a connection that could not be made or was dropped, an open
    circuit breaker, a missing uncached model with the Hub unreachable, and
    an HTTP 429/502/503 — the failure says nothing about the input, so the
    same input may well succeed later. Walks ``__cause__``, an
    ``original_error`` attribute (:class:`bibr.exceptions.UpstreamServiceError`)
    and unsuppressed ``__context__``, because the pipeline wraps service
    errors in its own types. A timeout is not an outage.

    ``http_status=False`` leaves the 429/502/503 answers out: the service
    answered, so it is up, if busy. What is left is a service that is gone.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [exc] if exc is not None else []
    while pending and len(seen) < _MAX_OUTAGE_DEPTH:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if _is_outage_link(current, http_status=http_status):
            return True
        original = getattr(current, "original_error", None)
        context = None if current.__suppress_context__ else current.__context__
        for linked in (current.__cause__, original, context):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return False
