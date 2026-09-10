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
