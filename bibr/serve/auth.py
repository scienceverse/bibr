"""Bearer-token auth dependency for the bibr HTTP API.

When ``Settings.auth.api_key`` is unset, the dependency is a no-op: the
endpoint is reachable without credentials. When set, the dependency
requires ``Authorization: Bearer <token>`` and rejects anything else
with HTTP 401. A keyless server is refused to browsers on other sites by
:func:`check_keyless_request`.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
from collections.abc import Mapping
from urllib.parse import urlsplit

from fastapi import Header, HTTPException, status

from bibr.config import Settings
from bibr.utils.hosts import is_loopback_host

logger = logging.getLogger(__name__)

# A bearer key shorter than this is trivially brute-forceable.
_MIN_API_KEY_LEN = 32

# Paths reachable without credentials: liveness/readiness probes carry none.
PUBLIC_PATHS = frozenset({"/health", "/ready"})


def validate_bind_auth(host: str, api_key: str | None) -> None:
    """Refuse a network-visible bind unless bearer authentication is enabled."""
    normalized = host.strip().strip("[]").lower()
    is_loopback = normalized == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(normalized).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback and not api_key:
        raise ValueError(
            f"Refusing to bind unauthenticated bibr server to {host!r}. "
            "Set AUTH_API_KEY or bind to 127.0.0.1/::1."
        )
    if api_key is not None and len(api_key) < _MIN_API_KEY_LEN:
        # Hard-fail on a network-visible bind; on loopback a short key is
        # lower-risk (local processes only) but still warned so it isn't carried
        # into a later non-loopback deployment unnoticed (audit L4).
        if not is_loopback:
            raise ValueError(
                f"AUTH_API_KEY must be at least {_MIN_API_KEY_LEN} characters "
                "for a network-visible bind"
            )
        logger.warning(
            "AUTH_API_KEY is shorter than %d characters; use a strong key even on loopback binds.",
            _MIN_API_KEY_LEN,
        )


def check_bearer(authorization: str | None) -> str | None:
    """Validate a bearer token; return the 401 detail string, or None if OK.

    Shared by the route dependency and the app-wide middleware so the two
    can't drift.
    """
    expected = Settings.auth.api_key
    if not expected:
        return None  # auth disabled (None or empty string)

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        return "Missing bearer token"

    # Compare bytes: ``compare_digest`` refuses str operands with non-ASCII
    # characters (TypeError), and Starlette decodes header bytes as latin-1,
    # so a stray high byte in the header would otherwise surface as a 500
    # instead of a 401. ``surrogateescape`` also covers a transport that hands
    # us undecodable bytes as lone surrogates. Encoding stays constant-time.
    if not hmac.compare_digest(token.encode("utf-8", "surrogateescape"), expected.encode("utf-8")):
        return "Invalid bearer token"

    return None


_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _loopback_url_host(value: str) -> bool:
    """Whether a Host header (``host[:port]``) or an Origin names a loopback host."""
    try:
        host = urlsplit(value if "://" in value else f"//{value}").hostname
    except ValueError:
        return False
    return host is not None and is_loopback_host(host)


def check_keyless_request(method: str, headers: Mapping[str, str]) -> tuple[int, str] | None:
    """Refuse a browser on another site; return ``(status, detail)`` or None if OK.

    Without ``AUTH_API_KEY`` the server may bind only loopback, and loopback is
    then the whole boundary, which a page open in the operator's browser can
    cross: a cross-site form POST needs no CORS preflight, and a DNS-rebinding
    page reaches 127.0.0.1 under a host name of its own and reads the answers.
    So a keyless server answers only a loopback ``Host``, and takes a
    state-changing request only from a loopback ``Origin`` (or one listed in
    ``CORS_ORIGINS``) or, without an ``Origin``, when ``Sec-Fetch-Site`` does
    not say ``cross-site``.

    ``bibr batch``, MCP clients and curl send a loopback Host and no Origin; the
    operator's own browser on ``http://127.0.0.1:<port>/docs`` sends a loopback
    Origin. With ``AUTH_API_KEY`` set the bearer token is the boundary (a
    browser never attaches it cross-site by itself) and nothing here applies,
    so a proxy or tunnel in front of the server needs the key.
    """
    if Settings.auth.api_key:
        return None
    host = headers.get("host")
    if host is not None and not _loopback_url_host(host):
        return (
            status.HTTP_421_MISDIRECTED_REQUEST,
            "Host not allowed: without AUTH_API_KEY bibr serve answers only loopback names",
        )
    if method.upper() in _SAFE_METHODS:
        return None
    cross_site = "Cross-site request refused: bibr serve has no AUTH_API_KEY"
    origin = headers.get("origin")
    if origin is not None:
        allowed = Settings.cors.origins
        if origin in allowed or "*" in allowed or _loopback_url_host(origin):
            return None
        return status.HTTP_403_FORBIDDEN, cross_site
    if (headers.get("sec-fetch-site") or "").lower() == "cross-site":
        return status.HTTP_403_FORBIDDEN, cross_site
    return None


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    detail = check_bearer(authorization)
    if detail is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )
