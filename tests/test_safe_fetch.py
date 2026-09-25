"""Tests for the SSRF-hardened URL fetch (``bibr/utils/safe_fetch.py``).

Everything is hermetic: DNS goes through a patched ``socket.getaddrinfo``
(``loop.getaddrinfo`` delegates to it) or an injected ``_resolve``, and HTTP
through ``httpx.MockTransport`` — no sockets are opened.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from bibr.utils.safe_fetch import (
    FetchFailedError,
    FetchTooLargeError,
    UnsafeUrlError,
    _resolve_public_ip,
    _validate_ip,
    _validate_url,
    fetch_url_safely,
)

PUBLIC_IP = "93.184.216.34"


def _addrinfo(*ips: str):
    return [
        (socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))
        for ip in ips
    ]


async def _resolve_fixed(host, *, url):  # noqa: ARG001 — resolver contract
    return PUBLIC_IP


# ---------------------------------------------------------------------------
# URL policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://example.org/paper.pdf", "only https"),
        ("ftp://example.org/paper.pdf", "only https"),
        ("file:///etc/passwd", "only https"),
        ("https://user:pass@example.org/p.pdf", "credentials"),
        ("https:///paper.pdf", "no host"),
        ("https://example.org:8443/paper.pdf", "port 443"),
    ],
)
def test_validate_url_rejections(url, reason):
    with pytest.raises(UnsafeUrlError, match=reason):
        _validate_url(url, allowed_hosts=None)


def test_validate_url_accepts_https_default_and_explicit_port():
    assert _validate_url("https://example.org/p.pdf", allowed_hosts=None).hostname == "example.org"
    assert _validate_url("https://example.org:443/p", allowed_hosts=None).port == 443


def test_validate_url_allowlist():
    allowed = ["arxiv.org", "Example.ORG."]
    for ok in (
        "https://arxiv.org/abs/1",
        "https://export.arxiv.org/pdf/1",
        "https://example.org/x",
    ):
        _validate_url(ok, allowed_hosts=allowed)
    with pytest.raises(UnsafeUrlError, match="allowlist"):
        _validate_url("https://evil.org/x", allowed_hosts=allowed)
    with pytest.raises(UnsafeUrlError, match="allowlist"):
        # Suffix must be a label boundary: notarxiv.org is not *.arxiv.org.
        _validate_url("https://notarxiv.org/x", allowed_hosts=allowed)


# ---------------------------------------------------------------------------
# Address policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",  # loopback
        "10.1.2.3",  # RFC1918
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # link-local / cloud metadata
        "100.64.0.1",  # CGNAT shared space
        "0.0.0.0",  # unspecified  # noqa: S104 — test datum, not a bind address
        "224.0.1.1",  # multicast (globally scoped)
        "255.255.255.255",
        "::1",  # v6 loopback
        "fe80::1",  # v6 link-local
        "fd00::1",  # v6 ULA
        "::ffff:10.0.0.1",  # v4-mapped private
    ],
)
def test_validate_ip_rejects_non_public(ip):
    with pytest.raises(UnsafeUrlError, match="non-public"):
        _validate_ip(ip, url="https://x/", host="x")


@pytest.mark.parametrize("ip", [PUBLIC_IP, "2606:2800:220:1:248:1893:25c8:1946"])
def test_validate_ip_accepts_public(ip):
    _validate_ip(ip, url="https://x/", host="x")


async def test_resolve_rejects_ip_literal_host():
    with pytest.raises(UnsafeUrlError, match="non-public"):
        await _resolve_public_ip("169.254.169.254", url="https://169.254.169.254/")


async def test_resolve_one_private_answer_taints_lookup(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo(PUBLIC_IP, "10.0.0.5"))
    with pytest.raises(UnsafeUrlError, match="non-public"):
        await _resolve_public_ip("evil.example", url="https://evil.example/")


async def test_resolve_returns_first_public_answer(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo(PUBLIC_IP, "1.1.1.1"))
    assert await _resolve_public_ip("ok.example", url="https://ok.example/") == PUBLIC_IP


async def test_resolve_failure_is_fetch_error(monkeypatch):
    def boom(*a, **k):
        raise socket.gaierror("NXDOMAIN")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(FetchFailedError, match="cannot resolve"):
        await _resolve_public_ip("gone.example", url="https://gone.example/")


# ---------------------------------------------------------------------------
# Fetch: pinning, redirects, size caps, filenames
# ---------------------------------------------------------------------------


async def test_fetch_pins_ip_and_keeps_hostname_for_tls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url_host"] = request.url.host
        seen["host_header"] = request.headers["host"]
        seen["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"%PDF-1.4 data")

    fetched = await fetch_url_safely(
        "https://example.org/papers/some.pdf",
        max_size=1_000_000,
        _resolve=_resolve_fixed,
        _transport=httpx.MockTransport(handler),
    )
    assert seen == {"url_host": PUBLIC_IP, "host_header": "example.org", "sni": "example.org"}
    assert fetched.content == b"%PDF-1.4 data"
    assert fetched.filename == "some.pdf"
    assert fetched.final_url == "https://example.org/papers/some.pdf"


async def test_fetch_redirect_cannot_reenter_private_network(monkeypatch):
    """The rebinding-via-redirect case: public host 302s to an internal name."""
    table = {"public.example": _addrinfo(PUBLIC_IP), "internal.example": _addrinfo("10.0.0.7")}
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: table[host])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "public.example":
            return httpx.Response(302, headers={"location": "https://internal.example/steal"})
        raise AssertionError("the internal host must never be contacted")

    with pytest.raises(UnsafeUrlError, match="non-public"):
        await fetch_url_safely(
            "https://public.example/p.pdf",
            max_size=1_000_000,
            _transport=httpx.MockTransport(handler),
        )


async def test_fetch_redirect_downgrade_to_http_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://example.org/p.pdf"})

    with pytest.raises(UnsafeUrlError, match="only https"):
        await fetch_url_safely(
            "https://example.org/p.pdf",
            max_size=1_000_000,
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )


async def test_fetch_follows_relative_redirect_then_succeeds():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(307, headers={"location": "/real.pdf"})
        return httpx.Response(200, content=b"ok")

    fetched = await fetch_url_safely(
        "https://example.org/start",
        max_size=1_000_000,
        _resolve=_resolve_fixed,
        _transport=httpx.MockTransport(handler),
    )
    assert fetched.final_url == "https://example.org/real.pdf"


async def test_fetch_redirect_loop_is_bounded():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.org/again"})

    with pytest.raises(FetchFailedError, match="redirects"):
        await fetch_url_safely(
            "https://example.org/p.pdf",
            max_size=1_000_000,
            max_redirects=3,
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )


async def test_fetch_declared_length_fails_fast():
    def handler(request: httpx.Request) -> httpx.Response:
        # httpx sets Content-Length from `content`, so this exercises the
        # header fast-fail before any body bytes are read.
        return httpx.Response(200, content=b"x" * 2048)

    with pytest.raises(FetchTooLargeError, match="declares 2048 bytes"):
        await fetch_url_safely(
            "https://example.org/big.pdf",
            max_size=1024,
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )


async def test_fetch_streaming_size_cap_without_content_length():
    class _Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(4):
                yield b"x" * 512

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_Chunks())  # no Content-Length header

    with pytest.raises(FetchTooLargeError, match="exceeds"):
        await fetch_url_safely(
            "https://example.org/big.pdf",
            max_size=1024,
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )


async def test_fetch_http_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(FetchFailedError, match="HTTP 404"):
        await fetch_url_safely(
            "https://example.org/gone.pdf",
            max_size=1024,
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )


async def test_fetch_filename_from_content_disposition_and_mime():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/cd":
            return httpx.Response(
                200,
                content=b"d",
                headers={"content-disposition": 'attachment; filename="../paper (1).pdf"'},
            )
        # No usable suffix anywhere: extension comes from Content-Type.
        return httpx.Response(200, content=b"d", headers={"content-type": "application/pdf"})

    with_cd = await fetch_url_safely(
        "https://example.org/cd",
        max_size=1024,
        _resolve=_resolve_fixed,
        _transport=httpx.MockTransport(handler),
    )
    assert with_cd.filename == "paper (1).pdf"

    from_mime = await fetch_url_safely(
        "https://example.org/records/8823",
        max_size=1024,
        _resolve=_resolve_fixed,
        _transport=httpx.MockTransport(handler),
    )
    assert from_mime.filename == "8823.pdf"


async def test_fetch_allowlist_enforced_before_any_network():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be contacted")

    with pytest.raises(UnsafeUrlError, match="allowlist"):
        await fetch_url_safely(
            "https://evil.org/p.pdf",
            max_size=1024,
            allowed_hosts=["arxiv.org"],
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )


@pytest.mark.parametrize(
    ("disposition", "url", "expected"),
    [
        ('attachment; filename="D:evil.pdf"', "https://x.org/y", "D_evil.pdf"),
        ('attachment; filename="C:\\\\Windows\\\\evil.pdf"', "https://x.org/y", "evil.pdf"),
        ('attachment; filename="\\\\\\\\server\\\\share\\\\p.pdf"', "https://x.org/y", "p.pdf"),
        ('attachment; filename="NUL.pdf"', "https://x.org/y", "_NUL.pdf"),
        ('attachment; filename="com1.pdf"', "https://x.org/y", "_com1.pdf"),
        ('attachment; filename="LPT9.tar.pdf"', "https://x.org/y", "_LPT9.tar.pdf"),
        ('attachment; filename="aux"', "https://x.org/y", "_aux.pdf"),
        ('attachment; filename="lpt10.pdf"', "https://x.org/y", "lpt10.pdf"),  # not a device
        ('attachment; filename="paper.pdf:stream"', "https://x.org/y", "paper.pdf_stream.pdf"),
        ('attachment; filename="paper.pdf. "', "https://x.org/y", "paper.pdf"),
        ('attachment; filename=".."', "https://x.org/files/p.pdf", "p.pdf"),
        (None, "https://x.org/files/D:evil.pdf", "D_evil.pdf"),
    ],
)
def test_filename_stays_one_component_inside_a_windows_directory(disposition, url, expected):
    """x-security-2: bibr mcp's chew_url joins this name to a temp directory, and
    on Windows a drive-relative "D:evil.pdf" discards the directory."""
    from pathlib import PureWindowsPath

    from bibr.utils.safe_fetch import _pick_filename

    headers = httpx.Headers({"content-disposition": disposition} if disposition else {})
    name = _pick_filename(headers, url, "application/pdf")
    tmp = PureWindowsPath(r"C:\Users\me\AppData\Local\Temp\bibr-mcp-url-ab12")

    assert name == expected
    assert (tmp / name).parent == tmp
