"""Audit regressions for the SSRF-guarded download (``bibr/utils/safe_fetch.py``).

Hermetic like ``test_safe_fetch.py``: DNS goes through a patched
``socket.getaddrinfo`` or an injected ``_resolve``, HTTP through
``httpx.MockTransport`` (or, for the proxy test, a stubbed TCP connect).
"""

from __future__ import annotations

import gzip
import socket

import httpcore
import httpx
import pytest

from bibr.utils.safe_fetch import (
    FetchFailedError,
    FetchTooLargeError,
    UnsafeUrlError,
    _validate_ip,
    fetch_url_safely,
)

PUBLIC_IP = "93.184.216.34"


async def _resolve_fixed(host, *, url):  # noqa: ARG001 — resolver contract
    return PUBLIC_IP


def _record_lookups(monkeypatch) -> list[str]:
    lookups: list[str] = []

    def fake_getaddrinfo(host, *a, **k):
        lookups.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    return lookups


class _Body(httpx.AsyncByteStream):
    """A network-like body: handed over undecoded, counting what is pulled."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pulled = 0

    async def __aiter__(self):
        self.pulled += len(self.data)
        yield self.data


# ---------------------------------------------------------------------------
# Decompression bombs: nothing is ever inflated
# ---------------------------------------------------------------------------


def _stacked_gzip(size: int, layers: int) -> bytes:
    data = bytes(size)
    for _ in range(layers):
        data = gzip.compress(data, 9)
    return data


async def test_stacked_gzip_bomb_is_refused_unread():
    """A few hundred wire bytes of "gzip, gzip, gzip" used to inflate in full
    before the size cap was checked (8 MiB here; gigabytes in the wild)."""
    body = _Body(_stacked_gzip(8 << 20, layers=3))
    assert len(body.data) < 1024

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-encoding": "gzip, gzip, gzip"}, stream=body)

    with pytest.raises(FetchFailedError, match="compressed response"):
        await fetch_url_safely(
            "https://example.org/bomb.pdf",
            max_size=1024,
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )
    assert body.pulled == 0


@pytest.mark.parametrize("encoding", ["gzip", "br", "deflate", "zstd", "x-gzip", "GZIP", "foo"])
async def test_any_content_encoding_is_refused(encoding):
    body = _Body(b"compressed-looking bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-encoding": encoding}, stream=body)

    with pytest.raises(FetchFailedError, match="only uncompressed downloads"):
        await fetch_url_safely(
            "https://example.org/p.pdf",
            max_size=1 << 20,
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )
    assert body.pulled == 0


@pytest.mark.parametrize("encoding", [None, "identity", " Identity "])
async def test_identity_is_requested_and_accepted(encoding):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["accept_encoding"] = request.headers.get("accept-encoding")
        headers = {"content-encoding": encoding} if encoding is not None else {}
        return httpx.Response(200, headers=headers, stream=_Body(b"%PDF-1.4 plain"))

    fetched = await fetch_url_safely(
        "https://example.org/p.pdf",
        max_size=1 << 20,
        _resolve=_resolve_fixed,
        _transport=httpx.MockTransport(handler),
    )
    assert seen["accept_encoding"] == "identity"
    assert fetched.content == b"%PDF-1.4 plain"


# ---------------------------------------------------------------------------
# IPv6 forms that embed or tunnel to an IPv4 address, or are not global unicast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "64:ff9b::a9fe:a9fe",  # NAT64 well-known prefix -> 169.254.169.254
        "64:ff9b::7f00:1",  # NAT64 -> 127.0.0.1
        "64:ff9b::a00:5",  # NAT64 -> 10.0.0.5
        "64:ff9b:1::a9fe:a9fe",  # NAT64 local-use prefix
        "::7f00:1",  # IPv4-compatible 127.0.0.1
        "::a9fe:a9fe",  # IPv4-compatible 169.254.169.254
        "::808:808",  # IPv4-compatible, deprecated even for a public IPv4
        "::ffff:0:a9fe:a9fe",  # IPv4-translated (SIIT)
        "2002:a9fe:a9fe::1",  # 6to4 around 169.254.169.254
        "2002:808:808::1",  # 6to4, deprecated
        "2001::1",  # Teredo
        "2001:0:4136:e378:8000:63bf:3fff:fdd2",  # Teredo
        "fec0::1",  # deprecated site-local
        "5f00::1",  # outside 2000::/3
        "2606:2800:220:1:248:1893:25c8:1946%eth0",  # zone index on a global address
    ],
)
def test_validate_ip_rejects_embedded_and_non_unicast_ipv6(ip):
    with pytest.raises(UnsafeUrlError, match="non-public"):
        _validate_ip(ip, url="https://x/", host="x")


@pytest.mark.parametrize("ip", ["64:ff9b::808:808", "64:ff9b::5db8:d822", "2a00:1450:4001::1"])
def test_validate_ip_accepts_nat64_of_public_ipv4_and_global_unicast(ip):
    """DNS64 synthesises 64:ff9b::/96 for IPv4-only hosts on IPv6-only networks."""
    _validate_ip(ip, url="https://x/", host="x")


async def test_nat64_answer_for_metadata_endpoint_is_refused(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("64:ff9b::a9fe:a9fe", 443))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be contacted")

    with pytest.raises(UnsafeUrlError, match="non-public"):
        await fetch_url_safely(
            "https://rebind.example/p.pdf",
            max_size=1024,
            _transport=httpx.MockTransport(handler),
        )


# ---------------------------------------------------------------------------
# Only the three documented errors escape, whatever the URL or reply
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "ascii_host"),
    [
        ("https://bücher.de/a.pdf", "xn--bcher-kva.de"),
        ("https://BÜCHER.de/a.pdf", "xn--bcher-kva.de"),
        ("https://例え.jp/a.pdf", "xn--r8jz45g.jp"),
    ],
)
async def test_idn_host_is_resolved_and_presented_idna_encoded(monkeypatch, url, ascii_host):
    lookups = _record_lookups(monkeypatch)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.headers["host"]
        seen["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"%PDF-1.4")

    fetched = await fetch_url_safely(url, max_size=1024, _transport=httpx.MockTransport(handler))
    assert lookups == [ascii_host]
    assert seen == {"host": ascii_host, "sni": ascii_host}
    assert fetched.filename == "a.pdf"


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("https://example.org:99999/p.pdf", "malformed URL"),
        ("https://example.org:abc/p.pdf", "malformed URL"),
        ("https://[::1/p.pdf", "malformed URL"),
        ("https://a..b/p.pdf", "IDNA"),
        ("https://" + "a" * 64 + ".org/p.pdf", "IDNA"),
        ("https://exa mple.org/p.pdf", "invalid host name"),
        ("https://%65xample.org/p.pdf", "invalid host name"),
    ],
)
async def test_malformed_url_is_a_policy_refusal(monkeypatch, url, reason):
    lookups = _record_lookups(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be contacted")

    with pytest.raises(UnsafeUrlError, match=reason):
        await fetch_url_safely(url, max_size=1024, _transport=httpx.MockTransport(handler))
    assert lookups == []


async def test_nul_in_host_cannot_slip_past_the_allowlist(monkeypatch):
    """getaddrinfo stops at NUL, so this name used to pass the arxiv.org
    allowlist and then resolve (and connect to) evil.example."""
    lookups = _record_lookups(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be contacted")

    with pytest.raises(UnsafeUrlError, match="invalid host name"):
        await fetch_url_safely(
            "https://evil.example\x00.arxiv.org/p.pdf",
            max_size=1024,
            allowed_hosts=["arxiv.org"],
            _transport=httpx.MockTransport(handler),
        )
    assert lookups == []


@pytest.mark.parametrize(
    "location",
    [
        # httpx parses these, but urljoin raises ValueError.
        b"https://a]b/",
        b"//[",
        # Joins fine; the next hop's port does not parse.
        b"https://h:99999/",
    ],
)
async def test_malformed_redirect_location_is_refused(location):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers=[(b"location", location)])
        raise AssertionError("the malformed target must not be contacted")

    with pytest.raises((FetchFailedError, UnsafeUrlError)):
        await fetch_url_safely(
            "https://example.org/start",
            max_size=1024,
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )


async def test_idn_redirect_target_is_followed(monkeypatch):
    lookups = _record_lookups(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "example.org":
            return httpx.Response(302, headers=[(b"location", "https://bücher.de/p.pdf".encode())])
        return httpx.Response(200, content=b"%PDF-1.4")

    fetched = await fetch_url_safely(
        "https://example.org/start", max_size=1024, _transport=httpx.MockTransport(handler)
    )
    assert lookups == ["example.org", "xn--bcher-kva.de"]
    assert fetched.final_url == "https://bücher.de/p.pdf"


async def test_non_ascii_content_length_does_not_escape():
    # "²".isdigit() is True but int("²") raises; h11 refuses it on the wire,
    # the guard should not depend on that.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers=[(b"content-length", "²".encode())], stream=_Body(b"%PDF")
        )

    fetched = await fetch_url_safely(
        "https://example.org/p.pdf",
        max_size=1024,
        _resolve=_resolve_fixed,
        _transport=httpx.MockTransport(handler),
    )
    assert fetched.content == b"%PDF"


async def test_non_printable_url_path_is_a_fetch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be contacted")

    with pytest.raises(FetchFailedError, match="download failed"):
        await fetch_url_safely(
            "https://example.org/a\x7fb.pdf",
            max_size=1024,
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )


# ---------------------------------------------------------------------------
# Environment proxies are never used
# ---------------------------------------------------------------------------


async def test_env_proxy_is_never_used(monkeypatch):
    """A CONNECT tunnel drops sni_hostname (TLS then verifies the pinned IP and
    fails for every site) and lets the proxy, not the IP pin, pick the
    destination. The fetch must connect directly whatever HTTPS_PROXY says."""
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://proxy.invalid:3128")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    connects: list[tuple[str, int]] = []

    # The real transport's TCP connect, stubbed: records where it would
    # connect and fails, so no socket is opened.
    async def connect_tcp(self, host, port, *args, **kwargs):  # noqa: ARG001
        connects.append((host, port))
        raise httpcore.ConnectError("stubbed")

    monkeypatch.setattr(httpcore.AnyIOBackend, "connect_tcp", connect_tcp)

    with pytest.raises(FetchFailedError, match="stubbed"):
        await fetch_url_safely(
            "https://example.org/p.pdf", max_size=1024, timeout=10, _resolve=_resolve_fixed
        )
    assert connects == [(PUBLIC_IP, 443)]


async def test_size_cap_still_applies_to_identity_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_Body(b"x" * 2048))

    with pytest.raises(FetchTooLargeError, match="exceeds"):
        await fetch_url_safely(
            "https://example.org/big.pdf",
            max_size=1024,
            _resolve=_resolve_fixed,
            _transport=httpx.MockTransport(handler),
        )
