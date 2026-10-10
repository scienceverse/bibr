"""SSRF-hardened download of a public HTTPS URL (``chew_url``).

Server-side fetch of a caller-supplied URL is the canonical SSRF surface: a
URL can point the server at its own network (cloud metadata endpoints,
internal admin panels, link-local services) or be re-pointed there mid-flight
via redirects or DNS rebinding. :func:`fetch_url_safely` closes those paths
with defense in depth:

- **HTTPS only, port 443 only, no userinfo.** No plaintext hops, no
  ``user:pass@host`` ambiguity, no probing internal services on odd ports.
  An internationalised hostname is IDNA-encoded, and that ASCII name is the
  one resolved and presented in ``Host`` and SNI.
- **Every resolved address must be public.** The hostname is resolved once
  per hop and *all* A/AAAA answers are checked: multicast and anything not
  ``is_global`` (private, loopback, link-local, CGNAT/shared, reserved,
  unspecified) is refused, with IPv4-mapped IPv6 unwrapped first so
  ``::ffff:10.0.0.1`` cannot smuggle a private IPv4 through the v6 check.
  NAT64's well-known prefix is unwrapped the same way; other IPv6 outside
  global unicast (``2000::/3``) and the 6to4/Teredo tunnel prefixes are
  refused. IP-literal hosts are validated directly.
- **The connection is pinned to the validated IP** (URL host swapped for the
  IP; original hostname carried in the ``Host`` header and the
  ``sni_hostname`` request extension, so TLS SNI and certificate
  verification still use the hostname). Validating and then letting the
  HTTP client re-resolve would leave the classic rebinding TOCTOU: a DNS
  answer that is public when checked and 169.254.169.254 when connected.
  Environment proxies (``HTTPS_PROXY`` and friends) are never used: a proxy
  would pick the destination itself, and httpcore's CONNECT tunnel drops
  ``sni_hostname``, so certificate verification would fail anyway.
- **Redirects are followed manually and re-validated per hop** (bounded),
  so a public URL cannot 302 into the internal network.
- **Bounded download**: a declared ``Content-Length`` over the cap fails
  fast, and the body is counted while streaming, so a lying header cannot
  overshoot ``max_size``. Nothing is decompressed: the request asks for
  ``identity`` and a response that declares any other ``Content-Encoding``
  is refused before its body is read, so a few hundred bytes of stacked
  gzip cannot inflate to gigabytes before the cap is checked. The whole
  fetch runs under one wall-clock deadline.
- **Optional host allowlist** for operators: exact match, with subdomains
  covered (``example.org`` admits ``cdn.example.org``).

``_resolve`` and ``_transport`` exist so tests can exercise every branch
hermetically; production callers never pass them.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlsplit

import httpx

from bibr.input.supported_files import SUPPORTED_EXTENSIONS

__all__ = [
    "FetchFailedError",
    "FetchTooLargeError",
    "FetchedFile",
    "UnsafeUrlError",
    "fetch_url_safely",
]


class UnsafeUrlError(ValueError):
    """The URL (or an address it resolves to) is refused by SSRF policy."""


class FetchTooLargeError(ValueError):
    """The download exceeds the caller's size cap."""


class FetchFailedError(RuntimeError):
    """The download failed for a non-policy reason (HTTP error, network)."""


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DEFAULT_MAX_REDIRECTS = 5
_DEFAULT_TIMEOUT_SECONDS = 90.0

# NAT64's well-known prefix (RFC 6052) carries the IPv4 destination in its low
# 32 bits. DNS64 synthesises it for IPv4-only hosts on IPv6-only networks, so
# the embedded address is validated rather than the prefix refused.
_NAT64_WELL_KNOWN = ipaddress.IPv6Network("64:ff9b::/96")
# Only 2000::/3 is allocated as global unicast. Outside it, is_global still
# passes IPv4-compatible ::/96 (::7f00:1), site-local fec0::/10 and the
# IPv4-translated ::ffff:0:0:0/96, and older Pythons also 64:ff9b:1::/48.
_IPV6_GLOBAL_UNICAST = ipaddress.IPv6Network("2000::/3")
# Tunnels to an embedded IPv4 address (6to4, Teredo), refused outright rather
# than left to each Python's is_global table (6to4 joined it only in 2024).
_IPV6_TUNNELS = (ipaddress.IPv6Network("2002::/16"), ipaddress.IPv6Network("2001::/32"))
# A DNS name once IDNA-encoded: letters, digits, hyphen, underscore, dots.
# Anything else fails deep in the resolver or TLS, and getaddrinfo cuts a name
# short at NUL, so "evil.example\0.arxiv.org" would pass the allowlist and
# resolve evil.example.
_HOSTNAME = re.compile(r"[A-Za-z0-9_.-]+")

# Content-Type → extension for downloads whose URL carries no usable suffix.
_MIME_TO_EXTENSION = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/xml": ".xml",
    "text/xml": ".xml",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "application/epub+zip": ".epub",
}

_CONTENT_DISPOSITION_FILENAME = re.compile(r'filename="?([^";]+)"?', re.IGNORECASE)
# Characters no Windows file name may hold, plus the ASCII controls. The name is
# joined to a temporary directory (``bibr mcp``'s chew_url), and on Windows a
# drive-relative "D:evil.pdf" would discard that directory and land on drive D.
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# DOS device names, reserved on Windows whatever the extension ("NUL.pdf").
_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10))}
)


@dataclass(frozen=True)
class FetchedFile:
    """A validated, size-bounded download."""

    content: bytes
    filename: str
    content_type: str | None
    final_url: str


def _reject(url: str, reason: str) -> UnsafeUrlError:
    return UnsafeUrlError(f"refusing to fetch {url}: {reason}")


def _validate_ip(ip_text: str, *, url: str, host: str) -> None:
    """Refuse any address that is not unambiguously public unicast."""
    try:
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(ip_text)
    except ValueError:
        raise _reject(url, f"{host} resolved to an unparseable address {ip_text!r}") from None
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.scope_id is not None:
            # A zone index only means something on a link-local address.
            raise _reject(url, f"{host} resolves to non-public address {ip_text}")
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in _NAT64_WELL_KNOWN:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        elif ip not in _IPV6_GLOBAL_UNICAST or any(ip in net for net in _IPV6_TUNNELS):
            raise _reject(url, f"{host} resolves to non-public address {ip_text}")
    # is_global is the closed test (False for private/loopback/link-local/
    # shared CGNAT/reserved/unspecified); globally-scoped multicast still
    # reports is_global, hence the explicit check.
    if ip.is_multicast or not ip.is_global:
        raise _reject(url, f"{host} resolves to non-public address {ip_text}")


def _host_allowed(host: str, allowed_hosts: list[str]) -> bool:
    host = host.lower().rstrip(".")
    for entry in allowed_hosts:
        entry = entry.lower().strip().rstrip(".")
        if entry and (host == entry or host.endswith("." + entry)):
            return True
    return False


def _validate_url(url: str, *, allowed_hosts: list[str] | None):
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as e:  # unbalanced IPv6 brackets, port not in 0-65535
        raise _reject(url, f"malformed URL ({e})") from None
    if parsed.scheme != "https":
        raise _reject(url, "only https:// URLs are fetched")
    if parsed.username is not None or parsed.password is not None:
        raise _reject(url, "credentials in the URL are not allowed")
    host = parsed.hostname
    if not host:
        raise _reject(url, "no host")
    if port not in (None, 443):
        raise _reject(url, "only port 443 is allowed")
    if allowed_hosts and not _host_allowed(host, allowed_hosts):
        raise _reject(url, f"host {host!r} is not in the configured allowlist")
    return parsed


def _ascii_host(host: str, *, url: str) -> str:
    """*host* as the ASCII name that is resolved and sent in ``Host`` and SNI."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return host  # IP literal, validated by the resolver
    try:
        # The codec socket.getaddrinfo applies to a str host, so the name
        # looked up and the name presented to the server cannot differ. It
        # also refuses empty and over-long labels.
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise _reject(url, f"cannot IDNA-encode host {host!r}") from None
    if not _HOSTNAME.fullmatch(ascii_host):
        raise _reject(url, f"invalid host name {host!r}")
    return ascii_host


async def _resolve_public_ip(host: str, *, url: str) -> str:
    """Resolve *host*, validate every answer, and return one address to pin."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass  # a hostname — resolve below
    else:
        _validate_ip(host, url=url, host=host)
        return host  # IP-literal host, already validated

    lookup = _ascii_host(host, url=url)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(lookup, 443, type=socket.SOCK_STREAM)
    except OSError as e:
        raise FetchFailedError(f"cannot resolve {host}: {e}") from e
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses:
        raise FetchFailedError(f"cannot resolve {host}: no addresses")
    # One poisoned answer taints the lookup: a resolver that can interleave
    # public and private addresses controls which one a later connection
    # would use.
    for address in addresses:
        _validate_ip(address, url=url, host=host)
    return addresses[0]


def _pin_netloc(ip: str) -> str:
    return f"[{ip}]" if ":" in ip else ip


def _safe_file_name(name: str) -> str:
    """*name* as a single file name that stays inside its directory on any OS."""
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip().rstrip(". ")
    if name.split(".", 1)[0].upper() in _WINDOWS_DEVICE_NAMES:
        name = "_" + name
    return name


def _pick_filename(headers: httpx.Headers, final_url: str, content_type: str | None) -> str:
    """Derive a filename whose extension survives bibr's extension/MIME check.

    The server picks it (Content-Disposition, else the URL path), so it is cut
    down to one plain path component before anyone joins it to a directory.
    """
    candidate = ""
    disposition = headers.get("content-disposition", "")
    match = _CONTENT_DISPOSITION_FILENAME.search(disposition)
    if match:
        candidate = _safe_file_name(PurePosixPath(match.group(1).replace("\\", "/")).name)
    if not candidate:
        candidate = _safe_file_name(PurePosixPath(urlsplit(final_url).path).name)
    candidate = candidate[:200] or "download"

    if PurePosixPath(candidate).suffix.lower() in SUPPORTED_EXTENSIONS:
        return candidate
    mime = (content_type or "").split(";")[0].strip().lower()
    return candidate + _MIME_TO_EXTENSION.get(mime, ".pdf")


async def fetch_url_safely(
    url: str,
    *,
    max_size: int,
    allowed_hosts: list[str] | None = None,
    max_redirects: int = _DEFAULT_MAX_REDIRECTS,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    _resolve=None,
    _transport: httpx.AsyncBaseTransport | None = None,
) -> FetchedFile:
    """Download *url* under the SSRF policy documented in the module docstring.

    Raises :class:`UnsafeUrlError` for policy refusals,
    :class:`FetchTooLargeError` past *max_size*, and
    :class:`FetchFailedError` for HTTP/network failures (including a
    compressed response). No other exception escapes for any URL or reply.
    """
    resolve = _resolve or _resolve_public_ip
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(
                transport=_transport,
                follow_redirects=False,  # every hop re-validates below
                timeout=httpx.Timeout(30.0),
                # Always connect directly to the pinned IP: no HTTPS_PROXY.
                # The CA overrides (SSL_CERT_FILE/SSL_CERT_DIR) still apply.
                trust_env=False,
                verify=httpx.create_ssl_context(),
            ) as client:
                current = url
                for _ in range(max_redirects + 1):
                    parsed = _validate_url(current, allowed_hosts=allowed_hosts)
                    assert parsed.hostname is not None  # noqa: S101 — _validate_url guarantees it
                    host = _ascii_host(parsed.hostname, url=current)
                    ip = await resolve(host, url=current)
                    pinned = parsed._replace(netloc=_pin_netloc(ip)).geturl()
                    request = client.build_request(
                        "GET",
                        pinned,
                        headers={"Host": host, "Accept-Encoding": "identity"},
                        # TLS SNI + certificate verification use the real
                        # hostname even though the TCP connection goes to
                        # the pinned IP.
                        extensions={"sni_hostname": host},
                    )
                    response = await client.send(request, stream=True)
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        await response.aclose()
                        if not location:
                            raise FetchFailedError(f"redirect without Location from {host}")
                        try:
                            current = urljoin(current, location)
                        except ValueError as e:
                            raise FetchFailedError(
                                f"malformed redirect Location from {host} ({e})"
                            ) from None
                        continue
                    try:
                        if response.status_code != 200:
                            raise FetchFailedError(f"HTTP {response.status_code} from {host}")
                        # Inflating is refused, not bounded: httpx's decoders
                        # expand each chunk in full before any size check, and
                        # "gzip, gzip, gzip" multiplies that per layer. What
                        # passes here leaves httpx nothing to decode.
                        encoding = response.headers.get("content-encoding", "").strip().lower()
                        if encoding not in ("", "identity"):
                            raise FetchFailedError(
                                f"{host} sent a compressed response (Content-Encoding: "
                                f"{encoding}); only uncompressed downloads are accepted"
                            )
                        declared = response.headers.get("content-length")
                        # isascii: "²".isdigit() is True, but int() rejects it.
                        if (
                            declared
                            and declared.isascii()
                            and declared.isdigit()
                            and int(declared) > max_size
                        ):
                            raise FetchTooLargeError(
                                f"{host} declares {declared} bytes (cap {max_size})"
                            )
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > max_size:
                                raise FetchTooLargeError(
                                    f"download from {host} exceeds {max_size} bytes"
                                )
                    finally:
                        await response.aclose()
                    content_type = response.headers.get("content-type")
                    return FetchedFile(
                        content=bytes(body),
                        filename=_pick_filename(response.headers, current, content_type),
                        content_type=content_type,
                        final_url=current,
                    )
                raise FetchFailedError(f"more than {max_redirects} redirects")
    except TimeoutError:
        raise FetchFailedError(f"download did not finish within {timeout:.0f}s") from None
    except (httpx.HTTPError, httpx.InvalidURL) as e:  # InvalidURL is no HTTPError
        raise FetchFailedError(f"download failed: {e}") from e
