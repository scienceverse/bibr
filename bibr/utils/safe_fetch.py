"""SSRF-hardened download of a public HTTPS URL (``chew_url``).

Server-side fetch of a caller-supplied URL is the canonical SSRF surface: a
URL can point the server at its own network (cloud metadata endpoints,
internal admin panels, link-local services) or be re-pointed there mid-flight
via redirects or DNS rebinding. :func:`fetch_url_safely` closes those paths
with defense in depth:

- **HTTPS only, port 443 only, no userinfo.** No plaintext hops, no
  ``user:pass@host`` ambiguity, no probing internal services on odd ports.
- **Every resolved address must be public.** The hostname is resolved once
  per hop and *all* A/AAAA answers are checked: multicast and anything not
  ``is_global`` (private, loopback, link-local, CGNAT/shared, reserved,
  unspecified) is refused, with IPv4-mapped IPv6 unwrapped first so
  ``::ffff:10.0.0.1`` cannot smuggle a private IPv4 through the v6 check.
  IP-literal hosts are validated directly.
- **The connection is pinned to the validated IP** (URL host swapped for the
  IP; original hostname carried in the ``Host`` header and the
  ``sni_hostname`` request extension, so TLS SNI and certificate
  verification still use the hostname). Validating and then letting the
  HTTP client re-resolve would leave the classic rebinding TOCTOU: a DNS
  answer that is public when checked and 169.254.169.254 when connected.
- **Redirects are followed manually and re-validated per hop** (bounded),
  so a public URL cannot 302 into the internal network.
- **Bounded download**: a declared ``Content-Length`` over the cap fails
  fast, and the body is counted as *decompressed* bytes while streaming, so
  neither a lying header nor a compressed bomb overshoots ``max_size``. The
  whole fetch runs under one wall-clock deadline.
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
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
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
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise _reject(url, "only https:// URLs are fetched")
    if parsed.username is not None or parsed.password is not None:
        raise _reject(url, "credentials in the URL are not allowed")
    host = parsed.hostname
    if not host:
        raise _reject(url, "no host")
    if parsed.port not in (None, 443):
        raise _reject(url, "only port 443 is allowed")
    if allowed_hosts and not _host_allowed(host, allowed_hosts):
        raise _reject(url, f"host {host!r} is not in the configured allowlist")
    return parsed


async def _resolve_public_ip(host: str, *, url: str) -> str:
    """Resolve *host*, validate every answer, and return one address to pin."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass  # a hostname — resolve below
    else:
        _validate_ip(host, url=url, host=host)
        return host  # IP-literal host, already validated

    lookup = host
    try:
        lookup.encode("ascii")
    except UnicodeEncodeError:
        try:
            lookup = lookup.encode("idna").decode("ascii")
        except UnicodeError:
            raise _reject(url, f"cannot IDNA-encode host {host!r}") from None
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


def _pick_filename(headers: httpx.Headers, final_url: str, content_type: str | None) -> str:
    """Derive a filename whose extension survives bibr's extension/MIME check."""
    candidate = ""
    disposition = headers.get("content-disposition", "")
    match = _CONTENT_DISPOSITION_FILENAME.search(disposition)
    if match:
        candidate = PurePosixPath(match.group(1).replace("\\", "/")).name
    if not candidate:
        candidate = PurePosixPath(urlsplit(final_url).path).name
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
    :class:`FetchTooLargeError` past *max_size* (decompressed), and
    :class:`FetchFailedError` for HTTP/network failures.
    """
    resolve = _resolve or _resolve_public_ip
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(
                transport=_transport,
                follow_redirects=False,  # every hop re-validates below
                timeout=httpx.Timeout(30.0),
            ) as client:
                current = url
                for _ in range(max_redirects + 1):
                    parsed = _validate_url(current, allowed_hosts=allowed_hosts)
                    host = parsed.hostname
                    assert host is not None  # noqa: S101 — _validate_url guarantees it
                    ip = await resolve(host, url=current)
                    pinned = parsed._replace(netloc=_pin_netloc(ip)).geturl()
                    request = client.build_request(
                        "GET",
                        pinned,
                        headers={"Host": host},
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
                        current = urljoin(current, location)
                        continue
                    try:
                        if response.status_code != 200:
                            raise FetchFailedError(f"HTTP {response.status_code} from {host}")
                        declared = response.headers.get("content-length")
                        if declared and declared.isdigit() and int(declared) > max_size:
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
    except httpx.HTTPError as e:
        raise FetchFailedError(f"download failed: {e}") from e
