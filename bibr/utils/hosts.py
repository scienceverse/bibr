"""Where a host name points: loopback, a private network, or the public internet.

Decided from the name alone, never by resolving it, so the answer cannot change
with DNS (a rebinding attacker controls DNS) and costs nothing at startup.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

# DNS suffixes reserved for, or conventionally used on, private networks:
# RFC 6761 (.localhost), RFC 6762 mDNS (.local), RFC 8375 (.home.arpa), ICANN's
# private-use .internal, and the common home-router .lan.
_PRIVATE_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")


def _normalize(host: str) -> str:
    return host.strip().strip("[]").rstrip(".").lower()


def _ip(name: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def is_loopback_host(host: str) -> bool:
    """``localhost``, a ``*.localhost`` name, or a loopback IP literal."""
    name = _normalize(host)
    if name == "localhost" or name.endswith(".localhost"):
        return True
    ip = _ip(name)
    return ip is not None and ip.is_loopback


def is_private_network_host(host: str) -> bool:
    """Whether *host* names a machine on the local or a private network.

    True for loopback, for an IP literal that is not globally routable (RFC 1918,
    the 100.64.0.0/10 range tailnets use, link-local, unique-local IPv6), for a
    single-label name such as ``gpu-box`` (resolved through local search domains
    or the hosts file) and for the private-use DNS suffixes above.
    """
    name = _normalize(host)
    if not name:
        return False
    if is_loopback_host(name):
        return True
    ip = _ip(name)
    if ip is not None:
        return not ip.is_global
    return "." not in name or name.endswith(_PRIVATE_SUFFIXES)


def refuse_public_plaintext(url: str, *, credential: str, opt_out: str) -> None:
    """Raise ``ValueError`` when *url* would carry *credential* in clear text.

    Plain ``http://`` stays allowed to loopback and private-network hosts (a
    local vLLM, a GPU box on the LAN or tailnet); only a public host is refused,
    since the bearer token would cross the internet readable by anyone on the
    path. *opt_out* names the explicit override in the error message.
    """
    parts = urlsplit(url.strip())
    host = parts.hostname
    if parts.scheme.lower() != "http" or not host or is_private_network_host(host):
        return
    raise ValueError(
        f"Refusing to send the {credential} over plain HTTP to the public host {host!r}. "
        f"Use an https:// URL, or {opt_out} if the network path is trusted."
    )
