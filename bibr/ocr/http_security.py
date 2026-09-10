"""Security policy shared by HTTP OCR transports."""

from __future__ import annotations

import ipaddress
import logging
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


def normalize_ocr_base_url(base_url: str) -> str:
    """Canonical server root for an OCR base URL.

    bibr appends ``/v1/models`` and ``/v1/chat/completions`` itself and probes
    ``/health`` at the root, so a configured URL that already ends in ``/v1``
    (the natural thing to copy from an OpenAI-compatible server's docs) was
    requested as ``/v1/v1/...`` and never became ready. Strip trailing slashes
    and one trailing ``/v1`` segment; everything else passes through.
    """
    url = base_url.strip().rstrip("/")
    if url.lower().endswith("/v1"):
        stripped = url[: -len("/v1")].rstrip("/")
        logger.warning(
            "OCR base URL %r ends in /v1; bibr adds the /v1 API prefix itself, so it will use %r",
            base_url,
            stripped,
        )
        url = stripped
    return url


def ocr_request_headers(
    base_url: str,
    api_key: str | None,
    *,
    allow_insecure_http: bool,
) -> dict[str, str]:
    """Validate the OCR URL and build its bearer-auth headers."""
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OCR base URL must be an absolute http:// or https:// URL")

    hostname = parsed.hostname.lower()
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False

    if parsed.scheme != "https" and not is_loopback and not allow_insecure_http:
        raise ValueError(
            f"Refusing insecure HTTP connection to remote OCR host {hostname!r}. "
            "Use HTTPS or explicitly set OCR_ALLOW_INSECURE_HTTP=true for a trusted private network."
        )
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}
