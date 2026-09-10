"""Small stdlib HTTP helper for managed local runtime probes."""

from __future__ import annotations

import http.client
import json
import logging
import socket
import urllib.parse
from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

# Rate limit applied to a managed local LLM server bibr started itself.
# ``LLM_RATE_LIMIT_RPM``'s 60 default guards a cloud provider's quota; against
# our own server there is no external budget to protect, and at roughly 5-8
# calls per paper that default caps the run near 8-12 papers/min however fast
# the GPU actually is. This is high enough to stop binding, so the server's own
# queue and ``LLM_MAX_CONCURRENCY`` become the real bound, while still
# smoothing a stampede. Operators who want a tighter leash set
# ``LLM_RATE_LIMIT_RPM`` explicitly, which suppresses this.
MANAGED_LOCAL_LLM_RATE_LIMIT_RPM = 600


class LocalHttpError(OSError):
    """Raised when a managed local-runtime HTTP request cannot complete."""


def request_bytes(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 5.0,
) -> tuple[int, str, bytes]:
    """Request a local runtime URL and return ``(status, reason, body)``.

    This intentionally uses ``http.client`` instead of ``urllib.request`` so
    probes cannot accidentally follow non-HTTP URL schemes.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LocalHttpError(f"unsupported local runtime URL: {url!r}")

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    conn_cls = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=dict(headers or {}))
        resp = conn.getresponse()
        return resp.status, resp.reason, resp.read()
    except (OSError, TimeoutError, http.client.HTTPException, ValueError) as e:
        raise LocalHttpError(str(e)) from e
    finally:
        conn.close()


def _port_is_held(base_url: str) -> bool:
    """Best-effort check that a process really holds *base_url*'s port.

    The ``/v1/models`` probe cannot separate "nothing is listening" from
    "something is listening but does not speak the OpenAI API" — a stubbed or
    proxied probe answers either way. Confirm with an actual TCP connect before
    telling the user their port is occupied.
    """
    parts = urllib.parse.urlsplit(base_url)
    host, port = parts.hostname, parts.port
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def guard_managed_server_port(
    service: str,
    *,
    base_url: str,
    model: str,
    server_label: str,
    request_fn: Callable[..., tuple[int, str, bytes]] | None = None,
) -> bool:
    """Pre-flight a managed local server's port before spawning a subprocess.

    Health polling alone cannot tell "our subprocess came up" from "a stale
    process left on the port answered", so a leftover server serving a
    different model would be silently reused and every request would 404.
    Probe ``/v1/models`` on the configured port first:

    - Nothing listening → return ``False`` (spawn normally).
    - Listener already serves *model* → return ``True`` (reuse it; the caller
      must not spawn, and must not kill a process it does not own).
    - Listener serves other model(s) → raise ``UpstreamServiceError`` naming
      both models.
    - Listener with an unusable ``/v1/models`` → raise ``UpstreamServiceError``
      naming the occupied port, rather than attempting a spawn that cannot
      bind it and failing with an unrelated-looking startup crash.

    *request_fn* lets callers route the probe through their module's
    ``request_bytes`` binding, matching how their other probes are issued.
    """
    if request_fn is None:
        request_fn = request_bytes
    models_url = f"{base_url}/v1/models"
    try:
        status, _reason, body = request_fn(models_url, timeout=5)
    except LocalHttpError:
        return False

    served_ids: list[str] = []
    if status == 200:
        try:
            data = json.loads(body.decode("utf-8"))
            served_ids = [
                str(item["id"])
                for item in data.get("data", [])
                if isinstance(item, dict) and "id" in item
            ]
        except (ValueError, AttributeError, TypeError):
            served_ids = []

    if model in served_ids:
        logger.info(
            "Reusing already-running %s server at %s (already serves %s)",
            server_label,
            base_url,
            model,
        )
        return True

    from bibr.exceptions import UpstreamServiceError

    port = urllib.parse.urlsplit(base_url).port

    if not served_ids:
        if not _port_is_held(base_url):
            logger.warning(
                "Could not verify what is listening at %s (its /v1/models returned "
                "HTTP %d), but nothing holds the port now — starting the managed %s "
                "server.",
                base_url,
                status,
                server_label,
            )
            return False
        # Something really holds the port and does not answer /v1/models usably.
        # The managed server cannot bind an occupied port, so spawning it would
        # die with an unrelated-looking startup crash (issue #82). Name the real
        # cause instead.
        raise UpstreamServiceError(
            service,
            f"port {port} is already in use by a process that is not a usable "
            f"{server_label} server — its /v1/models returned HTTP {status}, so bibr "
            f"cannot confirm it serves {model!r}. Starting the managed server would "
            f"fail anyway, because the port is taken. Stop whatever holds it "
            f"(`lsof -ti :{port} | xargs kill`, or `netstat -ano | findstr :{port}` "
            f"then `taskkill /PID <pid> /F` on Windows), or configure a free port, "
            f"then retry.",
        )

    raise UpstreamServiceError(
        service,
        f"port {port} already has a server listening that serves "
        f"{', '.join(repr(m) for m in served_ids)}, not the requested {model!r}. "
        f"A stale {server_label} process is likely left over from a previous run — "
        f"stop it (e.g. `lsof -ti :{port} | xargs kill`) or change the configured "
        "port, then retry.",
    )
