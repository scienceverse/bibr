"""Small stdlib helpers for managed local runtime probes and startup waits."""

from __future__ import annotations

import http.client
import json
import logging
import os
import secrets
import signal
import socket
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)


class StartupCancelled(Exception):
    """The owner of a managed server stopped its startup (Ctrl-C or task cancellation).

    A startup wait raises it once its stop event is set, so the constructor's
    own cleanup shuts the half-started process down, as it does on a timeout.
    ``await_owned`` then re-raises the owner's cancellation, not this error.
    """


def check_startup_stop(stop_event: threading.Event | None, server_label: str) -> None:
    """Raise :class:`StartupCancelled` if *stop_event* is set."""
    if stop_event is not None and stop_event.is_set():
        raise StartupCancelled(f"{server_label} startup was cancelled")


def pause_startup_poll(
    seconds: float, stop_event: threading.Event | None, server_label: str
) -> None:
    """Sleep between startup polls; wake early and raise once *stop_event* is set."""
    if stop_event is None:
        time.sleep(seconds)
    else:
        stop_event.wait(seconds)
    check_startup_stop(stop_event, server_label)


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


def new_server_api_key() -> str:
    """A random API key for one managed server launch.

    Runtimes that read their key from the environment (vLLM's
    ``VLLM_API_KEY``, llama.cpp's ``LLAMA_API_KEY``) get one, so another local
    user cannot drive the server bibr started.
    """
    return secrets.token_urlsafe(32)


def env_with_api_key(variable: str, api_key: str) -> dict[str, str]:
    """This process's environment plus a server key in *variable*.

    The key never goes on the command line, where ``ps`` shows it to every
    local user; a process's environment is readable only by its owner.
    """
    return {**os.environ, variable: api_key}


def bearer_headers(api_key: str | None) -> dict[str, str]:
    """``Authorization`` header for a managed server's key; empty without one."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


# Per-launch keys of the keyed servers this process started, by port. A second
# pipeline in the same process (a warm ``Chewer`` plus ``bibr.chew()``) finds
# its sibling's server keyed: the guard authenticates with the key kept here
# and shares that server, as it did before servers had keys.
_OWN_SERVER_KEYS: dict[int | None, str] = {}
_OWN_SERVER_KEYS_LOCK = threading.Lock()


def register_server_key(base_url: str, api_key: str) -> None:
    """Record the key of a server this process started and that is now ready."""
    with _OWN_SERVER_KEYS_LOCK:
        _OWN_SERVER_KEYS[urllib.parse.urlsplit(base_url).port] = api_key


def unregister_server_key(base_url: str, api_key: str) -> None:
    """Forget a stopped server's key, unless a newer server took over its port."""
    port = urllib.parse.urlsplit(base_url).port
    with _OWN_SERVER_KEYS_LOCK:
        if _OWN_SERVER_KEYS.get(port) == api_key:
            del _OWN_SERVER_KEYS[port]


def own_server_key(base_url: str) -> str:
    """The key of the server this process started on *base_url*'s port, or ``""``."""
    with _OWN_SERVER_KEYS_LOCK:
        return _OWN_SERVER_KEYS.get(urllib.parse.urlsplit(base_url).port, "")


def _stop_hint(port: int | None) -> str:
    return (
        f"`lsof -ti :{port} | xargs kill`, or `netstat -ano | findstr :{port}` "
        "then `taskkill /PID <pid> /F` on Windows"
    )


def kill_process_group(pid: int) -> None:
    """SIGKILL whatever is left of a ``start_new_session`` child's process group.

    When the server exits during startup, ``shutdown()`` finds no live process
    to signal, but its workers (vLLM's EngineCore) can outlive it and keep
    their VRAM. Best effort: never masks the startup error. No-op on Windows.
    """
    if os.name == "nt" or not hasattr(os, "killpg"):
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:  # noqa: BLE001 - cleanup must not replace the startup error
        logger.debug("Could not kill process group %s", pid, exc_info=True)


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
    refuse_reuse: Callable[[], str | None] | None = None,
) -> bool:
    """Pre-flight a managed local server's port before spawning a subprocess.

    Health polling alone cannot tell "our subprocess came up" from "a stale
    process left on the port answered", so a leftover server serving a
    different model would be silently reused and every request would 404.
    Probe ``/v1/models`` on the configured port first:

    - Nothing listening → return ``False`` (spawn normally).
    - Listener already serves *model* → return ``True`` (reuse it; the caller
      must not spawn and must not kill a process it does not own). A listener
      this process did not start is reused with a warning; the caller must
      not send it a key. *refuse_reuse*, when given, is asked first: a reason
      it returns is logged and raised as ``UpstreamServiceError`` instead.
    - Listener rejects the keyless probe (401/403) → retry with the key of the
      server this process started on that port (see ``register_server_key``);
      without one, or when it is rejected too, raise ``UpstreamServiceError``:
      it is most likely another bibr process's server, keyed with a
      per-launch key this process does not have.
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

    own_key = own_server_key(base_url)
    if status in (401, 403) and own_key:
        try:
            status, _reason, body = request_fn(
                models_url, headers=bearer_headers(own_key), timeout=5
            )
        except LocalHttpError:
            pass

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

    from bibr.exceptions import UpstreamServiceError

    port = urllib.parse.urlsplit(base_url).port

    if model in served_ids:
        refusal = refuse_reuse() if refuse_reuse is not None else None
        if refusal:
            message = (
                f"port {port} already has a server for {model!r} that bibr will not use "
                f"as its managed {server_label} server: {refusal} Stop that server "
                f"({_stop_hint(port)}) if you did not start it yourself, then retry."
            )
            # Logged as well: a fallback chain reports only which candidate
            # failed, and this names the port and the way out.
            logger.warning("%s", message)
            raise UpstreamServiceError(service, message)
        if own_key:
            logger.info(
                "Sharing the %s server this process started at %s (it serves %s)",
                server_label,
                base_url,
                model,
            )
            return True
        # Anything on loopback can list a model id, so say plainly that this
        # listener is not ours: it receives the documents bibr sends it.
        logger.warning(
            "Reusing a %s server already listening at %s (it serves %s) instead of "
            "starting one. bibr did not start it for this pipeline and cannot check who "
            "did; it will receive the document content bibr sends. Stop it if neither "
            "you nor another bibr run started it.",
            server_label,
            base_url,
            model,
        )
        return True

    if status in (401, 403):
        raise UpstreamServiceError(
            service,
            f"port {port} is held by a server that requires an API key (its /v1/models "
            f"returned HTTP {status}), most likely a {server_label} server started by "
            "another bibr process with its own per-launch key. bibr does not share a "
            f"server it cannot authenticate to. Stop it ({_stop_hint(port)}) or "
            "configure a free port, then retry.",
        )

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
