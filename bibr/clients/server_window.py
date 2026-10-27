"""The context window an OpenAI-compatible LLM server lists for its model.

vLLM lists each model's ``max_model_len`` under ``/v1/models``. bibr reads it
only after an answer stopped at a window bibr did not know
(``LLMClient._learn_window_from_truncation``). vLLM 0.31 rejected a prompt
whose completion cap ran past the window with HTTP 400 and named the window,
which bibr learns from (``_context_window_from_error``); vLLM 0.32 lowers the
cap to what is left instead, so the answer stops there and names nothing.

Answers are kept per server and model for the life of the process. A server
that answers without a window (another runtime, a router listing several
models, no ``/v1/models`` route) is kept as listing none. A server that does
not answer (down, timed out, HTTP 5xx) is asked again the next time. Never
raises.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from bibr.config import GlobalSettings

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 5.0
_CONNECT_TIMEOUT_SECONDS = 2.0

# (models URL, model) -> the window the server lists, or None when it answered
# without one.
_listed: dict[tuple[str, str], int | None] = {}


class _NoAnswer(Exception):
    """The server answered HTTP 5xx: starting, overloaded or failing."""


def clear() -> None:
    """Forget every server's answer."""
    _listed.clear()


def _models_url(settings: GlobalSettings) -> str | None:
    """``/models`` beside the chat route of an ``LLM_BASE_URL`` server, else ``None``."""
    llm = settings.llm
    base = (llm.base_url or "").strip().rstrip("/")
    if llm.provider.strip().lower() != "openai" or not base:
        return None
    return f"{base}/models"


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _window_in(listing: Any, model: str) -> int | None:
    """The ``max_model_len`` a ``/v1/models`` answer lists for *model*."""
    data = listing.get("data") if isinstance(listing, dict) else None
    entries = [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []
    named = [e for e in entries if e.get("id") == model]
    # A server that lists one model serves it under whatever name it is asked
    # for; a router that lists several may hold other windows.
    if not named and len(entries) == 1:
        named = entries
    return _positive_int(named[0].get("max_model_len")) if named else None


async def listed_window(
    settings: GlobalSettings, *, transport: httpx.AsyncBaseTransport | None = None
) -> int | None:
    """The context window the ``LLM_BASE_URL`` server lists for ``llm.model``.

    ``None`` for a cloud provider, a server that lists none, and a server
    that did not answer.
    """
    url = _models_url(settings)
    if url is None:
        return None
    key = (url, settings.llm.model)
    if key in _listed:
        return _listed[key]
    try:
        window = await _ask(settings, url, transport)
    except Exception as exc:  # noqa: BLE001 - a window the server did not give is unknown
        logger.debug("The LLM server at %s did not list its context window: %s", url, exc)
        return None
    _listed[key] = window
    return window


async def _ask(
    settings: GlobalSettings, url: str, transport: httpx.AsyncBaseTransport | None
) -> int | None:
    import httpx

    from bibr.utils.hosts import refuse_plaintext_llm_key

    llm = settings.llm
    headers: dict[str, str] = {}
    if llm.api_key:
        # The chat call carries the key under the same rule.
        refuse_plaintext_llm_key(url, llm.api_key, allow_insecure_http=llm.allow_insecure_http)
        headers["Authorization"] = f"Bearer {llm.api_key}"
    timeout = httpx.Timeout(_TIMEOUT_SECONDS, connect=_CONNECT_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(
        timeout=timeout, headers=headers, transport=transport, follow_redirects=False
    ) as client:
        response = await client.get(url)
    if response.status_code >= 500:
        raise _NoAnswer(f"HTTP {response.status_code}")
    if response.status_code != 200:
        return None
    try:
        listing = response.json()
    except ValueError:
        return None
    return _window_in(listing, llm.model)
