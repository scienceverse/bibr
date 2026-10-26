"""The context window of a self-hosted LLM server, read from the server itself.

Ollama and llama.cpp serve a fixed context (Ollama's ``num_ctx``, llama.cpp's
``-c``). Given a longer prompt, llama.cpp and Ollama's GGUF-template models
answer HTTP 400, and Ollama's library models cut the prompt from the front,
dropping the instructions. bibr's input cap is sized for cloud models
(``LLM_MAX_INPUT_CHARS`` 300,000 characters), so on a 16,384-token Ollama a long
front matter failed the paper or came back without its title and authors.

:func:`probe_server_context` asks the server which kind it is and, for Ollama
and llama.cpp, how long its context is:

- vLLM lists ``max_model_len`` under ``/v1/models``;
- llama.cpp's ``/props`` gives ``default_generation_settings.n_ctx``;
- Ollama answers ``/api/version``; ``/api/ps`` gives a loaded model's
  ``context_length``, and ``/api/show`` a ``num_ctx`` the model sets. The
  ``<arch>.context_length`` in ``/api/show`` is the trained maximum, not the
  context Ollama runs, so it is not used.

:func:`apply_server_context` then gives Ollama and llama.cpp the caps the
managed llama.cpp server uses and sets ``llm.context_tokens``. vLLM is left as
it was: bibr learns its window from a context-length rejection. Values the
user set are never replaced.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import httpx

    from bibr.config import GlobalSettings, LlmOptions

logger = logging.getLogger(__name__)

# Ollama and llama.cpp whose context could not be read are assumed to run the
# 16,384-token context bibr's managed llama.cpp server starts with.
ASSUMED_FIXED_CONTEXT_TOKENS = 16_384
# Servers that run one fixed context and reject or cut a longer prompt. Only
# these are sized before the first prompt.
FIXED_CONTEXT_SERVERS = ("ollama", "llama.cpp")

_PROBE_TIMEOUT_SECONDS = 5.0
_PROBE_CONNECT_TIMEOUT_SECONDS = 2.0
_NUM_CTX_RE = re.compile(r"^\s*num_ctx\s+(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ServerContext:
    """What a server said about itself."""

    # "other": the server answered, but on none of the routes above.
    server: Literal["ollama", "llama.cpp", "vllm", "other"]
    # ``None`` when not known, e.g. an Ollama whose model is not loaded and
    # sets no ``num_ctx``.
    context_tokens: int | None


class _NoAnswer(Exception):
    """A probe route answered HTTP 5xx: the server is starting or overloaded."""


def fixed_context_caps(context_tokens: int) -> tuple[tuple[str, int], ...]:
    """``llm`` caps that keep prompt plus completion inside a fixed context.

    They equal the historical llama.cpp constants at a 16,384-token context
    (completion at most a quarter of it) and scale with the context.
    """
    return (
        ("max_tokens", context_tokens // 4),
        ("max_input_chars", 24_000 * context_tokens // 16_384),
        ("ref_seg_window_chars", 12_000 * context_tokens // 16_384),
    )


def self_hosted_endpoint(settings: GlobalSettings) -> tuple[str, str] | None:
    """``(server root, OpenAI-compatible base)`` of a self-hosted LLM, else ``None``."""
    llm = settings.llm
    provider = llm.provider.strip().lower()
    if provider == "ollama":
        url = llm.ollama_base_url
    elif provider == "openai" and (llm.base_url or "").strip():
        url = llm.base_url or ""
    else:
        return None
    base = url.strip().rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    return root, f"{root}/v1"


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _ollama_name(model: str) -> str:
    """Ollama's name for *model*: an untagged name means ``:latest``.

    Casefolded, because Ollama matches model names without regard to case.
    """
    name = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
    return name.casefold()


async def _get_json(client: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> Any:
    """The JSON body of a 200 answer; ``None`` for a 4xx or a body that is not JSON."""
    response = await client.request(method, url, **kwargs)
    if response.status_code >= 500:
        raise _NoAnswer(f"{url} answered HTTP {response.status_code}")
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


async def _ollama_context(client: httpx.AsyncClient, root: str, model: str) -> int | None:
    # With OLLAMA_NUM_PARALLEL above 1, whether /api/ps reports the context of
    # one slot or of all slots together has not been checked.
    name = _ollama_name(model)
    try:
        ps = await _get_json(client, "GET", f"{root}/api/ps")
        models = ps.get("models") if isinstance(ps, dict) else None
        for entry in models if isinstance(models, list) else []:
            if not isinstance(entry, dict):
                continue
            names = {str(entry.get(key) or "").casefold() for key in ("name", "model")}
            if name in names and (window := _positive_int(entry.get("context_length"))):
                return window
        show = await _get_json(client, "POST", f"{root}/api/show", json={"model": model})
    except Exception as exc:  # noqa: BLE001 - a window Ollama did not give is unknown
        logger.debug("Ollama at %s did not give its context window: %s", root, exc)
        return None
    parameters = show.get("parameters") if isinstance(show, dict) else None
    match = _NUM_CTX_RE.search(parameters) if isinstance(parameters, str) else None
    return int(match.group(1)) if match else None


async def probe_server_context(
    settings: GlobalSettings, *, transport: httpx.AsyncBaseTransport | None = None
) -> ServerContext | None:
    """Ask a self-hosted LLM server which kind it is and how long its context is.

    ``None`` for a cloud provider, and when no answer could be read: the
    server is down or starting (HTTP 5xx), or its URL is malformed. Never
    raises: the extraction call that follows reports a broken endpoint itself.
    """
    try:
        return await _probe(settings, transport)
    except Exception as exc:  # noqa: BLE001 - the probe must never fail a call
        logger.debug("LLM server context probe failed: %s", exc)
        return None


async def _probe(
    settings: GlobalSettings, transport: httpx.AsyncBaseTransport | None
) -> ServerContext | None:
    import httpx

    from bibr.utils.hosts import refuse_plaintext_llm_key

    endpoint = self_hosted_endpoint(settings)
    if endpoint is None:
        return None
    root, base = endpoint
    llm = settings.llm
    ollama_provider = llm.provider.strip().lower() == "ollama"
    headers: dict[str, str] = {}
    if not ollama_provider and llm.api_key:
        try:
            refuse_plaintext_llm_key(base, llm.api_key, allow_insecure_http=llm.allow_insecure_http)
        except ValueError:
            return None
        headers["Authorization"] = f"Bearer {llm.api_key}"
    timeout = httpx.Timeout(_PROBE_TIMEOUT_SECONDS, connect=_PROBE_CONNECT_TIMEOUT_SECONDS)
    try:
        async with httpx.AsyncClient(
            timeout=timeout, headers=headers, transport=transport, follow_redirects=False
        ) as client:
            if not ollama_provider:
                listing = await _get_json(client, "GET", f"{base}/models")
                data = listing.get("data") if isinstance(listing, dict) else None
                entries = [e for e in data or [] if isinstance(e, dict)]
                named = [e for e in entries if e.get("id") == llm.model]
                # A server that lists one model serves it under whatever name;
                # a router or proxy that lists several may hold other windows.
                if not named and len(entries) == 1:
                    named = entries
                if named and (window := _positive_int(named[0].get("max_model_len"))):
                    return ServerContext("vllm", window)
                props = await _get_json(client, "GET", f"{root}/props")
                if isinstance(props, dict):
                    slot = props.get("default_generation_settings")
                    window = _positive_int(slot.get("n_ctx") if isinstance(slot, dict) else None)
                    if window := window or _positive_int(props.get("n_ctx")):
                        return ServerContext("llama.cpp", window)
            version = await _get_json(client, "GET", f"{root}/api/version")
            if isinstance(version, dict) and "version" in version:
                return ServerContext("ollama", await _ollama_context(client, root, llm.model))
    except (httpx.HTTPError, httpx.InvalidURL, _NoAnswer, ValueError, TypeError) as exc:
        logger.debug("LLM server context probe at %s failed: %s", root, exc)
        return None
    return None if ollama_provider else ServerContext("other", None)


def apply_server_context(llm: LlmOptions, found: ServerContext) -> None:
    """Size *llm*'s inputs to a fixed-context server (Ollama, llama.cpp).

    The caps are :func:`fixed_context_caps` of the user's
    ``LLM_CONTEXT_TOKENS`` when set, else of the server's window, else of one
    learned from a rejection, else of :data:`ASSUMED_FIXED_CONTEXT_TOKENS`.
    ``context_tokens`` becomes the server's window unless the user set it.
    Fields the user set (:attr:`bibr.config.LlmOptions.user_set_fields`) are
    kept, with a warning when they exceed the cap of a known window. Called
    again with a later answer, it moves the caps it set to that answer, never
    above the values they replaced. vLLM and other servers are left as they
    are.
    """
    if found.server not in FIXED_CONTEXT_SERVERS:
        return
    user_set = llm.user_set_fields
    user_window = llm.context_tokens if "context_tokens" in user_set else None
    if user_window and found.context_tokens and user_window > found.context_tokens:
        logger.warning(
            "%s LLM server — LLM_CONTEXT_TOKENS=%d is larger than the %d-token context the "
            "server reports; requests past the server's context will fail or be cut.",
            found.server,
            user_window,
            found.context_tokens,
        )
    window = user_window or found.context_tokens or llm.context_tokens
    tier = window or ASSUMED_FIXED_CONTEXT_TOKENS
    replaced = llm.server_cap_replaced
    for field, cap in fixed_context_caps(tier):
        current = getattr(llm, field)
        if field in user_set:
            if window and current > cap:
                logger.warning(
                    "%s LLM server — explicit llm.%s=%s exceeds the %d-token context "
                    "budget (~%s); requests past the context will fail or be cut.",
                    found.server,
                    field,
                    current,
                    window,
                    cap,
                )
            continue
        target = min(replaced.get(field, current), cap)
        if target != current:
            replaced.setdefault(field, current)
            setattr(llm, field, target)
            logger.info(
                "%s LLM server — llm.%s set to %d for a %d-token context%s",
                found.server,
                field,
                target,
                tier,
                "" if window else " (assumed; the server did not say)",
            )
    if found.context_tokens and user_window is None:
        llm.context_tokens = found.context_tokens
        logger.info(
            "%s LLM server — context window %d tokens; long inputs are cut to fit it",
            found.server,
            found.context_tokens,
        )
