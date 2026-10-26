"""The context window of a self-hosted LLM server, read from the server itself.

Ollama and llama.cpp serve a fixed context (Ollama's ``num_ctx``, llama.cpp's
``-c``). Given a longer prompt, llama.cpp and Ollama's GGUF-template models
answer HTTP 400, and Ollama's library models cut the prompt from the front,
dropping the instructions. bibr's input cap is sized for cloud models
(``LLM_MAX_INPUT_CHARS`` 300,000 characters), so on a 16,384-token Ollama a long
front matter failed the paper or came back without its title and authors.

:func:`probe_server_context` asks the server once, before the first prompt:

- vLLM lists ``max_model_len`` under ``/v1/models``;
- llama.cpp's ``/props`` gives ``default_generation_settings.n_ctx``;
- Ollama answers ``/api/version``; ``/api/ps`` gives a loaded model's
  ``context_length``, and ``/api/show`` a ``num_ctx`` the model sets. The
  ``<arch>.context_length`` in ``/api/show`` is the trained maximum, not the
  context Ollama runs, so it is not used.

:func:`apply_server_context` then sets ``llm.context_tokens`` and, for Ollama
and llama.cpp, the caps the managed llama.cpp server uses. Values the user set
are never replaced.
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

_PROBE_TIMEOUT_SECONDS = 5.0
_PROBE_CONNECT_TIMEOUT_SECONDS = 2.0
_NUM_CTX_RE = re.compile(r"^\s*num_ctx\s+(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ServerContext:
    """What a server said about its context window."""

    server: Literal["ollama", "llama.cpp", "vllm"]
    # ``None``: an Ollama whose model is not loaded and sets no ``num_ctx``.
    context_tokens: int | None


def fixed_context_caps(context_tokens: int) -> tuple[tuple[str, int], ...]:
    """``llm`` caps that keep prompt plus completion inside a fixed context.

    They equal the historical llama.cpp constants at a 16,384-token context
    (completion at most a quarter of it) and scale with a larger one.
    """
    return (
        ("max_tokens", context_tokens // 4),
        ("max_input_chars", 24_000 * context_tokens // 16_384),
        ("ref_seg_window_chars", 12_000 * context_tokens // 16_384),
    )


def _endpoint(settings: GlobalSettings) -> tuple[str, str] | None:
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
    """Ollama's name for *model*: an untagged name means ``:latest``."""
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


async def _get_json(client: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> Any:
    """The JSON body of a 200 answer, or ``None`` for any other answer."""
    response = await client.request(method, url, **kwargs)
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


async def _ollama_context(client: httpx.AsyncClient, root: str, model: str) -> int | None:
    name = _ollama_name(model)
    ps = await _get_json(client, "GET", f"{root}/api/ps")
    for entry in (ps or {}).get("models") or []:
        if (
            isinstance(entry, dict)
            and name in (entry.get("name"), entry.get("model"))
            and (window := _positive_int(entry.get("context_length")))
        ):
            return window
    show = await _get_json(client, "POST", f"{root}/api/show", json={"model": model})
    parameters = (show or {}).get("parameters")
    match = _NUM_CTX_RE.search(parameters) if isinstance(parameters, str) else None
    return int(match.group(1)) if match else None


async def probe_server_context(
    settings: GlobalSettings, *, transport: httpx.AsyncBaseTransport | None = None
) -> ServerContext | None:
    """Ask a self-hosted LLM server for its context window.

    ``None`` for a cloud provider, a server that is down, or one that says
    nothing about its context. Never raises: the extraction call that follows
    reports a broken endpoint itself.
    """
    import httpx

    from bibr.utils.hosts import refuse_plaintext_llm_key

    endpoint = _endpoint(settings)
    if endpoint is None:
        return None
    root, base = endpoint
    llm = settings.llm
    headers: dict[str, str] = {}
    if llm.provider.strip().lower() == "openai" and llm.api_key:
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
            if llm.provider.strip().lower() != "ollama":
                listing = await _get_json(client, "GET", f"{base}/models")
                entries = [e for e in (listing or {}).get("data") or [] if isinstance(e, dict)]
                named = [e for e in entries if e.get("id") == llm.model] or entries[:1]
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
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        logger.debug("LLM server context probe at %s failed: %s", root, exc)
    return None


def apply_server_context(llm: LlmOptions, found: ServerContext) -> None:
    """Size *llm*'s inputs to the server's context window.

    Sets ``context_tokens`` when the window is known. For Ollama and llama.cpp
    it also applies :func:`fixed_context_caps`, at
    :data:`ASSUMED_FIXED_CONTEXT_TOKENS` when Ollama did not say: those servers
    reject or cut a longer prompt rather than refuse to start. Fields the user
    set are kept, with a warning when they cannot fit the window.
    """
    # Assigning a field marks it as set, so take the user's set first.
    user_set = set(llm.model_fields_set)
    window = found.context_tokens
    if found.server in ("ollama", "llama.cpp"):
        tier = window or ASSUMED_FIXED_CONTEXT_TOKENS
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
            if current > cap:
                setattr(llm, field, cap)
                logger.info(
                    "%s LLM server — capping llm.%s to %d for a %d-token context%s",
                    found.server,
                    field,
                    cap,
                    tier,
                    "" if window else " (assumed; the server did not say)",
                )
    if window and "context_tokens" not in user_set:
        llm.context_tokens = window
        logger.info(
            "%s LLM server — context window %d tokens; long inputs are cut to fit it",
            found.server,
            window,
        )
