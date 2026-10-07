"""`LlmProvider` Protocol — per-provider adapter contract.

Each provider module (`google`, `openai`, `anthropic`, `groq`, `ollama`)
exports a class satisfying this Protocol and calls
``bibr.clients.providers.register`` at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    import instructor

    from bibr.config import GlobalSettings


def sdk_timeout_seconds(settings: GlobalSettings) -> float:
    """Timeout for one SDK request: the limit ``LLMClient`` puts on a call.

    Twice ``LLM_TIMEOUT_SECONDS``, the hard timeout of one attempt, so a
    request is bounded by the SDK itself without cutting short one that
    bibr's own limit allows.
    """
    return 2.0 * float(settings.llm.timeout_seconds)


def bound_sdk_client(client: Any, settings: GlobalSettings) -> Any:
    """Make the SDK client under an Instructor client send each request once.

    The OpenAI, Anthropic and Groq SDKs retry 408/409/429/5xx and dropped
    connections twice on their own and time out after ten minutes. bibr
    retries those failures itself, through the shared rate limiter and the
    circuit breaker, so SDK retries turned one call into up to nine requests
    that neither saw. These SDKs read both attributes on every request.
    """
    raw = getattr(client, "client", None)
    if raw is not None and hasattr(raw, "max_retries"):
        raw.max_retries = 0
        raw.timeout = sdk_timeout_seconds(settings)
    return client


@runtime_checkable
class LlmProvider(Protocol):
    """Per-provider adapter for instructor client + call kwargs."""

    name: ClassVar[str]

    def build_client(self) -> instructor.AsyncInstructor:
        """Construct a configured async instructor client."""
        ...

    def call_kwargs(self, reasoning_effort: str | None, max_tokens: int | None = None) -> dict:
        """Return per-call kwargs for ``client.create()``.

        ``max_tokens`` overrides the provider's default output cap for this one
        call (e.g. a smaller per-batch cap for reference parsing); ``None`` keeps
        the configured ``LLM_MAX_TOKENS``.
        """
        ...

    # Optional hook: providers may declare a ``transform_messages`` method
    # to rewrite the messages list (e.g. Anthropic prompt-cache markers).
    # Callers must check via ``hasattr`` because this is not part of the
    # required surface — only the Anthropic adapter currently implements it.
