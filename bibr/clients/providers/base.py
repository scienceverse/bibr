"""`LlmProvider` Protocol — per-provider adapter contract.

Each provider module (`google`, `openai`, `anthropic`, `groq`, `ollama`)
exports a class satisfying this Protocol and calls
``bibr.clients.providers.register`` at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    import instructor


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
