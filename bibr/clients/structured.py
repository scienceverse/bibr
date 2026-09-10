"""StructuredBackend — the transport seam under :class:`bibr.clients.llm.LLMClient`.

A backend performs exactly one structured-output call: task in (system +
user messages + response model), validated pydantic instance out. Everything
operational — rate limiting, concurrency gating, circuit breaking, transient
retries, timeouts, usage accounting — stays in ``LLMClient``, wrapped around
whichever backend is injected.

The default is ``bibr.clients.llm.InstructorBackend`` (Instructor chat
clients: google/openai/anthropic/groq/ollama). Alternate engines that don't
speak Instructor's chat/tool dialect — e.g. NuExtract 3's ``【template】``
DSL — implement this protocol and are passed to ``LLMClient(backend=...)``;
prompt text and response models come from ``bibr.clients.prompts.PROMPTS``
so backends render the same task definitions instead of drifting copies.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StructuredBackend(Protocol):
    """One structured-output call. Implementations must be side-effect free
    across calls (LLMClient may invoke ``create`` several times per task
    under its transient-retry loop)."""

    async def create(
        self,
        *,
        response_model: type,
        system: str,
        messages: list[dict],
        want_completion: bool,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        client_override: Any = None,
        on_protocol_hashes: Callable[[dict[str, str]], None] | None = None,
    ) -> tuple[Any, Any | None]:
        """Return ``(validated_model, raw_completion_or_None)``.

        ``raw_completion`` (when ``want_completion`` is true and the engine
        reports usage) is fed to the caller's usage accounting; return
        ``None`` when unavailable.
        """
        ...
