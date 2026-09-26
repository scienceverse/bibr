"""Ollama provider adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import instructor

from bibr.clients.providers import register
from bibr.config import snapshot_settings

if TYPE_CHECKING:
    from bibr.config import GlobalSettings


def ollama_openai_base_url(url: str) -> str:
    """The OpenAI-compatible API root of the Ollama server at ``url``.

    ``LLM_OLLAMA_BASE_URL`` names the server (``http://localhost:11434`` by
    default, and what ``bibr setup`` writes), but Ollama serves its
    OpenAI-compatible routes under ``/v1``. Instructor uses the base URL as
    given, so the bare server URL sent every request to ``/chat/completions``,
    which Ollama answers with 404. Both forms are accepted: ``/v1`` is appended
    unless the URL already ends in it.
    """
    base = url.strip().rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


@register
class OllamaProvider:
    name: ClassVar[str] = "ollama"

    def __init__(self, settings: GlobalSettings | None = None) -> None:
        self._settings = settings if settings is not None else snapshot_settings()

    def build_client(self) -> instructor.AsyncInstructor:
        return instructor.from_provider(
            f"ollama/{self._settings.llm.model}",
            async_client=True,
            base_url=ollama_openai_base_url(self._settings.llm.ollama_base_url),
        )

    def call_kwargs(self, reasoning_effort: str | None, max_tokens: int | None = None) -> dict:
        del reasoning_effort  # Ollama's OpenAI-compatible path has no equivalent setting.
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._settings.llm.max_tokens
        )
        return {
            "temperature": 0.0,
            "extra_body": {"options": {"num_predict": effective_max_tokens}},
        }
