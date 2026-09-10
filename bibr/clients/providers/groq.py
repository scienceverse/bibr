"""Groq provider adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import instructor

from bibr.clients.providers import register
from bibr.config import snapshot_settings

if TYPE_CHECKING:
    from bibr.config import GlobalSettings


@register
class GroqProvider:
    name: ClassVar[str] = "groq"

    def __init__(self, settings: GlobalSettings | None = None) -> None:
        self._settings = settings if settings is not None else snapshot_settings()

    def build_client(self) -> instructor.AsyncInstructor:
        api_key = self._settings.llm.api_key or self._settings.GROQ_API_KEY
        if not api_key:
            raise ValueError(
                "Groq API key required. Set LLM_API_KEY or GROQ_API_KEY environment variable."
            )
        return instructor.from_provider(
            f"groq/{self._settings.llm.model}", async_client=True, api_key=api_key
        )

    def call_kwargs(self, reasoning_effort: str | None, max_tokens: int | None = None) -> dict:  # noqa: ARG002
        return {
            "temperature": 0.0,
            "max_tokens": max_tokens or self._settings.llm.max_tokens,
        }
