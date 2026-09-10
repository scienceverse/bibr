"""Anthropic provider adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import instructor

from bibr.clients.providers import register
from bibr.config import snapshot_settings

if TYPE_CHECKING:
    from bibr.config import GlobalSettings


@register
class AnthropicProvider:
    name: ClassVar[str] = "anthropic"

    def __init__(self, settings: GlobalSettings | None = None) -> None:
        self._settings = settings if settings is not None else snapshot_settings()

    def build_client(self) -> instructor.AsyncInstructor:
        api_key = self._settings.llm.api_key or self._settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError(
                "Anthropic API key required. "
                "Set LLM_API_KEY or ANTHROPIC_API_KEY environment variable."
            )
        return instructor.from_provider(
            f"anthropic/{self._settings.llm.model}", async_client=True, api_key=api_key
        )

    def call_kwargs(self, reasoning_effort: str | None, max_tokens: int | None = None) -> dict:  # noqa: ARG002
        kwargs: dict = {"max_tokens": max_tokens or self._settings.llm.max_tokens}
        budget = self._settings.llm.thinking_budget
        if budget and budget > 0:
            # Extended thinking only runs at the API's default temperature —
            # sending any explicit value alongside it is rejected.
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            # Every other provider extracts at temperature 0 — google/groq/
            # ollama pin it, openai forwards ``LLM_TEMPERATURE`` (0.0 by
            # default). Omitting it here silently sampled at Anthropic's API
            # default of 1.0, so the same paper could yield different metadata
            # run to run.
            kwargs["temperature"] = self._settings.llm.temperature
        return kwargs

    def transform_messages(self, messages: list[dict]) -> list[dict]:
        """Place prompt-cache breakpoints on the cacheable content.

        The builders mark the large shared prefix of each user prompt
        (see :func:`bibr.clients.prompts.part`): the fenced document for the
        per-paper front-matter fan-out, the static instruction for batch
        tasks like reference parsing. Those parts become text blocks with
        ``cache_control`` so repeat calls reuse the prefix. The ~10-token
        system prompts are far below Anthropic's minimum cacheable length,
        so no breakpoint is spent on them. Prefixes under the per-model
        minimum are silently not cached (no error).
        """
        out: list[dict] = []
        for msg in messages:
            content = msg.get("content")
            if (
                isinstance(content, list)
                and content
                and all(isinstance(p, dict) and "cache" in p for p in content)
            ):
                blocks: list[dict] = []
                for p in content:
                    block = {"type": "text", "text": p["text"]}
                    if p["cache"]:
                        block["cache_control"] = {"type": "ephemeral"}
                    blocks.append(block)
                out.append({**msg, "content": blocks})
            else:
                out.append(msg)
        return out
