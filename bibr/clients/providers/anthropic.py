"""Anthropic provider adapter."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, ClassVar

import instructor

from bibr.clients.providers import register
from bibr.clients.providers.base import bound_sdk_client
from bibr.config import snapshot_settings

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

logger = logging.getLogger(__name__)

# Anthropic rejects extended thinking unless 1024 <= budget_tokens <
# max_tokens, and thinking tokens come out of max_tokens. The per-task caps
# in ``Settings.llm`` (paper_type 512, classification 1024, title/equations
# 4096) can be at or below a configured ``LLM_THINKING_BUDGET``, which the
# API answers with a 400 — and a cap just above the budget leaves almost no
# room for the answer itself.
_MIN_THINKING_BUDGET = 1024

# Minimum answer room kept above the thinking budget before thinking is
# sent. A 4096 cap with a 4000 budget would otherwise leave 96 tokens for
# the JSON answer and risk truncation.
_THINKING_OUTPUT_MARGIN = 1024

# The adapter does not stream, so a call's whole answer comes back in one HTTP
# response. The default LLM_MAX_TOKENS (65536) is above the output cap of
# Haiku 4.5 (64000; the setup default) and far above the ~16K Anthropic
# advises for a non-streaming call; on its default timeout the SDK refuses
# anything above ~21K outright. Calls without a task cap (reference
# segmentation, merged core metadata) failed under LLM_PROVIDER=anthropic.
_NONSTREAMING_MAX_TOKENS = 16_384


def _nonstreaming_cap(model: str) -> int:
    """Output cap of one non-streaming call to *model*.

    Also within the SDK's own per-model limit (8192 for Opus 4 and 4.1),
    which it enforces on a client left at its default timeout.
    """
    try:
        from anthropic._constants import MODEL_NONSTREAMING_TOKENS
    except ImportError:  # a private table; absent from other SDK versions
        return _NONSTREAMING_MAX_TOKENS
    return min(
        _NONSTREAMING_MAX_TOKENS, MODEL_NONSTREAMING_TOKENS.get(model, _NONSTREAMING_MAX_TOKENS)
    )


@functools.cache
def _warn_budget_never_fits(budget: int, cap: int) -> None:
    # Once per budget: the configuration, not the call, is at fault.
    logger.warning(
        "LLM_THINKING_BUDGET %d leaves no room for an answer within the %d output "
        "tokens of one Anthropic call; calls are sent without thinking",
        budget,
        cap,
    )


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
        client = instructor.from_provider(
            f"anthropic/{self._settings.llm.model}", async_client=True, api_key=api_key
        )
        return bound_sdk_client(client, self._settings)

    def call_kwargs(self, reasoning_effort: str | None, max_tokens: int | None = None) -> dict:  # noqa: ARG002
        cap = _nonstreaming_cap(self._settings.llm.model)
        requested = max_tokens or self._settings.llm.max_tokens
        kwargs: dict = {"max_tokens": min(requested, cap)}
        budget = self._settings.llm.thinking_budget
        if budget and budget > 0:
            effective = max(int(budget), _MIN_THINKING_BUDGET)
            if int(budget) < _MIN_THINKING_BUDGET:
                logger.info(
                    "LLM_THINKING_BUDGET %d is below Anthropic's minimum %d — using %d",
                    int(budget),
                    _MIN_THINKING_BUDGET,
                    effective,
                )
            if kwargs["max_tokens"] < effective + _THINKING_OUTPUT_MARGIN:
                # The task cap cannot fit thinking plus room for the answer
                # (budgets must stay below max_tokens) — send an ordinary
                # temperature-0 call rather than a request the API rejects
                # with a 400 or truncates.
                if effective + _THINKING_OUTPUT_MARGIN > cap:
                    _warn_budget_never_fits(effective, cap)
                logger.debug(
                    "thinking budget %d does not fit max_tokens %d — sending without thinking",
                    effective,
                    kwargs["max_tokens"],
                )
                kwargs["temperature"] = self._settings.llm.temperature
            else:
                # Extended thinking only runs at the API's default temperature —
                # sending any explicit value alongside it is rejected.
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": effective}
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
