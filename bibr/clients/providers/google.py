"""Google provider adapter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import instructor

from bibr.clients.providers import register
from bibr.config import snapshot_settings

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

logger = logging.getLogger(__name__)

# Gemini constrains ``thinkingConfig.thinkingBudget`` per model, and an illegal
# value is a hard 400 INVALID_ARGUMENT — which fails *every* LLM call in the
# pipeline, not just the one. Two constraints exist, and a blanket 0 violates
# the first:
#
#   * Newer models cannot disable thinking at all. This includes
#     ``gemini-3.5-flash-lite``, which is bibr's own default model, so the
#     stock google configuration was dead on arrival until this was resolved.
#   * Some models publish a minimum well above 1 — ``gemini-2.5-flash-lite``
#     rejects everything below 512, so clamping the first case up to 1
#     globally would break the second.
#
# Prefixes, so pinned point releases (``…-001``) inherit their family's rules.
# Verified against the live API 2026-07-25.
_NO_THINKING_DISABLE: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
)
_MIN_THINKING_BUDGET: dict[str, int] = {
    "gemini-2.5-flash-lite": 512,
}


def _resolve_thinking_budget(model: str, budget: int) -> int:
    """Map a configured ``budget`` onto a value ``model`` will actually accept.

    A negative budget means "let the model decide" (dynamic thinking) and is
    passed through as ``-1``; every Gemini model accepts it. Zero means
    "disabled" and is honoured wherever the model supports it, falling back to
    the cheapest legal budget where it does not. Positive budgets are raised to
    the model's published minimum rather than sent through to a 400.
    """
    model = (model or "").strip()
    if budget < 0:
        return -1
    minimum = next(
        (v for k, v in _MIN_THINKING_BUDGET.items() if model.startswith(k)),
        0,
    )
    if budget == 0:
        # "Disabled" must stay disabled wherever the model allows it — the
        # minimum applies to real thinking budgets, not to the off switch.
        if not any(model.startswith(p) for p in _NO_THINKING_DISABLE):
            return 0
        resolved = max(1, minimum)
        logger.debug(
            "%s cannot disable thinking — using the minimum budget %d instead of 0",
            model,
            resolved,
        )
        return resolved
    return max(budget, minimum)


@register
class GoogleProvider:
    name: ClassVar[str] = "google"

    def __init__(self, settings: GlobalSettings | None = None) -> None:
        self._settings = settings if settings is not None else snapshot_settings()

    def build_client(self) -> instructor.AsyncInstructor:
        api_key = self._settings.llm.api_key or self._settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError(
                "Google API key required. Set LLM_API_KEY or GOOGLE_API_KEY environment variable."
            )
        model_string = f"google/{self._settings.llm.model}"
        return instructor.from_provider(model_string, async_client=True, api_key=api_key)

    def call_kwargs(self, reasoning_effort: str | None, max_tokens: int | None = None) -> dict:  # noqa: ARG002
        # gemini-3 family models default to thinking-on, which causes the
        # model to emit a reasoning trace alongside the structured tool call.
        # Instructor sees the trace + the tool call as multiple function
        # calls and rejects with AssertionError. Always pass an explicit
        # thinking_config (default 0 = disabled) so behavior is deterministic
        # across model versions; raise budget via Settings.llm.thinking_budget
        # when reasoning is genuinely wanted, or set it negative for dynamic
        # thinking. The request is resolved against the model's own limits —
        # see _resolve_thinking_budget.
        budget = _resolve_thinking_budget(
            self._settings.llm.model, self._settings.llm.thinking_budget
        )
        return {
            "generation_config": {
                # Deterministic extraction: temperature 0 (matches the OpenAI
                # adapter). Title/author/DOI/reference extraction must not vary
                # run-to-run; sampling at temp 1.0 was an accuracy/consistency leak.
                "temperature": 0.0,
                "max_tokens": max_tokens or self._settings.llm.max_tokens,
            },
            "thinking_config": {"thinking_budget": budget},
        }
