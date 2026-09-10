"""Gemini thinking-budget resolution.

An illegal ``thinkingBudget`` is a 400 INVALID_ARGUMENT that kills every LLM
call in the pipeline, so the cases below are pinned to what the live API
actually accepted on 2026-07-25 rather than to what the docs imply.
"""

from bibr.clients.providers.google import GoogleProvider, _resolve_thinking_budget


class _Llm:
    def __init__(self, model, thinking_budget, max_tokens=4096):
        self.model = model
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.api_key = None


class _Settings:
    def __init__(self, model, thinking_budget):
        self.llm = _Llm(model, thinking_budget)
        self.GOOGLE_API_KEY = "test-key"


def test_default_model_cannot_disable_thinking_so_zero_becomes_legal():
    # gemini-3.5-flash-lite is bibr's default model and rejects budget 0.
    assert _resolve_thinking_budget("gemini-3.5-flash-lite", 0) == 1


def test_models_that_support_disabling_keep_zero():
    for model in ("gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"):
        assert _resolve_thinking_budget(model, 0) == 0


def test_zero_is_honoured_on_a_model_that_can_disable_despite_a_high_minimum():
    # 2.5-flash-lite accepts 0 (disabled) even though its minimum *budget* is 512.
    assert _resolve_thinking_budget("gemini-2.5-flash-lite", 0) == 0


def test_positive_budget_is_raised_to_the_model_minimum():
    # A blanket clamp-to-1 would 400 here; 2.5-flash-lite rejects everything <512.
    assert _resolve_thinking_budget("gemini-2.5-flash-lite", 1) == 512
    assert _resolve_thinking_budget("gemini-2.5-flash-lite", 511) == 512
    assert _resolve_thinking_budget("gemini-2.5-flash-lite", 1024) == 1024


def test_negative_budget_means_dynamic_and_is_no_longer_clamped_away():
    # The previous max(0, ...) made dynamic thinking unreachable.
    for model in ("gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-3.5-flash"):
        assert _resolve_thinking_budget(model, -1) == -1
        assert _resolve_thinking_budget(model, -999) == -1


def test_floating_latest_aliases_are_treated_as_undisableable():
    for model in ("gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.6-flash"):
        assert _resolve_thinking_budget(model, 0) == 1


def test_pinned_point_releases_inherit_family_rules():
    assert _resolve_thinking_budget("gemini-3.5-flash-lite-001", 0) == 1
    assert _resolve_thinking_budget("gemini-2.5-flash-lite-001", 1) == 512


def test_unknown_models_keep_the_configured_budget():
    assert _resolve_thinking_budget("some-future-model", 0) == 0
    assert _resolve_thinking_budget("some-future-model", 256) == 256
    assert _resolve_thinking_budget("", 0) == 0


def test_call_kwargs_emits_a_legal_budget_for_the_default_model():
    kwargs = GoogleProvider(_Settings("gemini-3.5-flash-lite", 0)).call_kwargs(None)
    assert kwargs["thinking_config"] == {"thinking_budget": 1}
    assert kwargs["generation_config"]["temperature"] == 0.0


def test_call_kwargs_still_disables_thinking_where_the_model_allows_it():
    kwargs = GoogleProvider(_Settings("gemini-3.1-flash-lite", 0)).call_kwargs(None)
    assert kwargs["thinking_config"] == {"thinking_budget": 0}
