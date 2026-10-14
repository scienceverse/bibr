"""Claude models that reject sampling parameters (Haiku, Sonnet, Opus 5.5, Fable 5.1).

Their API answers any ``temperature`` (even 0) with a 400 ("`temperature` is
deprecated for this model.") and rejects budget-style thinking, so every call
bibr made to them failed. Requests to them now carry neither; requests to
every other model are unchanged.
"""

import logging

import pytest

from bibr.clients.providers import anthropic as anthropic_provider
from bibr.clients.providers.anthropic import AnthropicProvider
from bibr.config import GlobalSettings

SAMPLING_FREE = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-fable-5-1",
    "claude-haiku-5-5-20261001",  # dated snapshot ids match by prefix
]


def _provider(**llm):
    settings = GlobalSettings(llm={"provider": "anthropic", **llm})
    return AnthropicProvider(settings=settings)


@pytest.mark.parametrize("model", SAMPLING_FREE)
def test_no_temperature_for_sampling_free_models(model):
    provider = _provider(model=model)
    assert provider.call_kwargs(None, max_tokens=512) == {"max_tokens": 512}
    body = provider.call_kwargs("low")
    assert "temperature" not in body
    assert "thinking" not in body


@pytest.mark.parametrize("model", SAMPLING_FREE)
def test_thinking_budget_is_dropped_with_a_warning(model, caplog):
    anthropic_provider._warn_thinking_budget_ignored.cache_clear()
    provider = _provider(model=model, thinking_budget=4096)
    with caplog.at_level(logging.WARNING, logger="bibr.clients.providers.anthropic"):
        assert provider.call_kwargs(None, max_tokens=8192) == {"max_tokens": 8192}
        # A cap too small for the budget gets no temperature either.
        assert provider.call_kwargs(None, max_tokens=512) == {"max_tokens": 512}
    warnings = [r for r in caplog.records if "LLM_THINKING_BUDGET" in r.message]
    assert len(warnings) == 1
    assert model in warnings[0].message


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-1"])
def test_other_models_are_unchanged(model):
    provider = _provider(model=model)
    assert provider.call_kwargs(None, max_tokens=512) == {
        "max_tokens": 512,
        "temperature": 0.0,
    }
    provider = _provider(model=model, thinking_budget=2048)
    assert provider.call_kwargs(None, max_tokens=8192) == {
        "max_tokens": 8192,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    }
