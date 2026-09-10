"""Every provider must extract at temperature 0.

Title/author/DOI/reference extraction must not vary run to run. ``google``,
``groq`` and ``ollama`` pin 0.0 outright; ``openai`` forwards
``LLM_TEMPERATURE`` (itself 0.0 by default). The Anthropic adapter forwarded
nothing at all, so it alone sampled at the Anthropic API default of 1.0. The
one legitimate exception is extended thinking, which is only accepted at the
API's own default temperature.
"""

import pytest

from bibr.clients import providers
from bibr.config import GlobalSettings

# Providers whose ``call_kwargs`` carry a plain ``temperature`` key.
_PLAIN = ["openai", "anthropic", "groq", "ollama"]


def _settings(**llm):
    return GlobalSettings(llm=llm)


@pytest.mark.parametrize("name", _PLAIN)
def test_provider_extracts_deterministically_by_default(name):
    provider = providers.get(name, settings=_settings(provider=name))
    assert provider.call_kwargs(None).get("temperature") == 0.0


def test_google_pins_temperature_zero():
    provider = providers.get("google", settings=_settings(provider="google"))
    assert provider.call_kwargs(None)["generation_config"]["temperature"] == 0.0


@pytest.mark.parametrize("name", ["openai", "anthropic"])
def test_configured_temperature_is_forwarded(name):
    """The two adapters that honour ``LLM_TEMPERATURE`` must actually send it."""
    provider = providers.get(name, settings=_settings(provider=name, temperature=0.7))
    assert provider.call_kwargs(None).get("temperature") == 0.7


def test_anthropic_omits_temperature_only_when_thinking_is_enabled():
    settings = _settings(provider="anthropic", thinking_budget=2048)
    kwargs = providers.get("anthropic", settings=settings).call_kwargs(None)

    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    # Extended thinking rejects an explicit temperature.
    assert "temperature" not in kwargs


def test_local_chat_template_options_are_forwarded_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
    settings = _settings(provider="openai", base_url="http://localhost:8000/v1")
    provider = providers.get("openai", settings=settings)
    kwargs = provider.call_kwargs(None)
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
    assert provider.call_kwargs(None)["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_real_openai_does_not_receive_local_chat_template_options():
    settings = _settings(provider="openai", chat_template_kwargs={"enable_thinking": False})
    assert "extra_body" not in providers.get("openai", settings=settings).call_kwargs(None)
