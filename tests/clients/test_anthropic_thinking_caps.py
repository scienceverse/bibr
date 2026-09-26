"""Anthropic thinking budget vs per-task max_tokens caps (clients-llm-8).

Anthropic rejects extended thinking unless ``1024 <= budget_tokens <
max_tokens``, and thinking tokens come out of ``max_tokens``. The per-task
caps (paper_type 512, classification 1024, title/equations 4096) can sit at
or below a configured ``LLM_THINKING_BUDGET``, which used to produce a
request the API answers with a 400. Capped tasks now fall back to
temperature-0 calls unless the cap leaves 1024 tokens above the budget for
the answer; tasks with room keep byte-identical requests.
"""

from bibr.clients.providers.anthropic import AnthropicProvider
from bibr.config import GlobalSettings

CAPS = {
    "paper_type": 512,
    "classification": 1024,
    "title": 4096,
    "equations": 4096,
}


def _provider(**llm):
    settings = GlobalSettings(llm={"provider": "anthropic", **llm})
    return AnthropicProvider(settings=settings), settings


def test_capped_tasks_skip_thinking_instead_of_400():
    provider, _ = _provider(thinking_budget=4096)
    for task, cap in CAPS.items():
        body = provider.call_kwargs(None, max_tokens=cap)
        assert body["max_tokens"] == cap, task
        assert "thinking" not in body, task
        assert body["temperature"] == 0.0, task


def test_sub_minimum_budget_is_clamped_not_sent_raw():
    provider, _ = _provider(thinking_budget=512)
    body = provider.call_kwargs(None)  # default max_tokens 65536
    assert body == {
        "max_tokens": 65536,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }


def test_valid_requests_are_unchanged():
    provider, _ = _provider(thinking_budget=2048)
    assert provider.call_kwargs(None) == {
        "max_tokens": 65536,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    }


def test_thinking_disabled_sends_temperature():
    provider, _ = _provider(thinking_budget=0)
    assert provider.call_kwargs(None, max_tokens=512) == {
        "max_tokens": 512,
        "temperature": 0.0,
    }


def test_cap_just_above_budget_still_skips_thinking():
    # Budget 4000 with the 4096 title cap leaves 96 tokens for the answer.
    provider, _ = _provider(thinking_budget=4000)
    body = provider.call_kwargs(None, max_tokens=4096)
    assert body["max_tokens"] == 4096
    assert "thinking" not in body
    assert body["temperature"] == 0.0


def test_thinking_margin_boundary():
    provider, _ = _provider(thinking_budget=4000)
    # Exactly budget + 1024 leaves room for the answer: thinking is sent.
    body = provider.call_kwargs(None, max_tokens=5024)
    assert body == {
        "max_tokens": 5024,
        "thinking": {"type": "enabled", "budget_tokens": 4000},
    }
    # One token less falls back to temperature 0.
    body = provider.call_kwargs(None, max_tokens=5023)
    assert "thinking" not in body
    assert body["temperature"] == 0.0


def test_raised_budget_margin_boundary(caplog):
    import logging

    provider, _ = _provider(thinking_budget=512)
    with caplog.at_level(logging.INFO, logger="bibr.clients.providers.anthropic"):
        body = provider.call_kwargs(None, max_tokens=2048)
    assert body == {
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    assert any("512" in record.message and "1024" in record.message for record in caplog.records)
    # 2047 is one token short of the raised budget + margin.
    body = provider.call_kwargs(None, max_tokens=2047)
    assert "thinking" not in body
    assert body["temperature"] == 0.0
