import re
from unittest import mock

import pytest

from bibr.clients.prompts import prompt_text


@pytest.fixture
def captured_prompt(monkeypatch):
    """Replace _invoke_structured to capture the assembled prompt."""
    from bibr.clients.llm import LLMClient

    captured = {"messages": None}

    async def fake_invoke(self, response_model, messages, system_prompt, **kw):
        captured["messages"] = messages
        out = mock.MagicMock()
        out.matches = []
        out.equations = []
        return out

    # Bypass rate limiter
    class _NoopLimiter:
        async def acquire(self):
            return None

        async def close(self):
            return None

    monkeypatch.setattr(LLMClient, "_invoke_structured", fake_invoke)
    monkeypatch.setattr(LLMClient, "limiter", property(lambda self: _NoopLimiter()))
    return captured


async def test_resolve_citations_uses_uuid_boundary(captured_prompt):
    from bibr.clients.llm import LLMClient

    c = LLMClient()
    await c.resolve_citations(
        ambiguous_citations=[(1, "(Smith, 2020)")],
        reference_summary=[{"bib_id": 1, "author": "Smith", "year": "2020", "title": "T"}],
    )
    content = prompt_text(captured_prompt["messages"][0]["content"])
    assert re.search(r"[0-9a-f]{32}", content), (
        "expected a UUID boundary marker in the resolve_citations prompt"
    )


async def test_extract_equations_uses_uuid_boundary(captured_prompt):
    from bibr.clients.llm import LLMClient

    c = LLMClient()
    await c.extract_equations(sentences=[(1, "t(28) = 3.42, p = .003")])
    content = prompt_text(captured_prompt["messages"][0]["content"])
    assert re.search(r"[0-9a-f]{32}", content)
    assert "--- START OF USER CONTENT ---" not in content
