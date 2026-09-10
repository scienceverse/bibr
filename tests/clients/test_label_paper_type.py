"""Tests for LLMClient.label_paper_type — the paper_type-only fallback tier."""

from __future__ import annotations

from unittest import mock

import pytest

from bibr.clients.llm import LLMClient
from bibr.clients.prompts import PROMPTS, prompt_text
from bibr.config import snapshot_settings
from bibr.schemas import PaperTypeLabel


def _make_client() -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client._settings = snapshot_settings()
    client._limiter = mock.AsyncMock()
    client._invoke_structured = mock.AsyncMock(
        return_value=PaperTypeLabel(paper_type="meta-analysis", confidence=0.82)
    )
    client._track_usage = False
    return client


async def test_label_paper_type_reuses_paper_type_label_prompt():
    client = _make_client()
    result = await client.label_paper_type("A meta-analytic review", "We pooled 42 studies.")

    assert isinstance(result, PaperTypeLabel)
    assert result.paper_type == "meta-analysis"
    assert result.confidence == pytest.approx(0.82)

    client._invoke_structured.assert_awaited_once()
    args, kwargs = client._invoke_structured.call_args
    # response_model is the paper_type_label spec's model.
    spec = PROMPTS["paper_type_label"]
    assert args[0] is spec.response_model is PaperTypeLabel
    # System prompt matches the spec.
    assert args[2] == spec.system
    # The user message embeds the title + abstract via the spec builder.
    user_content = args[1][0]["content"]
    text = prompt_text(user_content) if isinstance(user_content, list) else user_content
    assert "A meta-analytic review" in text
    assert "We pooled 42 studies." in text


async def test_label_paper_type_acquires_limiter():
    """The slot is taken inside ``_invoke_structured``, below its cache check,
    so drive the real funnel with a stub backend rather than stubbing the
    funnel itself — otherwise the acquisition under test is mocked away."""

    class _Backend:
        async def create(self, **kwargs):  # noqa: ARG002
            return PaperTypeLabel(paper_type="meta-analysis", confidence=0.82), None

    client = LLMClient(settings=snapshot_settings(), backend=_Backend())
    client._limiter = mock.AsyncMock()

    await client.label_paper_type("T", "A")

    client._limiter.acquire.assert_awaited_once()
