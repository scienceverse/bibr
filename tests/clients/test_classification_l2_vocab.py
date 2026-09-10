"""OECD L2 subdomain vocabulary must be spelled out in the classification prompt.

On the local vLLM path (Instructor Mode.JSON_SCHEMA) the model never sees pydantic
field descriptions -- the schema is only a decoding grammar there. The prompt text
is the only channel every backend sees, so the L2 label vocabulary must be built
directly from OECD_L2_MAP (single source of truth) instead of deferred to "the
schema", which is a dead reference on that path.
"""

from unittest import mock

from bibr.clients.llm import LLMClient
from bibr.clients.prompts import _CLASSIFICATION_PROMPT
from bibr.schemas import PaperClassificationLLM
from bibr.structure.paper_classifier import OECD_L2_MAP


def test_classification_prompt_has_no_dead_schema_deferral():
    assert "see schema" not in _CLASSIFICATION_PROMPT.lower()


def test_classification_prompt_contains_every_l2_label():
    for labels in OECD_L2_MAP.values():
        for label in labels:
            assert label in _CLASSIFICATION_PROMPT


async def test_classification_prompt_rendered_contains_l2_vocab():
    client = LLMClient()
    client._limiter = mock.Mock()
    client._limiter.acquire = mock.AsyncMock()
    captured = {}

    async def fake_invoke(response_model, messages, system_prompt, **kw):
        from bibr.clients.prompts import prompt_text

        captured["content"] = prompt_text(messages[0]["content"])
        return PaperClassificationLLM()

    with mock.patch.object(
        client, "_invoke_structured", new=mock.AsyncMock(side_effect=fake_invoke)
    ):
        await client.extract_paper_classification("some paper text")

    assert "see schema" not in captured["content"].lower()
    for labels in OECD_L2_MAP.values():
        for label in labels:
            assert label in captured["content"]
