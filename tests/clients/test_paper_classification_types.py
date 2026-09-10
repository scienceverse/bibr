"""Paper-type enumeration kept in sync between prompt, schema, and validator."""

from unittest import mock

from bibr.clients.llm import LLMClient
from bibr.schemas import PaperClassificationLLM

NOTICE_TYPES = ("corrigendum", "erratum", "retraction")


async def test_classification_prompt_enumerates_notice_types():
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
    for t in NOTICE_TYPES:
        assert t in captured["content"]


def test_paper_type_field_description_includes_notice_types():
    desc = PaperClassificationLLM.model_fields["paper_type"].description
    for t in NOTICE_TYPES:
        assert f"'{t}'" in desc
