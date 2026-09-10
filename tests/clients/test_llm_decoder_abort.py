"""A failed local grammar gets one independently validated recovery route."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import instructor
import pytest
from instructor.core.exceptions import InstructorRetryException
from openai import AsyncOpenAI

from bibr.clients import llm
from bibr.config import GlobalSettings
from bibr.schemas import TitleKeywordsLLM


@pytest.mark.parametrize("recovery_valid", [True, False])
async def test_decoder_abort_recovers_without_server_grammar(monkeypatch, recovery_valid):
    requests = []

    async def respond(request):
        requests.append(json.loads(request.content))
        content = '{"title":"A printed title","abstract":null}'
        abort = len(requests) == 1 or not recovery_valid
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{{"' if abort else content},
                        "finish_reason": "abort" if abort else "stop",
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:

        def create_client(mode_override=None, **_kwargs):
            sdk = AsyncOpenAI(api_key="test", base_url="http://localhost/v1", http_client=transport)
            return instructor.from_openai(
                sdk, mode=mode_override or instructor.Mode.JSON_SCHEMA, model="test"
            )

        monkeypatch.setattr(llm, "_create_client", create_client)
        client = llm.LLMClient(
            settings=GlobalSettings(
                llm={
                    "provider": "openai",
                    "base_url": "http://localhost/v1",
                    "max_retries": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
        )
        monkeypatch.setattr(client, "_acquire_rate_limit", AsyncMock())
        call = client._invoke_structured(
            TitleKeywordsLLM,
            [{"role": "user", "content": "A printed title"}],
            "Extract the metadata.",
            max_tokens=128,
        )
        if recovery_valid:
            result = await call
            assert result.title == "A printed title"
            assert result._abstract_explicitly_absent
        else:
            with pytest.raises(InstructorRetryException):
                await call
        assert len(requests) == 2
        assert requests[0]["response_format"]["type"] == "json_schema"
        assert "response_format" not in requests[1]
        assert requests[1]["max_tokens"] == 128
        assert requests[1]["chat_template_kwargs"] == {"enable_thinking": False}
        client._acquire_rate_limit.assert_awaited_once()


@pytest.mark.parametrize(
    ("provider", "base_url", "mode", "finish", "expected"),
    [
        ("openai", "http://localhost/v1", instructor.Mode.JSON_SCHEMA, "abort", True),
        ("openai", "http://localhost/v1", instructor.Mode.JSON, "abort", True),
        ("openai", "http://localhost/v1", instructor.Mode.MD_JSON, "abort", False),
        ("openai", "http://localhost/v1", instructor.Mode.JSON_SCHEMA, "content_filter", False),
        ("openai", "http://localhost/v1", instructor.Mode.JSON_SCHEMA, "length", False),
        ("openai", "http://localhost/v1", instructor.Mode.JSON_SCHEMA, "stop", False),
        ("openai", None, instructor.Mode.JSON_SCHEMA, "abort", False),
        ("google", "http://localhost/v1", instructor.Mode.JSON_SCHEMA, "abort", False),
    ],
)
def test_abort_recovery_is_limited_to_custom_endpoint_grammars(
    provider, base_url, mode, finish, expected
):
    client = llm.LLMClient(
        settings=GlobalSettings(llm={"provider": provider, "base_url": base_url})
    )
    error = ValueError("invalid completion")
    error.last_completion = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish)])
    assert client._can_recover_decoder_abort(error, SimpleNamespace(mode=mode)) is expected
