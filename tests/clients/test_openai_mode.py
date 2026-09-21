"""OpenAI provider instructor-mode selection for custom (local) base_urls."""

import json
from unittest.mock import AsyncMock

import httpx
import instructor
import pytest

from bibr.clients import llm
from bibr.clients.providers.openai import _resolve_instructor_mode
from bibr.config import GlobalSettings
from bibr.schemas import TitleKeywordsLLM


class TestResolveInstructorMode:
    def test_default_is_json_schema(self):
        assert _resolve_instructor_mode("") == instructor.Mode.JSON_SCHEMA

    def test_json_selects_json_object_mode(self):
        assert _resolve_instructor_mode("json") == instructor.Mode.JSON

    def test_md_json_selects_prompt_only_json_mode(self):
        assert _resolve_instructor_mode("md_json") == instructor.Mode.MD_JSON

    def test_tools_selects_tools(self):
        assert _resolve_instructor_mode("tools") == instructor.Mode.TOOLS

    def test_case_insensitive(self):
        assert _resolve_instructor_mode("JSON") == instructor.Mode.JSON

    def test_unknown_falls_back_to_json_schema(self):
        assert _resolve_instructor_mode("bogus") == instructor.Mode.JSON_SCHEMA


# --- wire requests (mocked transport) ---------------------------------------


def _chat_completion(content: str) -> dict:
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


async def _captured_request(monkeypatch, llm_settings: dict) -> tuple[httpx.Request, dict]:
    """Run one structured call through the real adapter; return the request sent."""
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, json=_chat_completion('{"title":"A printed title","abstract":null}')
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    real_from_provider = instructor.from_provider

    def from_provider(*args, **kwargs):
        return real_from_provider(*args, http_client=transport, **kwargs)

    monkeypatch.setattr(instructor, "from_provider", from_provider)
    client = llm.LLMClient(
        settings=GlobalSettings(llm={"provider": "openai", "api_key": "test", **llm_settings})
    )
    monkeypatch.setattr(client, "_acquire_rate_limit", AsyncMock())
    try:
        result = await client._invoke_structured(
            TitleKeywordsLLM,
            [{"role": "user", "content": "A printed title"}],
            "Extract the metadata.",
            max_tokens=128,
        )
    finally:
        await transport.aclose()
    assert result.title == "A printed title"
    assert len(requests) == 1
    return requests[0], json.loads(requests[0].content)


async def test_json_mode_sends_json_object_and_a_prompt_that_says_json(monkeypatch):
    """A json_object-only endpoint needs the format flag and the word "json" in the prompt."""
    request, body = await _captured_request(
        monkeypatch,
        {
            "base_url": "https://api.example.org",
            "model": "example-model",
            "instructor_mode": "json",
            "extra_body": {"thinking": {"type": "disabled"}},
            "reasoning_effort": "",
        },
    )

    assert str(request.url) == "https://api.example.org/chat/completions"
    assert body["model"] == "example-model"
    assert body["response_format"] == {"type": "json_object"}
    assert "tools" not in body
    system = body["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith("Extract the metadata.")
    assert "json" in system["content"]
    assert body["thinking"] == {"type": "disabled"}
    assert "chat_template_kwargs" not in body
    assert "reasoning_effort" not in body
    assert body["max_tokens"] == 128
    assert body["temperature"] == 0.0


@pytest.mark.parametrize(
    ("extra_body", "expected_extra"),
    [
        ({}, {}),
        ({"top_k": 20}, {"top_k": 20}),
    ],
)
async def test_default_mode_sends_json_schema_and_chat_template_options(
    monkeypatch, extra_body, expected_extra
):
    _, body = await _captured_request(
        monkeypatch,
        {
            "base_url": "http://localhost:8000/v1",
            "model": "served-model",
            "chat_template_kwargs": {"enable_thinking": False},
            "extra_body": extra_body,
            "reasoning_effort": "",
        },
    )

    assert body["response_format"]["type"] == "json_schema"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in body
    for key, value in expected_extra.items():
        assert body[key] == value
