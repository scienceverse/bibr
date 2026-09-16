"""Synthetic malformed envelopes must not select a valid nested value."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import instructor
import pytest
from openai import AsyncOpenAI

from bibr.clients.llm import InstructorBackend, LLMClient, usage_file_context
from bibr.clients.structured import StructuredResponseError
from bibr.clients.structured_json import recover_structured_object
from bibr.config import GlobalSettings
from bibr.schemas import TitleKeywordsLLM

MALFORMED = (
    r'{"title":"A printed study","abstract":"Measured \(7 \\pm 2\) units.","keywords":["growth"]}'
)


def test_outer_object_recovers_literal_math_backslashes_without_changing_values():
    recovered = recover_structured_object(f"```json\n{MALFORMED}\n```", TitleKeywordsLLM)

    assert recovered.value.title == "A printed study"
    assert recovered.value.abstract == r"Measured \(7 \pm 2\) units."
    assert recovered.value.keywords == ["growth"]
    assert recovered.repaired_backslashes == 2


def test_valid_escapes_and_unicode_are_not_reinterpreted():
    text = 'A quote " here; slash \\; tab\t; newline\n; Greek α.'
    recovered = recover_structured_object(json.dumps({"title": text}), TitleKeywordsLLM)

    assert recovered.value.title == text
    assert recovered.repaired_backslashes == 0


@pytest.mark.parametrize("prose", ["before", "after", "both"])
def test_one_explicit_fence_with_only_explanatory_prose_preserves_whole_object(prose):
    raw = '```json\n{"title":"A printed study","keywords":["growth"]}\n```'
    if prose in {"before", "both"}:
        raw = "The following values are printed in the document.\n\n" + raw
    if prose in {"after", "both"}:
        raw += "\n\nNo additional values were supplied."

    recovered = recover_structured_object(raw, TitleKeywordsLLM)

    assert recovered.value.title == "A printed study"
    assert recovered.value.keywords == ["growth"]
    assert recovered.repaired_backslashes == 0


@pytest.mark.parametrize(
    "outside",
    [
        '{"title":"Competing value"}',
        '["competing array"]',
        '{"unfinished":',
        '```json\n{"title":"Competing fence"}\n```',
        "```text\nExplanation\n```",
        "~~~json\n{}\n~~~",
    ],
)
@pytest.mark.parametrize("side", ["before", "after"])
def test_prose_fence_rejects_other_structured_material_anywhere(outside, side):
    fence = '```json\n{"title":"A printed study","keywords":["growth"]}\n```'
    raw = outside + "\n" + fence if side == "before" else fence + "\n" + outside

    with pytest.raises(StructuredResponseError):
        recover_structured_object(raw, TitleKeywordsLLM)


@pytest.mark.parametrize(
    "raw",
    [
        'Explanation\n```\n{"title":"Unnamed fence"}\n```',
        'Explanation\n```json\n{"title":"Truncated"\n```',
        'Explanation\n```json\n{"payload":{"title":"Nested"}}\n```',
        'Explanation\n```json\n["keywords only"]\n```',
        'Explanation\n```json\n{"title":"First", "title":"Second"}\n```',
    ],
)
def test_prose_fence_does_not_relax_outer_object_or_schema_guards(raw):
    with pytest.raises(StructuredResponseError):
        recover_structured_object(raw, TitleKeywordsLLM)


@pytest.mark.parametrize(
    "content",
    [
        '["growth"]',
        '{"abstract":"unfinished, "keywords":["growth"]}',
        '{"abstract":"unfinished',
        '{"abstract":"bad \\uXYZ1", "keywords":["growth"]}',
        '{"title":"first"}{"title":"second"}',
        'Note: {"title":"A printed study"}',
        '```json\n{"title":"first"}\n```\n```json\n{"title":"second"}\n```',
        '{"title":"first", "title":"second"}',
        '{"payload":{"title":"nested"}}',
        '{"TitleKeywordsLLM":["growth"]}',
        '{"TitleKeywordsLLM":{"payload":{"title":"nested"}}}',
        '{"title":NaN}',
        '{"title":{"not":"a string"}}',
    ],
)
def test_ambiguous_invalid_or_nested_content_is_never_silently_accepted(content):
    with pytest.raises(StructuredResponseError):
        recover_structured_object(content, TitleKeywordsLLM)


@pytest.mark.parametrize("fault", ["size", "depth", "repair-count"])
def test_recovery_is_bounded(fault):
    if fault == "size":
        content = json.dumps({"title": "x" * 262_144})
    elif fault == "depth":
        content = '{"keywords":' + "[" * 65 + "0" + "]" * 65 + "}"
    else:
        content = '{"title":"' + r"\(" * 129 + '"}'
    with pytest.raises(StructuredResponseError):
        recover_structured_object(content, TitleKeywordsLLM)


@pytest.mark.parametrize("track_usage", [False, True])
async def test_real_instructor_parse_failure_recovers_locally_and_retains_original_trace(
    track_usage,
):
    requests = []
    raw = f"```json\n{MALFORMED}\n```"

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": raw},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )

    settings = GlobalSettings(
        llm={
            "provider": "openai",
            "base_url": "http://localhost/v1",
            "capture_trace": True,
            "track_usage": track_usage,
        }
    )
    owner = LLMClient(settings=settings)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        sdk = AsyncOpenAI(api_key="test", base_url="http://localhost/v1", http_client=transport)
        client = instructor.from_openai(sdk, mode=instructor.Mode.MD_JSON, model="test")
        with usage_file_context("synthetic"):
            result, completion = await owner._backend.create(
                response_model=TitleKeywordsLLM,
                system="Extract metadata.",
                messages=[{"role": "user", "content": "A printed study"}],
                client_override=client,
                want_completion=track_usage,
            )
    assert len(requests) == 1
    assert result.title == "A printed study"
    assert result.abstract == r"Measured \(7 \pm 2\) units."
    assert (completion is not None) == track_usage
    failed, recovered = owner.traces_pop_file("synthetic")
    assert failed["parsed_ok"] is False
    assert "ValidationError" in failed["error"] or "InstructorRetryException" in failed["error"]
    assert failed["raw_completion"] == recovered["raw_completion"] == raw
    assert recovered["parsed_ok"] is True
    assert recovered["params"]["repaired_backslashes"] == 2


@pytest.mark.parametrize("failure", ["auth", "transport", "cancelled"])
async def test_backend_never_recovers_operational_failure_even_with_a_completion(failure):
    import asyncio

    if failure == "auth":
        error = RuntimeError("unauthorized")
        error.status_code = 401
    elif failure == "transport":
        error = TimeoutError("transport timed out")
    else:
        error = asyncio.CancelledError()
    error.last_completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=MALFORMED), finish_reason="stop")]
    )
    client = SimpleNamespace(create_with_completion=AsyncMock(side_effect=error))
    owner = LLMClient(settings=GlobalSettings())

    with pytest.raises(type(error)) as raised:
        await InstructorBackend(owner).create(
            response_model=TitleKeywordsLLM,
            system="Extract",
            messages=[],
            want_completion=True,
            client_override=client,
        )
    assert raised.value is error
    assert client.create_with_completion.await_count == 1


@pytest.mark.parametrize("track_usage", [False, True])
async def test_successful_instructor_nested_object_salvage_is_rejected(track_usage):
    # Instructor extracts the valid nested title after the outer JSON fails.
    # An all-default schema also permits a nested object of unrelated keys.
    raw = r'{"abstract":"bad \uXYZ1", "nested":{"title":"Nested substitute"}}'
    await _check_completed_response(raw, track_usage=track_usage, invalid=True)


@pytest.mark.parametrize("track_usage", [False, True])
async def test_instructor_explanatory_prose_and_one_fence_remains_one_request_and_trace(
    track_usage,
):
    raw = (
        "The document supplies the following values.\n\n"
        '```json\n{"title":"A printed study","keywords":["growth"]}\n```\n'
        "Only printed values are included."
    )
    await _check_completed_response(raw, track_usage=track_usage, invalid=False)


@pytest.mark.parametrize("track_usage", [False, True])
@pytest.mark.parametrize(
    "mode", [instructor.Mode.JSON, instructor.Mode.JSON_SCHEMA, instructor.Mode.MD_JSON]
)
@pytest.mark.parametrize("envelope", [False, True])
async def test_normal_json_response_or_explicit_schema_envelope_is_one_request_and_trace(
    track_usage, mode, envelope
):
    data = {"title": "A printed study", "keywords": ["growth"]}
    if envelope:
        data = {"TitleKeywordsLLM": data}
    await _check_completed_response(
        json.dumps(data), track_usage=track_usage, invalid=False, mode=mode
    )


async def _check_completed_response(raw, *, track_usage, invalid, mode=instructor.Mode.MD_JSON):
    requests, dispatches = [], []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": raw},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )

    owner = LLMClient(
        settings=GlobalSettings(
            llm={
                "provider": "openai",
                "base_url": "http://localhost/v1",
                "capture_trace": True,
                "track_usage": track_usage,
            }
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        sdk = AsyncOpenAI(api_key="test", base_url="http://localhost/v1", http_client=transport)
        client = instructor.from_openai(sdk, mode=mode, model="test")
        with usage_file_context("synthetic"):
            call = owner._backend.create(
                response_model=TitleKeywordsLLM,
                system="Extract metadata.",
                messages=[],
                client_override=client,
                want_completion=track_usage,
                on_dispatch=lambda: dispatches.append(True),
            )
            if invalid:
                with pytest.raises(StructuredResponseError) as raised:
                    await call
                # This particular fixture makes Instructor report success:
                # our success guard, not Instructor's exception path, rejected it.
                assert isinstance(raised.value.__cause__, StructuredResponseError)
            else:
                result, completion = await call
                assert result.title == "A printed study"
                assert result.keywords == ["growth"]
                assert (completion is not None) == track_usage
    assert len(requests) == len(dispatches) == 1
    traces = owner.traces_pop_file("synthetic")
    assert len(traces) == 1
    assert traces[0]["parsed_ok"] is not invalid
    assert traces[0]["raw_completion"] == raw


async def test_tool_call_mode_keeps_its_provider_representation():
    parsed = TitleKeywordsLLM(title="A printed study")
    completion = SimpleNamespace(content="This is not a JSON text response")
    client = SimpleNamespace(
        mode=instructor.Mode.TOOLS,
        create_with_completion=AsyncMock(return_value=(parsed, completion)),
        create=AsyncMock(return_value=parsed),
    )
    owner = LLMClient(settings=GlobalSettings())
    for want_completion in (False, True):
        result, raw = await owner._backend.create(
            response_model=TitleKeywordsLLM,
            system="Extract",
            messages=[],
            client_override=client,
            want_completion=want_completion,
        )
        assert result is parsed
        assert raw is (completion if want_completion else None)
    client.create.assert_awaited_once()
    client.create_with_completion.assert_awaited_once()
