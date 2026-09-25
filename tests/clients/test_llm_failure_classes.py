"""A failed LLM call raises the ``LlmCallError`` subclass that says how it failed.

Each class stays an ``UpstreamServiceError`` so the degrade sites behave as
before; the class and ``error_code`` separate a service that did not answer
from a response that was truncated or invalid, which serve reports as 422.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import instructor
import pytest
from instructor.core.exceptions import IncompleteOutputException, InstructorRetryException
from openai import AsyncOpenAI
from pydantic import ValidationError

from bibr.clients import llm
from bibr.config import GlobalSettings
from bibr.exceptions import (
    LlmCallError,
    LlmInvalidOutputError,
    LlmRejectedError,
    LlmServiceError,
    LlmTimeoutError,
    LlmTruncatedError,
    LlmUnreachableError,
    ProcessingError,
    UpstreamServiceError,
)
from bibr.schemas import AuthorsLLM
from bibr.utils.circuit_breaker import CircuitOpenError


class _StatusError(Exception):
    def __init__(self, status: int):
        super().__init__(f"Error code: {status}")
        self.status_code = status


def _retry_wrapping(cause: BaseException, completion=None) -> InstructorRetryException:
    try:
        raise InstructorRetryException(
            str(cause), last_completion=completion, n_attempts=1, total_usage=0
        ) from cause
    except InstructorRetryException as wrapped:
        return wrapped


def _validation_error(value="SECRET-MODEL-OUTPUT") -> ValidationError:
    try:
        AuthorsLLM.model_validate(
            {"authors": [{"given": "Ada", "family": "Lovelace", "corresponding": value}]}
        )
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


def _completion(finish_reason: str):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason)])


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (IncompleteOutputException(last_completion=_completion("length")), LlmTruncatedError),
        (
            _retry_wrapping(
                json.JSONDecodeError("Unterminated string", "{", 1), _completion("length")
            ),
            LlmTruncatedError,
        ),
        (TimeoutError("LLM call timed out after 240s"), LlmTimeoutError),
        (httpx.ReadTimeout("read timed out"), LlmTimeoutError),
        (_retry_wrapping(_StatusError(408)), LlmTimeoutError),
        (_retry_wrapping(_StatusError(503)), LlmServiceError),
        (_retry_wrapping(_StatusError(429)), LlmServiceError),
        (httpx.ConnectError("connection refused"), LlmUnreachableError),
        (httpx.RemoteProtocolError("server disconnected"), LlmUnreachableError),
        (httpx.ConnectTimeout("connect timed out"), LlmUnreachableError),
        (ConnectionResetError("reset by peer"), LlmUnreachableError),
        (CircuitOpenError("llm", 12.0), LlmUnreachableError),
        (httpx.DecodingError("garbled body"), LlmServiceError),
        (_retry_wrapping(_StatusError(401)), LlmRejectedError),
        (_retry_wrapping(_StatusError(400)), LlmRejectedError),
        (_validation_error(), LlmInvalidOutputError),
        (_retry_wrapping(_validation_error(), _completion("stop")), LlmInvalidOutputError),
        (json.JSONDecodeError("Expecting value", "x", 0), LlmInvalidOutputError),
        (AttributeError("'NoneType' object has no attribute 'text'"), LlmCallError),
    ],
)
def test_failure_is_classified_from_the_exception_chain(error, expected):
    wrapped = llm.llm_call_error("Failed to extract title/keywords", error)

    assert type(wrapped) is expected
    assert isinstance(wrapped, UpstreamServiceError)
    assert wrapped.service_name == "LLM"
    assert wrapped.original_error is error
    assert wrapped.error_code == expected.error_code
    assert llm.llm_failure_code(error) == expected.error_code
    assert str(wrapped).startswith("Error in LLM: Failed to extract title/keywords (")


def test_error_codes_are_stable():
    assert LlmCallError.error_code == "llm_failed"
    assert LlmServiceError.error_code == "llm_failed"
    assert LlmRejectedError.error_code == "llm_failed"
    assert LlmTimeoutError.error_code == "llm_timeout"
    assert LlmTruncatedError.error_code == "llm_truncated"
    assert LlmInvalidOutputError.error_code == "llm_invalid_output"
    assert LlmUnreachableError.error_code == "llm_failed"
    assert issubclass(LlmTimeoutError, LlmServiceError)
    assert issubclass(LlmUnreachableError, LlmServiceError)


def test_invalid_output_cause_names_locations_but_never_model_output():
    wrapped = llm.llm_call_error("Failed to extract title/keywords", _validation_error())

    assert "authors.0.corresponding [bool_parsing]" in str(wrapped)
    assert "SECRET-MODEL-OUTPUT" not in str(wrapped)


def test_rewrapping_keeps_class_and_cause():
    inner = llm.llm_call_error("Failed to extract authors", TimeoutError("timed out after 9s"))
    outer = llm.llm_call_error("Failed to extract core metadata", inner)

    assert type(outer) is LlmTimeoutError
    assert outer.cause == inner.cause
    assert "timed out after 9s" in str(outer)


def test_typed_processing_error_keeps_its_own_code():
    error = ProcessingError("invalid", error_code="llm_invalid_output")
    assert llm.llm_failure_code(error) == "llm_invalid_output"


async def test_task_wrappers_raise_the_typed_error():
    client = llm.LLMClient()
    client._acquire_rate_limit = AsyncMock()
    client._invoke_structured = AsyncMock(
        side_effect=IncompleteOutputException(last_completion=_completion("length"))
    )

    with pytest.raises(LlmTruncatedError) as raised:
        await client.extract_title_keywords("A printed title", file_hash="h")

    # Degrade sites that catch the base class still catch it.
    assert isinstance(raised.value, UpstreamServiceError)
    assert raised.value.error_code == "llm_truncated"


# ---------------------------------------------------------------------------
# The same classes from a real Instructor client over a mocked HTTP transport
# ---------------------------------------------------------------------------


def _chat(content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }


async def _title_call(monkeypatch, respond) -> LlmCallError:
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:

        def create_client(mode_override=None, **_kwargs):
            sdk = AsyncOpenAI(
                api_key="test",
                base_url="http://localhost/v1",
                http_client=transport,
                max_retries=0,
            )
            return instructor.from_openai(
                sdk, mode=mode_override or instructor.Mode.JSON_SCHEMA, model="test"
            )

        monkeypatch.setattr(llm, "_create_client", create_client)
        client = llm.LLMClient(
            settings=GlobalSettings(
                llm={"provider": "openai", "base_url": "http://localhost/v1", "max_retries": 1}
            )
        )
        monkeypatch.setattr(client, "_acquire_rate_limit", AsyncMock())
        monkeypatch.setattr(client, "_RETRY_BASE_DELAY", 0.0)
        monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())
        with pytest.raises(LlmCallError) as raised:
            await client.extract_title_keywords("A printed title", file_hash="h")
        return raised.value


async def test_real_client_invalid_json_is_invalid_output(monkeypatch):
    async def respond(request):
        return httpx.Response(200, json=_chat('{"title": ["not", "a", "string"]}'))

    error = await _title_call(monkeypatch, respond)
    assert type(error) is LlmInvalidOutputError


async def test_real_client_truncated_json_is_truncation(monkeypatch):
    async def respond(request):
        return httpx.Response(200, json=_chat('{"title": "A printed ti', "length"))

    error = await _title_call(monkeypatch, respond)
    assert type(error) is LlmTruncatedError


async def test_real_client_server_error_is_service_failure(monkeypatch):
    async def respond(request):
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    error = await _title_call(monkeypatch, respond)
    assert type(error) is LlmServiceError
    assert "503" in str(error)


async def test_real_client_auth_failure_is_rejected(monkeypatch):
    async def respond(request):
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    error = await _title_call(monkeypatch, respond)
    assert type(error) is LlmRejectedError


def test_cancellation_is_not_wrapped():
    # Task wrappers catch ``Exception``; cancellation must stay outside that.
    assert not issubclass(asyncio.CancelledError, Exception)


# ---------------------------------------------------------------------------
# core-api-6: a wrong-typed value is a ValidationError, not a crash
# ---------------------------------------------------------------------------

_REFERENCE = {
    "index": 1,
    "title": "R",
    "authors": "A",
    "year": 2020,
    "first_page": None,
    "volume": None,
    "container": None,
}


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        ("TitleKeywordsLLM", {"title": "T", "abstract": ["p1", "p2"]}),
        ("TitleKeywordsLLM", {"title": "T", "abstract": 7}),
        ("CoreMetadataLLM", {"title": "T", "authors": [], "abstract": {"text": "p"}}),
        ("PaperReferenceLLM", {**_REFERENCE, "bib_type": 7}),
        ("PaperReferenceLLM", {**_REFERENCE, "bib_type": ["book"]}),
    ],
)
def test_wrong_typed_llm_values_raise_validation_errors(model, payload):
    """Instructor re-asks only on a ValidationError; an AttributeError from a
    validator failed the call at once and read as an upstream outage."""
    from bibr import schemas

    with pytest.raises(ValidationError) as raised:
        getattr(schemas, model).model_validate(payload)
    assert {error["loc"][0] for error in raised.value.errors()} <= {"abstract", "bib_type"}


def test_well_typed_values_are_unchanged():
    from bibr.schemas import PaperReferenceLLM, TitleKeywordsLLM

    assert TitleKeywordsLLM.model_validate({"title": "T", "abstract": "  An abstract. "}).abstract
    assert TitleKeywordsLLM.model_validate({"title": "T", "abstract": "  "}).abstract is None
    reference = PaperReferenceLLM.model_validate({**_REFERENCE, "bib_type": "article"})
    assert reference.bib_type == "journal_article"
