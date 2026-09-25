"""A truncated or invalid title/keywords response no longer fails the paper.

The anchor call's failure used to raise out of the core fan-out, cancel the
reference task and fail the file. A response that fails the same way on every
retry now leaves its fields empty with a blocking ``VAL_METADATA_FIELD_FAILED``
issue, while authors, classification and references are kept. A service that
did not answer still fails the paper, so a retry can complete it.
"""

import asyncio
import json
from unittest import mock
from unittest.mock import AsyncMock

import httpx
import instructor
import pandas as pd
import pytest
from openai import AsyncOpenAI

from bibr.clients import llm
from bibr.clients.llm import LLMClient
from bibr.config import GlobalSettings
from bibr.exceptions import (
    LlmCallError,
    LlmInvalidOutputError,
    LlmRejectedError,
    LlmServiceError,
    LlmTimeoutError,
    LlmTruncatedError,
    ProcessingError,
    UpstreamServiceError,
)
from bibr.extract.extractor import MetadataExtractor
from bibr.paper import PaperReference
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.schemas import (
    AuthorLLM,
    AuthorsLLM,
    CoreMetadataLLM,
    PaperClassificationLLM,
    TitleKeywordsLLM,
)

_TITLE_FIELDS = set(TitleKeywordsLLM.model_fields)


def _truncated() -> LlmTruncatedError:
    return LlmTruncatedError("Failed to extract title/keywords", cause="token limit")


def _invalid() -> LlmInvalidOutputError:
    return LlmInvalidOutputError("Failed to extract title/keywords", cause="1 validation error")


def _client(monkeypatch, *, title=None, authors=None, classification=None, merged=False):
    client = LLMClient(settings=GlobalSettings(llm={"merged_core_metadata": merged}))
    for method, value in [
        ("extract_title_keywords", title or TitleKeywordsLLM(title="A printed title")),
        (
            "extract_authors",
            authors or AuthorsLLM(authors=[AuthorLLM(given="Mara", family="Quill")]),
        ),
        (
            "extract_paper_classification",
            classification or PaperClassificationLLM(paper_type="empirical"),
        ),
    ]:
        mocked = (
            AsyncMock(side_effect=value)
            if isinstance(value, BaseException)
            else AsyncMock(return_value=value)
        )
        monkeypatch.setattr(client, method, mocked)
    return client


@pytest.mark.parametrize(
    ("failure", "code"), [(_truncated(), "llm_truncated"), (_invalid(), "llm_invalid_output")]
)
async def test_deterministic_title_failure_keeps_the_other_calls(monkeypatch, failure, code):
    client = _client(monkeypatch, title=failure)

    combined = await client.extract_core_metadata("printed content")

    assert not combined.title and combined.abstract is None and combined.keywords == []
    assert not combined._abstract_explicitly_absent
    assert [(a.given, a.family) for a in combined.authors] == [("Mara", "Quill")]
    assert combined.paper_type == "empirical"
    assert combined._field_failures == dict.fromkeys(_TITLE_FIELDS, code)
    assert client.extract_title_keywords.await_count == 1
    assert client.extract_authors.await_count == 1


@pytest.mark.parametrize(
    "failure",
    [
        LlmTimeoutError("Failed to extract title/keywords", cause="timed out"),
        LlmServiceError("Failed to extract title/keywords", cause="HTTP 503"),
        LlmRejectedError("Failed to extract title/keywords", cause="HTTP 401"),
        LlmCallError("Failed to extract title/keywords", cause="AttributeError"),
    ],
)
async def test_title_failure_a_retry_could_fix_still_fails_the_paper(monkeypatch, failure):
    client = _client(monkeypatch, title=failure)

    with pytest.raises(UpstreamServiceError) as raised:
        await client.extract_core_metadata("printed content")

    assert raised.value is failure


async def test_untyped_title_failure_is_classified_before_the_decision(monkeypatch):
    client = _client(monkeypatch, title=RuntimeError("unexpected"))

    with pytest.raises(LlmCallError) as raised:
        await client.extract_core_metadata("printed content")

    assert type(raised.value) is LlmCallError
    assert isinstance(raised.value.original_error, RuntimeError)


@pytest.mark.parametrize(
    "other",
    [
        LlmTimeoutError("Failed to extract authors", cause="timed out"),
        LlmServiceError("Failed to extract authors", cause="HTTP 503"),
        RuntimeError("unexpected"),
    ],
)
async def test_other_call_failing_for_a_fixable_reason_is_not_hidden(monkeypatch, other):
    client = _client(monkeypatch, title=_invalid(), authors=other)

    with pytest.raises(UpstreamServiceError) as raised:
        await client.extract_core_metadata("printed content")

    if isinstance(other, LlmCallError):
        assert raised.value is other
    else:
        assert raised.value.original_error is other


async def test_cancelled_call_is_never_contained(monkeypatch):
    client = _client(monkeypatch, title=_invalid(), authors=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await client.extract_core_metadata("printed content")


async def test_typed_processing_failure_still_wins(monkeypatch):
    error = ProcessingError("LLM returned invalid structured output", error_code="x")
    client = _client(monkeypatch, title=_invalid(), authors=error)

    with pytest.raises(ProcessingError) as raised:
        await client.extract_core_metadata("printed content")

    assert raised.value is error


async def test_optional_call_failures_are_recorded_without_changing_values(monkeypatch):
    client = _client(
        monkeypatch,
        authors=LlmTimeoutError("Failed to extract authors", cause="timed out"),
        classification=LlmTruncatedError("Failed to classify paper", cause="token limit"),
    )

    combined = await client.extract_core_metadata("printed content")

    # The degrade policy is unchanged: authors empty, classification blank.
    assert combined.title == "A printed title"
    assert combined.authors == []
    assert combined.paper_type is None
    assert combined._field_failures == {
        "authors": "llm_timeout",
        "paper_type": "llm_truncated",
        "oecd_domain": "llm_truncated",
        "oecd_subdomain": "llm_truncated",
    }


async def test_successful_fan_out_records_no_failures(monkeypatch):
    combined = await _client(monkeypatch).extract_core_metadata("printed content")
    assert combined._field_failures == {}


async def test_merged_call_failure_marks_every_field_failed(monkeypatch):
    client = LLMClient(settings=GlobalSettings(llm={"merged_core_metadata": True}))
    monkeypatch.setattr(client, "extract_core_metadata_merged", AsyncMock(side_effect=_truncated()))

    combined = await client.extract_core_metadata("printed content")

    assert not combined.title and combined.authors == []
    assert combined._field_failures == dict.fromkeys(CoreMetadataLLM.model_fields, "llm_truncated")


async def test_merged_call_timeout_still_fails(monkeypatch):
    client = LLMClient(settings=GlobalSettings(llm={"merged_core_metadata": True}))
    error = LlmTimeoutError("Failed to extract core metadata", cause="timed out")
    monkeypatch.setattr(client, "extract_core_metadata_merged", AsyncMock(side_effect=error))

    with pytest.raises(LlmTimeoutError):
        await client.extract_core_metadata("printed content")


# ---------------------------------------------------------------------------
# Through the extractor: references and authors survive a failed title call
# ---------------------------------------------------------------------------


def _extractor(llm_client) -> MetadataExtractor:
    df = pd.DataFrame(
        {
            "section_name": ["Title", "Abstract"] + ["Introduction"] * 2 + ["References"] * 2,
            "text": [
                "A printed title",
                "Some abstract text.",
                "Intro one.",
                "Intro two.",
                "Smith J. (2020). Paper A. Nature, 10, 1-5.",
                "Jones A. (2019). Paper B. Science, 20, 10-15.",
            ],
            "page_number": [1, 1, 1, 1, 2, 2],
        }
    )
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = df
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = [
        PaperSection(0, "Title", 1, None, CanonicalSection.TITLE, 1.0),
        PaperSection(1, "Abstract", 2, None, CanonicalSection.ABSTRACT, 1.0),
        PaperSection(2, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
        PaperSection(3, "References", 2, None, CanonicalSection.REFERENCES, 1.0),
    ]
    contents.sentences = []
    contents.processing_warnings = []
    return MetadataExtractor(contents, llm_client=llm_client)


async def test_failed_title_call_keeps_references_and_blocks_promotion(monkeypatch):
    client = _client(monkeypatch, title=_truncated())
    ext = _extractor(client)
    refs = [
        PaperReference(
            bib_id=1,
            title="Paper A",
            authors="Smith, J.",
            year=2020,
            first_page=None,
            volume=None,
            container=None,
        )
    ]
    ext._extract_references = AsyncMock(return_value=refs)

    metadata = await ext.extract_all_metadata()

    assert metadata.references == refs
    assert metadata.title == ""
    assert [(a.given, a.family) for a in metadata.authors] == [("Mara", "Quill")]
    [issue] = [i for i in ext.validation_issues if i.code == "VAL_METADATA_FIELD_FAILED"]
    assert issue.blocking and issue.severity == "error"
    assert "reason:llm_truncated" in issue.evidence_ids
    assert {"field:title", "field:abstract", "field:keywords", "field:published"} <= set(
        issue.evidence_ids
    )


async def test_failed_title_call_skips_the_trained_classifier(monkeypatch):
    settings = GlobalSettings(
        llm={"merged_core_metadata": False}, ml={"paper_classifier_model_id": "some/model"}
    )
    client = _client(monkeypatch, title=_invalid())
    ext = _extractor(client)
    ext.core._settings = settings
    classify = AsyncMock()
    monkeypatch.setattr("bibr.structure.paper_classifier.classify_paper_async", classify)
    ext._extract_references = AsyncMock(return_value=[])

    metadata = await ext.extract_all_metadata()

    classify.assert_not_awaited()
    assert metadata.paper_type == ""
    [issue] = [i for i in ext.validation_issues if i.code == "VAL_METADATA_FIELD_FAILED"]
    assert {"field:paper_type", "field:oecd_l1", "field:oecd_l2"} <= set(issue.evidence_ids)


async def test_successful_title_call_raises_no_field_issue(monkeypatch):
    ext = _extractor(_client(monkeypatch))
    ext._extract_references = AsyncMock(return_value=[])

    metadata = await ext.extract_all_metadata()

    assert metadata.title == "A printed title"
    assert not [i for i in ext.validation_issues if i.code == "VAL_METADATA_FIELD_FAILED"]


async def test_failed_author_call_without_a_credit_statement_keeps_the_llm_source(monkeypatch):
    client = _client(
        monkeypatch, authors=LlmTimeoutError("Failed to extract authors", cause="timed out")
    )
    ext = _extractor(client)
    ext.core._recover_empty_authors = AsyncMock(return_value=[])
    ext._extract_references = AsyncMock(return_value=[])

    metadata = await ext.extract_all_metadata()

    # The CRediT harvest found nothing, so it is not the source of the field.
    assert metadata.authors == []
    assert metadata._field_sources["author"] == "llm"


# ---------------------------------------------------------------------------
# Local recovery of a finished title/keywords response (ported from PR #8)
# ---------------------------------------------------------------------------

_LATEX_TITLE_RESPONSE = (
    r'{"title":"A printed study","abstract":"Measured \(7 \\pm 2\) units.","keywords":["growth"]}'
)


async def _title_over_http(
    monkeypatch, content: str, finish_reason: str = "stop", *, merged: bool = False
):
    requests = []

    async def respond(request):
        requests.append(json.loads(request.content))
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
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:

        def create_client(mode_override=None, **_kwargs):
            sdk = AsyncOpenAI(
                api_key="test", base_url="http://localhost/v1", http_client=transport, max_retries=0
            )
            return instructor.from_openai(
                sdk, mode=mode_override or instructor.Mode.JSON_SCHEMA, model="test"
            )

        monkeypatch.setattr(llm, "_create_client", create_client)
        client = LLMClient(
            settings=GlobalSettings(
                llm={
                    "provider": "openai",
                    "base_url": "http://localhost/v1",
                    "merged_core_metadata": merged,
                }
            )
        )
        monkeypatch.setattr(client, "_acquire_rate_limit", AsyncMock())
        call = client.extract_core_metadata if merged else client.extract_title_keywords
        try:
            return await call("A printed study", file_hash="h"), requests
        except LlmCallError as exc:
            return exc, requests


async def test_invalid_backslash_escapes_are_recovered_from_the_finished_response(monkeypatch):
    result, requests = await _title_over_http(monkeypatch, _LATEX_TITLE_RESPONSE)

    assert isinstance(result, TitleKeywordsLLM)
    assert result.title == "A printed study"
    assert result.abstract == r"Measured \(7 \pm 2\) units."
    assert result.keywords == ["growth"]
    assert len(requests) == 1


async def test_truncated_response_is_never_completed_locally(monkeypatch):
    result, _ = await _title_over_http(monkeypatch, '{"title":"A printed st', "length")
    assert type(result) is LlmTruncatedError


async def test_nested_value_in_a_malformed_envelope_is_not_recovered(monkeypatch):
    # Instructor rejects the outer title; recovery must not reach for the
    # nested one either.
    result, _ = await _title_over_http(
        monkeypatch, '{"title": ["bad"], "payload": {"title": "Nested substitute"}}'
    )
    assert type(result) is LlmInvalidOutputError


def _recovery_crashes(monkeypatch):
    from bibr.clients import structured_json

    def crash(*_args, **_kwargs):
        raise TypeError("a validator bug")

    monkeypatch.setattr(structured_json, "recover_structured_object", crash)


async def test_a_crashing_recovery_keeps_the_typed_error(monkeypatch):
    _recovery_crashes(monkeypatch)

    result, _ = await _title_over_http(monkeypatch, _LATEX_TITLE_RESPONSE)

    assert type(result) is LlmInvalidOutputError


async def test_a_crashing_merged_recovery_is_still_contained(monkeypatch):
    # A raw error here used to reach the extractor's broad except, which
    # exports an empty record without the blocking field-failed issue.
    _recovery_crashes(monkeypatch)

    result, _ = await _title_over_http(monkeypatch, _LATEX_TITLE_RESPONSE, merged=True)

    assert isinstance(result, CoreMetadataLLM)
    assert result._field_failures == dict.fromkeys(
        CoreMetadataLLM.model_fields, "llm_invalid_output"
    )
