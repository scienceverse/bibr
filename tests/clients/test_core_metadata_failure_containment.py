"""Independent metadata is retained only with an explicit blocking carrier."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from bibr.clients.llm import LLMClient
from bibr.clients.structured import PartialCoreMetadataError, StructuredResponseError
from bibr.config import GlobalSettings
from bibr.exceptions import ProcessingError, UpstreamServiceError
from bibr.schemas import AuthorLLM, AuthorsLLM, PaperClassificationLLM, TitleKeywordsLLM


def _client(monkeypatch, *, title=None, authors=None, classification=None):
    client = LLMClient(settings=GlobalSettings(llm={"merged_core_metadata": False}))
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


async def test_title_failure_preserves_valid_author_and_classification_without_new_calls(
    monkeypatch,
):
    failure = StructuredResponseError("non_json")
    client = _client(monkeypatch, title=failure)

    with pytest.raises(PartialCoreMetadataError) as raised:
        await client.extract_core_metadata("printed content")

    partial = raised.value.partial_metadata
    assert not partial.title and partial.abstract is None
    assert not partial._abstract_explicitly_absent
    assert [(row.given, row.family) for row in partial.authors] == [("Mara", "Quill")]
    assert partial.paper_type == "empirical"
    assert {"title", "abstract", "keywords"}.issubset(raised.value.failed_fields)
    assert raised.value.__cause__ is failure
    assert client.extract_title_keywords.await_count == client.extract_authors.await_count == 1


async def test_author_parse_failure_retains_valid_title_with_explicit_failed_field(monkeypatch):
    client = _client(monkeypatch, authors=StructuredResponseError("schema_invalid"))
    with pytest.raises(PartialCoreMetadataError) as raised:
        await client.extract_core_metadata("printed content")
    assert raised.value.partial_metadata.title == "A printed title"
    assert raised.value.partial_metadata.authors == []
    assert raised.value.failed_fields == ("authors",)


@pytest.mark.parametrize("kind", ["auth", "timeout", "cancelled", "unrelated-processing"])
async def test_other_failures_cannot_be_hidden_by_title_parse_failure(monkeypatch, kind):
    if kind == "auth":
        original = RuntimeError("unauthorized")
        original.status_code = 401
        error = UpstreamServiceError("LLM", "authentication failed", original)
    elif kind == "timeout":
        error = TimeoutError("timed out")
    elif kind == "cancelled":
        error = asyncio.CancelledError()
    else:
        error = ProcessingError("Unrelated error", error_code="unrelated")
    client = _client(monkeypatch, title=StructuredResponseError("non_json"), authors=error)
    with pytest.raises(type(error)) as raised:
        await client.extract_core_metadata("printed content")
    if kind != "cancelled":
        assert raised.value is error


async def test_merged_failure_is_explicitly_partial_and_has_no_invented_authors(monkeypatch):
    client = LLMClient(settings=GlobalSettings(llm={"merged_core_metadata": True}))
    failure = StructuredResponseError("truncated")
    monkeypatch.setattr(client, "extract_core_metadata_merged", AsyncMock(side_effect=failure))
    with pytest.raises(PartialCoreMetadataError) as raised:
        await client.extract_core_metadata("printed content")
    assert raised.value.partial_metadata.authors == []
    assert raised.value.partial_metadata.title is None
    assert "authors" in raised.value.failed_fields
    assert raised.value.__cause__ is failure
