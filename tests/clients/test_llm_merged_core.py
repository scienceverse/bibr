"""Merged core-metadata extraction (LLM_MERGED_CORE_METADATA, opt-in).

One LLM call for title/abstract/keywords + authors + classification instead
of three — saves resending the same front-matter text 3x. Off by default;
the three-call path is the production default until the merged prompt is
quality-gated.
"""

from unittest import mock

import pytest

from bibr.clients.llm import LLMClient
from bibr.config import Settings
from bibr.exceptions import UpstreamServiceError
from bibr.schemas import (
    AuthorLLM,
    AuthorsLLM,
    CoreMetadataLLM,
    PaperClassificationLLM,
    TitleKeywordsLLM,
)

_TEXT = "Some Paper Title\nJane Doe, John Smith\nAbstract: things happened."


def _result_for(response_model):
    if response_model is TitleKeywordsLLM:
        return TitleKeywordsLLM(title="Some Paper Title", abstract="things", keywords=["k"])
    if response_model is AuthorsLLM:
        return AuthorsLLM(authors=[AuthorLLM(given="Jane", family="Doe")])
    if response_model is PaperClassificationLLM:
        return PaperClassificationLLM(paper_type="empirical")
    if response_model is CoreMetadataLLM:
        return CoreMetadataLLM(
            title="Some Paper Title",
            abstract="things",
            keywords=["k"],
            authors=[AuthorLLM(given="Jane", family="Doe")],
            paper_type="empirical",
        )
    raise AssertionError(f"unexpected response_model: {response_model}")


def _client_with_recorder(monkeypatch):
    client = LLMClient()
    calls = []

    async def fake_invoke(
        response_model,
        messages,
        system_prompt,
        reasoning_effort=None,
        client_override=None,
        max_tokens=None,
    ):
        calls.append(response_model)
        return _result_for(response_model)

    monkeypatch.setattr(client, "_invoke_structured", fake_invoke)
    return client, calls


async def test_default_uses_three_calls(monkeypatch):
    monkeypatch.setattr(Settings.llm, "merged_core_metadata", False, raising=False)
    client, calls = _client_with_recorder(monkeypatch)

    result = await client.extract_core_metadata(_TEXT)

    assert sorted(c.__name__ for c in calls) == [
        "AuthorsLLM",
        "PaperClassificationLLM",
        "TitleKeywordsLLM",
    ]
    assert result.title == "Some Paper Title"
    assert result.paper_type == "empirical"


async def test_merged_uses_single_call(monkeypatch):
    monkeypatch.setattr(Settings.llm, "merged_core_metadata", True, raising=False)
    client, calls = _client_with_recorder(monkeypatch)

    result = await client.extract_core_metadata(_TEXT)

    assert [c.__name__ for c in calls] == ["CoreMetadataLLM"]
    assert isinstance(result, CoreMetadataLLM)
    assert result.title == "Some Paper Title"
    assert result.paper_type == "empirical"


async def test_merged_ignores_selective_fanout_flag(monkeypatch):
    monkeypatch.setattr(Settings.llm, "merged_core_metadata", True, raising=False)
    client, calls = _client_with_recorder(monkeypatch)

    result = await client.extract_core_metadata(_TEXT, include_classification=False)

    assert [c.__name__ for c in calls] == ["CoreMetadataLLM"]
    assert result.paper_type == "empirical"


async def test_merged_failure_raises_upstream_error(monkeypatch):
    monkeypatch.setattr(Settings.llm, "merged_core_metadata", True, raising=False)
    client = LLMClient()
    monkeypatch.setattr(
        client,
        "_invoke_structured",
        mock.AsyncMock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(UpstreamServiceError):
        await client.extract_core_metadata(_TEXT)


@pytest.mark.parametrize("abstract_fields", [{}, {"abstract": None}, {"abstract": " "}])
async def test_core_preserves_explicit_null_versus_omitted_abstract(monkeypatch, abstract_fields):
    monkeypatch.setattr(Settings.llm, "merged_core_metadata", False)
    client = LLMClient()
    title = TitleKeywordsLLM(title="Paper", **abstract_fields)
    monkeypatch.setattr(client, "extract_title_keywords", mock.AsyncMock(return_value=title))
    monkeypatch.setattr(
        client,
        "extract_authors",
        mock.AsyncMock(return_value=AuthorsLLM(authors=[AuthorLLM(given="A", family="B")])),
    )
    combined = await client.extract_core_metadata(_TEXT, include_classification=False)
    assert ("abstract" in combined.model_fields_set) is ("abstract" in abstract_fields)
    assert combined._abstract_explicitly_absent is (abstract_fields == {"abstract": None})
