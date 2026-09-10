"""Schema-valid empty authors are a domain signal.

The client preserves the initial empty list; CoreMetadataExtractor owns the single grounded recovery using selected byline evidence. Blank classification recovery remains client-owned."""

from unittest import mock

from bibr.clients.llm import LLMClient
from bibr.schemas import AuthorLLM, AuthorsLLM, PaperClassificationLLM, TitleKeywordsLLM


def _title_kw():
    return TitleKeywordsLLM(title="A Normal Research Title", abstract="Abstract.", keywords=["k"])


def _empty():
    return AuthorsLLM(authors=[])


def _full():
    return AuthorsLLM(authors=[AuthorLLM(given="Charles", family="Hulme")])


def _empirical_cls():
    return PaperClassificationLLM(paper_type="empirical", oecd_domain="Social Sciences")


async def test_core_client_does_not_own_schema_valid_empty_author_recovery():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    client.extract_authors = mock.AsyncMock(return_value=_empty())
    client.extract_paper_classification = mock.AsyncMock(return_value=_empirical_cls())

    result = await client.extract_core_metadata("byline: Charles Hulme1", file_hash="h")

    assert result.authors == []
    assert client.extract_authors.await_count == 1


async def test_no_reroll_when_authors_present():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    client.extract_authors = mock.AsyncMock(return_value=_full())
    client.extract_paper_classification = mock.AsyncMock(return_value=_empirical_cls())

    result = await client.extract_core_metadata("x", file_hash="h")

    assert client.extract_authors.await_count == 1
    assert len(result.authors) == 1


async def test_hard_author_failure_degrades_to_empty_without_extra_calls():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    from bibr.exceptions import UpstreamServiceError

    client.extract_authors = mock.AsyncMock(
        side_effect=UpstreamServiceError("LLM", "Failed to extract authors")
    )
    client.extract_paper_classification = mock.AsyncMock(return_value=_empirical_cls())

    result = await client.extract_core_metadata("x", file_hash="h")

    assert result.title == "A Normal Research Title"
    assert result.authors == []
    assert client.extract_authors.await_count == 1


async def test_null_classification_rerolled_on_empty_author_signal():
    # The re-roll goes through the free-JSON client, which only differs from
    # the primary one for the openai provider; elsewhere it would re-send a
    # byte-identical request at temperature 0 and is skipped.
    client = LLMClient(settings=_openai_settings())
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    client.extract_authors = mock.AsyncMock(return_value=_empty())
    client.extract_paper_classification = mock.AsyncMock(
        side_effect=[PaperClassificationLLM(), _empirical_cls()]
    )

    result = await client.extract_core_metadata("byline: Charles Hulme1", file_hash="h")

    assert result.paper_type == "empirical"
    assert client.extract_authors.await_count == 1
    assert client.extract_paper_classification.await_count == 2
    assert client.extract_paper_classification.await_args_list[1].kwargs.get("json_mode") is True


async def test_null_classification_not_rerolled_when_json_mode_cannot_differ():
    """Providers without a mode knob would re-send an identical request."""
    client = LLMClient()  # default provider: google
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    client.extract_authors = mock.AsyncMock(return_value=_empty())
    client.extract_paper_classification = mock.AsyncMock(
        side_effect=[PaperClassificationLLM(), _empirical_cls()]
    )

    await client.extract_core_metadata("byline: Charles Hulme1", file_hash="h")

    assert client.extract_paper_classification.await_count == 1


async def test_classification_not_rerolled_when_authors_present():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    client.extract_authors = mock.AsyncMock(return_value=_full())
    # classification blank, but authors present -> no empty-author signal -> no reroll
    client.extract_paper_classification = mock.AsyncMock(
        side_effect=[PaperClassificationLLM(), _empirical_cls()]
    )

    result = await client.extract_core_metadata("x", file_hash="h")

    assert client.extract_paper_classification.await_count == 1
    assert result.paper_type is None


async def test_omitted_classification_is_not_rerolled_on_empty_author_signal():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    client.extract_authors = mock.AsyncMock(return_value=_empty())
    client.extract_paper_classification = mock.AsyncMock(return_value=_empirical_cls())

    result = await client.extract_core_metadata(
        "byline: Charles Hulme1",
        file_hash="h",
        include_classification=False,
    )

    assert result.authors == []
    assert client.extract_authors.await_count == 1
    client.extract_paper_classification.assert_not_awaited()


async def test_extract_authors_json_mode_routes_to_json_client():
    client = LLMClient()
    lim = mock.Mock()
    lim.acquire = mock.AsyncMock()
    client._limiter = lim
    sentinel = object()
    client._get_json_mode_client = mock.Mock(return_value=sentinel)

    captured = {}

    async def fake_invoke(
        response_model,
        messages,
        system_prompt,
        reasoning_effort=None,
        client_override=None,
        max_tokens=None,
    ):
        captured["client_override"] = client_override
        return _full()

    client._invoke_structured = fake_invoke

    await client.extract_authors("text", json_mode=True)
    assert captured["client_override"] is sentinel

    captured.clear()
    await client.extract_authors("text")  # default json_mode=False
    assert captured["client_override"] is None


def _openai_settings():
    from bibr.config import GlobalSettings

    return GlobalSettings(llm={"provider": "openai", "api_key": "sk-test"})
