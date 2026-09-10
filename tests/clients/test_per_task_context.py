"""Per-task context routing in LLMClient.extract_core_metadata.

The optional ``authors_text`` / ``classification_text`` kwargs route
task-specific slices into the authors and classification calls (and their
re-rolls) while title/keywords keeps the full ``text``. ``None`` falls back to
``text`` so the caller can leave the feature off transparently.
"""

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


async def test_slices_routed_to_authors_and_classification():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    client.extract_authors = mock.AsyncMock(return_value=_full())
    client.extract_paper_classification = mock.AsyncMock(return_value=_empirical_cls())

    await client.extract_core_metadata(
        "FULL", file_hash="h", authors_text="AUTHORS", classification_text="CLS"
    )

    assert client.extract_title_keywords.await_args.args[0] == "FULL"
    assert client.extract_authors.await_args.args[0] == "AUTHORS"
    assert client.extract_paper_classification.await_args.args[0] == "CLS"


async def test_none_kwargs_fall_back_to_text():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    client.extract_authors = mock.AsyncMock(return_value=_full())
    client.extract_paper_classification = mock.AsyncMock(return_value=_empirical_cls())

    await client.extract_core_metadata("FULL", file_hash="h")

    assert client.extract_authors.await_args.args[0] == "FULL"
    assert client.extract_paper_classification.await_args.args[0] == "FULL"


async def test_classification_reroll_uses_slice():
    from bibr.config import GlobalSettings

    # openai is the one provider whose free-JSON re-roll actually differs from
    # the primary call, so it is the one that still spends the extra request.
    client = LLMClient(settings=GlobalSettings(llm={"provider": "openai", "api_key": "sk-test"}))
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    # Schema-valid empty authors are returned unchanged (the extractor owns
    # author recovery), but the blank classification still re-rolls once on
    # the empty-author signal and must reuse its context slice.
    client.extract_authors = mock.AsyncMock(return_value=_empty())
    client.extract_paper_classification = mock.AsyncMock(
        side_effect=[PaperClassificationLLM(), _empirical_cls()]
    )

    await client.extract_core_metadata(
        "FULL", file_hash="h", authors_text="AUTHORS", classification_text="CLS"
    )

    assert client.extract_authors.await_count == 1
    assert client.extract_authors.await_args_list[0].args[0] == "AUTHORS"
    assert client.extract_paper_classification.await_count == 2
    assert client.extract_paper_classification.await_args_list[1].args[0] == "CLS"


async def test_classification_can_be_omitted_without_changing_other_calls():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock(return_value=_title_kw())
    client.extract_authors = mock.AsyncMock(return_value=_full())
    client.extract_paper_classification = mock.AsyncMock(return_value=_empirical_cls())

    result = await client.extract_core_metadata(
        "FULL",
        file_hash="h",
        authors_text="AUTHORS",
        classification_text="CLS",
        include_classification=False,
    )

    client.extract_title_keywords.assert_awaited_once()
    client.extract_authors.assert_awaited_once()
    client.extract_paper_classification.assert_not_awaited()
    assert result.paper_type is None
    assert result.oecd_domain is None
