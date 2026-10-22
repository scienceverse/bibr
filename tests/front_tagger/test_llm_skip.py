"""An accepted tagger answer replaces the title/keywords and authors calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from bibr.clients.llm import LLMClient
from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.front_tagger.assemble import TaggedFrontMatter
from bibr.schemas import AuthorLLM, AuthorsLLM, PaperClassificationLLM, TitleKeywordsLLM


def _tagged(*, title="Prevalence of Handedness", conf=0.99, problems=()):
    return TaggedFrontMatter(
        title_keywords=TitleKeywordsLLM(title=title, abstract="We studied.", keywords=["k"]),
        byline=AuthorsLLM(authors=[AuthorLLM(given="Thais", family="de Oliveira")]),
        confidence={"title": conf, "authors": conf, "abstract": conf},
        problems=list(problems),
    )


async def test_tagged_answer_skips_the_two_calls():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock()
    client.extract_authors = mock.AsyncMock()
    client.extract_paper_classification = mock.AsyncMock(
        return_value=PaperClassificationLLM(paper_type="empirical")
    )
    result = await client.extract_core_metadata("FULL", file_hash="h", tagged=_tagged())
    client.extract_title_keywords.assert_not_awaited()
    client.extract_authors.assert_not_awaited()
    client.extract_paper_classification.assert_awaited_once()
    assert result.title == "Prevalence of Handedness"
    assert [a.family for a in result.authors] == ["de Oliveira"]
    assert result._front_tagger_parts == {"title_keywords", "authors"}


def _extractor(result, mode="primary"):
    ml = SimpleNamespace(front_tagger_mode=mode, front_tagger_min_confidence=0.9)
    return SimpleNamespace(
        _settings=SimpleNamespace(ml=ml),
        contents=SimpleNamespace(front_tagger_result=result),
        file_hash="h",
    )


def _accepted(extractor, block_text):
    return CoreMetadataExtractor._accepted_front_tagger(extractor, block_text)


def test_confident_answer_in_the_selected_record_is_used():
    tagged = _tagged()
    block = "Running head: X\nPrevalence of  Handedness\nThais de Oliveira"
    used = _accepted(_extractor(tagged), block)
    assert used.title_keywords is tagged.title_keywords
    assert used.byline is tagged.byline


def test_an_unlinked_byline_still_saves_the_title_call():
    block = "Prevalence of Handedness\nThais de Oliveira"
    used = _accepted(_extractor(_tagged(problems=["affiliations_unlinked"])), block)
    assert used.title_keywords is not None and used.byline is None


async def test_a_partial_answer_asks_only_for_the_missing_call():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock()
    client.extract_authors = mock.AsyncMock(
        return_value=AuthorsLLM(authors=[AuthorLLM(given="Ann", family="Lee")])
    )
    client.extract_paper_classification = mock.AsyncMock(
        return_value=PaperClassificationLLM(paper_type="empirical")
    )
    tagged = _tagged()
    tagged.byline = None
    result = await client.extract_core_metadata("FULL", file_hash="h", tagged=tagged)
    client.extract_title_keywords.assert_not_awaited()
    client.extract_authors.assert_awaited_once()
    assert result.title == "Prevalence of Handedness"
    assert [a.family for a in result.authors] == ["Lee"]
    assert result._front_tagger_parts == {"title_keywords"}


def test_off_unsure_incomplete_or_elsewhere_goes_to_the_llm():
    block = "Prevalence of Handedness\nThais de Oliveira"
    assert _accepted(_extractor(_tagged(), mode="off"), block) is None
    assert _accepted(_extractor(_tagged(conf=0.5)), block) is None
    assert _accepted(_extractor(_tagged(problems=["no_title", "no_authors"])), block) is None
    assert _accepted(_extractor(_tagged(title="Another Article Entirely")), block) is None
    assert _accepted(_extractor(None), block) is None
