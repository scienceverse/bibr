"""An accepted tagger answer replaces the title/keywords and authors calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from bibr.clients.llm import LLMClient
from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.front_tagger.assemble import TaggedFrontMatter
from bibr.schemas import (
    AuthorLLM,
    AuthorsLLM,
    CoreMetadataLLM,
    PaperClassificationLLM,
    TitleKeywordsLLM,
)


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


# ---------------------------------------------------------------------------
# ML_FRONT_TAGGER_MODE=only: the tagger's whole answer, no LLM to ask
# ---------------------------------------------------------------------------


def _only(result, *, paged=True, mode="only"):
    """An extractor stand-in whose methods are the real ones, with no client at all."""
    ml = SimpleNamespace(
        front_tagger_mode=mode, front_tagger_min_confidence=0.9, paper_classifier_model_id="x/y"
    )
    contents = SimpleNamespace(
        front_tagger_result=result,
        region_summaries=[object()] if paged else [],
        processing_warnings=[],
    )
    ext = SimpleNamespace(_settings=SimpleNamespace(ml=ml), contents=contents, file_hash="h")
    ext._record_metadata_warning = lambda code, message: (
        CoreMetadataExtractor._record_metadata_warning(ext, code, message)
    )
    ext._tagged_core_metadata = lambda: CoreMetadataExtractor._tagged_core_metadata(ext)
    return ext


def _codes(ext):
    return [w.code for w in ext.contents.processing_warnings]


async def _core_call(ext):
    return await CoreMetadataExtractor._call_core_llm(ext, "FULL")


async def test_the_tagger_alone_answers_the_core_call_without_a_client():
    ext = _only(_tagged())  # no llm_client attribute: reaching for one raises
    result = await _core_call(ext)
    assert result.title == "Prevalence of Handedness"
    assert result.abstract == "We studied."
    assert result.keywords == ["k"]
    assert [a.family for a in result.authors] == ["de Oliveira"]
    assert result._front_tagger_parts == {"title_keywords", "authors"}
    assert _codes(ext) == []


async def test_a_flagged_answer_is_taken_whole_and_reported():
    from bibr.processing_warnings import WarningCode

    ext = _only(_tagged(conf=0.4, problems=["untagged_byline_word"]))
    result = await _core_call(ext)
    # Neither the low confidence nor the problem sends any part to a model.
    assert result.title == "Prevalence of Handedness"
    assert [a.family for a in result.authors] == ["de Oliveira"]
    assert _codes(ext) == [WarningCode.FRONT_TAGGER_FLAGGED]
    message = ext.contents.processing_warnings[0].message
    assert "untagged_byline_word" in message
    assert "low_confidence_title" in message and "low_confidence_authors" in message


async def test_a_confident_answer_raises_no_flag():
    ext = _only(_tagged(conf=0.9))
    await _core_call(ext)
    assert _codes(ext) == []


async def test_no_answer_on_a_paged_input_says_the_tagger_was_unavailable():
    from bibr.processing_warnings import NOT_FINAL_CODES, WarningCode

    ext = _only(None, paged=True)
    result = await _core_call(ext)
    assert result.title is None or result.title == ""
    assert result.authors == []
    assert result._front_tagger_parts == {"title_keywords", "authors"}
    assert _codes(ext) == [WarningCode.FRONT_TAGGER_UNAVAILABLE]
    # A model that failed to load may load next time, so the result is not cached.
    assert WarningCode.FRONT_TAGGER_UNAVAILABLE in NOT_FINAL_CODES


async def test_no_answer_on_an_input_without_pages_says_the_front_matter_was_not_read():
    from bibr.processing_warnings import NOT_FINAL_CODES, WarningCode

    ext = _only(None, paged=False)
    result = await _core_call(ext)
    assert result.authors == []
    assert _codes(ext) == [WarningCode.FRONT_MATTER_NOT_READ]
    # The input will not gain a layout, so the answer is final.
    assert WarningCode.FRONT_MATTER_NOT_READ not in NOT_FINAL_CODES
    assert "authors were not read" in ext.contents.processing_warnings[0].message


async def test_the_other_modes_still_ask_the_model():

    for mode in ("off", "primary"):
        ext = _only(_tagged(), mode=mode)
        ext.llm_client = SimpleNamespace(
            extract_core_metadata=mock.AsyncMock(
                return_value=CoreMetadataLLM(title="From model", authors=[])
            )
        )
        ext._accepted_front_tagger = lambda block: None
        assert (await _core_call(ext)).title == "From model"
        ext.llm_client.extract_core_metadata.assert_awaited_once()
        assert _codes(ext) == []


def test_the_record_the_calls_make_is_the_same_whoever_made_the_calls():
    from bibr.schemas import AuthorsLLM

    title_kw = TitleKeywordsLLM(title="T", keywords=["a"], journal="J")
    title_kw._repair_note = "repaired"
    authors = AuthorsLLM(authors=[AuthorLLM(given="Ann", family="Lee")])
    authors._salvaged_after = "Lee"

    record = CoreMetadataLLM.from_calls(
        title_kw,
        authors,
        PaperClassificationLLM(paper_type="empirical"),
        field_failures={"authors": "boom"},
        front_tagger_parts=frozenset({"authors"}),
    )

    assert (record.title, record.keywords, record.journal) == ("T", ["a"], "J")
    assert [a.family for a in record.authors] == ["Lee"]
    assert record.paper_type == "empirical"
    assert record._repair_note == "repaired"
    assert record._authors_salvaged_after == "Lee"
    assert record._field_failures == {"authors": "boom"}
    assert record._front_tagger_parts == {"authors"}
    # An abstract the answer never set stays unset, so nothing reads it as "absent".
    assert "abstract" not in record.model_fields_set
