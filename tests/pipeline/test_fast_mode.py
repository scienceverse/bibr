"""Fast mode makes no LLM call: one test per guard, under a tripwire.

Every site that would call a model in a full run is handed the stand-in client
(or tested for it) and must take its no-LLM path without reaching a model. A
site that catches the stand-in's refusal would hide a missed guard behind a
``*_FAILED`` warning, so the ``tripwire`` fixture records every way a model
could be reached and fails the test that reached one.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from bibr.clients import ref_llm, skipped
from bibr.clients.llm import LLMClient
from bibr.clients.skipped import SkippedLlmClient
from bibr.config import GlobalSettings
from bibr.models import PaperAuthor, PaperMetadata, PaperReference
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    RegionSummary,
)
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState
from bibr.processing_warnings import WarningCode
from bibr.schemas import CoreMetadataLLM


@pytest.fixture(autouse=True)
def tripwire(monkeypatch):
    """Every way a fast run could reach a model; the test ends on an empty list."""
    reached: list[str] = []

    def construct(self, *args, **kwargs):
        reached.append("LLMClient()")
        raise AssertionError("fast mode built an LLMClient")

    async def invoke(self, *args, **kwargs):
        reached.append("LLMClient._invoke_structured()")
        raise AssertionError("fast mode reached the model")

    async def parse_batch(*args, **kwargs):
        reached.append("ref_llm.parse_batch()")
        raise AssertionError("fast mode reached the reference parser server")

    refused = skipped._refused

    def refuse(name):
        reached.append(f"SkippedLlmClient.{name}()")
        return refused(name)

    monkeypatch.setattr(LLMClient, "__init__", construct)
    monkeypatch.setattr(LLMClient, "_invoke_structured", invoke)
    monkeypatch.setattr(ref_llm, "parse_batch", parse_batch)
    monkeypatch.setattr(skipped, "_refused", refuse)
    yield reached
    assert reached == []


def _codes(warnings) -> list[str]:
    return [w.code for w in warnings]


# ---------------------------------------------------------------------------
# Rows 5-7: core metadata. The tagger's answer stands in for the core call.
# ---------------------------------------------------------------------------


def _core_extractor(llm_client, llm_metadata: CoreMetadataLLM | None = None):
    from bibr.extract.extractor import MetadataExtractor

    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame(
        {
            "section_name": ["Title", "Abstract", "1. Introduction"],
            "text": ["A Normal Research Title", "Some abstract text.", "Some intro text."],
            "page_number": [1, 1, 1],
        }
    )
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = [
        PaperSection(0, "Title", 1, None, CanonicalSection.TITLE, 1.0),
        PaperSection(1, "Abstract", 2, None, CanonicalSection.ABSTRACT, 1.0),
        PaperSection(2, "1. Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
    ]
    contents.sentences = []
    contents.processing_warnings = []
    ext = MetadataExtractor(contents, llm_client=llm_client)
    answer = llm_metadata or CoreMetadataLLM(
        title="A Normal Research Title", abstract="Some abstract text.", authors=[]
    )
    ext.core._call_core_llm = AsyncMock(return_value=answer)
    return ext


def test_the_extractors_keep_a_falsy_client_without_building_a_real_one():
    from bibr.extract.extractor import MetadataExtractor
    from bibr.extract.ref_extractor import ReferenceExtractor

    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    client = SkippedLlmClient()

    assert ReferenceExtractor(contents, llm_client=client).llm_client is client
    assert MetadataExtractor(contents, llm_client=client).llm_client is client


async def test_empty_authors_skip_the_recovery_and_leave_no_trace():
    ext = _core_extractor(SkippedLlmClient())
    ext.core._recover_empty_authors = AsyncMock()

    await ext.extract_core_metadata()

    ext.core._recover_empty_authors.assert_not_awaited()
    assert ext.metadata.authors == []
    assert "VAL_AUTHOR_RECOVERY_DEGRADED" not in {i.code for i in ext.core.validation_issues}
    codes = _codes(ext.contents.processing_warnings)
    assert WarningCode.AUTHORS_EMPTY in codes
    assert WarningCode.AUTHORS_LLM_FAILED not in codes


def _tagger_alone_extractor(tagged, *, paged: bool = True):
    """A real extractor under fast settings: the tagger's answer, no client to ask."""
    from bibr.extract.extractor import MetadataExtractor
    from bibr.run_mode import fast_settings

    ext = _core_extractor(SkippedLlmClient())
    ext.contents.front_tagger_result = tagged
    ext.contents.region_summaries = (
        [RegionSummary(page=1, index=0, label="text", bbox=None)] if paged else []
    )
    settings = fast_settings(GlobalSettings())
    ext = MetadataExtractor(ext.contents, llm_client=SkippedLlmClient(), settings=settings)
    return ext


def _tagged(*, conf: float = 0.99, problems=(), title="A Tagged Title"):
    from bibr.front_tagger.assemble import TaggedFrontMatter
    from bibr.schemas import AuthorLLM, AuthorsLLM, TitleKeywordsLLM

    return TaggedFrontMatter(
        title_keywords=TitleKeywordsLLM(
            title=title, abstract="The tagger read this abstract.", keywords=["memory"]
        ),
        byline=AuthorsLLM(authors=[AuthorLLM(given="Ana", family="Ruiz")]),
        confidence={"title": conf, "authors": conf, "abstract": conf},
        problems=list(problems),
    )


async def test_fast_settings_take_the_core_metadata_from_the_tagger_alone():
    ext = _tagger_alone_extractor(_tagged())

    await ext.extract_core_metadata()

    assert ext.metadata.title == "A Tagged Title"
    assert ext.metadata.abstract == "The tagger read this abstract."
    assert ext.metadata.keywords == ["memory"]
    assert [(a.given, a.family) for a in ext.metadata.authors] == [("Ana", "Ruiz")]
    assert ext.metadata._field_decisions.sources()["author"] == "front_tagger"
    assert _codes(ext.contents.processing_warnings) == []


async def test_a_flagged_answer_is_kept_and_reported_not_sent_to_a_model():
    ext = _tagger_alone_extractor(_tagged(conf=0.3, problems=["affiliations_unlinked"]))

    await ext.extract_core_metadata()

    assert ext.metadata.title == "A Tagged Title"
    assert [a.family for a in ext.metadata.authors] == ["Ruiz"]
    warnings = {w.code: w.message for w in ext.contents.processing_warnings}
    assert "affiliations_unlinked" in warnings[WarningCode.FRONT_TAGGER_FLAGGED]
    assert "low_confidence_title" in warnings[WarningCode.FRONT_TAGGER_FLAGGED]


async def test_no_tagger_answer_on_a_paged_input_leaves_the_front_matter_empty():
    ext = _tagger_alone_extractor(None, paged=True)

    await ext.extract_core_metadata()

    codes = _codes(ext.contents.processing_warnings)
    assert WarningCode.FRONT_TAGGER_UNAVAILABLE in codes
    assert WarningCode.FRONT_MATTER_NOT_READ not in codes
    assert ext.metadata.authors == []
    assert not ext.metadata.abstract


async def test_the_full_run_settings_still_ask_the_client_for_the_core_call():
    from bibr.extract.extractor import MetadataExtractor
    from bibr.schemas import AuthorLLM

    ext = _core_extractor(SkippedLlmClient())
    answering = MagicMock(
        extract_core_metadata=AsyncMock(
            return_value=CoreMetadataLLM(
                title="From The Model", authors=[AuthorLLM(given="Bo", family="Lee")]
            )
        )
    )
    ext = MetadataExtractor(ext.contents, llm_client=answering, settings=GlobalSettings())

    await ext.extract_core_metadata()

    answering.extract_core_metadata.assert_awaited_once()
    assert ext.metadata.title == "From The Model"
    assert WarningCode.FRONT_TAGGER_FLAGGED not in _codes(ext.contents.processing_warnings)


async def test_the_credit_statement_still_recovers_authors_in_fast_mode():
    ext = _core_extractor(SkippedLlmClient())

    with mock.patch(
        "bibr.extract.core_metadata.credit_statement_authors", return_value=[("Ada", "Lovelace")]
    ):
        await ext.extract_core_metadata()

    assert [(a.given, a.family) for a in ext.metadata.authors] == [("Ada", "Lovelace")]
    assert "VAL_AUTHOR_CREDIT_RECOVERED" in {i.code for i in ext.core.validation_issues}


async def test_an_unavailable_paper_classifier_leaves_the_class_empty(monkeypatch):
    from bibr.structure import paper_classifier

    ext = _core_extractor(SkippedLlmClient())
    ext.core._settings = GlobalSettings(ml={"paper_classifier_model_id": "some/model"})
    ext.core._explicit_classifier_runtime = False
    monkeypatch.setattr(paper_classifier, "classify_paper_async", AsyncMock(return_value=None))

    await ext.extract_core_metadata()

    warnings = {w.code: w.message for w in ext.contents.processing_warnings}
    assert warnings[WarningCode.PAPER_CLASSIFIER_DEGRADED] == (
        "trained classifier unavailable; fast mode has no LLM fallback"
    )
    assert WarningCode.PAPER_CLASSIFICATION_FAILED not in warnings
    assert not ext.metadata.paper_type


async def test_a_low_confidence_paper_type_is_not_escalated(monkeypatch):
    from bibr.structure import paper_classifier

    settings = GlobalSettings(
        ml={
            "paper_classifier_model_id": "some/model",
            "paper_classifier_llm_escalation": True,
            "paper_classifier_min_confidence": 0.9,
        }
    )
    monkeypatch.setattr(
        paper_classifier,
        "classify_paper_async",
        AsyncMock(return_value=("", 0.0, "", 0.0, "empirical", 0.2)),
    )

    async def classify(client):
        ext = _core_extractor(client)
        ext.core._settings = settings
        ext.core._explicit_classifier_runtime = False
        return ext, await ext.core._classify_paper("T", "A", CoreMetadataLLM(authors=[]), "text")

    ext, (paper_type, *_rest) = await classify(SkippedLlmClient())
    assert paper_type == "empirical"
    assert ext.core._paper_type_source == "classifier"

    # The same paper with a client that answers is escalated: the guard, not the
    # settings, is what stops the call.
    answering = MagicMock(
        label_paper_type=AsyncMock(
            return_value=SimpleNamespace(paper_type="theoretical", confidence=0.8)
        )
    )
    ext, (paper_type, *_rest) = await classify(answering)
    assert paper_type == "theoretical"
    answering.label_paper_type.assert_awaited_once()


# ---------------------------------------------------------------------------
# Rows 8a and 8b: section classification
# ---------------------------------------------------------------------------


async def _trained_model_gives_unknown(monkeypatch, *, model_available: bool):
    import bibr.structure.section_classifier as sc

    calls = {"llm": 0}

    async def fake_trained(pairs):
        return [(CanonicalSection.UNKNOWN, 0.3, None) for _ in pairs]

    async def fake_llm(texts, **kwargs):
        calls["llm"] += 1
        return [(CanonicalSection.ETHICS, 0.85) for _ in texts]

    async def fake_get_model(*args):
        return object() if model_available else None

    monkeypatch.setattr(sc, "_classify_trained_batch", fake_trained)
    monkeypatch.setattr(sc, "_classify_llm_batch", fake_llm)
    monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)
    return sc, calls


async def test_section_escalation_is_not_sent_to_the_llm(monkeypatch):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.ml, "section_classifier_llm_escalation", True)
    sc, calls = await _trained_model_gives_unknown(monkeypatch, model_available=True)
    header = "limitations of existing theories"

    fast = await sc.classify_headers_batch_async([header], llm_client=SkippedLlmClient())
    assert calls["llm"] == 0
    assert fast == [(CanonicalSection.DISCUSSION, 0.6, None, "alias_prior")]

    # With a client that answers, the same header is escalated.
    answered = await sc.classify_headers_batch_async(
        ["patient consent and irb"], llm_client=object()
    )
    assert calls["llm"] == 1
    assert answered == [(CanonicalSection.ETHICS, 0.85, None, "llm")]


@pytest.mark.parametrize("model_id", [None, "some/model"])
async def test_headers_with_no_trained_model_keep_their_alias_prior(monkeypatch, model_id):
    from bibr.config import Settings
    from bibr.structure.section_classifier import SECTION_CLASSIFIER_DEGRADED_FAST_WARNING

    monkeypatch.setattr(Settings.ml, "section_classifier_model_id", model_id)
    sc, calls = await _trained_model_gives_unknown(monkeypatch, model_available=False)
    warnings: list = []

    results = await sc.classify_headers_batch_async(
        ["limitations of existing theories", "wholly unfamiliar heading"],
        llm_client=SkippedLlmClient(),
        degradation_warnings=warnings,
    )

    assert calls["llm"] == 0
    assert results == [
        (CanonicalSection.DISCUSSION, 0.6, None, "alias_prior"),
        (CanonicalSection.UNKNOWN, 0.0, None, None),
    ]
    # One warning, however many places noticed the missing tier.
    assert warnings == [SECTION_CLASSIFIER_DEGRADED_FAST_WARNING]
    assert warnings[0].code == WarningCode.SECTION_CLASSIFIER_DEGRADED


# ---------------------------------------------------------------------------
# Row 9: implicit sections
# ---------------------------------------------------------------------------


def _front_matter_contents() -> PaperContents:
    sections = [
        PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN),
        PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN),
        PaperSection(2, "Method", 1, 0, CanonicalSection.METHODS),
    ]
    sentences = [
        PaperSentence(1, "Abstract text.", 1, 1, page_number=1),
        PaperSentence(2, "More abstract.", 1, 2, page_number=1),
        PaperSentence(3, "Intro here.", 1, 3, page_number=1),
        PaperSentence(4, "More intro.", 1, 4, page_number=2),
        PaperSentence(5, "Method.", 2, 5, page_number=2),
    ]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={1: "Abstract text. More abstract. Intro here. More intro.", 2: "Method."},
        detected_title="Paper Title",
    )


def _sections_of(contents: PaperContents):
    return [(s.header, s.section_type) for s in contents.sections]


async def test_implicit_sections_take_the_positional_fallback_without_a_failure():
    from bibr.structure.implicit_sections import detect_implicit_sections

    # The failure path: the detector answered nothing, so the positional fallback decides.
    failed = _front_matter_contents()
    with mock.patch(
        "bibr.structure.implicit_sections._detect_via_llm",
        new_callable=AsyncMock,
        return_value=None,
    ):
        await detect_implicit_sections(failed, llm_client=MagicMock(close=AsyncMock()))

    fast = _front_matter_contents()
    with mock.patch(
        "bibr.structure.implicit_sections._detect_via_llm",
        new_callable=AsyncMock,
        side_effect=AssertionError("fast mode must not detect with the LLM"),
    ) as detect:
        await detect_implicit_sections(fast, llm_client=SkippedLlmClient())

    detect.assert_not_awaited()
    assert CanonicalSection.ABSTRACT in {s.section_type for s in fast.sections}
    assert _sections_of(fast) == _sections_of(failed)
    assert fast.processing_warnings == []


# ---------------------------------------------------------------------------
# Rows 10 to 12: references
# ---------------------------------------------------------------------------

_REF1 = "Smith, J., & Jones, K. (2020). Attention and memory. Psych Review, 12(3), 45-67."
_REF2 = "Doe, A. (2019). Seeing things clearly. Journal of Vision, 2, 11-20."
_REF3 = "Nguyen, T. H. (2021). Replication in the wild. Meta Science, 4(1), 1-19."
_REF4 = "World Health Organization. (2018). Global report on falls. WHO Press."


def _ref_contents(region_summaries=()) -> mock.Mock:
    contents = mock.Mock(spec=PaperContents)
    contents.native_ref_strings = None
    contents.ref_line_geometry = None
    contents.region_summaries = list(region_summaries)
    contents.processing_warnings = []
    contents.reference_yield_receipt = None
    contents.sections = []
    contents.sentences = []
    return contents


def _reference_extractor(contents, **settings):
    from bibr.extract.ref_extractor import ReferenceExtractor

    return ReferenceExtractor(
        contents, settings=GlobalSettings(**settings), llm_client=SkippedLlmClient()
    )


def _crf_segments(monkeypatch, segments):
    segmenter = mock.Mock()
    segmenter.segment = mock.Mock(return_value=segments)
    monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_a: segmenter)


def _attempts(extractor):
    return [(a.strategy, a.reason_flags) for a in extractor._segmentation_attempts]


async def test_reference_segmentation_has_no_llm_tier_in_its_fallback_chain(monkeypatch):
    extractor = _reference_extractor(
        _ref_contents(), REF_SEG_LLM_FALLBACK=True, REF_SEG_REGION_ANCHORS=False
    )
    _crf_segments(monkeypatch, ["Smith J. (2020). Reference. Journal 1:1-5."])

    segments = await extractor._segment_fallback_chain("source reference text")

    assert segments == ["Smith J. (2020). Reference. Journal 1:1-5."]
    assert _attempts(extractor) == [
        ("region", ("tier_disabled",)),
        ("llm_anchor", ("tier_disabled",)),
        ("crf", ()),
    ]
    [warning] = extractor.contents.processing_warnings
    assert warning.code == WarningCode.REF_SEG_CRF_FALLBACK
    assert warning.message == "fast mode has no LLM tier"


async def test_an_explicit_llm_segmentation_strategy_falls_through_to_the_crf(monkeypatch):
    extractor = _reference_extractor(_ref_contents())
    _crf_segments(monkeypatch, ["crf segment"])

    segments = await extractor._segment_llm_then_crf("some ref text", try_region=False)

    assert segments == ["crf segment"]
    assert _attempts(extractor) == [("llm_anchor", ("tier_disabled",)), ("crf", ())]
    assert [w.message for w in extractor.contents.processing_warnings] == [
        "fast mode has no LLM tier"
    ]


async def test_an_explicit_llm_segmentation_strategy_still_tries_the_region_anchors(monkeypatch):
    summaries = [
        RegionSummary(page=9, index=i, label="reference_content", bbox=None, content=ref)
        for i, ref in enumerate((_REF1, _REF2, _REF3, _REF4))
    ]
    extractor = _reference_extractor(_ref_contents(summaries))
    crf = mock.Mock()
    monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", crf)

    segments = await extractor._segment_llm_then_crf(
        "\n".join([_REF1, _REF2, _REF3, _REF4]), try_region=True
    )

    assert len(segments) == 4
    assert segments[0].startswith("Smith, J.")
    crf.assert_not_called()


@pytest.mark.parametrize("asked", ["llm", "llm-chunked", "ref-llm", "LLM"])
def test_an_llm_parse_strategy_is_coerced_to_the_local_parser(asked):
    from bibr.extract.ref_extractor import _resolve_ref_strategies

    settings = GlobalSettings()
    assert _resolve_ref_strategies("geom", asked, settings=settings, llm_skipped=True) == (
        "geom",
        "ner",
    )
    # Not in fast mode: the strategy is what was asked.
    assert _resolve_ref_strategies("geom", asked, settings=settings)[1] == asked.lower()


@pytest.mark.parametrize("asked", ["ner", "off"])
def test_the_local_parse_strategies_are_kept_in_fast_mode(asked):
    from bibr.extract.ref_extractor import _resolve_ref_strategies

    assert _resolve_ref_strategies(None, asked, settings=GlobalSettings(), llm_skipped=True)[1] == (
        asked
    )


async def test_extract_parses_with_the_local_parser_whatever_the_setting_says(monkeypatch):
    refs = [_REF1, _REF2, _REF3]
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    extractor = _reference_extractor(
        contents,
        REF_SEG_STRATEGY="llm",
        REF_PARSE_STRATEGY="llm",
        REF_NER_LLM_FALLBACK=True,
        REF_LLM_BASE_URL="http://reference.test/v1",
        REF_LLM_MODEL="trained-ref",
    )
    _crf_segments(monkeypatch, refs)
    parsed = [
        PaperReference(
            bib_id=index,
            title=f"Reference {index}",
            authors="Smith J",
            year=2020,
            container="Journal",
            volume=None,
            first_page=None,
        )
        for index, _ in enumerate(refs, start=1)
    ]
    ner = mock.Mock(return_value=parsed)
    monkeypatch.setattr(extractor, "_parse_references_ner", ner)
    ref_df = pd.DataFrame({"text": refs, "page_number": [9, 9, 9]})

    result = await extractor.extract(ref_df)

    ner.assert_called_once()
    assert [ref.title for ref in result] == [ref.title for ref in parsed]
    assert contents.reference_review_receipt is None


async def test_the_ner_review_is_skipped_for_a_configured_server(monkeypatch):
    from bibr.extract.ref_extractor import _parse_refs_via_ner

    extractor = _reference_extractor(
        _ref_contents(),
        REF_NER_LLM_FALLBACK=True,
        REF_LLM_BASE_URL="http://reference.test/v1",
        REF_LLM_MODEL="trained-ref",
    )
    review = AsyncMock()
    monkeypatch.setattr(extractor, "_parse_references_ner_review_aligned", review)
    monkeypatch.setattr(extractor, "_parse_references_ner", mock.Mock(return_value=["parsed"]))

    assert await _parse_refs_via_ner(extractor, "text", ["r1"]) == ["parsed"]
    review.assert_not_awaited()


async def test_the_aligned_ner_review_returns_the_ner_parse_untouched(monkeypatch):
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    extractor = _reference_extractor(
        contents,
        REF_NER_LLM_FALLBACK=True,
        REF_LLM_BASE_URL="http://reference.test/v1",
        REF_LLM_MODEL="trained-ref",
    )
    original = [None, mock.sentinel.second]
    monkeypatch.setattr(
        extractor, "_parse_references_ner_aligned", mock.Mock(return_value=original)
    )

    result = await extractor._parse_references_ner_review_aligned(["Work 2020", "Work 2021"])

    assert result is original
    assert contents.reference_review_receipt is None


# ---------------------------------------------------------------------------
# Row 13: Tier 3 citations
# ---------------------------------------------------------------------------


def _citation_inputs():
    """Two author-year citations: Jones is in the reference list, Smith is not."""
    sentences = [
        PaperSentence(
            text_id=1,
            text="This was shown by Smith (2018) in detail.",
            section_id=1,
            paragraph_id=1,
            page_number=1,
        ),
        PaperSentence(
            text_id=2,
            text="As Jones (2019) reported.",
            section_id=1,
            paragraph_id=2,
            page_number=1,
        ),
    ]
    sections = [PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0)]
    references = [
        PaperReference(
            bib_id=1,
            title="Another study",
            first_page=None,
            volume=None,
            authors="Jones",
            year=2019,
            container=None,
        )
    ]
    return sentences, sections, references


async def test_tier_three_is_skipped_and_tiers_one_and_two_still_run():
    from bibr.structure.citation_linker import detect_bib_xrefs_with_receipt

    sentences, sections, references = _citation_inputs()

    xrefs, receipt = await detect_bib_xrefs_with_receipt(
        sentences, sections, references, llm_client=SkippedLlmClient()
    )

    # The citation the reference list settles is linked on the deterministic tiers.
    assert [(x.text_id, x.xref_id, x.tier) for x in xrefs] == [(2, 1, "author-year")]
    # The other stays unlinked, and no Tier 3 candidate or failed call is recorded for it.
    [unlinked] = [c for c in receipt.candidates if not c.accepted]
    assert (unlinked.text_id, unlinked.style) == (1, "author-year")
    assert unlinked.rejection_reasons == ("no_reference_match",)


async def test_the_unlinked_citations_are_counted_on_the_paper():
    from bibr.pipeline.stages.post_parse import _link_citations

    sentences, sections, references = _citation_inputs()
    contents = PaperContents(
        sentences=sentences, sections=sections, tables=[], links=[], sections_text={}
    )
    metadata = PaperMetadata(doi="", title="T", references=references)

    await _link_citations(contents, metadata, "h", SkippedLlmClient())

    assert contents.llm_skipped == {"citations": 1}
    assert WarningCode.CITATION_LLM_FAILED not in _codes(contents.processing_warnings)


# ---------------------------------------------------------------------------
# Row 14: equations
# ---------------------------------------------------------------------------


def _equation_inputs():
    sections = [
        PaperSection(
            section_id=1,
            header="Results",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.RESULTS,
        )
    ]
    sentences = [
        PaperSentence(text_id=1, text="t(28) = 3.42, p < .001", section_id=1, paragraph_id=1),
        PaperSentence(text_id=2, text="Ambiguous (data 123) result.", section_id=1, paragraph_id=2),
    ]
    return sentences, sections


async def test_equations_keep_the_regex_rows_and_count_the_candidates_unread():
    from bibr.extract.equation_extractor import EquationExtractor

    sentences, sections = _equation_inputs()
    regex_only = await EquationExtractor().extract_with_llm_fallback(
        sentences, sections, llm_client=None
    )
    extractor = EquationExtractor()

    equations = await extractor.extract_with_llm_fallback(
        sentences, sections, llm_client=SkippedLlmClient()
    )

    assert regex_only
    assert equations == regex_only
    assert extractor.llm_skipped_candidates == 1
    assert extractor.llm_batch_failures == []


async def test_the_unread_equation_candidates_are_counted_on_the_paper():
    from bibr.pipeline.stages.post_parse import _extract_metadata_and_equations

    sentences, sections = _equation_inputs()
    contents = SimpleNamespace(
        sentences=sentences,
        sections=sections,
        preparsed_metadata=None,
        processing_warnings=[],
        equations=None,
        llm_skipped={},
    )
    metadata = PaperMetadata(doi="", title="T")
    extractor = SimpleNamespace(
        extract_all_metadata=AsyncMock(return_value=metadata), validation_issues=[]
    )

    with mock.patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor):
        result = await _extract_metadata_and_equations(contents, "h", False, SkippedLlmClient())

    assert result is metadata
    assert contents.equations
    assert contents.llm_skipped == {"equations": 1}
    assert contents.processing_warnings == []


# ---------------------------------------------------------------------------
# Row 15: research integrity
# ---------------------------------------------------------------------------


async def test_research_integrity_keeps_the_statements_unparsed():
    from bibr.extract.field_decisions import field_decisions_of
    from bibr.extract.research_integrity import extract_structured_integrity

    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    metadata = PaperMetadata(
        doi="",
        title="T",
        authors=[PaperAuthor(author_id=1, given="A", family="B", affiliation="Uni X")],
    )

    await extract_structured_integrity(contents, metadata, SkippedLlmClient(), "h")

    assert metadata.funding == []
    assert contents.llm_skipped == {"integrity": 1}
    assert contents.processing_warnings == []
    assert field_decisions_of(metadata).rules()["funding"] == "skipped_fast_mode"


async def test_nothing_to_parse_is_not_counted_as_skipped():
    from bibr.extract.research_integrity import extract_structured_integrity

    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    metadata = PaperMetadata(doi="", title="T")

    await extract_structured_integrity(contents, metadata, SkippedLlmClient(), "h")

    assert contents.llm_skipped == {}


# ---------------------------------------------------------------------------
# The stages around the guards
# ---------------------------------------------------------------------------


async def test_post_parse_runs_end_to_end_on_the_stand_in_client():
    from bibr.extract import ref_extractor
    from bibr.pipeline.stages.post_parse import post_parse
    from bibr.structure.section_classifier import SECTION_CLASSIFIER_DEGRADED_FAST_WARNING

    contents = PaperContents(
        sentences=[],
        sections=[
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(section_id=1, header="Introduction", level=1, parent_section_id=0),
            PaperSection(section_id=2, header="References", level=1, parent_section_id=0),
            PaperSection(section_id=3, header="Flurbulations", level=1, parent_section_id=0),
        ],
        tables=[],
        links=[],
        sections_text={0: "", 1: "intro body", 2: "ref 1; ref 2", 3: "something unfamiliar"},
        detected_title="My Paper",
    )
    extractor = MagicMock()
    extractor.extract_all_metadata = AsyncMock(return_value=PaperMetadata(doi="", title="T"))
    resolved: list[tuple[str, str]] = []
    real_resolve = ref_extractor._resolve_ref_strategies

    def resolve(*args, **kwargs):
        resolved.append(real_resolve(*args, **kwargs))
        return resolved[-1]

    with (
        mock.patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor),
        mock.patch("bibr.extract.ref_extractor._resolve_ref_strategies", resolve),
    ):
        paper = await post_parse(
            contents=contents,
            file_name="x.pdf",
            file_hash="deadbeef",
            llm_client=SkippedLlmClient(),
            ref_parse_strategy="llm",
        )

    # The strategy the run resolves is the local parser, not the one asked for.
    assert [parse for _seg, parse in resolved] == ["ner"]
    flurbulations = next(s for s in paper.contents.sections if s.header == "Flurbulations")
    assert flurbulations.section_type == CanonicalSection.UNKNOWN
    assert SECTION_CLASSIFIER_DEGRADED_FAST_WARNING in paper.contents.processing_warnings
    assert paper.contents.llm_skipped == {}


def _fast_settings() -> GlobalSettings:
    from bibr.run_mode import fast_settings

    return fast_settings(GlobalSettings())


def _segmented(parser) -> PaperContents:
    contents = parser.parse()
    texts = [[entry.text] for entry in parser.assembler.entries if entry.needs_segmentation]
    parser.apply_segmentation(contents, texts)
    parser.create_content_sections(contents)
    return contents


def _docx_contents(build) -> PaperContents:
    import io

    docx = pytest.importorskip("docx")
    from bibr.input.docx_native import DocxParser

    doc = docx.Document()
    build(doc)
    buf = io.BytesIO()
    doc.save(buf)
    return _segmented(DocxParser(buf.getvalue()))


def _titled_docx(doc) -> None:
    doc.add_heading("Memory and Attention in Toddlers", level=0)  # the Title style
    doc.add_paragraph("Ana Ruiz and Ben Okafor")
    doc.add_heading("Abstract", level=1)
    doc.add_paragraph("We studied memory in 400 toddlers. Attention predicted recall.")
    doc.add_heading("Introduction", level=1)
    doc.add_paragraph("Memory matters. It is studied widely.")


def _headingless_docx(doc) -> None:
    doc.add_paragraph("We studied memory in 400 toddlers. Attention predicted recall.")
    doc.add_paragraph("Memory matters. It is studied widely.")


def _epub_bytes() -> bytes:
    import io
    import zipfile

    container = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    package = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/"
         version="3.0">
  <metadata>
    <dc:title>Memory and Attention in Toddlers</dc:title>
    <dc:creator>Ana Ruiz</dc:creator>
  </metadata>
  <manifest><item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    chapter = b"""<html xmlns="http://www.w3.org/1999/xhtml"><body><section>
  <h1>Introduction</h1><p>Memory matters. It is studied widely.</p>
</section></body></html>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", package)
        zf.writestr("OEBPS/chapter1.xhtml", chapter)
    return buf.getvalue()


async def _fast_post_parse(contents, name, **kwargs):
    from bibr.pipeline.stages.post_parse import post_parse

    return await post_parse(
        contents,
        name,
        "deadbeef",
        llm_client=SkippedLlmClient(),
        settings=_fast_settings(),
        **kwargs,
    )


async def test_a_docx_has_no_layout_for_the_tagger_and_says_so():
    contents = _docx_contents(_titled_docx)
    assert contents.region_summaries == [] and contents.front_tagger_result is None

    paper = await _fast_post_parse(contents, "x.docx")

    codes = _codes(paper.contents.processing_warnings)
    assert WarningCode.FRONT_MATTER_NOT_READ in codes
    assert WarningCode.FRONT_TAGGER_UNAVAILABLE not in codes
    # What the document's own headings and sections say is kept; the byline is not read.
    assert paper.metadata.title == "Memory and Attention in Toddlers"
    assert paper.metadata.abstract.startswith("We studied memory in 400 toddlers")
    assert paper.metadata.authors == []


async def test_doc_info_fills_what_a_docx_without_a_layout_leaves_empty():
    doc_info = {"title": "Properties Title", "keywords": ["memory"]}

    paper = await _fast_post_parse(
        _docx_contents(_headingless_docx), "x.docx", ocr_metadata=doc_info
    )

    assert WarningCode.FRONT_MATTER_NOT_READ in _codes(paper.contents.processing_warnings)
    assert paper.metadata.title == "Properties Title"
    assert paper.metadata.keywords == ["memory"]
    assert paper.metadata._field_decisions.sources()["title"] == "doc_info"


async def test_a_paged_input_the_tagger_missed_keeps_the_front_matter_record_in_charge():
    # The same text with page regions: the tagger's model failing to load is not
    # "no layout", so the front-matter record still owns the fields and the
    # doc-info may not fill them.
    contents = _docx_contents(_headingless_docx)
    contents.region_summaries.append(RegionSummary(page=1, index=0, label="text", bbox=None))

    paper = await _fast_post_parse(
        contents, "x.pdf", ocr_metadata={"title": "Properties Title", "keywords": ["memory"]}
    )

    codes = _codes(paper.contents.processing_warnings)
    assert WarningCode.FRONT_TAGGER_UNAVAILABLE in codes
    assert WarningCode.FRONT_MATTER_NOT_READ not in codes
    assert not paper.metadata.title
    assert not paper.metadata.keywords


async def test_an_epub_takes_its_metadata_from_the_package_and_never_asks_the_tagger():
    from bibr.input.epub_native import EpubParser

    contents = _segmented(EpubParser(_epub_bytes()))
    assert contents.preparsed_metadata is not None

    paper = await _fast_post_parse(contents, "x.epub")

    codes = _codes(paper.contents.processing_warnings)
    assert paper.metadata.title == "Memory and Attention in Toddlers"
    assert [a.family for a in paper.metadata.authors] == ["Ruiz"]
    assert WarningCode.FRONT_MATTER_NOT_READ not in codes
    assert WarningCode.FRONT_TAGGER_UNAVAILABLE not in codes
    assert WarningCode.FRONT_TAGGER_FLAGGED not in codes


def _ctx(resources, config, *papers):
    return PipelineContext(
        file_states=[FileState(path=Path(p)) for p in papers or ("x.pdf",)],
        progress=NullProgress(),
        resources=resources,
        config=config,
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (RunConfig(mode="fast"), "skipped"),
        (RunConfig(mode="fast", no_llm=True), None),
        (RunConfig(mode="full", no_llm=True), None),
        (RunConfig(mode="full"), "resources"),
    ],
)
async def test_post_parse_is_given_the_client_the_mode_calls_for(config, expected):
    from bibr.pipeline.stages.post_parse import PostParseStage

    resources = MagicMock()
    ctx = _ctx(resources, config)
    ctx.file_states[0].contents = MagicMock(layout_hints=None)
    ctx.file_states[0].file_hash = "abc"

    with mock.patch(
        "bibr.pipeline.stages.post_parse.post_parse", AsyncMock(return_value=MagicMock())
    ) as post_parse:
        await PostParseStage().run(ctx)

    client = post_parse.await_args.kwargs["llm_client"]
    if expected == "skipped":
        assert isinstance(client, SkippedLlmClient)
    elif expected == "resources":
        assert client is resources.llm_client
    else:
        assert client is None


async def test_a_fast_run_never_asks_the_resources_for_a_client():
    from bibr.pipeline.stages.post_parse import PostParseStage

    resources = MagicMock()
    type(resources).llm_client = mock.PropertyMock(side_effect=AssertionError("built a client"))
    ctx = _ctx(resources, RunConfig(mode="fast"))
    ctx.file_states[0].contents = MagicMock(layout_hints=None)

    with mock.patch(
        "bibr.pipeline.stages.post_parse.post_parse", AsyncMock(return_value=MagicMock())
    ):
        await PostParseStage().run(ctx)

    assert ctx.file_states[0].error is None


def test_a_fast_run_reports_no_usage_and_builds_no_client():
    from bibr.local.pipeline import LocalPipeline

    pipeline = object.__new__(LocalPipeline)
    pipeline._config = RunConfig(mode="fast")
    pipeline._resources = MagicMock()
    type(pipeline._resources).llm_client = mock.PropertyMock(
        side_effect=AssertionError("built a client")
    )

    assert pipeline.llm_usage_snapshot() == {}
