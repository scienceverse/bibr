"""Silent fallbacks now leave a warning that names the failure.

Each of these paths degraded to an empty or default result with only a log
line, so the export looked like a paper that genuinely had nothing there. The
values are unchanged; the export now carries a registered warning whose
message starts with the typed error code.
"""

import asyncio
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import httpx
import pandas as pd
import pytest

from bibr.config import GlobalSettings
from bibr.exceptions import LlmInvalidOutputError, LlmTimeoutError, LlmTruncatedError
from bibr.models import FundingEntry, PaperAuthor, PaperMetadata, PaperReference
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.processing_warnings import WarningCode
from bibr.schemas import AuthorLLM, AuthorsLLM, CoreMetadataLLM, PaperClassificationLLM


def _codes(warnings) -> list[str]:
    return [w.code for w in warnings]


def _timeout(task="x"):
    return LlmTimeoutError(f"Failed to {task}", cause="timed out after 9s")


# ---------------------------------------------------------------------------
# Research integrity
# ---------------------------------------------------------------------------


async def test_research_integrity_failure_is_recorded():
    from bibr.extract.research_integrity import extract_structured_integrity

    contents = mock.Mock(spec=PaperContents)
    contents.sections = []
    contents.sentences = []
    contents.preparsed_metadata = None
    contents.processing_warnings = []
    metadata = PaperMetadata(
        doi="",
        title="T",
        authors=[PaperAuthor(author_id=1, given="A", family="B", affiliation="Uni X")],
    )
    client = SimpleNamespace(extract_research_integrity=AsyncMock(side_effect=_timeout()))

    await extract_structured_integrity(contents, metadata, client, "h")

    assert metadata.funding == []
    [warning] = contents.processing_warnings
    assert warning.code == WarningCode.RESEARCH_INTEGRITY_LLM_FAILED
    assert warning.message.startswith("llm_timeout: ")


# ---------------------------------------------------------------------------
# Section classification and implicit sections
# ---------------------------------------------------------------------------


async def test_section_classifier_llm_failure_is_recorded():
    from bibr.structure.section_classifier import classify_headers_batch_async

    client = SimpleNamespace(invoke_structured=AsyncMock(side_effect=TimeoutError("slow")))
    warnings = []
    settings = GlobalSettings()
    settings.ml.section_classifier_model_id = None

    results = await classify_headers_batch_async(
        ["unfamiliar heading"], llm_client=client, settings=settings, degradation_warnings=warnings
    )

    assert results == [(CanonicalSection.UNKNOWN, 0.0, None, None)]
    assert _codes(warnings) == [WarningCode.SECTION_CLASSIFIER_LLM_FAILED]
    assert warnings[0].message == "llm_timeout: 1 header(s) were not classified"


async def test_implicit_section_llm_failure_is_recorded():
    from bibr.structure.implicit_sections import _detect_via_llm

    client = SimpleNamespace(
        invoke_structured=AsyncMock(side_effect=LlmInvalidOutputError("x", cause="y")),
        _cap_input=lambda text: text,
    )
    sentences = [PaperSentence(text_id=1, text="One.", section_id=1, paragraph_id=1, page_number=1)]
    warnings = []

    assert await _detect_via_llm(client, sentences, "h", warnings=warnings) is None
    assert _codes(warnings) == [WarningCode.IMPLICIT_SECTIONS_LLM_FAILED]
    assert warnings[0].message.startswith("llm_invalid_output: ")


# ---------------------------------------------------------------------------
# Tier 3 citation linking
# ---------------------------------------------------------------------------


def _citation_inputs():
    sentences = [
        PaperSentence(
            text_id=1,
            text="As [Jones, 2019] reported.",
            section_id=1,
            paragraph_id=1,
            page_number=1,
        )
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


async def test_citation_llm_failure_is_on_the_receipt_and_warned():
    from bibr.pipeline.stages.post_parse import _link_citations
    from bibr.structure.citation_linker import detect_bib_xrefs_with_receipt

    sentences, sections, references = _citation_inputs()

    class FailingLlm:
        async def resolve_citations(self, ambiguous_citations, reference_summary, file_hash="x"):
            raise _timeout("resolve citations")

    xrefs, receipt = await detect_bib_xrefs_with_receipt(
        sentences, sections, references, llm_client=FailingLlm()
    )
    assert xrefs == []
    reasons = {r for c in receipt.candidates if not c.accepted for r in c.rejection_reasons}
    assert "llm_failed:llm_timeout" in reasons

    contents = SimpleNamespace(
        sentences=sentences,
        sections=sections,
        xrefs=[],
        citation_receipt=None,
        processing_warnings=[],
    )
    await _link_citations(contents, SimpleNamespace(references=references), "h", FailingLlm())
    [warning] = contents.processing_warnings
    assert warning.code == WarningCode.CITATION_LLM_FAILED
    assert warning.message.startswith("llm_timeout: ")


async def test_answered_citation_call_adds_no_warning():
    from bibr.pipeline.stages.post_parse import _link_citations

    sentences, sections, references = _citation_inputs()

    class QuietLlm:
        async def resolve_citations(self, ambiguous_citations, reference_summary, file_hash="x"):
            return []

    contents = SimpleNamespace(
        sentences=sentences,
        sections=sections,
        xrefs=[],
        citation_receipt=None,
        processing_warnings=[],
    )
    await _link_citations(contents, SimpleNamespace(references=references), "h", QuietLlm())
    assert contents.processing_warnings == []


async def test_llm_client_resolve_citations_raises_instead_of_returning_empty():
    from bibr.clients.llm import LLMClient

    client = LLMClient()
    client._acquire_rate_limit = AsyncMock()
    client._invoke_structured = AsyncMock(side_effect=TimeoutError("slow"))

    with pytest.raises(LlmTimeoutError):
        await client.resolve_citations([(1, "[Jones, 2019]")], [])


# ---------------------------------------------------------------------------
# Equation LLM fallback
# ---------------------------------------------------------------------------


async def test_failed_equation_batches_are_warned_with_their_count():
    from bibr.pipeline.stages.post_parse import _extract_metadata_and_equations

    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=1,
            header="Results",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.RESULTS,
        ),
    ]
    sentences = [
        PaperSentence(
            text_id=10,
            text="Weird stat layout (t: 3.42, p: .003) regex misses.",
            section_id=1,
            paragraph_id=1,
            page_number=None,
        )
    ]
    contents = SimpleNamespace(
        sentences=sentences,
        sections=sections,
        preparsed_metadata=None,
        processing_warnings=[],
        equations=None,
    )
    client = SimpleNamespace(
        extract_equations=AsyncMock(side_effect=LlmTruncatedError("x", cause="y"))
    )
    metadata = PaperMetadata(doi="", title="T")
    extractor = SimpleNamespace(
        extract_all_metadata=AsyncMock(return_value=metadata), validation_issues=[]
    )

    with mock.patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor):
        result = await _extract_metadata_and_equations(contents, "h", False, client)

    assert result is metadata
    [warning] = contents.processing_warnings
    assert warning.code == WarningCode.EQUATION_LLM_FALLBACK_FAILED
    assert warning.message.startswith("1 of 1 LLM batch(es) failed (llm_truncated)")


# ---------------------------------------------------------------------------
# Reference section
# ---------------------------------------------------------------------------


def _extractor(sections, texts, *, layout_hints=(), paper_sections=()):
    from bibr.extract.extractor import MetadataExtractor

    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame(
        {"section_name": sections, "text": texts, "page_number": [1] * len(texts)}
    )
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = list(layout_hints)
    contents.sections = list(paper_sections)
    contents.sentences = []
    contents.processing_warnings = []
    contents.reference_boundary_reason_flags = []
    return MetadataExtractor(contents, llm_client=mock.MagicMock())


async def test_missing_reference_section_is_warned():
    ext = _extractor(["Introduction", "Results"], ["Intro text.", "Results text."])
    ext.extract_core_metadata = AsyncMock()
    ext.metadata = PaperMetadata(doi="", title="T")

    metadata = await ext.extract_all_metadata()

    assert metadata.references == []
    assert _codes(ext.contents.processing_warnings) == [WarningCode.REF_SECTION_NOT_FOUND]
    assert ext.contents.processing_warnings[0].message.endswith("; the reference list is empty")


def test_inferred_reference_section_is_warned():
    ext = _extractor(
        ["Introduction", "Untitled"],
        ["Intro text.", "Smith J. (2020). A study. Nature, 1, 1-2."],
        layout_hints=[("reference", 1)],
        paper_sections=[
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(1, "Untitled", 2, None, CanonicalSection.UNKNOWN, 0.0),
        ],
    )

    ref_df = ext._collect_reference_rows()

    assert not ref_df.empty
    assert _codes(ext.contents.processing_warnings) == [WarningCode.REF_SECTION_INFERRED]


# ---------------------------------------------------------------------------
# ROR
# ---------------------------------------------------------------------------


def test_ror_http_errors_are_counted_and_warned():
    from bibr.clients.ror import RorClient
    from bibr.enrich.organizations import enrich_organizations
    from bibr.pipeline.enricher import RorEnricher

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    settings = GlobalSettings()
    client = RorClient(
        settings=settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    metadata = PaperMetadata(
        doi="",
        title="T",
        authors=[PaperAuthor(author_id=1, given="A", family="B", affiliation="Uni Two")],
        funding=[FundingEntry(funder="Wellcome Trust", award_ids=[])],
    )

    report = asyncio.run(enrich_organizations(metadata, client, timeout=10))

    assert (report.attempted, report.matched, report.failed) == (2, 0, 2)
    assert report.failure_reasons == ("HTTP 503",)

    with mock.patch(
        "bibr.enrich.organizations.enrich_organizations", AsyncMock(return_value=report)
    ):
        outcome = asyncio.run(
            RorEnricher(settings=settings).enrich(
                SimpleNamespace(paper=SimpleNamespace(metadata=metadata))
            )
        )
    assert [(w.code, w.message) for w in outcome.warnings] == [
        (
            WarningCode.ROR_MATCHING_FAILED,
            "2/2 ROR lookups failed (HTTP 503); those strings are unmatched",
        )
    ]


def test_ror_answer_without_a_match_is_not_a_failure():
    from bibr.clients.ror import RorClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    client = RorClient(
        settings=GlobalSettings(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert asyncio.run(client.lookup("Uni Two")) == (None, None)


# ---------------------------------------------------------------------------
# Authors and classification in the core fan-out
# ---------------------------------------------------------------------------


def _core_extractor(llm_metadata: CoreMetadataLLM, *, recovered=None):
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
    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = AsyncMock(return_value=llm_metadata)
    llm_client.json_mode_reroll_is_distinct = mock.Mock(return_value=recovered is not None)
    llm_client.extract_authors = AsyncMock(return_value=recovered or AuthorsLLM(authors=[]))
    return MetadataExtractor(contents, llm_client=llm_client)


def _core(**kwargs) -> CoreMetadataLLM:
    return CoreMetadataLLM(
        title="A Normal Research Title",
        abstract="Some abstract text.",
        authors=kwargs.pop("authors", []),
        paper_type=kwargs.pop("paper_type", None),
    )


async def test_failed_author_and_classification_calls_are_warned():
    llm_metadata = _core()
    llm_metadata._field_failures = {
        "authors": "llm_timeout",
        "paper_type": "llm_invalid_output",
        "oecd_domain": "llm_invalid_output",
        "oecd_subdomain": "llm_invalid_output",
    }
    ext = _core_extractor(llm_metadata)

    await ext.extract_core_metadata()

    codes = _codes(ext.contents.processing_warnings)
    assert WarningCode.AUTHORS_LLM_FAILED in codes
    assert WarningCode.PAPER_CLASSIFICATION_FAILED in codes
    assert not [i for i in ext.validation_issues if i.code == "VAL_METADATA_FIELD_FAILED"]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("llm_truncated", WarningCode.AUTHORS_TRUNCATED),
        ("llm_invalid_output", WarningCode.AUTHORS_PARTIAL),
    ],
)
async def test_salvaged_authors_are_warned_but_kept(code, expected):
    llm_metadata = _core(authors=[AuthorLLM(given="Ada", family="Lovelace")])
    llm_metadata._authors_salvaged_after = code
    ext = _core_extractor(llm_metadata)

    await ext.extract_core_metadata()

    assert [a.family for a in ext.metadata.authors] == ["Lovelace"]
    [warning] = [w for w in ext.contents.processing_warnings if w.code == expected]
    assert warning.message == f"{code}: kept the 1 leading author(s) of an unfinished response"


async def test_extract_authors_marks_a_salvage_with_its_failure():
    import json

    from bibr.clients.llm import LLMClient
    from bibr.extract.ref_extractor import IncompleteOutputException

    raw = '{"authors": [' + json.dumps({"given": "Ada", "family": "Lovelace"}) + ', {"given": "Gr'
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=raw, tool_calls=None), finish_reason="length"
            )
        ]
    )
    client = LLMClient()
    client._acquire_rate_limit = AsyncMock()
    client._invoke_structured = AsyncMock(
        side_effect=IncompleteOutputException(last_completion=completion)
    )

    result = await client.extract_authors("byline", file_hash="h")

    assert [a.family for a in result.authors] == ["Lovelace"]
    assert result._salvaged_after == "llm_truncated"


async def test_classifier_degraded_warning_says_when_the_fallback_failed_too(monkeypatch):
    from bibr.structure import paper_classifier

    llm_metadata = _core()
    ext = _core_extractor(llm_metadata)
    ext.core._settings = GlobalSettings(ml={"paper_classifier_model_id": "some/model"})
    ext.core._explicit_classifier_runtime = False
    monkeypatch.setattr(paper_classifier, "classify_paper_async", AsyncMock(return_value=None))
    ext.llm_client.extract_paper_classification = AsyncMock(side_effect=_timeout())

    await ext.extract_core_metadata()

    warnings = {w.code: w.message for w in ext.contents.processing_warnings}
    assert warnings[WarningCode.PAPER_CLASSIFIER_DEGRADED] == (
        "trained classifier unavailable; the LLM fallback failed too (llm_timeout)"
    )
    assert warnings[WarningCode.PAPER_CLASSIFICATION_FAILED].startswith("llm_timeout: ")


async def test_classifier_degraded_warning_is_unchanged_when_the_fallback_answers(monkeypatch):
    from bibr.structure import paper_classifier

    ext = _core_extractor(_core())
    ext.core._settings = GlobalSettings(ml={"paper_classifier_model_id": "some/model"})
    ext.core._explicit_classifier_runtime = False
    monkeypatch.setattr(paper_classifier, "classify_paper_async", AsyncMock(return_value=None))
    ext.llm_client.extract_paper_classification = AsyncMock(
        return_value=PaperClassificationLLM(paper_type="empirical")
    )

    await ext.extract_core_metadata()

    warnings = [
        (w.code, w.message)
        for w in ext.contents.processing_warnings
        if w.code.startswith("PAPER_CLASSIF")
    ]
    assert warnings == [
        (
            WarningCode.PAPER_CLASSIFIER_DEGRADED,
            "trained classifier unavailable; the LLM classified the paper",
        )
    ]
    assert ext.metadata.paper_type == "empirical"
