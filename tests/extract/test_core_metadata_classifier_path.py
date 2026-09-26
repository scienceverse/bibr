"""Tests for the trained-classifier swap in CoreMetadataExtractor.extract().

The classifier is injected as a fake (no real HF model). The dark default
(paper_classifier_model_id=None) is covered by the existing classification /
guard tests; here we exercise the configured path.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pandas as pd
import pytest

from bibr.config import Settings
from bibr.exceptions import UpstreamServiceError
from bibr.extract.extractor import MetadataExtractor
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.processing_warnings import WarningCode
from bibr.schemas import AuthorLLM, CoreMetadataLLM, PaperClassificationLLM, PaperTypeLabel
from bibr.structure import paper_classifier


def _build_extractor(llm_metadata: CoreMetadataLLM) -> MetadataExtractor:
    df = pd.DataFrame(
        {
            "section_name": ["Abstract", "1. Introduction"],
            "text": ["Some abstract text.", "Some intro text."],
            "page_number": [1, 1],
        }
    )
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = df
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = [
        PaperSection(0, "Abstract", 2, None, CanonicalSection.ABSTRACT, 1.0),
        PaperSection(1, "1. Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
    ]
    contents.sentences = []
    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = mock.AsyncMock(return_value=llm_metadata)
    llm_client.extract_paper_classification = mock.AsyncMock(
        return_value=PaperClassificationLLM(
            paper_type="commentary",
            oecd_domain="Humanities and the Arts",
            oecd_subdomain="Languages and Literature",
        )
    )
    llm_client.label_paper_type = mock.AsyncMock(
        return_value=PaperTypeLabel(paper_type="review", confidence=0.95)
    )
    return MetadataExtractor(contents, llm_client=llm_client)


def _base_llm_result() -> CoreMetadataLLM:
    return CoreMetadataLLM(
        title="Working Memory and Attention",
        abstract="We ran three experiments on working memory.",
        keywords=["memory"],
        authors=[AuthorLLM(given="A.", family="Researcher")],
        # These LLM classification fields must be IGNORED when the classifier
        # is configured — the classifier's output wins.
        oecd_domain="Humanities and the Arts",
        oecd_subdomain="Languages and Literature",
        paper_type="commentary",
    )


@pytest.fixture(autouse=True)
def _configure_classifier(monkeypatch):
    monkeypatch.setattr(Settings.ml, "paper_classifier_model_id", "fake/repo")
    monkeypatch.setattr(Settings.ml, "paper_classifier_min_confidence", 0.5)
    monkeypatch.setattr(Settings.ml, "paper_classifier_llm_escalation", True)
    # Never touch the real cache/HF hub.
    monkeypatch.setattr(paper_classifier, "_paper_model_cache", object())


class TestClassifierPopulatesFields:
    async def test_all_three_fields_and_confidences_populate(self, monkeypatch):
        async def fake_classify(title, abstract):  # noqa: ARG001
            return (
                "Social Sciences",
                0.93,
                "Psychology and Cognitive Sciences",
                0.71,
                "empirical",
                0.88,
            )

        monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)

        ext = _build_extractor(_base_llm_result())
        await ext.extract_core_metadata()

        m = ext.metadata
        assert m.oecd_l1 == "Social Sciences"
        assert m.oecd_l2 == "Psychology and Cognitive Sciences"
        assert m.paper_type == "empirical"
        assert m.oecd_confidence == pytest.approx(0.93)
        assert m.paper_type_confidence == pytest.approx(0.88)
        # High paper_type score → no LLM escalation.
        ext.llm_client.label_paper_type.assert_not_awaited()


class TestL2ConfidenceGate:
    async def test_low_confidence_l2_is_nulled(self, monkeypatch):
        monkeypatch.setattr(Settings.ml, "paper_classifier_l2_min_confidence", 0.5)

        async def fake_classify(title, abstract):  # noqa: ARG001
            # L2 below the gate → emitted as null; L1/paper_type unaffected.
            return ("Social Sciences", 0.93, "Sociology", 0.31, "empirical", 0.88)

        monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)

        ext = _build_extractor(_base_llm_result())
        await ext.extract_core_metadata()

        m = ext.metadata
        assert m.oecd_l2 == ""
        assert m.oecd_l1 == "Social Sciences"
        assert m.oecd_confidence == pytest.approx(0.93)

    async def test_high_confidence_l2_is_kept(self, monkeypatch):
        monkeypatch.setattr(Settings.ml, "paper_classifier_l2_min_confidence", 0.5)

        async def fake_classify(title, abstract):  # noqa: ARG001
            return ("Social Sciences", 0.93, "Sociology", 0.77, "empirical", 0.88)

        monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)

        ext = _build_extractor(_base_llm_result())
        await ext.extract_core_metadata()

        assert ext.metadata.oecd_l2 == "Sociology"


class TestPaperTypeEscalation:
    async def test_low_confidence_escalates_to_label_paper_type(self, monkeypatch):
        async def fake_classify(title, abstract):  # noqa: ARG001
            return (
                "Social Sciences",
                0.93,
                "Psychology and Cognitive Sciences",
                0.71,
                "empirical",
                0.20,  # below min_confidence → escalate paper_type only
            )

        monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)

        ext = _build_extractor(_base_llm_result())
        await ext.extract_core_metadata()

        ext.llm_client.label_paper_type.assert_awaited_once()
        m = ext.metadata
        # Escalated paper_type + its confidence replace the low-confidence one.
        assert m.paper_type == "review"
        assert m.paper_type_confidence == pytest.approx(0.95)
        # OECD is untouched by paper_type escalation.
        assert m.oecd_l1 == "Social Sciences"
        assert m.oecd_confidence == pytest.approx(0.93)

    async def test_no_escalation_when_disabled(self, monkeypatch):
        monkeypatch.setattr(Settings.ml, "paper_classifier_llm_escalation", False)

        async def fake_classify(title, abstract):  # noqa: ARG001
            return ("Social Sciences", 0.9, "", 0.0, "empirical", 0.10)

        monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)

        ext = _build_extractor(_base_llm_result())
        await ext.extract_core_metadata()

        ext.llm_client.label_paper_type.assert_not_awaited()
        assert ext.metadata.paper_type == "empirical"
        assert ext.metadata.paper_type_confidence == pytest.approx(0.10)


class TestGuardsStillFireOnClassifierPath:
    async def test_correction_notice_guard_fires(self, monkeypatch):
        async def fake_classify(title, abstract):  # noqa: ARG001
            return (
                "Social Sciences",
                0.9,
                "Psychology and Cognitive Sciences",
                0.8,
                "empirical",
                0.9,
            )

        monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)

        llm = _base_llm_result()
        llm.title = "Corrigendum: Working Memory and Attention"
        ext = _build_extractor(llm)
        await ext.extract_core_metadata()

        m = ext.metadata
        assert m.paper_type == "corrigendum"
        assert m.authors == []
        assert m.abstract == ""
        assert m.keywords == []
        assert m.oecd_l1 == ""
        assert m.oecd_l2 == ""
        # Guard-derived values carry no classifier confidence.
        assert m.oecd_confidence is None
        assert m.paper_type_confidence is None

    async def test_long_commentary_abstract_survives_core_and_finalization(self, monkeypatch):
        async def fake_classify(title, abstract):  # noqa: ARG001
            return (
                "Social Sciences",
                0.9,
                "Psychology and Cognitive Sciences",
                0.8,
                "commentary",
                0.9,
            )

        monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)

        llm = _base_llm_result()
        llm.paper_type = "commentary"
        llm.abstract = "x" * 3000
        llm.keywords = []
        ext = _build_extractor(llm)
        await ext.extract_core_metadata()

        assert ext.metadata.paper_type == "commentary"
        assert ext.metadata.abstract == llm.abstract

        from bibr.pipeline.stages.post_parse import _finalize_abstract_and_keywords

        _finalize_abstract_and_keywords(ext.core.contents, ext.metadata)
        assert ext.metadata.abstract == llm.abstract


class TestFallsBackWhenClassifierUnavailable:
    async def test_none_prediction_uses_llm_validation(self, monkeypatch):
        """classify_paper_async returning None (e.g. ml extra missing) must fall
        back to the LLM validation path and leave confidences None."""

        async def fake_classify(title, abstract):  # noqa: ARG001
            return None

        monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)

        ext = _build_extractor(_base_llm_result())
        await ext.extract_core_metadata()

        ext.llm_client.extract_paper_classification.assert_awaited_once()
        m = ext.metadata
        # LLM validation canonicalizes the LLM-provided fields.
        assert m.oecd_l1 == "Humanities and the Arts"
        assert m.oecd_l2 == "Languages and Literature"
        assert m.paper_type == "commentary"
        assert m.oecd_confidence is None
        assert m.paper_type_confidence is None


async def test_configured_classifier_omits_initial_broad_llm_call(monkeypatch):
    async def fake_classify(title, abstract):  # noqa: ARG001
        return ("Social Sciences", 0.93, "Sociology", 0.8, "empirical", 0.88)

    monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)
    ext = _build_extractor(_base_llm_result())

    await ext.extract_core_metadata()

    kwargs = ext.llm_client.extract_core_metadata.await_args.kwargs
    assert kwargs["include_classification"] is False
    ext.llm_client.extract_paper_classification.assert_not_awaited()


async def test_none_prediction_calls_delayed_broad_fallback(monkeypatch):
    async def fake_classify(title, abstract):  # noqa: ARG001
        return None

    monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)
    ext = _build_extractor(_base_llm_result())

    await ext.extract_core_metadata()

    ext.llm_client.extract_paper_classification.assert_awaited_once()
    fallback_text = ext.llm_client.extract_paper_classification.await_args.args[0]
    assert "Some abstract text." in fallback_text
    assert "Some intro text." not in fallback_text
    assert ext.metadata.paper_type == "commentary"
    assert ext.metadata.paper_type_confidence is None


async def test_classifier_exception_calls_delayed_broad_fallback(monkeypatch):
    async def fake_classify(title, abstract):  # noqa: ARG001
        raise RuntimeError("classifier unavailable")

    monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)
    ext = _build_extractor(_base_llm_result())

    await ext.extract_core_metadata()

    ext.llm_client.extract_paper_classification.assert_awaited_once()
    assert ext.metadata.paper_type == "commentary"


async def test_merged_core_reuses_classification_when_classifier_returns_none(monkeypatch):
    monkeypatch.setattr(Settings.llm, "merged_core_metadata", True)

    async def fake_classify(title, abstract):  # noqa: ARG001
        return None

    monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)
    ext = _build_extractor(_base_llm_result())

    await ext.extract_core_metadata()

    ext.llm_client.extract_paper_classification.assert_not_awaited()
    assert ext.metadata.paper_type == "commentary"
    assert ext.metadata.oecd_l1 == "Humanities and the Arts"
    assert ext.metadata.oecd_l2 == "Languages and Literature"


async def test_merged_core_reuses_classification_when_classifier_raises(monkeypatch):
    monkeypatch.setattr(Settings.llm, "merged_core_metadata", True)

    async def fake_classify(title, abstract):  # noqa: ARG001
        raise RuntimeError("classifier unavailable")

    monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)
    ext = _build_extractor(_base_llm_result())

    await ext.extract_core_metadata()

    ext.llm_client.extract_paper_classification.assert_not_awaited()
    assert ext.metadata.paper_type == "commentary"
    assert ext.metadata.oecd_l1 == "Humanities and the Arts"
    assert ext.metadata.oecd_l2 == "Languages and Literature"


async def test_delayed_broad_fallback_failure_keeps_core_metadata(monkeypatch):
    async def fake_classify(title, abstract):  # noqa: ARG001
        return None

    monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)
    ext = _build_extractor(_base_llm_result())
    ext.llm_client.extract_paper_classification.side_effect = UpstreamServiceError(
        "LLM", "classification down", None
    )

    await ext.extract_core_metadata()

    assert ext.metadata.title == "Working Memory and Attention"
    assert len(ext.metadata.authors) == 1
    assert ext.metadata.paper_type == ""
    assert ext.metadata.oecd_l1 == ""


async def test_classifier_cancellation_does_not_start_fallback(monkeypatch):
    async def fake_classify(title, abstract):  # noqa: ARG001
        raise asyncio.CancelledError

    monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)
    ext = _build_extractor(_base_llm_result())

    with pytest.raises(asyncio.CancelledError):
        await ext.extract_core_metadata()

    ext.llm_client.extract_paper_classification.assert_not_awaited()


async def test_no_configured_classifier_keeps_initial_broad_call(monkeypatch):
    monkeypatch.setattr(Settings.ml, "paper_classifier_model_id", None)
    ext = _build_extractor(_base_llm_result())

    await ext.extract_core_metadata()

    kwargs = ext.llm_client.extract_core_metadata.await_args.kwargs
    assert kwargs["include_classification"] is True
    ext.llm_client.extract_paper_classification.assert_not_awaited()
    assert ext.metadata.paper_type == "commentary"


class TestDegradedClassifierIsVisibleInTheExport:
    async def test_empty_input_takes_llm_path_with_no_degraded_warning(self, monkeypatch):
        """Empty title+abstract is missing input, not a classifier outage: the
        trained classifier must not run and no PAPER_CLASSIFIER_DEGRADED
        warning may reach the export — the LLM classifies from the full text.
        """
        classify_spy = mock.AsyncMock(
            side_effect=AssertionError("trained classifier must not run on empty input")
        )
        monkeypatch.setattr(paper_classifier, "classify_paper_async", classify_spy)

        llm = _base_llm_result()
        ext = _build_extractor(llm)
        ext.contents.processing_warnings = []
        result = await ext.core._classify_paper("", "", llm, "full classification text here")

        classify_spy.assert_not_awaited()
        # The delayed broad LLM fallback decides, with null confidences.
        assert result == (
            "commentary",
            "Humanities and the Arts",
            "Languages and Literature",
            None,
            None,
        )
        ext.llm_client.extract_paper_classification.assert_awaited_once()
        assert not [
            w
            for w in ext.contents.processing_warnings
            if w.code == WarningCode.PAPER_CLASSIFIER_DEGRADED
        ]

    async def test_whitespace_only_input_takes_llm_path_with_no_degraded_warning(self, monkeypatch):
        """Blank strings carry no signal either — same quiet LLM path."""
        classify_spy = mock.AsyncMock(
            side_effect=AssertionError("trained classifier must not run on blank input")
        )
        monkeypatch.setattr(paper_classifier, "classify_paper_async", classify_spy)

        llm = _base_llm_result()
        ext = _build_extractor(llm)
        ext.contents.processing_warnings = []
        result = await ext.core._classify_paper("   ", "  ", llm, "full classification text here")

        classify_spy.assert_not_awaited()
        assert result[0] == "commentary"
        assert not [
            w
            for w in ext.contents.processing_warnings
            if w.code == WarningCode.PAPER_CLASSIFIER_DEGRADED
        ]

    async def test_unavailable_classifier_marks_the_export(self, monkeypatch):
        """Configured but not answering (core install, failed load): the LLM decides
        and processing_warnings must say so."""

        async def fake_classify(title, abstract):  # noqa: ARG001
            return None

        monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)
        ext = _build_extractor(_base_llm_result())
        await ext.extract_core_metadata()
        assert any(
            w.code == WarningCode.PAPER_CLASSIFIER_DEGRADED
            for w in ext.contents.processing_warnings
        ), ext.contents.processing_warnings

    async def test_classifier_error_is_recorded_by_type_only(self, monkeypatch):
        async def fake_classify(title, abstract):  # noqa: ARG001
            raise RuntimeError("tokenizer exploded on private document text")

        monkeypatch.setattr(paper_classifier, "classify_paper_async", fake_classify)
        ext = _build_extractor(_base_llm_result())
        await ext.extract_core_metadata()
        warning = next(
            w
            for w in ext.contents.processing_warnings
            if w.code == WarningCode.PAPER_CLASSIFIER_DEGRADED
        )
        assert "RuntimeError" in warning.message
        assert "private document text" not in warning.message
