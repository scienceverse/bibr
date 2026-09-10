"""Tests for the corrigendum/erratum guard in MetadataExtractor."""

from unittest import mock

import pandas as pd
import pytest

from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.extract.extractor import MetadataExtractor
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.schemas import AuthorLLM, CoreMetadataLLM


class TestApplyCorrectionNoticeGuard:
    """Unit tests for CoreMetadataExtractor._apply_correction_notice_guard."""

    @pytest.mark.parametrize(
        "title, expected_is_notice, expected_kind",
        [
            # Real-world positive cases (mirror v12 poor papers).
            (
                'Erratum to "Socially Stratified Epigenetic Profiles..."',
                True,
                "erratum",
            ),
            (
                "Corrigendum: Causal Inference About Good and Bad Outcomes",
                True,
                "corrigendum",
            ),
            (
                'Corrigendum to "Indulgent Foods..."',
                True,
                "corrigendum",
            ),
            ("Correction: A Study of...", True, "corrigendum"),
            ("Retraction: Some Paper Title", True, "retraction"),
            # Case insensitivity.
            ("CORRIGENDUM: All Caps", True, "corrigendum"),
            ("erratum to 'lowercase'", True, "erratum"),
            # Leading whitespace tolerated.
            ("   Corrigendum: leading spaces", True, "corrigendum"),
            # Negative: legit research titles.
            (
                "Emotional Vocalizations Are Recognized Across Cultures",
                False,
                "",
            ),
            # Negative: false-positive bait — word starts with a prefix
            # but isn't the prefix itself.
            ("An Erratic Pattern of Decision-Making", False, ""),
            ("Corrective Feedback in Online Learning", False, ""),
            # Negative: prefix is plural / variant — alternation matches
            # the singular only.
            ("Corrigenda from the editors", False, ""),
            # Negative: prefix word appears mid-string but the regex is
            # anchored with ^\s* so it must be at the start.
            ("Some paper about corrigendum revisited", False, ""),
            # Negative: empty.
            ("", False, ""),
            (None, False, ""),
        ],
    )
    def test_classification(self, title, expected_is_notice, expected_kind):
        is_notice, kind = CoreMetadataExtractor._apply_correction_notice_guard(title)
        assert is_notice is expected_is_notice
        assert kind == expected_kind


def _build_extractor_for_e2e(llm_metadata: CoreMetadataLLM) -> MetadataExtractor:
    """Build a MetadataExtractor with a mocked llm_client and minimal contents.

    The contents have just enough rows to pass the empty-DataFrame guard at
    the top of extract_core_metadata; the actual LLM call is patched so
    none of the rows' content matters for the test.
    """
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
    return MetadataExtractor(contents, llm_client=llm_client)


def _fabricated_corrigendum_llm_result() -> CoreMetadataLLM:
    """LLM result that mirrors the v12 failure: title says corrigendum but
    the LLM has populated authors / abstract / keywords from the body."""
    return CoreMetadataLLM(
        title="Corrigendum: Causal Inference About Good and Bad Outcomes",
        abstract="Because of a data-entry error, some t statistics...",
        keywords=["causal inference", "learning"],
        authors=[
            AuthorLLM(given="H. M.", family="Dorfman"),
            AuthorLLM(given="R.", family="Bhui"),
            AuthorLLM(given="B. L.", family="Hughes"),
            AuthorLLM(given="S. J.", family="Gershman"),
        ],
        oecd_domain="Social Sciences",
        oecd_subdomain="Psychology",
        paper_type="commentary",
    )


class TestExtractCoreMetadataGuardIntegration:
    """End-to-end tests proving the guard fires inside extract_core_metadata."""

    async def test_corrigendum_title_clears_fabricated_fields(self):
        ext = _build_extractor_for_e2e(_fabricated_corrigendum_llm_result())
        await ext.extract_core_metadata()

        assert ext.metadata is not None
        assert ext.metadata.title == "Corrigendum: Causal Inference About Good and Bad Outcomes"
        assert ext.metadata.authors == []
        assert ext.metadata.abstract == ""
        assert ext.metadata.keywords == []
        assert ext.metadata.oecd_l1 == ""
        assert ext.metadata.oecd_l2 == ""
        assert ext.metadata.paper_type == "corrigendum"

    async def test_erratum_title_sets_paper_type_erratum(self):
        llm_result = _fabricated_corrigendum_llm_result()
        llm_result.title = 'Erratum to "Socially Stratified Epigenetic Profiles..."'
        ext = _build_extractor_for_e2e(llm_result)
        await ext.extract_core_metadata()

        assert ext.metadata.paper_type == "erratum"
        assert ext.metadata.authors == []
        assert ext.metadata.abstract == ""

    async def test_normal_title_preserves_llm_output(self):
        llm_result = _fabricated_corrigendum_llm_result()
        llm_result.title = "Emotional Vocalizations Are Recognized Across Cultures"
        # Use a paper_type the whitelist accepts so it isn't discarded for an
        # unrelated reason.
        llm_result.paper_type = "empirical"
        ext = _build_extractor_for_e2e(llm_result)
        await ext.extract_core_metadata()

        assert ext.metadata.title == "Emotional Vocalizations Are Recognized Across Cultures"
        assert len(ext.metadata.authors) == 4
        assert ext.metadata.abstract.startswith("Because of a data-entry error")
        assert ext.metadata.keywords == ["causal inference", "learning"]
        assert ext.metadata.paper_type == "empirical"
