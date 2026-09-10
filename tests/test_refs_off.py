"""Tests for the no-references mode (``--refs off`` / ``chew(refs="off")``).

References are the dominant LLM cost; ``off`` skips reference
segmentation+parsing entirely while keeping core metadata, sections, and
equations — unlike ``--no-llm``, which also drops those.
"""

from unittest import mock

import pandas as pd

from bibr.extract.extractor import MetadataExtractor
from bibr.models import PaperMetadata
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection


def _extractor_with_ref_section(**kwargs) -> MetadataExtractor:
    sections = ["Introduction"] * 3 + ["References"] * 3
    texts = [f"Intro {i}" for i in range(3)] + [
        "Smith J. (2020). Paper A. Nature, 10, 1-5.",
        "Jones A. (2019). Paper B. Science, 20, 10-15.",
        "Brown B. (2021). Paper C. Cell, 30, 100-110.",
    ]
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame({"section_name": sections, "text": texts})
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = [
        PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
        PaperSection(1, "References", 2, None, CanonicalSection.REFERENCES, 1.0),
    ]
    contents.sentences = []
    return MetadataExtractor(contents, **kwargs)


class TestExtractorRefsOff:
    async def test_off_skips_reference_extraction(self):
        ext = _extractor_with_ref_section(ref_parse_strategy="off")
        ext.extract_core_metadata = mock.AsyncMock()
        ext.metadata = PaperMetadata(doi="", title="Test", keywords=[], authors=[])
        ext._collect_reference_rows = mock.MagicMock(
            side_effect=AssertionError("ref rows must not be collected with refs=off")
        )
        ext._extract_references = mock.AsyncMock(
            side_effect=AssertionError("ref extraction must not run with refs=off")
        )

        meta = await ext.extract_all_metadata()

        assert meta.references == []
        assert meta.references_incomplete is False
        ext.extract_core_metadata.assert_awaited()

    async def test_off_from_settings_skips_reference_extraction(self, monkeypatch):
        import bibr.config

        monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "off")
        ext = _extractor_with_ref_section()
        ext.extract_core_metadata = mock.AsyncMock()
        ext.metadata = PaperMetadata(doi="", title="Test", keywords=[], authors=[])
        ext._collect_reference_rows = mock.MagicMock(
            side_effect=AssertionError("ref rows must not be collected with refs=off")
        )

        meta = await ext.extract_all_metadata()

        assert meta.references == []
        assert meta.references_incomplete is False


class TestNativeRefsOff:
    async def test_off_clears_stale_preparsed_references_and_failure_state(self):
        from bibr.config import snapshot_settings
        from bibr.models import PaperReference
        from bibr.pipeline.stages.post_parse import _resolve_preparsed_references

        stale_ref = PaperReference(
            bib_id=1,
            title="Stale",
            first_page=None,
            volume=None,
            authors=None,
            year=2020,
            container=None,
        )
        metadata = PaperMetadata(
            doi="10.1/native",
            title="Native",
            references=[stale_ref],
            references_incomplete=True,
        )
        metadata._references_incomplete_diagnostic = "RuntimeError: stale failure"
        contents = mock.MagicMock()
        contents.native_references = [stale_ref]

        result = await _resolve_preparsed_references(
            contents,
            metadata,
            "deadbeef",
            mock.MagicMock(),
            "native",
            "off",
            settings=snapshot_settings(),
        )

        assert result.references == []
        assert result.references_incomplete is False
        assert result._references_incomplete_diagnostic == ""

    async def test_no_llm_off_clears_stale_preparsed_reference_state(self):
        from bibr.paper_contents import PaperContents
        from bibr.pipeline.stages.post_parse import _extract_metadata_and_equations

        metadata = PaperMetadata(
            doi="10.1/native",
            title="Native",
            references_incomplete=True,
        )
        metadata._references_incomplete_diagnostic = "RuntimeError: stale failure"
        contents = PaperContents(
            sentences=[],
            sections=[],
            tables=[],
            links=[],
            sections_text={},
            preparsed_metadata=metadata,
        )

        result = await _extract_metadata_and_equations(
            contents,
            file_hash="deadbeef",
            no_llm=True,
            llm_client=None,
            ref_parse_strategy="off",
        )

        assert result.references == []
        assert result.references_incomplete is False
        assert result._references_incomplete_diagnostic == ""


class TestUndercountWarningGate:
    async def test_refs_off_suppresses_undercount_warning(self):
        """0 refs vs many citations must NOT warn when refs are off by design."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from bibr.pipeline.stages.post_parse import post_parse

        contents = PaperContents(
            sentences=[],
            sections=[PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)],
            tables=[],
            links=[],
            sections_text={0: ""},
            detected_title="My Paper",
        )

        with (
            patch(
                "bibr.pipeline.stages.post_parse._classify_sections",
                AsyncMock(return_value=None),
            ),
            patch(
                "bibr.structure.implicit_sections.detect_implicit_sections",
                AsyncMock(return_value=None),
            ),
            patch(
                "bibr.pipeline.stages.post_parse._extract_metadata_and_equations",
                AsyncMock(return_value=PaperMetadata(doi="", title="T")),
            ),
            patch(
                "bibr.pipeline.stages.post_parse._link_citations",
                AsyncMock(return_value=None),
            ),
            patch(
                "bibr.pipeline.stages.post_parse._low_reference_count_warning",
                MagicMock(return_value="UNDERCOUNT WARNING"),
            ),
        ):
            paper = await post_parse(
                contents=contents,
                file_name="x.pdf",
                file_hash="deadbeef",
                no_llm=False,
                llm_client=MagicMock(),
                ref_parse_strategy="off",
            )

        assert "UNDERCOUNT WARNING" not in paper.processing_warnings


class TestPipelineRefsOff:
    def _enrichment_stage(self, pipeline):
        from bibr.pipeline.stages.enrich import EnrichmentStage

        # Cloud-LLM pipelines stream the back half: EnrichmentStage lives
        # inside the terminal composite stage rather than the top-level list.
        candidates = list(pipeline._stages) + [
            s._enrich for s in pipeline._stages if hasattr(s, "_enrich")
        ]
        (stage,) = [s for s in candidates if isinstance(s, EnrichmentStage)]
        return stage

    def test_off_drops_crossref_enricher(self, monkeypatch):
        import bibr.config
        from bibr.local.pipeline import LocalPipeline

        monkeypatch.setattr(bibr.config.Settings.crossref, "enrich", True)
        pipe = LocalPipeline(llm_backend="cloud", crossref=True, ref_parse_strategy="off")
        assert self._enrichment_stage(pipe)._enrichers == []

    def test_default_keeps_crossref_enricher(self, monkeypatch):
        import bibr.config
        from bibr.local.pipeline import LocalPipeline

        monkeypatch.setattr(bibr.config.Settings.crossref, "enrich", True)
        pipe = LocalPipeline(llm_backend="cloud", crossref=True)
        assert len(self._enrichment_stage(pipe)._enrichers) == 1


class TestConfigAcceptsOff:
    def test_ref_parse_strategy_literal_accepts_off(self):
        from bibr.config import GlobalSettings

        settings = GlobalSettings(REF_PARSE_STRATEGY="off")
        assert settings.REF_PARSE_STRATEGY == "off"
