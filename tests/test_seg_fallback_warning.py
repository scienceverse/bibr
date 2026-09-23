"""Reference-segmentation CRF-fallback surfacing.

The LLM anchor-segmenter falls back to the CRF segmenter on failure
(zero usable spans, or any exception). Each fallback must be recorded on
``PaperContents.processing_warnings`` with the stable ``REF_SEG_CRF_FALLBACK``
code so fallback frequency can be measured across an eval corpus.
A directly configured ``crf`` strategy is not a fallback and records nothing.
"""

from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

from bibr.extract.extractor import MetadataExtractor
from bibr.extract.ref_extractor import ReferenceExtractor
from bibr.paper_contents import PaperContents, PaperSection
from bibr.processing_warnings import ProcessingWarning, WarningCode

REF_TEXT = "Smith, J. (2020). A. Journal, 1, 1-10.\nDoe, A. (2019). B. Journal, 2, 11-20."


def _extractor(llm_client) -> MetadataExtractor:
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = mock.Mock()
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = []
    contents.sentences = []
    contents.processing_warnings = []
    return MetadataExtractor(contents, llm_client=llm_client)


def _fake_crf(monkeypatch):
    fake_seg = mock.Mock()
    fake_seg.segment = mock.Mock(return_value=["Smith, J. (2020). A.", "Doe, A. (2019). B."])
    monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_a: fake_seg)
    return fake_seg


class TestSegmentReferencesFallbackWarning:
    async def test_fallback_on_exception_records_warning(self, monkeypatch):
        fake_seg = _fake_crf(monkeypatch)
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(side_effect=RuntimeError("seg down"))
        ext = _extractor(llm)

        await ext.refs._segment_references(REF_TEXT, "llm")

        fake_seg.segment.assert_called_once()
        warnings = ext.contents.processing_warnings
        assert len(warnings) == 1
        assert warnings[0].code == WarningCode.REF_SEG_CRF_FALLBACK
        assert "RuntimeError" in warnings[0].message
        assert "seg down" in warnings[0].message

    async def test_fallback_on_zero_spans_records_warning(self, monkeypatch):
        fake_seg = _fake_crf(monkeypatch)
        llm = mock.Mock()
        # Anchors that do not occur in the ref text → 0 snapped spans.
        llm.segment_references = mock.AsyncMock(return_value=["NONEXISTENT ANCHOR"])
        ext = _extractor(llm)

        await ext.refs._segment_references(REF_TEXT, "llm")

        fake_seg.segment.assert_called_once()
        warnings = ext.contents.processing_warnings
        assert len(warnings) == 1
        assert warnings[0].code == WarningCode.REF_SEG_CRF_FALLBACK
        assert "LLM segmentation produced 0 usable spans" in warnings[0].message

    async def test_direct_crf_strategy_records_nothing(self, monkeypatch):
        fake_seg = _fake_crf(monkeypatch)
        llm = mock.Mock()
        ext = _extractor(llm)

        await ext.refs._segment_references(REF_TEXT, "crf")

        fake_seg.segment.assert_called_once()
        assert ext.contents.processing_warnings == []

    async def test_llm_success_records_nothing(self):
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(
            return_value=["Smith, J. (2020).", "Doe, A. (2019)."]
        )
        ext = _extractor(llm)

        ref_strings = await ext.refs._segment_references(REF_TEXT, "llm")

        assert len(ref_strings) == 2
        assert ext.contents.processing_warnings == []

    def test_code_is_stable(self):
        # Eval tooling counts this exact code — do not rename.
        assert WarningCode.REF_SEG_CRF_FALLBACK == "REF_SEG_CRF_FALLBACK"


class TestSegmentReferencesTrainingDataCapture:
    """LLM segmentation is captured for CRF training; fallback output is not."""

    async def test_llm_success_saves_seg_training_data(self, monkeypatch):
        saver = mock.Mock()
        monkeypatch.setattr(ReferenceExtractor, "_save_seg_training_data", saver)
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(
            return_value=["Smith, J. (2020).", "Doe, A. (2019)."]
        )
        ext = _extractor(llm)

        ref_strings = await ext.refs._segment_references(REF_TEXT, "llm")

        saver.assert_called_once_with(REF_TEXT, ref_strings, settings=ext.refs._settings)

    async def test_crf_fallback_does_not_save_seg_training_data(self, monkeypatch):
        _fake_crf(monkeypatch)
        saver = mock.Mock()
        monkeypatch.setattr(ReferenceExtractor, "_save_seg_training_data", saver)
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(side_effect=RuntimeError("seg down"))
        ext = _extractor(llm)

        await ext.refs._segment_references(REF_TEXT, "llm")

        saver.assert_not_called()

    async def test_direct_crf_does_not_save_seg_training_data(self, monkeypatch):
        _fake_crf(monkeypatch)
        saver = mock.Mock()
        monkeypatch.setattr(ReferenceExtractor, "_save_seg_training_data", saver)
        ext = _extractor(mock.Mock())

        await ext.refs._segment_references(REF_TEXT, "crf")

        saver.assert_not_called()


class TestCrfHardFailureGuard:
    """The CRF last resort must never let a populated references region silently
    export 0 segments — it recovers via a zero-LLM marker split, and on a truly
    unrecoverable region surfaces a HIGH-severity warning. Must hold whether CRF
    returns [] OR raises (the LIGO 118->0: geom->LLM(timeout)->CRF(raise/empty)->0).
    """

    NUMBERED = "\n".join(
        f"[{i}] Author {i}, B. C., and Someone {i}, D. E. A reasonably long title "
        f"of paper number {i}. Physical Review Letters, {i}, 2016."
        for i in range(1, 13)
    )
    PROSE = (
        "This is a long block of reference-like prose carrying no line-start "
        "numbering markers anywhere, so the marker splitter cannot recover it. "
    ) * 3

    @staticmethod
    def _ext(llm):
        return _extractor(llm)

    async def test_crf_raise_recovers_via_marker_split(self, monkeypatch):
        # _get_ner_segmenter() itself raises (e.g. missing checkpoint) — the
        # guard must catch it, not propagate to a silent references=[].
        def _raise():
            raise RuntimeError("NER_SEG_CKPT load failed")

        monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", _raise)
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(side_effect=RuntimeError("LLM seg timeout"))
        ext = self._ext(llm)

        refs = await ext.refs._segment_references(self.NUMBERED, "llm")

        assert len(refs) == 12
        assert refs[0].startswith("[1]")

    async def test_crf_empty_recovers_via_marker_split(self, monkeypatch):
        fake = mock.Mock()
        fake.segment = mock.Mock(return_value=[])
        monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_a: fake)
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(side_effect=RuntimeError("x"))
        ext = self._ext(llm)

        refs = await ext.refs._segment_references(self.NUMBERED, "llm")

        assert len(refs) == 12

    async def test_unrecoverable_region_records_hard_failure_warning(self, monkeypatch):
        fake = mock.Mock()
        fake.segment = mock.Mock(return_value=[])
        monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_a: fake)
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(side_effect=RuntimeError("x"))
        ext = self._ext(llm)

        refs = await ext.refs._segment_references(self.PROSE, "llm")

        assert refs == []
        assert any(w.code == WarningCode.REF_SEG_FAILED for w in ext.contents.processing_warnings)

    def test_marker_split_refs_pure(self):
        from bibr.extract.extractor import _marker_split_refs

        out = _marker_split_refs("[1] A. Title.\n[2] B. Title.\n[3] C. Title.")
        assert out == ["[1] A. Title.", "[2] B. Title.", "[3] C. Title."]
        # fewer than 3 markers → decline (don't fabricate from a stray "1.")
        assert _marker_split_refs("[1] A.\n[2] B.") == []
        assert _marker_split_refs("plain prose, no markers at all here.") == []


class TestPaperContentsProcessingWarnings:
    def test_field_defaults_to_empty_list(self):
        contents = _minimal_contents()
        assert contents.processing_warnings == []

    def test_instances_do_not_share_the_list(self):
        a, b = _minimal_contents(), _minimal_contents()
        a.processing_warnings.append(ProcessingWarning(WarningCode.REF_SEG_CRF_FALLBACK, "w"))
        assert b.processing_warnings == []


def _minimal_contents() -> PaperContents:
    return PaperContents(
        sentences=[],
        sections=[PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)],
        tables=[],
        links=[],
        sections_text={0: ""},
        detected_title="My Paper",
    )


class TestWarningPropagation:
    async def test_contents_warnings_reach_paper(self):
        from bibr.pipeline.stages.post_parse import post_parse

        contents = _minimal_contents()
        warning = ProcessingWarning(
            WarningCode.REF_SEG_CRF_FALLBACK, "LLM segmentation error: RuntimeError('seg down')"
        )
        contents.processing_warnings.append(warning)

        paper = await post_parse(
            contents=contents,
            file_name="x.pdf",
            file_hash="deadbeef",
            no_llm=True,
        )

        assert warning in paper.processing_warnings

    async def test_contents_warnings_precede_post_parse_warnings(self):
        from bibr.models import PaperMetadata
        from bibr.pipeline.stages.post_parse import post_parse

        contents = _minimal_contents()
        seg_warning = ProcessingWarning(
            WarningCode.REF_SEG_CRF_FALLBACK, "LLM segmentation error: ValueError('x')"
        )
        post_parse_warning = ProcessingWarning(
            WarningCode.REF_UNDER_EXTRACTION_SUSPECTED, "POST PARSE WARNING"
        )
        contents.processing_warnings.append(seg_warning)

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
                mock.Mock(return_value=post_parse_warning),
            ),
        ):
            paper = await post_parse(
                contents=contents,
                file_name="x.pdf",
                file_hash="deadbeef",
                no_llm=False,
                llm_client=MagicMock(),
            )

        assert paper.processing_warnings == [seg_warning, post_parse_warning]

    async def test_warning_reaches_exported_json(self):
        from bibr.export.json_export import export_paper_to_json
        from bibr.pipeline.stages.post_parse import post_parse

        contents = _minimal_contents()
        warning = ProcessingWarning(
            WarningCode.REF_SEG_CRF_FALLBACK, "LLM segmentation produced 0 usable spans"
        )
        contents.processing_warnings.append(warning)

        paper = await post_parse(
            contents=contents,
            file_name="x.pdf",
            file_hash="deadbeef",
            no_llm=True,
        )
        paper.extraction = {
            "producer": {"name": "bibr", "version": "0.0.0-test"},
            "completed_at": "2026-07-24T10:00:00Z",
            "settings": {
                "ref_seg": "geom",
                "ref_parse": "ner",
                "crossref_enrich": False,
                "consolidate": "off",
            },
        }
        result = export_paper_to_json(paper)

        assert {
            "code": "REF_SEG_CRF_FALLBACK",
            "message": "LLM segmentation produced 0 usable spans",
        } in result["extraction"]["warnings"]
