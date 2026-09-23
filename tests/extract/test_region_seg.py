"""Layout-region anchor segmentation — the zero-cost cascade fallback tier.

``reference_content`` regions align to individual bib entries with high
recall but carry noise (CRediT/funding blocks, page-break continuation
fragments). The tier filters region starts to genuine reference onsets,
snaps them onto ``ref_text`` with the shared anchor machinery, and only
claims the segmentation when enough anchors align.
"""

from unittest import mock
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from bibr.config import Settings
from bibr.extract.region_seg import (
    region_anchor_texts,
    region_chunks,
    segment_by_region_anchors,
)
from bibr.paper_contents import PaperContents, RegionSummary
from bibr.processing_warnings import WarningCode


def _rs(label: str, content: str, page: int = 9, index: int = 0) -> RegionSummary:
    return RegionSummary(page=page, index=index, label=label, bbox=None, content=content)


_REF1 = "Smith, J., & Jones, K. (2020). Attention and memory. Psych Review, 12(3), 45-67."
_REF2 = "Doe, A. (2019). Seeing things clearly. Journal of Vision, 2, 11-20."
_REF3 = "Nguyen, T. H. (2021). Replication in the wild. Meta Science, 4(1), 1-19."
_REF4 = "World Health Organization. (2018). Global report on falls. WHO Press."


class TestRegionAnchorTexts:
    def test_keeps_reference_onsets_in_order(self):
        summaries = [
            _rs("reference_content", _REF1, index=0),
            _rs("reference_content", _REF2, index=1),
            _rs("reference_content", _REF4, index=2),
        ]
        anchors = region_anchor_texts(summaries)
        assert len(anchors) == 3
        assert anchors[0].startswith("Smith, J.")
        assert anchors[2].startswith("World Health Organization. (2018)")

    def test_filters_noise_and_continuation_regions(self):
        summaries = [
            _rs("reference_content", "Author contributions: J.S. conceived the study; K.J."),
            _rs("reference_content", _REF1),
            # page-break continuation fragment: starts mid-reference
            _rs("reference_content", "of Applied Psychology, 103(2), 182-214."),
            _rs("reference_content", _REF2),
            _rs("reference_content", "This research was funded by the ERC under grant 12345."),
        ]
        anchors = region_anchor_texts(summaries)
        assert len(anchors) == 2
        assert anchors[0].startswith("Smith")
        assert anchors[1].startswith("Doe")

    def test_ignores_non_reference_labels_and_empty_content(self):
        summaries = [
            _rs("text", _REF1),
            _rs("reference_content", ""),
            _rs("reference_content", None),  # type: ignore[arg-type]
            _rs("reference_content", _REF2),
        ]
        anchors = region_anchor_texts(summaries)
        assert len(anchors) == 1
        assert anchors[0].startswith("Doe")

    def test_collapses_ocr_whitespace_in_anchor(self):
        noisy = "Smith,\t\r \xa0J.,\t&\xa0Jones, K. (2020). Attention and memory."
        anchors = region_anchor_texts([_rs("reference_content", noisy)])
        assert anchors == [
            "Smith, J., & Jones, K. (2020). Attention and memory."[:80],
        ]

    def test_keeps_accented_author_date_onsets_with_full_given_names(self):
        summaries = [
            _rs(
                "reference_content",
                "Bühler, Charlotte (ed.) 1922. Quellen und Studien zur Jugendkunde.",
                index=0,
            ),
            _rs(
                "reference_content",
                "Bühring, Gerald 2007. Charlotte Bühler oder Der Lebenslauf.",
                index=1,
            ),
        ]

        anchors = region_anchor_texts(summaries)

        assert len(anchors) == 2
        assert anchors[0].startswith("Bühler, Charlotte")
        assert anchors[1].startswith("Bühring, Gerald")

    @pytest.mark.parametrize(
        "onset",
        [
            "1 European Commission. A pharmaceutical strategy for Europe. Brussels; 2020.",
            "12 Lindsay S, Cagliostro E. A systematic review. Disabil Rehabil. 2018;40:1.",
            "(1) Viñuales J. The Paris Agreement. Cambridge University Press; 2017.",
            "1- Birnie K, Petrie K. Illness perceptions. J Psychosom Res. 2011;70:12.",
            "3 – Smith J, Doe A. Trial design. Lancet. 2019;393:1.",
            # Vancouver with no numbering at all (author-initials lead).
            "Lindsay S, Cagliostro E, Albarico M. A systematic review of vocational.",
            "Rutten-van Mölken M, Karimi M. Comparing outcomes. Health Policy. 2020.",
            # Quoted-title humanities style.
            'Mayorga Hernández, María Isabel, "Modelos 3D y levantamiento", 2024.',
            # CJK author lead followed by a year.
            "田中太郎・鈴木花子 2019 認知心理学の展望 心理学評論 62(1) 1-20.",
        ],
    )
    def test_promotes_non_author_date_reference_onsets(self, onset):
        assert region_anchor_texts([_rs("reference_content", onset)]) != []

    @pytest.mark.parametrize(
        "prose",
        [
            "methods, results from 2020 were discussed in the funding statement.",
            "The report, published in 2020, summarizes the evidence.",
            "Funding support, awarded in 2021, came from the ERC.",
            # Loosened numbering must not swallow prose that opens with a
            # figure. (Prose opening with a decimal — "1.5 million people" —
            # is a pre-existing `_NUMBERED` false positive; that pattern is a
            # trained-GBM feature and is deliberately left byte-stable.)
            "12 patients were enrolled in the trial and followed for two years.",
        ],
    )
    def test_unicode_fallback_does_not_promote_prose(self, prose):
        anchors = region_anchor_texts(
            [
                _rs(
                    "reference_content",
                    prose,
                )
            ]
        )

        assert anchors == []


class TestSegmentByRegionAnchors:
    def _summaries(self):
        return [
            _rs("reference_content", _REF1, index=0),
            _rs("reference_content", _REF2, index=1),
            _rs("reference_content", _REF3, index=2),
            _rs("reference_content", _REF4, index=3),
        ]

    def test_recovers_refs_from_flat_text(self):
        ref_text = "\n".join([_REF1, _REF2, _REF3, _REF4])
        segments = segment_by_region_anchors(ref_text, self._summaries())
        assert segments is not None
        assert len(segments) == 4
        assert segments[0].startswith("Smith, J.")
        assert segments[1].startswith("Doe, A.")
        assert segments[3].startswith("World Health Organization.")

    def test_multiline_refs_span_to_next_anchor(self):
        # refs wrapped over lines: each segment must run to the next onset
        ref_text = _REF1.replace("Psych Review,", "Psych\nReview,") + "\n" + _REF2
        segments = segment_by_region_anchors(
            ref_text,
            self._summaries()[:2] + [_rs("reference_content", _REF3)],
        )
        # only 2 of 3 anchors align (REF3 is absent) — still >= alignment floor
        assert segments is not None
        assert len(segments) == 2
        assert "Review," in segments[0]

    def test_returns_none_when_too_few_usable_anchors(self):
        ref_text = "\n".join([_REF1, _REF2])
        summaries = [_rs("reference_content", _REF1), _rs("reference_content", _REF2)]
        assert segment_by_region_anchors(ref_text, summaries) is None

    def test_returns_none_when_anchors_do_not_align(self):
        # summaries from a different document must not claim the segmentation
        ref_text = "\n".join([_REF1, _REF2, _REF3, _REF4])
        other = [
            _rs("reference_content", "Brown, B. (1999). Completely unrelated. Elsewhere, 1, 1."),
            _rs("reference_content", "Green, G. (1998). Also unrelated. Nowhere, 2, 2."),
            _rs("reference_content", "White, W. (1997). Still unrelated. Anywhere, 3, 3."),
        ]
        assert segment_by_region_anchors(ref_text, other) is None


class TestCascadeWiring:
    """LLM seg failure must try region anchors before the CRF last resort."""

    def _extractor(self, llm_client, summaries):
        from bibr.extract.extractor import MetadataExtractor

        ref_texts = [_REF1, _REF2, _REF3, _REF4]
        df = pd.DataFrame({"section_name": ["References"] * 4, "text": ref_texts})
        contents = mock.Mock(spec=PaperContents)
        contents.sentences_df = df
        contents.detected_headers = []
        contents.detected_footers = []
        contents.layout_hints = []
        contents.sections = []
        contents.sentences = []
        contents.region_summaries = summaries
        return MetadataExtractor(contents, llm_client=llm_client)

    async def test_llm_failure_falls_back_to_region_anchors(self, monkeypatch):
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(side_effect=TimeoutError("boom"))
        summaries = [_rs("reference_content", r) for r in (_REF1, _REF2, _REF3, _REF4)]
        ext = self._extractor(llm, summaries)
        crf = mock.Mock()
        monkeypatch.setattr(ext.refs, "_crf_segment_or_recover", crf, raising=True)

        ref_text = "\n".join([_REF1, _REF2, _REF3, _REF4])
        segments = await ext.refs._segment_llm_then_crf(ref_text)

        assert len(segments) == 4
        assert segments[0].startswith("Smith, J.")
        crf.assert_not_called()

    async def test_region_tier_disabled_falls_through_to_crf(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings, "REF_SEG_REGION_ANCHORS", False)
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(side_effect=TimeoutError("boom"))
        summaries = [_rs("reference_content", r) for r in (_REF1, _REF2, _REF3, _REF4)]
        ext = self._extractor(llm, summaries)
        crf = mock.Mock(return_value=["crf-seg"])
        monkeypatch.setattr(ext.refs, "_crf_segment_or_recover", crf, raising=True)

        segments = await ext.refs._segment_llm_then_crf("some ref text")

        assert segments == ["crf-seg"]
        crf.assert_called_once()

    async def test_region_tier_misalignment_falls_through_to_crf(self, monkeypatch):
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(side_effect=TimeoutError("boom"))
        # summaries that do not exist in the ref text → tier declines
        summaries = [
            _rs("reference_content", "Brown, B. (1999). Unrelated. Elsewhere, 1, 1."),
            _rs("reference_content", "Green, G. (1998). Unrelated. Nowhere, 2, 2."),
            _rs("reference_content", "White, W. (1997). Unrelated. Anywhere, 3, 3."),
        ]
        ext = self._extractor(llm, summaries)
        crf = mock.Mock(return_value=["crf-seg"])
        monkeypatch.setattr(ext.refs, "_crf_segment_or_recover", crf, raising=True)

        segments = await ext.refs._segment_llm_then_crf("\n".join([_REF1, _REF2, _REF3, _REF4]))

        assert segments == ["crf-seg"]
        crf.assert_called_once()

    def test_default_setting_is_on(self, monkeypatch):
        from bibr.config import GlobalSettings

        monkeypatch.delenv("REF_SEG_REGION_ANCHORS", raising=False)
        assert GlobalSettings().REF_SEG_REGION_ANCHORS is True


class TestSourceRecallBackstop:
    """The region tier's alignment threshold is blind to what the onset filter
    discarded. The reference section's source-record count is the independent
    lower bound that catches a segmentation that silently kept 5 of 40 refs."""

    def _extractor(self, llm_client, summaries):
        from bibr.extract.extractor import MetadataExtractor

        ref_texts = [_REF1, _REF2, _REF3, _REF4]
        df = pd.DataFrame({"section_name": ["References"] * 4, "text": ref_texts})
        contents = mock.Mock(spec=PaperContents)
        contents.sentences_df = df
        contents.detected_headers = []
        contents.detected_footers = []
        contents.layout_hints = []
        contents.sections = []
        contents.sentences = []
        contents.region_summaries = summaries
        return MetadataExtractor(contents, llm_client=llm_client)

    def test_shortfall_needs_a_meaningful_record_count(self, monkeypatch):
        monkeypatch.setattr(Settings, "REF_SEG_MIN_SOURCE_RECALL", 0.6)
        ext = self._extractor(mock.Mock(), [])

        ext.refs._source_record_count = None
        assert ext.refs._source_recall_shortfall(1) is None
        # Below the 5-record floor a short reference list is indistinguishable
        # from a truncated one.
        ext.refs._source_record_count = 4
        assert ext.refs._source_recall_shortfall(1) is None
        ext.refs._source_record_count = 40
        assert ext.refs._source_recall_shortfall(5) == 40
        assert ext.refs._source_recall_shortfall(24) is None

    def test_zero_setting_disables_the_backstop(self, monkeypatch):
        monkeypatch.setattr(Settings, "REF_SEG_MIN_SOURCE_RECALL", 0.0)
        ext = self._extractor(mock.Mock(), [])
        ext.refs._source_record_count = 40

        assert ext.refs._source_recall_shortfall(1) is None

    async def test_shortfall_escalates_to_the_llm_tier(self, monkeypatch):
        monkeypatch.setattr(Settings, "REF_SEG_MIN_SOURCE_RECALL", 0.6)
        llm = mock.Mock()
        llm.segment_references = AsyncMock(return_value=[_REF1, _REF2, _REF3, _REF4])
        summaries = [_rs("reference_content", r) for r in (_REF1, _REF2, _REF3)]
        ext = self._extractor(llm, summaries)
        ext.refs._source_record_count = 40

        segments = await ext.refs._segment_fallback_chain("\n".join([_REF1, _REF2, _REF3, _REF4]))

        assert len(segments) == 4
        llm.segment_references.assert_awaited_once()
        assert any(
            w.code == WarningCode.REF_SEG_REGION_CASCADE for w in ext.contents.processing_warnings
        )

    async def test_reserved_region_segmentation_beats_crf_when_the_llm_fails(self, monkeypatch):
        monkeypatch.setattr(Settings, "REF_SEG_MIN_SOURCE_RECALL", 0.6)
        llm = mock.Mock()
        llm.segment_references = AsyncMock(side_effect=TimeoutError("boom"))
        summaries = [_rs("reference_content", r) for r in (_REF1, _REF2, _REF3)]
        ext = self._extractor(llm, summaries)
        ext.refs._source_record_count = 40
        crf = mock.Mock(return_value=["crf-seg"])
        monkeypatch.setattr(ext.refs, "_crf_segment_or_recover", crf, raising=True)

        segments = await ext.refs._segment_fallback_chain("\n".join([_REF1, _REF2, _REF3, _REF4]))

        assert len(segments) == 3
        assert segments[0].startswith("Smith, J.")
        crf.assert_not_called()
        assert any(
            w.code == WarningCode.REF_SEG_REGION_RECOVERY for w in ext.contents.processing_warnings
        )

    def test_default_setting_is_a_permissive_floor(self, monkeypatch):
        from bibr.config import GlobalSettings

        monkeypatch.delenv("REF_SEG_MIN_SOURCE_RECALL", raising=False)
        assert GlobalSettings().REF_SEG_MIN_SOURCE_RECALL == 0.6


# ----------------------------------------------------------------------------
# Cascade reorder: region anchors (zero cost) run BEFORE the LLM tier on the
# geom-decline path. REF_SEG_LLM_FALLBACK gates the LLM tier off entirely, and
# "region" is a selectable primary strategy.
# ----------------------------------------------------------------------------

REF_TEXT = "\n".join([_REF1, _REF2, _REF3, _REF4])
ANCHOR_1 = _REF1


def _cascade_extractor(llm_client, summaries):
    from bibr.extract.extractor import MetadataExtractor

    ref_texts = [_REF1, _REF2, _REF3, _REF4]
    df = pd.DataFrame({"section_name": ["References"] * 4, "text": ref_texts})
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = df
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = []
    contents.sentences = []
    contents.region_summaries = summaries
    contents.processing_warnings = []
    contents.ref_line_geometry = None  # force geom to decline
    return MetadataExtractor(contents, llm_client=llm_client)


@pytest.fixture
def extractor_with_regions(monkeypatch):
    """Geom declines (no ref_line_geometry); region anchors align with REF_TEXT."""
    llm = mock.Mock()
    llm.segment_references = AsyncMock(return_value=[])
    summaries = [_rs("reference_content", r) for r in (_REF1, _REF2, _REF3, _REF4)]
    ext = _cascade_extractor(llm, summaries)
    fake_ner = mock.Mock()
    fake_ner.segment = mock.Mock(return_value=["fallback seg"])
    monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_a: fake_ner)
    return ext


@pytest.fixture
def extractor_no_regions(monkeypatch):
    """Geom declines; no layout regions available (DOCX / no-ml install)."""
    llm = mock.Mock()
    llm.segment_references = AsyncMock(return_value=[])
    ext = _cascade_extractor(llm, [])
    fake_ner = mock.Mock()
    fake_ner.segment = mock.Mock(return_value=["fallback seg"])
    monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_a: fake_ner)
    return ext


async def test_geom_cascade_tries_region_before_llm(extractor_with_regions):
    """When geom declines, region anchors win WITHOUT any LLM call."""
    ext = extractor_with_regions  # regions align; geom declines (no ref_line_geometry)
    ext.llm_client.segment_references = AsyncMock(
        side_effect=AssertionError("LLM must not be called")
    )
    segments = await ext.refs._segment_references(REF_TEXT, "geom")
    assert len(segments) >= 3
    ext.llm_client.segment_references.assert_not_awaited()


async def test_llm_fallback_flag_off_skips_llm(extractor_no_regions, monkeypatch):
    """REF_SEG_LLM_FALLBACK=False: geom-decline goes region→CRF, never LLM."""
    ext = extractor_no_regions
    ext.refs._settings.REF_SEG_LLM_FALLBACK = False
    ext.llm_client.segment_references = AsyncMock(
        side_effect=AssertionError("LLM must not be called")
    )
    await ext.refs._segment_references(REF_TEXT, "geom")  # must not raise
    ext.llm_client.segment_references.assert_not_awaited()


async def test_region_strategy_primary(extractor_with_regions):
    """REF_SEG_STRATEGY=region segments by regions without geom or LLM."""
    ext = extractor_with_regions
    ext.llm_client.segment_references = AsyncMock(
        side_effect=AssertionError("LLM must not be called")
    )
    segments = await ext.refs._segment_references(REF_TEXT, "region")
    assert len(segments) >= 3
    ext.llm_client.segment_references.assert_not_awaited()


async def test_explicit_llm_strategy_still_llm_first(extractor_with_regions):
    """REF_SEG_STRATEGY=llm keeps LLM as the primary tier."""
    ext = extractor_with_regions
    ext.llm_client.segment_references = AsyncMock(return_value=[ANCHOR_1])
    await ext.refs._segment_references(REF_TEXT, "llm")
    ext.llm_client.segment_references.assert_awaited_once()


# ----------------------------------------------------------------------------
# Final-review finding 1/2: REF_SEG_STRATEGY=region is an explicit PRIMARY
# selection. REF_SEG_REGION_ANCHORS gates the FALLBACK TIER only (mirrors
# REF_SEG_LLM_FALLBACK vs an explicit "llm" strategy) so an explicit "region"
# selection must: (1) run even when that flag is off, (2) record a named
# decline reason (never silent) before chaining onward, and (3) never emit
# the fallback-tier recovery warning on success — that warning is reserved
# for region running as a cascade fallback tier.
# ----------------------------------------------------------------------------


async def test_region_strategy_primary_bypasses_disabled_flag(extractor_with_regions, monkeypatch):
    """Explicit region primary still runs region seg with REF_SEG_REGION_ANCHORS=false."""
    monkeypatch.setattr(Settings, "REF_SEG_REGION_ANCHORS", False)
    ext = extractor_with_regions
    ext.llm_client.segment_references = AsyncMock(
        side_effect=AssertionError("LLM must not be called")
    )
    segments = await ext.refs._segment_references(REF_TEXT, "region")
    assert len(segments) >= 3
    ext.llm_client.segment_references.assert_not_awaited()


async def test_region_strategy_primary_success_has_no_recovery_warning(extractor_with_regions):
    """Explicit region primary SUCCESS must not carry the fallback-tier recovery warning."""
    ext = extractor_with_regions
    ext.llm_client.segment_references = AsyncMock(
        side_effect=AssertionError("LLM must not be called")
    )
    segments = await ext.refs._segment_references(REF_TEXT, "region")
    assert len(segments) >= 3
    assert not any(
        w.code == WarningCode.REF_SEG_REGION_RECOVERY for w in ext.contents.processing_warnings
    )


async def test_region_strategy_primary_decline_no_regions_records_reason(extractor_no_regions):
    """No layout regions at all: primary region decline must be recorded, not silent,
    and must still chain onward (to LLM/CRF) rather than dead-ending."""
    ext = extractor_no_regions
    segments = await ext.refs._segment_references(REF_TEXT, "region")
    assert segments == ["fallback seg"]
    ext.llm_client.segment_references.assert_awaited_once()
    assert any(
        w.code == WarningCode.REF_SEG_REGION_CASCADE for w in ext.contents.processing_warnings
    )
    assert ext.refs._segmentation_attempts[0].reason_flags == ("no_summaries",)


async def test_region_strategy_primary_decline_misalignment_records_reason(monkeypatch):
    """Region summaries present but misaligned with ref_text: primary decline must be
    recorded (few anchors / misalignment), then chain onward."""
    llm = mock.Mock()
    llm.segment_references = AsyncMock(return_value=[])
    unrelated = [
        _rs("reference_content", "Brown, B. (1999). Completely unrelated. Elsewhere, 1, 1."),
        _rs("reference_content", "Green, G. (1998). Also unrelated. Nowhere, 2, 2."),
        _rs("reference_content", "White, W. (1997). Still unrelated. Anywhere, 3, 3."),
    ]
    ext = _cascade_extractor(llm, unrelated)
    fake_ner = mock.Mock()
    fake_ner.segment = mock.Mock(return_value=["fallback seg"])
    monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_a: fake_ner)

    segments = await ext.refs._segment_references(REF_TEXT, "region")

    assert segments == ["fallback seg"]
    ext.llm_client.segment_references.assert_awaited_once()
    assert any(
        w.code == WarningCode.REF_SEG_REGION_CASCADE for w in ext.contents.processing_warnings
    )
    assert ext.refs._segmentation_attempts[0].reason_flags == ("too_few_or_misaligned_anchors",)


@pytest.mark.parametrize("as_fallback", [False, True])
def test_region_error_records_attempt_in_primary_and_fallback(monkeypatch, as_fallback):
    summaries = [_rs("reference_content", ref) for ref in (_REF1, _REF2, _REF3)]
    ext = _cascade_extractor(mock.Mock(), summaries)
    monkeypatch.setattr(
        "bibr.extract.ref_extractor.segment_by_region_anchors",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("region failed")),
    )

    assert ext.refs._segment_region_anchors(REF_TEXT, as_fallback=as_fallback) is None

    assert ext.refs._segmentation_attempts[-1].strategy == "region"
    assert ext.refs._segmentation_attempts[-1].reason_flags == ("segmentation_error",)


async def test_geom_cascade_region_fallback_success_still_records_recovery_warning(
    extractor_with_regions,
):
    """Non-regression: region running as the geom-decline FALLBACK TIER (not an explicit
    primary selection) must still record the recovery warning on success."""
    ext = extractor_with_regions
    ext.llm_client.segment_references = AsyncMock(
        side_effect=AssertionError("LLM must not be called")
    )
    segments = await ext.refs._segment_references(REF_TEXT, "geom")
    assert len(segments) >= 3
    assert any(
        w.code == WarningCode.REF_SEG_REGION_RECOVERY for w in ext.contents.processing_warnings
    )


# ----------------------------------------------------------------------------
# region_chunks(): parse-sized chunking from region anchors
# ----------------------------------------------------------------------------


@pytest.fixture
def ref_text_and_summaries():
    """Aligned (ref_text, region_summaries) with >= 4 refs, matching REF_TEXT."""
    summaries = [
        _rs("reference_content", _REF1, index=0),
        _rs("reference_content", _REF2, index=1),
        _rs("reference_content", _REF3, index=2),
        _rs("reference_content", _REF4, index=3),
    ]
    return REF_TEXT, summaries


def _no_ws(s: str) -> str:
    """Whitespace-normalize for exact-content comparisons (drop all runs)."""
    return "".join(s.split())


class TestRegionChunks:
    def test_region_chunks_groups_spans(self, ref_text_and_summaries):
        ref_text, summaries = ref_text_and_summaries
        target = len(ref_text) // 2
        chunks = region_chunks(ref_text, summaries, target_chars=target)
        assert chunks is not None
        # Pin the fixture's actual current chunking: 3 chunks of [80, 140, 69]
        # chars at target_chars=len(ref_text)//2 (== 145 for this fixture).
        assert [len(c) for c in chunks] == [80, 140, 69]
        # Exact reconstruction (not a substring check): whitespace-normalized
        # chunks must equal the whitespace-normalized full segmentation —
        # this fails if the trailing chunk (or any content) is dropped.
        segments = segment_by_region_anchors(ref_text, summaries)
        assert segments is not None
        assert _no_ws("".join(chunks)) == _no_ws("".join(segments))
        # every chunk starts at a reference onset
        starts = {s[:30] for s in segments}
        assert all(any(c.startswith(st[:20]) for st in starts) for c in chunks)

    def test_region_chunks_single_chunk_when_target_large(self, ref_text_and_summaries):
        ref_text, summaries = ref_text_and_summaries
        chunks = region_chunks(ref_text, summaries, target_chars=10**6)
        assert chunks is not None
        assert len(chunks) == 1

    def test_region_chunks_declines_like_segmenter(self, ref_text_and_summaries):
        ref_text, _ = ref_text_and_summaries
        assert region_chunks(ref_text, []) is None

    def test_region_chunks_declines_when_anchors_do_not_align(self, ref_text_and_summaries):
        # mirrors TestSegmentByRegionAnchors.test_returns_none_when_anchors_do_not_align:
        # summaries from a different document must not claim a chunking either.
        ref_text, _ = ref_text_and_summaries
        other = [
            _rs("reference_content", "Brown, B. (1999). Completely unrelated. Elsewhere, 1, 1."),
            _rs("reference_content", "Green, G. (1998). Also unrelated. Nowhere, 2, 2."),
            _rs("reference_content", "White, W. (1997). Still unrelated. Anywhere, 3, 3."),
        ]
        assert region_chunks(ref_text, other) is None

    _SHORT_A = "Smith, J. (2020). Short title one. Journal A, 1, 1-5."
    _LONG_B = (
        "Doe, A., Brown, B., Green, C., White, D., Black, E. (2019). "
        "A very long reference title that goes on and on and on and on and on and on and on and on. "
        * 3
        + "Journal of Very Long Titles, 12(3), 100-200."
    )
    _SHORT_C = "Nguyen, T. (2021). Short title two. Journal C, 2, 6-10."

    @pytest.mark.parametrize(
        ("order", "expected_lens"),
        [
            pytest.param(("A", "B", "C"), [53, 497, 55], id="overlength_span_in_middle"),
            pytest.param(("B", "A", "C"), [497, 53, 55], id="overlength_span_first"),
        ],
    )
    def test_region_chunks_overlength_span_is_own_chunk(self, order, expected_lens):
        by_name = {"A": self._SHORT_A, "B": self._LONG_B, "C": self._SHORT_C}
        refs = [by_name[n] for n in order]
        ref_text = "\n".join(refs)
        summaries = [_rs("reference_content", r, index=i) for i, r in enumerate(refs)]
        # target_chars smaller than the long span alone, but big enough that
        # the two short spans (if adjacent) would otherwise be a candidate
        # to merge with each other.
        target = len(self._SHORT_A) + len(self._SHORT_C)
        assert target < len(self._LONG_B)

        chunks = region_chunks(ref_text, summaries, target_chars=target)
        assert chunks is not None
        assert [len(c) for c in chunks] == expected_lens
        # the overlength span is isolated as its own chunk, verbatim
        long_chunk = next(c for c in chunks if len(c) == len(self._LONG_B))
        assert long_chunk == self._LONG_B
        # neighbors survive intact (no drops) — exact reconstruction holds
        segments = segment_by_region_anchors(ref_text, summaries)
        assert segments is not None
        assert _no_ws("".join(chunks)) == _no_ws("".join(segments))
        assert _no_ws("".join(chunks)) == _no_ws(ref_text)
