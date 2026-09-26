import threading
from unittest.mock import AsyncMock, MagicMock, patch

from bibr.extract.ref_extractor import ReferenceExtractor
from bibr.paper_contents import PaperContents
from bibr.processing_warnings import WarningCode


def _extractor(ref_line_geometry):
    contents = PaperContents(
        sentences=[],
        sections=[],
        tables=[],
        links=[],
        sections_text={},
        ref_line_geometry=ref_line_geometry,
    )
    ex = ReferenceExtractor(contents, file_hash="h", llm_client=MagicMock())
    ex.llm_client.segment_references = AsyncMock(return_value=["Aknin, L. (2013)."])
    return ex


_GEO = [
    {
        "text": "Aknin, L. (2013).",
        "page": 5,
        "x0": 72.0,
        "y_top": 710.0,
        "x1": 180.0,
        "y_bottom": 700.0,
        "font_size": 10.0,
    }
]

# ``_segment_geom`` now derives ref strings by slicing spans out of ref_text
# (segment_spans contract), so fixtures that check exact output content need
# ref_text and spans to agree. _REF_TEXT is the concatenation of two refs;
# _SPANS_2 covers both, _SPANS_1 only the first.
_REF1 = "Aknin, L. (2013)."
_REF2 = "Carstensen, L. (2009)."
_REF_TEXT = _REF1 + _REF2
_SPANS_2 = [(0, len(_REF1)), (len(_REF1), len(_REF1) + len(_REF2))]
_SPANS_1 = [(0, len(_REF1))]


async def test_geom_high_confidence_uses_geom_no_llm():
    ex = _extractor(_GEO)
    seg = MagicMock()
    # yield 2/2 = 1.0 (healthy), confidence 0.95 (>= threshold)
    seg.segment_spans.return_value = (_SPANS_2, 0.95, 2, 2)
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch.object(ex, "_save_seg_training_data") as saver,
    ):
        out = await ex._segment_references(_REF_TEXT, "geom")
    assert out == [_REF1, _REF2]
    ex.llm_client.segment_references.assert_not_awaited()
    # Geom output is a model prediction, not an LLM label: never captured as
    # segmenter training data.
    saver.assert_not_called()


async def test_geom_absent_geometry_cascades_to_llm():
    ex = _extractor(None)
    with (
        patch("bibr.extract.ref_extractor.segment_by_anchors", return_value=["Aknin, L. (2013)."]),
        patch.object(ex, "_save_seg_training_data"),
    ):
        out = await ex._segment_references("ref blob", "geom")
    assert out == ["Aknin, L. (2013)."]
    ex.llm_client.segment_references.assert_awaited_once()
    assert any(w.code == WarningCode.REF_SEG_GEOM_CASCADE for w in ex.contents.processing_warnings)


async def test_geom_low_confidence_cascades_to_llm():
    ex = _extractor(_GEO)
    seg = MagicMock()
    # yield 1/1 = 1.0 (healthy) but confidence below threshold -> cascades on
    # the confidence check, not the yield check.
    seg.segment_spans.return_value = (_SPANS_1, 0.10, 1, 1)
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch("bibr.extract.ref_extractor.segment_by_anchors", return_value=["Aknin, L. (2013)."]),
        patch.object(ex, "_save_seg_training_data") as saver,
    ):
        ex._settings.REF_GEOM_SEG_CASCADE_THRESHOLD = 0.5
        await ex._segment_references(_REF_TEXT, "geom")
    ex.llm_client.segment_references.assert_awaited_once()
    assert any(w.code == WarningCode.REF_SEG_GEOM_CASCADE for w in ex.contents.processing_warnings)
    # The LLM tier the cascade reached is still captured, once, with its labels.
    saver.assert_called_once_with(_REF_TEXT, ["Aknin, L. (2013)."], settings=ex._settings)


async def test_geom_low_align_yield_cascades_to_llm():
    """Yield gate declines even at high confidence (see REF_GEOM_MIN_ALIGN_YIELD
    in config.py for the OOD rationale); rides the same cascade path as a
    sub-threshold confidence result."""
    ex = _extractor(_GEO)
    seg = MagicMock()
    # 120 labeled boundaries, only 10 survive alignment: yield 0.083 < 0.5.
    seg.segment_spans.return_value = ([(0, 5)] * 10, 0.95, 120, 10)
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch("bibr.extract.ref_extractor.segment_by_anchors", return_value=["Aknin, L. (2013)."]),
        patch.object(ex, "_save_seg_training_data"),
    ):
        await ex._segment_references(_REF_TEXT, "geom")
    ex.llm_client.segment_references.assert_awaited_once()
    warnings = ex.contents.processing_warnings
    assert any(w.code == WarningCode.REF_SEG_GEOM_CASCADE for w in warnings)
    assert any("labeled 120" in w.message and "aligned 10" in w.message for w in warnings)
    assert any("0.08" in w.message for w in warnings)  # 10/120 rounded


async def test_geom_healthy_align_yield_uses_geom_no_llm():
    """yield >= REF_GEOM_MIN_ALIGN_YIELD: geom result is used unchanged, no
    fallback -- the gate must not fire on healthy alignment."""
    ex = _extractor(_GEO)
    seg = MagicMock()
    seg.segment_spans.return_value = (_SPANS_2, 0.95, 2, 2)  # yield 1.0
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch.object(ex, "_save_seg_training_data"),
    ):
        out = await ex._segment_references(_REF_TEXT, "geom")
    assert out == [_REF1, _REF2]
    ex.llm_client.segment_references.assert_not_awaited()


async def test_geom_zero_labeled_cascades_to_llm():
    """labeled == 0: yield is undefined (no evidence) -- decline rather than
    divide by zero or trust an empty labeling."""
    ex = _extractor(_GEO)
    seg = MagicMock()
    seg.segment_spans.return_value = ([], 0.95, 0, 0)
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch("bibr.extract.ref_extractor.segment_by_anchors", return_value=["Aknin, L. (2013)."]),
        patch.object(ex, "_save_seg_training_data"),
    ):
        await ex._segment_references(_REF_TEXT, "geom")
    ex.llm_client.segment_references.assert_awaited_once()
    warnings = ex.contents.processing_warnings
    assert any(w.code == WarningCode.REF_SEG_GEOM_CASCADE for w in warnings)
    assert any("labeled 0" in w.message for w in warnings)
    geom_attempt = next(
        attempt for attempt in ex._segmentation_attempts if attempt.strategy == "geom"
    )
    assert geom_attempt.credible_starts is None


async def test_geom_segmenter_error_cascades_to_llm():
    ex = _extractor(_GEO)
    seg = MagicMock()
    seg.segment_spans.side_effect = RuntimeError("boom")
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch("bibr.extract.ref_extractor.segment_by_anchors", return_value=["Aknin, L. (2013)."]),
        patch.object(ex, "_save_seg_training_data"),
    ):
        await ex._segment_references(_REF_TEXT, "geom")
    ex.llm_client.segment_references.assert_awaited_once()
    assert any(w.code == WarningCode.REF_SEG_GEOM_CASCADE for w in ex.contents.processing_warnings)


async def test_geom_segment_runs_off_event_loop():
    """The GBM predict must not block the event loop: the geom segmenter's
    synchronous ``segment_spans`` compute is offloaded via ``asyncio.to_thread``,
    so it runs on a worker thread (a different ident than the loop thread), and
    the returned ref strings flow through unchanged."""
    ex = _extractor(_GEO)
    loop_ident = threading.get_ident()
    recorded: dict[str, int] = {}
    seg = MagicMock()

    def fake_segment_spans(ref_text, lines):
        recorded["ident"] = threading.get_ident()
        return (_SPANS_2, 0.95, 2, 2)

    seg.segment_spans.side_effect = fake_segment_spans
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch.object(ex, "_save_seg_training_data"),
    ):
        out = await ex._segment_references(_REF_TEXT, "geom")
    assert out == [_REF1, _REF2]
    assert recorded["ident"] != loop_ident


from bibr.paper_contents import RegionSummary


def _ref_onset_regions(n):
    """n ``reference_content`` regions whose content reads as a reference onset,
    so ``region_anchor_texts`` counts all n."""
    return [
        RegionSummary(
            page=1,
            index=i,
            label="reference_content",
            bbox=None,
            content="Smith, J. (2020). A study of things.",
        )
        for i in range(n)
    ]


async def test_geom_segment_count_below_region_ratio_cascades():
    """Confident, well-aligned geom that emits far too few spans vs the layout
    reference-onset count (arXiv/ACL under-segmentation) must decline."""
    ex = _extractor(_GEO)
    ex.contents.region_summaries = _ref_onset_regions(50)
    ref_text = "x" * 300
    spans = [(i * 5, i * 5 + 5) for i in range(28)]  # 28 < 0.6 * 50 = 30
    seg = MagicMock()
    seg.segment_spans.return_value = (spans, 0.95, 28, 28)  # confident, yield 1.0
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch("bibr.extract.ref_extractor.segment_by_anchors", return_value=["Aknin, L. (2013)."]),
        patch.object(ex, "_save_seg_training_data"),
    ):
        await ex._segment_references(ref_text, "geom")
    ex.llm_client.segment_references.assert_awaited_once()
    warnings = ex.contents.processing_warnings
    assert any(w.code == WarningCode.REF_SEG_GEOM_CASCADE for w in warnings)
    assert any("segment count" in w.message and "region onsets" in w.message for w in warnings)


async def test_geom_segment_count_within_region_ratio_kept():
    """Geom span count within GEOM_MIN_REGION_RATIO of the region-onset count is
    healthy — the gate must not fire."""
    ex = _extractor(_GEO)
    ex.contents.region_summaries = _ref_onset_regions(50)
    ref_text = "x" * 300
    spans = [(i * 5, i * 5 + 5) for i in range(48)]  # 48 >= 30
    expected = [ref_text[s:e] for s, e in spans]
    seg = MagicMock()
    seg.segment_spans.return_value = (spans, 0.95, 48, 48)
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch.object(ex, "_save_seg_training_data"),
    ):
        out = await ex._segment_references(ref_text, "geom")
    assert out == expected
    ex.llm_client.segment_references.assert_not_awaited()


async def test_geom_near_under_yield_cascades_when_region_anchors_fully_align():
    """Paper-190 shape: 38 geom spans for 47 perfectly aligned layout starts
    is an under-yield even though it clears the coarse 0.6 safety ratio."""
    ex = _extractor(_GEO)
    ref_text = "x" * 500
    spans = [(i * 10, i * 10 + 10) for i in range(38)]
    seg = MagicMock()
    seg.segment_spans.return_value = (spans, 0.95, 38, 38)
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch.object(ex, "_region_onset_count", return_value=47),
        patch.object(ex, "_aligned_region_onset_count", return_value=47),
        patch.object(ex, "_segment_fallback_chain", new=AsyncMock(return_value=["recovered"])),
    ):
        out = await ex._segment_references(ref_text, "geom")

    assert out == ["recovered"]
    attempt = ex._segmentation_attempts[-1]
    assert attempt.reason_flags == ("low_segment_count",)
    assert attempt.credible_starts == 47


async def test_geom_near_under_yield_keeps_coarse_ratio_when_region_alignment_is_partial():
    """A noisy/partially aligned region stream retains the conservative 0.6
    gate, so the stricter paper-190 rule cannot force a dubious fallback."""
    ex = _extractor(_GEO)
    ref_text = "x" * 500
    spans = [(i * 10, i * 10 + 10) for i in range(38)]
    expected = [ref_text[start:end] for start, end in spans]
    seg = MagicMock()
    seg.segment_spans.return_value = (spans, 0.95, 38, 38)
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch.object(ex, "_region_onset_count", return_value=47),
        patch.object(ex, "_aligned_region_onset_count", return_value=30),
        patch.object(ex, "_save_seg_training_data"),
    ):
        out = await ex._segment_references(ref_text, "geom")

    assert out == expected


async def test_geom_receipt_credible_starts_use_aligned_physical_offsets():
    ex = _extractor(_GEO)
    ref_text = "x" * 300
    spans = [(0, 50), (50, 100)]
    seg = MagicMock()
    # Gate behavior still uses 2/4 == 0.5; receipt denominator uses the two
    # aligned physical starts, not four raw labels.
    seg.segment_spans.return_value = (spans, 0.95, 4, 2)
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch.object(ex, "_save_seg_training_data"),
    ):
        out = await ex._segment_references(ref_text, "geom")

    assert out == [ref_text[start:end] for start, end in spans]
    assert ex._segmentation_attempts[-1].credible_starts == 2


async def test_geom_leading_orphan_span_does_not_inflate_credible_starts():
    ex = _extractor(_GEO)
    ref_text = "x" * 300
    # starts_to_spans may preserve text before the first aligned B-REF as a
    # leading orphan span. Three emitted spans still represent two aligned
    # physical starts according to the segmenter receipt.
    spans = [(0, 30), (30, 80), (80, 130)]
    seg = MagicMock()
    seg.segment_spans.return_value = (spans, 0.95, 4, 2)
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch.object(ex, "_save_seg_training_data"),
    ):
        out = await ex._segment_references(ref_text, "geom")

    assert out == [ref_text[start:end] for start, end in spans]
    assert ex._segmentation_attempts[-1].credible_starts == 2


async def test_geom_segment_count_gate_skipped_without_regions():
    """No region info -> the gate is silently skipped (a tiny geom result that
    would fail the ratio is still returned)."""
    ex = _extractor(_GEO)  # PaperContents defaults region_summaries to []
    ref_text = "x" * 300
    spans = [(i * 5, i * 5 + 5) for i in range(3)]
    expected = [ref_text[s:e] for s, e in spans]
    seg = MagicMock()
    seg.segment_spans.return_value = (spans, 0.95, 3, 3)
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch.object(ex, "_save_seg_training_data"),
    ):
        out = await ex._segment_references(ref_text, "geom")
    assert out == expected
    ex.llm_client.segment_references.assert_not_awaited()


# A located 20-entry list on pages 1-2 and a second, unrelated 20-entry list on
# pages 3-4 (a multi-article PDF, or supplementary references the locator did
# not select). Only the first list is in ref_text.
_MAIN_LIST = [
    f"Author{chr(65 + i)}, {chr(65 + i)}. ({2000 + i}). Main-list title {i}. Journal, {i}, 1-2."
    for i in range(20)
]
_SECOND_LIST = [
    f"Other{chr(65 + i)}, {chr(65 + i)}. ({2010 + i}). Supplement title {i}. Journal, {i}, 3-4."
    for i in range(20)
]


async def _extract_main_list(ref_line_geometry, geom_spans):
    import pandas as pd

    from bibr.extract.anchor_snap import starts_to_spans
    from bibr.paper import PaperReference

    ex = _extractor(ref_line_geometry)
    ex.contents.region_summaries = [
        RegionSummary(page=1 + i // 10, index=i, label="reference_content", bbox=None, content=text)
        for i, text in enumerate(_MAIN_LIST + _SECOND_LIST)
    ]
    ref_df = pd.DataFrame(
        {"text": _MAIN_LIST, "page_number": [1 + i // 10 for i in range(len(_MAIN_LIST))]}
    )
    ref_text = "\n".join(_MAIN_LIST)
    seg = MagicMock()
    spans = starts_to_spans(ref_text, [ref_text.find(ref) for ref in _MAIN_LIST])
    seg.segment_spans.return_value = (spans, 0.99, 20, 20) if geom_spans else ([], 0.0, 0, 0)

    def parse(self, segments):
        return [
            PaperReference(
                bib_id=i,
                title=segment,
                authors=None,
                year=None,
                container=None,
                volume=None,
                first_page=None,
            )
            for i, segment in enumerate(segments, start=1)
        ]

    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=seg),
        patch.object(ReferenceExtractor, "_parse_references_ner_aligned", parse),
    ):
        ex._ref_seg_strategy, ex._ref_parse_strategy = "geom", "ner"
        refs = await ex.extract(ref_df)
    attempts = [
        (attempt.strategy, attempt.selected, attempt.reason_flags)
        for attempt in ex.contents.reference_yield_receipt.attempts
    ]
    return ex, [ref.title for ref in refs], attempts


async def test_geom_segment_count_gate_counts_only_onsets_on_the_reference_pages():
    ex, titles, attempts = await _extract_main_list(_GEO, geom_spans=True)

    assert titles == _MAIN_LIST
    assert attempts == [("geom", True, ())]
    ex.llm_client.segment_references.assert_not_awaited()


async def test_reference_region_summaries_keep_regions_without_a_page():
    ex, _titles, _attempts = await _extract_main_list(_GEO, geom_spans=True)
    pageless = RegionSummary(page=None, index=99, label="reference_content", bbox=None)
    ex.contents.region_summaries = [*ex.contents.region_summaries, pageless]

    kept = ex._reference_region_summaries()

    assert [summary.content for summary in kept[:-1]] == _MAIN_LIST
    assert kept[-1] is pageless


async def test_chunked_parse_aligns_only_onsets_on_the_reference_pages():
    from bibr.extract import ref_extractor
    from bibr.schemas import PaperReferenceLLM

    ex, _titles, _attempts = await _extract_main_list(_GEO, geom_spans=True)
    ex.llm_client.extract_references_chunk = AsyncMock(
        return_value=[
            PaperReferenceLLM(
                index=1,
                title="T",
                first_page=None,
                volume=None,
                authors="A",
                year=2000,
                container=None,
            )
        ]
    )
    seen_pages: list[set] = []
    real_region_chunks = ref_extractor.region_chunks

    def spy(ref_text, summaries, **kwargs):
        seen_pages.append({summary.page for summary in summaries})
        return real_region_chunks(ref_text, summaries, **kwargs)

    with patch.object(ref_extractor, "region_chunks", side_effect=spy):
        await ex._parse_references_llm_chunked("\n".join(_MAIN_LIST), _MAIN_LIST)

    assert seen_pages == [{1, 2}]


async def test_region_tier_aligns_only_onsets_on_the_reference_pages():
    ex, titles, attempts = await _extract_main_list(None, geom_spans=False)

    assert titles == _MAIN_LIST
    assert attempts == [
        ("geom", False, ("source_geometry_unavailable",)),
        ("region", True, ()),
    ]
    ex.llm_client.segment_references.assert_not_awaited()


async def test_crf_strategy_unchanged():
    ex = _extractor(_GEO)
    seg = MagicMock()
    seg.segment.return_value = (["x"],)
    with patch("bibr.extract.ref_extractor._get_ner_segmenter") as ner:
        ner.return_value.segment.return_value = ["A. (2013).", "B. (2009)."]
        out = await ex._segment_references("ref blob", "crf")
    assert out == ["A. (2013).", "B. (2009)."]
