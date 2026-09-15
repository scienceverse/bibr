"""Evidence-bearing receipt for reference segmentation and parse yield."""

from dataclasses import FrozenInstanceError, fields, is_dataclass
from unittest import mock

import pandas as pd
import pytest

from bibr import paper_contents
from bibr.extract.ref_extractor import (
    REF_PARSE_STRATEGIES,
    ReferenceExtractor,
    _sequence_references,
    _source_character_coverage,
)
from bibr.models import PaperReference
from bibr.paper_contents import PaperContents

ReferenceSegmentationAttempt = getattr(paper_contents, "ReferenceSegmentationAttempt", None)
ReferenceYieldReceipt = getattr(paper_contents, "ReferenceYieldReceipt", None)


def _contents(native: list[str] | None = None) -> mock.Mock:
    contents = mock.Mock(spec=PaperContents)
    contents.native_ref_strings = native
    contents.ref_line_geometry = None
    contents.region_summaries = []
    contents.processing_warnings = []
    contents.reference_yield_receipt = None
    contents.sections = []
    contents.sentences = []
    return contents


def _ref(index: int) -> PaperReference:
    return PaperReference(
        bib_id=index,
        title=f"Reference {index}",
        authors="Smith J",
        year=2020,
        container="Journal",
        volume=None,
        first_page=None,
    )


def test_reference_yield_receipt_has_required_frozen_contract():
    assert is_dataclass(ReferenceYieldReceipt)
    assert [field.name for field in fields(ReferenceYieldReceipt)] == [
        "credible_source_starts",
        "attempts",
        "selected_spans",
        "source_character_coverage",
        "parsed_count",
        "valid_count",
        "duplicate_rate",
        "reason_flags",
        "losses",
    ]
    receipt = ReferenceYieldReceipt(None, (), (), None, 0, 0, 0.0, ())
    with pytest.raises(FrozenInstanceError):
        receipt.parsed_count = 1  # type: ignore[misc]


async def test_native_segmentation_records_offsets_counts_and_attempt(monkeypatch):
    assert ReferenceSegmentationAttempt is not None
    assert ReferenceYieldReceipt is not None
    refs = [
        "1. Smith J. First source-backed reference. 2020. Journal of Tests 10:1-5.",
        "2. Doe A. Second source-backed reference. 2021. Journal of Tests 11:6-9.",
        "3. Roe B. Third source-backed reference. 2022. Journal of Tests 12:10-15.",
    ]
    contents = _contents(refs)
    extractor = ReferenceExtractor(contents, seg_strategy="geom", parse_strategy="ner")

    async def parse(_extractor, _ref_text, _segments):
        return [_ref(1), _ref(2), _ref(3)]

    monkeypatch.setitem(REF_PARSE_STRATEGIES, "ner", parse)
    await extractor.extract(pd.DataFrame({"text": refs}))

    receipt = contents.reference_yield_receipt
    assert isinstance(receipt, ReferenceYieldReceipt)
    assert receipt.credible_source_starts == 3
    assert receipt.parsed_count == 3
    assert receipt.valid_count == 3
    assert receipt.source_character_coverage == 1.0
    assert receipt.duplicate_rate == 0.0
    assert len(receipt.selected_spans) == 3
    assert receipt.attempts == (
        ReferenceSegmentationAttempt(
            strategy="native",
            spans=receipt.selected_spans,
            credible_starts=3,
            selected=True,
            reason_flags=(),
        ),
    )


async def test_ord62_style_under_capture_emits_nonblocking_low_yield(monkeypatch):
    # ord62 has 32 JATS references but only 6 old-output bib rows.  Five compact
    # source occurrences are enough to exercise the same sharp-yield gate.
    refs = [
        f"{i}. Author {i}. Source reference title {i}. Journal. 20{i:02d}." for i in range(1, 6)
    ]
    contents = _contents(refs)
    extractor = ReferenceExtractor(contents, seg_strategy="native", parse_strategy="ner")

    async def parse(_extractor, _ref_text, _segments):
        return [_ref(1), _ref(2)]

    monkeypatch.setitem(REF_PARSE_STRATEGIES, "ner", parse)
    await extractor.extract(pd.DataFrame({"text": refs}))

    receipt = contents.reference_yield_receipt
    assert receipt.credible_source_starts == 5
    assert receipt.valid_count == 2
    assert "credible_start_under_yield" in receipt.reason_flags
    assert [issue.code for issue in extractor.validation_issues] == ["VAL_REF_LOW_YIELD"]
    issue = extractor.validation_issues[0]
    assert issue.severity == "warning"
    assert issue.blocking is False


async def test_source_record_under_yield_is_receipted_without_other_denominator(monkeypatch):
    refs = [
        f"Author{i}, Alice. Source reference title {i}. Journal. 20{i:02d}." for i in range(1, 11)
    ]
    contents = _contents(None)
    extractor = ReferenceExtractor(contents, seg_strategy="geom", parse_strategy="ner")

    async def segment(_ref_text, _strategy):
        return refs

    async def parse(_extractor, _ref_text, _segments):
        return [_ref(index) for index in range(1, 7)]

    monkeypatch.setattr(extractor, "_segment_references", segment)
    monkeypatch.setitem(REF_PARSE_STRATEGIES, "ner", parse)

    await extractor.extract(pd.DataFrame({"text": refs}))

    receipt = contents.reference_yield_receipt
    assert receipt.credible_source_starts is None
    assert receipt.valid_count == 6
    assert "credible_start_under_yield" not in receipt.reason_flags
    assert "parse_under_yield" not in receipt.reason_flags
    assert "source_record_under_yield" in receipt.reason_flags
    assert [issue.code for issue in extractor.validation_issues] == ["VAL_REF_LOW_YIELD"]
    assert extractor.validation_issues[0].blocking is False


async def test_paper190_source_record_under_yield_is_not_just_below_threshold(monkeypatch):
    refs = [
        f"Author{i}, Alice. Source reference title {i}. Journal. 20{i % 100:02d}."
        for i in range(1, 48)
    ]
    contents = _contents(None)
    extractor = ReferenceExtractor(contents, seg_strategy="geom", parse_strategy="ner")

    async def segment(_ref_text, _strategy):
        return refs[:38]

    async def parse(_extractor, _ref_text, _segments):
        return [_ref(index) for index in range(1, 39)]

    monkeypatch.setattr(extractor, "_segment_references", segment)
    monkeypatch.setitem(REF_PARSE_STRATEGIES, "ner", parse)

    await extractor.extract(pd.DataFrame({"text": refs}))

    receipt = contents.reference_yield_receipt
    assert receipt.valid_count == 38
    assert "source_record_under_yield" in receipt.reason_flags
    assert [issue.code for issue in extractor.validation_issues] == ["VAL_REF_LOW_YIELD"]


async def test_unavailable_start_denominator_does_not_invent_low_yield(monkeypatch):
    ref_text = "Smith J. A single unnumbered source reference. Journal. 2020."
    contents = _contents(None)
    extractor = ReferenceExtractor(contents, seg_strategy="crf", parse_strategy="ner")
    monkeypatch.setattr(extractor, "_crf_segment_or_recover", lambda _text: [ref_text])

    async def parse(_extractor, _ref_text, _segments):
        return []

    monkeypatch.setitem(REF_PARSE_STRATEGIES, "ner", parse)
    await extractor.extract(pd.DataFrame({"text": [ref_text]}))

    receipt = contents.reference_yield_receipt
    assert receipt.credible_source_starts is None
    assert "credible_start_under_yield" not in receipt.reason_flags
    assert extractor.validation_issues == []


async def test_partial_source_offsets_are_retained_without_overstating_coverage(monkeypatch):
    first = "1. Smith J. First source reference. 2020. Journal of Tests 10:1-5."
    third = "3. Roe B. Third source reference. 2022. Journal of Tests 12:10-15."
    unavailable = "2. Doe A. Segment text transformed outside the source. 2021. Journal 11:6-9."
    contents = _contents(None)
    extractor = ReferenceExtractor(contents, seg_strategy="crf", parse_strategy="ner")

    async def segment(_text, _strategy):
        return [first, unavailable, third]

    seen_segments = []

    async def parse(_extractor, _ref_text, segments):
        seen_segments.extend(segments)
        return [_ref(1), _ref(2), _ref(3)]

    monkeypatch.setattr(extractor, "_segment_references", segment)
    monkeypatch.setitem(REF_PARSE_STRATEGIES, "ner", parse)
    await extractor.extract(pd.DataFrame({"text": [first, third]}))

    receipt = contents.reference_yield_receipt
    assert seen_segments == [first, unavailable, third]
    assert len(receipt.selected_spans) == 2
    assert receipt.source_character_coverage is None
    assert "source_offsets_partially_unavailable" in receipt.reason_flags


def test_known_empty_selected_spans_have_zero_source_coverage():
    assert _source_character_coverage("nonempty reference source", ()) == 0.0


def test_sequence_drops_whitespace_only_reference_without_mutating_text_fields():
    blank = PaperReference(
        bib_id=1,
        title="  \t",
        authors=" \n",
        year=None,
        container=None,
        volume=None,
        first_page=None,
    )
    kept = PaperReference(
        bib_id=9,
        title="  Grounded title  ",
        authors="  Smith J  ",
        year=2020,
        container="Journal",
        volume=None,
        first_page=None,
    )

    result = _sequence_references([blank, kept])

    assert result == [kept]
    assert result[0].bib_id == 1
    assert blank.title == "  \t"
    assert blank.authors == " \n"
    assert kept.title == "  Grounded title  "
    assert kept.authors == "  Smith J  "


def test_duplicate_rate_never_discounts_independent_credible_starts():
    contents = _contents(None)
    extractor = ReferenceExtractor(contents)
    extractor._credible_source_starts = 10
    spans = tuple((i * 10, i * 10 + 8) for i in range(5))
    refs = [_ref(i) for i in range(1, 6)]

    extractor._record_yield_receipt(
        "x" * 60,
        spans,
        parsed_refs=refs,
        duplicate_rate=1 / 6,
        duplicate_reasons=("duplicate_source_span",),
    )

    receipt = contents.reference_yield_receipt
    assert receipt.credible_source_starts == 10
    assert receipt.valid_count == 5
    assert receipt.duplicate_rate == pytest.approx(1 / 6)
    assert "credible_start_under_yield" in receipt.reason_flags
    assert [issue.code for issue in extractor.validation_issues] == ["VAL_REF_LOW_YIELD"]
    assert "of 10 credible/selected starts" in extractor.validation_issues[0].message


def test_region_receipt_uses_unique_aligned_source_offsets(monkeypatch):
    first = "Smith J. (2020). First reference. Journal 1:1-5."
    second = "Doe A. (2021). Second reference. Journal 2:6-9."
    ref_text = f"{first}\n{second}"
    contents = _contents(None)
    contents.region_summaries = [mock.Mock(), mock.Mock(), mock.Mock()]
    extractor = ReferenceExtractor(contents)
    monkeypatch.setattr(
        "bibr.extract.ref_extractor.region_anchor_texts", lambda _summaries: [first, first, second]
    )
    monkeypatch.setattr(
        "bibr.extract.ref_extractor.segment_by_region_anchors",
        lambda _text, _summaries: [first, second],
    )

    result = extractor._segment_region_anchors(ref_text, as_fallback=False)

    assert result == [first, second]
    assert extractor._segmentation_attempts[-1].credible_starts == 2


async def test_attempt_history_records_declines_before_single_selected_marker(monkeypatch):
    refs = [
        f"{i}. Author {i}. Detailed extended source reference title {i}. "
        f"20{i:02d}. Journal {i}:1-5."
        for i in range(1, 4)
    ]
    ref_text = "\n".join(refs)
    contents = _contents(None)
    extractor = ReferenceExtractor(contents, seg_strategy="geom", parse_strategy="ner")
    extractor.llm_client.segment_references = mock.AsyncMock(side_effect=RuntimeError("LLM down"))
    monkeypatch.setattr(
        "bibr.extract.ref_extractor._get_ner_segmenter",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("CRF down")),
    )

    segments = await extractor._segment_references(ref_text, "geom")

    assert segments == refs
    assert [
        (attempt.strategy, attempt.reason_flags) for attempt in extractor._segmentation_attempts
    ] == [
        ("geom", ("source_geometry_unavailable",)),
        ("region", ("no_summaries",)),
        ("llm_anchor", ("segmentation_error",)),
        ("crf", ("segmentation_error",)),
        ("marker_split", ()),
    ]
    assert sum(attempt.selected for attempt in extractor._segmentation_attempts) == 1


async def test_disabled_fallback_tiers_are_recorded_before_crf_selection(monkeypatch):
    contents = _contents(None)
    extractor = ReferenceExtractor(contents)
    extractor._settings.REF_SEG_REGION_ANCHORS = False
    extractor._settings.REF_SEG_LLM_FALLBACK = False
    fake_ner = mock.Mock()
    fake_ner.segment.return_value = ["Smith J. (2020). Reference. Journal 1:1-5."]
    monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_args: fake_ner)

    segments = await extractor._segment_fallback_chain("source reference text")

    assert segments == ["Smith J. (2020). Reference. Journal 1:1-5."]
    assert [
        (attempt.strategy, attempt.reason_flags) for attempt in extractor._segmentation_attempts
    ] == [
        ("region", ("tier_disabled",)),
        ("llm_anchor", ("tier_disabled",)),
        ("crf", ()),
    ]
    assert sum(attempt.selected for attempt in extractor._segmentation_attempts) == 1


@pytest.mark.parametrize("crf_result", [[], RuntimeError("CRF failure")])
def test_crf_rejection_is_recorded_before_marker_selection(monkeypatch, crf_result):
    refs = [
        f"{i}. Author {i}. Detailed extended source reference title {i}. "
        f"20{i:02d}. Journal {i}:1-5."
        for i in range(1, 4)
    ]
    extractor = ReferenceExtractor(_contents(None))
    fake_ner = mock.Mock()
    if isinstance(crf_result, Exception):
        fake_ner.segment.side_effect = crf_result
    else:
        fake_ner.segment.return_value = crf_result
    monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_args: fake_ner)

    assert extractor._crf_segment_or_recover("\n".join(refs)) == refs

    crf_attempt, marker_attempt = extractor._segmentation_attempts
    assert crf_attempt.strategy == "crf"
    expected = ("segmentation_error",) if isinstance(crf_result, Exception) else ("no_segments",)
    assert crf_attempt.reason_flags == expected
    assert crf_attempt.selected is False
    assert marker_attempt.strategy == "marker_split"
    assert marker_attempt.selected is True


async def test_failed_reuse_replaces_stale_receipt_with_current_failure_evidence(monkeypatch):
    ref = "1. Smith J. Source reference. 2020. Journal 1:1-5."
    contents = _contents([ref])
    extractor = ReferenceExtractor(contents, parse_strategy="ner")

    async def parse_ok(_extractor, _ref_text, _segments):
        return [_ref(1)]

    monkeypatch.setitem(REF_PARSE_STRATEGIES, "ner", parse_ok)
    await extractor.extract(pd.DataFrame({"text": [ref]}))
    successful = contents.reference_yield_receipt
    assert successful is not None
    later = "2. Doe A. A different source reference. 2021. Journal 2:6-10."
    contents.native_ref_strings = [later]

    async def parse_failure(_extractor, _ref_text, _segments):
        raise RuntimeError("parse failed")

    monkeypatch.setitem(REF_PARSE_STRATEGIES, "ner", parse_failure)
    with pytest.raises(RuntimeError, match="parse failed"):
        await extractor.extract(pd.DataFrame({"text": [later]}))

    failed = contents.reference_yield_receipt
    assert failed is not successful
    assert failed.parsed_count == failed.valid_count == 0
    assert len(failed.losses.unresolved) == 1
    assert failed.losses.unresolved[0].reason == "parser_failed"
    assert failed.losses.unresolved[0].source_text == later
    assert failed.selected_spans == ((0, len(later)),)

    # An error before source segmentation has no new parse evidence to assert.
    monkeypatch.setattr(
        extractor, "_segment_references", mock.AsyncMock(side_effect=RuntimeError("segment failed"))
    )
    with pytest.raises(RuntimeError, match="segment failed"):
        await extractor.extract(pd.DataFrame({"text": [ref]}))

    assert contents.reference_yield_receipt is None
