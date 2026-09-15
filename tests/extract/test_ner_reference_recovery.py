"""Optional recovery preserves local successes and accounts for every missing slot."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from bibr.config import snapshot_settings
from bibr.exceptions import UpstreamServiceError
from bibr.extract.ref_extractor import ReferenceExtractor
from bibr.paper_contents import PaperContents
from bibr.schemas import PaperReferenceLLM
from tests.extract.test_reference_losses import FIRST, SECOND

THIRD = "Green B. Third reference title. Journal of Testing. 2022;3:11-15."


def test_numeric_recovery_fields_require_whole_printed_numbers():
    from bibr.extract.reference_recovery import ungrounded_fields

    assert ungrounded_fields({"year": 202, "volume": "2", "first_page": "1"}, "2021;32:10-15") == (
        "year",
        "volume",
        "first_page",
    )
    assert (
        ungrounded_fields({"year": 2021, "volume": "32", "last_page": "15"}, "2021;32:10-15") == ()
    )


def _ref(title="Second reference title", authors="Doe A", **kwargs):
    fields = {
        "index": 1,
        "title": title,
        "authors": authors,
        "year": None,
        "container": None,
        "volume": None,
        "first_page": None,
    }
    return PaperReferenceLLM(**(fields | kwargs))


def _extractor(monkeypatch, *, budget=2, result=None, timeout=60):
    settings = snapshot_settings()
    settings.REF_NER_RECOVERY_MAX_SEGMENTS = budget
    settings.REF_NER_RECOVERY_TIMEOUT = timeout
    client = Mock(extract_references=AsyncMock(return_value=result or [_ref()]))
    contents = PaperContents([], [], [], [], {})
    ext = ReferenceExtractor(contents, llm_client=client, parse_strategy="ner", settings=settings)
    monkeypatch.setattr(ext, "_segment_references", AsyncMock(return_value=[FIRST, SECOND, THIRD]))
    first = _ref("First reference title", "Smith J")
    last = _ref("Third reference title", "Green B")
    monkeypatch.setattr(
        ext, "_parse_references_ner_aligned", Mock(return_value=[first, None, last])
    )
    frame = pd.DataFrame({"text": [FIRST, SECOND, THIRD], "text_id": [10, 20, 30]})
    return ext, client, frame, first, last


async def test_recovers_only_missing_slot_preserving_successful_objects_and_source_order(
    monkeypatch,
):
    ext, client, frame, first, last = _extractor(monkeypatch)
    result = await ext.extract(frame)
    assert result[0] is first and result[2] is last
    assert [ref.title for ref in result] == [
        "First reference title",
        "Second reference title",
        "Third reference title",
    ]
    assert [ref.bib_id for ref in result] == [1, 2, 3]
    client.extract_references.assert_awaited_once_with(
        f"1. {SECOND}", file_hash="unknown", start_index=1, expected_count=1
    )
    receipt = ext.contents.reference_yield_receipt
    assert receipt.losses.unresolved == ()
    assert receipt.recovery.recovered_count == 1
    attempt = receipt.recovery.attempts[0]
    assert (attempt.segment_index, attempt.outcome) == (1, "recovered")
    assert attempt.source_span == (len(FIRST) + 1, len(FIRST) + 1 + len(SECOND))


async def test_disabled_recovery_keeps_ner_local(monkeypatch):
    ext, client, frame, first, last = _extractor(monkeypatch, budget=0)
    assert await ext.extract(frame) == [first, last]
    client.extract_references.assert_not_awaited()
    assert ext.contents.reference_yield_receipt.recovery is None


@pytest.mark.parametrize("fault", ["duplicate", "index", "untrusted", "title", "year", "stub"])
async def test_unsupported_or_ambiguous_responses_leave_original_loss_visible(monkeypatch, fault):
    ref = _ref()
    if fault == "index":
        ref.index = 2
    elif fault == "untrusted":
        ref.mark_index_untrusted()
    elif fault == "title":
        ref.title = "A plausible but unprinted title"
    elif fault == "year":
        ref.year = 1998
    elif fault == "stub":
        ref.title = ref.authors = None
    ext, _, frame, first, last = _extractor(
        monkeypatch, result=[ref, ref] if fault == "duplicate" else [ref]
    )
    assert await ext.extract(frame) == [first, last]
    receipt = ext.contents.reference_yield_receipt
    assert receipt.recovery.recovered_count == 0
    assert receipt.losses.unresolved[0].source_text_ids == (20,)


async def test_segment_budget_leaves_unattempted_entries_in_denominator(monkeypatch):
    ext, client, frame, _, last = _extractor(monkeypatch, budget=1)
    ext._parse_references_ner_aligned.return_value = [None, None, last]
    client.extract_references.return_value = [_ref("First reference title", "Smith J")]
    result = await ext.extract(frame)
    assert len(result) == 2
    assert client.extract_references.await_count == 1
    receipt = ext.contents.reference_yield_receipt
    assert receipt.recovery.stop_reason == "segment_budget"
    assert receipt.losses.unresolved[0].source_text_ids == (20,)


async def test_total_timeout_preserves_local_results(monkeypatch):
    ext, client, frame, first, last = _extractor(monkeypatch, timeout=0.01)

    async def slow(*_args, **_kwargs):
        await asyncio.sleep(10)

    client.extract_references.side_effect = slow
    assert await ext.extract(frame) == [first, last]
    receipt = ext.contents.reference_yield_receipt
    assert receipt.recovery.stop_reason == "timeout"
    assert receipt.recovery.attempts[0].outcome == "timeout"
    assert len(receipt.losses.unresolved) == 1


async def test_cancellation_propagates(monkeypatch):
    ext, client, frame, _, _ = _extractor(monkeypatch)
    client.extract_references.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await ext.extract(frame)


async def test_optional_upstream_failure_stops_recovery_without_erasing_local_success(monkeypatch):
    ext, client, frame, first, last = _extractor(monkeypatch)
    client.extract_references.side_effect = UpstreamServiceError("LLM", "unavailable")
    assert await ext.extract(frame) == [first, last]
    assert ext.contents.reference_yield_receipt.recovery.stop_reason == "upstream_unavailable"


async def test_rewritten_segment_is_not_sent_as_source_evidence(monkeypatch):
    ext, client, frame, first, last = _extractor(monkeypatch)
    ext._segment_references.return_value = [FIRST, SECOND.replace("Second", "Invented"), THIRD]
    assert await ext.extract(frame) == [first, last]
    client.extract_references.assert_not_awaited()
    assert ext.contents.reference_yield_receipt.losses.unlocated_unresolved_count == 1


async def test_recovery_accepts_printed_non_latin_fields(monkeypatch):
    ext, client, frame, _, _ = _extractor(monkeypatch)
    source = "山田太郎. 植物の成長と光. 園芸研究. 2021;2:6-10."
    frame.loc[1, "text"] = source
    ext._segment_references.return_value = [FIRST, source, THIRD]
    client.extract_references.return_value = [_ref("植物の成長と光", "山田太郎", year=2021)]
    result = await ext.extract(frame)
    assert result[1].title == "植物の成長と光"
    assert ext.contents.reference_yield_receipt.recovery.recovered_count == 1
