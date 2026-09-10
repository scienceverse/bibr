"""Tests for LLMClient.extract_references batch index handling.

The caller numbers each batch entry from ``start_index`` and later uses
``ref.index`` to look up the original segment string, so a ref dropped
mid-batch must not shift the indices of the refs after it.
"""

import logging
from unittest import mock

import pytest

from bibr.clients.llm import LLMClient
from bibr.schemas import PaperReferenceList, PaperReferenceLLM


@pytest.fixture
def client():
    c = LLMClient()
    c._limiter = mock.Mock()
    c._limiter.acquire = mock.AsyncMock()
    return c


def _llm_ref(idx, title):
    return PaperReferenceLLM(
        index=idx,
        title=title,
        authors="Author",
        first_page=None,
        last_page=None,
        volume=None,
        issue=None,
        year=2020,
        container=None,
        doi=None,
    )


async def _extract(client, refs, text, start_index):
    fake = PaperReferenceList(references=refs)
    with mock.patch.object(client, "_invoke_structured", new=mock.AsyncMock(return_value=fake)):
        return await client.extract_references(text, start_index=start_index)


async def test_dropped_mid_batch_ref_keeps_reported_indices(client):
    text = "1. Ref A text\n2. Ref B text\n3. Ref C text"
    refs = await _extract(client, [_llm_ref(1, "A"), _llm_ref(3, "C")], text, 1)
    assert [r.index for r in refs] == [1, 3]
    assert [r.title for r in refs] == ["A", "C"]


async def test_dropped_ref_logs_warning(client, caplog):
    text = "1. Ref A text\n2. Ref B text\n3. Ref C text"
    with caplog.at_level(logging.WARNING, logger="bibr.clients.llm"):
        await _extract(client, [_llm_ref(1, "A"), _llm_ref(3, "C")], text, 1)
    assert any("2/3" in r.message for r in caplog.records)


async def test_full_batch_valid_indices_preserved(client):
    text = "6. Ref A text\n7. Ref B text"
    refs = await _extract(client, [_llm_ref(6, "A"), _llm_ref(7, "B")], text, 6)
    assert [r.index for r in refs] == [6, 7]


async def test_trusted_indices_returned_in_text_order(client):
    text = "1. Ref A text\n2. Ref B text\n3. Ref C text"
    refs = await _extract(client, [_llm_ref(3, "C"), _llm_ref(1, "A")], text, 1)
    assert [r.index for r in refs] == [1, 3]
    assert [r.title for r in refs] == ["A", "C"]


async def test_duplicate_indices_fall_back_to_positional(client):
    text = "1. Ref A text\n2. Ref B text\n3. Ref C text"
    refs = await _extract(client, [_llm_ref(2, "A"), _llm_ref(2, "B")], text, 1)
    assert [r.index for r in refs] == [1, 2]


async def test_out_of_range_indices_fall_back_to_positional(client):
    text = "6. Ref A text\n7. Ref B text"
    refs = await _extract(client, [_llm_ref(1, "A"), _llm_ref(2, "B")], text, 6)
    assert [r.index for r in refs] == [6, 7]


async def test_unnumbered_text_falls_back_to_positional(client):
    text = "Smith J. Ref A text\nDoe A. Ref B text"
    refs = await _extract(client, [_llm_ref(1, "A"), _llm_ref(2, "B")], text, 4)
    assert [r.index for r in refs] == [4, 5]


# ---------------------------------------------------------------------------
# Duplicate indices with a matching count (round-3 audit)
# ---------------------------------------------------------------------------


async def test_duplicate_index_with_matching_count_is_marked_untrusted(client):
    """A repeat means the model was not tracking entries one-to-one.

    The count check alone cannot see it: three rows for three entries labelled
    1, 2, 2 passes ``len(refs) == len(expected)`` while entry 3 is missing and
    every position after the repeat is off by one. Leaving these trusted let a
    segment-anchored backfill stamp the neighbouring entry's printed DOI onto
    the wrong reference.
    """
    text = "1. Ref A text\n2. Ref B text\n3. Ref C text"
    refs = await _extract(client, [_llm_ref(1, "A"), _llm_ref(2, "B"), _llm_ref(2, "C")], text, 1)
    assert [r.index for r in refs] == [1, 2, 3]
    assert [r.index_trusted for r in refs] == [False, False, False]


async def test_duplicate_index_with_mismatched_count_is_still_untrusted(client):
    text = "1. Ref A text\n2. Ref B text\n3. Ref C text"
    refs = await _extract(client, [_llm_ref(2, "A"), _llm_ref(2, "B")], text, 1)
    assert all(not r.index_trusted for r in refs)


async def test_a_pure_renumbering_stays_trusted(client):
    """Batch numbered 1..n instead of from start_index.

    Positional re-indexing fixes that exactly, so turning the backfills off
    would cost enrichment for nothing.
    """
    text = "6. Ref A text\n7. Ref B text"
    refs = await _extract(client, [_llm_ref(1, "A"), _llm_ref(2, "B")], text, 6)
    assert [r.index for r in refs] == [6, 7]
    assert all(r.index_trusted for r in refs)


async def test_a_complete_in_order_batch_stays_trusted(client):
    text = "1. Ref A text\n2. Ref B text\n3. Ref C text"
    refs = await _extract(client, [_llm_ref(1, "A"), _llm_ref(2, "B"), _llm_ref(3, "C")], text, 1)
    assert all(r.index_trusted for r in refs)
