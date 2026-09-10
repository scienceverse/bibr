"""``llm-chunked`` reference parse strategy — chunk-tolerant LLM parse over
region-aligned chunks of the reference list. The model finds its own
reference boundaries (no numbered-list contract), so upstream
under/over-segmentation is non-critical on this path.

Failure discipline mirrors the batched-LLM parse path
(``_parse_references_llm``): a degenerate failure (timeout / output-cap
loop) skips the retry and falls straight to the NER parser on that chunk's
members; a transient failure is retried once before falling back.
"""

import asyncio
from unittest import mock
from unittest.mock import AsyncMock, patch

import pytest

from bibr.exceptions import UpstreamServiceError
from bibr.extract.ref_extractor import (
    REF_PARSE_STRATEGIES,
    ReferenceExtractor,
    _chunk_ner_members,
)
from bibr.paper import PaperReference
from bibr.paper_contents import PaperContents, RegionSummary
from bibr.schemas import PaperReferenceLLM

_REF1 = "Smith, J., & Jones, K. (2020). Attention and memory. Psych Review, 12(3), 45-67."
_REF2 = "Doe, A. (2019). Seeing things clearly. Journal of Vision, 2, 11-20."
_REF3 = "Nguyen, T. H. (2021). Replication in the wild. Meta Science, 4(1), 1-19."
_REF4 = "World Health Organization. (2018). Global report on falls. WHO Press."

SEGMENTS = [_REF1, _REF2, _REF3, _REF4]
REF_TEXT = "\n".join(SEGMENTS)

FAKE_LLM_REF = PaperReferenceLLM(
    index=1,
    title="Fake Title",
    first_page=None,
    volume=None,
    authors="Fake, A.",
    year=2020,
    container="Fake Journal",
)

FAKE_REF = PaperReference(
    bib_id=1,
    title="Fake Title",
    first_page=None,
    volume=None,
    authors="Fake, A.",
    year=2020,
    container="Fake Journal",
)


def _rs(label: str, content: str, page: int = 9, index: int = 0) -> RegionSummary:
    return RegionSummary(page=page, index=index, label=label, bbox=None, content=content)


def _extractor(llm_client, summaries) -> ReferenceExtractor:
    contents = mock.Mock(spec=PaperContents)
    contents.region_summaries = summaries
    contents.processing_warnings = []
    return ReferenceExtractor(contents, llm_client=llm_client)


@pytest.fixture
def extractor_with_regions():
    """Layout regions align with REF_TEXT/SEGMENTS — region_chunks() claims it."""
    llm = mock.Mock()
    summaries = [_rs("reference_content", r) for r in SEGMENTS]
    return _extractor(llm, summaries)


@pytest.fixture
def extractor_no_regions():
    """No usable regions (DOCX / non-native) — falls back to segment grouping."""
    llm = mock.Mock()
    return _extractor(llm, [])


async def test_chunked_parse_uses_region_chunks(extractor_with_regions):
    ext = extractor_with_regions
    ext.llm_client.extract_references_chunk = AsyncMock(return_value=[FAKE_LLM_REF])
    refs = await ext._parse_references_llm_chunked(REF_TEXT, SEGMENTS)
    assert ext.llm_client.extract_references_chunk.await_count >= 1
    assert [r.bib_id for r in refs] == list(range(1, len(refs) + 1))  # contiguous


async def test_chunked_parse_falls_back_to_segment_grouping(extractor_no_regions):
    """No usable regions → chunks built by grouping the segmented refs."""
    ext = extractor_no_regions
    ext.llm_client.extract_references_chunk = AsyncMock(return_value=[FAKE_LLM_REF])
    refs = await ext._parse_references_llm_chunked(REF_TEXT, SEGMENTS)
    assert refs
    sent = ext.llm_client.extract_references_chunk.await_args_list[0].args[0]
    assert SEGMENTS[0] in sent


async def test_chunked_parse_chunk_failure_falls_back_to_ner(extractor_with_regions):
    ext = extractor_with_regions
    ext.llm_client.extract_references_chunk = AsyncMock(side_effect=TimeoutError)
    with patch.object(type(ext), "_parse_references_ner", return_value=[FAKE_REF]) as ner:
        refs = await ext._parse_references_llm_chunked(REF_TEXT, SEGMENTS)
    assert ner.called and refs


async def test_chunked_parse_transient_failure_retries_once(extractor_with_regions):
    """A non-degenerate failure (e.g. a 5xx) is retried once before any fallback."""
    ext = extractor_with_regions
    ext.llm_client.extract_references_chunk = AsyncMock(
        side_effect=[RuntimeError("503"), [FAKE_LLM_REF]]
    )
    refs = await ext._parse_references_llm_chunked(REF_TEXT, SEGMENTS)
    assert ext.llm_client.extract_references_chunk.await_count == 2
    assert refs


async def test_chunked_parse_empty_segments_returns_empty(extractor_no_regions):
    ext = extractor_no_regions
    ext.llm_client.extract_references_chunk = AsyncMock(return_value=[FAKE_LLM_REF])
    refs = await ext._parse_references_llm_chunked("", [])
    assert refs == []
    ext.llm_client.extract_references_chunk.assert_not_awaited()


def test_registry_has_llm_chunked():
    assert "llm-chunked" in REF_PARSE_STRATEGIES


async def test_chunked_parse_cancelled_during_retry_propagates(extractor_with_regions):
    """Cancellation is not a parse failure. If the retry await is cancelled
    (request timeout / shutdown), it must propagate — never be swallowed into
    an NER fallback that lets the cancelled coroutine keep running."""
    ext = extractor_with_regions
    # First call: transient error → triggers the retry. Retry: cancelled.
    ext.llm_client.extract_references_chunk = AsyncMock(
        side_effect=[RuntimeError("503"), asyncio.CancelledError()]
    )
    with patch.object(type(ext), "_parse_references_ner", return_value=[FAKE_REF]) as ner:
        with pytest.raises(asyncio.CancelledError):
            await ext._parse_references_llm_chunked(REF_TEXT, SEGMENTS)
    ner.assert_not_called()


async def test_batched_parse_cancelled_during_retry_propagates(extractor_with_regions, monkeypatch):
    """Same cancellation discipline on the batched-LLM parse path."""
    ext = extractor_with_regions
    # One batch so the retry await is deterministic (single gather task).
    monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 50)
    ext.llm_client.extract_references = AsyncMock(
        side_effect=[RuntimeError("503"), asyncio.CancelledError()]
    )
    with patch.object(type(ext), "_parse_references_ner", return_value=[FAKE_REF]) as ner:
        with pytest.raises(asyncio.CancelledError):
            await ext._parse_references_llm(REF_TEXT, SEGMENTS)
    ner.assert_not_called()


class TestChunkNerMembers:
    """NER-fallback member selection for one region chunk — must not silently
    drop a reference that straddles a region-chunk boundary."""

    def test_clean_tiling_returns_segments_unchanged(self):
        seg = ["Smith 2020. Title A.", "Doe 2019. Title B.", "Ng 2021. Title C."]
        chunk = "\n".join(seg)
        assert _chunk_ner_members(chunk, seg) == seg

    def test_boundary_straddled_segment_is_recovered_not_dropped(self):
        # The chunk ends mid-reference: only "Ng 2021." of the third ref lands
        # in this chunk (its tail is in the next chunk). The full third segment
        # is therefore not a substring of the chunk.
        seg = ["Smith 2020. Title A.", "Doe 2019. Title B.", "Ng 2021. Title C."]
        chunk = "Smith 2020. Title A.\nDoe 2019. Title B.\nNg 2021."
        members = _chunk_ner_members(chunk, seg)
        assert "Smith 2020. Title A." in members
        assert "Doe 2019. Title B." in members
        # The straddled head is recovered rather than lost entirely.
        assert "Ng 2021." in members
        assert all(m != "Ng 2021. Title C." for m in members)

    def test_no_match_numbered_chunk_marker_splits(self):
        chunk = "[1] Alpha ref.\n[2] Beta ref.\n[3] Gamma ref."
        members = _chunk_ner_members(chunk, ["totally unrelated segment text here"])
        assert members == ["[1] Alpha ref.", "[2] Beta ref.", "[3] Gamma ref."]

    def test_no_match_unnumbered_chunk_falls_back_to_whole_chunk(self):
        chunk = "Prose without any list markers at all."
        assert _chunk_ner_members(chunk, ["unrelated"]) == [chunk]


async def test_chunked_parse_preserves_chunk_order_despite_completion_inversion(
    extractor_no_regions, monkeypatch
):
    """Chunk 0's LLM call is the slower of the two (completes AFTER chunk 1's),
    but the assembled refs must still follow document (chunk) order — not
    completion order — with bib_ids contiguous across chunks."""
    ext = extractor_no_regions
    # Shrink the fallback chunk-grouping target so SEGMENTS split into exactly
    # two chunks: [REF1, REF2] and [REF3, REF4].
    monkeypatch.setattr("bibr.extract.ref_extractor._CHUNK_TARGET_CHARS", 150)

    chunk0_refs = [
        PaperReferenceLLM(
            index=1,
            title="Chunk0-Ref1",
            first_page=None,
            volume=None,
            authors="A, A.",
            year=2020,
            container="J0",
        ),
        PaperReferenceLLM(
            index=2,
            title="Chunk0-Ref2",
            first_page=None,
            volume=None,
            authors="B, B.",
            year=2020,
            container="J0",
        ),
    ]
    chunk1_refs = [
        PaperReferenceLLM(
            index=1,
            title="Chunk1-Ref1",
            first_page=None,
            volume=None,
            authors="C, C.",
            year=2021,
            container="J1",
        ),
        PaperReferenceLLM(
            index=2,
            title="Chunk1-Ref2",
            first_page=None,
            volume=None,
            authors="D, D.",
            year=2021,
            container="J1",
        ),
    ]

    async def fake_extract(chunk, file_hash=None):
        if _REF1 in chunk:
            # First chunk deliberately completes LAST.
            await asyncio.sleep(0.05)
            return chunk0_refs
        return chunk1_refs

    ext.llm_client.extract_references_chunk = AsyncMock(side_effect=fake_extract)
    refs = await ext._parse_references_llm_chunked(REF_TEXT, SEGMENTS)

    assert [r.title for r in refs] == [
        "Chunk0-Ref1",
        "Chunk0-Ref2",
        "Chunk1-Ref1",
        "Chunk1-Ref2",
    ]
    assert [r.bib_id for r in refs] == [1, 2, 3, 4]


async def test_chunked_parse_all_chunks_fail_raises_upstream_error(
    extractor_with_regions, monkeypatch
):
    """Every chunk's LLM call fails AND the NER fallback also raises — the
    strategy must fail loud with UpstreamServiceError, never silently
    return []."""
    ext = extractor_with_regions
    ext.llm_client.extract_references_chunk = AsyncMock(side_effect=TimeoutError("stuck"))

    with (
        patch.object(type(ext), "_parse_references_ner", side_effect=RuntimeError("ner exploded")),
        pytest.raises(UpstreamServiceError, match="reference parse chunks failed"),
    ):
        await ext._parse_references_llm_chunked(REF_TEXT, SEGMENTS)
