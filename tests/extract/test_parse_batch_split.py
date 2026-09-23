"""Batched-LLM reference parse (``--refs llm``) degenerate-failure recovery.

On small-context servers (e.g. NuExtract3 at ``max_model_len=8192``) a full
``REF_PARSE_BATCH_SIZE`` batch can overflow the window and fail with
``IncompleteOutputException``. That failure is size-induced, not a greedy
loop — so before surrendering the batch to the NER parser, the extractor
splits it and retries the halves (then quarters). Every surviving degradation
to NER is recorded on ``processing_warnings`` so "full precision" runs are
never silently downgraded in the output JSON.
"""

from unittest import mock
from unittest.mock import AsyncMock, patch

from bibr.extract.ref_extractor import (
    IncompleteOutputException,
    ReferenceExtractor,
)
from bibr.paper import PaperReference
from bibr.paper_contents import PaperContents
from bibr.processing_warnings import WarningCode
from bibr.schemas import PaperReferenceLLM

_REFS = [
    "Smith, J., & Jones, K. (2020). Attention and memory. Psych Review, 12(3), 45-67.",
    "Doe, A. (2019). Seeing things clearly. Journal of Vision, 2, 11-20.",
    "Nguyen, T. H. (2021). Replication in the wild. Meta Science, 4(1), 1-19.",
    "World Health Organization. (2018). Global report on falls. WHO Press.",
]
REF_TEXT = "\n".join(_REFS)


def _llm_ref(index: int) -> PaperReferenceLLM:
    return PaperReferenceLLM(
        index=index,
        title=f"Title {index}",
        first_page=None,
        volume=None,
        authors="Fake, A.",
        year=2020,
        container="Fake Journal",
    )


FAKE_NER_REF = PaperReference(
    bib_id=1,
    title="Ner Title",
    first_page=None,
    volume=None,
    authors="Ner, A.",
    year=2020,
    container="Ner Journal",
)


def _extractor() -> ReferenceExtractor:
    contents = mock.Mock(spec=PaperContents)
    contents.region_summaries = []
    contents.processing_warnings = []
    return ReferenceExtractor(contents, llm_client=mock.Mock())


async def test_degenerate_batch_splits_and_recovers(monkeypatch):
    """Full batch overflows → halves succeed → all refs LLM-parsed, no NER."""
    monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 4)
    ext = _extractor()

    async def fake_extract(numbered, *, file_hash, start_index, expected_count):
        if expected_count == 4:
            raise IncompleteOutputException()
        return [_llm_ref(start_index + i) for i in range(expected_count)]

    ext.llm_client.extract_references = AsyncMock(side_effect=fake_extract)
    with patch.object(
        type(ext), "_parse_references_ner_aligned", return_value=[FAKE_NER_REF]
    ) as ner:
        refs = await ext._parse_references_llm(REF_TEXT, _REFS)

    ner.assert_not_called()
    assert len(refs) == 4
    assert [r.bib_id for r in refs] == [1, 2, 3, 4]
    # 1 full-batch attempt + 2 halves
    assert ext.llm_client.extract_references.await_count == 3
    assert any(
        w.code == WarningCode.REF_PARSE_SPLIT_RECOVERY for w in ext.contents.processing_warnings
    )


async def test_split_exhausted_falls_back_to_ner_with_warning(monkeypatch):
    """Every split level still overflows → NER fallback, recorded on the paper."""
    monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 4)
    ext = _extractor()
    ext.llm_client.extract_references = AsyncMock(side_effect=IncompleteOutputException())
    with patch.object(
        type(ext), "_parse_references_ner_aligned", return_value=[FAKE_NER_REF]
    ) as ner:
        refs = await ext._parse_references_llm(REF_TEXT, _REFS)

    assert ner.called
    assert refs
    assert any(
        w.code == WarningCode.REF_PARSE_NER_FALLBACK for w in ext.contents.processing_warnings
    )


async def test_singleton_degenerate_batch_skips_split(monkeypatch):
    """A 1-ref batch can't be split — degenerate failure goes straight to NER."""
    monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 1)
    ext = _extractor()
    single = _REFS[:1]
    ext.llm_client.extract_references = AsyncMock(side_effect=IncompleteOutputException())
    with patch.object(
        type(ext), "_parse_references_ner_aligned", return_value=[FAKE_NER_REF]
    ) as ner:
        refs = await ext._parse_references_llm(single[0], single)

    assert ner.called
    assert refs
    # No split attempts: exactly the one original call.
    assert ext.llm_client.extract_references.await_count == 1
    assert any(
        w.code == WarningCode.REF_PARSE_NER_FALLBACK for w in ext.contents.processing_warnings
    )


async def test_partial_split_recovery_merges_in_order(monkeypatch):
    """One half recovers via LLM, the other exhausts to NER — refs merge in
    document order and both the recovery and the fallback are recorded."""
    monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 4)
    ext = _extractor()

    async def fake_extract(numbered, *, file_hash, start_index, expected_count):
        # Full batch and every sub-batch containing ref 3 overflow; the
        # leading half (refs 1-2) parses fine.
        if expected_count == 4 or start_index >= 3:
            raise IncompleteOutputException()
        return [_llm_ref(start_index + i) for i in range(expected_count)]

    ext.llm_client.extract_references = AsyncMock(side_effect=fake_extract)

    def fake_ner(self, segs):
        return [
            PaperReference(
                bib_id=i + 1,
                title=f"Ner {i}",
                first_page=None,
                volume=None,
                authors="Ner, A.",
                year=2020,
                container="Ner Journal",
            )
            for i in range(len(segs))
        ]

    with patch.object(
        type(ext), "_parse_references_ner_aligned", autospec=True, side_effect=fake_ner
    ):
        refs = await ext._parse_references_llm(REF_TEXT, _REFS)

    assert len(refs) == 4
    assert [r.bib_id for r in refs] == [1, 2, 3, 4]
    # LLM titles for the recovered half, NER titles for the failed half.
    assert refs[0].title == "Title 1"
    assert refs[1].title == "Title 2"
    assert refs[2].title.startswith("Ner")
    warnings = ext.contents.processing_warnings
    assert any(w.code == WarningCode.REF_PARSE_SPLIT_RECOVERY for w in warnings)
    assert any(w.code == WarningCode.REF_PARSE_NER_FALLBACK for w in warnings)


async def test_transient_failure_still_retried_not_split(monkeypatch):
    """The existing transient-retry discipline is untouched: a 5xx retries the
    identical batch once and succeeds without any split or NER involvement."""
    monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 4)
    ext = _extractor()
    ok = [_llm_ref(i + 1) for i in range(4)]
    ext.llm_client.extract_references = AsyncMock(side_effect=[RuntimeError("503"), ok])
    with patch.object(
        type(ext), "_parse_references_ner_aligned", return_value=[FAKE_NER_REF]
    ) as ner:
        refs = await ext._parse_references_llm(REF_TEXT, _REFS)

    ner.assert_not_called()
    assert len(refs) == 4
    assert ext.llm_client.extract_references.await_count == 2
    assert not ext.contents.processing_warnings
