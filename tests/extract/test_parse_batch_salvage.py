"""Batched-LLM reference parse (``--refs llm``) truncated-completion salvage.

When a batch runs to its output-token cap, its raw completion still carries the
complete leading reference objects. The extractor salvages that positionally
aligned prefix and routes only the un-parsed tail to the split / NER fallback,
recording the salvage on ``processing_warnings`` so a downgraded batch is never
silent in the output JSON.
"""

import json
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock, patch

from bibr.exceptions import UpstreamServiceError
from bibr.extract.ref_extractor import (
    PARSE_FALLBACK_WARNING_PREFIX,
    PARSE_SALVAGE_RECOVERY_PREFIX,
    PARSE_SPLIT_RECOVERY_PREFIX,
    IncompleteOutputException,
    ReferenceExtractor,
)
from bibr.paper import PaperReference
from bibr.paper_contents import PaperContents
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


def _salv_obj(index: int) -> dict:
    return {
        "index": index,
        "title": f"Salv {index}",
        "authors": "Salv, A.",
        "year": 2020,
        "container": "Salv Journal",
        "first_page": None,
        "volume": None,
    }


def _truncated_completion(*complete_indices: int) -> SimpleNamespace:
    """An OpenAI-shaped completion whose references array is cut off mid-object
    after the given complete objects."""
    body = ", ".join(json.dumps(_salv_obj(i)) for i in complete_indices)
    content = f'{{"references": [{body}, {{"index": 99, "title": "Trunc'
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _degenerate(*complete_indices: int) -> UpstreamServiceError:
    """The wrapped exception the LLM client raises on an output-cap truncation,
    carrying the partial completion the same way production does."""
    inner = IncompleteOutputException(last_completion=_truncated_completion(*complete_indices))
    return UpstreamServiceError("LLM", "Failed to extract references", inner)


def _extractor() -> ReferenceExtractor:
    contents = mock.Mock(spec=PaperContents)
    contents.region_summaries = []
    contents.processing_warnings = []
    return ReferenceExtractor(contents, llm_client=mock.Mock())


async def test_salvaged_prefix_kept_tail_reparsed(monkeypatch):
    """Full batch truncates after refs 1-2 → those are salvaged, the tail (3-4)
    is re-parsed in smaller batches, no NER, salvage + split both recorded."""
    monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 4)
    ext = _extractor()

    async def fake_extract(numbered, *, file_hash, start_index, expected_count):
        if expected_count == 4:
            raise _degenerate(1, 2)
        return [_llm_ref(start_index + i) for i in range(expected_count)]

    ext.llm_client.extract_references = AsyncMock(side_effect=fake_extract)
    with patch.object(type(ext), "_parse_references_ner_aligned", return_value=[]) as ner:
        refs = await ext._parse_references_llm(REF_TEXT, _REFS)

    ner.assert_not_called()
    assert [r.bib_id for r in refs] == [1, 2, 3, 4]
    # Salvaged refs keep the truncated completion's content; the tail is LLM-reparsed.
    assert refs[0].title == "Salv 1"
    assert refs[1].title == "Salv 2"
    assert refs[2].title == "Title 3"
    assert refs[3].title == "Title 4"
    warnings = ext.contents.processing_warnings
    assert any(w.startswith(PARSE_SALVAGE_RECOVERY_PREFIX) for w in warnings)
    assert any(w.startswith(PARSE_SPLIT_RECOVERY_PREFIX) for w in warnings)


async def test_salvaged_prefix_kept_tail_falls_back_to_ner(monkeypatch):
    """Truncation salvages refs 1-2; the tail keeps overflowing every split
    level and surrenders to NER — prefix kept, both events recorded."""
    monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 4)
    ext = _extractor()

    ext.llm_client.extract_references = AsyncMock(side_effect=_degenerate(1, 2))

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

    assert [r.bib_id for r in refs] == [1, 2, 3, 4]
    assert refs[0].title == "Salv 1"
    assert refs[1].title == "Salv 2"
    assert refs[2].title.startswith("Ner")
    assert refs[3].title.startswith("Ner")
    warnings = ext.contents.processing_warnings
    assert any(w.startswith(PARSE_SALVAGE_RECOVERY_PREFIX) for w in warnings)
    assert any(w.startswith(PARSE_FALLBACK_WARNING_PREFIX) for w in warnings)


async def test_full_salvage_skips_fallback_entirely(monkeypatch):
    """When the truncation happens after the last complete object, salvage
    covers the whole batch and no retry / NER runs at all."""
    monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 4)
    ext = _extractor()

    ext.llm_client.extract_references = AsyncMock(side_effect=_degenerate(1, 2, 3, 4))
    with patch.object(type(ext), "_parse_references_ner_aligned", return_value=[]) as ner:
        refs = await ext._parse_references_llm(REF_TEXT, _REFS)

    ner.assert_not_called()
    # Exactly the one failed full-batch call — no split, no re-parse.
    assert ext.llm_client.extract_references.await_count == 1
    assert [r.title for r in refs] == ["Salv 1", "Salv 2", "Salv 3", "Salv 4"]
    assert any(
        w.startswith(PARSE_SALVAGE_RECOVERY_PREFIX) for w in ext.contents.processing_warnings
    )


async def test_misaligned_indices_decline_salvage(monkeypatch):
    """A completion whose leading object is NOT at the batch's start index is
    not trustworthy for positional mapping → no salvage, the existing split
    path handles the whole batch."""
    monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 4)
    ext = _extractor()

    async def fake_extract(numbered, *, file_hash, start_index, expected_count):
        if expected_count == 4:
            # Leading object claims index 2, breaking the offset+j contract at j=0.
            raise _degenerate(2, 3)
        return [_llm_ref(start_index + i) for i in range(expected_count)]

    ext.llm_client.extract_references = AsyncMock(side_effect=fake_extract)
    with patch.object(type(ext), "_parse_references_ner_aligned", return_value=[]) as ner:
        refs = await ext._parse_references_llm(REF_TEXT, _REFS)

    ner.assert_not_called()
    # No salvage recorded; the full batch is recovered by the split path instead.
    assert not any(
        w.startswith(PARSE_SALVAGE_RECOVERY_PREFIX) for w in ext.contents.processing_warnings
    )
    assert [r.title for r in refs] == ["Title 1", "Title 2", "Title 3", "Title 4"]
