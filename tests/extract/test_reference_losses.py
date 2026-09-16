"""Loss evidence remains source-backed and outside successfully parsed records."""

from dataclasses import asdict, replace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from bibr.export.models import ReferenceYieldLossesExport
from bibr.extract.ref_extractor import ReferenceExtractor
from bibr.extract.reference_losses import reference_losses
from bibr.models import PaperReference
from bibr.paper_contents import PaperContents
from bibr.schemas import PaperReferenceLLM

FIRST = "Smith J. First reference title. Journal of Testing. 2020;1:1-5."
SECOND = "Doe A. Second reference title. Journal of Testing. 2021;2:6-10."


def test_duplicate_text_retains_the_unresolved_physical_occurrence():
    losses = reference_losses(
        [(FIRST, 11), (FIRST, 12)],
        [FIRST, FIRST],
        [FIRST, FIRST],
        [FIRST, FIRST],
        {1: "no_title_or_authors"},
        parse_alignment_available=True,
    )
    assert len(losses.unresolved) == 1
    row = losses.unresolved[0]
    assert row.source_text_ids == (12,)
    assert row.source_span == (len(FIRST) + 1, 2 * len(FIRST) + 1)


def test_segmentation_and_filtering_losses_are_distinct_from_parse_failure():
    source = [
        (FIRST, 1),
        ("Unsegmented source evidence.", 2),
        ("Supplemental material", 3),
        (SECOND, 4),
    ]
    losses = reference_losses(
        source,
        [FIRST, "Supplemental material", SECOND],
        [FIRST, SECOND],
        [FIRST, SECOND],
        {1: "no_title_or_authors"},
        parse_alignment_available=True,
    )
    assert (
        losses.selected_source_row_count,
        losses.segmented_count,
        losses.retained_segment_count,
        losses.filtered_segment_count,
    ) == (4, 3, 2, 1)
    assert [(row.stage, row.source_text_ids) for row in losses.unresolved] == [
        ("segmentation", (2,)),
        ("filtering", (3,)),
        ("parsing", (4,)),
    ]
    assert losses.unresolved[-1].source_text == SECOND
    assert ReferenceYieldLossesExport.model_validate(asdict(losses)).unresolved[
        -1
    ].source_text_ids == [4]


def test_rewritten_model_text_is_never_claimed_as_source_evidence():
    losses = reference_losses(
        [(FIRST, 1)],
        ["A model rewrite"],
        ["A model rewrite"],
        ["A model rewrite"],
        {0: "no_title_or_authors"},
        parse_alignment_available=False,
    )
    assert losses.unresolved == ()
    assert not losses.parse_alignment_available
    assert not losses.source_alignment_available
    assert losses.unlocated_unresolved_count == 1


async def test_ner_none_preserves_source_without_increasing_parsed_count(monkeypatch):
    contents = PaperContents([], [], [], [], {})
    extractor = ReferenceExtractor(
        contents, llm_client=object(), seg_strategy="native", parse_strategy="ner"
    )
    monkeypatch.setattr(extractor, "_segment_references", AsyncMock(return_value=[FIRST, SECOND]))
    good = PaperReference(
        bib_id=1,
        title="First reference title",
        authors="Smith J",
        year=2020,
        container="Journal",
        volume=None,
        first_page=None,
    )
    monkeypatch.setattr(extractor, "_parse_references_ner_aligned", lambda segments: [good, None])

    refs = await extractor.extract(pd.DataFrame({"text": [FIRST, SECOND], "text_id": [11, 12]}))

    assert refs == [good]
    receipt = contents.reference_yield_receipt
    assert receipt.parsed_count == receipt.valid_count == 1
    assert receipt.losses.unresolved[0].source_text == SECOND
    assert receipt.losses.unresolved[0].source_text_ids == (12,)
    assert "unresolved_parse_segments" in receipt.reason_flags


async def test_parser_error_preserves_evidence_and_original_exception(monkeypatch):
    contents = PaperContents([], [], [], [], {})
    extractor = ReferenceExtractor(
        contents, llm_client=object(), seg_strategy="native", parse_strategy="ner"
    )
    monkeypatch.setattr(extractor, "_segment_references", AsyncMock(return_value=[FIRST]))
    error = RuntimeError("failed local parser")
    monkeypatch.setattr(
        extractor, "_parse_references_ner", lambda segments: (_ for _ in ()).throw(error)
    )

    with pytest.raises(RuntimeError) as raised:
        await extractor.extract(pd.DataFrame({"text": [FIRST], "text_id": [9]}))

    assert raised.value is error
    receipt = contents.reference_yield_receipt
    assert receipt.parsed_count == 0
    assert receipt.losses.unresolved[0].reason == "parser_failed"
    assert receipt.losses.unresolved[0].source_text == FIRST


async def test_skipped_llm_entry_keeps_source_after_existing_ner_recovery_fails(monkeypatch):
    contents = PaperContents([], [], [], [], {})
    good = PaperReferenceLLM(
        index=1,
        title="First reference title",
        authors="Smith J",
        year=2020,
        container="Journal of Testing",
        volume=None,
        first_page=None,
    )
    client = Mock(extract_references=AsyncMock(return_value=[good]))
    extractor = ReferenceExtractor(contents, llm_client=client, parse_strategy="llm")
    monkeypatch.setattr(extractor, "_segment_references", AsyncMock(return_value=[FIRST, SECOND]))
    recovery = Mock(return_value=[None])
    monkeypatch.setattr(extractor, "_parse_references_ner_aligned", recovery)

    parsed = await extractor.extract(pd.DataFrame({"text": [FIRST, SECOND], "text_id": [11, 12]}))

    assert len(parsed) == 1 and parsed[0].title == good.title
    client.extract_references.assert_awaited_once()
    recovery.assert_called_once_with([SECOND])
    receipt = contents.reference_yield_receipt
    assert receipt.parsed_count == receipt.valid_count == 1
    assert receipt.losses.unresolved[0].source_text_ids == (12,)
    assert receipt.losses.unresolved[0].reason == "no_aligned_parsed_reference"


def test_export_keeps_raw_loss_in_diagnostics_only():
    from bibr.export import export_paper_to_json
    from tests.test_media_export_107 import _extraction_block, _paper

    paper = _paper()
    paper.extraction = _extraction_block()
    losses = reference_losses(
        [(SECOND, 12)],
        [SECOND],
        [SECOND],
        [SECOND],
        {0: "no_title_or_authors"},
        parse_alignment_available=True,
    )
    paper.contents.reference_yield_receipt = replace(
        paper.contents.reference_yield_receipt, losses=losses
    )

    output = export_paper_to_json(paper, validate=False)

    assert output["bib"] == []
    receipt = output["extraction"]["diagnostics"]["reference_yield"]
    assert receipt["losses"]["unresolved"][0]["source_text"] == SECOND
    assert receipt["parsed_count"] == 2  # existing receipt stays untouched
