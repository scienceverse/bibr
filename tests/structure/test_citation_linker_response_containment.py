"""Malformed optional linking responses preserve confirmed links and honest receipts."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bibr.clients.structured import StructuredResponseError
from bibr.exceptions import ProcessingError
from bibr.models import PaperReference
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
from bibr.structure.citation_linker import _resolve_with_llm, detect_bib_xrefs_with_receipt


def citation_fixture():
    sections = [
        PaperSection(1, "Introduction", 1, None, CanonicalSection.INTRODUCTION),
        PaperSection(2, "References", 1, None, CanonicalSection.REFERENCES),
    ]
    body = [
        "Prior work [1] established the method.",
        "Jones (2021) reported an independent replication.",
        "The interval was [.12, .44].",
        "An unknown numeric marker [9] remains.",
        "A distinct report [Zed] merits review.",
        "Unknown (2018) discussed a different cohort.",
        "A second mention of [Zed] remains unresolved.",
        "The study kept [1] while [Zed] remains ambiguous.",
    ]
    sentences = [PaperSentence(i, text, 1, i, page_number=1) for i, text in enumerate(body, 1)]
    refs = []
    for i, (authors, year) in enumerate(
        [("Smith AB", 2020), ("Jones CD", 2021), ("Brown EF", 2022), ("Green GH", 2023)], 1
    ):
        text_id = 100 + i
        sentences.append(
            PaperSentence(
                text_id, f"[{i}] {authors}. Printed study {i}. {year}.", 2, text_id, page_number=2
            )
        )
        refs.append(
            PaperReference(
                bib_id=i,
                title=f"Printed study {i}",
                authors=authors,
                year=year,
                text_id=text_id,
                first_page=None,
                volume=None,
                container=None,
            )
        )
    return PaperContents(sentences, sections, [], [], {}), refs


@pytest.mark.parametrize(
    "category",
    ["empty", "non_json", "non_object", "truncated", "trailing_content", "schema_invalid"],
)
async def test_invalid_response_keeps_deterministic_links_and_marks_only_offered_spans(
    category, caplog
):
    contents, refs = citation_fixture()
    expected, original_receipt = await detect_bib_xrefs_with_receipt(
        contents.sentences, contents.sections, refs
    )
    error = StructuredResponseError(category, last_completion="RAW-PROVIDER-SECRET")
    error.category = "MUTABLE-ATTRIBUTE-SECRET"
    client = SimpleNamespace(resolve_citations=AsyncMock(side_effect=error))

    links, receipt = await detect_bib_xrefs_with_receipt(
        contents.sentences, contents.sections, refs, llm_client=client
    )

    assert links == expected
    assert {(link.text_id, link.xref_id, link.tier) for link in links} == {
        (1, 1, "numeric"),
        (2, 2, "author-year"),
        (8, 1, "numeric"),
    }
    accepted_before = [candidate for candidate in original_receipt.candidates if candidate.accepted]
    assert [candidate for candidate in receipt.candidates if candidate.accepted] == accepted_before
    flagged = [
        candidate
        for candidate in receipt.candidates
        if any(reason.startswith("llm_response_invalid:") for reason in candidate.rejection_reasons)
    ]
    assert {candidate.text_id for candidate in flagged} == {5, 6, 7, 8}
    for candidate in flagged:
        assert not candidate.accepted
        assert f"llm_response_invalid:{category}" in candidate.rejection_reasons
        sentence = next(row for row in contents.sentences if row.text_id == candidate.text_id)
        assert candidate.raw == sentence.text[candidate.start : candidate.end]
    assert not any(candidate.text_id in {3, 4} for candidate in flagged)
    assert receipt.unique_linked_bib_fraction == 0.5
    assert receipt.resolved_candidate_fraction == len(accepted_before) / len(receipt.candidates)
    client.resolve_citations.assert_awaited_once()
    offered = client.resolve_citations.await_args.kwargs["ambiguous_citations"]
    assert sum(text == "[Zed]" for _, text in offered) == 1
    assert "RAW-PROVIDER-SECRET" not in repr(receipt) + caplog.text
    assert "MUTABLE-ATTRIBUTE-SECRET" not in repr(receipt) + caplog.text


@pytest.mark.parametrize(
    "error",
    [ProcessingError("typed transport", error_code="llm_invalid_output"), asyncio.CancelledError()],
)
async def test_generic_typed_failures_and_cancellation_still_propagate(error):
    contents, refs = citation_fixture()
    client = SimpleNamespace(resolve_citations=AsyncMock(side_effect=error))
    with pytest.raises(type(error)) as raised:
        await detect_bib_xrefs_with_receipt(
            contents.sentences, contents.sections, refs, llm_client=client
        )
    assert raised.value is error


@pytest.mark.parametrize("error", [TimeoutError("raw timeout"), RuntimeError("raw failure")])
async def test_existing_untyped_best_effort_behavior_does_not_get_structured_marker(error):
    contents, refs = citation_fixture()
    client = SimpleNamespace(resolve_citations=AsyncMock(side_effect=error))
    links, receipt = await detect_bib_xrefs_with_receipt(
        contents.sentences, contents.sections, refs, llm_client=client
    )
    assert {(link.text_id, link.xref_id) for link in links} == {(1, 1), (2, 2), (8, 1)}
    assert not any(
        reason.startswith("llm_response_invalid:")
        for candidate in receipt.candidates
        for reason in candidate.rejection_reasons
    )


async def test_direct_llm_resolver_keeps_typed_error_contract():
    _, refs = citation_fixture()
    error = StructuredResponseError("truncated")
    client = SimpleNamespace(resolve_citations=AsyncMock(side_effect=error))
    with pytest.raises(StructuredResponseError) as raised:
        await _resolve_with_llm([(5, "[Zed]")], refs, client, "source-hash")
    assert raised.value is error
