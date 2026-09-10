"""Tests for research-integrity extraction: verbatim statement copy (no LLM),
structured funding/contributions LLM schema, role mapping, export, skip logic.
"""

import importlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from bibr.export.json_export import export_paper_to_json
from bibr.extract.research_integrity import (
    _collect_affiliations,
    copy_integrity_statements,
    extract_structured_integrity,
)
from bibr.input.file import InputFile, InputFormat
from bibr.models import PaperAuthor, PaperMetadata
from bibr.paper import Paper
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.schemas import ResearchIntegrityLLM

# ── helpers ────────────────────────────────────────────────────────────


def _contents(sections_spec: list[tuple[int, str, CanonicalSection, list[str]]]) -> PaperContents:
    """Build PaperContents from ``(section_id, header, type, [sentences])`` tuples."""
    sections = [PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)]
    sentences = []
    tid = 1
    for section_id, header, section_type, sents in sections_spec:
        sections.append(
            PaperSection(
                section_id=section_id,
                header=header,
                level=1,
                parent_section_id=0,
                section_type=section_type,
            )
        )
        for s in sents:
            sentences.append(
                PaperSentence(text_id=tid, text=s, section_id=section_id, paragraph_id=tid)
            )
            tid += 1
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
    )


def _author(author_id: int, given: str, family: str, affiliation: str = "") -> PaperAuthor:
    return PaperAuthor(author_id=author_id, given=given, family=family, affiliation=affiliation)


def _paper(metadata: PaperMetadata, contents: PaperContents) -> Paper:
    input_file = InputFile(
        path=Path("/tmp/test.pdf"),
        file_hash="abc123",
        input_format=InputFormat(
            file_extension=".pdf", detected_mime_type="application/pdf", file_type="pdf"
        ),
    )
    return Paper(input_file=input_file, metadata=metadata, contents=contents)


# ── A) verbatim statement copy (no LLM) ─────────────────────────────────


def test_copy_statements_each_type_populated():
    contents = _contents(
        [
            (1, "Funding", CanonicalSection.FUNDING, ["This work was funded by the NSF."]),
            (2, "Conflicts", CanonicalSection.COI, ["The authors declare no conflict."]),
            (3, "Ethics", CanonicalSection.ETHICS, ["Approved by the IRB."]),
            (4, "Data", CanonicalSection.OPEN_DATA, ["Data are available on OSF."]),
        ]
    )
    metadata = PaperMetadata(doi="", title="T")
    copy_integrity_statements(contents, metadata)
    assert metadata.funding_statement == "This work was funded by the NSF."
    assert metadata.coi_statement == "The authors declare no conflict."
    assert metadata.ethics_statement == "Approved by the IRB."
    assert metadata.data_availability == "Data are available on OSF."


def test_copy_statements_absent_is_none():
    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hello."])])
    metadata = PaperMetadata(doi="", title="T")
    copy_integrity_statements(contents, metadata)
    assert metadata.funding_statement is None
    assert metadata.coi_statement is None
    assert metadata.ethics_statement is None
    assert metadata.data_availability is None


def test_copy_statements_multiple_same_type_joined():
    contents = _contents(
        [
            (1, "Funding", CanonicalSection.FUNDING, ["Grant A."]),
            (2, "Funding", CanonicalSection.FUNDING, ["Grant B."]),
        ]
    )
    metadata = PaperMetadata(doi="", title="T")
    copy_integrity_statements(contents, metadata)
    assert metadata.funding_statement == "Grant A.\n\nGrant B."


def test_copy_statements_works_without_llm_client():
    # The copy is a pure function — no client involved.
    contents = _contents([(1, "Funding", CanonicalSection.FUNDING, ["Funded by X."])])
    metadata = PaperMetadata(doi="", title="T")
    copy_integrity_statements(contents, metadata)
    assert metadata.funding_statement == "Funded by X."


# ── B) LLM schema parsing ───────────────────────────────────────────────


def test_llm_schema_parses_realistic_payload():
    r = ResearchIntegrityLLM.model_validate(
        {
            "funding": [
                {"funder": "National Science Foundation", "award_ids": ["R01-MH123", "834861"]},
                {"funder": "Wellcome Trust", "award_ids": []},
            ],
            "contributions": [
                {"author": "J.W.", "roles": ["Conceptualization", "wrote the manuscript"]},
                {"author": "A.B.", "roles": ["Data curation"]},
            ],
        }
    )
    assert r.funding[0].funder == "National Science Foundation"
    assert r.funding[0].award_ids == ["R01-MH123", "834861"]
    assert r.funding[1].award_ids == []
    assert r.contributions[0].roles == ["Conceptualization", "wrote the manuscript"]


def test_llm_schema_null_handling():
    # Nulls on the list fields and on award_ids/roles coerce to empty lists.
    r = ResearchIntegrityLLM.model_validate({"funding": None, "contributions": None})
    assert r.funding == []
    assert r.contributions == []
    r2 = ResearchIntegrityLLM.model_validate(
        {"funding": [{"funder": "NSF", "award_ids": None}], "contributions": []}
    )
    assert r2.funding[0].award_ids == []


def test_llm_schema_unwraps_schema_name_envelope():
    # LLMResponse base unwraps {"ResearchIntegrityLLM": {...}} envelopes.
    r = ResearchIntegrityLLM.model_validate(
        {"ResearchIntegrityLLM": {"funding": [], "contributions": []}}
    )
    assert r.funding == []


# ── C) role mapping ─────────────────────────────────────────────────────


async def _run_structured(contents, metadata, result: ResearchIntegrityLLM):
    client = AsyncMock()
    client.extract_research_integrity = AsyncMock(return_value=result)
    await extract_structured_integrity(contents, metadata, client, "hash")
    return client


async def test_role_mapping_full_name_match():
    contents = _contents(
        [(1, "Author Contributions", CanonicalSection.AUTHOR_CONTRIBUTIONS, ["stmt"])]
    )
    metadata = PaperMetadata(
        doi="", title="T", authors=[_author(1, "Jakub", "Werner"), _author(2, "Alice", "Brown")]
    )
    result = ResearchIntegrityLLM.model_validate(
        {"contributions": [{"author": "Jakub Werner", "roles": ["Conceptualization"]}]}
    )
    await _run_structured(contents, metadata, result)
    assert metadata.authors[0].role == ["Conceptualization"]
    assert metadata.authors[1].role == []


async def test_role_mapping_initials_match():
    contents = _contents(
        [(1, "Author Contributions", CanonicalSection.AUTHOR_CONTRIBUTIONS, ["stmt"])]
    )
    metadata = PaperMetadata(
        doi="", title="T", authors=[_author(1, "Jakub", "Werner"), _author(2, "Alice", "Brown")]
    )
    result = ResearchIntegrityLLM.model_validate(
        {"contributions": [{"author": "J.W.", "roles": ["Writing"]}]}
    )
    await _run_structured(contents, metadata, result)
    assert metadata.authors[0].role == ["Writing"]
    assert metadata.authors[1].role == []


async def test_role_mapping_repeated_author_entries_merge():
    # Role-keyed statements ("Conceptualization: J.W.; Writing: J.W.") produce
    # one entry per role for the same author — roles must merge, not overwrite.
    contents = _contents(
        [(1, "Author Contributions", CanonicalSection.AUTHOR_CONTRIBUTIONS, ["stmt"])]
    )
    metadata = PaperMetadata(doi="", title="T", authors=[_author(1, "Jakub", "Werner")])
    result = ResearchIntegrityLLM.model_validate(
        {
            "contributions": [
                {"author": "J.W.", "roles": ["Conceptualization"]},
                {"author": "J.W.", "roles": ["Writing", "Conceptualization"]},
            ]
        }
    )
    await _run_structured(contents, metadata, result)
    assert metadata.authors[0].role == ["Conceptualization", "Writing"]


async def test_role_mapping_ambiguous_dropped():
    contents = _contents(
        [(1, "Author Contributions", CanonicalSection.AUTHOR_CONTRIBUTIONS, ["stmt"])]
    )
    # Two authors share family "Wang" → ambiguous family token dropped.
    metadata = PaperMetadata(
        doi="", title="T", authors=[_author(1, "Li", "Wang"), _author(2, "Wei", "Wang")]
    )
    result = ResearchIntegrityLLM.model_validate(
        {"contributions": [{"author": "Wang", "roles": ["Analysis"]}]}
    )
    await _run_structured(contents, metadata, result)
    assert metadata.authors[0].role == []
    assert metadata.authors[1].role == []


async def test_role_mapping_unmatched_dropped_and_no_contrib_keeps_empty():
    contents = _contents(
        [(1, "Author Contributions", CanonicalSection.AUTHOR_CONTRIBUTIONS, ["stmt"])]
    )
    metadata = PaperMetadata(
        doi="", title="T", authors=[_author(1, "Jakub", "Werner"), _author(2, "Alice", "Brown")]
    )
    result = ResearchIntegrityLLM.model_validate(
        {"contributions": [{"author": "Zoe Zimmer", "roles": ["Conceptualization"]}]}
    )
    await _run_structured(contents, metadata, result)
    # Unmatched entry dropped; both authors keep [].
    assert metadata.authors[0].role == []
    assert metadata.authors[1].role == []


async def test_structured_funding_stored():
    contents = _contents([(1, "Funding", CanonicalSection.FUNDING, ["Funded by NSF grant 123."])])
    metadata = PaperMetadata(doi="", title="T", funding_statement="Funded by NSF grant 123.")
    result = ResearchIntegrityLLM.model_validate(
        {"funding": [{"funder": "NSF", "award_ids": ["123"]}]}
    )
    await _run_structured(contents, metadata, result)
    assert len(metadata.funding) == 1
    assert metadata.funding[0].funder == "NSF"
    assert metadata.funding[0].award_ids == ["123"]


async def test_structured_funding_uses_selected_statement_not_all_funding_sections():
    contents = _contents(
        [
            (
                1,
                "Funding in public policy",
                CanonicalSection.FUNDING,
                ["Funding can affect institutional priorities."],
            ),
            (
                2,
                "Acknowledgments",
                CanonicalSection.ACKNOWLEDGMENT,
                ["This work was supported by NSF grant $^{123}$ ."],
            ),
        ]
    )
    integrity = importlib.import_module("bibr.extract.integrity_statements")
    resolution = integrity.resolve_integrity_statements(contents, mode="active")
    contents.finalize_text()
    metadata = PaperMetadata(doi="", title="T")
    integrity.apply_integrity_resolution(contents, metadata, resolution)
    result = ResearchIntegrityLLM.model_validate(
        {"funding": [{"funder": "NSF", "award_ids": ["123"]}]}
    )
    client = AsyncMock()
    client.extract_research_integrity = AsyncMock(return_value=result)

    await extract_structured_integrity(
        contents,
        metadata,
        client,
        "hash",
        integrity_resolution=resolution,
    )

    kwargs = client.extract_research_integrity.await_args.kwargs
    assert kwargs["funding_text"] == "This work was supported by NSF grant 123 ."
    assert "institutional priorities" not in kwargs["funding_text"]


async def test_structured_funding_preserves_prepopulated_native_statement():
    contents = _contents(
        [(1, "Funding", CanonicalSection.FUNDING, ["Supported by PDF Foundation."])]
    )
    integrity = importlib.import_module("bibr.extract.integrity_statements")
    resolution = integrity.resolve_integrity_statements(contents, mode="active")
    metadata = PaperMetadata(
        doi="",
        title="Native",
        funding_statement="Supported by the Native JATS Foundation.",
    )
    contents.preparsed_metadata = metadata
    result = ResearchIntegrityLLM.model_validate(
        {"funding": [{"funder": "Native JATS Foundation", "award_ids": []}]}
    )
    client = AsyncMock()
    client.extract_research_integrity = AsyncMock(return_value=result)

    await extract_structured_integrity(
        contents,
        metadata,
        client,
        "hash",
        integrity_resolution=resolution,
    )

    assert client.extract_research_integrity.await_args.kwargs["funding_text"] == (
        "Supported by the Native JATS Foundation."
    )


async def test_shadow_structured_funding_uses_bounded_selection_not_legacy_scalar():
    contents = _contents(
        [
            (
                1,
                "Funding in public policy",
                CanonicalSection.FUNDING,
                ["Funding can affect institutional priorities."],
            ),
            (
                2,
                "Acknowledgments",
                CanonicalSection.ACKNOWLEDGMENT,
                ["This work was supported by NSF grant 123."],
            ),
        ]
    )
    integrity = importlib.import_module("bibr.extract.integrity_statements")
    resolution = integrity.resolve_integrity_statements(contents, mode="shadow")
    metadata = PaperMetadata(doi="", title="T")
    integrity.apply_integrity_resolution(contents, metadata, resolution)
    assert metadata.funding_statement == "Funding can affect institutional priorities."

    result = ResearchIntegrityLLM.model_validate(
        {"funding": [{"funder": "NSF", "award_ids": ["123"]}]}
    )
    client = AsyncMock()
    client.extract_research_integrity = AsyncMock(return_value=result)
    await extract_structured_integrity(
        contents,
        metadata,
        client,
        "hash",
        integrity_resolution=resolution,
    )

    assert client.extract_research_integrity.await_args.kwargs["funding_text"] == (
        "This work was supported by NSF grant 123."
    )


async def test_shadow_exact_legacy_section_scalar_and_bounded_structured_funding_diverge():
    contents = _contents(
        [
            (
                1,
                "Funding",
                CanonicalSection.FUNDING,
                [
                    "This work was supported by NSF grant 123.",
                    "Ethics: Not applicable.",
                ],
            )
        ]
    )
    contents.sentences[1].paragraph_id = contents.sentences[0].paragraph_id
    integrity = importlib.import_module("bibr.extract.integrity_statements")
    resolution = integrity.resolve_integrity_statements(contents, mode="shadow")
    metadata = PaperMetadata(doi="", title="T")

    integrity.apply_integrity_resolution(contents, metadata, resolution)

    assert metadata.funding_statement == (
        "This work was supported by NSF grant 123. Ethics: Not applicable."
    )
    assert any(
        issue.code == "VAL_STATEMENT_SUSPECT" and "funding_statement" in issue.evidence_ids
        for issue in resolution.issues
    )

    result = ResearchIntegrityLLM.model_validate(
        {"funding": [{"funder": "NSF", "award_ids": ["123"]}]}
    )
    client = AsyncMock()
    client.extract_research_integrity = AsyncMock(return_value=result)
    await extract_structured_integrity(
        contents,
        metadata,
        client,
        "hash",
        integrity_resolution=resolution,
    )

    assert client.extract_research_integrity.await_args.kwargs["funding_text"] == (
        "This work was supported by NSF grant 123."
    )


async def test_legacy_structured_funding_uses_exact_compatibility_scalar():
    contents = _contents(
        [
            (
                1,
                "Funding",
                CanonicalSection.FUNDING,
                [
                    "This work was supported by NSF grant $^{123}$ .",
                    "Ethics: Not applicable.",
                ],
            )
        ]
    )
    contents.sentences[1].paragraph_id = contents.sentences[0].paragraph_id
    integrity = importlib.import_module("bibr.extract.integrity_statements")
    resolution = integrity.resolve_integrity_statements(contents, mode="legacy")
    contents.finalize_text()
    metadata = PaperMetadata(doi="", title="T")
    integrity.apply_integrity_resolution(contents, metadata, resolution)
    assert metadata.funding_statement == (
        "This work was supported by NSF grant $^{123}$ . Ethics: Not applicable."
    )

    result = ResearchIntegrityLLM.model_validate(
        {"funding": [{"funder": "NSF", "award_ids": ["123"]}]}
    )
    client = AsyncMock()
    client.extract_research_integrity = AsyncMock(return_value=result)
    await extract_structured_integrity(
        contents,
        metadata,
        client,
        "hash",
        integrity_resolution=resolution,
    )

    assert client.extract_research_integrity.await_args.kwargs["funding_text"] == (
        "This work was supported by NSF grant $^{123}$ . Ethics: Not applicable."
    )


# ── D) skip logic ───────────────────────────────────────────────────────


async def test_legacy_structured_funding_ignores_lexical_only_scalar():
    contents = _contents(
        [
            (
                1,
                "Acknowledgments",
                CanonicalSection.ACKNOWLEDGMENT,
                ["This work was supported by NSF grant 123."],
            ),
            (
                2,
                "Author Contributions",
                CanonicalSection.AUTHOR_CONTRIBUTIONS,
                ["J.W. wrote the manuscript."],
            ),
        ]
    )
    integrity = importlib.import_module("bibr.extract.integrity_statements")
    resolution = integrity.resolve_integrity_statements(contents, mode="legacy")
    metadata = PaperMetadata(doi="", title="T")
    integrity.apply_integrity_resolution(contents, metadata, resolution)
    assert metadata.funding_statement == "This work was supported by NSF grant 123."

    client = AsyncMock()
    client.extract_research_integrity = AsyncMock(
        return_value=ResearchIntegrityLLM.model_validate({})
    )
    await extract_structured_integrity(
        contents,
        metadata,
        client,
        "hash",
        integrity_resolution=resolution,
    )

    assert client.extract_research_integrity.await_args.kwargs["funding_text"] == ""


async def test_skip_llm_when_no_relevant_sections():
    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hello."])])
    metadata = PaperMetadata(doi="", title="T")
    client = AsyncMock()
    client.extract_research_integrity = AsyncMock()
    await extract_structured_integrity(contents, metadata, client, "hash")
    client.extract_research_integrity.assert_not_called()
    assert metadata.funding == []


async def test_llm_called_when_only_contributions_present():
    contents = _contents(
        [(1, "Author Contributions", CanonicalSection.AUTHOR_CONTRIBUTIONS, ["JW wrote it."])]
    )
    metadata = PaperMetadata(doi="", title="T", authors=[_author(1, "Jakub", "Werner")])
    result = ResearchIntegrityLLM.model_validate({"contributions": []})
    client = await _run_structured(contents, metadata, result)
    client.extract_research_integrity.assert_called_once()


async def test_no_funding_entries_when_funding_text_empty():
    # Funding statement absent but affiliations present: the LLM still runs for
    # affiliations, but any funding entry it hallucinates must be discarded.
    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hello."])])
    metadata = PaperMetadata(
        doi="",
        title="T",
        authors=[_author(1, "Jakub", "Werner", affiliation="University of Twente")],
    )
    result = ResearchIntegrityLLM.model_validate(
        {
            "funding": [{"funder": "Phantom Foundation", "award_ids": ["999"]}],
            "affiliations": [{"index": 1, "institution": "University of Twente"}],
        }
    )
    client = await _run_structured(contents, metadata, result)
    client.extract_research_integrity.assert_called_once()
    assert metadata.funding == []
    # affiliation parse is preserved
    assert len(metadata.affiliations) == 1
    assert metadata.affiliations[0].institution == "University of Twente"


# ── F) affiliations ─────────────────────────────────────────────────────


def test_collect_affiliations_split_dedupe_and_shared_author_ids():
    authors = [
        _author(1, "A", "One", "Dept of Psychology, Univ X; Univ X, London, UK"),
        _author(2, "B", "Two", "Univ X, London, UK"),
        _author(3, "C", "Three", ""),
    ]
    unique, author_ids = _collect_affiliations(authors)
    # First-seen order preserved; components split on "; " and stripped.
    assert unique == ["Dept of Psychology, Univ X", "Univ X, London, UK"]
    # Shared affiliation carries both authors that print it, in author order.
    assert author_ids == [[1], [1, 2]]


def test_collect_affiliations_dedupes_within_single_author():
    authors = [_author(1, "A", "One", "Univ X; Univ X")]
    unique, author_ids = _collect_affiliations(authors)
    assert unique == ["Univ X"]
    assert author_ids == [[1]]


async def test_affiliation_gate_fires_with_affiliations_only():
    # No funding / contributions sections, but authors carry affiliations.
    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hello."])])
    metadata = PaperMetadata(
        doi="", title="T", authors=[_author(1, "A", "One", "Univ X, London, UK")]
    )
    result = ResearchIntegrityLLM.model_validate(
        {"affiliations": [{"index": 1, "institution": "Univ X", "city": "London", "country": "UK"}]}
    )
    client = await _run_structured(contents, metadata, result)
    client.extract_research_integrity.assert_called_once()
    assert len(metadata.affiliations) == 1
    aff = metadata.affiliations[0]
    assert aff.text == "Univ X, London, UK"
    assert aff.institution == "Univ X"
    assert aff.city == "London"
    assert aff.country == "UK"
    assert aff.department is None
    assert aff.author_ids == [1]


async def test_affiliation_out_of_range_index_dropped():
    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hi."])])
    metadata = PaperMetadata(doi="", title="T", authors=[_author(1, "A", "One", "Univ X")])
    result = ResearchIntegrityLLM.model_validate(
        {
            "affiliations": [
                {"index": 5, "institution": "Bogus"},
                {"index": 1, "institution": "Univ X"},
            ]
        }
    )
    await _run_structured(contents, metadata, result)
    assert len(metadata.affiliations) == 1
    assert metadata.affiliations[0].institution == "Univ X"


async def test_affiliation_duplicate_index_first_wins():
    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hi."])])
    metadata = PaperMetadata(doi="", title="T", authors=[_author(1, "A", "One", "Univ X")])
    result = ResearchIntegrityLLM.model_validate(
        {
            "affiliations": [
                {"index": 1, "institution": "First"},
                {"index": 1, "institution": "Second"},
            ]
        }
    )
    await _run_structured(contents, metadata, result)
    assert metadata.affiliations[0].institution == "First"


async def test_affiliation_omitted_entry_still_exported_with_none():
    # Two affiliations sent; LLM only parses the second — the first is still
    # emitted verbatim with all parse fields None.
    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hi."])])
    metadata = PaperMetadata(
        doi="",
        title="T",
        authors=[_author(1, "A", "One", "Univ X; Univ Y")],
    )
    result = ResearchIntegrityLLM.model_validate(
        {"affiliations": [{"index": 2, "institution": "Univ Y"}]}
    )
    await _run_structured(contents, metadata, result)
    assert [a.text for a in metadata.affiliations] == ["Univ X", "Univ Y"]
    assert metadata.affiliations[0].institution is None
    assert metadata.affiliations[1].institution == "Univ Y"


async def test_affiliation_whitespace_and_none_string_coercion():
    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hi."])])
    metadata = PaperMetadata(doi="", title="T", authors=[_author(1, "A", "One", "Univ X")])
    result = ResearchIntegrityLLM.model_validate(
        {
            "affiliations": [
                {
                    "index": 1,
                    "institution": "  Univ X  ",
                    "department": "   ",
                    "city": "None",
                    "country": "null",
                }
            ]
        }
    )
    await _run_structured(contents, metadata, result)
    aff = metadata.affiliations[0]
    assert aff.institution == "Univ X"  # stripped
    assert aff.department is None  # whitespace-only → None
    assert aff.city is None  # "None" string coerced
    assert aff.country is None  # "null" string coerced


async def test_affiliation_output_order_follows_input_not_llm():
    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hi."])])
    metadata = PaperMetadata(doi="", title="T", authors=[_author(1, "A", "One", "Univ X; Univ Y")])
    result = ResearchIntegrityLLM.model_validate(
        {
            "affiliations": [
                {"index": 2, "institution": "Univ Y"},
                {"index": 1, "institution": "Univ X"},
            ]
        }
    )
    await _run_structured(contents, metadata, result)
    assert [a.text for a in metadata.affiliations] == ["Univ X", "Univ Y"]


# ── E) export ───────────────────────────────────────────────────────────


def test_export_statements_in_metadata_and_funding_roundtrips():
    from bibr.models import FundingEntry

    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hello."])])
    metadata = PaperMetadata(
        doi="10.1/x",
        title="T",
        authors=[_author(1, "Jakub", "Werner")],
        funding_statement="Funded by the NSF.",
        coi_statement="No conflict.",
        ethics_statement="IRB approved.",
        data_availability="On OSF.",
        funding=[FundingEntry(funder="NSF", award_ids=["123"])],
    )
    metadata.authors[0].role = ["Conceptualization"]
    out = export_paper_to_json(_paper(metadata, contents))

    assert out["metadata"]["funding_statement"] == "Funded by the NSF."
    assert out["metadata"]["coi_statement"] == "No conflict."
    assert out["metadata"]["ethics_statement"] == "IRB approved."
    assert out["metadata"]["data_availability"] == "On OSF."
    assert out["funding"] == [{"funding_id": 1, "funder": "NSF", "award_ids": ["123"]}]
    assert out["author"][0]["role"] == ["Conceptualization"]


def test_export_defaults_when_absent():
    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hello."])])
    metadata = PaperMetadata(doi="10.1/x", title="T", authors=[_author(1, "A", "B")])
    out = export_paper_to_json(_paper(metadata, contents))
    assert out["metadata"]["funding_statement"] is None
    assert out["funding"] == []
    assert out["affiliation"] == []
    assert out["author"][0]["role"] == []


def test_export_affiliations_roundtrips():
    from bibr.models import Affiliation

    contents = _contents([(1, "Intro", CanonicalSection.INTRODUCTION, ["Hello."])])
    metadata = PaperMetadata(
        doi="10.1/x",
        title="T",
        authors=[_author(1, "Jakub", "Werner")],
        affiliations=[
            Affiliation(
                text="Dept of Psychology, Univ X, London, UK",
                institution="Univ X",
                department="Dept of Psychology",
                city="London",
                country="UK",
                author_ids=[1],
            ),
            Affiliation(text="Univ Y"),
        ],
    )
    out = export_paper_to_json(_paper(metadata, contents))
    assert out["affiliation"] == [
        {
            "affiliation_id": 1,
            "text": "Dept of Psychology, Univ X, London, UK",
            "institution": "Univ X",
            "department": "Dept of Psychology",
            "city": "London",
            "country": "UK",
            "author_ids": [1],
        },
        {
            "affiliation_id": 2,
            "text": "Univ Y",
            "institution": None,
            "department": None,
            "city": None,
            "country": None,
            "author_ids": [],
        },
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
