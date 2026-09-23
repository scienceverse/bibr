"""Shared export payload fixtures.

``completed_at`` is stripped from every fixture payload: it is the export's
only non-deterministic field, and leaving it in would make any full-payload
comparison flaky. The per-stage timings go the same way — they are wall-clock
too.

``extraction_block()`` lives here as the single definition of the minimal
``extraction`` skeleton, with ``extraction_export()`` as its typed twin for
tests that assert on the serializer rather than on a payload. Papers exported
outside the pipeline carry no ``extraction`` at all, so every test asserting on
telemetry (warnings, engines, diagnostics, enrichment, usage, regions) has to
attach one; before this module existed, several near-identical copies had
drifted apart across the export tests. Modules outside ``tests/export/`` import
them directly (``from tests.export.conftest import extraction_block``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bibr.export.json_export import _export_paper_payload
from bibr.export.models import ExtractionExport
from bibr.input.file import InputFile, InputFormat
from bibr.models import (
    Affiliation,
    BibAuthor,
    ExternalMatch,
    FundingEntry,
    MatchFunder,
    MatchOrganization,
    MatchSource,
    OrganizationMatch,
    PaperAuthor,
    PaperMetadata,
    PaperReference,
)
from bibr.paper import Paper
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperEquation,
    PaperFigure,
    PaperSection,
    PaperSentence,
    PaperTable,
    PaperURLLink,
    PaperXref,
)


def extraction_block(**overrides) -> dict:
    """A minimal ``extraction`` block, as ``ExportStage`` would build it."""
    block = {
        "producer": {"name": "bibr", "version": "0.0.0-test", "build_sha": None},
        "completed_at": "2026-07-24T10:00:00Z",
        "settings": {
            "ref_seg": "geom",
            "ref_parse": "ner",
            "crossref_enrich": False,
            "consolidate": "off",
        },
    }
    block.update(overrides)
    return block


def extraction_export(**overrides) -> ExtractionExport:
    """The same minimal block as :func:`extraction_block`, typed.

    Tests asserting on the *serializer* (which sub-blocks are omitted vs
    ``null``) need the model, not the dict. Both come from one definition so
    the typed and untyped skeletons cannot drift.
    """
    return ExtractionExport(**extraction_block(**overrides))


def _input_file() -> InputFile:
    return InputFile(
        path=Path("/tmp/demo.pdf"),
        file_hash="3a6eb0790f39ac87",
        sha256="3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
        input_format=InputFormat(
            file_extension=".pdf",
            detected_mime_type="application/pdf",
            file_type="pdf",
        ),
    )


def _contents(*, with_refs: bool) -> PaperContents:
    sentences = [
        PaperSentence(text_id=1, text="We measured the thing.", section_id=1, paragraph_id=1),
        PaperSentence(
            text_id=2,
            text="It replicated prior work [1].",
            section_id=2,
            paragraph_id=2,
            page_number=2,
        ),
    ]
    xrefs = [
        PaperXref(xref_id=1, xref_type="bib", contents="[1]", text_id=2, tier="numeric"),
        PaperXref(xref_id=1, xref_type="table", contents="Table 1", text_id=2),
    ]
    if not with_refs:
        # refs="off" leaves no bibliography to point at; the table xref stays.
        xrefs = xrefs[1:]
    table_df = pd.DataFrame([["1"]], columns=["A"])
    return PaperContents(
        sentences=sentences,
        xrefs=xrefs,
        sections=[
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Introduction",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.INTRODUCTION,
            ),
            PaperSection(
                section_id=2,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ],
        tables=[
            PaperTable(
                1,
                table_df,
                "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>",
                2,
                "Table 1. Values",
                2,
                [],
            )
        ],
        links=[
            PaperURLLink(
                url="https://example.org/data",
                section_id=2,
                paragraph_id=2,
                text_id=2,
                link_text="data",
            )
        ],
        sections_text={1: "We measured the thing.", 2: "It replicated prior work [1]."},
        figures=[PaperFigure(1, 2, None, "Figure 1. Plot", 2, [])],
        equations=[PaperEquation(text_id=2, grp_id=1, lhs="t", comp="=", rhs="3.42", df="28")],
    )


def _metadata(*, with_refs: bool) -> PaperMetadata:
    references = []
    if with_refs:
        references = [
            PaperReference(
                bib_id=1,
                title="A Prior Study",
                first_page="10",
                last_page="20",
                volume="7",
                authors="Smith, J., & Doe, A.",
                editors="Roe, R.",
                year=2020,
                container="Journal of Things",
                doi="10.1234/prior",
                bib_type="journal_article",
                text_id=None,
                match={
                    MatchSource.CROSSREF: ExternalMatch(
                        id="10.1234/prior",
                        # Enrichment scores 0-100; the export publishes 0-1.
                        score=98.0,
                        title="A Prior Study",
                        authors=[BibAuthor(given="Jane", family="Smith")],
                        year=2020,
                        container="Journal of Things",
                        doi="10.1234/prior",
                    )
                },
            )
        ]
    return PaperMetadata(
        doi="10.1234/demo",
        title="A Demonstration Paper",
        abstract="We demonstrate the export schema.",
        keywords=["schema", "export"],
        paper_type="empirical",
        paper_type_confidence=0.91,
        oecd_l1="Social Sciences",
        oecd_l2="Psychology and Cognitive Sciences",
        oecd_confidence=0.77,
        authors=[
            PaperAuthor(
                author_id=1,
                given="Jane",
                family="Smith",
                affiliation="Department of Things, Example University",
                email="jane@example.org",
                corresponding=True,
                orcid="0000-0002-1825-0097",
                role=["conceptualization"],
            )
        ],
        references=references,
        journal="Journal of Demonstrations",
        volume="12",
        issue="3",
        first_page="100",
        last_page="115",
        issn="1234-5678",
        publisher="Example Press",
        published="2026-01-15",
        license="CC-BY-4.0",
        funding_statement="Funded by the NSF under award 12345.",
        coi_statement="The authors declare no conflict of interest.",
        ethics_statement="Approved by the IRB.",
        data_availability="Data are available on OSF.",
        funding=[FundingEntry(funder="NSF", award_ids=["12345"])],
        affiliations=[
            Affiliation(
                text="Department of Things, Example University",
                institution="Example University",
                department="Department of Things",
                city="Exampleville",
                country="Netherlands",
                author_ids=[1],
            )
        ],
        match={
            MatchSource.CROSSREF: ExternalMatch(
                id="10.1234/demo",
                score=99.0,
                title="A Demonstration Paper",
                authors=[
                    BibAuthor(
                        given="Jane",
                        family="Smith",
                        orcid="https://orcid.org/0000-0002-1825-0097",
                        affiliation=[
                            MatchOrganization(
                                name="Example University", ror="https://ror.org/0abcde123"
                            )
                        ],
                    )
                ],
                year=2026,
                container="Journal of Demonstrations",
                doi="10.1234/demo",
                license_url="http://creativecommons.org/licenses/by/4.0/",
                funders=[
                    MatchFunder(
                        name="National Science Foundation",
                        funder_doi="10.13039/100000001",
                        award_ids=["12345"],
                    )
                ],
            )
        },
        affiliation_match={
            "Department of Things, Example University": OrganizationMatch(
                service_id="https://ror.org/0abcde123",
                score=1.0,
                name="Example University",
                country_code="NL",
            )
        },
        funder_match={
            "NSF": OrganizationMatch(
                service_id="https://ror.org/021nxhr62",
                score=1.0,
                name="U.S. National Science Foundation",
                country_code="US",
                funder_doi="10.13039/100000001",
            )
        },
        enrichment_complete=True if with_refs else None,
    )


def _demo_paper(*, with_refs: bool) -> Paper:
    paper = Paper(
        input_file=_input_file(),
        metadata=_metadata(with_refs=with_refs),
        contents=_contents(with_refs=with_refs),
    )
    paper.text_quality = 0.87
    paper.extraction = extraction_block()
    return paper


def as_parsed(paper: Paper) -> None:
    """Reshape the demo paper the way ``create_content_sections`` leaves one.

    Each caption and footnote is a synthetic section holding its sentence, and
    a float points at its section while remembering the body section it sits
    in. The export turns these into caption rows and a ``footnote`` row.
    """
    contents = paper.contents
    contents.sections += [
        PaperSection(
            section_id=3,
            header="Figure 1",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.FIGURE,
            synthetic_kind="figure",
        ),
        PaperSection(
            section_id=4,
            header="Table 1",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.TABLE,
            synthetic_kind="table",
        ),
        PaperSection(
            section_id=5,
            header="Footnote 1",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.FOOTNOTE,
            synthetic_kind="footnote",
            footnote_label="*",
        ),
    ]
    contents.sentences += [
        PaperSentence(
            text_id=3, text="Figure 1. Plot", section_id=3, paragraph_id=3, page_number=2
        ),
        PaperSentence(
            text_id=4, text="Table 1. Values", section_id=4, paragraph_id=4, page_number=2
        ),
        PaperSentence(
            text_id=5, text="* Collected in 2020.", section_id=5, paragraph_id=5, page_number=2
        ),
    ]
    for item, own_section in ((contents.figures[0], 3), (contents.tables[0], 4)):
        item._body_section_id = item.section_id
        item.section_id = own_section
    contents.xrefs += [
        PaperXref(xref_id=1, xref_type="figure", contents="Figure 1", text_id=2),
        PaperXref(xref_id=5, xref_type="foot", contents="*", text_id=2),
    ]


def _strip_nondeterministic(payload: dict) -> dict:
    extraction = payload.get("extraction")
    if isinstance(extraction, dict):
        extraction.pop("completed_at", None)
        timings = extraction.get("timings")
        if isinstance(timings, dict):
            timings.pop("stages", None)
            timings.pop("total_seconds", None)
    return payload


@pytest.fixture
def demo_paper() -> Paper:
    """A fully-populated Paper: every root table has at least one row."""
    return _demo_paper(with_refs=True)


@pytest.fixture
def demo_paper_refs_off() -> Paper:
    """The same paper from a ``refs="off"`` run — no bibliography at all."""
    return _demo_paper(with_refs=False)


@pytest.fixture
def export_payload(demo_paper) -> dict:
    """A fully-populated payload from the shared demo paper."""
    return _strip_nondeterministic(_export_paper_payload(demo_paper))


@pytest.fixture
def export_payload_refs_off(demo_paper_refs_off) -> dict:
    """A payload from a ``refs="off"`` run — every root table must still exist."""
    return _strip_nondeterministic(_export_paper_payload(demo_paper_refs_off))
