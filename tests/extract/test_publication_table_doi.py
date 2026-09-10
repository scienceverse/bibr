"""Identity boxes must be spatially owned and distinct from cited publications."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from bibr.config import Settings
from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates
from bibr.extract.front_matter import (
    FrontMatterBlock,
    FrontMatterCandidate,
    FrontMatterResolution,
)
from bibr.input.file import InputFile
from bibr.models import PaperMetadata
from bibr.paper import Paper
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    PaperTable,
    Provenance,
    RegionSummary,
)
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState

BOX = "Vol. 12(3), pp. 45-51 DOI: 10.4321/JOURNAL.2025.42 ISSN 1234-567X Copyright 2025"


def _contents(text=BOX):
    title = FrontMatterCandidate(
        candidate_id="title",
        source_kind="sentence",
        reading_order=2,
        page=1,
        bbox=(100, 250, 900, 320),
        region_label="title",
        font_size=18,
        font_bold=True,
        section_id=1,
        text_ids=(1,),
        paragraph_id=1,
        raw_text="Independent article title",
        normalized_text="independent article title",
        roles=frozenset({"title"}),
    )
    block = FrontMatterBlock("article", ("title",), ("title",), pages=(1,))
    contents = PaperContents(
        sentences=[PaperSentence(1, title.raw_text, 1, 1, page_number=1)],
        sections=[
            PaperSection(0, "Root", 0, None, CanonicalSection.TITLE),
            PaperSection(1, "Title", 1, 0, CanonicalSection.TITLE),
            PaperSection(2, "Table 1", 1, 0, CanonicalSection.TABLE),
        ],
        tables=[
            PaperTable(
                table_id=7,
                df=pd.DataFrame([[text, "Journal of Examples"]]),
                tbl_html="",
                section_id=2,
                page_number=1,
                provenance=[Provenance(1, (80, 50, 950, 180))],
            )
        ],
        links=[],
        sections_text={},
    )
    contents.region_summaries = [
        RegionSummary(
            page=1,
            index=7,
            label="table",
            bbox=(80, 50, 950, 180),
            section_id=2,
            content=text[:200],
            raw_ocr_content=text,
        )
    ]
    contents.front_matter_resolution = FrontMatterResolution(
        candidates=(title,),
        blocks=(block,),
        selected_block_id="article",
        selection_method="single_block",
        reason_flags=(),
        allowed_text_ids=frozenset({1}),
        allowed_section_ids=frozenset({1}),
    )
    return contents


@pytest.mark.parametrize(
    "text",
    [
        BOX,
        "Vol. 12(3), pp. 45-51DOI: 10.4321/JOURNAL.2025.42Article Number: X123"
        "ISSN 1234-567XCopyright 2025",
    ],
)
def test_owned_publisher_box_recovers_doi_and_table_provenance(text):
    selection = select_doi_candidates(collect_doi_candidates(_contents(text)))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.4321/journal.2025.42"
    assert selection.selected.raw == "10.4321/JOURNAL.2025.42"
    assert selection.selected.source_kind == "publication_region"
    assert selection.selected.region_type == "publication_metadata"
    assert (selection.selected.page, selection.selected.section_id) == (1, 2)
    assert selection.selected.semantic_context == "article_self"


@pytest.mark.parametrize("case", ["caption", "below_title", "other_page", "no_geometry"])
def test_cited_publication_table_is_not_article_identity(case):
    contents = _contents()
    table = contents.tables[0]
    if case == "caption":
        table.caption = "Table 1. Publications included in this review."
    elif case == "below_title":
        contents.region_summaries[0].bbox = (80, 500, 950, 650)
    elif case == "other_page":
        contents.region_summaries[0].page = 2
    else:
        contents.region_summaries[0].bbox = None

    assert collect_doi_candidates(contents) == ()


@pytest.mark.parametrize("case", ["unselected", "no_title_geometry", "other_record"])
def test_box_without_unambiguous_article_ownership_is_excluded(case):
    contents = _contents()
    resolution = contents.front_matter_resolution
    if case == "unselected":
        resolution = replace(resolution, selected_block_id=None)
    elif case == "no_title_geometry":
        resolution = replace(resolution, candidates=(replace(resolution.candidates[0], bbox=None),))
    else:
        other_title = replace(resolution.candidates[0], candidate_id="other-title")
        resolution = replace(
            resolution,
            candidates=(*resolution.candidates, other_title),
            blocks=(
                *resolution.blocks,
                FrontMatterBlock("other", ("other-title",), ("other-title",)),
            ),
        )
    contents.front_matter_resolution = resolution

    assert collect_doi_candidates(contents) == ()


@pytest.mark.parametrize(
    "text",
    [
        "DOI: 10.4321/JOURNAL.2025.42 ISSN 1234-567X",  # no publication furniture
        "Vol. 12(3) DOI: 10.4321/JOURNAL.2025.42",  # no serial identifier
        BOX + " https://doi.org/10.4321/cited",  # one label, two identifiers
        BOX.replace("DOI:", "Data DOI:"),
        BOX.replace("DOI:", "Reference DOI:"),
        BOX.replace("DOI:", "Table DOI:"),
        BOX.replace("DOI:", "Journal DOI:"),
        BOX.replace("DOI:", "JOURNAL DOI:"),
        BOX.replace("DOI:", "Journal\nDOI:"),
        BOX.replace("10.4321/JOURNAL.2025.42", "10.4321/JOURNAL.2025.42.g001"),
    ],
)
def test_box_keeps_existing_non_article_and_ambiguity_exclusions(text):
    assert select_doi_candidates(collect_doi_candidates(_contents(text))).selected is None


def test_conflicting_article_identity_is_not_overridden_by_box():
    contents = _contents()
    contents.sentences.append(PaperSentence(2, "Article DOI: 10.4321/other", 1, 1, page_number=1))

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    assert [issue.code for issue in selection.issues] == ["VAL_DOI_AMBIGUOUS"]


async def test_identity_stage_exports_recovered_box_doi_without_llm_identity():
    from bibr.pipeline.stages.export import _build_extraction
    from bibr.pipeline.stages.identity import IdentityValidationStage

    source = InputFile(path="paper.pdf")
    source.file_hash = "a" * 16
    paper = Paper(
        input_file=source, contents=_contents(), metadata=PaperMetadata(title="Paper", doi="")
    )
    state = FileState(path=Path("paper.pdf"), paper=paper)
    ctx = PipelineContext(
        file_states=[state],
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(no_llm=True),
        settings=Settings,
    )

    await IdentityValidationStage().run(ctx)
    paper.extraction = _build_extraction(ctx, paper, state)
    payload = paper.export_to_json()

    assert payload["metadata"]["doi"] == "10.4321/journal.2025.42"
    selected = payload["extraction"]["identity"]["receipt"]["selected"]
    assert selected["source_kind"] == "publication_region"
    assert selected["normalized"] == payload["metadata"]["doi"]


def test_dropped_publisher_box_retains_complete_source_identity_evidence():
    from bibr.extract.front_matter import resolve_front_matter
    from bibr.structure.pdf_parser import PDFParser

    html = f"<table><tr><td>{BOX}</td><td>Journal of Examples</td></tr></table>"
    parser = PDFParser(
        [
            [
                {
                    "index": 0,
                    "label": "table",
                    "content": html,
                    "_raw_ocr_content": html,
                    "bbox_2d": [80, 50, 950, 180],
                },
                {
                    "index": 1,
                    "label": "text",
                    "native_label": "doc_title",
                    "content": "Independent scientific article title",
                    "bbox_2d": [100, 250, 900, 320],
                },
                {
                    "index": 2,
                    "label": "text",
                    "content": "Alice Smith and Bob Jones",
                    "bbox_2d": [100, 340, 900, 380],
                },
            ]
        ]
    )
    contents = parser.parse()
    parser.apply_segmentation(contents, [[text] for text in parser.assembler.segmentable_texts])
    parser.create_content_sections(contents)
    contents.front_matter_resolution, _ = resolve_front_matter(contents)

    assert contents.tables == []
    selected = select_doi_candidates(collect_doi_candidates(contents)).selected
    assert selected is not None
    assert selected.normalized == "10.4321/journal.2025.42"


def test_single_cell_label_is_not_promoted_to_a_data_table():
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(
        [[{"label": "table", "content": "<table><tr><td>Supplementary material</td></tr></table>"}]]
    )

    assert parser.parse().tables == []


def test_truncated_region_summary_cannot_supply_identity():
    contents = _contents()
    contents.region_summaries[0].raw_ocr_content = None
    assert collect_doi_candidates(contents) == ()
