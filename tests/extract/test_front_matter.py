"""Front-matter ownership IR and multi-record abstention using synthetic records.

All titles, bylines, text, IDs, and geometry below are invented to exercise ownership boundaries."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import FrozenInstanceError
from importlib import import_module

import pytest

from bibr.models import PaperMetadata
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)
from bibr.pipeline.identity import ExpectedIdentity
from bibr.validation import ValidationIssue


def _front_matter_module():
    try:
        return import_module("bibr.extract.front_matter")
    except ModuleNotFoundError:
        pytest.fail("front-matter ownership IR has not been implemented")


def test_generated_section_label_is_not_rendered_as_printed_front_matter():
    from bibr.extract.core_metadata import render_block_context

    title = _section(1, "A study of forest disease", section_type=CanonicalSection.TITLE)
    abstract = _section(2, "Abstract", section_type=CanonicalSection.ABSTRACT)
    abstract.header_is_synthetic = True
    contents = _contents(
        [_sentence(1, "Opening prose about forest disease.", paragraph_id=1, section_id=2)],
        sections=[title, abstract],
        detected_title=title.header,
    )
    resolution, _ = _front_matter_module().resolve_front_matter(contents)
    text = render_block_context(resolution)
    assert "Opening prose about forest disease." in text
    assert "Abstract" not in text

    abstract.header_is_synthetic = False
    resolution, _ = _front_matter_module().resolve_front_matter(contents)
    assert "Abstract" in render_block_context(resolution)


def _section(
    section_id: int,
    header: str,
    *,
    section_type: CanonicalSection = CanonicalSection.UNKNOWN,
    bbox: tuple[float, float, float, float] | None = None,
) -> PaperSection:
    provenance = [Provenance(page_no=1, bbox=bbox)] if bbox is not None else []
    return PaperSection(
        section_id=section_id,
        header=header,
        level=0 if section_id == 0 else 1,
        parent_section_id=None,
        section_type=section_type,
        provenance=provenance,
    )


def _sentence(
    text_id: int,
    text: str,
    *,
    paragraph_id: int,
    section_id: int = 0,
    page: int = 1,
    bbox: tuple[float, float, float, float] | None = None,
    label: str = "text",
    font_size: float = 9.0,
    font_bold: bool = False,
) -> PaperSentence:
    return PaperSentence(
        text_id=text_id,
        text=text,
        section_id=section_id,
        paragraph_id=paragraph_id,
        page_number=page,
        provenance=[Provenance(page_no=page, bbox=bbox)] if bbox is not None else [],
        region_meta={
            "region_type": label,
            "font_size": font_size,
            "font_bold": font_bold,
        },
    )


def _contents(
    sentences: list[PaperSentence],
    *,
    sections: list[PaperSection] | None = None,
    detected_title: str | None = None,
    region_summaries: list[RegionSummary] | None = None,
    preparsed_metadata: PaperMetadata | None = None,
) -> PaperContents:
    sections = sections or [_section(0, "Root")]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={section.section_id: "" for section in sections},
        detected_title=detected_title,
        region_summaries=region_summaries or [],
        preparsed_metadata=preparsed_metadata,
    )


def _paragraph(
    text_id: int,
    text: str,
    *,
    paragraph_id: int,
    bbox: tuple[float, float, float, float],
    label: str = "text",
    font_size: float = 9.0,
    font_bold: bool = False,
    section_id: int = 0,
    page: int = 1,
) -> PaperSentence:
    return _sentence(
        text_id,
        text,
        paragraph_id=paragraph_id,
        section_id=section_id,
        page=page,
        bbox=bbox,
        label=label,
        font_size=font_size,
        font_bold=font_bold,
    )


def _single_candidate(text: str, *, source_kind: str = "paragraph"):
    module = _front_matter_module()
    if source_kind == "heading":
        contents = _contents(
            [],
            sections=[
                _section(0, "Root"),
                _section(1, text, bbox=(90.0, 170.0, 380.0, 205.0)),
            ],
        )
    else:
        contents = _contents(
            [
                _paragraph(
                    1,
                    text,
                    paragraph_id=1,
                    bbox=(90.0, 170.0, 380.0, 205.0),
                )
            ]
        )
    candidates = module.collect_front_matter_candidates(contents)
    assert len(candidates) == 1
    return candidates[0]


def test_candidate_types_are_immutable_and_expose_the_required_contract():
    module = _front_matter_module()
    candidate = module.FrontMatterCandidate(
        candidate_id="candidate-1",
        source_kind="paragraph",
        reading_order=0,
        page=1,
        bbox=(10.0, 20.0, 300.0, 40.0),
        region_label="text",
        font_size=12.0,
        font_bold=True,
        section_id=0,
        text_ids=(1,),
        paragraph_id=1,
        raw_text="A Grounded Title",
        normalized_text="a grounded title",
        roles=frozenset({"title"}),
    )
    block = module.FrontMatterBlock(
        block_id="block-1",
        candidate_ids=(candidate.candidate_id,),
        title_candidate_ids=(candidate.candidate_id,),
    )
    resolution = module.FrontMatterResolution(
        candidates=(candidate,),
        blocks=(block,),
        selected_block_id=block.block_id,
        selection_method="unique_block",
        reason_flags=(),
        allowed_text_ids=frozenset({1}),
        allowed_section_ids=frozenset({0}),
    )

    with pytest.raises(FrozenInstanceError):
        candidate.raw_text = "mutated"
    with pytest.raises(FrozenInstanceError):
        resolution.selected_block_id = None


def test_paragraph_aggregation_preserves_full_authoritative_source_and_geometry():
    module = _front_matter_module()
    first = "A complete source paragraph begins here and deliberately exceeds the preview limit. "
    second = "Its second sentence preserves the authoritative tail. " + "tail " * 40
    contents = _contents(
        [
            _sentence(
                7,
                first,
                paragraph_id=3,
                bbox=(101.0, 202.0, 801.0, 260.0),
                font_size=13.5,
                font_bold=True,
            ),
            _sentence(
                8,
                second,
                paragraph_id=3,
                bbox=(101.0, 202.0, 801.0, 260.0),
                font_size=13.5,
                font_bold=True,
            ),
        ],
        region_summaries=[
            RegionSummary(
                page=1,
                index=9,
                label="text",
                bbox=(101.0, 202.0, 801.0, 260.0),
                font_size=99.0,
                font_bold=False,
                section_id=0,
                content="WRONG 200-CHAR REGION PREVIEW",
                raw_ocr_content="WRONG RAW REGION SUMMARY",
            )
        ],
    )

    candidates = module.collect_front_matter_candidates(contents)
    paragraph = next(candidate for candidate in candidates if candidate.paragraph_id == 3)

    assert paragraph.raw_text == f"{first.strip()} {second.strip()}"
    assert len(paragraph.raw_text) > 200
    assert paragraph.text_ids == (7, 8)
    assert paragraph.page == 1
    assert paragraph.bbox == (101.0, 202.0, 801.0, 260.0)
    assert paragraph.region_label == "text"
    assert paragraph.font_size == 13.5
    assert paragraph.font_bold is True
    assert "WRONG" not in paragraph.raw_text


def test_heading_only_name_is_a_byline_candidate():
    module = _front_matter_module()
    contents = _contents(
        [],
        sections=[
            _section(0, "Root"),
            _section(1, "María de la Cruz", bbox=(90.0, 170.0, 380.0, 205.0)),
        ],
    )

    candidates = module.collect_front_matter_candidates(contents)
    candidate = next(item for item in candidates if item.section_id == 1)

    assert candidate.source_kind == "heading"
    assert candidate.raw_text == "María de la Cruz"
    assert candidate.text_ids == ()
    assert candidate.bbox == (90.0, 170.0, 380.0, 205.0)
    assert "byline" in candidate.roles


@pytest.mark.parametrize(
    ("section_type", "byline"),
    [
        # 10.1370/afm.22.s1.6174 -- abstract book, byline headed "Presenters".
        (
            CanonicalSection.ACKNOWLEDGMENT,
            "Fanor Balderrama, Lyn Sibley, PhD, Samira Jeimy, PhD, FRCPC, Michelle Cohen, MD",
        ),
        (
            CanonicalSection.AUTHOR_CONTRIBUTIONS,
            "María Isabel Mayorga Hernández, Juan Carlos Pérez Ruiz",
        ),
        (CanonicalSection.ENDNOTE, "Н. О. Садовникава, А. М. Мирзаахмедов"),
    ],
)
def test_first_page_byline_survives_a_mistyped_section(section_type, byline):
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(1, byline, paragraph_id=1, section_id=1, bbox=(90.0, 170.0, 380.0, 205.0)),
            _paragraph(
                2,
                "Body text that belongs to the same mistyped section.",
                paragraph_id=2,
                section_id=1,
                bbox=(90.0, 240.0, 380.0, 300.0),
            ),
        ],
        sections=[_section(0, "Root"), _section(1, "Presenters", section_type=section_type)],
    )

    candidates = module.collect_front_matter_candidates(contents)
    texts = [candidate.raw_text for candidate in candidates]

    assert byline in texts
    assert "Body text that belongs to the same mistyped section." not in texts


def test_mistyped_section_byline_is_ignored_off_the_first_page():
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(
                1,
                "Alice Example, Bob Sample, Carla Scholar",
                paragraph_id=1,
                section_id=1,
                page=4,
                bbox=(90.0, 170.0, 380.0, 205.0),
            ),
            _paragraph(2, "Front matter.", paragraph_id=2, bbox=(90.0, 60.0, 380.0, 90.0)),
        ],
        sections=[
            _section(0, "Root"),
            _section(1, "Author contributions", section_type=CanonicalSection.AUTHOR_CONTRIBUTIONS),
        ],
    )

    candidates = module.collect_front_matter_candidates(contents)

    assert all(candidate.section_id != 1 for candidate in candidates)


def test_first_page_byline_heading_survives_a_mistyped_section():
    module = _front_matter_module()
    contents = _contents(
        [_paragraph(1, "Аннотация.", paragraph_id=1, bbox=(90.0, 300.0, 380.0, 330.0))],
        sections=[
            _section(0, "Root"),
            _section(
                1,
                "Н. О. Садовникава",
                section_type=CanonicalSection.ENDNOTE,
                bbox=(90.0, 170.0, 380.0, 205.0),
            ),
            _section(
                2,
                "Материалы и методы",
                section_type=CanonicalSection.METHODS,
                bbox=(90.0, 400.0, 380.0, 430.0),
            ),
        ],
    )

    candidates = module.collect_front_matter_candidates(contents)
    headings = {
        candidate.raw_text for candidate in candidates if candidate.source_kind == "heading"
    }

    assert "Н. О. Садовникава" in headings
    assert "Материалы и методы" not in headings


def test_printed_byline_suppresses_first_page_probation():
    """10.30574/wjarr.2022.14.3.0574 -- a related-works citation is byline-shaped.

    The paper prints its own byline, so probation must not run at all: admitting
    the citation gave the body heading above it byline anatomy, which rooted a
    second record and made selection fail closed.
    """

    module = _front_matter_module()
    citation = "Arjun Aman, Aryan Singh, Ayush Raj and Sandeep Raj"
    contents = _contents(
        [
            _paragraph(
                1,
                "Divya E, Jaishreenithi V, Keerthika S and S Yamuna",
                paragraph_id=1,
                bbox=(90.0, 150.0, 380.0, 180.0),
            ),
            _paragraph(
                2,
                "Abstract. Humans do require a lot of communication.",
                paragraph_id=2,
                bbox=(90.0, 200.0, 380.0, 260.0),
            ),
            _paragraph(
                3,
                citation,
                paragraph_id=3,
                section_id=2,
                bbox=(90.0, 600.0, 380.0, 630.0),
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                1,
                "1.1.1. An Efficient Bar/QR Code Recognition System",
                section_type=CanonicalSection.TITLE,
                bbox=(90.0, 560.0, 380.0, 590.0),
            ),
            _section(
                2,
                "Related works",
                section_type=CanonicalSection.AUTHOR_CONTRIBUTIONS,
                bbox=(90.0, 520.0, 380.0, 550.0),
            ),
        ],
    )

    candidates = module.collect_front_matter_candidates(contents)
    blocks = module.group_front_matter_blocks(candidates)

    assert citation not in [candidate.raw_text for candidate in candidates]
    assert len(blocks) == 1


def test_probation_row_never_roots_a_record():
    """Even when probation does fire, its rows carry no record anatomy."""

    module = _front_matter_module()
    byline = "María Isabel Mayorga Hernández, Juan Carlos Pérez Ruiz"
    contents = _contents(
        [
            _paragraph(
                1,
                byline,
                paragraph_id=1,
                section_id=2,
                bbox=(90.0, 600.0, 380.0, 630.0),
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                1,
                "1.1.1. An Efficient Bar/QR Code Recognition System",
                section_type=CanonicalSection.TITLE,
                bbox=(90.0, 560.0, 380.0, 590.0),
            ),
            _section(
                2,
                "Author contributions",
                section_type=CanonicalSection.AUTHOR_CONTRIBUTIONS,
                bbox=(90.0, 520.0, 380.0, 550.0),
            ),
        ],
    )

    candidates = module.collect_front_matter_candidates(contents)
    blocks = module.group_front_matter_blocks(candidates)
    probation = [
        candidate for candidate in candidates if module.BYLINE_PROBATION_ROLE in candidate.roles
    ]

    assert [candidate.raw_text for candidate in probation] == [byline]
    assert len(blocks) == 1


@pytest.mark.parametrize(
    "text",
    [
        "Mahdieh Khorsandifard1 · Kian Jafari1 · Arash Sheikhaleh1",
        "Alice Example; Bob Sample; Carla Scholar; David Researcher; "
        "Elena Author; Farid Scientist; Grace Analyst; Hugo Writer",
        "Smith JA · Doe BB",
        "J.A. Smith · B.B. Doe",
        "Dr Alice Example · Prof Bob Sample",
        "Alice O'Neill · Bob D'Arcy",
        "María-José O'Neill · Jean-Luc D'Arcy",
    ],
)
def test_bare_separator_person_shapes_are_precision_abstentions(text):
    candidate = _single_candidate(text, source_kind="paragraph")
    assert "byline" not in candidate.roles


@pytest.mark.parametrize(
    "text",
    [
        "Mahdieh Khorsandifard1 · Kian Jafari1 · Arash Sheikhaleh1",
        "Alice Example; Bob Sample; Carla Scholar; David Researcher; "
        "Elena Author; Farid Scientist; Grace Analyst; Hugo Writer",
        "Alice Example · Bob Sample",
        "Smith JA · Doe BB",
        "J.A. Smith · B.B. Doe",
        "Dr Alice Example · Prof Bob Sample",
        "Alice O'Neill · Bob D'Arcy",
        "María-José O'Neill · Jean-Luc D'Arcy",
    ],
)
def test_trusted_title_list_affiliation_sequence_promotes_contextual_byline(text):
    module = _front_matter_module()
    title = "Grounded Study of Community Health"
    contents = _contents(
        [
            _paragraph(
                1,
                title,
                paragraph_id=1,
                bbox=(80.0, 100.0, 720.0, 140.0),
                label="doc_title",
            ),
            _paragraph(
                2,
                text,
                paragraph_id=2,
                bbox=(100.0, 150.0, 700.0, 185.0),
            ),
            _paragraph(
                3,
                "Department of Biology, Example University",
                paragraph_id=3,
                bbox=(90.0, 195.0, 710.0, 225.0),
            ),
        ],
        detected_title=title,
    )

    collected = module.collect_front_matter_candidates(contents)
    collected_byline = next(candidate for candidate in collected if candidate.text_ids == (2,))
    assert "byline" not in collected_byline.roles

    resolution, issues = module.resolve_front_matter(contents)
    byline = next(candidate for candidate in resolution.candidates if candidate.text_ids == (2,))

    assert issues == ()
    assert resolution.selected_block_id is not None
    assert "byline" in byline.roles


@pytest.mark.parametrize(
    "text",
    [
        "Alice Example · Bob Sample",
        "Alice Example; Bob Sample",
        "Alice Example · Bob Sample · Carla Scholar",
    ],
)
def test_short_plain_title_case_lists_are_precision_abstentions(text):
    candidate = _single_candidate(text, source_kind="paragraph")
    assert "byline" not in candidate.roles


def test_contextual_promotion_preserves_mixed_byline_affiliation():
    module = _front_matter_module()
    title = "Grounded Study of Community Health"
    contents = _contents(
        [
            _paragraph(
                1,
                title,
                paragraph_id=1,
                bbox=(80.0, 100.0, 720.0, 140.0),
                label="doc_title",
            ),
            _paragraph(
                2,
                "Alice Example · Bob Sample · Department of Biology, Example University",
                paragraph_id=2,
                bbox=(90.0, 150.0, 710.0, 190.0),
            ),
        ],
        detected_title=title,
    )

    collected = module.collect_front_matter_candidates(contents)
    assert next(candidate for candidate in collected if candidate.text_ids == (2,)).roles == (
        frozenset({"affiliation"})
    )

    resolution, issues = module.resolve_front_matter(contents)
    mixed = next(candidate for candidate in resolution.candidates if candidate.text_ids == (2,))

    assert issues == ()
    assert mixed.roles == frozenset({"byline", "affiliation"})


@pytest.mark.parametrize(
    "text",
    [
        "Conceptualization · methodology · formal analysis · writing and review",
        "Department of Medicine · Example University · Amsterdam, Netherlands",
        "Smith J. A long article title. Journal Name 2024; 12: 1-8.",
        "This study examined education · health · income · and employment outcomes.",
        "Materials and Methods · Results and Discussion",
        "Public Health · Environmental Policy",
        "Amsterdam Netherlands · London United Kingdom",
        "Smith JA · Doe BB · Journal Name (2024) 12:e12345",
        "Resources: Alice Example · Resources: Bob Sample",
        "Study Design · Statistical Analysis",
        "Clinical Medicine · Health Economics",
        "Boston Massachusetts · Toronto Canada",
        "Santé Publique · Économie Sociale",
        "Educación Médica · Política Pública",
        "Économie de la Santé · Política de la Salud",
        "Evidence-Based Medicine · Patient-Centered Care",
        "WHO Guidance · OECD Report",
        "Phase2 Study · Model3 Evaluation",
        "Children’s Health · Women’s Studies",
        "U.S. Policy · U.K. Guidance",
        "Clinical WHO · Economic OECD",
        "Model Version2 · Study Phase3",
        "Public Health · Environmental Policy · Clinical Medicine · Health Economics",
        "Boston Massachusetts · Toronto Canada · London England · Paris France",
        "Santé Publique · Économie Sociale · Educación Médica · Política Pública",
        "WHO Guidance · OECD Report · UNESCO Policy · UNICEF Strategy",
        "Evidence-Based Medicine · Patient-Centered Care · Community-Based Research · "
        "Value-Based Health",
        "Phase2 Study · Model3 Evaluation · Cohort4 Analysis · Trial5 Results",
        "Smith JA · Doe BB · Brown CC · Journal Name",
        "Topic One; Topic Two; Topic Three; Topic Four; Topic Five; Topic Six; "
        "Topic Seven; Topic Eight",
        "Smith JA · Doe BB · Nature Medicine 2024",
        "Smith JA · Doe BB · Nature Medicine 12:e12345",
    ],
)
def test_separator_rich_prose_affiliations_and_references_are_not_bylines(text):
    candidate = _single_candidate(text, source_kind="paragraph")
    assert "byline" not in candidate.roles


@pytest.mark.parametrize(
    ("detected_title", "intervening", "affiliation_page", "affiliation_bbox"),
    [
        (None, (), 1, (90.0, 195.0, 710.0, 225.0)),
        (
            "Grounded Study of Community Health",
            ("Conference Track",),
            1,
            (90.0, 235.0, 710.0, 265.0),
        ),
        ("Grounded Study of Community Health", (), 2, (90.0, 195.0, 710.0, 225.0)),
        ("Grounded Study of Community Health", (), 1, (760.0, 195.0, 960.0, 225.0)),
    ],
)
def test_contextual_separator_promotion_requires_trusted_adjacent_compatible_record(
    detected_title,
    intervening,
    affiliation_page,
    affiliation_bbox,
):
    module = _front_matter_module()
    title = "Grounded Study of Community Health"
    rows = [
        _paragraph(
            1,
            title,
            paragraph_id=1,
            bbox=(80.0, 100.0, 720.0, 140.0),
            label="text" if detected_title is None else "doc_title",
        )
    ]
    rows.extend(
        _paragraph(
            index + 2,
            text,
            paragraph_id=index + 2,
            bbox=(100.0, 150.0 + index * 40.0, 700.0, 185.0 + index * 40.0),
        )
        for index, text in enumerate(intervening)
    )
    byline_id = len(rows) + 1
    rows.append(
        _paragraph(
            byline_id,
            "Alice Example · Bob Sample",
            paragraph_id=byline_id,
            bbox=(100.0, 150.0 + len(intervening) * 40.0, 700.0, 185.0 + len(intervening) * 40.0),
        )
    )
    affiliation_id = len(rows) + 1
    rows.append(
        _paragraph(
            affiliation_id,
            "Department of Biology, Example University",
            paragraph_id=affiliation_id,
            bbox=affiliation_bbox,
            page=affiliation_page,
        )
    )

    resolution, issues = module.resolve_front_matter(_contents(rows, detected_title=detected_title))
    candidate = next(
        candidate for candidate in resolution.candidates if candidate.text_ids == (byline_id,)
    )

    assert issues == ()
    assert "byline" not in candidate.roles


def test_generic_paragraph_title_does_not_anchor_contextual_separator_byline():
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(
                1,
                "Internal Topic Heading Overview",
                paragraph_id=1,
                bbox=(80.0, 100.0, 720.0, 140.0),
                label="paragraph_title",
            ),
            _paragraph(
                2,
                "Alice Example · Bob Sample",
                paragraph_id=2,
                bbox=(100.0, 150.0, 700.0, 185.0),
            ),
            _paragraph(
                3,
                "Department of Biology, Example University",
                paragraph_id=3,
                bbox=(90.0, 195.0, 710.0, 225.0),
            ),
        ]
    )

    resolution, issues = module.resolve_front_matter(contents)
    candidate = next(candidate for candidate in resolution.candidates if candidate.text_ids == (2,))

    assert issues == ()
    assert "byline" not in candidate.roles


@pytest.mark.parametrize("boundary_row", ["list", "affiliation"])
def test_contextual_separator_promotion_does_not_cross_abstract_boundary(boundary_row):
    module = _front_matter_module()
    title = "Grounded Study of Community Health"
    contents = _contents(
        [
            _paragraph(
                1,
                title,
                paragraph_id=1,
                bbox=(80.0, 100.0, 720.0, 140.0),
                label="doc_title",
            ),
            _paragraph(
                2,
                "Alice Example · Bob Sample",
                paragraph_id=2,
                bbox=(100.0, 150.0, 700.0, 185.0),
                label="abstract" if boundary_row == "list" else "text",
            ),
            _paragraph(
                3,
                "Department of Biology, Example University",
                paragraph_id=3,
                bbox=(90.0, 195.0, 710.0, 225.0),
                label="abstract" if boundary_row == "affiliation" else "text",
            ),
        ],
        detected_title=title,
    )

    resolution, issues = module.resolve_front_matter(contents)
    candidate = next(candidate for candidate in resolution.candidates if candidate.text_ids == (2,))

    assert issues == ()
    assert "byline" not in candidate.roles


def test_bare_separator_false_positive_cannot_manufacture_record_ownership():
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(
                1,
                "First Grounded Study",
                paragraph_id=1,
                bbox=(60.0, 100.0, 450.0, 140.0),
                label="doc_title",
            ),
            _paragraph(
                2,
                "Alice Example, Bob Sample",
                paragraph_id=2,
                bbox=(70.0, 150.0, 440.0, 180.0),
            ),
            _paragraph(
                3,
                "Department of Biology, Example University",
                paragraph_id=3,
                bbox=(70.0, 190.0, 440.0, 220.0),
            ),
            _paragraph(
                4,
                "Second Grounded Study",
                paragraph_id=4,
                bbox=(540.0, 100.0, 930.0, 140.0),
                label="paragraph_title",
            ),
            _paragraph(
                5,
                "Public Health · Environmental Policy · Clinical Medicine · Health Economics",
                paragraph_id=5,
                bbox=(550.0, 150.0, 920.0, 180.0),
            ),
            _paragraph(
                6,
                "Department of Medicine, Example University",
                paragraph_id=6,
                bbox=(550.0, 190.0, 920.0, 220.0),
            ),
        ],
        detected_title="First Grounded Study",
    )

    collected = module.collect_front_matter_candidates(contents)
    topic = next(candidate for candidate in collected if candidate.text_ids == (5,))
    blocks = module.group_front_matter_blocks(collected)
    resolution, issues = module.resolve_front_matter(contents)

    assert "byline" not in topic.roles
    assert len(blocks) == 1
    assert len(resolution.blocks) == 1
    assert resolution.selected_block_id == resolution.blocks[0].block_id
    assert issues == ()


def _trusted_sandwich_contents(middle_rows: list[str], *, title_label: str = "doc_title"):
    """Trusted title directly above ``middle_rows`` directly above an affiliation."""

    title = "Grounded Study of Community Health"
    rows = [
        _paragraph(
            1,
            title,
            paragraph_id=1,
            bbox=(80.0, 100.0, 720.0, 140.0),
            label=title_label,
        )
    ]
    rows.extend(
        _paragraph(
            index + 2,
            text,
            paragraph_id=index + 2,
            bbox=(100.0, 150.0 + index * 40.0, 700.0, 185.0 + index * 40.0),
        )
        for index, text in enumerate(middle_rows)
    )
    affiliation_id = len(rows) + 1
    rows.append(
        _paragraph(
            affiliation_id,
            "Department of Biology, Example University",
            paragraph_id=affiliation_id,
            bbox=(90.0, 150.0 + len(middle_rows) * 40.0, 710.0, 185.0 + len(middle_rows) * 40.0),
        )
    )
    return _contents(rows, detected_title=title)


@pytest.mark.parametrize(
    "text",
    [
        "Original Research · Open Access",
        "Research Article · Check for Updates",
        "Journal of Community Health · Springer Nature",
        "Program Abstracts · Annual Scientific Meeting",
        "History of Science · Philosophy of Mind",
    ],
)
def test_furniture_and_masthead_chunks_never_promote_to_contextual_byline(text):
    module = _front_matter_module()
    resolution, issues = module.resolve_front_matter(_trusted_sandwich_contents([text]))
    candidate = next(candidate for candidate in resolution.candidates if candidate.text_ids == (2,))

    assert issues == ()
    assert resolution.selected_block_id is not None
    assert "byline" not in candidate.roles


def test_wrapped_separator_byline_promotes_every_run_row():
    # 10.20448-class shape whose middle-dot author list wraps across two OCR
    # rows before the affiliation block.
    module = _front_matter_module()
    contents = _trusted_sandwich_contents(
        [
            "Mahdieh Khorsandifard1 · Kian Jafari1 · Arash Sheikhaleh1",
            "Sara Moradi2 · Ali Rezaei2",
        ]
    )

    resolution, issues = module.resolve_front_matter(contents)
    first = next(candidate for candidate in resolution.candidates if candidate.text_ids == (2,))
    second = next(candidate for candidate in resolution.candidates if candidate.text_ids == (3,))

    assert issues == ()
    assert "byline" in first.roles
    assert "byline" in second.roles


def test_promotion_requires_affiliation_anchor_after_name_list():
    # The affiliation anchor is the ONLY failing condition here: trusted title
    # above, compatible geometry, valid list shape, but a plain prose row after
    # the list instead of an affiliation.
    module = _front_matter_module()
    title = "Grounded Study of Community Health"
    contents = _contents(
        [
            _paragraph(
                1,
                title,
                paragraph_id=1,
                bbox=(80.0, 100.0, 720.0, 140.0),
                label="doc_title",
            ),
            _paragraph(
                2,
                "Alice Example · Bob Sample",
                paragraph_id=2,
                bbox=(100.0, 150.0, 700.0, 185.0),
            ),
            _paragraph(
                3,
                "Accepted after peer revision on request",
                paragraph_id=3,
                bbox=(90.0, 195.0, 710.0, 225.0),
            ),
        ],
        detected_title=title,
    )

    resolution, issues = module.resolve_front_matter(contents)
    candidate = next(candidate for candidate in resolution.candidates if candidate.text_ids == (2,))

    assert issues == ()
    assert "byline" not in candidate.roles


def test_detected_title_match_anchors_promotion_without_doc_title_label():
    module = _front_matter_module()
    contents = _trusted_sandwich_contents(
        ["Alice Example · Bob Sample"],
        title_label="text",
    )

    resolution, issues = module.resolve_front_matter(contents)
    candidate = next(candidate for candidate in resolution.candidates if candidate.text_ids == (2,))

    assert issues == ()
    assert "byline" in candidate.roles


def test_plain_topic_pair_promotion_is_a_documented_residual():
    # "Clinical Medicine · Health Economics" is lexically indistinguishable
    # from "Alice Example · Bob Sample": no function words, furniture labels,
    # masthead phrases, or prohibited evidence. Inside a trusted
    # title/affiliation sandwich the shape therefore promotes. This pins the
    # accepted residual so any future lexical repair flips it visibly; it is
    # not an endorsement of the promotion.
    module = _front_matter_module()
    resolution, issues = module.resolve_front_matter(
        _trusted_sandwich_contents(["Clinical Medicine · Health Economics"])
    )
    candidate = next(candidate for candidate in resolution.candidates if candidate.text_ids == (2,))

    assert issues == ()
    assert "byline" in candidate.roles


@pytest.mark.parametrize(
    "role",
    [
        "Conceptualization",
        "Data curation",
        "Formal analysis",
        "Funding acquisition",
        "Investigation",
        "Methodology",
        "Project administration",
        "Resources",
        "Software",
        "Supervision",
        "Validation",
        "Visualization",
        "Writing – original draft",
        "Writing – review and editing",
    ],
)
def test_credit_role_annotations_are_not_bylines(role):
    candidate = _single_candidate(
        f"{role}: Alice Example · {role}: Bob Sample",
        source_kind="paragraph",
    )
    assert "byline" not in candidate.roles


def test_two_composite_byline_affiliation_records_do_not_collapse_to_unique_selection():
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(
                1,
                "First Study of Community Health",
                paragraph_id=1,
                bbox=(60.0, 100.0, 450.0, 145.0),
                label="paragraph_title",
            ),
            _paragraph(
                2,
                "Alice Example, Bob Sample; Department of Biology, Example University",
                paragraph_id=2,
                bbox=(60.0, 150.0, 450.0, 190.0),
            ),
            _paragraph(
                3,
                "Second Study of Community Health",
                paragraph_id=3,
                bbox=(540.0, 100.0, 930.0, 145.0),
                label="paragraph_title",
            ),
            _paragraph(
                4,
                "Carla Scholar, David Researcher; Institute of Medicine, Example University",
                paragraph_id=4,
                bbox=(540.0, 150.0, 930.0, 190.0),
            ),
        ]
    )

    resolution, issues = module.resolve_front_matter(contents, target_required=True)

    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id is None
    assert resolution.selection_method == "abstained"
    assert "multiple_plausible_blocks" in resolution.reason_flags
    assert len(issues) == 1
    assert issues[0].code == "VAL_METADATA_MULTI_ITEM"
    assert issues[0].blocking is True


def test_heading_order_falls_back_to_section_structure_without_region_summaries():
    module = _front_matter_module()
    contents = _contents(
        [
            _sentence(
                10,
                "The first record abstract.",
                paragraph_id=1,
                section_id=2,
                label="abstract",
            ),
            _sentence(
                20,
                "The second record abstract.",
                paragraph_id=2,
                section_id=3,
                label="abstract",
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(1, "First Grounded Article Title", section_type=CanonicalSection.TITLE),
            _section(2, "María de la Cruz"),
            _section(3, "SECOND GROUNDED ARTICLE TITLE"),
        ],
        detected_title="First Grounded Article Title",
    )
    expected = ExpectedIdentity(
        queue_record_id="docx-heading-order",
        expected_title="Second Grounded Article Title",
    )

    resolution, issues = module.resolve_front_matter(
        contents,
        expected_identity=expected,
        target_required=True,
    )

    assert resolution.selected_block_id == "front-matter-block-2"
    assert resolution.allowed_text_ids == frozenset({20})
    assert issues == ()


# Synthetic proceedings page with three neighboring records sharing a section.
def _proceedings_composite_contents() -> PaperContents:
    # The long target abstract spans several source rows but shares one paragraph.
    target_rows = (
        "WHAT DO STUDENTS LEARN ABOUT RIVER SYSTEMS? AN ANALYSIS OF INTRODUCTORY SCIENCE TEXTBOOKS Morgan Sample, Avery Reader, and Riley Example 1. Example University, Example City, United States, 2. Sample Institute, Sample Town, United States",
        "Introductory science courses provide many students with their first sustained discussion of how rivers connect landscapes, ecosystems, and communities, but textbooks can organize these explanations in several different ways.",
        "This fictional classroom investigation compares the language used to describe river systems in a set of invented teaching materials assembled only to exercise document parsing and ownership boundaries.",
        "The example coding scheme records whether each passage introduces a process, gives a concrete illustration, describes an observation, or asks readers to connect two ideas in their own words.",
        "A passage can receive several codes because a single paragraph may explain an observation while also giving the learner a new question to consider during a later classroom activity.",
        "The materials are grouped by intended reading level so that the analysis can distinguish differences in vocabulary from differences in the concepts selected for discussion by their fictional authors.",
        "Several example chapters begin with changes in water flow, whereas others first discuss the movement of sediment before introducing the relationship between those two kinds of change.",
        "Another group of example passages asks students to compare drawings of river channels across seasons and describe which features remain stable and which features are expected to change.",
        "The exercise also records whether illustrations provide labels that can be understood independently or require the learner to consult surrounding prose before interpreting the drawing.",
        "This distinction is included because a caption and its associated paragraph may be separated when the same material is displayed in a different page layout or reading interface.",
        "The fictional comparison does not estimate a population effect and is not intended to support conclusions about any real textbook, school, publisher, or group of learners.",
        "Its purpose is to supply a sufficiently long abstract with complete sentences that a parser must keep with the correct title even when neighboring records share the same section identifier.",
        "A robust ownership decision should therefore preserve all of these rows while excluding the separate title, author list, and abstract printed in the adjacent column of the same page.",
        "The example authors use consistent terminology throughout the abstract so that shortening a region preview cannot silently remove later sentences from the selected record.",
        "Future classroom exercises could ask learners to design their own diagrams and explain how the arrangement of labels changes the interpretation of an otherwise identical illustration.",
        "The invented teaching materials demonstrate a complete record whose final sentence must remain available after title matching, and their fictional authors could be guided in this effort.",
    )
    summaries = [
        RegionSummary(
            page=1,
            index=index,
            label=label,
            bbox=bbox,
            section_id=1,
            content=preview,
        )
        for index, label, bbox, preview in (
            (2, "text", (80.0, 230.0, 470.0, 320.0), "TEACHING FIELD SCIENCE"),
            (3, "abstract", (80.0, 325.0, 490.0, 685.0), "Background: Although"),
            (4, "paragraph_title", (85.0, 700.0, 465.0, 735.0), "THREE STEPS PROVIDE"),
            (5, "text", (85.0, 740.0, 400.0, 770.0), "Robin Example"),
            (8, "abstract", (80.0, 825.0, 490.0, 920.0), "The Example Learning"),
            (10, "text", (510.0, 375.0, 900.0, 475.0), "WHAT DO STUDENTS"),
            # RegionSummary content is intentionally just the 200-character
            # preview; the second authoritative sentence remains much longer.
            (11, "text", (510.0, 485.0, 915.0, 920.0), target_rows[1][:200]),
        )
    ]
    return _contents(
        [
            _paragraph(
                30,
                "TEACHING FIELD SCIENCE: IS STRUCTURED WRITTEN FEEDBACK EFFECTIVE FOR LECTURES "
                "Taylor Sample, Jordan Reader, Casey Example, and Quinn Writer",
                paragraph_id=1,
                bbox=(80.0, 230.0, 470.0, 320.0),
                font_size=4.0,
                section_id=1,
            ),
            _paragraph(
                31,
                "Background: Classroom activities can be organized in several different ways, "
                "and each approach can guide learners toward a different kind of question.",
                paragraph_id=2,
                bbox=(80.0, 325.0, 490.0, 685.0),
                label="abstract",
                font_size=4.0,
                section_id=1,
            ),
            _paragraph(
                32,
                "THREE STEPS PROVIDE A STRUCTURAL FRAMEWORK FOR ORGANIZING EDUCATIONAL MATERIALS",
                paragraph_id=3,
                bbox=(85.0, 700.0, 465.0, 735.0),
                label="paragraph_title",
                font_size=6.5,
                font_bold=True,
                section_id=1,
            ),
            _paragraph(
                33,
                "Robin Example, Jamie Reader, Alex Sample, Morgan Writer",
                paragraph_id=4,
                bbox=(85.0, 740.0, 400.0, 770.0),
                font_size=4.0,
                section_id=1,
            ),
            _paragraph(
                34,
                "The Example Learning Foundation and the Institute for Classroom Learning "
                "provide a framework for helping system leaders.",
                paragraph_id=5,
                bbox=(80.0, 825.0, 490.0, 920.0),
                label="abstract",
                font_size=4.0,
                section_id=1,
            ),
            *[
                _paragraph(
                    text_id,
                    text,
                    paragraph_id=6,
                    bbox=(
                        (510.0, 375.0, 900.0, 475.0)
                        if text_id == 38
                        else (510.0, 485.0, 915.0, 920.0)
                    ),
                    label="text",
                    font_size=4.0,
                    section_id=1,
                )
                for text_id, text in zip(range(38, 54), target_rows, strict=True)
            ],
        ],
        sections=[
            _section(0, "Root"),
            _section(1, "Abstract", section_type=CanonicalSection.ABSTRACT),
        ],
        region_summaries=summaries,
    )


def test_proceedings_composite_splits_repeated_title_seeds_and_abstains_when_required():
    module = _front_matter_module()
    from bibr.pipeline.artifacts import ArtifactDisposition, disposition_for_issues

    resolution, issues = module.resolve_front_matter(
        _proceedings_composite_contents(),
        target_required=True,
    )

    assert len(resolution.blocks) == 3
    assert resolution.selected_block_id is None
    assert resolution.allowed_text_ids == frozenset()
    assert "multiple_plausible_blocks" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]
    assert issues[0].blocking is True
    assert disposition_for_issues(issues) is ArtifactDisposition.BLOCKED


def test_proceedings_composite_expected_title_selects_only_the_matching_record():
    module = _front_matter_module()
    expected = ExpectedIdentity(
        queue_record_id="synthetic-proceedings",
        expected_title=(
            "What Do Students Learn About River Systems? "
            "An Analysis of Introductory Science Textbooks"
        ),
        doi_required=True,
    )

    resolution, issues = module.resolve_front_matter(
        _proceedings_composite_contents(),
        expected_identity=expected,
        target_required=True,
    )

    assert resolution.selected_block_id == "front-matter-block-3"
    assert resolution.selection_method == "expected_title"
    assert resolution.allowed_text_ids == frozenset(range(38, 54))
    assert not resolution.allowed_text_ids.intersection({30, 31, 32, 33, 34})
    target_candidate = next(
        candidate
        for candidate in resolution.candidates
        if candidate.text_ids == tuple(range(38, 54))
    )
    assert len(target_candidate.raw_text) > 1_800
    assert "guided in this effort" in target_candidate.raw_text
    # All three proceedings records deliberately reuse section 1. Section IDs
    # alone cannot enforce ownership; consumers must intersect allowed text IDs.
    assert resolution.allowed_section_ids == frozenset({1})
    assert issues == ()


# Synthetic boundary control with an adjacent record and invented geometry.
def _synthetic_proceedings_multi_record_contents() -> PaperContents:
    return _contents(
        [
            _paragraph(
                1,
                "PROGRAM ABSTRACTS FROM THE EXAMPLE 2024 RESEARCH MEETING",
                paragraph_id=1,
                bbox=(60.0, 35.0, 930.0, 70.0),
                label="header",
                font_size=6.0,
                font_bold=True,
            ),
            _paragraph(
                2,
                "Community Gardens and Public Spaces",
                paragraph_id=2,
                bbox=(70.0, 90.0, 470.0, 120.0),
                label="paragraph_title",
                font_size=7.0,
                font_bold=True,
            ),
            _paragraph(
                3,
                "COMMUNITY GARDENING AND THE EXPERIENCE OF SHARED PUBLIC SPACES",
                paragraph_id=3,
                bbox=(70.0, 140.0, 470.0, 185.0),
                font_size=6.5,
                font_bold=True,
            ),
            _paragraph(
                4,
                "Morgan Example, Example University",
                paragraph_id=4,
                bbox=(70.0, 190.0, 470.0, 225.0),
            ),
            _paragraph(
                5,
                "This research compares community gardens with other shared public spaces, "
                "using observations and interviews about local participation.",
                paragraph_id=5,
                bbox=(70.0, 230.0, 470.0, 520.0),
                label="abstract",
            ),
            _paragraph(
                6,
                "SOCIAL CONNECTIONS AND VOLUNTEERING AMONG OLDER ADULTS",
                paragraph_id=6,
                bbox=(520.0, 140.0, 920.0, 185.0),
                font_size=6.5,
                font_bold=True,
            ),
            _paragraph(
                7,
                "Mei Lin and Jordan Smith",
                paragraph_id=7,
                bbox=(520.0, 190.0, 920.0, 225.0),
            ),
            _paragraph(
                8,
                "We examined social connections and volunteering in later life.",
                paragraph_id=8,
                bbox=(520.0, 230.0, 920.0, 520.0),
                label="abstract",
            ),
        ]
    )


def test_synthetic_proceedings_heading_does_not_create_an_extra_record_block():
    module = _front_matter_module()
    expected = ExpectedIdentity(
        queue_record_id="synthetic-proceedings",
        expected_title=("Community Gardening and the Experience of Shared Public Spaces"),
        doi_required=True,
    )

    resolution, issues = module.resolve_front_matter(
        _synthetic_proceedings_multi_record_contents(),
        expected_identity=expected,
        target_required=True,
    )

    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id == "front-matter-block-1"
    assert resolution.selection_method == "expected_title"
    assert frozenset({3, 4, 5}).issubset(resolution.allowed_text_ids)
    assert 6 not in resolution.allowed_text_ids
    assert issues == ()


def _native_target_contents() -> PaperContents:
    """Synthetic native text with target IDs and no OCR geometry."""

    rows = (
        "Community Gardening and the Experience of Shared Public Spaces",
        "Morgan Example",
        "Example University",
        "This fictional study describes how neighbors organize activities in a shared garden and explain their choices about the use of common space.",
        "Participants describe the same outdoor area in different ways depending on which activities they expect to organize there.",
        "Those interpretations influence the questions they raise during a planning meeting.",
        "Some example groups focus on collective events while others prefer individual activities.",
        "The comparison considers how the groups describe these preferences without assigning a correct choice.",
        "The invented record supplies native text with stable ownership identifiers and no OCR geometry.",
    )
    return _contents(
        [
            _sentence(
                text_id,
                text,
                paragraph_id=text_id,
                label=(
                    "paragraph_title" if text_id == 37 else "abstract" if text_id >= 40 else "text"
                ),
            )
            for text_id, text in zip(range(37, 46), rows, strict=True)
        ]
    )


def test_native_ownership_is_limited_to_target_ids():
    module = _front_matter_module()
    expected = ExpectedIdentity(
        queue_record_id="synthetic-native-target",
        expected_title=("Community Gardening and the Experience of Shared Public Spaces"),
    )

    resolution, issues = module.resolve_front_matter(
        _native_target_contents(), expected_identity=expected
    )

    assert resolution.selected_block_id == "front-matter-block-1"
    assert resolution.allowed_text_ids == frozenset(range(37, 46))
    assert all(candidate.bbox is None for candidate in resolution.candidates)
    assert issues == ()


def test_ordinary_within_record_headings_do_not_split_the_block():
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(
                1,
                "A Spatially Grounded Study of Healthy Aging",
                paragraph_id=1,
                bbox=(100.0, 90.0, 900.0, 140.0),
                label="doc_title",
                font_size=18.0,
                font_bold=True,
            ),
            _paragraph(
                2,
                "Ada Lovelace and Grace Hopper",
                paragraph_id=2,
                bbox=(100.0, 150.0, 900.0, 180.0),
            ),
            _paragraph(
                3,
                "Background",
                paragraph_id=3,
                bbox=(100.0, 200.0, 300.0, 225.0),
                label="paragraph_title",
                font_bold=True,
            ),
            _paragraph(
                4,
                "BACKGROUND AND OBJECTIVES OF THIS STUDY: healthy aging is shaped by social and "
                "spatial "
                "conditions.",
                paragraph_id=4,
                bbox=(100.0, 230.0, 900.0, 330.0),
                label="abstract",
            ),
            _paragraph(
                5,
                "Methods",
                paragraph_id=5,
                bbox=(100.0, 350.0, 300.0, 375.0),
                label="paragraph_title",
                font_bold=True,
            ),
        ]
    )

    resolution, issues = module.resolve_front_matter(contents, target_required=True)

    assert len(resolution.blocks) == 1
    assert resolution.selected_block_id == "front-matter-block-1"
    assert resolution.selection_method == "unique_block"
    assert issues == ()


def test_unique_expected_doi_and_coordinate_hint_select_a_single_block():
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(
                1,
                "FIRST GROUNDED ARTICLE TITLE",
                paragraph_id=1,
                bbox=(60.0, 100.0, 450.0, 145.0),
                font_bold=True,
            ),
            _paragraph(
                2,
                "Article DOI: 10.1234/first.record",
                paragraph_id=2,
                bbox=(60.0, 150.0, 450.0, 180.0),
            ),
            _paragraph(
                3,
                "SECOND GROUNDED ARTICLE TITLE",
                paragraph_id=3,
                bbox=(540.0, 100.0, 930.0, 145.0),
                font_bold=True,
            ),
            _paragraph(
                4,
                "Article DOI: 10.1234/second.record",
                paragraph_id=4,
                bbox=(540.0, 150.0, 930.0, 180.0),
            ),
        ]
    )
    doi_expected = ExpectedIdentity(
        queue_record_id="doi-target",
        expected_doi="10.1234/second.record",
        doi_required=True,
    )
    coordinate_expected = ExpectedIdentity(
        queue_record_id="coordinate-target",
        target_block_hint={"page": 1, "x": 700, "y": 120},
    )
    page_only_expected = ExpectedIdentity(
        queue_record_id="ambiguous-page-target",
        target_block_hint={"page": 1},
    )
    hash_expected = ExpectedIdentity(
        queue_record_id="doi-hash-target",
        expected_doi_sha256=hashlib.sha256(b"10.1234/second.record").hexdigest(),
        doi_required=True,
    )
    conflicting_expected = ExpectedIdentity(
        queue_record_id="conflicting-target",
        expected_doi="10.1234/first.record",
        target_block_hint={"page": 1, "x": 700, "y": 120},
        doi_required=True,
    )
    doi_title_conflict_expected = ExpectedIdentity(
        queue_record_id="doi-title-conflict",
        expected_doi="10.1234/first.record",
        expected_title="Second Grounded Article Title",
        doi_required=True,
    )

    doi_resolution, doi_issues = module.resolve_front_matter(
        contents,
        expected_identity=doi_expected,
        target_required=True,
    )
    coordinate_resolution, coordinate_issues = module.resolve_front_matter(
        contents,
        expected_identity=coordinate_expected,
        target_required=True,
    )
    page_resolution, page_issues = module.resolve_front_matter(
        contents,
        expected_identity=page_only_expected,
    )
    hash_resolution, hash_issues = module.resolve_front_matter(
        contents,
        expected_identity=hash_expected,
        target_required=True,
    )
    conflicting_resolution, conflicting_issues = module.resolve_front_matter(
        contents,
        expected_identity=conflicting_expected,
        target_required=True,
    )
    doi_title_resolution, doi_title_issues = module.resolve_front_matter(
        contents,
        expected_identity=doi_title_conflict_expected,
        target_required=True,
    )

    assert doi_resolution.selected_block_id == "front-matter-block-2"
    assert doi_resolution.selection_method == "expected_doi"
    assert doi_issues == ()
    assert coordinate_resolution.selected_block_id == "front-matter-block-2"
    assert coordinate_resolution.selection_method == "target_block_hint"
    assert coordinate_issues == ()
    assert page_resolution.selected_block_id is None
    assert "target_block_hint_ambiguous" in page_resolution.reason_flags
    assert [issue.code for issue in page_issues] == ["VAL_METADATA_MULTI_ITEM"]
    assert hash_resolution.selected_block_id == "front-matter-block-2"
    assert hash_resolution.selection_method == "expected_doi"
    assert hash_issues == ()
    assert conflicting_resolution.selected_block_id is None
    assert "expected_identity_conflict" in conflicting_resolution.reason_flags
    assert [issue.code for issue in conflicting_issues] == ["VAL_METADATA_MULTI_ITEM"]
    assert doi_title_resolution.selected_block_id is None
    assert "expected_identity_conflict" in doi_title_resolution.reason_flags
    assert [issue.code for issue in doi_title_issues] == ["VAL_METADATA_MULTI_ITEM"]


async def test_post_parse_attaches_resolution_between_classification_and_normalization(monkeypatch):
    module = _front_matter_module()
    from bibr.pipeline.stages import post_parse as post_parse_module

    contents = _contents([])
    events: list[str] = []
    resolution = module.FrontMatterResolution(
        candidates=(),
        blocks=(),
        selected_block_id=None,
        selection_method="no_candidates",
        reason_flags=("no_candidates",),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset(),
    )
    expected = ExpectedIdentity(
        queue_record_id="ordered-attachment",
        target_block_hint={"page": 1},
    )
    issue = ValidationIssue(
        code="VAL_METADATA_MULTI_ITEM",
        severity="error",
        message="multiple records",
        origin_stage="extract",
        blocking=True,
    )

    async def classify(*_args, **_kwargs):
        events.append("classify")

    def attach(actual_contents, actual_expected, *_args, **_kwargs):
        assert actual_contents is contents
        assert actual_expected is expected
        events.append("front_matter")
        actual_contents.front_matter_resolution = resolution
        return (issue,)

    async def normalize(actual_contents, *_args, **_kwargs):
        assert actual_contents.front_matter_resolution is resolution
        events.append("normalize")

    async def extract(*_args, **_kwargs):
        return PaperMetadata(doi="", title="")

    monkeypatch.setattr(post_parse_module, "_classify_sections", classify)
    monkeypatch.setattr(post_parse_module, "_attach_front_matter_resolution", attach, raising=False)
    monkeypatch.setattr(post_parse_module, "_normalize_section_structure", normalize)
    monkeypatch.setattr(post_parse_module, "_extract_metadata_and_equations", extract)
    monkeypatch.setattr(post_parse_module, "_resolve_title", lambda *_args: None)
    monkeypatch.setattr(post_parse_module, "_finalize_abstract_and_keywords", lambda *_args: None)

    paper = await post_parse_module.post_parse(
        contents,
        "source.pdf",
        "source-hash",
        no_llm=True,
        expected_identity=expected,
    )

    assert events[:3] == ["classify", "front_matter", "normalize"]
    assert paper.contents.front_matter_resolution is resolution
    assert paper.validation_issues == [issue]


def _realistic_single_record_contents() -> PaperContents:
    return _contents(
        [
            _sentence(
                11,
                "Ada Lovelace, Grace Hopper, and Katherine Johnson",
                paragraph_id=1,
                section_id=1,
            ),
            _sentence(
                12,
                "Department of Computing, Example University, Amsterdam, Netherlands",
                paragraph_id=2,
                section_id=1,
            ),
            _sentence(
                13,
                "We evaluate a deterministic approach to scientific metadata ownership.",
                paragraph_id=3,
                section_id=1,
                label="abstract",
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                1,
                "A Deterministic Approach to Scientific Metadata Ownership",
                section_type=CanonicalSection.TITLE,
            ),
        ],
        detected_title="A Deterministic Approach to Scientific Metadata Ownership",
    )


def _two_title_case_records() -> PaperContents:
    return _contents(
        [
            _paragraph(
                1,
                "First Study of Healthy Aging",
                paragraph_id=1,
                bbox=(60.0, 100.0, 450.0, 145.0),
                label="paragraph_title",
            ),
            _paragraph(
                2,
                "Ada Lovelace and Grace Hopper",
                paragraph_id=2,
                bbox=(60.0, 150.0, 450.0, 180.0),
            ),
            _paragraph(
                3,
                "We studied healthy aging in the first cohort.",
                paragraph_id=3,
                bbox=(60.0, 190.0, 450.0, 280.0),
                label="abstract",
            ),
            _paragraph(
                4,
                "Second Study of Healthy Aging",
                paragraph_id=4,
                bbox=(540.0, 100.0, 930.0, 145.0),
                label="paragraph_title",
            ),
            _paragraph(
                5,
                "Katherine Johnson and Dorothy Vaughan",
                paragraph_id=5,
                bbox=(540.0, 150.0, 930.0, 180.0),
            ),
            _paragraph(
                6,
                "We studied healthy aging in the second cohort.",
                paragraph_id=6,
                bbox=(540.0, 190.0, 930.0, 280.0),
                label="abstract",
            ),
        ]
    )


def test_title_section_paragraphs_form_one_record_and_article_title_heading_is_not_byline():
    module = _front_matter_module()

    resolution, issues = module.resolve_front_matter(_realistic_single_record_contents())

    assert len(resolution.blocks) == 1
    assert resolution.allowed_text_ids == frozenset({11, 12, 13})
    heading = next(
        candidate for candidate in resolution.candidates if candidate.source_kind == "heading"
    )
    assert "title" in heading.roles
    assert "byline" not in heading.roles
    assert issues == ()


def test_wrong_expected_title_on_sole_block_abstains_and_blocks():
    module = _front_matter_module()
    expected = ExpectedIdentity(queue_record_id="wrong-title", expected_title="A Different Paper")

    resolution, issues = module.resolve_front_matter(
        _realistic_single_record_contents(), expected_identity=expected
    )

    assert resolution.selected_block_id is None
    assert resolution.allowed_text_ids == frozenset()
    assert resolution.allowed_section_ids == frozenset()
    assert "expected_title_not_found" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]
    assert issues[0].blocking is True


@pytest.mark.parametrize(
    "identity_kwargs,reason",
    [
        ({"expected_doi": "10.9999/wrong.record"}, "expected_doi_not_found"),
        (
            {"expected_doi_sha256": hashlib.sha256(b"10.9999/wrong.record").hexdigest()},
            "expected_doi_sha256_not_found",
        ),
    ],
)
def test_wrong_expected_doi_or_hash_on_sole_block_abstains_and_blocks(identity_kwargs, reason):
    module = _front_matter_module()
    expected = ExpectedIdentity(queue_record_id="wrong-doi", **identity_kwargs)

    resolution, issues = module.resolve_front_matter(
        _realistic_single_record_contents(), expected_identity=expected
    )

    assert resolution.selected_block_id is None
    assert resolution.allowed_text_ids == frozenset()
    assert reason in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_title_case_paragraph_titles_split_two_complete_records():
    module = _front_matter_module()
    expected = ExpectedIdentity(
        queue_record_id="title-case-second",
        expected_title="Second Study of Healthy Aging",
    )

    resolution, issues = module.resolve_front_matter(
        _two_title_case_records(), expected_identity=expected
    )

    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id == "front-matter-block-2"
    assert resolution.allowed_text_ids == frozenset({4, 5, 6})
    assert issues == ()


@pytest.mark.parametrize(
    "hint",
    [
        {},
        {"unknown": 1},
        {"occurrence": 0},
        {"occurrence": True},
        {"page": 0},
        {"page": 1.5},
        {"x": 100.0},
        {"x": float("nan"), "y": 100.0},
        {"bbox": [10.0, 20.0, 30.0]},
        {"bbox": [30.0, 20.0, 10.0, 40.0]},
        {"bbox": [-1.0, 20.0, 30.0, 40.0]},
        {"occurrence": 1, "page": 1},
        {"page": 1, "x": 100.0, "y": 100.0, "bbox": [50.0, 50.0, 150.0, 150.0]},
    ],
)
def test_invalid_target_hints_fail_closed_with_a_stable_reason(hint):
    module = _front_matter_module()
    expected = ExpectedIdentity(queue_record_id="invalid-hint", target_block_hint=hint)

    resolution, issues = module.resolve_front_matter(
        _two_title_case_records(), expected_identity=expected, target_required=True
    )

    assert resolution.selected_block_id is None
    assert resolution.allowed_text_ids == frozenset()
    assert "target_block_hint_invalid" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_occurrence_and_bbox_hints_are_strict_and_one_based():
    module = _front_matter_module()
    contents = _two_title_case_records()

    first, first_issues = module.resolve_front_matter(
        contents,
        expected_identity=ExpectedIdentity(
            queue_record_id="first", target_block_hint={"occurrence": 1}
        ),
    )
    second, second_issues = module.resolve_front_matter(
        contents,
        expected_identity=ExpectedIdentity(
            queue_record_id="second", target_block_hint={"occurrence": 2}
        ),
    )
    bbox, bbox_issues = module.resolve_front_matter(
        contents,
        expected_identity=ExpectedIdentity(
            queue_record_id="bbox",
            target_block_hint={"page": 1, "bbox": [530.0, 90.0, 940.0, 290.0]},
        ),
    )

    assert first.selected_block_id == "front-matter-block-1"
    assert second.selected_block_id == "front-matter-block-2"
    assert bbox.selected_block_id == "front-matter-block-2"
    assert first_issues == second_issues == bbox_issues == ()


def test_valid_hint_with_no_match_abstains_even_for_a_sole_block():
    module = _front_matter_module()
    expected = ExpectedIdentity(
        queue_record_id="missing-occurrence", target_block_hint={"occurrence": 2}
    )

    resolution, issues = module.resolve_front_matter(
        _realistic_single_record_contents(), expected_identity=expected
    )

    assert resolution.selected_block_id is None
    assert "target_block_hint_not_found" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


@pytest.mark.parametrize(
    "middle_rows",
    [
        [
            ("BACKGROUND AND OBJECTIVES", "paragraph_title"),
            ("Healthy aging is shaped by social conditions.", "abstract"),
        ],
        [
            ("A MULTICENTER LONGITUDINAL ANALYSIS", "paragraph_title"),
            ("Ada Lovelace and Grace Hopper", "text"),
        ],
        [
            ("Une Étude Déterministe du Vieillissement", "paragraph_title"),
            ("Ada Lovelace and Grace Hopper", "text"),
        ],
        [
            (
                "ADA LOVELACE, GRACE HOPPER, KATHERINE JOHNSON, DOROTHY VAUGHAN, "
                "MARY JACKSON, ANNIE EASLEY, MARGARET HAMILTON, AND BARBARA LISKOV",
                "text",
            ),
        ],
        [
            (
                "DEPARTMENT OF COMPUTING, EXAMPLE UNIVERSITY, AMSTERDAM, NETHERLANDS",
                "text",
            ),
            ("Ada Lovelace and Grace Hopper", "text"),
        ],
        [
            ("SESSION 4: CIVIC ENGAGEMENT AND VOLUNTARISM", "paragraph_title"),
            ("Ada Lovelace and Grace Hopper", "text"),
        ],
    ],
    ids=[
        "structured-abstract-heading",
        "uppercase-subtitle",
        "bilingual-parallel-title",
        "uppercase-long-author-list",
        "uppercase-institution",
        "proceedings-session-masthead",
    ],
)
def test_record_anatomy_does_not_split_internal_or_masthead_rows(middle_rows):
    module = _front_matter_module()
    rows = [
        _paragraph(
            1,
            "A Deterministic Study of Healthy Aging",
            paragraph_id=1,
            bbox=(100.0, 80.0, 900.0, 130.0),
            label="doc_title",
        )
    ]
    for offset, (text, label) in enumerate(middle_rows, start=2):
        rows.append(
            _paragraph(
                offset,
                text,
                paragraph_id=offset,
                bbox=(100.0, 80.0 + offset * 60, 900.0, 120.0 + offset * 60),
                label=label,
            )
        )
    rows.append(
        _paragraph(
            len(rows) + 1,
            "We report results from one article record.",
            paragraph_id=len(rows) + 1,
            bbox=(100.0, 700.0, 900.0, 850.0),
            label="abstract",
        )
    )

    resolution, _ = module.resolve_front_matter(_contents(rows))

    assert len(resolution.blocks) == 1


def test_toc_only_titles_and_running_header_do_not_form_multiple_records():
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(
                1,
                "JOURNAL OF HEALTHY AGING",
                paragraph_id=1,
                bbox=(20.0, 10.0, 980.0, 35.0),
                label="header",
            ),
            _paragraph(
                2,
                "First Study of Healthy Aging",
                paragraph_id=2,
                bbox=(80.0, 100.0, 450.0, 135.0),
                label="paragraph_title",
            ),
            _paragraph(
                3,
                "Second Study of Healthy Aging",
                paragraph_id=3,
                bbox=(80.0, 150.0, 450.0, 185.0),
                label="paragraph_title",
            ),
            _paragraph(
                4,
                "Third Study of Healthy Aging",
                paragraph_id=4,
                bbox=(80.0, 200.0, 450.0, 235.0),
                label="paragraph_title",
            ),
        ]
    )

    resolution, _ = module.resolve_front_matter(contents)

    assert len(resolution.blocks) == 1


def test_mixed_summary_matches_preserve_source_candidate_order():
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(1, "First unmatched row", paragraph_id=1, bbox=(10, 10, 100, 20)),
            _paragraph(2, "Second matched row", paragraph_id=2, bbox=(10, 30, 100, 40)),
            _paragraph(3, "Third unmatched row", paragraph_id=3, bbox=(10, 50, 100, 60)),
        ],
        region_summaries=[
            RegionSummary(
                page=1,
                index=0,
                label="text",
                bbox=(10, 30, 100, 40),
                section_id=0,
                content="preview",
            )
        ],
    )

    candidates = module.collect_front_matter_candidates(contents)

    assert [candidate.text_ids for candidate in candidates] == [(1,), (2,), (3,)]


def test_multi_page_paragraph_has_no_single_page_or_union_bbox():
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(1, "First page fragment", paragraph_id=1, bbox=(10, 900, 990, 990)),
            _paragraph(
                2,
                "Second page fragment",
                paragraph_id=1,
                bbox=(10, 10, 990, 100),
                page=2,
            ),
        ]
    )

    candidate = module.collect_front_matter_candidates(contents)[0]

    assert candidate.page is None
    assert candidate.bbox is None


def test_single_sentence_with_multi_page_provenance_is_also_fail_closed():
    module = _front_matter_module()
    sentence = _sentence(1, "A sentence split across pages", paragraph_id=1)
    sentence.provenance = [
        Provenance(page_no=1, bbox=(10, 900, 990, 990)),
        Provenance(page_no=2, bbox=(10, 10, 990, 100)),
    ]
    contents = _contents([sentence])

    candidate = module.collect_front_matter_candidates(contents)[0]

    assert candidate.page is None
    assert candidate.bbox is None


def test_resolution_is_deterministic_immutable_and_does_not_mutate_contents():
    module = _front_matter_module()
    contents = _two_title_case_records()
    before = copy.deepcopy(contents)
    expected = ExpectedIdentity(
        queue_record_id="deterministic", target_block_hint={"occurrence": 2}
    )

    first = module.resolve_front_matter(contents, expected_identity=expected)
    second = module.resolve_front_matter(contents, expected_identity=expected)

    assert first == second
    assert contents == before
    with pytest.raises(FrozenInstanceError):
        first[0].allowed_text_ids = frozenset()


async def test_preparsed_native_metadata_is_authoritative_over_false_multi_item_diagnostic():
    from bibr.pipeline.stages.post_parse import post_parse

    native = PaperMetadata(doi="10.1234/native.record", title="Native Structured Title")
    contents = _contents(
        [
            _sentence(1, "UNKNOWN UPPERCASE BODY HEADING ONE", paragraph_id=1, section_id=1),
            _sentence(2, "UNKNOWN UPPERCASE BODY HEADING TWO", paragraph_id=2, section_id=2),
        ],
        sections=[
            _section(0, "Root"),
            _section(1, "UNKNOWN UPPERCASE BODY HEADING ONE"),
            _section(2, "UNKNOWN UPPERCASE BODY HEADING TWO"),
        ],
        detected_title="Native Structured Title",
        preparsed_metadata=native,
    )

    paper = await post_parse(
        contents,
        "article.xml",
        "native-hash",
        no_llm=True,
        expected_identity=ExpectedIdentity(
            queue_record_id="native-record",
            expected_title="Wrong OCR-Derived Target",
        ),
    )

    assert paper.metadata.title == "Native Structured Title"
    assert paper.metadata.doi == "10.1234/native.record"
    assert "VAL_METADATA_MULTI_ITEM" not in {issue.code for issue in paper.validation_issues}


def test_multi_item_blocker_is_deduped_and_exported_as_non_promotable():
    module = _front_matter_module()
    from bibr.export.json_export import _apply_output_validation
    from bibr.pipeline.artifacts import ArtifactDisposition, disposition_for_issues

    _, issues = module.resolve_front_matter(
        _two_title_case_records(),
        expected_identity=ExpectedIdentity(
            queue_record_id="missing-title", expected_title="No Such Record"
        ),
    )
    payload = _apply_output_validation(
        {"info": {}, "bib": [], "bib_match": [], "info_match": []},
        [issues[0], issues[0]],
    )
    exported = [
        issue
        for issue in payload["validation"]["issues"]
        if issue["code"] == "VAL_METADATA_MULTI_ITEM"
    ]

    assert len(exported) == 1
    assert exported[0]["blocking"] is True
    assert payload["validation"]["promotable"] is False
    assert disposition_for_issues(issues) is ArtifactDisposition.BLOCKED


def test_abstract_section_ownership_blocks_novel_structured_heading_title_seed():
    module = _front_matter_module()
    contents = _contents(
        [
            _sentence(
                1,
                "Ada Lovelace and Grace Hopper",
                paragraph_id=1,
                section_id=1,
            ),
            _sentence(
                2,
                "Department of Computing, Example University",
                paragraph_id=2,
                section_id=1,
            ),
            _sentence(
                3,
                "BACKGROUND AND OBJECTIVES OF THIS STUDY",
                paragraph_id=3,
                section_id=2,
                label="paragraph_title",
            ),
            _sentence(
                4,
                "This structured abstract reports the objectives and principal findings.",
                paragraph_id=4,
                section_id=2,
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                1,
                "A Deterministic Study of Healthy Aging",
                section_type=CanonicalSection.TITLE,
            ),
            _section(2, "Abstract", section_type=CanonicalSection.ABSTRACT),
        ],
        detected_title="A Deterministic Study of Healthy Aging",
    )

    resolution, issues = module.resolve_front_matter(contents)
    structured_heading = next(
        candidate for candidate in resolution.candidates if candidate.text_ids == (3,)
    )

    assert "abstract" in structured_heading.roles
    assert "title" not in structured_heading.roles
    assert len(resolution.blocks) == 1
    assert resolution.allowed_text_ids == frozenset({1, 2, 3, 4})
    assert issues == ()


@pytest.mark.parametrize(
    "second_title",
    [
        "Loneliness, Social Isolation, and Healthy Aging",
        "Loneliness Anxiety and Healthy Aging",
        "LONELINESS, SOCIAL ISOLATION, AND HEALTHY AGING",
    ],
)
def test_punctuation_and_connective_paragraph_titles_outrank_byline_shape(second_title):
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(
                1,
                "A First Record About Community Health",
                paragraph_id=1,
                bbox=(60, 100, 450, 145),
                label="paragraph_title",
            ),
            _paragraph(
                2,
                "Ada Lovelace and Grace Hopper",
                paragraph_id=2,
                bbox=(60, 150, 450, 180),
            ),
            _paragraph(
                3,
                "We report the first record's methods and findings.",
                paragraph_id=3,
                bbox=(60, 190, 450, 280),
                label="abstract",
            ),
            _paragraph(
                4,
                second_title,
                paragraph_id=4,
                bbox=(540, 100, 930, 145),
                label="paragraph_title",
            ),
            _paragraph(
                5,
                "Katherine Johnson and Dorothy Vaughan",
                paragraph_id=5,
                bbox=(540, 150, 930, 180),
            ),
            _paragraph(
                6,
                "We report the second record's methods and findings.",
                paragraph_id=6,
                bbox=(540, 190, 930, 280),
                label="abstract",
            ),
        ]
    )
    expected = ExpectedIdentity(queue_record_id="punctuation-title", expected_title=second_title)

    resolution, issues = module.resolve_front_matter(contents, expected_identity=expected)
    candidate = next(item for item in resolution.candidates if item.text_ids == (4,))

    assert candidate.roles == frozenset({"heading", "title"})
    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id == "front-matter-block-2"
    assert resolution.allowed_text_ids == frozenset({4, 5, 6})
    assert issues == ()


def _realistic_toc_contents() -> PaperContents:
    return _contents(
        [
            _paragraph(
                1,
                "TABLE OF CONTENTS",
                paragraph_id=1,
                bbox=(50, 40, 950, 75),
                label="paragraph_title",
            ),
            _paragraph(
                2,
                "Loneliness, Social Isolation, and Healthy Aging",
                paragraph_id=2,
                bbox=(80, 100, 450, 140),
                label="paragraph_title",
            ),
            _paragraph(
                3,
                "Ada Lovelace and Grace Hopper",
                paragraph_id=3,
                bbox=(80, 145, 450, 175),
            ),
            _paragraph(
                4,
                "Loneliness Anxiety and Healthy Aging",
                paragraph_id=4,
                bbox=(520, 100, 920, 140),
                label="paragraph_title",
            ),
            _paragraph(
                5,
                "Katherine Johnson and Dorothy Vaughan",
                paragraph_id=5,
                bbox=(520, 145, 920, 175),
            ),
        ]
    )


def test_realistic_toc_title_byline_pairs_are_not_extractable_untargeted():
    module = _front_matter_module()

    resolution, issues = module.resolve_front_matter(_realistic_toc_contents())

    assert len(resolution.blocks) == 1
    assert resolution.selected_block_id is None
    assert resolution.allowed_text_ids == frozenset()
    assert "toc_listing" in resolution.reason_flags
    assert issues == ()


def test_expected_title_cannot_authorize_a_merged_toc_listing():
    module = _front_matter_module()
    expected = ExpectedIdentity(
        queue_record_id="targeted-toc",
        expected_title="Loneliness Anxiety and Healthy Aging",
    )

    resolution, issues = module.resolve_front_matter(
        _realistic_toc_contents(), expected_identity=expected
    )

    assert resolution.selected_block_id is None
    assert resolution.allowed_text_ids == frozenset()
    assert "toc_listing" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]
    assert issues[0].blocking is True


def test_minimal_repeated_title_byline_records_split_outside_toc_context():
    module = _front_matter_module()
    contents = _contents(
        [
            _paragraph(
                1,
                "First Minimal Record About Healthy Aging",
                paragraph_id=1,
                bbox=(80, 100, 450, 140),
                label="paragraph_title",
            ),
            _paragraph(
                2,
                "Ada Lovelace and Grace Hopper",
                paragraph_id=2,
                bbox=(80, 145, 450, 175),
            ),
            _paragraph(
                3,
                "Second Minimal Record About Healthy Aging",
                paragraph_id=3,
                bbox=(520, 100, 920, 140),
                label="paragraph_title",
            ),
            _paragraph(
                4,
                "Katherine Johnson and Dorothy Vaughan",
                paragraph_id=4,
                bbox=(520, 145, 920, 175),
            ),
        ]
    )
    expected = ExpectedIdentity(
        queue_record_id="minimal-second",
        expected_title="Second Minimal Record About Healthy Aging",
    )

    resolution, issues = module.resolve_front_matter(contents, expected_identity=expected)

    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id == "front-matter-block-2"
    assert resolution.allowed_text_ids == frozenset({3, 4})
    assert issues == ()


def test_two_structured_abstract_subheadings_with_abstract_bodies_stay_one_record():
    module = _front_matter_module()
    contents = _contents(
        [
            _sentence(1, "Ada Lovelace and Grace Hopper", paragraph_id=1, section_id=1),
            _sentence(
                2,
                "BACKGROUND AND OBJECTIVES OF THIS STUDY",
                paragraph_id=2,
                section_id=2,
                label="paragraph_title",
            ),
            _sentence(
                3,
                "We describe the background and objectives.",
                paragraph_id=3,
                section_id=2,
                label="abstract",
            ),
            _sentence(
                4,
                "DESIGN SETTING AND PARTICIPANT RECRUITMENT",
                paragraph_id=4,
                section_id=2,
                label="paragraph_title",
            ),
            _sentence(
                5,
                "Participants were recruited through community clinics.",
                paragraph_id=5,
                section_id=2,
                label="abstract",
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                1,
                "A Deterministic Study of Healthy Aging",
                section_type=CanonicalSection.TITLE,
            ),
            _section(2, "Abstract", section_type=CanonicalSection.ABSTRACT),
        ],
        detected_title="A Deterministic Study of Healthy Aging",
    )

    resolution, issues = module.resolve_front_matter(contents)

    assert len(resolution.blocks) == 1
    assert resolution.selected_block_id == "front-matter-block-1"
    assert resolution.allowed_text_ids == frozenset({1, 2, 3, 4, 5})
    assert not any(
        "title" in candidate.roles
        for candidate in resolution.candidates
        if candidate.section_id == 2
    )
    assert issues == ()


def test_shared_abstract_title_byline_affiliation_records_split_and_select_locally():
    module = _front_matter_module()
    contents = _contents(
        [
            _sentence(
                1,
                "First Shared Abstract Record",
                paragraph_id=1,
                section_id=1,
                label="paragraph_title",
            ),
            _sentence(2, "Ada Lovelace and Grace Hopper", paragraph_id=2, section_id=1),
            _sentence(
                3,
                "Department of Computing, Example University",
                paragraph_id=3,
                section_id=1,
            ),
            _sentence(
                4,
                "Second Shared Abstract Record",
                paragraph_id=4,
                section_id=1,
                label="paragraph_title",
            ),
            _sentence(
                5,
                "Katherine Johnson and Dorothy Vaughan",
                paragraph_id=5,
                section_id=1,
            ),
            _sentence(
                6,
                "Institute for Population Health, Example University",
                paragraph_id=6,
                section_id=1,
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(1, "Abstract", section_type=CanonicalSection.ABSTRACT),
        ],
    )
    expected = ExpectedIdentity(
        queue_record_id="shared-abstract-second",
        expected_title="Second Shared Abstract Record",
    )

    resolution, issues = module.resolve_front_matter(contents, expected_identity=expected)

    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id == "front-matter-block-2"
    assert resolution.allowed_text_ids == frozenset({4, 5, 6})
    assert issues == ()


def _in_this_issue_listing(*, publisher: bool) -> PaperContents:
    rows = [
        _paragraph(
            1,
            "  In   This   Issue  ",
            paragraph_id=1,
            bbox=(50, 40, 950, 75),
            label="paragraph_title",
        )
    ]
    next_id = 2
    if publisher:
        rows.append(
            _paragraph(
                next_id,
                "Oxford University Press",
                paragraph_id=next_id,
                bbox=(50, 80, 950, 105),
            )
        )
        next_id += 1
    rows.extend(
        [
            _paragraph(
                next_id,
                "Loneliness, Social Isolation, and Healthy Aging",
                paragraph_id=next_id,
                bbox=(80, 120, 450, 160),
                label="paragraph_title",
            ),
            _paragraph(
                next_id + 1,
                "Ada Lovelace and Grace Hopper",
                paragraph_id=next_id + 1,
                bbox=(80, 165, 450, 195),
            ),
            _paragraph(
                next_id + 2,
                "Loneliness Anxiety and Healthy Aging",
                paragraph_id=next_id + 2,
                bbox=(520, 120, 920, 160),
                label="paragraph_title",
            ),
            _paragraph(
                next_id + 3,
                "Katherine Johnson and Dorothy Vaughan",
                paragraph_id=next_id + 3,
                bbox=(520, 165, 920, 195),
            ),
        ]
    )
    return _contents(rows)


@pytest.mark.parametrize("publisher", [False, True], ids=["plain", "publisher-line"])
def test_in_this_issue_title_byline_listings_are_not_extractable_untargeted(publisher):
    module = _front_matter_module()

    resolution, issues = module.resolve_front_matter(_in_this_issue_listing(publisher=publisher))

    assert resolution.selected_block_id is None
    assert resolution.allowed_text_ids == frozenset()
    assert "toc_listing" in resolution.reason_flags
    assert issues == ()


@pytest.mark.parametrize("publisher", [False, True], ids=["plain", "publisher-line"])
def test_in_this_issue_expected_title_cannot_authorize_listing_entry(publisher):
    module = _front_matter_module()
    expected = ExpectedIdentity(
        queue_record_id="targeted-in-this-issue",
        expected_title="Loneliness Anxiety and Healthy Aging",
    )

    resolution, issues = module.resolve_front_matter(
        _in_this_issue_listing(publisher=publisher), expected_identity=expected
    )

    assert resolution.selected_block_id is None
    assert resolution.allowed_text_ids == frozenset()
    assert "toc_listing" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]
    assert issues[0].blocking is True


def test_shared_abstract_title_byline_abstract_records_split_and_select_locally():
    module = _front_matter_module()
    contents = _contents(
        [
            _sentence(
                1,
                "First Shared Abstract Body Record",
                paragraph_id=1,
                section_id=1,
                label="paragraph_title",
            ),
            _sentence(2, "Ada Lovelace and Grace Hopper", paragraph_id=2, section_id=1),
            _sentence(
                3,
                "The first explicit abstract body describes healthy aging.",
                paragraph_id=3,
                section_id=1,
                label="abstract",
            ),
            _sentence(
                4,
                "Second Shared Abstract Body Record",
                paragraph_id=4,
                section_id=1,
                label="paragraph_title",
            ),
            _sentence(
                5,
                "Katherine Johnson and Dorothy Vaughan",
                paragraph_id=5,
                section_id=1,
            ),
            _sentence(
                6,
                "The second explicit abstract body describes social isolation.",
                paragraph_id=6,
                section_id=1,
                label="abstract",
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(1, "Abstract", section_type=CanonicalSection.ABSTRACT),
        ],
    )
    expected = ExpectedIdentity(
        queue_record_id="shared-abstract-body-second",
        expected_title="Second Shared Abstract Body Record",
    )

    resolution, issues = module.resolve_front_matter(contents, expected_identity=expected)

    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id == "front-matter-block-2"
    assert resolution.allowed_text_ids == frozenset({4, 5, 6})
    assert issues == ()


def test_abstract_subhead_cannot_count_itself_as_byline_or_expected_article():
    module = _front_matter_module()
    contents = _contents(
        [
            _sentence(
                1,
                "STUDY DESIGN AND PARTICIPANT RECRUITMENT",
                paragraph_id=1,
                section_id=1,
                label="paragraph_title",
            ),
            _sentence(
                2,
                "Participants were recruited through community clinics.",
                paragraph_id=2,
                section_id=1,
                label="abstract",
            ),
            _sentence(
                3,
                "DATA COLLECTION AND STATISTICAL ANALYSIS",
                paragraph_id=3,
                section_id=1,
                label="paragraph_title",
            ),
            _sentence(
                4,
                "Data were collected prospectively and analyzed with preregistered models.",
                paragraph_id=4,
                section_id=1,
                label="abstract",
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(1, "Abstract", section_type=CanonicalSection.ABSTRACT),
        ],
    )
    expected = ExpectedIdentity(
        queue_record_id="structured-subhead-target",
        expected_title="DATA COLLECTION AND STATISTICAL ANALYSIS",
    )

    resolution, issues = module.resolve_front_matter(contents, expected_identity=expected)

    assert len(resolution.blocks) == 1
    assert resolution.selected_block_id is None
    assert resolution.allowed_text_ids == frozenset()
    assert "expected_title_not_found" in resolution.reason_flags
    assert not any(
        "title" in candidate.roles for candidate in resolution.candidates if candidate.text_ids
    )
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def _single_article_with_furniture(furniture: str):
    """One complete record plus one furniture heading the classifier typed TITLE."""

    title = "Emerging forest disease in Europe and North America"
    sections = [
        _section(0, "Root"),
        _section(1, furniture, section_type=CanonicalSection.TITLE, bbox=(60.0, 40.0, 240.0, 60.0)),
        _section(2, title, section_type=CanonicalSection.TITLE, bbox=(60.0, 100.0, 700.0, 150.0)),
        _section(
            3,
            "Abstract",
            section_type=CanonicalSection.ABSTRACT,
            bbox=(60.0, 320.0, 200.0, 340.0),
        ),
    ]
    sentences = [
        _paragraph(
            1,
            "DOI: 10.1515/ffp-2017-0016",
            paragraph_id=1,
            bbox=(60.0, 70.0, 300.0, 90.0),
            section_id=1,
        ),
        _paragraph(
            2,
            "Tomasz Oszako, Jacek Olchowik, Adam Szaniawski, Stanislaw Drozdowski",
            paragraph_id=2,
            bbox=(60.0, 160.0, 700.0, 190.0),
            section_id=2,
        ),
        _paragraph(
            3,
            "Department of Forest Protection, Forest Research Institute, Warsaw, Poland",
            paragraph_id=3,
            bbox=(60.0, 200.0, 700.0, 230.0),
            section_id=2,
        ),
        _paragraph(
            4,
            "Ash dieback and oak decline emerged across European forests over the last decade.",
            paragraph_id=4,
            bbox=(60.0, 350.0, 700.0, 420.0),
            section_id=3,
        ),
    ]
    return _contents(sentences, sections=sections, detected_title=title)


@pytest.mark.parametrize(
    "furniture",
    [
        "SHORT COMMUNICATION",
        "ORIGINAL ARTICLE",
        "RESEARCH ARTICLE",
        "OPEN ACCESS",
        "Check for updates",
        "ARTICLE INFO",
        "Article info",
        "How to cite this article",
        "COPYRIGHT",
        "You may also like",
    ],
)
def test_furniture_does_not_create_competing_record(furniture):
    module = _front_matter_module()
    contents = _single_article_with_furniture(furniture)

    resolution, issues = module.resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is not None
    assert not any(issue.code == "VAL_METADATA_MULTI_ITEM" for issue in issues)
    selected = next(
        block for block in resolution.blocks if block.block_id == resolution.selected_block_id
    )
    furniture_ids = {
        candidate.candidate_id
        for candidate in resolution.candidates
        if candidate.normalized_text == furniture.casefold()
    }
    assert furniture_ids.isdisjoint(selected.title_candidate_ids)


@pytest.mark.parametrize(
    "heading",
    [
        "Resumen",
        "RESUMO",
        "Abstrak",
        "Аннотация",
        "Referencias",
        "Bibliografi",
        "Hasil dan Pembahasan",
        "Palabras clave",
    ],
)
def test_multilingual_section_headings_do_not_create_competing_record(heading):
    module = _front_matter_module()
    contents = _single_article_with_furniture(heading)

    resolution, issues = module.resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is not None
    assert not any(issue.code == "VAL_METADATA_MULTI_ITEM" for issue in issues)


def _bilingual_single_article(*, parallel_as_heading: bool):
    """EN title with a same-record translated title (10.1590 / 10.17981 shapes)."""

    title = "MOVEMENT ASSESSMENT BATTERY FOR CHILDREN: THEORETICAL ADEQUACY"
    parallel = "Bateria de avaliacao motora para criancas: adequacao teorica do instrumento"
    sections = [
        _section(0, "Root"),
        _section(1, title, section_type=CanonicalSection.TITLE, bbox=(60.0, 60.0, 700.0, 110.0)),
        _section(
            2,
            "Abstract",
            section_type=CanonicalSection.ABSTRACT,
            bbox=(60.0, 320.0, 200.0, 340.0),
        ),
    ]
    if parallel_as_heading:
        sections.insert(
            2,
            _section(
                3,
                parallel,
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 115.0, 700.0, 150.0),
            ),
        )
        parallel_rows = []
    else:
        parallel_rows = [
            _paragraph(
                5,
                parallel,
                paragraph_id=5,
                bbox=(60.0, 115.0, 700.0, 150.0),
                section_id=1,
            )
        ]
    sentences = parallel_rows + [
        _paragraph(
            1,
            "Patrik Felipe Nazario, Luciana Ferreira, Jorge Both, Jose Luiz Lopes Vieira",
            paragraph_id=1,
            bbox=(60.0, 160.0, 700.0, 190.0),
            section_id=1,
        ),
        _paragraph(
            2,
            "https://doi.org/10.1590/1984-0462/2022/40/2020205",
            paragraph_id=2,
            bbox=(60.0, 200.0, 420.0, 220.0),
            section_id=1,
        ),
        _paragraph(
            3,
            "This study verified the internal structure of the motor assessment instrument.",
            paragraph_id=3,
            bbox=(60.0, 350.0, 700.0, 400.0),
            section_id=2,
        ),
    ]
    return _contents(sentences, sections=sections, detected_title=title)


@pytest.mark.parametrize("parallel_as_heading", [False, True])
def test_parallel_language_titles_share_one_record(parallel_as_heading):
    module = _front_matter_module()
    contents = _bilingual_single_article(parallel_as_heading=parallel_as_heading)

    resolution, issues = module.resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is not None
    assert len(resolution.blocks) == 1
    assert not any(issue.blocking for issue in issues)


def _article_with_weak_developed_seed():
    """A complete main record plus a weak seed developed only by a bare heading.

    Mirrors the 10.59784_matriks.v3i2.92 shape where a late uppercase heading
    (typed TITLE by the model) owns an empty translated abstract heading and
    nothing else.
    """

    title = "Correlation of blood sodium and potassium levels with body mass index"
    sections = [
        _section(0, "Root"),
        _section(1, title, section_type=CanonicalSection.TITLE, bbox=(60.0, 60.0, 700.0, 110.0)),
        _section(
            2,
            "Abstract",
            section_type=CanonicalSection.ABSTRACT,
            bbox=(60.0, 260.0, 200.0, 280.0),
        ),
        _section(
            3,
            "CLINICAL PRACTICE PERSPECTIVES",
            section_type=CanonicalSection.TITLE,
            bbox=(60.0, 520.0, 460.0, 550.0),
        ),
        _section(
            4,
            "Abstract",
            section_type=CanonicalSection.ABSTRACT,
            bbox=(60.0, 560.0, 200.0, 580.0),
        ),
    ]
    sentences = [
        _paragraph(
            1,
            "Vijay Pandey, Hemant Kumar Dutt, Ganesh Singh, Amritha Vinod",
            paragraph_id=1,
            bbox=(60.0, 120.0, 700.0, 150.0),
            section_id=1,
        ),
        _paragraph(
            2,
            "Department of Pharmacology, Government Medical College, Uttarakhand",
            paragraph_id=2,
            bbox=(60.0, 160.0, 700.0, 190.0),
            section_id=1,
        ),
        _paragraph(
            3,
            "DOI: 10.7324/JAPS.2017.70127",
            paragraph_id=3,
            bbox=(60.0, 200.0, 340.0, 220.0),
            section_id=1,
        ),
        _paragraph(
            4,
            "Serum electrolyte levels were correlated with body mass index in adults.",
            paragraph_id=4,
            bbox=(60.0, 290.0, 700.0, 340.0),
            section_id=2,
        ),
    ]
    return _contents(sentences, sections=sections, detected_title=title)


def test_complete_main_record_dominates_weak_title_seed():
    module = _front_matter_module()
    contents = _article_with_weak_developed_seed()

    resolution, issues = module.resolve_front_matter(contents, target_required=True)

    assert len(resolution.blocks) == 2
    assert resolution.selected_block_id is not None
    assert resolution.selection_method == "coherent_dominance"
    assert not any(issue.blocking for issue in issues)
    selected = next(
        block for block in resolution.blocks if block.block_id == resolution.selected_block_id
    )
    main_title_ids = {
        candidate.candidate_id
        for candidate in resolution.candidates
        if candidate.normalized_text.startswith("correlation of blood sodium")
    }
    assert main_title_ids & set(selected.title_candidate_ids)


def _two_complete_articles():
    sections = [
        _section(0, "Root"),
        _section(
            1,
            "First independent study of community health",
            section_type=CanonicalSection.TITLE,
            bbox=(60.0, 60.0, 700.0, 100.0),
        ),
        _section(
            2,
            "Second independent study of environmental policy",
            section_type=CanonicalSection.TITLE,
            bbox=(60.0, 360.0, 700.0, 400.0),
        ),
    ]
    sentences = [
        _paragraph(
            1,
            "Alice Example, Bob Sample",
            paragraph_id=1,
            bbox=(60.0, 110.0, 700.0, 140.0),
            section_id=1,
        ),
        _paragraph(
            2,
            "DOI: 10.1000/first.2024.1",
            paragraph_id=2,
            bbox=(60.0, 150.0, 340.0, 170.0),
            section_id=1,
        ),
        _paragraph(
            3,
            "Carla Scholar, David Researcher",
            paragraph_id=3,
            bbox=(60.0, 410.0, 700.0, 440.0),
            section_id=2,
        ),
        _paragraph(
            4,
            "DOI: 10.1000/second.2024.2",
            paragraph_id=4,
            bbox=(60.0, 450.0, 340.0, 470.0),
            section_id=2,
        ),
    ]
    return _contents(sentences, sections=sections)


def test_two_independently_complete_records_still_abstain():
    module = _front_matter_module()

    resolution, issues = module.resolve_front_matter(
        _two_complete_articles(),
        target_required=True,
    )

    assert resolution.selected_block_id is None
    assert any(issue.code == "VAL_METADATA_MULTI_ITEM" and issue.blocking for issue in issues)


def test_supplied_expected_identity_failure_disables_coherent_dominance():
    # The weak-seed fixture would win coherent dominance untargeted, but a
    # supplied expected title that matches nothing must stay fail-closed.
    module = _front_matter_module()
    contents = _article_with_weak_developed_seed()
    expected = ExpectedIdentity(
        queue_record_id="dominance-guard",
        expected_title="An entirely different benchmark paper title",
    )

    resolution, issues = module.resolve_front_matter(
        contents,
        expected_identity=expected,
        target_required=True,
    )

    assert resolution.selected_block_id is None
    assert "expected_title_not_found" in resolution.reason_flags
    assert any(issue.code == "VAL_METADATA_MULTI_ITEM" and issue.blocking for issue in issues)


def test_competitor_with_own_byline_and_doi_blocks_dominance():
    module = _front_matter_module()
    contents = _two_complete_articles()

    resolution, _ = module.resolve_front_matter(contents, target_required=False)

    assert resolution.selected_block_id is None
    assert resolution.selection_method == "abstained"


def _article_with_classifier_typed_byline(byline: str) -> PaperContents:
    """A byline classified as TITLE must not create a second article record.

    The classifier has no dedicated byline class. The title and byline are both valid evidence even when the byline is assigned a high-confidence TITLE label."""
    title = "Judgment Under Uncertainty In Applied Risk Assessment"
    return _contents(
        [
            _paragraph(
                1,
                "DOI: 10.1177/09567976241249183",
                paragraph_id=1,
                bbox=(60.0, 160.0, 340.0, 180.0),
                section_id=1,
            ),
            _paragraph(
                2,
                "Department of Psychology, The Ohio State University",
                paragraph_id=2,
                bbox=(60.0, 240.0, 700.0, 260.0),
                section_id=2,
            ),
            _paragraph(
                3,
                "We report two studies of probabilistic reasoning under time pressure.",
                paragraph_id=3,
                bbox=(60.0, 320.0, 700.0, 380.0),
                section_id=3,
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                1, title, section_type=CanonicalSection.TITLE, bbox=(60.0, 90.0, 700.0, 130.0)
            ),
            _section(
                2, byline, section_type=CanonicalSection.TITLE, bbox=(60.0, 200.0, 700.0, 225.0)
            ),
            _section(3, "Abstract", section_type=CanonicalSection.ABSTRACT),
        ],
        detected_title=title,
    )


@pytest.mark.parametrize(
    "byline",
    [
        "Michael L. DeKay1 and Shiyu Dou1,2",
        "Misun Kim1,5 and Christian F. Doeller1,2,3,4",
        "Hannah C. Williamson",
        "Jason McInerney4, and Braden Thue3",
    ],
)
def test_classifier_typed_byline_does_not_root_a_second_record(byline):
    module = _front_matter_module()

    resolution, issues = module.resolve_front_matter(
        _article_with_classifier_typed_byline(byline),
        target_required=True,
    )

    assert len(resolution.blocks) == 1
    assert resolution.selected_block_id is not None
    assert not any(issue.blocking for issue in issues)
    byline_candidate = next(
        candidate for candidate in resolution.candidates if candidate.raw_text.strip() == byline
    )
    # Marked, not demoted: the row keeps its title role so nothing downstream
    # loses a candidate; it is only denied the right to root a record.
    assert module.CLASSIFIED_BYLINE_TITLE_ROLE in byline_candidate.roles
    assert "title" in byline_candidate.roles


def test_capitalised_kicker_is_not_treated_as_a_classifier_typed_byline():
    # "At least 65% capitalised words" describes an article-type kicker just as
    # well as a byline. Without positive person-name evidence (a middle initial,
    # an affiliation superscript) the row must keep its ability to root a
    # record, or every capitalised heading silently merges the blocks.
    module = _front_matter_module()

    resolution, _ = module.resolve_front_matter(
        _article_with_classifier_typed_byline("CLINICAL PRACTICE PERSPECTIVES"),
        target_required=True,
    )

    kicker = next(
        candidate
        for candidate in resolution.candidates
        if candidate.raw_text.strip() == "CLINICAL PRACTICE PERSPECTIVES"
    )
    assert module.CLASSIFIED_BYLINE_TITLE_ROLE not in kicker.roles
