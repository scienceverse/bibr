"""Front matter read from position and printed numbering, not from guessed body types.

Whether a section's rows reach the front matter used to turn on whether the
section classifier typed it as body text or left it untyped. A retrained
classifier that leaves numbered body headings untyped fed whole body sections
in, and one that guesses "introduction" for a byline set as a heading under the
title dropped the byline. Every title, name and row below is invented.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bibr.config import GlobalSettings
from bibr.extract.core_metadata import render_block_context
from bibr.extract.front_matter import collect_front_matter_candidates, resolve_front_matter
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
)
from bibr.run_mode import fast_settings
from bibr.structure.section_tree import body_numbered_section_ids

U = CanonicalSection.UNKNOWN


def _section(
    section_id: int,
    header: str,
    section_type: CanonicalSection = U,
    *,
    top: float = 100.0,
    page: int = 1,
    level: int = 1,
    parent: int | None = 0,
    source: str | None = None,
    score: float = 0.0,
) -> PaperSection:
    return PaperSection(
        section_id=section_id,
        header=header,
        level=level,
        parent_section_id=parent,
        section_type=section_type,
        classification_score=score,
        classification_source=source,
        provenance=[Provenance(page_no=page, bbox=(90.0, top, 900.0, top + 15.0))],
    )


def _root() -> PaperSection:
    return PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)


def _row(
    text_id: int,
    text: str,
    *,
    section_id: int,
    top: float,
    page: int = 1,
    label: str = "text",
) -> PaperSentence:
    return PaperSentence(
        text_id=text_id,
        text=text,
        section_id=section_id,
        paragraph_id=text_id,
        page_number=page,
        provenance=[Provenance(page_no=page, bbox=(90.0, top, 900.0, top + 20.0))],
        region_meta={"region_type": label, "font_size": 9.0, "font_bold": False},
    )


def _contents(
    sections: list[PaperSection], sentences: list[PaperSentence], *, title: str
) -> PaperContents:
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={section.section_id: "" for section in sections},
        detected_title=title,
        region_summaries=[],
    )


def _reading(contents: PaperContents) -> list[tuple[str, int | None, frozenset[str]]]:
    return [
        (candidate.raw_text, candidate.section_id, candidate.roles)
        for candidate in collect_front_matter_candidates(contents)
    ]


NUMBERED_TITLE = "Regional Water Supply After the Utility Reform"


def _numbered_paper(*, flipped: bool) -> PaperContents:
    """A proceedings paper whose body headings are numbered I., II., III.

    ``flipped`` leaves every body heading after the first one untyped, as a
    classifier that is unsure of them does; otherwise they carry body types.
    """

    def body(section_type: CanonicalSection) -> tuple[CanonicalSection, str | None, float]:
        return (U, None, 0.4) if flipped else (section_type, "model", 0.95)

    second, second_source, second_score = body(CanonicalSection.INTRODUCTION)
    sub, sub_source, sub_score = body(CanonicalSection.RESULTS)
    third, third_source, third_score = body(CanonicalSection.DISCUSSION)
    sections = [
        _root(),
        _section(1, NUMBERED_TITLE, CanonicalSection.TITLE, top=60.0, source="title", score=1.0),
        _section(
            2, "Abstract", CanonicalSection.ABSTRACT, top=200.0, source="exact_alias", score=1.0
        ),
        _section(
            3,
            "Keywords—water supply; regional planning; quality control",
            CanonicalSection.KEYWORDS,
            top=300.0,
            source="exact_alias",
            score=1.0,
        ),
        _section(
            4,
            "I. HOW REGIONAL WATER SUPPLY IS ORGANISED TODAY",
            CanonicalSection.INTRODUCTION,
            top=340.0,
            source="model",
            score=0.95,
        ),
        _section(
            5,
            "II. COMPETITION BETWEEN OPERATORS AND QUALITY CONTROL OF THE WATER NETWORK",
            second,
            page=2,
            source=second_source,
            score=second_score,
        ),
        _section(
            6,
            "B. Licences for New Treatment Plants",
            sub,
            page=2,
            top=400.0,
            level=2,
            parent=5,
            source=sub_source,
            score=sub_score,
        ),
        _section(7, "III. LESSONS", third, page=3, source=third_source, score=third_score),
        _section(
            8,
            "REFERENCES",
            CanonicalSection.REFERENCES,
            page=3,
            top=600.0,
            source="exact_alias",
            score=1.0,
        ),
    ]
    sentences = [
        _row(1, "Anna Berger, Tomas Lind", section_id=1, top=100.0),
        _row(2, "Northfield University, Northfield 1000", section_id=1, top=130.0),
        _row(
            3,
            "The regional supply of drinking water changed after the reform of the utilities.",
            section_id=2,
            top=220.0,
            label="abstract",
        ),
        _row(4, "The utilities were reorganised in three stages.", section_id=4, top=360.0),
        _row(
            5,
            "Competition between the operators was opened step by step.",
            section_id=5,
            top=120.0,
            page=2,
        ),
        _row(6, "Clara Ott, Peter Holm and Ida Brun", section_id=6, top=420.0, page=2),
        _row(
            7,
            "The boards should agree on one quality standard.",
            section_id=7,
            top=120.0,
            page=3,
        ),
        _row(8, "[1] A. Author, Water and towns, 2001.", section_id=8, top=620.0, page=3),
    ]
    return _contents(sections, sentences, title=NUMBERED_TITLE)


def test_numbered_body_headings_read_the_same_typed_or_untyped():
    typed = _numbered_paper(flipped=False)
    untyped = _numbered_paper(flipped=True)

    assert _reading(untyped) == _reading(typed)
    assert {section for _, section, _ in _reading(untyped)} <= {0, 1, 2, 3}


@pytest.mark.parametrize("flipped", [False, True])
def test_untyped_numbered_body_headings_do_not_split_the_record(flipped):
    contents = _numbered_paper(flipped=flipped)
    # Fast mode: the tagger has read the title and byline off the page.
    contents.front_tagger_result = SimpleNamespace(
        title_keywords=SimpleNamespace(title=NUMBERED_TITLE),
        byline=SimpleNamespace(
            authors=[SimpleNamespace(family="Berger"), SimpleNamespace(family="Lind")]
        ),
    )

    resolution, issues = resolve_front_matter(
        contents, target_required=True, settings=fast_settings(GlobalSettings())
    )

    assert issues == ()
    assert len(resolution.blocks) == 1
    assert resolution.selected_block_id is not None
    context = render_block_context(resolution)
    assert NUMBERED_TITLE in context
    assert "Anna Berger, Tomas Lind" in context
    assert "COMPETITION BETWEEN OPERATORS" not in context


def _late_rows_paper(late_header: str, late_row: str, *, numbered_late: bool) -> PaperContents:
    """A numbered paper with an untyped section after its conclusions."""

    late = f"5. {late_header}" if numbered_late else late_header
    sections = [
        _root(),
        _section(1, NUMBERED_TITLE, CanonicalSection.TITLE, top=60.0, source="title", score=1.0),
        _section(
            2, "Abstract", CanonicalSection.ABSTRACT, top=200.0, source="exact_alias", score=1.0
        ),
        _section(
            3,
            "1. Introduction",
            CanonicalSection.INTRODUCTION,
            top=340.0,
            source="exact_alias",
            score=1.0,
        ),
        _section(
            4,
            "4. Conclusions",
            CanonicalSection.DISCUSSION,
            page=4,
            source="exact_alias",
            score=1.0,
        ),
        _section(5, late, U, page=4, top=500.0),
    ]
    sentences = [
        _row(1, "Anna Berger, Tomas Lind", section_id=1, top=100.0),
        _row(
            2,
            "The regional supply of drinking water changed after the reform.",
            section_id=2,
            top=220.0,
            label="abstract",
        ),
        _row(3, "The utilities were reorganised in three stages.", section_id=3, top=360.0),
        _row(4, "The reform worked where the boards cooperated.", section_id=4, top=120.0, page=4),
        _row(5, late_row, section_id=5, top=520.0, page=4),
    ]
    return _contents(sections, sentences, title=NUMBERED_TITLE)


def test_unnumbered_untyped_section_after_the_body_stays_front_matter():
    row = "Department of Hydrology, Northfield University, Northfield"
    contents = _late_rows_paper("Affiliations", row, numbered_late=False)

    texts = [text for text, _, _ in _reading(contents)]

    assert "Affiliations" in texts
    assert row in texts


def test_numbered_untyped_section_after_the_body_is_body_text():
    row = "The boards should publish their quality reports every year."
    contents = _late_rows_paper("Recommendations", row, numbered_late=True)

    texts = [text for text, _, _ in _reading(contents)]

    assert "5. Recommendations" not in texts
    assert row not in texts


def test_numbered_untyped_row_before_the_body_stays_front_matter():
    sections = [
        _root(),
        _section(1, NUMBERED_TITLE, CanonicalSection.TITLE, top=60.0, source="title", score=1.0),
        _section(
            2, "Abstract", CanonicalSection.ABSTRACT, top=200.0, source="exact_alias", score=1.0
        ),
        _section(3, "2: Email:", U, top=400.0),
        _section(
            4,
            "Introduction",
            CanonicalSection.INTRODUCTION,
            page=2,
            source="exact_alias",
            score=1.0,
        ),
        _section(
            5,
            "3. Sources of the Law",
            CanonicalSection.INTRODUCTION,
            page=3,
            source="model",
            score=0.97,
        ),
    ]
    sentences = [
        _row(1, "Anna Berger, Tomas Lind", section_id=1, top=100.0),
        _row(2, "Law courts read treaties in two ways.", section_id=2, top=220.0, label="abstract"),
        _row(3, "anna.berger@example.org", section_id=3, top=420.0),
        _row(4, "Treaties bind the parties.", section_id=4, top=100.0, page=2),
    ]
    contents = _contents(sections, sentences, title=NUMBERED_TITLE)

    texts = [text for text, _, _ in _reading(contents)]

    assert "anna.berger@example.org" in texts


HEADED_TITLE = "地方都市の公共交通と住民の移動"
HEADED_NAME = "山 田 花 子"
BODY_ROWS = (
    "路線バスの本数は年々減っている。",
    "住民の多くは自家用車で移動する。",
)


def _byline_heading_paper(
    section_type: CanonicalSection,
    *,
    source: str | None,
    score: float,
    header: str = HEADED_NAME,
    abstract_first: bool = False,
) -> PaperContents:
    """A short essay whose only body section is headed by the author's name."""

    sections = [
        _root(),
        _section(1, HEADED_TITLE, CanonicalSection.TITLE, top=60.0, source="title", score=1.0),
    ]
    sentences = []
    if abstract_first:
        sections.append(
            _section(
                3, "Abstract", CanonicalSection.ABSTRACT, top=100.0, source="exact_alias", score=1.0
            )
        )
        sentences.append(
            _row(
                10,
                "This essay reviews bus services.",
                section_id=3,
                top=110.0,
                label="abstract",
            )
        )
    sections.append(_section(2, header, section_type, top=140.0, source=source, score=score))
    sentences += [
        _row(1, BODY_ROWS[0], section_id=2, top=200.0),
        _row(2, BODY_ROWS[1], section_id=2, top=120.0, page=2),
    ]
    return _contents(sections, sentences, title=HEADED_TITLE)


def test_guessed_body_type_under_the_title_reads_as_untyped():
    guessed = _byline_heading_paper(CanonicalSection.INTRODUCTION, source="model", score=0.6)
    untyped = _byline_heading_paper(U, source=None, score=0.99)

    assert _reading(guessed) == _reading(untyped)
    resolution, _ = resolve_front_matter(guessed)
    assert HEADED_NAME in render_block_context(resolution)


@pytest.mark.parametrize(
    ("source", "score", "header", "abstract_first"),
    [
        # A type the model is sure of.
        ("model", 0.95, HEADED_NAME, False),
        # An alias match on the heading text.
        ("exact_alias", 1.0, HEADED_NAME, False),
        # A heading printed with a body number.
        ("model", 0.6, "1. 研究の背景", False),
        # Past the abstract, the body has begun.
        ("model", 0.6, HEADED_NAME, True),
    ],
)
def test_body_type_stays_where_it_is_no_guess_under_the_title(
    source, score, header, abstract_first
):
    contents = _byline_heading_paper(
        CanonicalSection.INTRODUCTION,
        source=source,
        score=score,
        header=header,
        abstract_first=abstract_first,
    )

    texts = [text for text, _, _ in _reading(contents)]

    assert BODY_ROWS[1] not in texts


def test_body_numbered_section_ids_follow_the_document_scheme():
    sections = [
        _root(),
        _section(1, "1. Introduction", CanonicalSection.INTRODUCTION),
        _section(2, "2.3 Sampling", CanonicalSection.METHODS),
        _section(3, "A.1 Proofs", U),
        _section(4, "I Want to Believe", U),
        _section(5, "75 years of water law", U),
        _section(6, "Chapter 3: Results", CanonicalSection.RESULTS),
    ]

    assert body_numbered_section_ids(sections) == frozenset({1, 2, 6})
    roman = [_root(), _section(1, "I. SCOPE", U), _section(2, "II. METHOD", U)]
    assert body_numbered_section_ids(roman) == frozenset({1, 2})


GAP_TITLE = "Groundwater Recharge Under Irrigated Orchards"
GAP_BYLINE = "Anna Berger1 and Tomas Lind2"
GAP_ROW = "E-mail: anna.berger@example.org; Tel.: +47 22 85 50 00"


def _gap_paper(
    section_type: CanonicalSection,
    *,
    source: str | None,
    score: float,
    after_abstract: bool = False,
    header: str = "Correspondence",
) -> PaperContents:
    """A paper that prints a "Correspondence" block between its byline and abstract.

    ``after_abstract`` moves the block below the abstract, out of the gap.
    """

    corr_top = 300.0 if after_abstract else 160.0
    abstract_top = 200.0 if after_abstract else 240.0
    sections = [
        _root(),
        _section(1, GAP_TITLE, CanonicalSection.TITLE, top=60.0, source="title", score=1.0),
        _section(
            3,
            "Abstract",
            CanonicalSection.ABSTRACT,
            top=abstract_top,
            source="exact_alias",
            score=1.0,
        ),
        _section(2, header, section_type, top=corr_top, source=source, score=score),
        _section(
            4,
            "1. Introduction",
            CanonicalSection.INTRODUCTION,
            top=400.0,
            source="exact_alias",
            score=1.0,
        ),
    ]
    if not after_abstract:
        sections[2], sections[3] = sections[3], sections[2]
    sentences = [
        _row(1, GAP_BYLINE, section_id=1, top=100.0),
        _row(2, GAP_ROW, section_id=2, top=corr_top + 20.0),
        _row(
            3,
            "Orchard irrigation raises recharge in dry years.",
            section_id=3,
            top=abstract_top + 20.0,
            label="abstract",
        ),
        _row(4, "Recharge under orchards is poorly measured.", section_id=4, top=420.0),
    ]
    return _contents(sections, sentences, title=GAP_TITLE)


def test_guessed_back_matter_type_between_title_and_abstract_reads_as_front_matter():
    guessed = _gap_paper(CanonicalSection.ACKNOWLEDGMENT, source="model", score=0.96)
    untyped = _gap_paper(U, source=None, score=0.99)

    texts = [text for text, _, _ in _reading(guessed)]

    assert GAP_ROW in texts
    assert _reading(guessed) == _reading(untyped)


@pytest.mark.parametrize(
    ("source", "header", "after_abstract"),
    [
        # A type matched from the printed heading keeps its type.
        ("exact_alias", "Acknowledgments", False),
        # Below the abstract the block is outside the gap.
        ("model", "Correspondence", True),
    ],
)
def test_back_matter_type_outside_a_guess_in_the_gap_stays(source, header, after_abstract):
    contents = _gap_paper(
        CanonicalSection.ACKNOWLEDGMENT,
        source=source,
        score=0.96,
        after_abstract=after_abstract,
        header=header,
    )

    texts = [text for text, _, _ in _reading(contents)]

    assert GAP_ROW not in texts
