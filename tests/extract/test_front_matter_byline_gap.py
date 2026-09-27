"""Byline rows the header classifier mistypes between the title and the abstract.

Each case is cut down from a dev paper whose page-1 byline cells were promoted to
headings (or merged into a correspondence paragraph) and typed as endnote or
acknowledgment sections, so part of the byline never reached the front matter.
Only names and layout order are kept; bodies are placeholders.
"""

from __future__ import annotations

import pytest

from bibr.extract.core_metadata import render_author_context, render_block_context
from bibr.extract.front_matter import (
    BYLINE_PROBATION_ROLE,
    FrontRolePolicy,
    _front_gap_heading_is_byline,
    collect_front_matter_candidates,
    group_front_matter_blocks,
    resolve_front_matter,
)
from bibr.extract.front_role import FrontRolePredictions, RoleScores
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)


def _section(
    section_id: int,
    header: str,
    section_type: CanonicalSection,
    *,
    top: float,
    page: int = 1,
) -> PaperSection:
    return PaperSection(
        section_id=section_id,
        header=header,
        level=0 if section_id == 0 else 1,
        parent_section_id=None if section_id == 0 else 0,
        section_type=section_type,
        provenance=[]
        if section_id == 0
        else [Provenance(page_no=page, bbox=(90.0, top, 900.0, top + 15.0))],
    )


def _sentence(
    text_id: int,
    text: str,
    *,
    section_id: int,
    paragraph_id: int,
    top: float,
    page: int = 1,
    label: str = "text",
) -> PaperSentence:
    return PaperSentence(
        text_id=text_id,
        text=text,
        section_id=section_id,
        paragraph_id=paragraph_id,
        page_number=page,
        provenance=[Provenance(page_no=page, bbox=(90.0, top, 900.0, top + 20.0))],
        region_meta={"region_type": label, "font_size": 9.0, "font_bold": False},
    )


def _contents(
    sections: list[PaperSection],
    sentences: list[PaperSentence],
    *,
    detected_title: str,
    region_summaries: list[RegionSummary] | None = None,
) -> PaperContents:
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={section.section_id: "" for section in sections},
        detected_title=detected_title,
        region_summaries=region_summaries or [],
    )


def _author_context(contents: PaperContents) -> str:
    resolution, _ = resolve_front_matter(contents, target_required=True)
    return render_author_context(resolution, full_text=render_block_context(resolution))


GRID_TITLE = "AssertCoder: LLM-Based Assertion Generation via Multimodal Specification Extraction"


def _grid_byline_contents(*, closing_type: CanonicalSection | None) -> PaperContents:
    """W4414739865: a 2x2 IEEE author grid, one heading per cell."""

    sections = [
        _section(0, "Root", CanonicalSection.UNKNOWN, top=0.0),
        _section(1, GRID_TITLE, CanonicalSection.TITLE, top=60.0),
        _section(2, "Enyuan Tian", CanonicalSection.ENDNOTE, top=120.0),
        _section(3, "Yiwei Ci, Qiusong Yang*", CanonicalSection.ENDNOTE, top=120.0),
        _section(4, "Yufeng Li", CanonicalSection.UNKNOWN, top=200.0),
        # Typed TITLE in the paper, where the front-role model's masthead-free
        # "other" score keeps it from rooting a record; this fixture carries no
        # model scores, so it keeps the cell out of that question.
        _section(5, "Zhichao Lyu", CanonicalSection.UNKNOWN, top=200.0),
    ]
    sentences = [
        _sentence(
            1,
            "Institute of Software, Chinese Academy of Sciences",
            section_id=3,
            paragraph_id=1,
            top=140.0,
        ),
        _sentence(
            2,
            "Institute of Computing Technology, Chinese Academy of Sciences",
            section_id=4,
            paragraph_id=2,
            top=220.0,
        ),
        _sentence(
            3,
            "Institute of Software, Chinese Academy of Sciences Beijing, China",
            section_id=5,
            paragraph_id=3,
            top=220.0,
        ),
    ]
    if closing_type is not None:
        sections.append(_section(6, "Abstract", closing_type, top=300.0))
        sentences.append(
            _sentence(
                4,
                "Assertion-based verification is critical.",
                section_id=6,
                paragraph_id=4,
                top=320.0,
                label="abstract",
            )
        )
    return _contents(sections, sentences, detected_title=GRID_TITLE)


def test_grid_byline_cells_typed_as_endnotes_join_the_front_matter():
    contents = _grid_byline_contents(closing_type=CanonicalSection.ABSTRACT)

    candidates = collect_front_matter_candidates(contents)
    by_text = {candidate.raw_text: candidate for candidate in candidates}

    for cell in ("Enyuan Tian", "Yiwei Ci, Qiusong Yang*"):
        assert {"byline", BYLINE_PROBATION_ROLE} <= by_text[cell].roles
    assert len(group_front_matter_blocks(candidates)) == 1
    context = _author_context(contents)
    for name in ("Enyuan Tian", "Yiwei Ci", "Qiusong Yang", "Yufeng Li", "Zhichao Lyu"):
        assert name in context


def test_front_gap_needs_a_closing_abstract_or_body_section():
    contents = _grid_byline_contents(closing_type=None)

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert "Enyuan Tian" not in texts
    assert "Yiwei Ci, Qiusong Yang*" not in texts


NULLING_TITLE = "Meta-Learner with Linear Nulling"


def test_name_over_email_cells_join_the_front_matter():
    """W2807000842: three name-over-address columns.

    The parser made the first and last cells headings and left the middle one
    as a paragraph under the first; the classifier typed the first an endnote.
    """

    contents = _contents(
        [
            _section(0, "Root", CanonicalSection.UNKNOWN, top=0.0),
            _section(1, NULLING_TITLE, CanonicalSection.TITLE, top=90.0),
            _section(2, "Sung Whan Yoon shyoon8@kaist.ac.kr", CanonicalSection.ENDNOTE, top=180.0),
            _section(3, "Jaekyun Moon jmoon@kaist.edu", CanonicalSection.TITLE, top=180.0),
            _section(4, "Abstract", CanonicalSection.ABSTRACT, top=300.0),
        ],
        [
            _sentence(
                1, "Jun Seo\r\ntjwns0630@kaist.ac.kr", section_id=2, paragraph_id=1, top=180.0
            ),
            _sentence(
                2,
                "School of Electrical Engineering,\r\nKorea Advanced Institute of Science and Technology (KAIST)",
                section_id=3,
                paragraph_id=2,
                top=220.0,
            ),
            _sentence(
                3,
                "We propose a meta-learning algorithm.",
                section_id=4,
                paragraph_id=3,
                top=320.0,
                label="abstract",
            ),
        ],
        detected_title=NULLING_TITLE,
    )

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert "Sung Whan Yoon shyoon8@kaist.ac.kr" in texts
    assert "Jun Seo\r\ntjwns0630@kaist.ac.kr" in texts
    context = _author_context(contents)
    for name in ("Sung Whan Yoon", "Jun Seo", "Jaekyun Moon"):
        assert name in context


ORIENTATION_TITLE = "Adaptation-induced sharpening of orientation tuning curves in the"


def test_byline_merged_into_a_correspondence_paragraph_is_admitted_on_its_first_region():
    """W4390186873: reading order put the byline row after the
    "- Corresponding author:" heading, and the paragraph join merged it with the
    correspondence rows. The heading is typed acknowledgment; the abstract is on
    page 2."""

    byline_row = "Afef Ouelhazi a\r\n, Vishal Bharmauria b, Stéphane Molotchnikoff a"
    merged = (
        byline_row + " Dr. Vishal Bharmauria Department of Psychology Toronto, Canada "
        "Email: author@example.org"
    )
    byline_box = (113.0, 181.0, 660.0, 200.0)
    contents = _contents(
        [
            _section(0, "Root", CanonicalSection.UNKNOWN, top=0.0),
            _section(1, ORIENTATION_TITLE, CanonicalSection.TITLE, top=91.0),
            _section(2, "mouse visual cortex", CanonicalSection.TITLE, top=133.0),
            _section(3, "- Corresponding author:", CanonicalSection.ACKNOWLEDGMENT, top=341.0),
            _section(4, "ABSTRACT", CanonicalSection.ABSTRACT, top=100.0, page=2),
        ],
        [
            _sentence(
                1,
                "a. Département de Sciences Biologiques, Université de Montréal",
                section_id=2,
                paragraph_id=1,
                top=227.0,
            ),
            PaperSentence(
                text_id=2,
                text=merged,
                section_id=3,
                paragraph_id=2,
                page_number=1,
                provenance=[
                    Provenance(page_no=1, bbox=byline_box),
                    Provenance(page_no=1, bbox=(107.0, 359.0, 304.0, 376.0)),
                ],
                region_meta={"region_type": "text", "font_size": 6.7, "font_bold": False},
            ),
            _sentence(
                3,
                "Orientation selectivity is an emergent property.",
                section_id=4,
                paragraph_id=3,
                top=131.0,
                page=2,
                label="abstract",
            ),
        ],
        detected_title=ORIENTATION_TITLE,
        region_summaries=[
            RegionSummary(
                page=1, index=5, label="text", bbox=byline_box, section_id=3, content=byline_row
            ),
            RegionSummary(
                page=1,
                index=6,
                label="text",
                bbox=(107.0, 359.0, 304.0, 376.0),
                section_id=3,
                content="Dr. Vishal Bharmauria",
            ),
        ],
    )

    candidates = collect_front_matter_candidates(contents)

    merged_candidate = next(candidate for candidate in candidates if candidate.raw_text == merged)
    assert BYLINE_PROBATION_ROLE in merged_candidate.roles
    assert "Stéphane Molotchnikoff" in _author_context(contents)


def test_a_title_typed_body_heading_opens_no_front_gap():
    """10.30574/wjarr.2022.14.3.0574: the classifier typed a numbered body heading
    TITLE. Only the parser's detected title opens the gap, so the byline-shaped
    related-works citation after it stays out, even with a body section below."""

    citation = "Arjun Aman, Aryan Singh, Ayush Raj and Sandeep Raj"
    contents = _contents(
        [
            _section(0, "Root", CanonicalSection.UNKNOWN, top=0.0),
            _section(1, "Bar and QR Code Recognition for Retail", CanonicalSection.TITLE, top=60.0),
            _section(2, "Abstract", CanonicalSection.ABSTRACT, top=200.0),
            _section(
                3,
                "1.1.1. An Efficient Bar/QR Code Recognition System",
                CanonicalSection.TITLE,
                top=560.0,
            ),
            _section(4, "Related works", CanonicalSection.AUTHOR_CONTRIBUTIONS, top=580.0),
            _section(5, "2. Methods", CanonicalSection.METHODS, top=700.0),
        ],
        [
            _sentence(
                1,
                "Divya E, Jaishreenithi V, Keerthika S and S Yamuna",
                section_id=1,
                paragraph_id=1,
                top=100.0,
            ),
            _sentence(
                2,
                "Humans do require a lot of communication.",
                section_id=2,
                paragraph_id=2,
                top=220.0,
                label="abstract",
            ),
            _sentence(3, citation, section_id=4, paragraph_id=3, top=600.0),
        ],
        detected_title="Bar and QR Code Recognition for Retail",
    )

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert citation not in texts


def _title_page_with_byline(gap_header: str, gap_text: str, *, regions: int = 0) -> PaperContents:
    """A title page that prints its byline, then a mistyped section before the abstract."""

    title = "Trust in Science Across Cultures"
    gap_box = (90.0, 320.0, 900.0, 340.0)
    return _contents(
        [
            _section(0, "Root", CanonicalSection.UNKNOWN, top=0.0),
            _section(1, title, CanonicalSection.TITLE, top=90.0),
            _section(2, gap_header, CanonicalSection.ACKNOWLEDGMENT, top=300.0),
            _section(3, "Abstract", CanonicalSection.ABSTRACT, top=100.0, page=2),
        ],
        [
            _sentence(1, "Anna Berg and Carl Dahl", section_id=1, paragraph_id=1, top=140.0),
            _sentence(
                2,
                "Department of Psychology, University of Oslo",
                section_id=1,
                paragraph_id=2,
                top=170.0,
            ),
            _sentence(3, gap_text, section_id=2, paragraph_id=3, top=320.0),
            _sentence(
                4,
                "We surveyed trust in science.",
                section_id=3,
                paragraph_id=4,
                top=120.0,
                page=2,
                label="abstract",
            ),
        ],
        detected_title=title,
        region_summaries=[
            RegionSummary(
                page=1, index=index, label="text", bbox=gap_box, section_id=2, content=gap_text
            )
            for index in range(regions)
        ],
    )


@pytest.mark.parametrize(
    "gap_header",
    [
        "Author Note",  # osf_cv6px and six other APA manuscripts
        "* Corresponding Author:",  # W4315563590
        "Credit Author Statement",  # osf_ytws5
        "THE DECISION TO REFINANCE",  # W1511304478
    ],
)
def test_field_label_headings_in_the_gap_stay_out_on_a_page_with_a_byline(gap_header):
    contents = _title_page_with_byline(gap_header, "We thank the participants.")

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert "Anna Berg and Carl Dahl" in texts
    assert gap_header not in texts


def test_a_one_region_publisher_line_in_the_gap_stays_out():
    """W2947837352: the publisher's line reads as a byline
    ("Wilson & Lafleur"), but on a page that prints one, a paragraph in the gap
    needs name evidence, and a one-region paragraph gets no preview pass."""

    contents = _title_page_with_byline("Publisher", "Éditions Wilson & Lafleur, inc.", regions=1)

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert "Éditions Wilson & Lafleur, inc." not in texts


@pytest.mark.parametrize(
    "gap_header",
    [
        "Data Availability",
        "Handling Editor: Jane Smith",
        "Edited by Jane Smith",
        "Received 12 March 2020",
        "Competing Interests",
        "Specialty Section",
        "Citation: Anna Berg",  # any labelled line
    ],
)
def test_editorial_and_metadata_headings_in_the_gap_stay_out(gap_header):
    """Editorial and article-metadata lines share the title-to-abstract gap with
    the byline and pass its capital-ratio test; a name among them belongs to an
    editor, so none of them may become byline evidence."""

    contents = _title_page_with_byline(gap_header, "We thank the participants.")

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert "Anna Berg and Carl Dahl" in texts
    assert gap_header not in texts


def _byline_vote(page: int, index: int) -> FrontRolePredictions:
    scores = RoleScores(probs={"byline": 0.97, "other": 0.03}, top="byline", confidence=0.97)
    return FrontRolePredictions({(page, index): scores}, model_version="test")


@pytest.mark.parametrize(
    ("header", "admitted"),
    [
        ("Edited by Jane Smith", False),
        ("Handling Editor: Jane Smith jane.smith@example.org", False),
        ("Reviewed by: John Doe", False),
        # A name over its author's e-mail address is still a byline cell, and a
        # plain two-name cell still passes on its shape.
        ("Sung Whan Yoon shyoon8@kaist.ac.kr", True),
        ("Yiwei Ci, Qiusong Yang*", True),
    ],
)
def test_gap_heading_rule_rejects_editorial_lines_before_any_byline_evidence(header, admitted):
    """Neither the classifier's byline vote nor an e-mail address admits an
    editorial line in the gap."""

    summary = RegionSummary(
        page=1, index=4, label="paragraph_title", bbox=(90.0, 300.0, 900.0, 315.0), section_id=2
    )

    verdict = _front_gap_heading_is_byline(header, summary, _byline_vote(1, 4), FrontRolePolicy())

    assert verdict is admitted


def test_an_editor_over_an_email_address_stays_out_of_the_no_byline_rescue():
    """On a title page that prints no byline, the no-byline rescue admits a
    name over an e-mail address in the gap. An editor's line there reads the
    same way and must not become the page's only byline."""

    title = "Trust in Science Across Cultures"
    editor = "Handling Editor: Jane Smith jane.smith@example.org"
    contents = _contents(
        [
            _section(0, "Root", CanonicalSection.UNKNOWN, top=0.0),
            _section(1, title, CanonicalSection.TITLE, top=90.0),
            _section(2, editor, CanonicalSection.ACKNOWLEDGMENT, top=300.0),
            _section(3, "Abstract", CanonicalSection.ABSTRACT, top=100.0, page=2),
        ],
        [
            _sentence(
                1,
                "Department of Psychology, University of Oslo",
                section_id=1,
                paragraph_id=1,
                top=170.0,
            ),
            _sentence(2, "We thank the participants.", section_id=2, paragraph_id=2, top=320.0),
            _sentence(
                3,
                "We surveyed trust in science.",
                section_id=3,
                paragraph_id=3,
                top=120.0,
                page=2,
                label="abstract",
            ),
        ],
        detected_title=title,
    )

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert title in texts
    assert editor not in texts


def test_an_editor_over_an_email_address_in_a_gap_paragraph_stays_out():
    """The gap admits a paragraph that is a name over an e-mail address; an
    editor's line has the same shape."""

    gap_text = "Handling Editor: Jane Smith jane.smith@example.org"
    contents = _title_page_with_byline("Publisher", gap_text, regions=1)

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert "Anna Berg and Carl Dahl" in texts
    assert gap_text not in texts


def test_a_merged_paragraph_led_by_an_editorial_line_stays_out():
    """The gap admits a paragraph over several regions whose first region
    reads as a byline (the merged byline and correspondence block above); an
    editorial credit leading a merged block reads as a byline too."""

    first_row = "Edited by Jane Smith and John Doe"
    merged = first_row + " Department of Psychology, University of Oslo, Norway"
    first_box = (113.0, 320.0, 660.0, 339.0)
    second_box = (107.0, 341.0, 504.0, 358.0)
    contents = _title_page_with_byline("Publisher", "placeholder")
    contents.sentences[2] = PaperSentence(
        text_id=3,
        text=merged,
        section_id=2,
        paragraph_id=3,
        page_number=1,
        provenance=[Provenance(page_no=1, bbox=first_box), Provenance(page_no=1, bbox=second_box)],
        region_meta={"region_type": "text", "font_size": 6.7, "font_bold": False},
    )
    contents.region_summaries = [
        RegionSummary(
            page=1, index=5, label="text", bbox=first_box, section_id=2, content=first_row
        ),
        RegionSummary(
            page=1,
            index=6,
            label="text",
            bbox=second_box,
            section_id=2,
            content="Department of Psychology",
        ),
    ]

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert "Anna Berg and Carl Dahl" in texts
    assert merged not in texts
