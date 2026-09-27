"""A byline printed above the title that layout labelled a page header.

scans_eval80/W4319033756 sets "Hubert Heinen" over "German-Texan Attitudes toward
the Civil War"; scans_eval80/W4391777158 sets "Eva-Maria Biermann-Ratjen" over its
chapter title, beside the journal's page head. Layout labels both name rows
headers, the parser files them with the running heads, and neither front matter
had the author's name ahead of the abstract. Geometry is from the papers; body
text is placeholder.
"""

from __future__ import annotations

import pytest

from bibr.extract.core_metadata import render_author_context, render_block_context
from bibr.extract.front_matter import (
    BYLINE_PROBATION_ROLE,
    collect_front_matter_candidates,
    resolve_front_matter,
)
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)

TITLE = "German-Texan Attitudes toward the Civil War"
TITLE_BOX = (219.0, 300.0, 843.0, 321.0)


def _contents(
    heads: list[tuple[str, tuple[float, float, float, float]]],
    *,
    byline: str | None = None,
    title: str = TITLE,
) -> PaperContents:
    sections = [
        PaperSection(
            section_id=0,
            header="Root",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
            provenance=[],
        ),
        PaperSection(
            section_id=1,
            header=title,
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.TITLE,
            provenance=[Provenance(page_no=1, bbox=TITLE_BOX)],
        ),
    ]
    body_box = (63.0, 366.0, 843.0, 608.0)
    sentences = [
        PaperSentence(
            text_id=1,
            text="Texans and others have long scrutinized these attitudes.",
            section_id=1,
            paragraph_id=1,
            page_number=1,
            provenance=[Provenance(page_no=1, bbox=body_box)],
            region_meta={"region_type": "text", "font_size": 9.0, "font_bold": False},
        )
    ]
    if byline is not None:
        sentences.insert(
            0,
            PaperSentence(
                text_id=0,
                text=byline,
                section_id=1,
                paragraph_id=0,
                page_number=1,
                provenance=[Provenance(page_no=1, bbox=(219.0, 330.0, 843.0, 345.0))],
                region_meta={"region_type": "text", "font_size": 9.0, "font_bold": False},
            ),
        )
    summaries = [
        RegionSummary(page=1, index=index, label="header", bbox=bbox, content=text)
        for index, (text, bbox) in enumerate(heads)
    ]
    summaries += [
        RegionSummary(page=1, index=len(heads), label="doc_title", bbox=TITLE_BOX, content=title),
        RegionSummary(page=1, index=len(heads) + 1, label="text", bbox=body_box, content="Texans"),
    ]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={0: "", 1: ""},
        detected_title=title,
        detected_headers=[text for text, _ in heads],
        region_summaries=summaries,
    )


def test_name_row_labelled_header_above_the_title_reaches_the_author_context():
    contents = _contents([("Hubert Heinen", (65.0, 229.0, 270.0, 249.0))])

    candidates = collect_front_matter_candidates(contents)
    head = next(candidate for candidate in candidates if candidate.raw_text == "Hubert Heinen")
    resolution, _ = resolve_front_matter(contents, target_required=True)

    assert BYLINE_PROBATION_ROLE in head.roles
    assert not head.roles & {"byline", "title"}
    assert resolution.selected_block_id is not None
    context = render_author_context(resolution, full_text=render_block_context(resolution))
    assert context.startswith("Hubert Heinen\n" + TITLE)


def test_only_the_name_row_of_a_two_part_page_head_is_admitted():
    contents = _contents(
        [
            ("Eva-Maria Biermann-Ratjen", (101.0, 240.0, 357.0, 259.0)),
            ("PERSON 1 (1998) 64-68", (767.0, 253.0, 924.0, 268.0)),
        ]
    )

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert "Eva-Maria Biermann-Ratjen" in texts
    assert "PERSON 1 (1998) 64-68" not in texts


@pytest.mark.parametrize(
    ("text", "bbox"),
    [
        ("Journal of Southern History", (65.0, 229.0, 270.0, 249.0)),  # not a name
        ("BMC Public Health", (65.0, 229.0, 270.0, 249.0)),  # a banner in capitals
        ("CASE REPORT", (65.0, 229.0, 270.0, 249.0)),
        # Name-shaped journal and article-type heads (audience_eval200/W4248753238,
        # audience_ci60/W4410919437).
        ("Educational Review", (102.0, 229.0, 227.0, 244.0)),
        ("Original Manuscript", (102.0, 229.0, 280.0, 248.0)),
        ("Hubert Heinen", (65.0, 60.0, 270.0, 80.0)),  # far above the title
        ("Hubert Heinen", (65.0, 330.0, 270.0, 350.0)),  # below the title
    ],
)
def test_other_page_heads_stay_out(text, bbox):
    contents = _contents([(text, bbox)])

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert text not in texts


def test_a_running_head_that_repeats_the_title_stays_out():
    """audience_eval200/osf_7fvr9 sets its short title as the running head."""

    title = "Remythologising Satan: A New Version of The Fall of Lucifer."
    contents = _contents([("Remythologising Satan.", (116.0, 229.0, 315.0, 246.0))], title=title)

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert "Remythologising Satan." not in texts


@pytest.mark.parametrize("text", ["Hubert Heinen", "Critical Inquiry"])
def test_page_heads_wait_for_a_page_without_a_byline(text):
    """On a page that prints a byline, a journal name can still pass the same
    name-shape test ("Critical Inquiry"), so a page head joins only a first
    page with no byline ahead of its abstract."""

    contents = _contents([(text, (65.0, 229.0, 270.0, 249.0))], byline="Anna Berg and Carl Dahl")

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert "Anna Berg and Carl Dahl" in texts
    assert text not in texts


CHAPTER_TITLE = (
    "Das Phänomen Aggression betrachtet im Rahmen der Klientenzentrierten Entwicklungspsychologie"
)
KEYWORDS = (
    "Keywords: Aggression, Selbstentwicklungstendenz, Selbstbehauptungstendenz, "
    "analytische Objektbeziehungstheorie."
)


def _chapter_contents() -> PaperContents:
    """scans_eval80/W4391777158: name and journal page heads, title, abstract, keywords."""

    title_box = (102.0, 122.0, 900.0, 177.0)
    abstract_box = (101.0, 315.0, 928.0, 376.0)
    keywords_box = (101.0, 377.0, 881.0, 392.0)
    heads = [
        ("Eva-Maria Biermann-Ratjen", (101.0, 81.0, 357.0, 100.0)),
        ("PERSON 1 (1998) 64-68", (767.0, 53.0, 924.0, 68.0)),
    ]
    sections = [
        PaperSection(
            section_id=0,
            header="Root",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
            provenance=[],
        ),
        PaperSection(
            section_id=1,
            header=CHAPTER_TITLE,
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.TITLE,
            provenance=[Provenance(page_no=1, bbox=title_box)],
        ),
        PaperSection(
            section_id=2,
            header="Abstract",
            level=2,
            parent_section_id=1,
            section_type=CanonicalSection.ABSTRACT,
            provenance=[Provenance(page_no=1, bbox=(101.0, 285.0, 169.0, 300.0))],
        ),
    ]
    sentences = [
        PaperSentence(
            text_id=index,
            text=text,
            section_id=2,
            paragraph_id=index,
            page_number=1,
            provenance=[Provenance(page_no=1, bbox=bbox)],
            region_meta={"region_type": "text", "font_size": 9.0, "font_bold": False},
        )
        for index, (text, bbox) in enumerate(
            [
                ("Verschiedene Aggressionskonzepte werden diskutiert.", abstract_box),
                (KEYWORDS, keywords_box),
            ],
            start=1,
        )
    ]
    summaries = [
        RegionSummary(page=1, index=index, label="header", bbox=bbox, content=text)
        for index, (text, bbox) in enumerate(heads)
    ]
    summaries += [
        RegionSummary(page=1, index=2, label="doc_title", bbox=title_box, content=CHAPTER_TITLE),
        RegionSummary(
            page=1,
            index=3,
            label="paragraph_title",
            bbox=(101.0, 285.0, 169.0, 300.0),
            content="Abstract",
        ),
        RegionSummary(page=1, index=4, label="abstract", bbox=abstract_box, content="Verschiedene"),
        RegionSummary(page=1, index=5, label="text", bbox=keywords_box, content=KEYWORDS),
    ]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={0: "", 1: "", 2: ""},
        detected_title=CHAPTER_TITLE,
        detected_headers=[text for text, _ in heads] + [CHAPTER_TITLE],
        region_summaries=summaries,
    )


def test_a_page_head_joins_when_the_only_bylines_follow_the_abstract():
    """German nouns are capitalised, so the keyword line reads as a byline, and
    the page no longer takes the no-byline rescue; the name over the title is
    still the only byline printed ahead of the abstract."""

    contents = _chapter_contents()

    candidates = collect_front_matter_candidates(contents)
    by_text = {candidate.raw_text: candidate for candidate in candidates}
    resolution, _ = resolve_front_matter(contents, target_required=True)

    assert "byline" in by_text[KEYWORDS].roles
    assert by_text["Eva-Maria Biermann-Ratjen"].roles == {BYLINE_PROBATION_ROLE}
    assert "PERSON 1 (1998) 64-68" not in by_text
    context = render_author_context(resolution, full_text=render_block_context(resolution))
    assert context.startswith("Eva-Maria Biermann-Ratjen\n" + CHAPTER_TITLE)
