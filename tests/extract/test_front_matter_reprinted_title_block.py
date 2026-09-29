"""The byline of a title block that a repository cover page reprints.

A journal distributed through a repository adds a cover page that prints the
article's title, its translated titles and its byline; the article's first
page prints all three again. The two rows under the title repeat on both pages
as a block, so the parser files every copy with the running heads, and the
front matter had no byline: the author call read only the closing author
biographies and returned the first author alone. Geometry is from the cover
page of such an article; names and text are placeholders.
"""

from __future__ import annotations

import pytest

from bibr.extract.core_metadata import render_author_context, render_block_context
from bibr.extract.front_matter import (
    BYLINE_PROBATION_ROLE,
    _looks_like_person_name_list,
    collect_front_matter_candidates,
    resolve_front_matter,
)
from bibr.extract.title_subtitle import fold_printed_subtitle
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)

TITLE = "Shared gardens in the city. Meeting places for their growers"
TRANSLATED = (
    "Jardins partagés en ville. Des lieux de rencontre pour leurs jardiniers\n"
    "Gedeelde tuinen in de stad. Ontmoetingsplekken voor hun tuiniers"
)
BYLINE = "Marie Dubois, Pieter Janssens, Anne-Sophie Martin et Luc Van\xa0Damme"
TITLE_BOX = (139.0, 206.0, 642.0, 251.0)
TRANSLATED_BOX = (138.0, 255.0, 678.0, 294.0)
BYLINE_BOX = (139.0, 309.0, 761.0, 327.0)
EDITION_BOX = (140.0, 416.0, 269.0, 430.0)
URL_BOX = (138.0, 426.0, 464.0, 466.0)
URL_TEXT = "URL : https://journals.example.org/town/1234\nISSN : 0000-0000"


def _contents(
    block: list[tuple[str, str, tuple[float, float, float, float]]],
    *,
    filed: set[str],
    body: tuple[str, ...] = (),
) -> PaperContents:
    """A cover page: the title, then *block*'s rows, then the edition fields.

    *filed* are the block rows the parser filed with the running heads; the
    others stay in the sentence stream as a paragraph of the title section.
    *body* are sentences of the article's introduction on the next page.
    """

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
            header=TITLE,
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.TITLE,
            provenance=[Provenance(page_no=1, bbox=TITLE_BOX)],
        ),
        PaperSection(
            section_id=2,
            header="Édition électronique",
            level=2,
            parent_section_id=1,
            section_type=CanonicalSection.UNKNOWN,
            provenance=[Provenance(page_no=1, bbox=EDITION_BOX)],
        ),
        PaperSection(
            section_id=3,
            header="Introduction",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.INTRODUCTION,
            provenance=[Provenance(page_no=2, bbox=(139.0, 100.0, 300.0, 115.0))],
        ),
    ]
    summaries = [
        RegionSummary(page=1, index=0, label="doc_title", bbox=TITLE_BOX, content=TITLE),
    ]
    sentences: list[PaperSentence] = []
    for index, (label, text, bbox) in enumerate(block, start=1):
        summaries.append(RegionSummary(page=1, index=index, label=label, bbox=bbox, content=text))
        if text in filed:
            continue
        sentences.append(
            PaperSentence(
                text_id=len(sentences) + 1,
                text=text,
                section_id=1,
                paragraph_id=len(sentences) + 1,
                page_number=1,
                provenance=[Provenance(page_no=1, bbox=bbox)],
                region_meta={"region_type": label, "font_size": 6.0, "font_bold": False},
            )
        )
    edition_index = len(block) + 1
    summaries += [
        RegionSummary(
            page=1,
            index=edition_index,
            label="paragraph_title",
            bbox=EDITION_BOX,
            content="Édition électronique",
        ),
        RegionSummary(
            page=1, index=edition_index + 1, label="text", bbox=URL_BOX, content=URL_TEXT
        ),
    ]
    sentences.append(
        PaperSentence(
            text_id=len(sentences) + 1,
            text=URL_TEXT,
            section_id=2,
            paragraph_id=len(sentences) + 1,
            page_number=1,
            provenance=[Provenance(page_no=1, bbox=URL_BOX)],
            region_meta={"region_type": "text", "font_size": 4.6, "font_bold": False},
        )
    )
    for text in body:
        sentences.append(
            PaperSentence(
                text_id=len(sentences) + 1,
                text=text,
                section_id=3,
                paragraph_id=len(sentences) + 1,
                page_number=2,
                provenance=[Provenance(page_no=2, bbox=(139.0, 130.0, 760.0, 160.0))],
                region_meta={"region_type": "text", "font_size": 6.0, "font_bold": False},
            )
        )
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={0: "", 1: "", 2: "", 3: " ".join(body)},
        detected_title=TITLE,
        # Each filed row is printed on the cover and on the first page.
        detected_headers=[text for text in filed for _ in range(2)],
        region_summaries=summaries,
    )


COVER_BLOCK = [("text", TRANSLATED, TRANSLATED_BOX), ("text", BYLINE, BYLINE_BOX)]


def test_the_byline_of_a_reprinted_title_block_reaches_the_author_context():
    contents = _contents(COVER_BLOCK, filed={TRANSLATED, BYLINE})

    candidates = collect_front_matter_candidates(contents)
    by_text = {candidate.raw_text: candidate for candidate in candidates}
    resolution, _ = resolve_front_matter(contents, target_required=True)

    assert {"byline", BYLINE_PROBATION_ROLE} <= by_text[BYLINE].roles
    assert "title" not in by_text[BYLINE].roles
    # The translated titles stay with the running heads.
    assert TRANSLATED not in by_text
    assert resolution.selected_block_id is not None
    block = render_block_context(resolution)
    assert block.startswith(TITLE + "\n" + BYLINE + "\n")
    assert render_author_context(resolution, full_text=block).startswith(BYLINE + "\n")


def test_a_reprinted_byline_follows_its_title_when_region_order_is_partial():
    contents = _contents(COVER_BLOCK, filed={TRANSLATED, BYLINE})
    # A cover row whose box no layout region matches: region order no longer
    # covers every candidate, so candidates keep source order, which has no
    # slot for a row the parser filed with the running heads.
    contents.sentences.append(
        PaperSentence(
            text_id=len(contents.sentences) + 1,
            text="This document was generated automatically.",
            section_id=2,
            paragraph_id=len(contents.sentences) + 1,
            page_number=1,
            provenance=[Provenance(page_no=1, bbox=(139.0, 688.0, 551.0, 702.0))],
            region_meta={"region_type": "text", "font_size": 4.6, "font_bold": False},
        )
    )

    resolution, _ = resolve_front_matter(contents, target_required=True)
    title, _ = fold_printed_subtitle(TITLE, resolution)

    block = render_block_context(resolution)
    assert block.startswith(TITLE + "\n" + BYLINE + "\n")
    # The heading the cover sets next is not the title's subtitle.
    assert title == TITLE


@pytest.mark.parametrize(
    ("block", "filed"),
    [
        # A repeated row that is not a list of names.
        (
            [
                ("text", TRANSLATED, TRANSLATED_BOX),
                ("text", "Town Studies, Collection générale", BYLINE_BOX),
            ],
            {TRANSLATED, "Town Studies, Collection générale"},
        ),
        # A row the parser kept ends the block: a name list after it is not the
        # title block's byline.
        (COVER_BLOCK, {BYLINE}),
        # Layout labelled the row a header, not body text.
        (
            [("text", TRANSLATED, TRANSLATED_BOX), ("header", BYLINE, BYLINE_BOX)],
            {TRANSLATED, BYLINE},
        ),
    ],
)
def test_other_rows_filed_with_the_running_heads_stay_out(block, filed):
    contents = _contents(block, filed=filed)

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert not filed & set(texts)


ABOVE_TITLE_BOX = (139.0, 150.0, 761.0, 168.0)


def test_a_name_list_ahead_of_the_title_in_reading_order_is_not_in_its_block():
    block = [("text", TRANSLATED, TRANSLATED_BOX), ("text", BYLINE, ABOVE_TITLE_BOX)]
    contents = _contents(block, filed={TRANSLATED, BYLINE})
    # Printed above the title, the name list comes first on the page.
    contents.region_summaries.sort(key=lambda summary: summary.bbox[1])
    for index, summary in enumerate(contents.region_summaries):
        summary.index = index

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert BYLINE not in texts


def test_a_name_list_printed_above_the_title_is_not_a_reprinted_byline():
    # Reading order puts the name list after the title and the filed row, but
    # the page prints it above the title: it is not the title block's byline.
    block = [("text", TRANSLATED, TRANSLATED_BOX), ("text", BYLINE, ABOVE_TITLE_BOX)]
    contents = _contents(block, filed={TRANSLATED, BYLINE})

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert BYLINE not in texts


TITLE_CASE_TRANSLATION = "Urban Gardens and Public Life"


def test_a_title_case_translated_title_is_not_a_byline():
    # Set in title case, the translated title splits into capitalised chunks
    # as a byline does, but the paper prints its words in lower case.
    block = [("text", TITLE_CASE_TRANSLATION, TRANSLATED_BOX), ("text", BYLINE, BYLINE_BOX)]
    contents = _contents(
        block,
        filed={TITLE_CASE_TRANSLATION, BYLINE},
        body=("Shared plots bring public life to urban gardens.",),
    )

    by_text = {
        candidate.raw_text: candidate for candidate in collect_front_matter_candidates(contents)
    }

    assert TITLE_CASE_TRANSLATION not in by_text
    assert "byline" in by_text[BYLINE].roles


def test_a_name_list_of_the_title_words_is_not_a_byline():
    # The short title in title case, filed under the title as a running head.
    short_title = "Shared Gardens and Meeting Places"
    contents = _contents([("text", short_title, TRANSLATED_BOX)], filed={short_title})

    texts = [candidate.raw_text for candidate in collect_front_matter_candidates(contents)]

    assert short_title not in texts


def test_names_in_the_authors_addresses_are_not_common_words():
    # The paper prints the authors' names in lower case only in their e-mail
    # addresses, written out or with "[at]".
    contents = _contents(
        COVER_BLOCK,
        filed={TRANSLATED, BYLINE},
        body=(
            "Contact: marie.dubois[at]example.org, pieter.janssens@example.org.",
            "Anne-Sophie Martin (anne-sophie.martin[at]example.org) and luc.vandamme@example.org.",
        ),
    )

    by_text = {
        candidate.raw_text: candidate for candidate in collect_front_matter_candidates(contents)
    }

    assert "byline" in by_text[BYLINE].roles


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (BYLINE, True),
        ("Anna Berg and Carl Dahl", True),
        ("Anna Berg & Carl Dahl", True),
        ("Anna Berg, Carl Dahl en Els de Vries", True),
        ("Anna Berg", False),  # one name
        ("Town Studies, Collection générale", False),
        ("Journal of Town Life, Ghent", False),
        ("MARIE DUBOIS, PIETER JANSSENS", False),  # a banner in capitals
        ("Marie Dubois, 2026", False),
        # Affiliations, in English and in the languages of the byline.
        ("Northfield University, Vrije Universiteit Zuid", False),
        ("Técnico de Investigación, Departamento de Horticultura", False),
    ],
)
def test_person_name_list_shape(text, expected):
    assert _looks_like_person_name_list(text) is expected
