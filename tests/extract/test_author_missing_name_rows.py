"""A byline row that names nobody is not an author the list is missing.

The byline role also lands on a letter's salutation, a "Cite This:" line, the
article history, a copyright line or a journal banner. Each reads as one
two-word name, so VAL_AUTHOR_MISSING fired on PMC papers whose extracted
author list matched JATS exactly.
"""

from __future__ import annotations

from bibr.extract.core_metadata import assess_author_grounding, build_byline_group
from bibr.extract.front_matter import (
    FrontMatterBlock,
    FrontMatterCandidate,
    FrontMatterResolution,
)
from bibr.paper import PaperAuthor


def _row(index: int, text: str) -> FrontMatterCandidate:
    return FrontMatterCandidate(
        candidate_id=f"c{index}",
        source_kind="paragraph",
        reading_order=index,
        page=1,
        bbox=None,
        region_label="text",
        font_size=None,
        font_bold=None,
        section_id=1,
        text_ids=(index,),
        paragraph_id=None,
        raw_text=text,
        normalized_text=" ".join(text.casefold().split()),
        roles=frozenset({"byline"}),
    )


def _codes(byline_rows: list[str], authors: list[tuple[str, str]]) -> dict[str, int]:
    rows = tuple(_row(index, text) for index, text in enumerate(byline_rows, start=1))
    resolution = FrontMatterResolution(
        candidates=rows,
        blocks=(
            FrontMatterBlock(
                block_id="selected",
                candidate_ids=tuple(row.candidate_id for row in rows),
                title_candidate_ids=(),
            ),
        ),
        selected_block_id="selected",
        selection_method="unique_block",
        reason_flags=(),
        allowed_text_ids=frozenset(row.text_ids[0] for row in rows),
        allowed_section_ids=frozenset({1}),
    )
    extracted = [
        PaperAuthor(author_id=index, given=given, family=family, affiliation="")
        for index, (given, family) in enumerate(authors, start=1)
    ]
    issues = assess_author_grounding(extracted, build_byline_group(resolution))
    return {issue.code: issue.count for issue in issues}


def test_a_letters_salutation_is_not_a_missing_author():
    codes = _codes(["Takenori Mishiba1,2, 1", "Dear Editor"], [("Takenori", "Mishiba")])

    assert codes == {}


def test_citation_history_copyright_and_banner_rows_are_not_missing_authors():
    rows = [
        "Elvira Sattarova 1*, Saman Lashkari1, Frederik Rask Dalby 2",
        "Cite This: ACS Omega 2026, 11, 40460−40467",
        "Accepted: July 22, 2026",
        "Received: April 29, 2026",
        "© The Author(s) 2026",
        "JMIR MEDICAL EDUCATION",
        "Published: 4 September 2026 Copyright: © 2026 by the authors. This article is "
        "distributed under the terms and conditions of the Creative Commons Attribution license",
    ]
    authors = [("Elvira", "Sattarova"), ("Saman", "Lashkari"), ("Frederik Rask", "Dalby")]

    assert _codes(rows, authors) == {}


def test_a_dropped_row_with_affiliation_markers_is_still_missing():
    rows = ["Ana Silva1, Jonas Byrne2", "Kim Lee3"]

    codes = _codes(rows, [("Ana", "Silva"), ("Jonas", "Byrne")])

    assert codes == {"VAL_AUTHOR_MISSING": 1}


def test_a_dropped_row_of_two_plain_names_is_still_missing():
    rows = ["Ana Silva, Jonas Byrne", "Kim Lee, Tomas Ford"]

    codes = _codes(rows, [("Ana", "Silva"), ("Jonas", "Byrne")])

    assert codes == {"VAL_AUTHOR_MISSING": 2}


def test_a_lone_row_is_evidence_when_no_author_is_found_at_all():
    codes = _codes(["Dear Editor"], [("Takenori", "Mishiba")])

    assert codes == {"VAL_AUTHOR_UNGROUNDED": 1, "VAL_AUTHOR_MISSING": 1}
