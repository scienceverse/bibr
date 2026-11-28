"""A manuscript title set as two paragraphs, the first ending on a colon."""

from __future__ import annotations

from bibr.extract.front_matter import (
    FrontMatterCandidate,
    _record_title_indices,
    _select_agreeing_record,
    group_front_matter_blocks,
)

TITLE = "Becoming a climate activist:"
SUBTITLE = "A qualitative investigation of movement participation"


def _candidate(order: int, text: str, roles: list[str]) -> FrontMatterCandidate:
    return FrontMatterCandidate(
        candidate_id=f"front-matter-candidate-{order}",
        source_kind="paragraph",
        reading_order=order,
        page=None,
        bbox=None,
        region_label=None,
        font_size=None,
        font_bold=None,
        section_id=0,
        text_ids=(order,),
        paragraph_id=order,
        raw_text=text,
        normalized_text=text.casefold(),
        roles=frozenset(roles),
        model_roles=frozenset(),
        model_scores=(),
    )


# A .docx title page (title, byline, affiliation, notes) and then the
# manuscript, which prints the title again above its abstract.
TITLE_PAGE = (
    _candidate(1, TITLE, ["title"]),
    _candidate(2, SUBTITLE, []),
    _candidate(3, "Anna A. Sach, Cameron Brick, Roos Herings and Nils B. Jostmann", ["byline"]),
    _candidate(4, "Department of Psychology, University of Amsterdam", ["affiliation"]),
    _candidate(5, "Corresponding author: a.a.sach@uva.nl", []),
)
MANUSCRIPT = (
    _candidate(6, TITLE, ["title", "heading"]),
    _candidate(7, SUBTITLE, ["title", "heading"]),
    _candidate(8, "Abstract", ["abstract", "heading"]),
    _candidate(
        9, "Climate movements have attracted many newcomers in the past years.", ["abstract"]
    ),
)


def test_the_row_after_a_title_ending_on_a_colon_roots_no_record():
    candidates = TITLE_PAGE + MANUSCRIPT

    roots = _record_title_indices(candidates, allow_byline_only=True)

    assert roots == frozenset({0, 5})
    blocks = group_front_matter_blocks(candidates)
    assert [len(block.candidate_ids) for block in blocks] == [5, 4]


def test_the_title_page_record_is_selected_over_the_reprinted_title():
    candidates = TITLE_PAGE + MANUSCRIPT
    blocks = group_front_matter_blocks(candidates)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}

    selected, _ = _select_agreeing_record(blocks, by_id, detected_title=TITLE)

    assert selected is not None
    assert selected.block_id == blocks[0].block_id


def test_a_title_row_without_a_colon_still_lets_the_next_title_root_a_record():
    candidates = (
        _candidate(1, "Becoming a climate activist", ["title"]),
        _candidate(2, SUBTITLE, ["title"]),
        _candidate(3, "Abstract", ["abstract", "heading"]),
    )

    assert _record_title_indices(candidates, allow_byline_only=True) == frozenset({1})
