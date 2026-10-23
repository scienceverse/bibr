"""A Word manuscript's bare "Abstract" paragraph opens the abstract span.

Word authors often type the label as a bold Normal paragraph rather than a
Heading style, so the native DOCX parse keeps it as a body row with the
abstract role. The span selector only opened at a heading, so the abstract
under it was never grounded: an empty span on 24 of 40 sampled PsyArXiv DOCX
files that printed one.
"""

from bibr.extract.front_matter import (
    FrontMatterBlock,
    FrontMatterCandidate,
    FrontMatterResolution,
)
from bibr.paper_contents import PaperContents, PaperSection, PaperSentence
from bibr.structure.implicit_sections import select_abstract_span

ABSTRACT = (
    "Attentional control deficits are often invoked to explain differences in how "
    "people deal with distraction, and we test that account here."
)


def _select(rows: list[tuple[str, set[str]]], *, page: int | None = None):
    sentences = [
        PaperSentence(index, text, 1, index, page_number=page)
        for index, (text, _) in enumerate(rows, start=1)
    ]
    candidates = tuple(
        FrontMatterCandidate(
            candidate_id=f"c{index}",
            source_kind="paragraph",
            reading_order=index,
            page=page,
            bbox=None,
            region_label=None,
            font_size=None,
            font_bold=None,
            section_id=1,
            text_ids=(index,),
            paragraph_id=index,
            raw_text=text,
            normalized_text=" ".join(text.casefold().split()),
            roles=frozenset(roles),
        )
        for index, (text, roles) in enumerate(rows, start=1)
    )
    contents = PaperContents(
        sentences=sentences,
        sections=[PaperSection(0, "Root", 0, None), PaperSection(1, "Root", 1, 0)],
        tables=[],
        links=[],
        sections_text={},
    )
    resolution = FrontMatterResolution(
        candidates=candidates,
        blocks=(
            FrontMatterBlock(
                block_id="selected",
                candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
                title_candidate_ids=(),
            ),
        ),
        selected_block_id="selected",
        selection_method="test",
        reason_flags=(),
        allowed_text_ids=frozenset(range(1, len(rows) + 1)),
        allowed_section_ids=frozenset({1}),
    )
    return select_abstract_span(contents, resolution)


def test_a_bare_abstract_paragraph_opens_the_span_without_its_label():
    selected = _select(
        [
            ("Putting Attentional Control to a Normative Test", {"title"}),
            ("Dominik Grätz", {"byline"}),
            ("Abstract", {"abstract"}),
            (ABSTRACT, set()),
            ("Keywords: attention, aging", set()),
        ]
    )

    assert selected.text == ABSTRACT
    assert selected.reason_flags == ("explicit_abstract",)


def test_the_span_stops_at_an_unstyled_heading_after_the_label():
    for heading in ("Introduction", "Putting Attentional Control to a Normative Test"):
        selected = _select(
            [
                ("Abstract:", {"abstract"}),
                (ABSTRACT, set()),
                (heading, set()),
                ("The purpose of attentional selection is to focus the mind.", set()),
            ]
        )

        assert selected.text == ABSTRACT


def test_a_second_abstract_paragraph_is_kept():
    second = "The findings show that the effect is stable across groups."

    selected = _select([("Abstract", {"abstract"}), (ABSTRACT, set()), (second, set())])

    assert selected.text == f"{ABSTRACT} {second}"


def test_the_label_needs_the_abstract_role():
    assert _select([("Abstract", set()), (ABSTRACT, set())]).text == ""


def test_a_paged_abstract_paragraph_is_left_to_the_layout():
    assert _select([("Abstract", {"abstract"}), (ABSTRACT, set())], page=1).text == ""
