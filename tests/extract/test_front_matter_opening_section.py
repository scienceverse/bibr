"""A body-typed opening section is read as front matter when nothing else is.

The real case is a short commentary: its title is the header of its only body
section, the author and affiliation are that section's first row, and the
whole text follows under it. The section classifier typed the section as an
introduction, so no row reached front matter and the paper abstained with no
candidate at all. All text below is invented.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import bibr.extract.front_matter as front_matter
from bibr.extract.front_matter import collect_front_matter_candidates, resolve_front_matter
from bibr.paper_contents import CanonicalSection, PaperContents

from .test_front_matter_record_agreement import _contents, _row, _section, _selected_text

BYLINE = "Maria J. Example (Example University, New Zealand)"


def _commentary(
    *,
    header: str = "The logic of confidence intervals1",
    first_row: str = BYLINE,
    section_type: CanonicalSection = CanonicalSection.INTRODUCTION,
    first_row_section: int = 1,
    first_row_page: int = 1,
    header_page: int = 1,
    synthetic_header: bool = False,
) -> PaperContents:
    opening = _section(
        1,
        header,
        section_type=section_type,
        bbox=(60.0, 60.0, 460.0, 90.0),
        page=header_page,
    )
    return _contents(
        [
            _row(1, first_row, y=110.0, section_id=first_row_section, page=first_row_page),
            _row(
                2,
                "Several authors recently argued that confidence intervals are misread.",
                y=150.0,
                section_id=1,
            ),
            _row(
                3, "Their argument rests on a syllogism about probabilities.", y=190.0, section_id=1
            ),
            _row(
                4,
                "Example, M. J. (2015). Intervals. Journal of Examples, 1, 1-10.",
                y=100.0,
                section_id=2,
                page=3,
            ),
        ],
        sections=[
            _section(0, "Root"),
            replace(opening, header_is_synthetic=True) if synthetic_header else opening,
            _section(
                2,
                "References",
                section_type=CanonicalSection.REFERENCES,
                bbox=(60.0, 60.0, 460.0, 80.0),
                page=3,
            ),
        ],
    )


def test_an_opening_section_typed_as_introduction_is_read_as_front_matter():
    contents = _commentary()

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selection_method == "unique_block"
    selected = _selected_text(resolution)
    assert "The logic of confidence intervals1" in selected
    assert BYLINE in selected
    assert issues == ()
    # The paper's own sections keep their types.
    assert contents.sections[1].section_type == CanonicalSection.INTRODUCTION


@pytest.mark.parametrize(
    ("header", "first_row"),
    [
        # No person named in the first row.
        ("The logic of confidence intervals1", "Several authors recently argued this point."),
        # A name without an initial, superscript or affiliation.
        ("The logic of confidence intervals1", "Maria Example"),
        # An ordinary body heading is not a title.
        ("Introduction", BYLINE),
        # A numbered heading is not a title.
        ("1. The logic of confidence intervals", BYLINE),
    ],
)
def test_an_opening_section_without_title_and_byline_evidence_stays_out(header, first_row):
    contents = _commentary(header=header, first_row=first_row)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert collect_front_matter_candidates(contents) == ()
    assert resolution.selection_method == "no_candidates"
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_a_page_with_candidates_never_reads_its_opening_section(monkeypatch):
    contents = _commentary(section_type=CanonicalSection.UNKNOWN)
    monkeypatch.setattr(
        front_matter,
        "_opening_section_as_front_matter",
        lambda *args, **kwargs: pytest.fail("the seed runs only when nothing is found"),
    )

    candidates = collect_front_matter_candidates(contents)

    assert candidates


def test_the_default_commentary_opens_with_a_readable_section():
    # Each guard case below differs from this page in one respect only.
    assert front_matter._opening_section_as_front_matter(_commentary()) is not None


@pytest.mark.parametrize(
    "overrides",
    [
        # The section is already typed as front matter and still gave no row.
        {"section_type": CanonicalSection.UNKNOWN},
        {"section_type": CanonicalSection.TITLE},
        # A header the parser made up is no printed title.
        {"synthetic_header": True},
        # The first page does not open with the section.
        {"header_page": 2},
        # One word, or more than thirty, is no title.
        {"header": "Intervals"},
        {"header": " ".join(["interval"] * 31)},
        # An ordinary body heading of several words is no title either.
        {"header": "Materials and Methods"},
        # The first row belongs to another section, or to a later page.
        {"first_row_section": 2},
        {"first_row_page": 2},
        # A long first row is prose, even when it names a person.
        {"first_row": BYLINE + " " + "The argument is restated here at length. " * 8},
        # Initials and an affiliation, but no surname.
        {"first_row": "M. J. (University of Examples, New Zealand)"},
    ],
    ids=[
        "typed-unknown",
        "typed-title",
        "synthetic-header",
        "header-on-page-2",
        "one-word-header",
        "long-header",
        "ordinary-heading",
        "first-row-in-another-section",
        "first-row-on-page-2",
        "long-first-row",
        "no-surname",
    ],
)
def test_the_opening_section_stays_out_when_one_guard_fails(overrides):
    assert front_matter._opening_section_as_front_matter(_commentary(**overrides)) is None
