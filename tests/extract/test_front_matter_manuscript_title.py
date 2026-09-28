"""The title of a manuscript set entirely in body font seeds the front matter.

The real case is an anonymised submission: its first page prints the title as
one plain text row in the body font, then the submission dates, and the
abstract follows on page 2 under its own heading. No layout label, capitals or
heading marks the title row, and the front-role classifier scores it as
abstract text (0.90), so the record had no title seed and the paper exported
no title. Other text is invented.
"""

from __future__ import annotations

from bibr.extract.front_matter import collect_front_matter_candidates, resolve_front_matter
from bibr.extract.front_role import FrontRolePredictions
from bibr.extract.title_candidates import selected_title_candidate
from bibr.paper_contents import CanonicalSection, Provenance

from .test_front_matter_model_roles import (
    _contents,
    _scores,
    _section,
    _sentence,
    _summary,
)

_TITLE = (
    "Do construal levels affect athletes' imagery and performance outcomes? It depends on the task!"
)
_TITLE_BBOX = (90.0, 190.0, 864.0, 207.0)
_DATES_BBOX = (365.0, 265.0, 634.0, 282.0)
_HEADING_BBOX = (479.0, 73.0, 567.0, 93.0)
_ABSTRACT_BBOX = (90.0, 107.0, 899.0, 356.0)


def _manuscript(
    title=_TITLE,
    *,
    heading="Abstract",
    heading_type=CanonicalSection.ABSTRACT,
    title_scores=None,
):
    abstract_section = _section(1, heading, heading_type, level=2)
    abstract_section.provenance = [Provenance(page_no=2, bbox=_HEADING_BBOX)]
    sentences = [
        _sentence(1, title, paragraph_id=1, bbox=_TITLE_BBOX),
        _sentence(
            2,
            "Date of Submission: 15.03.2016 Date of re-submission: 12.06.2016",
            paragraph_id=2,
            bbox=_DATES_BBOX,
        ),
        _sentence(
            3,
            "We tested whether the framing of an imagery script changes how well athletes "
            "perform a reactive task.",
            paragraph_id=3,
            bbox=_ABSTRACT_BBOX,
            label="text",
            section_id=1,
            page=2,
        ),
    ]
    summaries = [
        _summary(1, "text", _TITLE_BBOX),
        _summary(2, "text", _DATES_BBOX),
        _summary(0, "paragraph_title", _HEADING_BBOX, section_id=1, page=2),
        _summary(1, "text", _ABSTRACT_BBOX, section_id=1, page=2),
    ]
    predictions = FrontRolePredictions(
        {
            (1, 1): title_scores or _scores(abstract=0.90),
            (1, 2): _scores(other=0.99),
            (2, 0): _scores(heading=0.99),
            (2, 1): _scores(abstract=0.95),
        },
        model_version="test",
    )
    return _contents(
        sentences,
        summaries,
        sections=[_section(0, "Root"), abstract_section],
        predictions=predictions,
    )


def _first(candidates):
    return next(candidate for candidate in candidates if candidate.page == 1)


def test_body_font_title_row_above_the_abstract_heading_seeds_the_title():
    contents = _manuscript()

    candidates = collect_front_matter_candidates(contents)

    first = _first(candidates)
    assert first.raw_text == _TITLE
    assert first.roles == frozenset({"title"})
    assert first.model_roles == frozenset()
    resolution, _issues = resolve_front_matter(contents)
    candidate, _reason = selected_title_candidate(resolution, journal=None, publisher=None)
    assert candidate is not None
    assert candidate.value == _TITLE


def test_without_a_printed_abstract_heading_the_row_stays_abstract_text():
    candidates = collect_front_matter_candidates(
        _manuscript(heading="Introduction", heading_type=CanonicalSection.INTRODUCTION)
    )

    assert not any("title" in candidate.roles for candidate in candidates)
    assert "abstract" in _first(candidates).roles


def test_a_first_row_closed_like_prose_is_not_seeded():
    sentence = "We asked whether construal levels affect imagery and performance outcomes."

    candidates = collect_front_matter_candidates(_manuscript(sentence))

    assert not any("title" in candidate.roles for candidate in candidates)


def test_a_first_row_with_its_own_evidence_is_not_seeded():
    affiliation = "Department of Sport Science, University of Somewhere, Somewhere"

    candidates = collect_front_matter_candidates(_manuscript(affiliation))

    assert not any("title" in candidate.roles for candidate in candidates)


def test_a_page_with_a_title_seed_keeps_its_rows_as_they_are():
    # The classifier names the row a title: the ordinary pass seeds it, and
    # the last resort must not also run.
    candidates = collect_front_matter_candidates(_manuscript(title_scores=_scores(title=0.9)))

    first = _first(candidates)
    assert "title" in first.roles
    assert first.model_roles == frozenset({"title"})
