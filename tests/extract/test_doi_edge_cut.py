"""A DOI cut short by a text-layer region's box never outranks the full DOI.

In a SAGE paper the text-layer header box ended the line
"https://doi.org/10.1177/11779322231160397" at "...311603", so the header gave
a truncated candidate beside the printed DOI. The native fill marks a region
whose box cuts a printed line (``RegionSummary.native_text_edge_cut``); a DOI
from it that a longer candidate extends by an end a cut can lose is rejected
as ``edge_truncated``.
"""

from __future__ import annotations

from dataclasses import replace

from bibr.extract.doi_identity import (
    EDGE_TRUNCATED,
    LINE_JOIN_OVERRUN,
    _reject_edge_cuts,
    collect_doi_candidates,
    select_doi_candidates,
)
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, RegionSummary

_CUT_HEADER = "DOI: 10.1177/117793222311603\nBioinformatics and Biology Insights\nVolume 17: 1-14"
_FULL_FOOTER = "https://doi.org/10.1177/11779322231160397"


def _contents(headers: list[str], footers: list[str], *, cut: bool) -> PaperContents:
    return PaperContents(
        sentences=[],
        sections=[PaperSection(0, "Root", 0, None, CanonicalSection.TITLE)],
        tables=[],
        links=[],
        sections_text={},
        detected_headers=headers,
        detected_footers=footers,
        region_summaries=[
            RegionSummary(
                page=1,
                index=0,
                label="header",
                bbox=None,
                content=headers[0][:200],
                source="text_layer",
                native_text_edge_cut=cut,
            )
        ],
    )


def _by_value(selection) -> dict[str, str | None]:
    return {c.normalized: c.rejection_reason for c in selection.candidates}


def test_cut_short_header_doi_cannot_outrank_the_full_doi():
    selection = select_doi_candidates(
        collect_doi_candidates(_contents([_CUT_HEADER], [_FULL_FOOTER], cut=True))
    )

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1177/11779322231160397"
    assert _by_value(selection)["10.1177/117793222311603"] == EDGE_TRUNCATED


def test_without_a_cut_the_header_doi_keeps_its_rank():
    """The rule needs the geometry: a shorter DOI alone proves nothing."""
    selection = select_doi_candidates(
        collect_doi_candidates(_contents([_CUT_HEADER], [_FULL_FOOTER], cut=False))
    )

    assert _by_value(selection)["10.1177/117793222311603"] is None


def test_a_component_extension_does_not_reject_the_article_doi():
    """PLOS figure DOIs extend the article DOI with letters; they are other DOIs."""
    header = "PLOS ONE | https://doi.org/10.1371/journal.pone.0279511"
    footer = "https://doi.org/10.1371/journal.pone.0279511.g001"
    selection = select_doi_candidates(
        collect_doi_candidates(_contents([header], [footer], cut=True))
    )

    assert _by_value(selection)["10.1371/journal.pone.0279511"] is None


def test_a_rejected_longer_doi_does_not_reject_the_cut_one():
    """A cited reference that happens to extend the DOI is no evidence of a cut."""
    found = collect_doi_candidates(_contents([_CUT_HEADER], [_FULL_FOOTER], cut=False))
    cut = next(c for c in found if c.normalized == "10.1177/117793222311603")
    cited = replace(
        next(c for c in found if c.normalized == "10.1177/11779322231160397"),
        rejection_reason="reference_candidate",
    )

    kept = _reject_edge_cuts([cut, cited], {id(cut)})

    assert kept[0].rejection_reason is None


# --- the page number glued to a running footer's DOI -------------------------
#
# A running footer prints the DOI and then the page number. From page 10 on
# the text layer of one layout leaves no gap between them, so every later page
# gives a DOI of its own, and the box of the earlier footers cuts their line:
# the real DOI was rejected as cut short and the paper had none.

_FOOTER = "March 2021 Volume 3 Issue 2 10.1234/journal.00777-21"
_STEM = "10.1234/journal.00777-21"


def _running_footers(spaced: list[int], glued: list[int], *, cut: bool = False) -> PaperContents:
    footers = [f"{_FOOTER} {page}" for page in spaced] + [f"{_FOOTER}{page}" for page in glued]
    return PaperContents(
        sentences=[],
        sections=[PaperSection(0, "Root", 0, None, CanonicalSection.TITLE)],
        tables=[],
        links=[],
        sections_text={},
        detected_headers=[],
        detected_footers=footers,
        region_summaries=[
            RegionSummary(
                page=page,
                index=0,
                label="footer",
                bbox=None,
                content=f"{_FOOTER} {page}",
                source="text_layer",
                native_text_edge_cut=cut,
            )
            for page in spaced
        ],
    )


def test_a_page_number_glued_to_the_footer_doi_is_not_another_doi():
    selection = select_doi_candidates(
        collect_doi_candidates(_running_footers([8, 9], [10, 11, 12]))
    )

    assert selection.selected is not None
    assert selection.selected.normalized == _STEM
    assert selection.issues == ()
    reasons = _by_value(selection)
    assert [reasons[f"{_STEM}{page}"] for page in (10, 11, 12)] == [LINE_JOIN_OVERRUN] * 3


def test_glued_page_numbers_do_not_make_the_footer_doi_a_cut_one():
    """The mined case: the footers that kept the gap sit in boxes that cut their line."""
    selection = select_doi_candidates(
        collect_doi_candidates(_running_footers([8, 9], [10, 11, 12], cut=True))
    )

    assert selection.selected is not None
    assert selection.selected.normalized == _STEM
    assert _by_value(selection)[_STEM] is None


def test_one_longer_footer_doi_is_not_a_page_number():
    """A single extension may be the full DOI beside a cut one: the cut rule decides."""
    selection = select_doi_candidates(
        collect_doi_candidates(_running_footers([8, 9], [10], cut=True))
    )

    reasons = _by_value(selection)
    assert reasons[f"{_STEM}10"] is None
    assert reasons[_STEM] == EDGE_TRUNCATED


def test_numeric_ends_far_apart_are_not_page_numbers():
    """Two readings of one DOI's last digits ("...34", "...84") are not a page count."""
    selection = select_doi_candidates(collect_doi_candidates(_running_footers([8], [34, 84])))

    reasons = _by_value(selection)
    assert reasons[f"{_STEM}34"] is None
    assert reasons[f"{_STEM}84"] is None
