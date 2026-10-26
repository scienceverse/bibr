"""A DOI cut short by a text-layer region's box never outranks the full DOI.

10.1177_11779322231160397 (SAGE): the text-layer header box ended the line
"https://doi.org/10.1177/11779322231160397" at "...311603", so the header gave
a truncated candidate beside the printed DOI. The native fill marks a region
whose box cuts a printed line (``RegionSummary.native_text_edge_cut``); a DOI
from it that a longer candidate extends by an end a cut can lose is rejected
as ``edge_truncated``.
"""

from __future__ import annotations

from bibr.extract.doi_identity import (
    EDGE_TRUNCATED,
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
