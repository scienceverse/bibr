"""A printed references heading in any listed language beats the weaker guesses.

EXCITE dev: on 40753 the front-role model took the one-row German body section
"Auswertung" for the reference header and the "Literatur" list on p19 was never
read; on 32265 the layout fallback took the last unclassified section (a two-row
talk title) after the "Literatur" list.
"""

from __future__ import annotations

import pytest

from bibr.extract.front_role import FrontRolePredictions, RoleScores
from bibr.extract.ref_locator import RefLocator, _is_reference_heading
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    RegionSummary,
)

LIST = [
    "Brandstätter, H. (1999). Wirtschaftspsychologie. Göttingen: Hogrefe.",
    "Kirchler, E. (2007). The economic psychology of tax behaviour. Cambridge: CUP.",
    "Wahl, I. & Kirchler, E. (2006). Steuermoral. Wirtschaftspsychologie, 8, 13-22.",
]


def _contents(sections, rows, *, predictions=None, layout_hints=None) -> PaperContents:
    """PaperContents from ``(header, type)`` sections and ``(section_id, text)`` rows."""
    paper_sections = [PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN)]
    for section_id, (header, section_type) in enumerate(sections, start=1):
        paper_sections.append(PaperSection(section_id, header, 1, 0, section_type))
    sentences = [
        PaperSentence(
            text_id=text_id,
            text=text,
            section_id=section_id,
            paragraph_id=text_id,
            page_number=section_id,
        )
        for text_id, (section_id, text) in enumerate(rows, start=1)
    ]
    return PaperContents(
        sentences=sentences,
        sections=paper_sections,
        tables=[],
        links=[],
        sections_text={s.section_id: "" for s in paper_sections},
        region_summaries=[
            RegionSummary(
                page=s.section_id,
                index=0,
                label="paragraph_title",
                bbox=None,
                section_id=s.section_id,
            )
            for s in paper_sections[1:]
        ],
        front_role_predictions=predictions,
        layout_hints=layout_hints or [],
    )


def _ref_header_on(page: int) -> FrontRolePredictions:
    probs = {"ref_header": 0.95, "heading": 0.05}
    return FrontRolePredictions(
        {(page, 0): RoleScores(probs=probs, top="ref_header", confidence=0.95)}, model_version="t"
    )


def test_printed_german_heading_beats_the_front_role_header():
    contents = _contents(
        [
            ("Einleitung", CanonicalSection.INTRODUCTION),
            ("Auswertung", CanonicalSection.UNKNOWN),
            ("Ergebnisse", CanonicalSection.RESULTS),
            ("Literatur", CanonicalSection.UNKNOWN),
        ],
        [
            (1, "Die Mehrwertsteuer wurde 2007 erhöht."),
            (
                2,
                "Bei der Darstellung unserer Ergebnisse orientieren wir uns an den Fragestellungen.",
            ),
            (3, "Die Befragten erwarteten steigende Preise."),
            *((4, ref) for ref in LIST),
        ],
        predictions=_ref_header_on(2),
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert list(rows["text"]) == LIST
    assert contents.sections[2].section_type == CanonicalSection.UNKNOWN


def test_front_role_header_without_a_dated_row_is_not_the_list():
    # No printed heading the grammar knows; the model still points at a body heading.
    contents = _contents(
        [
            ("Einleitung", CanonicalSection.INTRODUCTION),
            ("Auswertung", CanonicalSection.UNKNOWN),
            ("Schrifttum", CanonicalSection.REFERENCES),
        ],
        [
            (1, "Die Mehrwertsteuer wurde erhöht."),
            (
                2,
                "Bei der Darstellung unserer Ergebnisse orientieren wir uns an den Fragestellungen.",
            ),
            *((3, ref) for ref in LIST),
        ],
        predictions=_ref_header_on(2),
    )

    locator = RefLocator(contents)
    rows = locator.collect_reference_rows()

    assert list(rows["text"]) == LIST
    assert "front_role_ref_header" not in contents.reference_boundary_reason_flags


def test_printed_heading_beats_the_last_unclassified_section():
    contents = _contents(
        [
            ("Fazit", CanonicalSection.DISCUSSION),
            ("Selected References", CanonicalSection.UNKNOWN),
            ("Soziale Nachhaltigkeit und Arbeit", CanonicalSection.UNKNOWN),
        ],
        [
            (1, "Die Hochschulen stehen vor neuen Aufgaben."),
            *((2, ref) for ref in LIST),
            (
                3,
                "Decent Work, Regulierung von Arbeitsbedingungen in globalen Wirtschaftsstrukturen.",
            ),
            (3, "(Guido Becke, Eva Senghaas-Knobloch)"),
        ],
        layout_hints=[("reference_content", 2)],
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert list(rows["text"]) == LIST
    assert contents.sections[3].section_type == CanonicalSection.UNKNOWN


@pytest.mark.parametrize(
    "heading",
    [
        "Zitierte und verwendete Quellen:",
        "Verwendete Literatur",
        "Literatur und Quellen",
        "Quellen und Literatur",
        "Quellenverzeichnis",
    ],
)
def test_german_source_headings_are_reference_headings(heading):
    assert _is_reference_heading(heading)


@pytest.mark.parametrize("heading", ["Quellen", "Quellen der Daten", "Auswertung"])
def test_bare_or_qualified_quellen_is_not_a_reference_heading(heading):
    # "Quellen:" alone also opens German table notes.
    assert not _is_reference_heading(heading)
