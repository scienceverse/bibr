"""A translated title under the title is not the byline that stops probation.

Cut down from a CC BY trilingual nursing article (PMC13497355): the English
title, then its Portuguese and Spanish titles in one paragraph, then one bold
"Name ORCID: ..." row per author that the section classifier promoted to an
acknowledgment heading. The translated titles read as a name list (capitalized
chunks, commas, "de" particles), so the page looked like it printed a byline,
byline probation never ran, and the paper came out with no authors.
"""

from __future__ import annotations

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
)

TITLE = (
    "Symptoms of Depression, Anxiety, and Stress Among Nursing Students and Coping Strategies Used"
)
TRANSLATED = (
    "Sintomas de Depressão, Ansiedade, Estresse entre Estudantes de Enfermagem e Estratégias de "
    "Coping Utilizadas Síntomas de Depresión, Ansiedad y Estrés entre Estudiantes de Enfermería "
    "y Estrategias de Afrontamiento Utilizadas"
)
AUTHOR_ROWS = (
    "Ana Paula Porto CruzI ORCID: 0009-0003-9057-8665",
    "Bárbara Beatriz Coelho LimaIV ORCID: 0009-0002-1066-3222",
)


def _section(
    section_id: int, header: str, section_type: CanonicalSection, *, top: float
) -> PaperSection:
    return PaperSection(
        section_id=section_id,
        header=header,
        level=0 if section_id == 0 else 1,
        parent_section_id=None if section_id == 0 else 0,
        section_type=section_type,
        provenance=[]
        if section_id == 0
        else [Provenance(page_no=1, bbox=(90.0, top, 500.0, top + 15.0))],
    )


def _sentence(
    text_id: int, text: str, *, section_id: int, paragraph_id: int, top: float
) -> PaperSentence:
    return PaperSentence(
        text_id=text_id,
        text=text,
        section_id=section_id,
        paragraph_id=paragraph_id,
        page_number=1,
        provenance=[Provenance(page_no=1, bbox=(90.0, top, 500.0, top + 20.0))],
        region_meta={"region_type": "text", "font_size": 9.0, "font_bold": False},
    )


def _contents(second_row: str) -> PaperContents:
    sections = [
        _section(0, "Root", CanonicalSection.UNKNOWN, top=0.0),
        _section(1, TITLE, CanonicalSection.TITLE, top=60.0),
        _section(2, AUTHOR_ROWS[0], CanonicalSection.ACKNOWLEDGMENT, top=160.0),
        _section(3, AUTHOR_ROWS[1], CanonicalSection.ACKNOWLEDGMENT, top=200.0),
        _section(4, "ABSTRACT", CanonicalSection.ABSTRACT, top=140.0),
    ]
    sentences = [
        _sentence(1, second_row, section_id=1, paragraph_id=1, top=100.0),
        _sentence(
            2,
            "IUniversidade Federal de Rondonópolis. Rondonópolis, Mato Grosso, Brazil.",
            section_id=3,
            paragraph_id=2,
            top=230.0,
        ),
        _sentence(
            3,
            "Objectives: to analyze symptoms of depression among nursing students.",
            section_id=4,
            paragraph_id=3,
            top=160.0,
        ),
    ]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={section.section_id: "" for section in sections},
        detected_title=TITLE,
        region_summaries=[],
    )


def _probation_rows(contents: PaperContents) -> list[str]:
    return [
        candidate.raw_text
        for candidate in collect_front_matter_candidates(contents)
        if BYLINE_PROBATION_ROLE in candidate.roles
    ]


def test_translated_titles_leave_probation_to_read_the_author_headings():
    contents = _contents(TRANSLATED)

    resolution, _ = resolve_front_matter(contents, target_required=True)
    context = render_author_context(resolution, full_text=render_block_context(resolution))

    assert [row for row in AUTHOR_ROWS if row in _probation_rows(contents)] == list(AUTHOR_ROWS)
    assert "Ana Paula Porto Cruz" in context
    assert "Bárbara Beatriz Coelho Lima" in context


def test_a_byline_in_another_language_still_suppresses_probation():
    contents = _contents("Maria Clara Santana da Silva, Ana Paula de Souza e João dos Santos")

    assert _probation_rows(contents) == []
