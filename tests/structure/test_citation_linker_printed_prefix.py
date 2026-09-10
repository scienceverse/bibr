"""Printed reference-marker evidence and bibliography ownership.

Numeric fallback linking requires evidence that printed reference markers agree
with bibliography IDs. Bare leading numbers can occur in prose and can also
belong to a bibliography whose numbering differs from internal IDs. Broadening
marker punctuation alone would therefore permit links to the wrong references.

These synthetic cases retain the conservative grammar and exercise the explicit
marker/ID agreement check. The final test documents the remaining section-scan
limitation: unanchored rows are accepted without proving that agreement.
"""

from bibr.models import PaperReference
from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence
from bibr.structure.citation_linker import (
    _printed_reference_sources,
)


def _reference(bib_id: int, text_id: int | None = None) -> PaperReference:
    # text_id defaults to None: native/JATS references routinely lack it, which
    # is the path that falls back to scanning REFERENCES-typed sentences.
    return PaperReference(
        bib_id=bib_id,
        title=f"Study {bib_id}",
        first_page=None,
        volume=None,
        authors=None,
        year=None,
        container=None,
        text_id=text_id,
    )


def _sections() -> list[PaperSection]:
    return [
        PaperSection(
            section_id=1,
            header="Introduction",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.INTRODUCTION,
        ),
        PaperSection(
            section_id=2,
            header="References",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.REFERENCES,
        ),
    ]


def _sentence(text_id: int, text: str, section_id: int = 2) -> PaperSentence:
    return PaperSentence(text_id=text_id, text=text, section_id=section_id, paragraph_id=text_id)


def test_delimiter_free_numbering_is_not_printed_evidence():
    """The Vancouver bare-number form must stay unrecognised.

    Supporting this syntax requires a marker-to-bibliography agreement check,
    because punctuation alone cannot distinguish reference numbers from prose.
    """

    rows = [
        "1 Example Committee. A fictional planning study. Example City; 2020.",
        "2 Sample P, Reader A. A fictional survey of shared spaces. Example Press; 2021.",
        "3 Приклад П. Умовне дослідження. Київ; 2022.",
        "4 - García J. An invented classroom study. 2023.",
    ]
    sentences = [_sentence(100 + index, row) for index, row in enumerate(rows)]
    references = [_reference(bib_id) for bib_id in (1, 2, 3, 4)]

    assert set(_printed_reference_sources(sentences, _sections(), references)) == set()


def test_delimited_numbering_is_printed_evidence():
    """The delimited forms are accepted as printed evidence.

    A bracket, a paren, a superscript brace, or a period/paren terminator after
    a leading number is punctuation running prose does not produce.  The
    no-space period form ("12.Приклад") is included because this module's
    ``(\\d{1,4})[.)]`` never required a following space.
    """

    rows = [
        "[1] Smith J. A study of things. Journal of Things. 2019;4:11-20.",
        "(2) Jones A. Another study. Journal of Other Things. 2020;5:21-30.",
        "3. Brown K. A third study. Journal of Third Things. 2021;6:31-40.",
        "^{4} Green L. A fourth study. Journal of Fourth Things. 2022;7:41-50.",
        "5.Приклад П. Умовне дослідження. Київ; 2022.",
    ]
    sentences = [_sentence(100 + index, row) for index, row in enumerate(rows)]
    references = [_reference(bib_id) for bib_id in (1, 2, 3, 4, 5)]

    assert set(_printed_reference_sources(sentences, _sections(), references)) == {1, 2, 3, 4, 5}


def test_anchored_path_requires_marker_to_equal_bib_id():
    """A row reached through its own ``text_id`` must print its own number.

    This is the agreement check the section-scan path lacks.  Here every row
    prints a marker two positions ahead of the bib entry it IS, so the anchored
    path yields nothing.
    """

    sentences = [
        _sentence(100, "[3] Smith J. A study of things. 2019."),
        _sentence(101, "[4] Jones A. Another study. 2020."),
    ]
    references = [_reference(1, text_id=100), _reference(2, text_id=101)]
    sections = _sections()

    sources = _printed_reference_sources(sentences, sections, references)

    assert 1 not in sources
    assert 2 not in sources


def test_section_scan_admits_markers_without_row_ownership():
    """Documents a known section-scan ownership limitation.

    The same two rows as above, with the references' ``text_id`` cleared so the
    anchored path cannot run.  The section scan then admits markers 3 and 4 on
    nothing more than "3 and 4 are valid bib_ids" — it never checks that the row
    printing "3" is bib entry 3.  Evidence for bib entries 3 and 4 is
    manufactured out of rows belonging to entries 1 and 2.

    An agreement check is required before adding bare-number support. When
    ownership is enforced here, replace this characterization with its inverse.
    """

    sentences = [
        _sentence(100, "[3] Smith J. A study of things. 2019."),
        _sentence(101, "[4] Jones A. Another study. 2020."),
    ]
    references = [_reference(bib_id) for bib_id in (1, 2, 3, 4)]

    assert set(_printed_reference_sources(sentences, _sections(), references)) == {3, 4}
