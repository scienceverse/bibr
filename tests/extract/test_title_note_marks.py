"""A note marker or review-status tag the model copied leaves the title.

The real cases: a nursing journal article whose title row ends "... solid
waste management1" over a note "1 Paper extracted from Doctoral
Dissertation ...", with its authors marked 2 to 6; a scanned agricultural
journal article whose OCR sets the title's note marker as "vivero¹" but lost
the number of the note itself, with its authors marked ² to ⁵; a Portuguese
article whose title carries two notes glued as "BIBLIOMÉTRICO12", with its
authors marked 3 to 6; an open-review article whose title row carries its
review status "[version 1; peer review: 1 approved, 2 approved with
reservations]"; and a scanned conference paper whose title ends "Companies *"
over a note "* This is a revised and condensed version ...". The guard case
is a Portuguese article whose title ends "exercise1" while its only author's
affiliation is marked 1 too. The authors' names and other text are invented.
"""

import dataclasses

import pytest

from bibr.extract.title_marks import drop_title_note_marker
from bibr.paper_contents import RegionSummary
from bibr.schemas import AuthorLLM, CoreMetadataLLM

from .test_core_metadata_author_guards import _candidate, _contents, _extractor, _resolution

_WASTE_TITLE = (
    "Building sustainability indicators in the health dimension for solid waste management"
)
_WASTE_BYLINE = (
    "Beatriz Lima Rocha2 Carlos da Silva Prado3 Marta Sofia Pires Luz4 "
    "Joana Alves Reis5 Paula Maria Dias Moura6"
)
_WASTE_NOTES = (
    "1 Paper extracted from Doctoral Dissertation, presented to a nursing school.",
    "2 PhD, Professor Adjunto, Departamento de Engenharia Ambiental.",
    "3 PhD, Adjunct Professor, Departamento de Educação Física.",
)


def _record(title_row, byline, *, byline_roles=frozenset({"byline"})):
    return _resolution(
        _candidate("c1", title_row, roles=frozenset({"title", "heading"}), source_kind="heading"),
        _candidate("c2", byline, roles=byline_roles, text_ids=(1,)),
        _candidate("c3", "Abstract text of the article.", roles=frozenset({"abstract"})),
    )


def _page(*notes, title_row="", label="footnote"):
    contents = _contents()
    contents.region_summaries = [
        RegionSummary(page=1, index=0, label="doc_title", bbox=None, content=title_row),
        *(
            RegionSummary(page=1, index=10 + number, label=label, bbox=None, content=note)
            for number, note in enumerate(notes)
        ),
    ]
    return contents


def test_a_digit_the_first_note_on_the_title_page_starts_with_is_dropped():
    title = f"{_WASTE_TITLE}1"

    kept, issue = drop_title_note_marker(
        title, _record(title, _WASTE_BYLINE), _page(*_WASTE_NOTES, title_row=title)
    )

    assert kept == _WASTE_TITLE
    assert issue is not None
    assert issue.code == "VAL_TITLE_REGROUNDED"
    assert issue.evidence_ids == ("c1", "reason:title_note_marker")
    assert not issue.blocking


def test_a_digit_the_first_note_starts_with_is_dropped_under_an_unmarked_byline():
    # No byline marker to continue from: the note's number is the evidence.
    title = f"{_WASTE_TITLE}1"
    byline = "Beatriz Lima Rocha, Carlos da Silva Prado"

    kept, issue = drop_title_note_marker(title, _record(title, byline), _page(*_WASTE_NOTES[:1]))

    assert kept == _WASTE_TITLE
    assert issue is not None


@pytest.mark.parametrize(
    ("title", "notes", "label"),
    [
        # A numbered row on the title page that is body text, not a note.
        (f"{_WASTE_TITLE}1", _WASTE_NOTES[:1], "text"),
        (f"{_WASTE_TITLE}1", _WASTE_NOTES[:1], "footer"),
        # The stem left without the digit is too short to be a title.
        ("On pellagra1", _WASTE_NOTES[:1], "footnote"),
        # The only note that opens with a star opens with two of them.
        (
            "Some Considerations on the Empirical Research of Goal Systems of Insurance Companies *",
            ("** Translated from the German by the editors.",),
            "footnote",
        ),
    ],
)
def test_a_mark_without_a_title_page_note_of_its_own_is_kept(title, notes, label):
    record = _record(title, "Beatriz Lima Rocha, Carlos da Silva Prado")

    assert drop_title_note_marker(title, record, _page(*notes, label=label)) == (title, None)


def test_a_superscript_digit_before_the_byline_markers_is_dropped():
    # The OCR lost the note's own number, so the byline's markers carry the
    # evidence: the title is note 1 and the authors are 2 to 5.
    stem = "Patogenicidad de Myrothecium roridum y Rhizoctonia solani en cafetos en el vivero"
    byline = "Rosa del P. Marín², Luis Ortega³, Pedro Gómez⁴\ny Óscar Vidal⁵"
    notes = ("Manuscrito sometido a la junta editorial.", r"\(^{2}\)Investigadora, Departamento.")

    kept, issue = drop_title_note_marker(f"{stem}¹", _record(f"{stem}¹", byline), _page(*notes))

    assert kept == stem
    assert issue is not None


def test_two_glued_digits_read_as_the_first_two_notes_are_dropped():
    stem = "POSSIBILIDADES E DESAFIOS DA OFERTA DE LIBRAS NO BRASIL: UM ESTUDO BIBLIOMÉTRICO"
    byline = "Clara Mendes Sato3\r\nUniversidade Federal do Tocantins4\r\nPalmas, Tocantins"
    notes = ("1 Editora responsável pela avaliação.", "2 Copyright © 2022.", "3 e-mail")

    kept, _ = drop_title_note_marker(f"{stem}12", _record(f"{stem}12", byline), _page(*notes))

    assert kept == stem


def test_the_review_status_tag_after_the_title_is_dropped():
    stem = (
        "Co-production of guidance and resources to implement principled participant "
        "information leaflets (PrinciPILs)"
    )
    tag = "[version 1; peer review: 1 approved, 2 approved with reservations]"
    # The title row breaks inside the tag; the model joins the pieces.
    record = _resolution(
        _candidate(
            "c1",
            f"{stem} [version 1; peer review: 1 approved, 2 approved with",
            roles=frozenset({"title", "heading"}),
            source_kind="heading",
        ),
        _candidate("c2", "reservations]", roles=frozenset(), text_ids=(2,)),
        _candidate("c3", "Ana Silva1, Rui Costa1,2", roles=frozenset({"byline"})),
    )

    kept, issue = drop_title_note_marker(f"{stem} {tag}", record, _page())

    assert kept == stem
    assert issue is not None
    assert issue.evidence_ids == ("reason:title_review_status_tag",)


def test_a_review_status_tag_the_record_does_not_print_is_kept():
    stem = (
        "Co-production of guidance and resources to implement principled participant "
        "information leaflets (PrinciPILs)"
    )
    title = f"{stem} [version 1; peer review: 2 approved]"

    assert drop_title_note_marker(title, _record(stem, "Ana Silva1"), _page()) == (title, None)


def test_a_review_status_tag_inside_the_title_is_kept():
    title = "Trial leaflets [version 2; peer review: 2 approved] revisited in practice"
    record = _record(title, "Ana Silva1")

    assert drop_title_note_marker(title, record, _page()) == (title, None)


def test_a_note_symbol_that_opens_a_note_on_the_title_page_is_dropped():
    stem = "Some Considerations on the Empirical Research of Goal Systems of Insurance Companies"
    title = f"{stem} *"
    notes = ("* This is a revised and condensed version of a paper.", "** University.")

    kept, issue = drop_title_note_marker(
        title, _record(title, "By a Named Author**"), _page(*notes)
    )

    assert kept == stem
    assert issue is not None


def test_a_note_symbol_without_its_note_is_kept():
    title = "Geschlechtschromatin in Ovarialtumoren*"

    assert drop_title_note_marker(title, _record(title, "Von A. Autor"), _page()) == (title, None)


def test_a_note_symbol_ending_a_title_in_capitals_is_dropped():
    stem = "FROM CHRONOGENETICS TO PHENOGENESIS"
    notes = ("* Text of a lecture for an international congress.",)

    kept, issue = drop_title_note_marker(
        f"{stem}*", _record(f"{stem}*", "ANA SILVA"), _page(*notes)
    )

    assert kept == stem
    assert issue is not None


@pytest.mark.parametrize(
    "byline_row",
    [
        # The record took the note in as a byline row: its star opens the row.
        "* Text of a lecture for an international congress.",
        # A byline row that repeats the title, star and all.
        "FROM CHRONOGENETICS TO PHENOGENESIS* ANA SILVA",
    ],
)
def test_a_star_the_byline_rows_print_only_as_the_note_or_the_title_is_dropped(byline_row):
    stem = "FROM CHRONOGENETICS TO PHENOGENESIS"
    notes = ("* Text of a lecture for an international congress.",)

    kept, _ = drop_title_note_marker(f"{stem}*", _record(f"{stem}*", byline_row), _page(*notes))

    assert kept == stem


@pytest.mark.parametrize(
    ("title", "byline", "note"),
    [
        # The star of a name: a capital inside the last word.
        ("Memory-bounded heuristic search with IDA*", "Ana Silva", "* Supported by a grant."),
        (
            "Model checking the branching-time logic CTL\u2217",
            "Ana Silva",
            "\u2217This work was supported by a grant.",
        ),
        ("Foundations of RDF* and SPARQL*", "Ana Silva", "* Supported by a grant."),
        # The byline uses the star, so the note is the author's.
        (
            "Any-angle path planning on grids with Theta*",
            "Ana Silva*",
            "* This paper extends a conference version.",
        ),
        # The note that opens with the star is about an author.
        ("Any-angle path planning on grids with Theta*", "Ana Silva", "* Corresponding author."),
        (
            "Any-angle path planning on grids with Theta*",
            "Ana Silva",
            "* Ana Silva\r\nana.silva@example.org",
        ),
    ],
)
def test_a_star_that_is_part_of_a_name_or_an_author_note_is_kept(title, byline, note):
    assert drop_title_note_marker(title, _record(title, byline), _page(note)) == (title, None)


@pytest.mark.parametrize(
    ("title", "byline", "notes"),
    [
        # The star of an algorithm's name, with a corresponding-author note.
        ("Path planning for mobile robots with A*", "Ana Silva*", ("* Corresponding author.",)),
        # A squared statistic, with the byline marked from 3.
        ("Explained variance beyond the adjusted R\u00b2", "Ana Silva3", ("3 A note.",)),
    ],
)
def test_a_mark_after_a_short_token_is_kept(title, byline, notes):
    assert drop_title_note_marker(title, _record(title, byline), _page(*notes)) == (title, None)


def test_a_digit_the_affiliations_also_use_is_kept():
    # Note 1 on the page and affiliation 1 in the byline: the numbers are
    # shared, so the digit is not told apart from title text.
    title = "Child in court: intersection of justice spaces and participation rights exercise1"
    byline = "LIMA, Ana Cláudia (Portugal, Braga)\r\n1*\r\n1Universidade do Minho"
    notes = ("1 This article was presented at a conference on child studies.",)

    assert drop_title_note_marker(title, _record(title, byline), _page(*notes)) == (title, None)


@pytest.mark.parametrize(
    ("title", "byline", "notes"),
    [
        # A gene name the byline numbers as affiliation 1.
        ("Germline mutations in BRCA1", "Ana Silva1, Rui Costa2", ("1 Department of Genetics.",)),
        # No numbered note and no byline markers.
        ("Oxidative stress signalling through Nrf2", "Ana Silva, Rui Costa", ()),
        # Numbered notes exist, but the first one is not this number.
        ("Oxidative stress signalling through Nrf2", "Ana Silva", ("1 A note.", "2 A note.")),
        # Digits that are title text.
        ("Coagulation disorders in COVID-19", "Ana Silva2", ("1 A note.",)),
        ("Ocean uptake of anthropogenic CO2", "Ana Silva3", ("2 A note.",)),
        ("Working memory load in Study 1", "Ana Silva2", ("1 A note.",)),
    ],
)
def test_digits_without_note_evidence_are_kept(title, byline, notes):
    assert drop_title_note_marker(title, _record(title, byline), _page(*notes)) == (title, None)


# An invented title whose last word is long enough for a plain digit.
_SURVEY_TITLE = "Seasonal patterns of household water consumption in rural districts"
_ARTICLE_NOTE = "1 Paper presented at a regional meeting on water supply."


@pytest.mark.parametrize(
    "byline",
    [
        # Markers after a two-letter name, after a space, after a letter and a
        # comma, after a star, and before an ORCID mark: the byline uses 1.
        "Wei Li1, Hua Zhang2",
        "Ana Silva 1 | Rui Costa2",
        "Jane Doe a,1, John Roe b,*",
        "Ana Silva* 1 and Rui Costa2",
        "Ana Silva 1 \u25cf\nRui Costa2",
    ],
)
def test_a_digit_a_title_page_byline_marker_uses_is_kept(byline):
    title = f"{_SURVEY_TITLE}1"

    assert drop_title_note_marker(title, _record(title, byline), _page(_ARTICLE_NOTE)) == (
        title,
        None,
    )


@pytest.mark.parametrize(
    "note",
    [
        "1 These authors contributed equally to this work.",
        "1 Corresponding author: Ana Silva, Lisbon.",
        "1 ana.silva@example.org",
        "1 Department of Geography, University of Lisbon, Portugal.",
    ],
)
def test_a_digit_whose_first_note_is_about_an_author_is_kept(note):
    title = f"{_SURVEY_TITLE}1"

    assert drop_title_note_marker(title, _record(title, "Ana Silva, Rui Costa"), _page(note)) == (
        title,
        None,
    )


def test_a_plain_digit_before_the_byline_markers_without_its_note_is_kept():
    # A plain digit glued to a word is read only against a numbered note, not
    # against the byline's markers.
    title = f"{_SURVEY_TITLE}1"

    assert drop_title_note_marker(title, _record(title, "Ana Silva2, Rui Costa3"), _page()) == (
        title,
        None,
    )


def test_a_raised_digit_is_not_read_against_markers_of_other_pages():
    # The title page's byline is unmarked; a numbered heading later in the
    # record carries a byline role.
    title = f"{_SURVEY_TITLE}\u00b9"
    record = _record(title, "Ana Silva, Rui Costa")
    heading = dataclasses.replace(
        _candidate("c4", "2 Materials and methods", roles=frozenset({"byline", "heading"})),
        page=2,
    )
    record = dataclasses.replace(
        record,
        candidates=(*record.candidates, heading),
        blocks=(
            dataclasses.replace(
                record.blocks[0], candidate_ids=(*record.blocks[0].candidate_ids, "c4")
            ),
        ),
    )

    assert drop_title_note_marker(title, record, _page()) == (title, None)


@pytest.mark.parametrize(
    ("title", "note"),
    [
        # Gene and formula names: a capital inside the last word.
        ("Germline mutations in BRCA1", _ARTICLE_NOTE),
        ("Photocatalytic degradation of dyes over TiO2", "2 Paper presented at a meeting."),
        ("Androgen regulation of TMPRSS2", "2 Paper presented at a meeting."),
        # A last word shorter than six letters.
        ("Oxidative stress signalling through Keap1", _ARTICLE_NOTE),
    ],
)
def test_a_digit_that_ends_a_name_is_kept(title, note):
    assert drop_title_note_marker(title, _record(title, "Ana Silva"), _page(note)) == (
        title,
        None,
    )


def test_a_marker_the_title_row_does_not_print_is_kept():
    record = _record(_WASTE_TITLE, _WASTE_BYLINE)

    assert drop_title_note_marker(f"{_WASTE_TITLE}1", record, _page(*_WASTE_NOTES)) == (
        f"{_WASTE_TITLE}1",
        None,
    )


async def test_extracted_title_drops_its_note_marker():
    title = f"{_WASTE_TITLE}1"
    ext = _extractor(
        _record(title, _WASTE_BYLINE),
        CoreMetadataLLM(
            title=title,
            authors=[AuthorLLM(given="Beatriz Lima", family="Rocha")],
            keywords=[],
        ),
        contents=_page(*_WASTE_NOTES, title_row=title),
    )

    metadata = await ext.extract()

    assert metadata.title == _WASTE_TITLE
    assert "reason:title_note_marker" in [
        evidence for issue in ext.validation_issues for evidence in issue.evidence_ids
    ]
