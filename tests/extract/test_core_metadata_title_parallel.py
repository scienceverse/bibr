"""A title the model joined with its translation keeps only the version printed first.

The two real cases come from dev-set papers: a Spanish legal article whose
English title is printed on the rows under the Spanish one (the model joined
them with " / "), and a Dutch report whose title row prints the English
translation in brackets after it. Other titles and names are invented.
"""

import pytest

from bibr.extract.title_subtitle import drop_parallel_title
from bibr.schemas import AuthorLLM, CoreMetadataLLM

from .test_core_metadata_author_guards import _candidate, _extractor, _resolution

_SPANISH = (
    "Un estira y afloja: La definición de las reglas para la libre expresión en las "
    "plataformas de redes sociales"
)
_ENGLISH = "Tug of War: The definition of the rules for freedom of expression on social networks"
# The selected record as the metadata prompt sees it: the Spanish title
# heading, then the English title opening the next row, broken as printed.
_PRINTED = (
    f"{_SPANISH}\n"
    "Tug of War: The definition of the rules for\r\n"
    "freedom of expression on social networks DOI: https://doi.org/10.17981/example"
)
_DUTCH = "PERCEPTUELE STRUKTUREN VAN SYNTHETISCHE EN NATUURLIJKE KLINKERS"
_DUTCH_ENGLISH = "PERCEPTUAL STRUCTURES OF SYNTHETIC AND NATURAL VOWELS"


def test_joined_versions_keep_the_one_printed_first():
    title, issue = drop_parallel_title(f"{_SPANISH} / {_ENGLISH}", _PRINTED)

    assert title == _SPANISH
    assert issue is not None
    assert issue.code == "VAL_TITLE_REGROUNDED"
    assert issue.evidence_ids == ("reason:title_parallel_versions_joined",)
    assert not issue.blocking


def test_printed_order_wins_over_the_order_the_model_joined_them_in():
    assert drop_parallel_title(f"{_ENGLISH} / {_SPANISH}", _PRINTED)[0] == _SPANISH


@pytest.mark.parametrize(
    "printed",
    [
        # Only the English title is printed: the Spanish half is the model's own.
        f"{_ENGLISH}\nDOI: https://doi.org/10.17981/example",
        "Resumen\nEste trabajo estudia la regulación.",
    ],
)
def test_a_join_with_a_version_the_page_does_not_print_is_left_alone(printed):
    joined = f"{_SPANISH} / {_ENGLISH}"

    assert drop_parallel_title(joined, printed) == (joined, None)


def test_a_half_carrying_words_the_page_does_not_print_is_left_alone():
    # "Corrigendum: " is not printed before the Spanish title; keeping the
    # English half would drop the notice prefix the correction guard reads.
    joined = f"Corrigendum: {_SPANISH} / {_ENGLISH}"

    assert drop_parallel_title(joined, _PRINTED) == (joined, None)


def test_translation_bracketed_after_the_title_row_is_dropped():
    title, issue = drop_parallel_title(f"{_DUTCH} [{_DUTCH_ENGLISH}]", "")

    assert title == _DUTCH
    assert issue is not None


def test_versions_in_two_scripts_keep_the_one_printed_first():
    russian = "Сезонная смена окраски листьев в школьных садах"
    english = "Seasonal colour change of leaves in school gardens"

    title, _ = drop_parallel_title(f"{russian} / {english}", f"{russian}\n{english}")

    assert title == russian


@pytest.mark.parametrize(
    "title",
    [
        # One language on both sides: "A Scoping Review" uses no English
        # function word, so the language test needs two on each side.
        "The Role of the Rural Context in the Transition to Adulthood / A Scoping Review",
        "Input / output of the school garden network",
        "Seasonal colour change in small allotment gardens / Lessons for the classroom",
        "Reading the garden [Review of a book for young gardeners]",
        # Too short to be a title version.
        "Colour / Farbe",
        "Seasonal colour change in allotment gardens [Farbwechsel]",
        # Three versions: not a two-version join.
        f"{_SPANISH} / {_ENGLISH} / {_DUTCH_ENGLISH}",
    ],
)
def test_titles_that_are_not_two_language_versions_are_left_alone(title):
    assert drop_parallel_title(title, title) == (title, None)


async def test_extracted_title_keeps_the_version_printed_first():
    resolution = _resolution(
        _candidate("c1", _SPANISH, roles=frozenset({"title", "heading"}), source_kind="heading"),
        _candidate("c2", _ENGLISH, roles=frozenset()),
        _candidate("c3", "Rodrigo Cetina Presuel", roles=frozenset({"byline"}), text_ids=(3,)),
    )
    ext = _extractor(
        resolution,
        CoreMetadataLLM(
            title=f"{_SPANISH} / {_ENGLISH}",
            authors=[AuthorLLM(given="Rodrigo", family="Cetina Presuel")],
            keywords=[],
        ),
    )

    metadata = await ext.extract()

    # The subtitle fold leaves the English row out again: it is a parallel title.
    assert metadata.title == _SPANISH
    assert "reason:title_parallel_versions_joined" in [
        evidence for issue in ext.validation_issues for evidence in issue.evidence_ids
    ]
