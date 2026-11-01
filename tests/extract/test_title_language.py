"""The language a title gives the record when the input declares none.

Every title below is invented.
"""

from __future__ import annotations

import pytest

from bibr.extract.front_matter import fill_language, language_from_title
from bibr.models import PaperMetadata


@pytest.mark.parametrize(
    ("title", "language"),
    [
        ("The effects of sleep loss on memory in older adults", "en"),
        ("La inteligencia emocional en el contexto educativo de los adolescentes", "es"),
        ("Les mondes de la lecture dans les villes de province", "fr"),
        ("Estudo sobre os efeitos da leitura em crianças do ensino básico", "pt"),
        ("Eine Untersuchung der Lesegewohnheiten von Kindern und Jugendlichen", "de"),
        ("Влияние сна на память у людей старшего возраста: обзор данных", "ru"),
        ("Перші кроки реабілітації та українська практика", "uk"),
        # Four words are enough where the function words decide.
        ("Les œuvres littéraires modernes", "fr"),
    ],
)
def test_a_title_gives_the_language_it_is_printed_in(title, language):
    assert language_from_title(title) == language


@pytest.mark.parametrize(
    "title",
    [
        None,
        "",
        # Too short to trust, even where one function word is in the vocabulary.
        "Theory of mind",
        "Les mondes",
        # No language's function words win.
        "Deep Learning Pipelines Image Retrieval Benchmarks",
        "HIGH QUALITY HUMAN RESOURCES DEVELOPMENT",
        # A script names no language: Russian printed without its telltale letters,
        # Chinese, Greek.
        "Сравнительная оценка метода измерения",
        "城市老年人孤独感与社会支持研究",
        "Μελέτη της εργαζόμενης μνήμης σε ηλικιωμένους",
    ],
)
def test_a_title_that_cannot_be_told_gives_none(title):
    assert language_from_title(title) is None


def test_fill_language_sets_an_empty_language_from_the_title():
    metadata = PaperMetadata(doi="10.1/x", title="The effects of sleep loss on memory in adults")
    fill_language(metadata)
    assert metadata.language == "en"


def test_fill_language_keeps_what_the_input_declared():
    metadata = PaperMetadata(
        doi="10.1/x", title="The effects of sleep loss on memory in adults", language="de"
    )
    fill_language(metadata)
    assert metadata.language == "de"


def test_fill_language_leaves_the_field_empty_when_the_title_says_nothing():
    metadata = PaperMetadata(
        doi="10.1/x", title="Deep Learning Pipelines Image Retrieval Benchmarks"
    )
    fill_language(metadata)
    assert metadata.language is None
