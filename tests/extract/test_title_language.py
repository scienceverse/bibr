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


@pytest.mark.parametrize(
    ("title", "language"),
    [
        # Polish by its letters, and without them by two words only Polish has.
        ("Rola szkoły w procesie wychowania młodzieży", "pl"),
        ("Ocena jakosci uslug dla pacjentow oraz personelu", "pl"),
        ("Het onderwijs van de toekomst voor het kind", "nl"),
        # One Indonesian function word is enough: no other language shares them.
        ("Pengaruh pendidikan terhadap siswa kelas lima", "id"),
        # "do" is a Portuguese word before it is a Polish one; "do" and "we" are
        # English words as well.
        ("O significado do trabalho na vida das pessoas", "pt"),
        ("What do we know about sleep and memory", "en"),
        ("Artificial intelligence in physiology and where do we stand", "en"),
    ],
)
def test_the_guess_names_a_language_the_words_support(title, language):
    assert language_from_title(title) == language


@pytest.mark.parametrize(
    "title",
    [
        # A single-letter word is no evidence: "I" and "D" are Polish and French words.
        "Type I Error Rate Inflation",
        "Vitamin D Supplementation Trial Results",
        # One function word that English shares ("over", "die") does not make a
        # Dutch or German title.
        "Over Time: Attitudes Change Fast",
        "Die Young or Live Forever",
        "Aufsichtsgesetz der Volksrepublik China",
        # Scandinavian titles share Polish "i" and "og" with nothing in the table.
        "Skolen i Norge og verden i dag",
        # A capital dotted İ casefolds to "i" and a combining dot, which cut the
        # word at "sosi" + "al" and gave Spanish "al".
        "ELEKTRON DÖVLƏT MÜHİTİNDƏ SOSİAL KREDİTİN QİYMƏTLƏNDİRİLMƏSİ",
    ],
)
def test_the_guess_gives_none_where_the_words_are_too_thin(title):
    assert language_from_title(title) is None


@pytest.mark.parametrize(
    "title",
    [
        # ъ is Bulgarian as much as Russian.
        "Съвременно образование в училище и обществото",
        # Kazakh, Uzbek, Serbian and Belarusian letters name none of the two.
        "Оқушылардың химия пәніне қызығушылығын арттыру",
        "Ўзбекистонда таълим тизимини ривожлантириш",
        "Настава у школама у Србији данас",
        "Беларускія школы і выхаванне моладзі",
    ],
)
def test_a_cyrillic_title_of_another_language_is_not_called_russian_or_ukrainian(title):
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
