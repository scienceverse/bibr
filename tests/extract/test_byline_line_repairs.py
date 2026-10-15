"""Author names the model read across a byline's line breaks.

Two jats_bs_v1 author misses: a one-name-per-line byline whose first two
names came back as one person, and a JMIR group author whose first word
wrapped onto the line above.
"""

from __future__ import annotations

from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.paper import PaperAuthor


def _author(author_id: int, given: str, family: str, **extra) -> PaperAuthor:
    return PaperAuthor(author_id=author_id, given=given, family=family, affiliation="", **extra)


def _extractor() -> CoreMetadataExtractor:
    extractor = object.__new__(CoreMetadataExtractor)
    extractor.validation_issues = []
    return extractor


ONE_PER_LINE = (
    "Ana Maria Lopes\n"
    "Bea Kowalski1\n"
    "Carl Ode\n"
    "Example Institute, Paris, France\n"
    "Correspondence: Ana Maria Lopes (ana@example.org)"
)


def test_two_names_on_two_lines_read_as_one_person_are_split():
    authors = [
        _author(1, "Ana Maria Lopes Bea", "Kowalski", email="ana@example.org", corresponding=True),
        _author(2, "Carl", "Ode"),
    ]

    kept, transforms = _extractor()._sanitize_authors(authors, ONE_PER_LINE)

    assert [(a.author_id, a.given, a.family) for a in kept] == [
        (1, "Ana Maria", "Lopes"),
        (2, "Bea", "Kowalski"),
        (3, "Carl", "Ode"),
    ]
    assert all(a.email is None and not a.corresponding for a in kept[:2])
    assert "line_merged_authors_split" in transforms


def test_a_long_name_printed_whole_is_left_alone():
    context = "Ana Maria Lopes Bea Kowalski\nAna Maria Lopes\nBea Kowalski"
    authors = [_author(1, "Ana Maria Lopes Bea", "Kowalski")]

    kept, transforms = _extractor()._sanitize_authors(authors, context)

    assert [(a.given, a.family) for a in kept] == [("Ana Maria Lopes Bea", "Kowalski")]
    assert transforms == []


WRAPPED_GROUP = (
    "Ann Lee1 , MD, PhD; Bo Chen2 , PhD; Cy Pham1,2 , PhD; ZChart\n"
    "Study Panel3\n"
    "1 Example University"
)


def test_a_group_name_wrapped_after_a_semicolon_gets_its_head_back():
    authors = [
        _author(1, "Ann", "Lee"),
        _author(2, "Bo", "Chen"),
        _author(3, "Cy", "Pham"),
        _author(4, "", "Study Panel"),
    ]

    kept, _ = _extractor()._sanitize_authors(authors, WRAPPED_GROUP)

    assert kept[3].family == "ZChart Study Panel"


def test_a_degree_before_the_break_is_not_taken_as_the_head():
    context = "Ann Lee1 , MD, PhD; Bo Chen2 , PhD\nStudy Panel3"
    authors = [_author(1, "Ann", "Lee"), _author(2, "Bo", "Chen"), _author(3, "", "Study Panel")]

    kept, _ = _extractor()._sanitize_authors(authors, context)

    assert kept[2].family == "Study Panel"


def test_another_authors_name_before_the_break_is_not_taken_as_the_head():
    context = "Ann Lee; Chen\nStudy Panel"
    authors = [_author(1, "Ann", "Lee"), _author(2, "Bo", "Chen"), _author(3, "", "Study Panel")]

    kept, _ = _extractor()._sanitize_authors(authors, context)

    assert kept[2].family == "Study Panel"
