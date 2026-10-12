"""A group byline tail the model returned as a person (#118)."""

from __future__ import annotations

from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.paper import PaperAuthor


def _author(author_id: int, given: str, family: str, affiliation: str = "") -> PaperAuthor:
    return PaperAuthor(author_id=author_id, given=given, family=family, affiliation=affiliation)


_GROUP_BYLINE = (
    "Ann Lee, Bo Chen, Cy Pham, for the ABC/1234 Study Collaborators‡\n"
    "Summary Background Example text."
)


def _group_extractor() -> CoreMetadataExtractor:
    extractor = object.__new__(CoreMetadataExtractor)
    extractor.validation_issues = []
    return extractor


def test_a_group_tail_returned_as_a_person_becomes_a_group_author():
    authors = [
        _author(1, "Ann", "Lee", "Example University"),
        _author(2, "Bo", "Chen"),
        _author(3, "Cy", "Pham"),
        _author(4, "", "ABC/1234", "Example University"),
    ]

    kept, transforms = _group_extractor()._sanitize_authors(authors, _GROUP_BYLINE)

    assert [(a.author_id, a.given, a.family) for a in kept] == [
        (1, "Ann", "Lee"),
        (2, "Bo", "Chen"),
        (3, "Cy", "Pham"),
        (4, "", "ABC/1234 Study Collaborators"),
    ]
    assert "organization" in kept[3].role
    assert kept[3].affiliation == ""
    assert transforms == ["group_authors_rewritten"]


def test_a_group_the_model_also_returned_is_not_added_twice():
    from bibr.schemas import AuthorLLM

    converted = CoreMetadataExtractor._convert_llm_authors(
        [
            AuthorLLM(given="Ann", family="Lee"),
            AuthorLLM(given="", family="ABC/1234 Study Collaborators", role=["organization"]),
            AuthorLLM(given="Abc", family="1234"),
        ]
    )
    # "collaborators" now counts as a group word, so the organisation entry
    # survives the fragment filter instead of being dropped.
    assert [(a.given, a.family) for a in converted] == [
        ("Ann", "Lee"),
        ("", "ABC/1234 Study Collaborators"),
        ("Abc", "1234"),
    ]

    kept, _ = _group_extractor()._sanitize_authors(converted, _GROUP_BYLINE)

    assert [(a.author_id, a.given, a.family) for a in kept] == [
        (1, "Ann", "Lee"),
        (2, "", "ABC/1234 Study Collaborators"),
    ]


def test_people_and_unprinted_groups_are_left_alone():
    authors = [
        _author(1, "Ann", "Lee"),
        _author(2, "J0hn", "Smith"),
        _author(3, "", "EXAMPLE/22"),
        _author(4, "", "on behalf of the Example Trial Group"),
    ]

    kept, transforms = _group_extractor()._sanitize_authors(authors, "Ann Lee, J0hn Smith")

    assert [(a.given, a.family) for a in kept] == [
        ("Ann", "Lee"),
        ("J0hn", "Smith"),
        ("", "EXAMPLE/22"),
        ("", "Example Trial Group"),
    ]
    assert "organization" not in (kept[2].role or [])
    assert "organization" in kept[3].role
    assert transforms == ["group_authors_rewritten"]
