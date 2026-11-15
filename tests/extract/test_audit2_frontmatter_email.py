"""Corresponding-author e-mail harvesting: envelope misreads, shared surnames, braced lists."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bibr.extract.author_email_harvester import AuthorEmailHarvester
from bibr.models import PaperAuthor


def _make_contents_with_text(lines: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        sentences=[SimpleNamespace(text=t, section_id=0, text_id=i) for i, t in enumerate(lines)],
        sections=[SimpleNamespace(section_id=0, section_type="abstract", header="")],
    )


def _authors(*names: tuple[str, str]) -> list[PaperAuthor]:
    return [
        PaperAuthor(author_id=i, given=given, family=family, affiliation="X")
        for i, (given, family) in enumerate(names, 1)
    ]


@pytest.mark.parametrize("marker", ["✉", "pg"])
def test_an_envelope_misread_as_pg_pairs_the_address_like_the_glyph(marker):
    # GLM-OCR prints the envelope as "pg ": the line it opens names the corresponding
    # address exactly as the glyph does, so the other byline address is not promoted.
    contents = _make_contents_with_text(
        [
            "Jane Doe, John Smith",
            f"{marker} Jane Doe, jane.doe@uni.edu",
            "John Smith, john.smith@uni.edu",
        ]
    )
    authors = _authors(("Jane", "Doe"), ("John", "Smith"))
    AuthorEmailHarvester(contents).harvest(authors)

    assert [(a.email, a.corresponding) for a in authors] == [
        ("jane.doe@uni.edu", True),
        ("john.smith@uni.edu", False),
    ]


@pytest.mark.parametrize("given", ["Mary Ann", "Mary A.", "Mary-Ann"])
def test_a_shared_surname_goes_to_the_author_whose_given_name_the_address_spells(given):
    contents = _make_contents_with_text(
        [f"Mark Smith and {given} Smith", "Correspondence: mary.smith@uni.edu"]
    )
    authors = _authors(("Mark", "Smith"), (given, "Smith"))
    AuthorEmailHarvester(contents).harvest(authors)

    assert [a.email for a in authors] == [None, "mary.smith@uni.edu"]


def test_a_braced_address_list_gives_each_author_their_address():
    contents = _make_contents_with_text(
        ["Alice Lee, Bob Kim", "University of Example", "{alice.lee, bob.kim}@uni.edu"]
    )
    authors = _authors(("Alice", "Lee"), ("Bob", "Kim"))
    AuthorEmailHarvester(contents).harvest(authors)

    assert [a.email for a in authors] == ["alice.lee@uni.edu", "bob.kim@uni.edu"]
