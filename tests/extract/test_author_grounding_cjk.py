"""Grounding CJK author names against bylines printed unspaced or letter-spaced.

A Japanese family-sociology paper prints its byline as 奥　山　正　司 (one
kanji per token). The extracted author 奥山 正司 never matched it, so the answer
was discarded as fabricated and the paper shipped without authors.
"""

from __future__ import annotations

import pytest

from bibr.extract.core_metadata import _author_token_variants, grounded_authors_in_context
from bibr.paper import PaperAuthor


def _author(given: str, family: str) -> PaperAuthor:
    return PaperAuthor(author_id=1, affiliation="", given=given, family=family)


@pytest.mark.parametrize(
    "printed",
    [
        "奥　山　正　司",  # letter-spaced with ideographic spaces, as printed
        "奥 山 正 司",  # the same after space folding
        "奥山正司",  # unspaced
        "奥山 正司",  # family and given name apart
    ],
)
def test_cjk_author_grounds_however_the_byline_is_spaced(printed):
    context = f"一人暮らし高齢者世帯の急増と社会福祉・社会保障\n{printed}\n私が高齢化や高齢者の研究を始めたのは"

    grounded, rejected = grounded_authors_in_context([_author("正司", "奥山")], context)

    assert len(grounded) == 1
    assert rejected == []


def test_cjk_author_does_not_ground_inside_a_longer_word():
    grounded, rejected = grounded_authors_in_context(
        [_author("正司", "奥山")], "奥山町の正司さんが来た"
    )

    assert grounded == []
    assert len(rejected) == 1


def test_unspaced_chinese_byline_grounds_each_author():
    authors = [
        _author("伟", "王"),
        PaperAuthor(author_id=2, affiliation="", given="明", family="李"),
    ]

    grounded, rejected = grounded_authors_in_context(authors, "王伟, 李明")

    assert [author.family for author in grounded] == ["王", "李"]
    assert rejected == []


def test_letter_spaced_names_of_two_authors_ground_apart():
    authors = [
        _author("伟", "王"),
        PaperAuthor(author_id=2, affiliation="", given="明", family="李"),
    ]

    grounded, rejected = grounded_authors_in_context(authors, "王 伟 李 明")

    assert [author.family for author in grounded] == ["王", "李"]
    assert rejected == []


def test_only_cjk_names_gain_joined_and_per_character_variants():
    assert ("奥山正司",) in _author_token_variants(_author("正司", "奥山"))
    assert ("奥", "山", "正", "司") in _author_token_variants(_author("正司", "奥山"))
    assert _author_token_variants(_author("Anna", "Smith")) == (
        ("anna", "smith"),
        ("smith", "anna"),
    )


@pytest.mark.parametrize(
    "printed",
    [
        "佐々木太郎",  # unspaced
        "佐　々　木　太　郎",  # letter-spaced, as the byline above is printed
    ],
)
def test_a_name_with_the_kanji_repetition_mark_grounds_like_any_cjk_name(printed):
    """々 (U+3005) repeats the kanji before it and is common in Japanese family
    names (佐々木); it sits outside the Han block, so without it the name got no
    joined or per-character variant and read as fabricated."""

    context = f"一人暮らし高齢者世帯の急増と社会福祉・社会保障\n{printed}\n私が高齢化や高齢者の研究を始めたのは"

    grounded, rejected = grounded_authors_in_context([_author("太郎", "佐々木")], context)

    assert ("佐々木太郎",) in _author_token_variants(_author("太郎", "佐々木"))
    assert len(grounded) == 1
    assert rejected == []
