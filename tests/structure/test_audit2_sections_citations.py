"""Author-year matching and equation cross-reference edge cases."""

import pytest

from bibr.models import PaperReference
from bibr.paper_contents import PaperSentence
from bibr.structure.citation_matcher import match_with_candidates
from bibr.structure.xref_utils import detect_xrefs


def _ref(bib_id: int, authors: str, year: int | None, suffix: str | None = None):
    return PaperReference(
        bib_id=bib_id,
        title=f"Title {bib_id}",
        first_page=None,
        volume=None,
        authors=authors,
        year=year,
        container=None,
        year_suffix=suffix,
    )


def _linked(text: str, refs: list[PaperReference], *, rules: bool) -> list[int]:
    _xrefs, _ambiguous, candidates = match_with_candidates(
        [PaperSentence(0, text, 0, 0)], refs, rules=rules
    )
    return sorted({bib_id for c in candidates if c.accepted for bib_id in c.bib_ids})


_SMITH = [
    _ref(1, "Smith, J.", 2020, "a"),
    _ref(2, "Smith, J.", 2020, "b"),
    _ref(3, "Smith, J.", 2020, "c"),
]


@pytest.mark.parametrize("rules", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Prior work (Smith, 2020a, b) agrees.", [1, 2]),
        ("Prior work (Smith, 2020a,b) agrees.", [1, 2]),
        ("Prior work (Smith 2020a, b, c) agrees.", [1, 2, 3]),
        ("Prior work (Smith, 2020b and c) agrees.", [2, 3]),
        ("Smith (2020a, b) argued it.", [1, 2]),
        ("Prior work (Jones, 2019; Smith, 2020a, b) agrees.", [1, 2]),
    ],
)
def test_bare_year_suffix_list_links_every_suffix(text, expected, rules):
    assert _linked(text, _SMITH, rules=rules) == expected


@pytest.mark.parametrize("rules", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        "Prior work (Smith, 2020a, p. 5) agrees.",
        "Prior work (Smith, 2020a, pp. 5-6) agrees.",
        "Prior work (Smith, 2020a, e.g.) agrees.",
    ],
)
def test_a_pinpoint_letter_is_no_year_suffix(text, rules):
    assert _linked(text, _SMITH, rules=rules) == [1]


@pytest.mark.parametrize("rules", [False, True])
@pytest.mark.parametrize(
    ("listed", "cited"),
    [
        ("O'Brien, D.", "(O’Brien, 2011)"),
        ("O’Brien, D.", "(O'Brien, 2011)"),
        ("O’Brien, D.", "O'Brien (2011) found"),
        ("D'Souza, M., & Lee, K.", "(D’Souza & Lee, 2011)"),
    ],
)
def test_apostrophe_variants_match(listed, cited, rules):
    assert _linked(cited, [_ref(1, listed, 2011)], rules=rules) == [1]


def _equations(text: str) -> list[tuple[int, str]]:
    return [
        (x.xref_id, x.contents)
        for x in detect_xrefs([PaperSentence(1, text, 1, 1)], [], [])
        if x.xref_type == "equation"
    ]


@pytest.mark.parametrize(
    ("text", "expected_ids", "contents"),
    [
        ("Combining Eqs. (1)-(3) gives the bound.", [1, 2, 3], "Eqs. (1)-(3)"),
        ("Combining Eqs. (1)–(3) gives the bound.", [1, 2, 3], "Eqs. (1)–(3)"),
        ("Equations (2) and (3) are exact.", [2, 3], "Equations (2) and (3)"),
        ("From Eqs. (4), (5) and (7) we get it.", [4, 5, 7], "Eqs. (4), (5) and (7)"),
        ("From eqs. (4) & (6) we get it.", [4, 6], "eqs. (4) & (6)"),
        ("See Eq. (2.3) and (2.4) here.", [2], "Eq. (2.3) and (2.4)"),
    ],
)
def test_parenthesised_equation_lists_keep_every_number(text, expected_ids, contents):
    found = _equations(text)
    assert [number for number, _ in found] == expected_ids
    assert {span for _, span in found} == {contents}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Substituting Eq. (5) into the bound.", [(5, "Eq. (5)")]),
        ("(see Equations 1 and 7)", [(1, "Equations 1 and 7"), (7, "Equations 1 and 7")]),
        ("Using Eq. (3) (Smith, 2020) we bound it.", [(3, "Eq. (3)")]),
        ("Eq. (2) and (Smith, 2020) agree.", [(2, "Eq. (2)")]),
    ],
)
def test_single_parenthesised_equation_unchanged(text, expected):
    assert _equations(text) == expected
