"""Recall guards for the reference evidence sent to citation resolution."""

import pytest

from bibr.schemas import CitationMatch
from bibr.structure.citation_shortlist import (
    partition_shortlist_matches,
    shortlist_references,
)


def ref(bib_id, author="Smith J", year=2020):
    return {"bib_id": bib_id, "author": author, "year": year, "title": f"Study {bib_id}"}


def bibliography():
    # Deliberately unrelated IDs/order: these are IDs, never citation numbers.
    return [
        ref(71),
        ref(42, "Smith, A.", "2020b"),
        ref(900, "Smith B", 2018),
        ref(15, "Jones B", 2020),
        ref(3, None, 1997),
        ref(2, "Other A", None),
        *[ref(i, "Unrelated A", 1980) for i in range(100, 140)],
    ]


@pytest.mark.parametrize(
    "text", ["(Smith, 2020)", "Smith (2020)", "Smith, 2020", "[Smith\n et al.,\n2020a]"]
)
def test_union_keeps_all_name_or_year_matches_and_incomplete_records(text):
    refs = bibliography()
    result = shortlist_references([(18, text)], refs)
    assert result.narrowed
    assert result.references == refs[:6]
    assert set(next(iter(result.candidate_ids.values()))) == {71, 42, 900, 15, 3, 2}


@pytest.mark.parametrize("author", ["van der Waals J, García A", "van der Waals, J., & García, A."])
def test_compound_names_accents_and_year_suffixes_do_not_prune_variants(author):
    refs = [ref(7, author, "2020b"), ref(9, "Garcia B", 1999), *bibliography()[6:]]
    result = shortlist_references([(1, "(van der Waals & García, 2020a)")], refs)
    assert result.narrowed
    assert [r["bib_id"] for r in result.references] == [7, 9]


def test_multicitation_union_includes_every_year_and_separate_citation():
    refs = [*bibliography(), ref(800, "Black Z", 2022), ref(888, "White Y", 2023)]
    result = shortlist_references(
        [(1, "(Smith, 2020, 2018; Black, 2022)"), (2, "White (2023)")], refs
    )
    assert result.narrowed
    assert {r["bib_id"] for r in result.references} == {71, 42, 900, 15, 3, 2, 800, 888}
    assert 888 not in result.candidate_ids["(Smith, 2020, 2018; Black, 2022)"]


@pytest.mark.parametrize(
    "text",
    [
        "[1]",
        "[71, 42]",
        "[1–3]",
        "(Unknown, 2020)",
        "(Sm1th, 2020)",
        "(Smith, 2033)",
        "(Smith, in press)",
        "(Smith, 2020; Doe, in press)",
        "(Smith, 2020; Unknown, 2021)",
        "(Smith, 2020; [???])",
        "Smith (2020) missing work",
        "(Smith, 2020; doi:unknown)",
        "(Smith, 2020, 1899)",
        "(2020)",
    ],
)
def test_uncertain_citation_keeps_full_bibliography(text):
    refs = bibliography()
    result = shortlist_references([(4, text)], refs)
    assert not result.narrowed
    assert result.references is refs


def test_one_uncertain_citation_keeps_entire_request_full():
    refs = bibliography()
    result = shortlist_references([(1, "(Smith, 2020)"), (2, "[1]")], refs)
    assert result.references is refs


def test_small_absolute_or_relative_savings_keep_full_request():
    small = [ref(1), ref(2, "Other B", 2001)]
    assert shortlist_references([(1, "Smith (2020)")], small).reason == "small_saving"
    big = [
        {**ref(1), "title": "Large retained record " * 300},
        {**ref(2, "Other B", 2001), "title": "Excluded record " * 50},
    ]
    assert shortlist_references([(1, "Smith (2020)")], big).reason == "small_saving"


@pytest.mark.parametrize("refs", [[], [ref(1)], [ref(1), ref(1)], [ref("b1")]])
def test_small_empty_or_noninteger_bibliographies_keep_legacy_request(refs):
    assert not shortlist_references([(1, "(Smith, 2020)")], refs).narrowed


def test_candidate_specific_validation_and_normalized_text_preserve_linker_semantics():
    cites = [(10, "[Smith\n et al., 2020]"), (20, "Jones (2021)"), (30, "Missing (2022)")]
    result = [
        CitationMatch(text_id=999, citation_text="[Smith et al., 2020]", bib_id=71),
        CitationMatch(text_id=20, citation_text="Jones (2021)", bib_id=71),
        CitationMatch(text_id=30, citation_text="Invented (2022)", bib_id=5),
    ]
    accepted, pending = partition_shortlist_matches(
        cites, result, {"[Smith et al., 2020]": frozenset({71}), "Jones (2021)": frozenset({5})}
    )
    assert accepted == result[:1]
    assert pending == cites[1:]


@pytest.mark.parametrize("targets", [[], [None], [71, None], [71, 42], [999]])
def test_missing_conflicting_or_invalid_answers_require_expansion(targets):
    cites = [(10, "Smith (2020)")]
    matches = [CitationMatch(text_id=10, citation_text=cites[0][1], bib_id=i) for i in targets]
    assert partition_shortlist_matches(cites, matches, {cites[0][1]: frozenset({71, 42})}) == (
        [],
        cites,
    )


def test_identical_duplicate_answers_and_repeated_citations_do_not_expand():
    text = "Smith (2020)"
    match = CitationMatch(text_id=9, citation_text=text, bib_id=71)
    assert partition_shortlist_matches(
        [(1, text), (2, text)], [match, match], {text: frozenset({71})}
    ) == ([match], [])
