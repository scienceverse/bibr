"""Splitting printed author and editor lists (bibr.export.names)."""

from __future__ import annotations

import pytest

from bibr.export.names import PrintedName, is_repeated_author_rule, split_printed_names


def _split(text: str, **kwargs) -> tuple[list[dict[str, str]], bool]:
    names, et_al = split_printed_names(text, **kwargs)
    return [n.as_dict() for n in names], et_al


def P(family: str, given: str | None = None, suffix: str | None = None) -> dict[str, str]:
    return PrintedName(family=family, given=given, suffix=suffix).as_dict()


def L(literal: str) -> dict[str, str]:
    return {"literal": literal}


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        # APA / Harvard: Family, Initials
        (
            "Carney, D. R., Cuddy, A. J. C., & Yap, A. J.",
            [P("Carney", "D. R."), P("Cuddy", "A. J. C."), P("Yap", "A. J.")],
        ),
        ("Smith, J. A. and Doe, B.", [P("Smith", "J. A."), P("Doe", "B.")]),
        (
            "Dehaene, S., Changeux, J.-P., & Sergent, C.",
            [P("Dehaene", "S."), P("Changeux", "J.-P."), P("Sergent", "C.")],
        ),
        ("García-López, J., & O'Brien, M.", [P("García-López", "J."), P("O'Brien", "M.")]),
        # particles stay with the family name
        (
            "Smith, J. A., Doe, B., & van der Berg, C.",
            [P("Smith", "J. A."), P("Doe", "B."), P("van der Berg", "C.")],
        ),
        ("de Groot, A. D.", [P("de Groot", "A. D.")]),
        ("Berg, C. van der", [P("van der Berg", "C.")]),
        ("Ortega y Gasset, J.", [P("Ortega y Gasset", "J.")]),
        # suffixes
        ("Smith, J. A., Jr., & Doe, B.", [P("Smith", "J. A.", "Jr."), P("Doe", "B.")]),
        ("King, M. L., Jr.; Doe, A.", [P("King", "M. L.", "Jr."), P("Doe", "A.")]),
        # written-out given names, Chicago's given-first co-authors
        ("Smith, John, and Jane Doe", [P("Smith", "John"), P("Doe", "Jane")]),
        ("Smith, John A., Doe, Jane", [P("Smith", "John A."), P("Doe", "Jane")]),
        # Vancouver
        (
            "Smith JA, Doe B, van der Berg C",
            [P("Smith", "JA"), P("Doe", "B"), P("van der Berg", "C")],
        ),
        ("Kahneman D., Tversky A.", [P("Kahneman", "D."), P("Tversky", "A.")]),
        # given name first
        ("J. A. Smith and B. Doe", [P("Smith", "J. A."), P("Doe", "B.")]),
        (
            "J. A. Smith, B. Doe, and C. van der Berg",
            [P("Smith", "J. A."), P("Doe", "B."), P("van der Berg", "C.")],
        ),
        # one name per ;-separated part
        ("Smith, J.; Doe, A.", [P("Smith", "J."), P("Doe", "A.")]),
        ("Smith J; Doe A", [P("Smith", "J"), P("Doe", "A")]),
        # APA 7's ellipsis before the last author
        ("Smith, J., Doe, A., . . . Zed, Z.", [P("Smith", "J."), P("Doe", "A."), P("Zed", "Z.")]),
        # a family name alone
        ("Plato", [P("Plato")]),
    ],
)
def test_person_names(printed, expected):
    assert _split(printed) == (expected, False)


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("World Health Organization", [L("World Health Organization")]),
        ("Open Science Collaboration", [L("Open Science Collaboration")]),
        ("American Psychiatric Association", [L("American Psychiatric Association")]),
        (
            "Department of Health and Human Services",
            [L("Department of Health and Human Services")],
        ),
        (
            "Smith, J., & World Health Organization",
            [P("Smith", "J."), L("World Health Organization")],
        ),
    ],
)
def test_organizations_are_one_literal_name(printed, expected):
    assert _split(printed) == (expected, False)


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("Smith, J., Doe, A., et al.", [P("Smith", "J."), P("Doe", "A.")]),
        ("Smith et al.", [P("Smith")]),
        ("Smith JA, Doe B, et al", [P("Smith", "JA"), P("Doe", "B")]),
        ("Smith, J., and others", [P("Smith", "J.")]),
    ],
)
def test_et_al_is_cut_off_and_reported(printed, expected):
    assert _split(printed) == (expected, True)


@pytest.mark.parametrize(
    "printed",
    [
        "张三, 李四",  # no letter case to tell names apart
        "SMITH, J., & DOE, A.",  # "DOE" reads as initials
        "Smith, J., Doe, A., Roe",  # a name without initials among inverted ones
    ],
)
def test_lists_that_fit_no_style_stay_whole(printed):
    names, et_al = _split(printed)
    assert names == [L(printed)]
    assert et_al is False


def test_editor_lists_lose_their_markers():
    assert _split("In J. Smith & B. Doe (Eds.),", editors=True) == (
        [P("Smith", "J."), P("Doe", "B.")],
        False,
    )
    assert _split("Smith, J. (Ed.)", editors=True) == ([P("Smith", "J.")], False)


def test_rule_gives_no_names():
    assert is_repeated_author_rule("——. 1987.")
    assert is_repeated_author_rule("---, & Doe, A.")
    assert not is_repeated_author_rule("Smith, J.")
    assert split_printed_names("——") == ([], False)


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_nothing_printed(empty):
    assert split_printed_names(empty) == ([], False)
