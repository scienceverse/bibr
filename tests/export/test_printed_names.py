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
        # A given name never follows "and": these are two people, not one
        # inverted name with family "John Smith".
        ("John Smith and Jane Doe", [P("Smith", "John"), P("Doe", "Jane")]),
        (
            "Sepp Hochreiter and Jürgen Schmidhuber",
            [P("Hochreiter", "Sepp"), P("Schmidhuber", "Jürgen")],
        ),
        (
            "Chelsea Finn, Pieter Abbeel, and Sergey Levine",
            [P("Finn", "Chelsea"), P("Abbeel", "Pieter"), P("Levine", "Sergey")],
        ),
        # Two full names in a row are two people, with or without an "and".
        ("John Smith, Jane Doe", [P("Smith", "John"), P("Doe", "Jane")]),
        ("Jane Doe, John A. Smith", [P("Doe", "Jane"), P("Smith", "John A.")]),
        (
            "Anita Williams Woolley, Christopher F Chabris, Alex Pentland",
            [P("Woolley", "Anita Williams"), P("Chabris", "Christopher F"), P("Pentland", "Alex")],
        ),
        # A two-word family name with a one-word given name is still inverted.
        ("García Márquez, Gabriel", [P("García Márquez", "Gabriel")]),
        (
            "Smith, John, García Márquez, Gabriel",
            [P("Smith", "John"), P("García Márquez", "Gabriel")],
        ),
        # Chicago: after the first author, everyone is given name first.
        (
            "Hochreiter, Sepp, and Jürgen Schmidhuber",
            [P("Hochreiter", "Sepp"), P("Schmidhuber", "Jürgen")],
        ),
        (
            "Fieldhouse, Edward, Jane Green, Geoffrey Evans, and Chris Prosser",
            [
                P("Fieldhouse", "Edward"),
                P("Green", "Jane"),
                P("Evans", "Geoffrey"),
                P("Prosser", "Chris"),
            ],
        ),
        (
            "Abbott, William, Thomas E. Brownlee, Tom Clifford, Richard Page",
            [
                P("Abbott", "William"),
                P("Brownlee", "Thomas E."),
                P("Clifford", "Tom"),
                P("Page", "Richard"),
            ],
        ),
        # a particle inside a given-name-first family name
        (
            "Lucas Souza, Gabriel de Oliveira Ramos, and Celia Ralha",
            [P("Souza", "Lucas"), P("de Oliveira Ramos", "Gabriel"), P("Ralha", "Celia")],
        ),
    ],
)
def test_full_names_are_never_merged_into_one_person(printed, expected):
    assert _split(printed) == (expected, False)


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        # Vancouver initials that spell a suffix
        ("Smith JR", [P("Smith", "JR")]),
        ("Smith JR, Doe AB", [P("Smith", "JR"), P("Doe", "AB")]),
        ("Wild SR, Jones KC", [P("Wild", "SR"), P("Jones", "KC")]),
        ("Lee SR, Kim II", [P("Lee", "SR"), P("Kim", "II")]),
        # ... and real suffixes, in their own segment or around the initials
        ("King, M. L., Jr.", [P("King", "M. L.", "Jr.")]),
        ("Smith, J. A., III, & Doe, B.", [P("Smith", "J. A.", "III"), P("Doe", "B.")]),
        ("Thorpe RJ Jr, Doe A", [P("Thorpe", "RJ", "Jr"), P("Doe", "A")]),
        ("Gates WH III, Doe A", [P("Gates", "WH", "III"), P("Doe", "A")]),
        ("Reynolds III CF, Doe A", [P("Reynolds", "CF", "III"), P("Doe", "A")]),
        ("Simone CB 2nd, Neal JW", [P("Simone", "CB", "2nd"), P("Neal", "JW")]),
        # a suffix printed after the family name of an inverted name
        ("Mroczek, D. K., & Spiro III, A.", [P("Mroczek", "D. K."), P("Spiro", "A.", "III")]),
        ("Sheffi, Y., & Rice Jr., J. B.", [P("Sheffi", "Y."), P("Rice", "J. B.", "Jr.")]),
        # given name first
        ("Martin Luther King Jr.", [P("King", "Martin Luther", "Jr.")]),
    ],
)
def test_suffixes_and_initials_that_look_like_them(printed, expected):
    assert _split(printed) == (expected, False)


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        # The list's last period went with the reference's own.
        ("Smith, M.R", [P("Smith", "M.R")]),
        ("Buss, K.A., Kiel, E.J", [P("Buss", "K.A."), P("Kiel", "E.J")]),
        ("Cohen, W.M.; Levinthal, D.A", [P("Cohen", "W.M."), P("Levinthal", "D.A")]),
        ("Smith, J.A., & Doe, B.C", [P("Smith", "J.A."), P("Doe", "B.C")]),
        ("Kobzev E.A, Jaja I.F", [P("Kobzev", "E.A"), P("Jaja", "I.F")]),
        # hyphenated initials, the second letter often lowercase
        ("Sartre, J.-P", [P("Sartre", "J.-P")]),
        ("Kanai, R., & Tseng, C.-h", [P("Kanai", "R."), P("Tseng", "C.-h")]),
        ("Meng L-f, Toh W-c", [P("Meng", "L-f"), P("Toh", "W-c")]),
        ("L.-d. Zhang and Y.-t. Wu", [P("Zhang", "L.-d."), P("Wu", "Y.-t.")]),
        # a typographic hyphen (U+2010) inside a name
        ("K. Shearer and J. Magill\u2010Evans", [P("Shearer", "K."), P("Magill\u2010Evans", "J.")]),
        # a meta-analysis asterisk before the list
        ("*Smith, J., & Doe, A.", [P("Smith", "J."), P("Doe", "A.")]),
    ],
)
def test_initials_without_their_last_period(printed, expected):
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
