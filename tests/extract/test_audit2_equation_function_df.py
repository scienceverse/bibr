"""A function's argument is not a statistic's degrees of freedom."""

import pytest

from bibr.extract.equation_extractor import EquationExtractor
from bibr.paper_contents import PaperSentence


def _equations(text: str):
    sentence = PaperSentence(text_id=1, text=text, section_id=0, paragraph_id=0)
    return [
        (eq.lhs, eq.df, eq.comp, eq.rhs)
        for eq in EquationExtractor().extract_from_sentences([sentence], [])
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Let F(x) = 3 for every x in the domain.", ("F(x)", "", "=", "3")),
        ("The survival function S(t) = 0.82 at one year.", ("S(t)", "", "=", "0.82")),
    ],
)
def test_a_function_value_keeps_its_argument_in_the_name(text, expected):
    assert _equations(text) == [expected]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reliability was high, ICC(2,1) = .87.", ("ICC", "2,1", "=", ".87")),
        ("The effect held, t(28) = 2.10, p = .04.", ("t", "28", "=", "2.10")),
    ],
)
def test_a_numeric_parenthesis_is_still_split_off(text, expected):
    assert _equations(text)[0] == expected
