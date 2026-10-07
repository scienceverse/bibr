"""Audit fixes: linear-time line-stream and equation-filter regexes, and their exact matches.

The rewritten patterns must answer exactly as before on real text; the timing
guards use inputs on which the old patterns ran for tens of seconds while
the new ones take milliseconds, so their one-second budget stays loose on a
busy machine.
"""

from __future__ import annotations

import re
import time

import pytest

from bibr.extract import equation_extractor as eq
from bibr.extract import ref_line_stream as rls
from bibr.paper_contents import PaperSentence

# Budget for an input the old patterns took 15-60 s on.
_BUDGET_S = 1.0


def _seconds(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Super-linear patterns on hostile lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "method", "text"),
    [
        # Overlapping letter runs around the lower-case letter of a surname.
        pytest.param("_ENDS_WITH_BYLINE", "search", "Aa" * 1500, id="byline"),
        # Every split of a hyphenated run into up to three surnames.
        pytest.param("_AUTHOR_INITIALS_LINE", "match", "A-" * 1200, id="initials"),
        pytest.param("_AUTHOR_GIVEN_LINE", "match", "A-" * 1600, id="given-name"),
        # Adjacent optional whitespace runs on a page-edge text-layer line.
        pytest.param("_PAGE_NUMBER_LINE", "match", "\n " * 20000 + ")", id="page-number"),
        # A word retried to its end from every locator in it.
        pytest.param("_ENDS_WITH_LOCATOR", "search", "doi:" * 16000 + " x", id="locator"),
        pytest.param("_CLOSES_ENTRY", "search", "https://" * 12000 + " x", id="closes-entry"),
    ],
)
def test_line_stream_patterns_are_linear(pattern, method, text):
    compiled = getattr(rls, pattern)
    assert _seconds(getattr(compiled, method), text) < _BUDGET_S


def test_unclosed_parentheses_do_not_stall_the_fallback_filter():
    # "(1 (1 (1 …" with no ")" took the old regex cubic time per sentence.
    text = "Values were " + "(1 " * 3300
    assert _seconds(eq._has_statistical_paren, text) < _BUDGET_S
    assert eq._digit_paren_spans(text) == []


def test_citation_piece_with_a_long_space_run_is_linear():
    assert _seconds(eq._is_citation_piece, "Smith" + " " * 40000 + "x") < _BUDGET_S


def test_lhs_with_a_long_space_run_is_linear():
    # Spaces before the df (a LaTeX LHS), and inside it (the structured LHS takes them)
    lhs = "a" + " " * 120000 + "b(1)"
    assert _seconds(eq._split_lhs_df, lhs) < _BUDGET_S
    assert eq._split_lhs_df(lhs) == (lhs[:-3], "1")
    df = "1" + " " * 120000 + "2"
    assert _seconds(eq._split_lhs_df, f"t({df})") < _BUDGET_S
    assert eq._split_lhs_df(f"t({df})") == ("t", df)


def test_spaced_df_sentence_extracts_quickly():
    text = "We found t(1" + " " * 120000 + "2) = 3.40 here."
    sentence = PaperSentence(text_id=1, text=text, section_id=1, paragraph_id=1)
    start = time.perf_counter()
    found = eq.EquationExtractor().extract_from_sentences([sentence], [])
    assert time.perf_counter() - start < _BUDGET_S
    assert [(e.lhs, e.rhs) for e in found] == [("t", "3.40")]


def test_hostile_reference_lines_segment_quickly():
    lines = ["Smith, J. (2001). A title. Journal, 1, 2-3.", "Aa" * 1500, "A-" * 1200]
    stream = rls.LineStream(
        lines=[
            rls.StreamLine(
                text=text,
                page=1,
                bbox=(0.0, float(i), 10.0, float(i) + 1),
                region=0,
                region_label="text",
                region_first=i == 0,
            )
            for i, text in enumerate(lines)
        ]
    )
    start = time.perf_counter()
    segmentation = rls.segment_line_stream(stream)
    assert time.perf_counter() - start < _BUDGET_S
    assert segmentation is not None


# ---------------------------------------------------------------------------
# Same answers as before
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hoffman BJ, Lance CE.", True),
        ("12. Twenge JM.", True),
        ("[3] Twenge JM.", True),
        ("Twenge JM.", True),
        # A later capital starts the first surname: "Silva" of "deSilva".
        ("deSilva AB, McDonald CD.", True),
        ("Über Müller AB, Øster CD.", True),
        ("New York NY.", False),
        ("Smith AB, deSilva CD.", False),
        ("Smith AB, Jones CD", False),
        ("ABBA AB, Jones CD.", False),
        ("Smith AB, JONES CD.", False),
        ("see Smith AB.", False),
    ],
)
def test_ends_with_byline_matches_as_before(text, expected):
    assert bool(rls._ENDS_WITH_BYLINE.search(text)) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Smith, J. (2001). Title.", True),
        ("Smith-Jones, A. B. (2001).", True),
        ("van der Berg-Smith, J.-L. (2001).", True),
        ("Jean-Pierre de la Fontaine, J. (2001).", True),
        ("O'Brien, JL, & Doe, K.", True),
        ("A-B-C-D, J.", True),
        ("Smith Jones Brown, J.", True),
        ("Smith Jones Brown Green, J.", False),
        ("Psychology, 25(3), 1-10.", False),
        ("Cambridge, MA: Press.", False),
        ("Smith-, J.", False),
    ],
)
def test_author_initials_line_matches_as_before(text, expected):
    assert bool(rls._AUTHOR_INITIALS_LINE.match(text)) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://doi.org/10.1000/xyz.", True),
        ("Retrieved from https://example.org/a ;", True),
        ("doi: 10.1234/abc", True),
        ("DOI:10.1234/abc,", True),
        # "doi:" ends a word whose first locator is the URL before it.
        ("See https://example.org/doi: S0140", True),
        ("(https://example.org/x)", True),
        ("https://example.org and more", False),
        ("https://", False),
        ("x10.1234/abc", False),
    ],
)
def test_ends_with_locator_matches_as_before(text, expected):
    assert bool(rls._ENDS_WITH_LOCATOR.search(text)) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Available at https://example.org/a", True),
        ("10.1234/abc", True),
        ("Journal, 12, 3-4.", True),
        ("reported in", False),
        ("see https:// later", False),
    ],
)
def test_closes_entry_matches_as_before(text, expected):
    assert bool(rls._CLOSES_ENTRY.search(text)) is expected


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("12", 12),
        ("  - 12 -  ", 12),
        ("page 7", 7),
        ("Seite 3 von 9", 3),
        ("12 of 30", 12),
        ("xiv", 14),
        ("12345", None),
        ("Smith 12", None),
    ],
)
def test_page_number_lines_read_as_before(text, value):
    assert rls._page_number_value(text) == value


def _old_digit_paren_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\([^)]*\d[^)]*\)", text)]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no parentheses 12",
        "(1)",
        "(a) (b 2) (c)",
        "(see (left) 2) after (1)",
        "(a (b (c) (1)",
        "((1)",
        "(1 (2 (3",
        ")(1)(",
        "(t(28) = 3.42, p = .003)",
    ],
)
def test_digit_paren_spans_match_the_old_regex(text):
    assert eq._digit_paren_spans(text) == _old_digit_paren_spans(text)


def test_digit_outside_a_nested_parenthesis_keeps_the_candidate():
    # The group runs to the first ")", so the CI before "(adjusted)" counts.
    assert eq._has_statistical_paren("Effects held (95% CI 1.2 to 3.4 (adjusted)).") is True
    assert eq._has_statistical_paren("As shown before (Smith, 2020; Jones, 2019).") is False


@pytest.mark.parametrize(
    ("lhs", "expected"),
    [
        ("t(28)", ("t", "28")),
        ("F( 2, 47 )", ("F", "2, 47")),
        ("χ² (4, N=200) ", ("χ²", "4, N=200")),
        ("y\n(1)", ("y", "1")),
        (" (28)", (" (28)", "")),
        ("t(28", ("t(28", "")),
        ("f(a(b))", ("f(a(b))", "")),
    ],
)
def test_split_lhs_df_as_before(lhs, expected):
    assert eq._split_lhs_df(lhs) == expected


def test_latex_lhs_with_spaces_still_splits_its_df():
    text = "We fit $\\chi^2" + " " * 50 + "(1) = 3.84$ here."
    sentence = PaperSentence(text_id=1, text=text, section_id=1, paragraph_id=1)
    found = eq.EquationExtractor().extract_from_sentences([sentence], [])
    assert [(e.lhs, e.df, e.rhs) for e in found] == [("\\chi^2", "1", "3.84")]


# ---------------------------------------------------------------------------
# Character classes: × (U+00D7) is no capital, ÷ (U+00F7) no lower-case letter
# ---------------------------------------------------------------------------


def test_multiplication_sign_does_not_open_a_dash_led_entry():
    assert rls._is_dash_start("— × 10 cells") is False
    assert rls._is_dash_start("— Ödman, K. 1931.") is True


def test_division_sign_does_not_break_a_word():
    assert rls._CONTINUES_NEXT.search("a ÷-") is None
    assert rls._CONTINUES_NEXT.search("Müller-") is not None


def test_division_sign_is_no_given_name_letter():
    assert rls._AUTHOR_GIVEN_LINE.match("Smith, B÷n") is None
    assert rls._AUTHOR_GIVEN_LINE.match("Smith, Börje") is not None


def test_multiplication_sign_opens_no_citation():
    assert eq._is_citation_piece("×, 2020") is False
    assert eq._is_citation_piece("Ørsted, 2020") is True


# ---------------------------------------------------------------------------
# Roman numerals: canonical forms only
# ---------------------------------------------------------------------------


def _roman(value: int) -> str:
    out = ""
    for amount, symbol in (
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        while value >= amount:
            out += symbol
            value -= amount
    return out


def test_canonical_roman_numerals_keep_their_values():
    for value in range(1, 200):
        token = _roman(value)
        assert rls._roman_value(token) == value
        assert rls._roman_value(token.lower()) == value


@pytest.mark.parametrize("token", ["IIX", "IC", "VX", "IIII", "XXXX", "LL", "civil", "ill", ""])
def test_non_canonical_roman_numerals_are_no_numbers(token):
    assert rls._roman_value(token) is None


def test_a_roman_looking_word_is_no_list_marker():
    assert rls._parse_marker("CIVIL. Engineering handbook, 2001.") is None
    assert rls._parse_marker("XIV. Smith, J. (2001).") == ("roman", 14)
