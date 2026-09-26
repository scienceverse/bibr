"""Direct coverage for ``clean_ocr_content``, the per-region OCR cleanup.

It repairs GLM-OCR prose artifacts (a literal ``\\t`` tab marker, dot-leader
runs, a list marker glued to its text, decode loops). Each repair must leave
alone what only looks like that artifact: LaTeX commands that start with
``\\t``, decimals, DOIs and abbreviations, and the text after a repeated run.
"""

from __future__ import annotations

import pytest

from bibr.ocr.postprocess import _find_consecutive_repeat, clean_ocr_content


@pytest.mark.parametrize(
    "latex",
    [
        r"\theta_{t+1} = \theta_t - \eta \nabla L",
        r"\tau = 2\pi",
        r"\text{RMSE} = \sqrt{x}",
        r"\tilde{x} = 1",
        r"\times",
    ],
)
def test_formula_keeps_latex_commands_that_start_with_t(latex):
    assert clean_ocr_content(latex, formula=True) == latex


def test_formula_still_drops_a_stray_tab_marker():
    assert clean_ocr_content(r"\t\theta = 1\t", formula=True) == r"\theta = 1"


def test_prose_still_drops_the_tab_marker():
    assert clean_ocr_content(r"\tThe results were robust.") == "The results were robust."


@pytest.mark.parametrize(
    "latex",
    [
        # The real Paddle fixture ``formula_simple`` after normalization.
        r"(a)_{n}=(a;q)_{n}=\prod_{k=0}^{n-1}(1-aq^{k}),",
        r"0.5\sum_i x_i",
        r"1.x + y",
        r"a....b",
    ],
)
def test_formula_skips_the_prose_repairs(latex):
    assert clean_ocr_content(latex, formula=True) == latex


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.text", "1. text"),
        ("12)text", "12) text"),
        ("A)text", "A) text"),
        ("(a)text", "(a) text"),
        ("(1)text", "(1) text"),
        ("（1）text", "(1) text"),
        ("1）text", "1) text"),
        ("1.   text", "1. text"),
        # After a closing parenthesis a digit or quote cannot continue a
        # number or an abbreviation.
        ("(1)12 participants were excluded.", "(1) 12 participants were excluded."),
        ("（2）30 items were dropped.", "(2) 30 items were dropped."),
        ("(a)“Always” was coded as 5.", "(a) “Always” was coded as 5."),
        ("1)“Never” was coded as 1.", "1) “Never” was coded as 1."),
        ("1.“Never” was coded as 1.", "1. “Never” was coded as 1."),
    ],
)
def test_list_markers_are_still_normalised(raw, expected):
    assert clean_ocr_content(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "3.14 is the ratio of a circumference to its diameter.",
        "0.5 of the participants responded.",
        "10.1038/s41586-020-2649-2",
        r"0.5\sum_i x_i",
        "U.S. Adults and Social Media Use",
        "e.g. the effect was small.",
        "i.e., the effect was small.",
        "A.B. Smith and C. Jones",
        "(1)-(3) hold for every n.",
        "(a), (b) and (c) are shown.",
    ],
)
def test_numbers_dois_and_abbreviations_are_not_list_markers(text):
    assert clean_ocr_content(text) == text


def test_repeated_punctuation_is_still_collapsed():
    assert clean_ocr_content("Wait.......") == "Wait..."


def test_dot_leader_table_of_contents_keeps_every_entry():
    """A table of contents over 2,048 characters with spaced dot leaders was
    cut to its first entry ("1 Introduction . . . . .")."""
    entries = [f"{n} Section heading number {n}" for n in range(1, 19)]
    toc = "\r\n".join(f"{entry} {'. ' * 55}{n}" for n, entry in enumerate(entries, start=1))
    assert len(toc) >= 2048

    cleaned = clean_ocr_content(toc)

    for entry in entries:
        assert entry in cleaned
    assert cleaned.endswith("18")


def test_consecutive_repeat_keeps_the_text_after_the_run():
    unit = "ABCDEFGHIJKL"

    result = _find_consecutive_repeat("Opening words. " + unit * 12 + " closing words.")

    assert result == "Opening words. " + unit + " closing words."


def test_decode_loop_longer_than_the_search_window_is_collapsed():
    """The repeat search sees the first 50,000 characters. A decode loop that
    runs on past them (GLM allows 16,384 tokens) kept every copy beyond."""
    unit = "The results were robust. "
    loop = "Intro paragraph. " + unit * 2600

    assert len(loop) > 60_000
    assert clean_ocr_content(loop) == "Intro paragraph. The results were robust"
    assert clean_ocr_content(loop + "Closing words.") == (
        "Intro paragraph. The results were robust. Closing words."
    )
