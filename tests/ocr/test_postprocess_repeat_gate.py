"""Consecutive-repeat detection: cost and equivalence.

``_find_consecutive_repeat`` uses a backreference + non-greedy quantifier
pattern whose cost is superlinear in the searched length (2,048 chars ≈ 16 ms,
32,768 ≈ 4.3 s, 50,000 ≈ 11 s). On 131,199 real OCR regions, 737 of the 943
regions over 2,048 characters produced no repeat at all — pure waste with an
unbounded tail. An exact O(n) n-gram precondition skips the search in that
case without changing any result.
"""

from __future__ import annotations

import random
import time

from bibr.ocr.postprocess import _find_consecutive_repeat, _has_repeated_ngram


def _unrepetitive(length: int) -> str:
    """Text in which no 10-character gram recurs, so no repeat can exist."""
    rng = random.Random(20260903)  # noqa: S311 - deterministic test fixture, not crypto
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    words = []
    total = 0
    while total < length:
        word = "".join(rng.choice(alphabet) for _ in range(rng.randint(4, 11)))
        words.append(word)
        total += len(word) + 1
    return " ".join(words)[:length]


def test_genuine_repeat_is_still_truncated():
    text = "Table 1 continued from the previous page. " * 40

    result = _find_consecutive_repeat(text)

    assert result == "Table 1 continued from the previous page. "


def test_repeat_after_a_prefix_keeps_the_prefix():
    text = _unrepetitive(400) + "ABCDEFGHIJKL" * 40

    result = _find_consecutive_repeat(text)

    assert result is not None
    assert result.endswith("ABCDEFGHIJKL")
    assert result.startswith(text[:100])


def test_ngram_gate_is_a_necessary_condition_not_a_heuristic():
    """A unit repeated N times contains its own leading 10-gram N times."""
    unit = "abcdefghijklmno"

    assert _has_repeated_ngram(unit * 10, 10, 10)
    assert not _has_repeated_ngram(unit * 9, 10, 10)


def test_non_repetitive_region_is_rejected_without_running_the_regex():
    text = _unrepetitive(50_000)

    assert not _has_repeated_ngram(text, 10, 10)

    started = time.perf_counter()
    assert _find_consecutive_repeat(text) is None
    elapsed = time.perf_counter() - started

    # The unguarded search took ~11 s at this length.
    assert elapsed < 0.5, f"took {elapsed:.3f}s"


def test_over_length_region_is_also_bounded():
    """Past ``_MAX_SEARCH_LEN`` the unit ceiling follows the searched length,
    not the input length — deriving it from the input left the cap
    ineffective, so a 100,000-character region cost twice a 50,000-character
    one (23 s measured)."""
    text = _unrepetitive(120_000)

    started = time.perf_counter()
    assert _find_consecutive_repeat(text) is None
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"took {elapsed:.3f}s"
