"""Regression tests for Fix 1C (O(N log N) repeat detector, linear tab strip, pre-OCR formula dedup)."""

from __future__ import annotations

import random
import re
import time

import pytest
from PIL import Image

from bibr.ocr.postprocess import _find_consecutive_repeat, clean_ocr_content
from bibr.pipeline.stages.ocr import ocr_page_regions


def _regex_reference(s: str, min_unit_len: int = 10, min_repeats: int = 10) -> str | None:
    n = len(s)
    if n < min_unit_len * min_repeats:
        return None
    search_s = s[:50_000] if n > 50_000 else s
    max_unit_len = len(search_s) // min_repeats
    if max_unit_len < min_unit_len:
        return None
    pattern = re.compile(
        r"(.{" + str(min_unit_len) + "," + str(max_unit_len) + r"}?)\1{" + str(min_repeats - 1) + ",}",
        re.DOTALL,
    )
    match = pattern.search(search_s)
    if match is None:
        return None
    unit = match.group(1)
    end = match.end()
    while s.startswith(unit, end):
        end += len(unit)
    if unit.startswith(s[end:]):
        end = n
    return s[: match.start()] + unit + s[end:]


def test_find_consecutive_repeat_matches_regex_on_diverse_and_random_inputs() -> None:
    rng = random.Random(2026)  # noqa: S311 — deterministic fuzz test seed
    cases: list[tuple[str, int, int]] = [
        ("a" * 150, 10, 10),
        ("prefix" + "a" * 150 + "suffix", 10, 10),
        ("Z" + ("ABCDEFGHIJKLMNOPQRSTUV" * 10) + ("0123456789" * 12), 10, 10),
        ("P" + ("0123456789" * 20) + "Q", 10, 10),
        ("head\n" + ("line1\nline2\n" * 11) + "tail", 10, 10),
        ("start " + ("0123456789abc" * 10) + "01234", 10, 10),
        ("pq" + ("abcdefghijkl" * 2) + "rs", 10, 2),
        ("pq" + ("abcdefghijkl" * 3) + "rs", 10, 3),
        ("xx" + ("abababababab" * 10) + "yy", 10, 10),
    ]
    for _ in range(100):
        min_u = rng.choice([5, 10])
        min_r = rng.choice([2, 3, 5, 10])
        alphabet = rng.choice(["ab", "abcd", "abcdef0123456789\n"])
        prefix = "".join(rng.choices(alphabet, k=rng.randint(0, 60)))
        suffix = "".join(rng.choices(alphabet, k=rng.randint(0, 60)))
        unit = "".join(rng.choices(alphabet, k=rng.randint(min_u, min_u * 3)))
        reps = rng.randint(min_r - 1, min_r + 2)
        s = prefix + unit * reps + unit[: rng.randint(0, len(unit))] + suffix
        cases.append((s, min_u, min_r))

    for s, min_u, min_r in cases:
        assert _find_consecutive_repeat(s, min_unit_len=min_u, min_repeats=min_r) == _regex_reference(
            s, min_unit_len=min_u, min_repeats=min_r
        )


def test_find_consecutive_repeat_is_fast_on_large_html_table() -> None:
    rows = [
        f"<tr><td>Row{r:04d}</td><td>{r * 1.23:.2f}</td><td>p &lt; 0.{(r % 99) + 1:02d}</td></tr>"
        for r in range(380)
    ]
    html = "<table>" + "".join(rows) + "</table>"
    t0 = time.perf_counter()
    assert _find_consecutive_repeat(html) is None
    assert (time.perf_counter() - t0) < 0.2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"foo\t\t", "foo"),
        ("foo\\t\\t\n", "foo"),
        ("foo\\t   ", r"foo\t"),
        ("foo\\t\n\n", r"foo\t"),
        ("a" + r"\t" * 500 + "x", "a" + r"\t" * 500 + "x"),
    ],
)
def test_clean_ocr_content_trailing_tabs_exact_and_linear(raw: str, expected: str) -> None:
    assert clean_ocr_content(raw) == expected


@pytest.mark.asyncio
async def test_ocr_page_regions_skips_ocr_for_contained_formula_regions() -> None:
    img = Image.new("RGB", (200, 200), "white")
    regions = [
        {"label": "text", "bbox_2d": [100, 100, 900, 300], "task_type": "text"},
        # Fully inside the text region above -> must NOT invoke ocr_fn
        {"label": "inline_formula", "bbox_2d": [200, 120, 400, 180], "task_type": "formula"},
        # Standalone display formula outside any text region -> MUST invoke ocr_fn
        {"label": "display_formula", "bbox_2d": [100, 400, 900, 500], "task_type": "formula"},
    ]
    calls: list[str] = []

    async def fake_ocr(_crop, prompt: str) -> str:
        calls.append(prompt)
        return f"read:{len(calls)}"

    out = await ocr_page_regions(img, regions, 0, "paper.pdf", fake_ocr, None)
    assert len(calls) == 2
    assert [r["native_label"] for r in out] == ["text", "display_formula"]
