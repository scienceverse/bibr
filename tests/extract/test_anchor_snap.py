"""Tests for anchor→span snapping (LLM segmentation back-end)."""

from bibr.extract.anchor_snap import (
    find_anchor_starts,
    segment_by_anchors,
    starts_to_spans,
)

REF_TEXT = (
    "Smith, J. (2020). A study of things. Journal of Things, 1, 1-10.\n"
    "Doe, A. (2019). Another study. Journal of Stuff, 2, 11-20.\n"
    "Roe, B. (2018). Third study. Journal of Items, 3, 21-30."
)


def test_exact_anchor_match_finds_three_starts():
    anchors = ["Smith, J. (2020).", "Doe, A. (2019).", "Roe, B. (2018)."]
    starts = find_anchor_starts(REF_TEXT, anchors)
    assert len(starts) == 3
    assert starts == sorted(starts)


def test_segment_by_anchors_returns_three_refs_in_order():
    anchors = ["Smith, J. (2020).", "Doe, A. (2019).", "Roe, B. (2018)."]
    refs = segment_by_anchors(REF_TEXT, anchors)
    assert len(refs) == 3
    assert refs[0].startswith("Smith")
    assert refs[1].startswith("Doe")
    assert refs[2].startswith("Roe")
    assert refs[2].endswith("21-30.")  # trailing content kept


def test_fuzzy_match_tolerates_ocr_noise():
    # Realistic ~30-char opening with a classic OCR error ('of' → '0f').
    # Exact match fails, so the fuzzy fallback must still snap it to position 0.
    starts = find_anchor_starts(REF_TEXT, ["Smith, J. (2020). A study 0f things"])
    assert starts == [0]


def test_fuzzy_match_locates_noisy_anchor_mid_text():
    starts = find_anchor_starts(REF_TEXT, ["Doe, A. (2019). An0ther study."])
    assert starts == [REF_TEXT.index("Doe")]


def test_unsnappable_anchor_is_dropped_not_crashed():
    refs = segment_by_anchors(REF_TEXT, ["ZZZ nonexistent anchor text here"])
    assert refs == []


def test_empty_anchors_returns_empty():
    assert segment_by_anchors(REF_TEXT, []) == []
    assert starts_to_spans(REF_TEXT, []) == []


def test_trailing_whitespace_trimmed():
    text = "Smith, J. (2020). Title.   \nDoe, A. (2019). Title2."
    spans = starts_to_spans(text, [0, text.index("Doe")])
    s, e = spans[0]
    assert text[e - 1] not in " \t\n"


def test_duplicate_truncated_anchors_claim_distinct_positions():
    """Two refs sharing a >ANCHOR_LEN-char prefix must yield two distinct spans.

    After truncation to ANCHOR_LEN the two anchors are identical; the
    ``claimed`` set must route the second anchor to the second occurrence.
    """
    shared = "Anderson, P. Q., & Borkowski, R. S. "  # 36 chars > ANCHOR_LEN
    text = f"{shared}(2020a). First study. J. A, 1, 1-9.\n{shared}(2020b). Second study. J. B, 2, 10-19."
    anchors = [f"{shared}(2020a)", f"{shared}(2020b)"]

    starts = find_anchor_starts(text, anchors)

    assert len(starts) == 2
    assert starts == [0, text.index(shared, 1)]
    refs = segment_by_anchors(text, anchors)
    assert len(refs) == 2
    assert "(2020a)" in refs[0]
    assert "(2020b)" in refs[1]


def test_out_of_order_anchors_yield_text_ordered_spans():
    """Anchors emitted out of text order still produce sorted, complete spans."""
    anchors = ["Roe, B. (2018).", "Smith, J. (2020).", "Doe, A. (2019)."]
    refs = segment_by_anchors(REF_TEXT, anchors)
    assert len(refs) == 3
    assert refs[0].startswith("Smith")
    assert refs[1].startswith("Doe")
    assert refs[2].startswith("Roe")


def test_fuzzy_match_on_claimed_position_is_dropped():
    """A fuzzy candidate landing on an already-claimed start is dropped, not duplicated."""
    anchors = [
        "Smith, J. (2020). A study of things",  # exact → claims position 0
        "Smith, J. (2020). A study 0f things",  # OCR-noisy dup → fuzzy lands on 0 → dropped
    ]
    starts = find_anchor_starts(REF_TEXT, anchors)
    assert starts == [0]


# --- Embedded-sublist collisions (exp #2 re-gate residual: lead-author drops) ---
# A sibling reference's truncated anchor can exact-match *inside* a longer
# reference's author list (same research group, sub-list authorship). The
# snap must place such anchors at the true line-start reference, not the
# embedded mid-reference occurrence. Fixtures mirror the six judged failures.


def test_embedded_sublist_year_disambiguation():
    """Caballero/Hoshi-Kashyap: truncation cuts the year, full anchor + line start resolve it."""
    text = (
        "Caballero, R. J., Hoshi, T., & Kashyap, A. K. (2008). Zombie lending and "
        "depressed restructuring in Japan. American Economic Review, 98(5), 1943-1977.\n"
        "Hoshi, T. (2006). Economics of the living dead. Japanese Economic Review, "
        "57(1), 30-49.\n"
        "Hoshi, T., & Kashyap, A. K. (2010). Will the U.S. bank recapitalization "
        "succeed? Journal of Financial Economics, 97(3), 319-336."
    )
    anchors = [
        "Caballero, R. J., Hoshi, T., & Kashyap, A",
        "Hoshi, T. (2006). Economics of the living ",
        "Hoshi, T., & Kashyap, A. K. (2010). Will t",
    ]
    refs = segment_by_anchors(text, anchors)
    assert len(refs) == 3
    assert refs[0].startswith("Caballero") and "1943-1977" in refs[0]
    assert refs[1].startswith("Hoshi, T. (2006)")
    assert refs[2].startswith("Hoshi, T., & Kashyap, A. K. (2010)")


def test_embedded_identical_author_list_prefers_line_start():
    """Torralbo/Walther: identical author sub-list — only the line start disambiguates."""
    text = (
        "Torralbo, A., Walther, D. B., Chai, B., Caddigan, E., Fei-Fei, L., & "
        "Beck, D. M. (2013). Good exemplars of natural scene categories elicit "
        "clearer patterns. PLOS ONE, 8(3), e58594.\n"
        "Walther, D. B., Chai, B., Caddigan, E., Fei-Fei, L., & Beck, D. M. (2011). "
        "Simple line drawings suffice. PNAS, 108(23), 9661-9666."
    )
    anchors = [
        "Torralbo, A., Walther, D. B., Chai, B.,",
        "Walther, D. B., Chai, B., Caddigan, E.,",
    ]
    refs = segment_by_anchors(text, anchors)
    assert len(refs) == 2
    assert refs[0].startswith("Torralbo") and "e58594" in refs[0]
    assert refs[1].startswith("Walther, D. B., Chai, B.")
    assert "(2011)" in refs[1]


def test_triple_collision_claims_in_text_order():
    """Guan/Heltzel: one embedded + two real same-prefix refs all claim correctly."""
    text = (
        "Guan, K., Heltzel, G., & Laurin, K. (2024). Moral dimensions of political "
        "attitudes. In P. A. Robbins (Ed.), Handbook. Routledge.\n"
        "Heltzel, G., & Laurin, K. (2020). Polarization in America. Current Opinion, "
        "34, 179-184.\n"
        "Heltzel, G., & Laurin, K. (2021). Seek and ye shall be fine. Psychological "
        "Science, 32(11), 1696-1708."
    )
    anchors = [
        "Guan, K., Heltzel, G., & Lauri",
        "Heltzel, G., & Laurin, K. (2020). Polar",
        "Heltzel, G., & Laurin, K. (2021). Seek ",
    ]
    refs = segment_by_anchors(text, anchors)
    assert len(refs) == 3
    assert refs[0].startswith("Guan") and "Routledge" in refs[0]
    assert refs[1].startswith("Heltzel") and "(2020)" in refs[1] and "179-184" in refs[1]
    assert refs[2].startswith("Heltzel") and "(2021)" in refs[2]


def test_mid_line_refs_without_line_starts_still_segment():
    """Refs flowed on one line: mid-line claiming must keep working (no hard line-start rule)."""
    text = (
        "Smith, J. (2020). A study. Journal A, 1, 1-10. "
        "Doe, A. (2019). Another. Journal B, 2, 11-20."
    )
    refs = segment_by_anchors(text, ["Smith, J. (2020).", "Doe, A. (2019)."])
    assert len(refs) == 2
    assert refs[0].startswith("Smith")
    assert refs[1].startswith("Doe")


# --- Pre-first-anchor safety net (D2: leading reference drops) ---
# When the first reference's anchor fails to snap (OCR noise / fuzzy miss),
# everything before the first *found* anchor was silently discarded, dropping a
# genuine leading reference. starts_to_spans must emit that text as its own span
# (the downstream segment_filter discards it if it is only header/byline noise).


def test_pre_first_anchor_text_emitted_as_leading_span():
    # Only the Doe anchor's start is supplied; the Smith reference precedes it.
    spans = starts_to_spans(REF_TEXT, [REF_TEXT.index("Doe")])
    assert len(spans) == 2
    assert REF_TEXT[spans[0][0] : spans[0][1]].startswith("Smith")
    assert REF_TEXT[spans[0][0] : spans[0][1]].endswith("1, 1-10.")
    assert REF_TEXT[spans[1][0] : spans[1][1]].startswith("Doe")


def test_leading_whitespace_before_first_anchor_emits_no_span():
    text = "\n\n  Smith, J. (2020). Title.\nDoe, A. (2019). Title2."
    spans = starts_to_spans(text, [text.index("Smith"), text.index("Doe")])
    # Whitespace ahead of the first anchor is not a reference — no phantom span.
    assert len(spans) == 2
    assert text[spans[0][0] : spans[0][1]].startswith("Smith")
    assert text[spans[1][0] : spans[1][1]].startswith("Doe")


def test_first_anchor_at_zero_unchanged():
    # Regression: when the first anchor already sits at position 0 there is no
    # leading text, so the span set is exactly the per-anchor spans.
    spans = starts_to_spans(REF_TEXT, [0, REF_TEXT.index("Doe"), REF_TEXT.index("Roe")])
    assert len(spans) == 3
    assert spans[0][0] == 0


def test_segment_by_anchors_recovers_dropped_leading_reference():
    refs = segment_by_anchors(REF_TEXT, ["Doe, A. (2019).", "Roe, B. (2018)."])
    assert len(refs) == 3
    assert refs[0].startswith("Smith")
    assert refs[1].startswith("Doe")
    assert refs[2].startswith("Roe")
