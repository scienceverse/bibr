"""Ordered line→text alignment (the geometry segmenter's span path).

The geom segmenter knows the exact ordered sequence of reference-section
lines and which of them open a reference (B-REF). ``align_line_starts``
exploits that order: it walks the lines against ``ref_text`` with a
monotonic cursor, so a sibling reference's opening embedded inside an
EARLIER reference's author list (sub-list authorship) can never steal the
match — the cursor has already passed it. The global anchor search
(``find_anchor_starts``) has to disambiguate those collisions with
line-start and scoring heuristics; the ordered path excludes them by
construction.
"""

from bibr.extract.anchor_snap import align_line_starts

REF_TEXT = (
    "Smith, J. (2020). A study of things. Journal of Things, 1, 1-10.\n"
    "Doe, A. (2019). Another study. Journal of Stuff, 2, 11-20.\n"
    "Roe, B. (2018). Third study. Journal of Items, 3, 21-30."
)

# The layout lines behind REF_TEXT: each reference wraps onto two physical
# lines; rows were joined with spaces when ref_text was built.
LINES = [
    "Smith, J. (2020). A study of things.",
    "Journal of Things, 1, 1-10.",
    "Doe, A. (2019). Another study. Journal",
    "of Stuff, 2, 11-20.",
    "Roe, B. (2018). Third study.",
    "Journal of Items, 3, 21-30.",
]
BOUNDS = [True, False, True, False, True, False]


def test_boundary_lines_align_to_their_text_starts():
    starts = align_line_starts(REF_TEXT, LINES, BOUNDS)
    assert starts == [0, REF_TEXT.index("Doe"), REF_TEXT.index("Roe")]


def test_embedded_sublist_mid_line_resolved_by_cursor():
    """Sub-list authorship with the year truncated out of the probe AND no
    line start at the true opening: the global search ties and picks the
    embedded (wrong, earlier) occurrence; the cursor path must pick the true
    one because it has already walked past reference 1's lines.
    """
    text = (
        "Alpha, B., Gamma, D., Delta, E. F., Epsilon, G. H., & Zeta, I. (2001). "
        "One thing. Journal A, 1, 1-9. "
        "Gamma, D., Delta, E. F., Epsilon, G. H., & Zeta, I. (2002). "
        "Other thing. Journal B, 2, 2-8."
    )
    lines = [
        "Alpha, B., Gamma, D., Delta, E. F., Epsilon, G. H., & Zeta, I. (2001).",
        "One thing. Journal A, 1, 1-9.",
        "Gamma, D., Delta, E. F., Epsilon, G. H., & Zeta, I. (2002).",
        "Other thing. Journal B, 2, 2-8.",
    ]
    starts = align_line_starts(text, lines, [True, False, True, False])
    true_second = text.index("Gamma, D., Delta, E. F., Epsilon, G. H., & Zeta, I. (2002)")
    assert starts == [0, true_second]


def test_furniture_lines_are_skipped_without_breaking_alignment():
    """Page numbers and running headers exist in the geometry stream but not
    in ref_text (or only case-mangled). They must neither crash nor drag the
    cursor forward past genuine references.
    """
    lines = [
        "Smith, J. (2020). A study of things.",
        "Journal of Things, 1, 1-10.",
        "17",  # page number — too short to be a reference opening
        "JOURNAL OF THINGS",  # running header — case-mangled, absent from ref_text
        "Doe, A. (2019). Another study. Journal",
        "of Stuff, 2, 11-20.",
        "Roe, B. (2018). Third study.",
    ]
    bounds = [True, False, False, False, True, False, True]
    starts = align_line_starts(REF_TEXT, lines, bounds)
    assert starts == [0, REF_TEXT.index("Doe"), REF_TEXT.index("Roe")]


def test_wide_letter_spacing_line_text_is_collapsed_before_matching():
    """Text-layer lines with pathological whitespace (tabs/CR/NBSP between
    words) must still exact-match against the whitespace-collapsed ref_text.
    """
    text = "1. Zietsch, B. (2020). Eye colour. Journal C, 3, 1-5.\n2. Other, A. (2019). Thing."
    lines = [
        "1.\t\r \xa0Zietsch,\t\r \xa0B.\t\r \xa0(2020).",
        "Eye colour. Journal C, 3, 1-5.",
        "2.\t\r \xa0Other,\t\r \xa0A.\t\r \xa0(2019). Thing.",
    ]
    starts = align_line_starts(text, lines, [True, False, True])
    assert starts == [0, text.index("2. Other")]


def test_short_boundary_probe_is_skipped_not_misplaced():
    # A too-short B-REF line ("Ibid.") cannot be located reliably — skip it
    # rather than fuzzy-matching it somewhere destructive.
    starts = align_line_starts(
        REF_TEXT, ["Smith, J. (2020). A study of things.", "Ibid."], [True, True]
    )
    assert starts == [0]


def test_boundary_line_missing_from_ref_text_is_dropped():
    """A B-REF line whose text never made it into ref_text (dropped row) is
    dropped as a boundary; the following reference must still align.
    """
    lines = [
        "Smith, J. (2020). A study of things.",
        "Vanished, Z. (1999). Not in the row text at all.",
        "Roe, B. (2018). Third study.",
    ]
    starts = align_line_starts(REF_TEXT, lines, [True, True, True])
    assert starts == [0, REF_TEXT.index("Roe")]


def test_order_violation_recovers_boundary_behind_cursor():
    """The layout line stream (pdfium content order) and the row text
    (layout-region order) can locally disagree about the order of
    same-first-author references. A boundary whose only exact match lies
    BEHIND the cursor must be claimed there (soft monotonicity) instead of
    dropped — without moving the cursor backward.
    """
    text = (
        "Johnson, E. J., Shu, S. B., & Weber, E. (2012). Beyond nudges. "
        "Marketing Letters, 23, 487-504.\n"
        "Johnson, E. J., & Goldstein, D. (2003). Do defaults save lives? "
        "Science, 302, 1338-1339.\n"
        "Kahneman, D. (2011). Thinking fast and slow. FSG."
    )
    lines = [  # the two Johnsons arrive swapped relative to `text`
        "Johnson, E. J., & Goldstein, D. (2003).",
        "Do defaults save lives? Science, 302, 1338-1339.",
        "Johnson, E. J., Shu, S. B., & Weber, E. (2012).",
        "Beyond nudges. Marketing Letters, 23, 487-504.",
        "Kahneman, D. (2011). Thinking fast and slow. FSG.",
    ]
    starts = align_line_starts(text, lines, [True, False, True, False, True])
    assert sorted(starts) == [
        0,
        text.index("Johnson, E. J., & Goldstein"),
        text.index("Kahneman"),
    ]


def test_behind_cursor_fallback_never_reclaims_a_start():
    """A duplicated boundary line must not re-claim the start the first copy
    already took via the behind-cursor fallback.
    """
    text = "Smith, J. (2020). A study of things. Journal of Things, 1, 1-10."
    lines = ["Smith, J. (2020). A study of things.", "Smith, J. (2020). A study of things."]
    starts = align_line_starts(text, lines, [True, True])
    assert starts == [0]


def test_ocr_noisy_boundary_line_recovered_by_fuzzy():
    lines = [
        "Smith, J. (2020). A study of things.",
        "Journal of Things, 1, 1-10.",
        "D0e, A, (2019), An0ther study.",  # OCR noise — exact fails, fuzzy snaps
    ]
    starts = align_line_starts(REF_TEXT, lines, [True, False, True])
    assert starts == [0, REF_TEXT.index("Doe")]


def test_repeated_openings_claim_successive_occurrences():
    """Identical truncated openings (2020a/2020b twins): the cursor must send
    the second line to the second occurrence, never re-claim the first.
    """
    shared = "Anderson, P. Q., & Borkowski, R. S. "
    text = (
        f"{shared}(2020a). First study. Journal A, 1, 1-9.\n"
        f"{shared}(2020b). Second study. Journal B, 2, 10-19."
    )
    lines = [f"{shared}(2020a). First study.", f"{shared}(2020b). Second study."]
    starts = align_line_starts(text, lines, [True, True])
    assert starts == [0, text.index(shared, 1)]


def test_empty_inputs():
    assert align_line_starts(REF_TEXT, [], []) == []
    assert align_line_starts("", ["Smith, J. (2020). A study."], [True]) == []
