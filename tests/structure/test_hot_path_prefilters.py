"""The hot-path prefilters must be necessary conditions, not heuristics.

``bibr serve`` runs one LitServe worker with ``enable_async=True``, so
synchronous regex CPU inside a coroutine delays every co-resident request.
Several sweeps now skip their expensive pass when a cheap prescan fails.
Each prescan is a strict superset of what the sweep can match — these tests
pin that, because a prescan that is merely *usually* right silently drops
real extractions.
"""

from __future__ import annotations

import random

import pytest


def _corpus(n: int = 3000) -> list[str]:
    rng = random.Random(20260904)  # noqa: S311 - deterministic test fixture, not crypto
    words = [
        "the",
        "study",
        "reported",
        "Table",
        "3",
        "and",
        "Fig.",
        "2b",
        "alongside",
        "Equation",
        "(4)",
        "in",
        "Section",
        "2.1",
        "with",
        "supplementary",
        "material",
        "p",
        "=",
        ".03",
        "chi2",
        "12,345",
        "participants",
        "stable",
        "notable",
        "frequency",
        "sequence",
        "database",
        "equal",
        "tab",
        "tbl",
        "eq",
        "figs",
        "Supplemental",
        "Data",
        "§§2.1",
        "cm^{2}",
        "effective^{9}",
        "data",
        "availability",
        "funding",
        "statement",
        "ethics",
        "approval",
    ]
    return [
        " ".join(rng.choice(words) for _ in range(rng.randint(4, 30))) + rng.choice(". ?!")
        for _ in range(n)
    ]


def test_xref_prescan_never_hides_a_real_match():
    from bibr.structure import xref_utils as xu

    patterns = [
        xu.TABLE_XREF_RE,
        xu.FIGURE_XREF_RE,
        xu.SUPP_NAMED_XREF_RE,
        xu.EQUATION_XREF_RE,
        xu.SECTION_XREF_RE,
    ]

    for text in _corpus():
        if any(pattern.search(text) for pattern in patterns):
            assert xu._XREF_PRESCAN_RE.search(text), text


def test_broad_equation_prescan_never_hides_a_real_match():
    from bibr.extract import equation_extractor as ee

    for text in _corpus():
        if ee._BROAD_EQUATION_RE.search(text):
            assert ee._BROAD_OP_PRESCAN_RE.search(text), text


def test_acronym_prescan_never_hides_a_real_match():
    from bibr.structure import citation_linker as cl

    for text in _corpus():
        if cl._ACRONYM_DEFINITION_RE.search(text):
            assert cl._ACRONYM_PARENTHETICAL_RE.search(text), text


@pytest.mark.parametrize(
    "field",
    ["funding_statement", "coi_statement", "ethics_statement", "data_availability"],
)
def test_combined_field_anchor_matches_the_individual_patterns(field):
    from bibr.extract.statement_scan import _ANCHOR_ANY, _ANCHORS

    texts = [
        *_corpus(400),
        "Funding: this work was supported by the NIH under grant no. 12345.",
        "Data availability: all data are available on OSF at https://osf.io/abcd.",
        "The authors declare no conflicts of interest.",
        "Ethical approval was granted by the institutional review board.",
        "Competing interests: none declared.",
        "Informed consent was obtained from all participants.",
        "Code availability: the analysis code is deposited in a repository.",
        "The data generated in this study are available from the corresponding author.",
    ]
    for text in texts:
        expected = any(pattern.search(text) for pattern in _ANCHORS[field])
        assert bool(_ANCHOR_ANY[field].search(text)) is expected, (field, text)


def test_alphanumeric_linewrap_guard_never_hides_a_real_match():
    from bibr.input import consolidate_text as ct

    texts = [
        "Vitamin B-\n12 was measured.",
        "The 5-\nHT receptor was blocked.",
        "No wrap here at all.",
        "A hyphen - but no newline.",
        "Word-\n wrapped across lines.",
        *_corpus(400),
    ]
    for text in texts:
        if ct._ALPHANUMERIC_LINEWRAP_RE.search(text):
            assert ct._HYPHEN_AT_LINEBREAK_RE.search(text), text


def test_alphanumeric_linewrap_bridging_still_works():
    from bibr.input.consolidate_text import _bridge_alphanumeric_linewraps

    assert _bridge_alphanumeric_linewraps("Vitamin B-\n12 was measured.") == (
        "Vitamin B-12 was measured."
    )
    assert _bridge_alphanumeric_linewraps("no wrap") == "no wrap"
