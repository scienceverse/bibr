"""Tests for paper type and OECD domain constants and validation.

Classification is now handled by the LLM during metadata extraction.
These tests cover the taxonomy constants and validation functions.
"""

from typing import get_args

from bibr.structure.paper_classifier import (
    ALL_OECD_L2_LABELS,
    OECD_L1_LABELS,
    OECD_L2_MAP,
    OECD_L2_TO_L1,
    PAPER_TYPE_LABELS,
    OECDDomainLiteral,
    OECDSubdomainLiteral,
    PaperType,
    PaperTypeLiteral,
    canonicalize_oecd_l2_any,
    validate_oecd_l1,
    validate_oecd_l2,
)

# ---------------------------------------------------------------------------
# PaperType enum tests
# ---------------------------------------------------------------------------


class TestPaperType:
    def test_all_values(self):
        assert PaperType.EMPIRICAL == "empirical"
        assert PaperType.REVIEW == "review"
        assert PaperType.META_ANALYSIS == "meta-analysis"
        assert PaperType.CASE_STUDY == "case-study"
        assert PaperType.COMMENTARY == "commentary"
        assert PaperType.UNKNOWN == "unknown"

    def test_from_string(self):
        assert PaperType("empirical") == PaperType.EMPIRICAL
        assert PaperType("meta-analysis") == PaperType.META_ANALYSIS


# ---------------------------------------------------------------------------
# OECD L1 validation tests
# ---------------------------------------------------------------------------


class TestValidateOecdL1:
    def test_exact_match(self):
        assert validate_oecd_l1("Social Sciences") == "Social Sciences"

    def test_all_labels_exact(self):
        for label in OECD_L1_LABELS:
            assert validate_oecd_l1(label) == label

    def test_none_returns_empty(self):
        assert validate_oecd_l1(None) == ""

    def test_empty_string_returns_empty(self):
        assert validate_oecd_l1("") == ""
        assert validate_oecd_l1("   ") == ""

    def test_case_insensitive_match(self):
        assert validate_oecd_l1("social sciences") == "Social Sciences"
        assert validate_oecd_l1("NATURAL SCIENCES") == "Natural Sciences"

    def test_fuzzy_match(self):
        assert validate_oecd_l1("Medical and Health Science") == "Medical and Health Sciences"
        assert validate_oecd_l1("Engineering & Technology") == "Engineering and Technology"

    def test_hallucinated_label_returns_empty(self):
        assert validate_oecd_l1("Quantum Physics") == ""
        assert validate_oecd_l1("Computer Science") == ""


# ---------------------------------------------------------------------------
# OECD L2 validation tests
# ---------------------------------------------------------------------------


class TestValidateOecdL2:
    def test_exact_match(self):
        assert (
            validate_oecd_l2("Social Sciences", "Psychology and Cognitive Sciences")
            == "Psychology and Cognitive Sciences"
        )

    def test_all_labels_exact(self):
        for l1, labels in OECD_L2_MAP.items():
            for label in labels:
                assert validate_oecd_l2(l1, label) == label

    def test_case_insensitive_match(self):
        assert (
            validate_oecd_l2("Social Sciences", "economics and business")
            == "Economics and Business"
        )

    def test_near_miss_bare_head_noun(self):
        # Real LLM outputs observed in the wild: bare "Psychology" instead of
        # the canonical "Psychology and Cognitive Sciences".
        assert (
            validate_oecd_l2("Social Sciences", "Psychology") == "Psychology and Cognitive Sciences"
        )
        assert (
            validate_oecd_l2("Social Sciences", "psychology") == "Psychology and Cognitive Sciences"
        )

    def test_near_miss_singular_sciences(self):
        # "Cognitive Science" (singular) vs canonical plural "...Sciences".
        assert (
            validate_oecd_l2("Social Sciences", "Cognitive Science")
            == "Psychology and Cognitive Sciences"
        )

    def test_near_miss_economics(self):
        assert validate_oecd_l2("Social Sciences", "Economics") == "Economics and Business"

    def test_cross_l1_canonical_still_works(self):
        assert (
            validate_oecd_l2("Medical and Health Sciences", "Clinical Medicine")
            == "Clinical Medicine"
        )

    def test_garbage_returns_empty(self):
        assert validate_oecd_l2("Social Sciences", "Astrology") == ""
        assert validate_oecd_l2("Social Sciences", "Basket Weaving") == ""

    def test_none_returns_empty(self):
        assert validate_oecd_l2("Social Sciences", None) == ""

    def test_empty_string_returns_empty(self):
        assert validate_oecd_l2("Social Sciences", "") == ""
        assert validate_oecd_l2("Social Sciences", "   ") == ""

    def test_unknown_l1_returns_empty(self):
        assert validate_oecd_l2("Not A Real Domain", "Psychology") == ""
        assert validate_oecd_l2("", "Psychology") == ""

    def test_history_does_not_cross_match_arts_sibling(self):
        # "History" is a substring of the Arts entry's parenthetical example
        # ("Arts (arts, history of arts, ...)"), which used to tie 100 vs
        # 100 against the correct "History and Archaeology" match.
        assert validate_oecd_l2("Humanities and the Arts", "History") == "History and Archaeology"
        assert (
            validate_oecd_l2("Humanities and the Arts", "Arts")
            == "Arts (arts, history of arts, performing arts, music)"
        )

    def test_earth_sciences_does_not_cross_match_natural_science_siblings(self):
        assert (
            validate_oecd_l2("Natural Sciences", "Earth Sciences")
            == "Earth and Related Environmental Sciences"
        )

    def test_ambiguous_generic_head_noun_discarded(self):
        # Every "Engineering and Technology" subfield contains the word
        # "Engineering", and the two Biotechnology entries tie at 100 for a
        # bare "Biotechnology" query — genuinely ambiguous, must not guess.
        assert validate_oecd_l2("Engineering and Technology", "Engineering") == ""
        assert validate_oecd_l2("Engineering and Technology", "Biotechnology") == ""

    def test_all_l2_map_entries_self_match_without_cross_sibling_collision(self):
        # Sanity sweep: every canonical L2 label, when passed back through
        # validate_oecd_l2, must resolve to itself (never a sibling), even
        # through the fuzzy path forced by lowercasing + whitespace noise.
        for l1, labels in OECD_L2_MAP.items():
            for label in labels:
                noisy = f"  {label.lower()}  "
                assert validate_oecd_l2(l1, noisy) == label


# ---------------------------------------------------------------------------
# OECD taxonomy integrity tests
# ---------------------------------------------------------------------------


class TestOECDTaxonomy:
    def test_all_l1_have_l2(self):
        for l1 in OECD_L1_LABELS:
            assert l1 in OECD_L2_MAP, f"Missing L2 map for L1: {l1}"
            assert len(OECD_L2_MAP[l1]) >= 4, f"Too few L2 categories for {l1}"

    def test_no_extra_l2_keys(self):
        for key in OECD_L2_MAP:
            assert key in OECD_L1_LABELS, f"L2 key {key} not in L1 labels"


# ---------------------------------------------------------------------------
# Literal aliases (guided-decoding grammars) must track the runtime lists
# ---------------------------------------------------------------------------


class TestLiteralAliasesTrackLists:
    def test_domain_literal_matches_l1_labels(self):
        assert list(get_args(OECDDomainLiteral)) == OECD_L1_LABELS

    def test_subdomain_literal_matches_flattened_l2(self):
        assert list(get_args(OECDSubdomainLiteral)) == ALL_OECD_L2_LABELS

    def test_paper_type_literal_matches_labels(self):
        assert list(get_args(PaperTypeLiteral)) == PAPER_TYPE_LABELS

    def test_flattened_l2_length(self):
        assert len(ALL_OECD_L2_LABELS) == sum(len(v) for v in OECD_L2_MAP.values())


class TestOecdL2ParentInvariant:
    def test_every_l2_belongs_to_exactly_one_l1(self):
        # The cross-L1 rescue relies on this: a canonical L2 → single parent L1.
        assert len(OECD_L2_TO_L1) == len(ALL_OECD_L2_LABELS)
        for l1, labels in OECD_L2_MAP.items():
            for label in labels:
                assert OECD_L2_TO_L1[label] == l1


# ---------------------------------------------------------------------------
# canonicalize_oecd_l2_any — L1-agnostic L2 canonicalization
# ---------------------------------------------------------------------------


class TestCanonicalizeOecdL2Any:
    def test_exact(self):
        assert (
            canonicalize_oecd_l2_any("Computer and Information Sciences")
            == "Computer and Information Sciences"
        )

    def test_cross_l1_computer_science(self):
        # OECD files Computer Science under Natural Sciences, not Engineering.
        assert canonicalize_oecd_l2_any("Computer Science") == "Computer and Information Sciences"

    def test_near_miss_bare_head_noun(self):
        assert canonicalize_oecd_l2_any("Psychology") == "Psychology and Cognitive Sciences"

    def test_garbage_returns_empty(self):
        assert canonicalize_oecd_l2_any("Astrology") == ""

    def test_none_returns_empty(self):
        assert canonicalize_oecd_l2_any(None) == ""
        assert canonicalize_oecd_l2_any("   ") == ""

    def test_ambiguous_biotechnology_discarded(self):
        # Several "Biotechnology" entries across L1s tie → margin 0 → discard.
        assert canonicalize_oecd_l2_any("Biotechnology") == ""

    def test_all_labels_self_match(self):
        for label in ALL_OECD_L2_LABELS:
            assert canonicalize_oecd_l2_any(f"  {label.lower()}  ") == label
