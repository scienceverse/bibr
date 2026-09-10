"""Metric-definition regressions using synthetic titles, names, and DOI examples."""

from evaluation.validation_metrics import (
    abstract_ned,
    abstract_rouge_l,
    authors_family_f1,
    authors_fullname_f1,
    doi_match,
    title_soft_containment,
    title_soft_match,
)


class TestDoiAssertedAgainstEmptyGold:
    """M3: doi_match was recall-only — an invented DOI cost nothing."""

    def test_asserted_doi_against_empty_gold_is_zero(self):
        # 10.7769_gesec.v7i3.557: gold prints no DOI in its text layer, bibr
        # asserts "10.5555/example.asserted" and used to score None (unscored).
        assert doi_match("10.5555/example.asserted", "") == 0.0

    def test_both_empty_still_abstains(self):
        assert doi_match("", "") is None
        assert doi_match("   ", "") is None

    def test_junk_extraction_that_normalizes_empty_still_abstains(self):
        # normalize_doi() strips a bare "doi:" to nothing — no assertion made.
        assert doi_match("doi:", "") is None

    def test_gold_present_behaviour_unchanged(self):
        assert doi_match("10.5555/123", "10.5555/123") == 1.0
        assert doi_match("10.5555/123", "10.5555/456") == 0.0
        assert doi_match("", "10.5555/123") == 0.0

    def test_abstract_metrics_keep_abstaining(self):
        """Deliberately NOT given the doi_match rule — see the module notes.

        Gold's empty ``abstract`` is a transcription gap, not an adjudicated
        "this paper prints no abstract" (unlike gold's ``doi_printed``), so
        penalizing an asserted abstract would score gold defects as bibr errors.
        """
        assert abstract_rouge_l("Some asserted abstract text", "") is None
        assert abstract_ned("Some asserted abstract text", "") is None


class TestTitleContainmentLengthFloor:
    """M2: containment forgave bibr truncating the title by more than half."""

    def test_bilingual_title_half_dropped_is_not_a_match(self):
        gold = (
            "Como Crescem as Plantas? Um Estudo de Pequenos Jardins em Ambientes Urbanos "
            "da Escola de Exemplo / How Do Plants Grow? A Study of Small Urban Gardens and "
            "Their Changing Conditions at the Example School Throughout the Year"
        )
        pred = (
            "Como Crescem as Plantas? Um Estudo de Pequenos Jardins em Ambientes Urbanos "
            "da Escola de Exemplo"
        )
        assert title_soft_match(pred, gold) == 0.0

    def test_japanese_subtitle_dropped_is_not_a_match(self):
        assert (
            title_soft_match(
                "庭の花を数えてみよう", "庭の花を数えてみよう 花の色と季節の変化について"
            )
            == 0.0
        )

    def test_near_length_containment_still_matches(self):
        # A trailing-word difference (>= 70% of the longer title retained) is
        # still forgiven, in either direction.
        gold = "Seasonal plant growth and garden observations in local schools"
        pred = "Seasonal plant growth and garden observations in local"
        assert title_soft_match(pred, gold) == 1.0
        assert title_soft_match(gold, pred) == 1.0

    def test_exact_soft_match_unaffected(self):
        assert title_soft_match("Hello, World!", "Hello World") == 1.0

    def test_scriptio_continua_eligibility_floor_kept(self):
        """The character floor rejects tiny spaceless prefixes."""
        assert title_soft_match("庭の", "庭の花を数えてみよう 花の色と季節の変化について") == 0.0
        # …and a spaceless prefix that is both substantial and long enough
        # relative to the whole still matches (12 of 17 chars).
        assert (
            title_soft_match("庭の花と季節の変化の関係", "庭の花と季節の変化の関係 について") == 1.0
        )

    def test_single_word_prefix_still_rejected(self):
        assert title_soft_match("Brain", "Brain Imaging in Neural Correlates") == 0.0


class TestTitleSoftContainmentCounter:
    """M2: make reliance on the containment branch visible in the artifact."""

    def test_exact_match_is_not_containment(self):
        assert title_soft_containment("Hello World", "hello world") == 0.0

    def test_containment_match_is_flagged(self):
        gold = "Seasonal plant growth and garden observations in local schools"
        pred = "Seasonal plant growth and garden observations in local"
        assert title_soft_containment(pred, gold) == 1.0

    def test_non_match_is_not_containment(self):
        assert title_soft_containment("Hello World", "Goodbye World") == 0.0

    def test_no_gold_title_abstains(self):
        assert title_soft_containment("Hello World", "") is None


class TestAuthorsFamilyEmptyFamilyFallback:
    """M1a: authors_f1 returned 1.0 for a zero-author prediction."""

    def test_zero_authors_against_empty_family_gold_is_zero(self):
        gold = [{"given": "Director-General", "family": ""}]
        assert authors_family_f1([], gold) == 0.0

    def test_empty_family_gold_matched_by_family_carrying_extraction(self):
        gold = [{"given": "Director-General", "family": ""}]
        ext = [{"given": "", "family": "Director-General"}]
        assert authors_family_f1(ext, gold) == 1.0

    def test_given_only_extraction_counts_against_precision(self):
        gold = [{"given": "Robin", "family": "Example"}]
        ext = [{"given": "Robin", "family": ""}]
        assert authors_family_f1(ext, gold) == 0.0

    def test_both_genuinely_empty_still_one(self):
        assert authors_family_f1([], []) == 1.0

    def test_normal_family_matching_unchanged(self):
        gold = [{"family": "Smith", "given": "J"}, {"family": "Jones", "given": "A"}]
        assert authors_family_f1(gold, gold) == 1.0
        ext = [{"family": "Smith", "given": "J"}, {"family": "Brown", "given": "B"}]
        assert abs(authors_family_f1(ext, gold) - 0.5) < 1e-9


class TestAuthorsFullnameOrderInsensitive:
    """M1b: the "boundary-insensitive" metric was order-sensitive."""

    def test_family_first_byline_matches(self):
        gold = [{"given": "Kamaria Robin", "family": "Example"}]
        ext = [{"given": "Example Kamaria", "family": "Robin"}]
        assert authors_fullname_f1(ext, gold) == 1.0

    def test_boundary_shift_still_perfect(self):
        gold = [{"given": "Amélia Campos", "family": "de Exemplo"}]
        ext = [{"given": "Amélia", "family": "Campos de Exemplo"}]
        assert authors_fullname_f1(ext, gold) == 1.0

    def test_initial_abbreviation_still_matches(self):
        gold = [{"given": "John", "family": "Smith"}]
        ext = [{"given": "J.", "family": "Smith"}]
        assert authors_fullname_f1(ext, gold) == 1.0

    def test_different_people_still_do_not_match(self):
        gold = [{"given": "Anna", "family": "Kowalska"}]
        ext = [{"given": "Piotr", "family": "Nowak"}]
        assert authors_fullname_f1(ext, gold) == 0.0


class TestAuthorsFullnameTokenGuard:
    """M1c: Jaro-Winkler's prefix boost scored a dropped surname 0.92+."""

    def test_dropped_surname_does_not_match(self):
        gold = [
            {"given": "Robin", "family": "Example"},
            {"given": "Avery Morgan", "family": "Sample"},
        ]
        ext = [{"given": "Robin", "family": ""}, {"given": "Avery Morgan Sample", "family": ""}]
        assert authors_fullname_f1(ext, gold) == 0.5

    def test_middle_initial_dropped_still_matches(self):
        gold = [{"given": "John A.", "family": "Smith"}]
        ext = [{"given": "John", "family": "Smith"}]
        assert authors_fullname_f1(ext, gold) == 1.0

    def test_extra_extracted_token_still_matches(self):
        # The guard is one-directional: only a SHORTER candidate is suspect.
        gold = [{"given": "John", "family": "Smith"}]
        ext = [{"given": "John Andrew", "family": "Smith"}]
        assert authors_fullname_f1(ext, gold) == 1.0
