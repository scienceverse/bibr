"""Metric-definition regressions using synthetic titles, names, and DOI examples."""

from evaluation.evaluate import extract_comparable_from_json, ref_field_counts
from evaluation.validation_metrics import (
    _gold_ref_matchable,
    _ref_similarity,
    abstract_ned,
    abstract_rouge_l,
    authors_family_f1,
    authors_fullname_f1,
    doi_match,
    match_references,
    ref_matching_f1,
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


class TestReferencesPrintedWithoutTitles:
    """A reference list without titles (chemistry, physics) has only names and numbers to match on.

    The first-author level read the first word of the printed author string, so
    a list printed initials first ("J. M. Smith") never matched its gold.
    """

    GOLD = {"authors_list": ["Example", "Sample"], "author": "Example", "year": "2010"}

    def test_initials_first_string_matches_the_gold_family_name(self):
        ext = {"author": "R. T. Example, A. Sample", "authors": "R. T. Example, A. Sample"}
        assert _ref_similarity({**ext, "year": "2010"}, self.GOLD) == 0.7

    def test_family_first_string_still_matches(self):
        ext = {"author": "Example, R. T., & Sample, A.", "authors": "Example, R. T., & Sample, A."}
        assert _ref_similarity({**ext, "year": "2010"}, self.GOLD) == 0.7

    def test_spelled_out_given_name_first_matches(self):
        ext = {"authors": "Robin Example and Avery Sample", "year": "2010"}
        assert _ref_similarity(ext, self.GOLD) == 0.7

    def test_initials_after_the_family_name_match(self):
        ext = {"authors": "Example RT, Sample A", "year": "2010"}
        assert _ref_similarity(ext, self.GOLD) == 0.7

    def test_a_later_author_is_not_the_first_author(self):
        ext = {"authors": "A. Sample, R. T. Example", "year": "2010"}
        assert _ref_similarity(ext, self.GOLD) == 0.0

    def test_another_year_does_not_match(self):
        ext = {"authors": "R. T. Example, A. Sample", "year": "2011"}
        assert _ref_similarity(ext, self.GOLD) == 0.0

    def test_compound_family_name_is_found_by_its_longest_word(self):
        gold = {"authors_list": ["van der Sample"], "year": "2001"}
        assert _ref_similarity({"authors": "J. van der Sample", "year": "2001"}, gold) == 0.7
        assert _ref_similarity({"authors": "Sample, J. van der", "year": "2001"}, gold) == 0.7
        assert _ref_similarity({"authors": "J. van Example", "year": "2001"}, gold) == 0.0

    @staticmethod
    def _printed(authors: str, year: str = "1999") -> dict:
        # The evaluator gives a printed author string under both keys.
        return {"author": authors, "authors": authors, "year": year}

    def test_two_printed_strings_agree_on_the_name_words_of_the_shorter(self):
        assert _ref_similarity(self._printed("J. Example"), self._printed("Example, John")) == 0.7
        assert _ref_similarity(self._printed("John Example"), self._printed("Example J")) == 0.7
        assert _ref_similarity(self._printed("Robin Example"), self._printed("Avery Sample")) == 0.0

    def test_a_shared_first_initial_is_not_a_shared_first_author(self):
        assert _ref_similarity(self._printed("R. Example"), self._printed("R. Sample")) == 0.0
        assert _ref_similarity(self._printed("R.T. Example"), self._printed("R.T. Sample")) == 0.0
        assert _ref_similarity(self._printed("R. Example"), self._printed("R. Example")) == 0.7

    def test_two_family_first_strings_still_match_on_their_first_word(self):
        # The rule from before the name-word comparison is kept beside it.
        assert _ref_similarity(self._printed("Example RT"), self._printed("Example AB")) == 0.7

    def test_year_volume_and_first_page_identify_a_reference(self):
        gold = {"year": "2010", "volume": "35", "first_page": "487", "container": "J. Synth. Ex."}
        assert _ref_similarity({"year": "2010", "volume": "35", "first_page": "487"}, gold) == 0.8
        assert _ref_similarity({"year": "2010", "volume": "35", "first_page": "e487"}, gold) == 0.0
        assert _ref_similarity({"year": "2010", "volume": "36", "first_page": "487"}, gold) == 0.0
        # A page range in one field still gives its first page.
        assert (
            _ref_similarity({"year": "2010", "volume": "35", "first_page": "487-499"}, gold) == 0.8
        )

    def test_coordinates_need_all_three_values_on_both_sides(self):
        gold = {"year": "2010", "first_page": "487"}
        assert _ref_similarity({"year": "2010", "first_page": "487"}, gold) == 0.0

    def test_coordinates_pair_two_references_of_one_author_and_year(self):
        gold = [
            {"authors_list": ["Example"], "year": "2019", "volume": "7", "first_page": "100"},
            {"authors_list": ["Example"], "year": "2019", "volume": "9", "first_page": "250"},
        ]
        ext = [
            {"authors": "R. Example", "year": "2019", "volume": "9", "first_page": "250"},
            {"authors": "R. Example", "year": "2019", "volume": "7", "first_page": "100"},
        ]
        assert set(match_references(ext, gold).pairs) == {(0, 1), (1, 0)}

    def test_a_titleless_list_printed_initials_first_scores_full_marks(self):
        gold = [
            {"authors_list": ["Example", "Sample"], "author": "Example", "year": "2010"},
            {"authors_list": ["Sample"], "author": "Sample", "year": "1989", "volume": "14"},
            {"authors_list": ["Placeholder"], "author": "Placeholder", "year": "1997"},
        ]
        ext = [
            {"author": "R. T. Example, A. Sample", "authors": "R. T. Example, A. Sample"},
            {"author": "A. Sample", "authors": "A. Sample", "volume": "14"},
            {"author": "C. K. Placeholder", "authors": "C. K. Placeholder"},
        ]
        for ref, year in zip(ext, ("2010", "1989", "1997"), strict=True):
            ref["year"] = year
        assert ref_matching_f1(ext, gold) == 1.0

    def test_a_title_match_still_outranks_the_levels_below_it(self):
        gold = {"title": "A synthetic study of classroom gardens", "year": "2010", "volume": "3"}
        ext = {**gold, "first_page": "12", "authors": "R. Example"}
        assert _ref_similarity(ext, {**gold, "first_page": "12"}) == 0.9

    def test_unstructured_gold_needs_the_author_as_a_whole_word(self):
        # "em" is in "system" and "ja" in "Jakarta": an initials block is no evidence.
        gold = {"unstructured": "Placeholder, K. (1994). A system of examples from Jakarta. 4, 12."}
        assert _ref_similarity({"authors": "Sample EM", "year": "1994"}, gold) == 0.0
        assert _ref_similarity({"authors": "Sample JA", "year": "1994"}, gold) == 0.0
        # Another author's initials are not this author either.
        gold = {"unstructured": "Placeholder EM (1994) A study of synthetic examples. J Ex 4:12"}
        assert _ref_similarity({"authors": "Sample EM", "year": "1994"}, gold) == 0.0
        assert _ref_similarity({"authors": "Placeholder EM", "year": "1994"}, gold) == 0.6

    def test_unstructured_gold_matches_a_two_letter_family_name(self):
        gold = {"unstructured": "Ex, R. (2010). Notes on synthetic gardens. J. Ex. 4, 12."}
        assert _ref_similarity({"authors": "R. Ex", "year": "2010"}, gold) == 0.6

    def test_unstructured_gold_matches_an_initials_first_extraction(self):
        gold = {
            "unstructured": "Éxample, R. T. (2010). Cognitive biases in decisions. J. Ex. 4, 12."
        }
        ext = {
            "author": "R. T. Éxample",
            "authors": "R. T. Éxample",
            "year": "2010",
            "title": "Cognitive biases in decisions",
        }
        assert _ref_similarity(ext, gold) == 0.6


class TestGoldReferencesWithNothingToMatch:
    """Publisher XML gives some references as plain text; their gold entry is empty.

    No extraction can match such an entry. Counted as a miss, and its correct
    extraction as a spurious reference, it cost every tool the same points and
    capped a perfect extraction below 1.0.
    """

    FULL = {"title": "A synthetic study of classroom gardens", "year": "2010"}
    OTHER = {"title": "Field notes on imaginary rivers", "year": "2012"}
    EXTRA = {"title": "An unrelated report on paper kites", "year": "2014"}

    def test_a_perfect_extraction_scores_full_marks(self):
        assert ref_matching_f1([self.FULL, self.OTHER], [self.FULL, {}]) == 1.0

    def test_one_unmatched_extraction_is_forgiven_per_empty_gold_reference(self):
        # Three extracted, one matched: one of the other two may be the empty
        # gold reference, the third is spurious. Precision 1/2, recall 1/1.
        score = ref_matching_f1([self.FULL, self.OTHER, self.EXTRA], [self.FULL, {}])
        assert round(score, 4) == 0.6667

    def test_missing_the_empty_reference_costs_nothing(self):
        assert ref_matching_f1([self.FULL], [self.FULL, {}]) == 1.0

    def test_a_missed_matchable_reference_still_costs_recall(self):
        score = ref_matching_f1([self.FULL], [self.FULL, self.OTHER, {}])
        assert round(score, 4) == 0.6667

    def test_nothing_to_score_when_every_gold_reference_is_empty(self):
        assert ref_matching_f1([self.FULL], [{}, {"container": "J. Ex."}]) is None
        assert ref_matching_f1([], [{}]) is None

    def test_no_extraction_against_matchable_gold_is_zero(self):
        assert ref_matching_f1([], [self.FULL, {}]) == 0.0

    def test_spurious_references_against_no_gold_are_zero(self):
        assert ref_matching_f1([self.FULL], []) == 0.0

    def test_matchable_is_what_some_level_of_the_cascade_can_use(self):
        matchable = [
            {"doi": "10.5555/example.1"},
            {"title": "A synthetic study"},
            {"year": "2010", "volume": "35", "first_page": "487"},
            {"year": "2010", "authors_list": ["Example"]},
            {"year": "2010", "author": "Example, R."},
            {"unstructured": "Example, R. (2010). A synthetic study. J. Ex. 4, 12."},
        ]
        unmatchable = [
            {},
            {"year": "2010"},
            {"container": "J. Ex.", "volume": "35"},
            {"year": "2010", "volume": "35"},
            {"authors_list": ["Example"]},
            {"unstructured": "too short"},
        ]
        assert all(_gold_ref_matchable(ref) for ref in matchable)
        assert not any(_gold_ref_matchable(ref) for ref in unmatchable)

    def test_no_extraction_matches_an_unmatchable_gold_reference(self):
        rich = {
            "doi": "10.5555/example.1",
            "title": "A synthetic study",
            "author": "Example, R.",
            "authors": "Example, R.",
            "year": "2010",
            "container": "J. Ex.",
            "volume": "35",
            "first_page": "487",
        }
        for gold in (
            {},
            {"year": "2010"},
            {"year": "2010", "volume": "35"},
            {"container": "J. Ex."},
        ):
            assert _ref_similarity(rich, gold) == 0.0

    def test_a_set_aside_reference_is_in_no_field_denominator(self):
        gold = [self.FULL, {"year": "2010"}]
        match = match_references([self.FULL], gold)
        assert match.gold_matchable == (True, False)
        assert match.gold_has_field["year"] == (True, False)
        counts = ref_field_counts([self.FULL], gold, match=match)
        assert counts["gold_refs"] == 2
        assert counts["gold_refs_unmatchable"] == 1
        assert counts["year_gold"] == 1

    def test_a_gold_record_may_carry_the_printed_string(self):
        record = {"bib": [{"bib_id": 1, "unstructured": "Example, R. (2010). A study. J. Ex. 4."}]}
        gold = extract_comparable_from_json(record, is_gold=True)
        assert gold["references"][0]["unstructured"].startswith("Example, R.")
        assert _gold_ref_matchable(gold["references"][0])
        # A prediction's key of that name is not read: an export has none.
        assert extract_comparable_from_json(record)["references"] == [{}]
        ext = [{"author": "Example R", "authors": "Example R", "year": "2010"}]
        assert ref_matching_f1(ext, gold["references"]) == 1.0
