"""Tests for evaluation harness — metrics and runner."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from evaluation.validation_metrics import (
    FLOORS,
    abstract_ned,
    abstract_rouge_l,
    authors_family_f1,
    authors_fullname_f1,
    doi_match,
    first_author_match,
    normalize_doi,
    paper_passes_floors,
    pass_rate,
    ref_field_scores,
    ref_matching_f1,
    references_count_ratio,
    title_exact_match,
    title_similarity,
    title_soft_match,
)

# ============================================================================
# Metric Tests
# ============================================================================


class TestTitleMetrics:
    def test_exact_match_identical(self):
        assert title_exact_match("Hello World", "Hello World") == 1.0

    def test_exact_match_different(self):
        assert title_exact_match("Hello World", "Goodbye World") == 0.0

    def test_exact_match_case_insensitive(self):
        assert title_exact_match("Hello World", "hello world") == 1.0

    def test_exact_match_stripped(self):
        assert title_exact_match("  Hello World  ", "Hello World") == 1.0

    def test_similarity_close(self):
        # Levenshtein correctly rates this ~0.85 (one word insertion in a 6-word title)
        score = title_similarity(
            "Seasonal Changes in Garden Plants", "Seasonal Color Changes in Garden Plants"
        )
        assert score > 0.8

    def test_similarity_identical(self):
        assert title_similarity("Hello", "Hello") == 1.0

    def test_similarity_completely_different(self):
        score = title_similarity("abc", "xyz")
        assert score < 0.5

    def test_missing_gold_title_excluded(self):
        """No gold title → nothing to evaluate against → None, not a penalty."""
        assert title_exact_match("Extracted Title", "") is None
        assert title_soft_match("Extracted Title", "") is None
        assert title_similarity("Extracted Title", "") is None

    def test_missing_extracted_title_penalized(self):
        """Gold has a title but extraction is empty → genuine miss → 0.0."""
        assert title_exact_match("", "Gold Title") == 0.0
        assert title_soft_match("", "Gold Title") == 0.0
        assert title_similarity("", "Gold Title") == 0.0


class TestTitleSoftMatch:
    def test_exact_match(self):
        assert title_soft_match("Hello World", "Hello World") == 1.0

    def test_case_insensitive(self):
        assert title_soft_match("Hello World", "hello world") == 1.0

    def test_punctuation_ignored(self):
        assert title_soft_match("Hello, World!", "Hello World") == 1.0

    def test_different(self):
        assert title_soft_match("Hello World", "Goodbye World") == 0.0

    def test_containment_subtitle_drop_rejected(self):
        """A dropped subtitle is a real extraction defect, not a tolerable variant."""
        assert (
            title_soft_match(
                "Planning a Garden: Comparing Plant Growth Across Different Classroom Conditions",
                "Planning a Garden",
            )
            == 0.0
        )
        assert (
            title_soft_match(
                "Planning a Garden",
                "Planning a Garden: Comparing Plant Growth Across Different Classroom Conditions",
            )
            == 0.0
        )

    def test_containment_single_word_rejected(self):
        """Single-word prefixes don't match (risk of false positives)."""
        assert title_soft_match("Brain", "Brain Imaging in Neural Correlates") == 0.0

    def test_containment_short_two_word_title_rejected(self):
        """Short two-word prefixes fall well under the length floor."""
        assert (
            title_soft_match("Growing Plants", "Growing Plants: Comparing Classroom Environments")
            == 0.0
        )
        assert (
            title_soft_match("Looking at Leaves", "Looking at Leaves: Comparing Seasonal Colors")
            == 0.0
        )

    def test_containment_not_word_boundary(self):
        """Prefix must end at a word boundary."""
        assert title_soft_match("The Garde", "The Garden Observation Learning Project") == 0.0

    def test_containment_survives_trailing_punctuation(self):
        """The floor forgives what it is meant to: near-identical strings."""
        assert (
            title_soft_match(
                "The Garden Observation Learning Project: A Study",
                "The Garden Observation Learning Project",
            )
            == 1.0
        )

    def test_containment_scriptio_continua_matches_latin_treatment(self):
        """Spaceless scripts must be scored the same way as space-delimited ones."""
        assert (
            title_soft_match(
                "庭の花を数えてみよう", "庭の花を数えてみよう 花の色と季節の変化について"
            )
            == 0.0
        )

    def test_containment_scriptio_continua_short_prefix_rejected(self):
        """The character floor still gates coincidental short prefixes."""
        assert title_soft_match("庭の", "庭の花を数えてみよう 花の色と季節の変化について") == 0.0

    def test_containment_long_single_latin_word_still_rejected(self):
        """The floor is scoped to spaceless scripts — a long Latin single word
        stays ineligible, since space-delimited text has no excuse."""
        assert (
            title_soft_match("Neuropsychopharmacology", "Neuropsychopharmacology and Behaviour")
            == 0.0
        )


class TestDoiMetrics:
    def test_match_identical(self):
        assert doi_match("10.5555/example.identical", "10.5555/example.identical") == 1.0

    def test_match_case_insensitive(self):
        assert doi_match("10.5555/ABC", "10.5555/abc") == 1.0

    def test_match_different(self):
        assert doi_match("10.5555/123", "10.5555/456") == 0.0

    def test_match_with_url_prefix(self):
        assert doi_match("https://doi.org/10.5555/123", "10.5555/123") == 1.0

    def test_match_both_urls(self):
        assert doi_match("http://dx.doi.org/10.5555/123", "https://doi.org/10.5555/123") == 1.0

    def test_match_with_doi_prefix(self):
        assert doi_match("doi:10.5555/123", "10.5555/123") == 1.0

    def test_asserting_a_doi_against_empty_gold_is_penalized(self):
        """Gold has no printed DOI but bibr asserts one → 0.0, not free."""
        assert doi_match("10.5555/123", "") == 0.0
        assert doi_match("", "") is None

    def test_missing_extracted_doi_penalized(self):
        """Gold has a DOI but extraction doesn't → genuine miss → 0.0."""
        assert doi_match("", "10.5555/123") == 0.0


class TestDoiNormalization:
    def test_normalize_bare(self):
        assert normalize_doi("10.5555/123") == "10.5555/123"

    def test_normalize_https_doi_org(self):
        assert normalize_doi("https://doi.org/10.5555/123") == "10.5555/123"

    def test_normalize_http_dx(self):
        assert normalize_doi("http://dx.doi.org/10.5555/123") == "10.5555/123"

    def test_normalize_doi_prefix(self):
        assert normalize_doi("DOI: 10.5555/123") == "10.5555/123"

    def test_normalize_trailing_punctuation(self):
        assert normalize_doi("10.5555/123.") == "10.5555/123"

    def test_normalize_empty(self):
        assert normalize_doi("") == ""

    def test_normalize_case(self):
        assert normalize_doi("10.5555/ABC") == "10.5555/abc"

    def test_normalize_collapses_repeated_slashes(self):
        assert normalize_doi("10.5555//example.suffix") == "10.5555/example.suffix"


class TestAbstractMetrics:
    def test_rouge_l_identical(self):
        text = "This is a test abstract about memory and encoding."
        assert abstract_rouge_l(text, text) == 1.0

    def test_rouge_l_partial(self):
        ext = "This is a test abstract."
        gt = "This is a test abstract about memory and encoding."
        score = abstract_rouge_l(ext, gt)
        assert 0 < score < 1

    def test_rouge_l_empty_extracted(self):
        assert abstract_rouge_l("", "Some abstract text") == 0.0

    def test_rouge_l_empty_ground_truth_excluded(self):
        """No gold abstract → nothing to evaluate against → None."""
        assert abstract_rouge_l("Some abstract text", "") is None

    def test_rouge_l_both_empty_excluded(self):
        assert abstract_rouge_l("", "") is None

    def test_rouge_l_no_overlap(self):
        assert abstract_rouge_l("cat dog", "xyz abc") == 0.0

    def test_ned_empty_ground_truth_excluded(self):
        assert abstract_ned("Some abstract text", "") is None

    def test_ned_both_empty_excluded(self):
        assert abstract_ned("", "") is None

    def test_ned_empty_extracted_penalized(self):
        assert abstract_ned("", "Some abstract text") == 0.0


class TestAuthorMetrics:
    def test_family_f1_perfect(self):
        authors = [{"family": "Smith", "given": "J"}, {"family": "Jones", "given": "A"}]
        assert authors_family_f1(authors, authors) == 1.0

    def test_family_f1_partial(self):
        ext = [{"family": "Smith", "given": "J"}, {"family": "Brown", "given": "B"}]
        gt = [{"family": "Smith", "given": "J"}, {"family": "Jones", "given": "A"}]
        score = authors_family_f1(ext, gt)
        # Smith matches, Brown/Jones don't. precision=1/2, recall=1/2, F1=0.5
        assert abs(score - 0.5) < 0.001

    def test_family_f1_both_empty(self):
        assert authors_family_f1([], []) == 1.0

    def test_family_f1_one_empty(self):
        assert authors_family_f1([], [{"family": "Smith"}]) == 0.0

    def test_family_f1_case_insensitive(self):
        ext = [{"family": "DE JONG"}]
        gt = [{"family": "de Jong"}]
        assert authors_family_f1(ext, gt) == 1.0


class TestAuthorFullnameMetrics:
    def test_boundary_shift_is_perfect(self):
        gold = [{"given": "Amélia Campos", "family": "de Exemplo"}]
        ext = [{"given": "Amélia", "family": "Campos de Exemplo"}]
        assert authors_fullname_f1(ext, gold) == 1.0

    def test_particle_boundary_is_perfect(self):
        gold = [{"given": "Samir", "family": "Karim ul Exemplar"}]
        ext = [{"given": "Samir Karim ul", "family": "Exemplar"}]
        assert authors_fullname_f1(ext, gold) == 1.0

    def test_dropped_author_penalized(self):
        gold = [{"given": "A", "family": f"Fam{i}"} for i in range(6)]
        ext = gold[:5]
        f1 = authors_fullname_f1(ext, gold)
        assert 0.90 < f1 < 1.0  # 2*(1.0*5/6)/(1+5/6) ≈ 0.909

    def test_family_only_entry_used(self):
        gold = [{"given": "", "family": "Smith"}]
        ext = [{"family": "Smith"}]
        assert authors_fullname_f1(ext, gold) == 1.0

    def test_both_empty(self):
        assert authors_fullname_f1([], []) == 1.0

    def test_one_empty(self):
        assert authors_fullname_f1([], [{"given": "J", "family": "Smith"}]) == 0.0

    def test_diacritic_insensitive(self):
        gold = [{"given": "Lucía Bárbara", "family": "Ejemplo"}]
        ext = [{"given": "Lucia Barbara", "family": "Ejemplo"}]
        assert authors_fullname_f1(ext, gold) == 1.0


class TestFloorsGateOnFullname:
    def test_floors_key_is_fullname(self):
        assert "authors_fullname_f1" in FLOORS
        assert "authors_f1" not in FLOORS
        assert FLOORS["authors_fullname_f1"] == 0.9


class TestFirstAuthorMatch:
    def test_identical(self):
        ext = [{"family": "Smith", "given": "John"}]
        gt = [{"family": "Smith", "given": "John"}]
        assert first_author_match(ext, gt) == 1.0

    def test_case_insensitive(self):
        ext = [{"family": "DE JONG", "given": "A"}]
        gt = [{"family": "de Jong", "given": "A"}]
        assert first_author_match(ext, gt) == 1.0

    def test_both_empty(self):
        assert first_author_match([], []) == 1.0

    def test_one_empty(self):
        assert first_author_match([], [{"family": "Smith"}]) == 0.0

    def test_different(self):
        ext = [{"family": "Smith"}]
        gt = [{"family": "Jones"}]
        score = first_author_match(ext, gt)
        assert score < 0.8

    def test_fuzzy_match(self):
        # Minor typo should still get high score
        ext = [{"family": "Smithe"}]
        gt = [{"family": "Smith"}]
        score = first_author_match(ext, gt)
        assert score > 0.8


class TestRefMatchingF1:
    def test_both_empty(self):
        assert ref_matching_f1([], []) == 1.0

    def test_one_empty(self):
        assert ref_matching_f1([], [{"title": "A paper"}]) == 0.0

    def test_match_by_doi(self):
        ext = [{"doi": "10.5555/123"}, {"doi": "10.5555/456"}]
        gt = [{"DOI": "10.5555/123"}, {"DOI": "10.5555/456"}]
        assert ref_matching_f1(ext, gt) == 1.0

    def test_match_by_doi_normalized(self):
        ext = [{"doi": "https://doi.org/10.5555/123"}]
        gt = [{"DOI": "10.5555/123"}]
        assert ref_matching_f1(ext, gt) == 1.0

    def test_match_by_title(self):
        ext = [{"title": "A Synthetic Study of Classroom Gardens"}]
        gt = [{"article_title": "A Synthetic Study of Classroom Gardens"}]
        assert ref_matching_f1(ext, gt) == 1.0

    def test_match_by_article_title_hyphenated(self):
        """Crossref uses hyphenated 'article-title' keys."""
        ext = [{"title": "A Synthetic Study of Classroom Gardens"}]
        gt = [{"article-title": "A Synthetic Study of Classroom Gardens"}]
        assert ref_matching_f1(ext, gt) == 1.0

    def test_match_by_author_year(self):
        ext = [{"author": "Smith J.", "year": "2020"}]
        gt = [{"author": "Smith, John", "year": "2020"}]
        assert ref_matching_f1(ext, gt) == 1.0

    def test_match_by_unstructured(self):
        """Many Crossref refs only have an 'unstructured' citation string."""
        ext = [{"author": "Smith", "year": "2020", "title": "Cognitive biases in decisions"}]
        gt = [
            {
                "unstructured": "Smith, J. (2020). Cognitive biases in decisions. "
                "Journal of Psychology, 45, 123-130."
            }
        ]
        assert ref_matching_f1(ext, gt) == 1.0

    def test_no_match(self):
        ext = [{"title": "Paper A"}]
        gt = [{"title": "Completely Different Paper"}]
        assert ref_matching_f1(ext, gt) == 0.0

    def test_partial_match(self):
        ext = [
            {"doi": "10.5555/111"},
            {"title": "Unknown Paper"},
        ]
        gt = [
            {"DOI": "10.5555/111"},
            {"article-title": "A Different Paper"},
        ]
        # 1 match: P=1/2, R=1/2, F1=0.5
        assert abs(ref_matching_f1(ext, gt) - 0.5) < 0.01


class TestRefFieldScores:
    def test_sparse_gt_title_excluded(self):
        """When GT has no titles, ref_title_acc is None (no data), not 0.0."""
        ext = [{"doi": "10.5555/123", "title": "A Paper Title"}]
        gt = [{"DOI": "10.5555/123"}]  # DOI-only GT, no title
        scores = ref_field_scores(ext, gt)
        # No GT titles to compare → None (excluded), not a penalty
        assert scores["ref_title_acc"] is None
        # DOI recall should be perfect
        assert scores["ref_doi_recall"] == 1.0

    def test_sparse_gt_year_excluded(self):
        """When GT has no years, ref_year_acc is None (no data)."""
        ext = [{"doi": "10.5555/123", "year": "2020"}]
        gt = [{"DOI": "10.5555/123"}]  # No year in GT
        scores = ref_field_scores(ext, gt)
        assert scores["ref_year_acc"] is None

    def test_gt_without_dois_excluded(self):
        """When no GT ref has a DOI (paper prints none), ref_doi_recall is None."""
        ext = [{"title": "Paper A", "doi": "10.5555/123"}]
        gt = [{"article-title": "Paper A"}]
        scores = ref_field_scores(ext, gt)
        assert scores["ref_doi_recall"] is None
        assert scores["ref_title_acc"] == 1.0

    def test_empty_gold_refs_all_excluded(self):
        """No gold refs at all → nothing to evaluate → all None."""
        scores = ref_field_scores([{"title": "Paper A"}], [])
        assert scores["ref_title_acc"] is None
        assert scores["ref_year_acc"] is None
        assert scores["ref_doi_recall"] is None

    def test_empty_extraction_with_gold_fields_penalized(self):
        """Gold has fields but nothing was extracted → genuine miss → 0.0."""
        gt = [{"DOI": "10.5555/123", "article-title": "Paper A", "year": "2020"}]
        scores = ref_field_scores([], gt)
        assert scores["ref_title_acc"] == 0.0
        assert scores["ref_year_acc"] == 0.0
        assert scores["ref_doi_recall"] == 0.0

    def test_doi_recall_counts_unmatched_gold_refs(self):
        """Recall denominator is ALL gold refs with DOIs, not just matched pairs."""
        ext = [{"doi": "10.5555/111", "title": "Paper A"}]
        gt = [
            {"DOI": "10.5555/111", "article-title": "Paper A"},
            {"DOI": "10.5555/222", "article-title": "A Completely Unrelated Work"},
        ]
        scores = ref_field_scores(ext, gt)
        # One of two gold DOIs recovered → 0.5, even though the second ref went unmatched
        assert scores["ref_doi_recall"] == 0.5

    def test_gt_with_title_correct(self):
        """When GT has titles, correct extraction scores 1.0."""
        ext = [{"doi": "10.5555/123", "title": "A Synthetic Study of Classroom Gardens"}]
        gt = [{"DOI": "10.5555/123", "article-title": "A Synthetic Study of Classroom Gardens"}]
        scores = ref_field_scores(ext, gt)
        assert scores["ref_title_acc"] == 1.0

    def test_gt_with_year_correct(self):
        ext = [{"doi": "10.5555/123", "year": "2020"}]
        gt = [{"DOI": "10.5555/123", "year": "2020"}]
        scores = ref_field_scores(ext, gt)
        assert scores["ref_year_acc"] == 1.0

    def test_mixed_sparse_and_complete_gt(self):
        """Only pairs where GT has the field count toward the denominator."""
        ext = [
            {"doi": "10.5555/1", "title": "Paper A", "year": "2020"},
            {"doi": "10.5555/2", "title": "Paper B", "year": "2021"},
            {"doi": "10.5555/3", "title": "Paper C", "year": "2019"},
        ]
        gt = [
            {"DOI": "10.5555/1", "article-title": "Paper A", "year": "2020"},  # complete
            {"DOI": "10.5555/2"},  # DOI-only
            {"DOI": "10.5555/3"},  # DOI-only
        ]
        scores = ref_field_scores(ext, gt)
        # Only 1 GT ref has a title → 1/1 = 1.0
        assert scores["ref_title_acc"] == 1.0
        # Only 1 GT ref has a year → 1/1 = 1.0
        assert scores["ref_year_acc"] == 1.0
        # All 3 GT refs have DOIs → 3/3 = 1.0
        assert scores["ref_doi_recall"] == 1.0


class TestReferenceMetrics:
    def test_count_ratio_equal(self):
        assert references_count_ratio(41, 41) == 1.0

    def test_count_ratio_half(self):
        assert references_count_ratio(20, 40) == 0.5

    def test_count_ratio_both_zero(self):
        assert references_count_ratio(0, 0) == 1.0

    def test_count_ratio_one_zero(self):
        assert references_count_ratio(0, 10) == 0.0

    def test_count_ratio_over(self):
        # min(50, 40) / max(50, 40) = 40/50 = 0.8
        assert references_count_ratio(50, 40) == 0.8


class TestPaperPassesFloors:
    def _passing_row(self) -> dict:
        return dict(FLOORS)  # every metric exactly at its own floor

    def test_all_at_floor_passes(self):
        assert paper_passes_floors(self._passing_row()) is True

    def test_one_below_floor_fails(self):
        row = self._passing_row()
        row["title_soft"] = FLOORS["title_soft"] - 0.01
        assert paper_passes_floors(row) is False

    def test_none_value_does_not_fail(self):
        row = self._passing_row()
        row["doi_match"] = None  # paper prints no DOI — excluded, not penalized
        assert paper_passes_floors(row) is True

    def test_missing_key_does_not_fail(self):
        row = {"title_soft": 1.0}  # other floor metrics absent entirely
        assert paper_passes_floors(row) is True

    def test_custom_floors(self):
        row = {"foo": 0.5}
        assert paper_passes_floors(row, floors={"foo": 0.6}) is False
        assert paper_passes_floors(row, floors={"foo": 0.4}) is True


class TestPassRate:
    def test_empty_rows_is_none(self):
        assert pass_rate([]) is None

    def test_all_pass(self):
        row = dict.fromkeys(FLOORS, 1.0)
        assert pass_rate([row, row, row]) == 1.0

    def test_all_fail(self):
        row = dict.fromkeys(FLOORS, 0.0)
        row["doi_match"] = None  # doi_match is 0/1/None; keep it excluded, fail via others
        assert pass_rate([row, row]) == 0.0

    def test_mixed(self):
        good = dict.fromkeys(FLOORS, 1.0)
        bad = dict.fromkeys(FLOORS, 0.0)
        assert pass_rate([good, good, bad]) == pytest.approx(2 / 3)


class TestGoldDoiPrinted:
    """The join-key DOI (filename/paper_id) must not leak into doi_match scoring."""

    def _gold_dir(self, tmp_path, info_doi):
        gold = {
            "paper_id": "10.5555/example.join-key",
            "info": {
                "doi": info_doi,
                "file_name": "10.5555_example.join-key.pdf",
                "title": "T",
                "abstract": "A",
            },
            "author": [],
            "bib": [],
        }
        d = tmp_path / "gold"
        d.mkdir()
        (d / "10.5555_example.join-key.json").write_text(__import__("json").dumps(gold))
        return d

    def test_loader_separates_printed_doi_from_join_key(self, tmp_path):
        from evaluation.evaluate import load_ground_truth_gold

        df = load_ground_truth_gold([self._gold_dir(tmp_path, None)])
        row = df.iloc[0]
        # join key keeps the paper_id fallback…
        assert row["doi"] == "10.5555/example.join-key"
        # …but the printed DOI is empty: the page shows none
        assert row["doi_printed"] == ""

    def test_loader_printed_doi_when_present(self, tmp_path):
        from evaluation.evaluate import load_ground_truth_gold

        df = load_ground_truth_gold([self._gold_dir(tmp_path, "10.5555/example.join-key")])
        assert df.iloc[0]["doi_printed"] == "10.5555/example.join-key"

    def test_score_paper_uses_printed_doi(self):
        from evaluation.evaluate import score_paper

        gt = {"title": "T", "doi": "10.5555/x", "doi_printed": "", "abstract": "A", "authors": []}
        ext = {"title": "T", "doi": "", "abstract": "A", "authors": []}
        assert score_paper(ext, gt)["doi_match"] is None

    def test_score_paper_falls_back_to_doi_without_printed_column(self):
        """Legacy Crossref GT has no doi_printed — keep scoring against doi."""
        from evaluation.evaluate import score_paper

        gt = {"title": "T", "doi": "10.5555/x", "abstract": "A", "authors": []}
        ext = {"title": "T", "doi": "10.5555/x", "abstract": "A", "authors": []}
        assert score_paper(ext, gt)["doi_match"] == 1.0


class TestSaveResultsNoneHandling:
    """None metric values (gold lacks the field) must be excluded, not averaged as 0."""

    def _results_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"doi": "10.1/a", "file_name": "a.pdf", "doi_match": 1.0, "title_soft": 1.0},
                {"doi": "10.1/b", "file_name": "b.pdf", "doi_match": None, "title_soft": 0.0},
            ]
        )

    def test_mean_excludes_none(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._results_df(), out)
        data = _json.loads(out.read_text())
        assert data["metrics"]["doi_match"]["mean"] == 1.0
        assert data["metrics"]["title_soft"]["mean"] == 0.5

    def test_metric_reports_evaluated_count(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._results_df(), out)
        data = _json.loads(out.read_text())
        assert data["metrics"]["doi_match"]["n"] == 1
        assert data["metrics"]["title_soft"]["n"] == 2

    def test_per_paper_serializes_none_as_null(self, tmp_path):
        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._results_df(), out)
        raw = out.read_text()
        assert "NaN" not in raw  # json.dump writes bare NaN for np.nan — invalid JSON
        import json as _json

        record = _json.loads(raw)["per_paper"][1]
        assert record["doi_match"] is None


# ============================================================================
# Runner Tests
# ============================================================================


# Mock dataclasses matching bibr's structure for testing
@dataclass
class _MockAuthor:
    family: str
    given: str
    affiliation: str = ""
    email: str = ""
    orcid: str = ""
    corresponding: bool = False


@dataclass
class _MockReference:
    bib_id: int


@dataclass
class _MockMetadata:
    doi: str = ""
    title: str = ""
    keywords: list[str] = field(default_factory=list)
    authors: list[_MockAuthor] = field(default_factory=list)
    references: list[_MockReference] = field(default_factory=list)


@dataclass
class _MockSentence:
    text: str
    section_id: int


@dataclass
class _MockSection:
    section_id: int
    section_type: str | None


@dataclass
class _MockContents:
    sections: list[_MockSection] = field(default_factory=list)
    sentences: list[_MockSentence] = field(default_factory=list)


@dataclass
class _MockPaper:
    metadata: _MockMetadata | None = None
    contents: _MockContents | None = None


class TestExtractComparable:
    def test_extract_basic(self):
        from bibr.paper_contents import CanonicalSection
        from evaluation.evaluate import extract_comparable

        paper = _MockPaper(
            metadata=_MockMetadata(
                doi="10.5555/123",
                title="Test Paper",
                keywords=["memory", "encoding"],
                authors=[_MockAuthor("Smith", "John"), _MockAuthor("Jones", "Alice")],
                references=[_MockReference(1), _MockReference(2)],
            ),
            contents=_MockContents(
                sections=[
                    _MockSection(section_id=1, section_type=CanonicalSection.ABSTRACT),
                    _MockSection(section_id=2, section_type=CanonicalSection.INTRODUCTION),
                ],
                sentences=[
                    _MockSentence("First abstract sentence.", section_id=1),
                    _MockSentence("Second abstract sentence.", section_id=1),
                    _MockSentence("Intro sentence.", section_id=2),
                ],
            ),
        )

        result = extract_comparable(paper)
        assert result["title"] == "Test Paper"
        assert result["doi"] == "10.5555/123"
        assert len(result["authors"]) == 2
        assert result["authors"][0]["family"] == "Smith"
        assert result["abstract"] == "First abstract sentence. Second abstract sentence."
        assert result["reference_count"] == 2

    def test_extract_no_abstract(self):
        from evaluation.evaluate import extract_comparable

        paper = _MockPaper(
            metadata=_MockMetadata(title="No Abstract Paper"),
            contents=_MockContents(),
        )
        result = extract_comparable(paper)
        assert result["abstract"] == ""


class TestScorePaper:
    def test_score_paper_perfect(self):
        from evaluation.evaluate import score_paper

        data = {
            "title": "Test Paper",
            "doi": "10.5555/123",
            "authors": [{"family": "Smith", "given": "J"}],
            "abstract": "This is the abstract.",
            "reference_count": 10,
            "references": [{"doi": "10.5555/456", "title": "Ref One"}],
        }
        scores = score_paper(data, data)

        assert scores["title_soft"] == 1.0
        assert scores["doi_match"] == 1.0
        assert scores["abstract_rouge_l"] == 1.0
        assert scores["authors_f1"] == 1.0
        assert scores["first_author"] == 1.0
        assert scores["ref_count_ratio"] == 1.0
        assert scores["ref_matching_f1"] == 1.0

    def test_score_paper_all_keys_present(self):
        from evaluation.evaluate import METRIC_COLS, score_paper

        ext = {
            "title": "A",
            "doi": "B",
            "authors": [],
            "abstract": "",
            "reference_count": 0,
        }
        gt = {
            "title": "C",
            "doi": "D",
            "authors": [],
            "abstract": "",
            "reference_count": 0,
        }
        scores = score_paper(ext, gt)

        for col in METRIC_COLS:
            assert col in scores, f"Missing metric: {col}"
            # None = gold lacks the field (e.g. empty abstract here) → excluded
            assert scores[col] is None or isinstance(scores[col], float), (
                f"Metric {col} is not float or None"
            )
            if scores[col] is not None:
                assert 0 <= scores[col] <= 1, f"Metric {col} out of range: {scores[col]}"


class TestFrontMatterAbstentions:
    """A front-matter abstention is an explicit refusal.

    When no record can be selected safely, a blocking VAL_METADATA_MULTI_ITEM issue and empty metadata must be represented separately from an incorrect asserted answer."""

    @staticmethod
    def _export(*, blocking_code: str | None = None, title: str = "T") -> dict:
        data: dict = {
            "info": {"title": title, "doi": "10.1/x", "file_name": "p.pdf"},
            "author": [{"author_id": 1, "given": "Jo", "family": "Smith"}],
        }
        if blocking_code:
            data["validation"] = {"issues": [{"code": blocking_code, "blocking": True, "count": 1}]}
        return data

    def test_multi_item_marks_abstained(self):
        from evaluation.evaluate import extract_comparable_from_json

        row = extract_comparable_from_json(self._export(blocking_code="VAL_METADATA_MULTI_ITEM"))
        assert row["abstained"] is True

    def test_clean_export_not_abstained(self):
        from evaluation.evaluate import extract_comparable_from_json

        assert extract_comparable_from_json(self._export())["abstained"] is False

    def test_other_blocking_code_is_not_an_abstention(self):
        """Only front-matter abstention codes exempt front-matter metrics — a
        blocking DOI-identity error is a failure and must still be scored."""
        from evaluation.evaluate import extract_comparable_from_json

        row = extract_comparable_from_json(self._export(blocking_code="VAL_DOI_MISMATCH"))
        assert row["abstained"] is False

    def test_non_blocking_multi_item_is_not_an_abstention(self):
        from evaluation.evaluate import extract_comparable_from_json

        data = self._export()
        data["validation"] = {"issues": [{"code": "VAL_METADATA_MULTI_ITEM", "blocking": False}]}
        assert extract_comparable_from_json(data)["abstained"] is False

    def test_suppressed_metrics_nulled_but_doi_and_refs_kept(self):
        """info.doi survives abstention via extract/doi_identity, and references
        are extracted independently of front matter — both stay scored."""
        import pandas as pd

        from evaluation.evaluate import _apply_abstentions

        df = _apply_abstentions(
            pd.DataFrame(
                [
                    {
                        "file_name": "abstained.pdf",
                        "abstained": True,
                        "title_soft": 0.0,
                        "authors_f1": 0.0,
                        "authors_fullname_f1": 0.0,
                        "doi_match": 1.0,
                        "ref_matching_f1": 0.95,
                    },
                    {
                        "file_name": "scored.pdf",
                        "abstained": False,
                        "title_soft": 0.0,
                        "authors_f1": 0.0,
                        "authors_fullname_f1": 0.0,
                        "doi_match": 1.0,
                        "ref_matching_f1": 0.95,
                    },
                ]
            )
        )
        abstained = df[df["file_name"] == "abstained.pdf"].iloc[0]
        assert pd.isna(abstained["title_soft"])
        assert pd.isna(abstained["authors_f1"])
        assert pd.isna(abstained["authors_fullname_f1"])
        assert abstained["doi_match"] == 1.0
        assert abstained["ref_matching_f1"] == 0.95

        # A real failure on an identical row is untouched.
        scored = df[df["file_name"] == "scored.pdf"].iloc[0]
        assert scored["title_soft"] == 0.0

    @staticmethod
    def _rows_with_one_abstention() -> list[dict]:
        """One abstaining paper, one passing paper, one failing paper.

        The abstaining row is what _apply_abstentions leaves behind: title_soft
        and authors_fullname_f1 nulled, doi_match and ref_matching_f1 surviving
        and good.
        """
        return [
            {"abstained": True, "doi_match": 1.0, "ref_matching_f1": 0.95},
            {
                "abstained": False,
                "title_soft": 1.0,
                "doi_match": 1.0,
                "authors_fullname_f1": 1.0,
                "ref_matching_f1": 0.95,
            },
            {
                "abstained": False,
                "title_soft": 0.0,
                "doi_match": 1.0,
                "authors_fullname_f1": 1.0,
                "ref_matching_f1": 0.95,
            },
        ]

    def test_raw_pass_rate_would_pass_an_abstention_vacuously(self):
        """Why neither pre-v4 rule worked, stated as an executable fact.

        paper_passes_floors ignores None metrics, so an abstaining row is judged
        on its two SURVIVING floors (doi_match, ref_matching_f1) and passes
        without asserting a title or an author. Feeding the abstention-nulled
        rows straight to pass_rate() therefore reports 2/3.
        """
        from evaluation.validation_metrics import paper_passes_floors, pass_rate

        rows = self._rows_with_one_abstention()
        assert paper_passes_floors(rows[0]) is True
        assert pass_rate(rows) == pytest.approx(2 / 3)

    def test_abstained_scored_as_failure_lowers_pass_rate(self):
        """v4: the abstention stays in the denominator and counts as a FAIL.

        pass_rate must therefore be strictly below pass_rate_excl_abstained
        whenever anything abstains — the survivors-only figure can only ever be
        the more flattering of the two.
        """
        from evaluation.evaluate import _pass_rate_full_cohort, _split_abstained
        from evaluation.validation_metrics import pass_rate

        rows = self._rows_with_one_abstention()
        scored, abstained = _split_abstained(rows)
        assert len(abstained) == 1
        assert len(scored) == 2
        # pre-v4 headline: 1 of 2 survivors passes.
        assert pass_rate(scored) == 0.5
        # v4 headline: 1 of 3 attempted papers passes.
        assert _pass_rate_full_cohort(rows) == pytest.approx(1 / 3)
        assert _pass_rate_full_cohort(rows) < pass_rate(scored)

    def test_save_results_pass_rate_is_below_excl_abstained(self, tmp_path):
        """The same inequality at the artifact surface, on both emitted keys."""
        import json

        import pandas as pd

        from evaluation.evaluate import save_results

        rows = self._rows_with_one_abstention()
        for i, row in enumerate(rows):
            row["file_name"] = f"{i}.pdf"
        out = tmp_path / "eval.json"
        save_results(pd.DataFrame(rows), out)
        data = json.loads(out.read_text())
        assert data["pass_rate"] == pytest.approx(1 / 3)
        assert data["pass_rate_excl_abstained"] == 0.5
        assert data["pass_rate"] < data["pass_rate_excl_abstained"]

    def test_save_results_reports_abstention_rate(self, tmp_path):
        """The exclusion is only safe if abstaining is visible and gateable."""
        import json

        import pandas as pd

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(
            pd.DataFrame(
                [
                    {"file_name": "a.pdf", "abstained": True, "title_soft": None},
                    {"file_name": "b.pdf", "abstained": False, "title_soft": 1.0},
                    {"file_name": "c.pdf", "abstained": False, "title_soft": 1.0},
                    {"file_name": "d.pdf", "abstained": False, "title_soft": 1.0},
                ]
            ),
            out,
        )
        data = json.loads(out.read_text())
        assert data["papers_evaluated"] == 4
        assert data["papers_scored"] == 3
        assert data["abstained"] == 1
        assert data["abstention_rate"] == 0.25
        assert data["abstained_ids"] == ["a.pdf"]


class TestExtractComparableFromJson:
    def test_abstract_prefers_info_field(self):
        """info.abstract is the judge-corrected authority; the section join
        picks up front-matter noise misclassified into the abstract section."""
        from evaluation.evaluate import extract_comparable_from_json

        data = {
            "info": {"title": "T", "doi": "10.1/x", "abstract": "Clean abstract."},
            "text": [
                {"text_id": 1, "text": "https://doi.org/10.1/x reuse guidelines", "section_id": 1},
                {"text_id": 2, "text": "Clean abstract.", "section_id": 1},
            ],
            "section": [{"section_id": 1, "section_name": "Abstract", "section_type": "abstract"}],
        }
        assert extract_comparable_from_json(data)["abstract"] == "Clean abstract."

    @staticmethod
    def _section_only_abstract() -> dict:
        """Export with ABSTRACT section text but no ``/info/abstract``."""
        return {
            "info": {"title": "T", "doi": "10.1/x"},
            "text": [{"text_id": 1, "text": "Joined abstract sentence.", "section_id": 1}],
            "section": [{"section_id": 1, "section_name": "Abstract", "section_type": "abstract"}],
        }

    def test_abstract_section_join_is_gold_only(self):
        """The section-text fallback fires for GOLD and only for gold (D2).

        Gold stores the abstract exclusively as ABSTRACT section text, so it must
        keep the join. A prediction is scored on `/info/abstract` alone — the
        field a consumer actually reads — so an export that publishes none scores
        an empty abstract instead of being silently repaired by the evaluator.
        """
        from evaluation.evaluate import extract_comparable_from_json

        gold = extract_comparable_from_json(self._section_only_abstract(), is_gold=True)
        assert gold["abstract"] == "Joined abstract sentence."

        prediction = extract_comparable_from_json(self._section_only_abstract())
        assert prediction["abstract"] == ""

    def test_gold_keeps_section_join_while_prediction_scores_zero(self):
        """The asymmetry has to survive end to end, not just in the extractor.

        Gold recovers its abstract from section text; the prediction that
        publishes no `/info/abstract` is measured as empty. Before D2 both sides
        joined sections and this pair scored a free 1.0.
        """
        from evaluation.evaluate import extract_comparable_from_json
        from evaluation.validation_metrics import abstract_rouge_l

        gold = extract_comparable_from_json(self._section_only_abstract(), is_gold=True)
        prediction = extract_comparable_from_json(self._section_only_abstract())
        assert abstract_rouge_l(prediction["abstract"], gold["abstract"]) == 0.0
        # ...and an export that DOES publish the abstract still scores 1.0.
        published = self._section_only_abstract()
        published["info"]["abstract"] = "Joined abstract sentence."
        good = extract_comparable_from_json(published)
        assert abstract_rouge_l(good["abstract"], gold["abstract"]) == 1.0

    def test_basic(self):
        from evaluation.evaluate import extract_comparable_from_json

        data = {
            "paper_id": "test_paper.pdf",
            "info": {
                "title": "Test Paper",
                "doi": "10.5555/123",
                "keywords": ["memory"],
            },
            "author": [
                {"author_id": 1, "given": "John", "family": "Smith"},
            ],
            "text": [
                {"text_id": 1, "text": "Abstract sentence one.", "section_id": 1},
                {"text_id": 2, "text": "Intro sentence.", "section_id": 2},
            ],
            "section": [
                {"section_id": 1, "section_name": "Abstract", "section_type": "abstract"},
                {"section_id": 2, "section_name": "Intro", "section_type": "introduction"},
            ],
            "bib": [
                {
                    "bib_id": 1,
                    "title": "Reference Paper",
                    "authors": ["Smith J."],
                    "year": 2020,
                    "doi": "10.5555/456",
                },
            ],
        }
        result = extract_comparable_from_json(data)
        assert result["title"] == "Test Paper"
        assert result["doi"] == "10.5555/123"
        # No `/info/abstract` in this export, and it is scored as a PREDICTION —
        # the ABSTRACT section text is deliberately NOT joined in (D2). Gold
        # would get "Abstract sentence one." here; see
        # test_abstract_section_join_is_gold_only.
        assert result["abstract"] == ""
        assert extract_comparable_from_json(data, is_gold=True)["abstract"] == (
            "Abstract sentence one."
        )
        assert len(result["authors"]) == 1
        assert result["authors"][0]["family"] == "Smith"
        assert result["reference_count"] == 1
        assert len(result["references"]) == 1
        assert result["references"][0]["doi"] == "10.5555/456"
        assert result["file_name"] == "test_paper.pdf"

    def test_empty_data(self):
        from evaluation.evaluate import extract_comparable_from_json

        result = extract_comparable_from_json({})
        assert result["title"] == ""
        assert result["doi"] == ""
        assert result["abstract"] == ""
        assert result["reference_count"] == 0


class TestEvaluate:
    def test_evaluate_matches_by_doi(self):
        from bibr.paper_contents import CanonicalSection
        from evaluation.evaluate import evaluate

        paper1 = _MockPaper(
            metadata=_MockMetadata(
                doi="10.5555/001",
                title="Paper One",
                keywords=["a"],
                authors=[_MockAuthor("Smith", "J")],
                references=[_MockReference(1)],
            ),
            contents=_MockContents(
                sections=[_MockSection(1, CanonicalSection.ABSTRACT)],
                sentences=[_MockSentence("Abstract one.", 1)],
            ),
        )
        paper2 = _MockPaper(
            metadata=_MockMetadata(
                doi="10.5555/002",
                title="Paper Two",
                keywords=["b"],
                authors=[_MockAuthor("Jones", "A")],
                references=[_MockReference(1), _MockReference(2)],
            ),
            contents=_MockContents(
                sections=[_MockSection(1, CanonicalSection.ABSTRACT)],
                sentences=[_MockSentence("Abstract two.", 1)],
            ),
        )

        gt_df = pd.DataFrame(
            [
                {
                    "doi": "10.5555/001",
                    "title": "Paper One",
                    "authors": [{"family": "Smith", "given": "J"}],
                    "abstract": "Abstract one.",
                    "reference_count": 1,
                },
                {
                    "doi": "10.5555/002",
                    "title": "Paper Two",
                    "authors": [{"family": "Jones", "given": "A"}],
                    "abstract": "Abstract two.",
                    "reference_count": 2,
                },
            ]
        )

        results = evaluate([paper1, paper2], gt_df)
        assert len(results) == 2
        assert "doi" in results.columns
        assert results["title_soft"].tolist() == [1.0, 1.0]
        assert results["doi_match"].tolist() == [1.0, 1.0]

    def test_evaluate_unmatched_doi_skipped(self):
        from evaluation.evaluate import evaluate

        paper = _MockPaper(
            metadata=_MockMetadata(doi="10.5555/999", title="Unknown"),
            contents=_MockContents(),
        )
        gt_df = pd.DataFrame(
            [
                {
                    "doi": "10.5555/001",
                    "title": "Other",
                    "authors": [],
                    "abstract": "",
                    "reference_count": 0,
                }
            ]
        )
        results = evaluate([paper], gt_df)
        assert len(results) == 0


class TestRefFieldScoresDeepFields:
    """ref_author_acc / ref_journal_acc / ref_volume_acc / ref_pages_acc.

    Same None-semantics as the existing trio: None when NO gold ref carries
    the field; matched pairs missing the field on the extraction side count
    as wrong; gold-has-field-but-zero-matched-pairs scores 0.0.
    """

    GOLD = [
        {
            "title": "Reading development",
            "authors": "Caravolas, M., Lervåg, A., & Hulme, C",
            "year": "2013",
            "doi": "10.5555/0956797612473122",
            "container": "Psychological Science",
            "volume": "24",
            "first_page": "1398",
            "last_page": "1407",
        },
        {
            "title": "A second ref",
            "authors": "Smith, J",
            "year": "2020",
            "doi": "10.1000/second",
            "container": "Tiny Journal",
            "volume": "7",
            "first_page": "10",
            "last_page": "20",
        },
    ]

    def _ext(self, **overrides):
        ext = [dict(self.GOLD[0]), dict(self.GOLD[1])]
        ext[0].update(overrides)
        return ext

    def test_perfect_extraction_scores_one(self):
        scores = ref_field_scores(self._ext(), self.GOLD)
        assert scores["ref_author_acc"] == 1.0
        assert scores["ref_journal_acc"] == 1.0
        assert scores["ref_volume_acc"] == 1.0
        assert scores["ref_pages_acc"] == 1.0

    def test_missing_author_in_extraction_fails_pair(self):
        # Drop Hulme from the matched ref: gold surname set not covered.
        scores = ref_field_scores(self._ext(authors="Caravolas, M., & Lervåg, A"), self.GOLD)
        assert scores["ref_author_acc"] == 0.5

    def test_structured_authors_list_with_full_given_names_still_correct(self):
        # GROBID-style: families list, given names irrelevant — recall on gold
        # surnames, so extra tokens never hurt.
        ext = self._ext(authors=None)
        del ext[0]["authors"]
        ext[0]["authors_list"] = ["Caravolas", "Lervåg", "Hulme"]
        scores = ref_field_scores(ext, self.GOLD)
        assert scores["ref_author_acc"] == 1.0

    def test_diacritics_fuzz_tolerated_in_author(self):
        scores = ref_field_scores(
            self._ext(authors="Caravolas, M., Lervag, A., & Hulme, C"), self.GOLD
        )
        assert scores["ref_author_acc"] == 1.0

    def test_journal_fuzzy_threshold(self):
        scores = ref_field_scores(self._ext(container="Psychological  Science."), self.GOLD)
        assert scores["ref_journal_acc"] == 1.0
        scores = ref_field_scores(self._ext(container="Nature"), self.GOLD)
        assert scores["ref_journal_acc"] == 0.5

    def test_volume_exact(self):
        scores = ref_field_scores(self._ext(volume="25"), self.GOLD)
        assert scores["ref_volume_acc"] == 0.5

    def test_pages_first_and_last_must_match(self):
        scores = ref_field_scores(self._ext(last_page="1408"), self.GOLD)
        assert scores["ref_pages_acc"] == 0.5

    def test_pages_gold_without_last_page_compares_first_only(self):
        gold = [dict(self.GOLD[0])]
        del gold[0]["last_page"]
        ext = [dict(gold[0], last_page="9999")]
        scores = ref_field_scores(ext, gold)
        assert scores["ref_pages_acc"] == 1.0

    @pytest.mark.parametrize(
        ("range_field", "printed_range"),
        [
            ("last_page", "1398-1407"),
            ("last_page", "1398–1407"),
            ("first_page", "1398—1407"),
        ],
    )
    def test_pages_range_in_either_slot_normalizes_printed_endpoints(
        self, range_field, printed_range
    ):
        gold = [dict(self.GOLD[0])]
        ext = [dict(gold[0])]
        ext[0]["first_page"] = ""
        ext[0]["last_page"] = ""
        ext[0][range_field] = printed_range

        scores = ref_field_scores(ext, gold)

        assert scores["ref_pages_acc"] == 1.0

    def test_pages_compact_range_keeps_last_page_verbatim(self):
        gold = [
            {
                "title": "Compact pages",
                "year": "2020",
                "doi": "10.1/compact",
                "first_page": "666",
                "last_page": "74",
            }
        ]
        ext = [dict(gold[0], first_page="", last_page="666–74")]

        assert ref_field_scores(ext, gold)["ref_pages_acc"] == 1.0

    def test_pages_range_still_fails_on_wrong_required_endpoint(self):
        gold = [dict(self.GOLD[0])]
        ext = [dict(gold[0], first_page="", last_page="1398–1408")]

        assert ref_field_scores(ext, gold)["ref_pages_acc"] == 0.0

    def test_none_when_gold_lacks_field_entirely(self):
        gold = [{"title": "Only title", "year": "2020", "doi": "10.1/x"}]
        ext = [dict(gold[0], container="J", volume="1", first_page="1", authors="A, B")]
        scores = ref_field_scores(ext, gold)
        assert scores["ref_author_acc"] is None
        assert scores["ref_journal_acc"] is None
        assert scores["ref_volume_acc"] is None
        assert scores["ref_pages_acc"] is None

    def test_zero_when_gold_has_field_but_extraction_empty(self):
        scores = ref_field_scores([], self.GOLD)
        assert scores["ref_author_acc"] == 0.0
        assert scores["ref_journal_acc"] == 0.0
        assert scores["ref_volume_acc"] == 0.0
        assert scores["ref_pages_acc"] == 0.0

    def test_multiword_surname_list_vs_string_gold_symmetric(self):
        # GROBID extracts the full compound family name as one list entry;
        # gold is the bibr-style string. Tokenization must converge so this
        # correct extraction is scored correct.
        gold = [
            {
                "title": "Czech reading",
                "authors": "Seidlová Málková, G., & Hulme, C",
                "year": "2017",
                "doi": "10.1/czech",
                "container": "Journal of Memory",
            }
        ]
        ext = [dict(gold[0])]
        del ext[0]["authors"]
        ext[0]["authors_list"] = ["Seidlová Málková", "Hulme"]
        scores = ref_field_scores(ext, gold)
        assert scores["ref_author_acc"] == 1.0

    def test_particle_surname_list_vs_string_gold(self):
        gold = [
            {
                "title": "Particles",
                "authors": "van der Berg, J",
                "year": "2019",
                "doi": "10.1/particle",
            }
        ]
        ext = [dict(gold[0])]
        del ext[0]["authors"]
        ext[0]["authors_list"] = ["van der Berg"]
        assert ref_field_scores(ext, gold)["ref_author_acc"] == 1.0

    def test_journal_fallback_key(self):
        gold = [{"title": "T", "year": "2020", "doi": "10.1/j", "container": "Tiny Journal"}]
        ext = [{"title": "T", "year": "2020", "doi": "10.1/j", "journal": "Tiny Journal"}]
        assert ref_field_scores(ext, gold)["ref_journal_acc"] == 1.0


# ============================================================================
# extract_comparable_from_json — deep ref field passthrough
# ============================================================================


class TestExtractComparableRefFields:
    def test_bib_deep_fields_passed_through(self):
        from evaluation.evaluate import extract_comparable_from_json

        data = {
            "info": {"title": "T", "doi": "10.1/x", "file_name": "x.pdf"},
            "author": [],
            "bib": [
                {
                    "bib_id": 0,
                    "title": "Ref",
                    "authors": "Smith, A., & Jones, B",
                    "year": "2020",
                    "doi": "10.1/r0",
                    "container": "Tiny Journal",
                    "volume": "7",
                    "first_page": "10",
                    "last_page": "20",
                }
            ],
        }
        ref = extract_comparable_from_json(data)["references"][0]
        assert ref["authors"] == "Smith, A., & Jones, B"
        assert ref["container"] == "Tiny Journal"
        assert ref["volume"] == "7"
        assert ref["first_page"] == "10"
        assert ref["last_page"] == "20"
        # Existing whole-string key must survive unchanged
        assert ref["author"] == "Smith, A., & Jones, B"

    def test_diagnostic_metric_cols_include_deep_ref_fields(self):
        from evaluation.evaluate import DIAGNOSTIC_METRIC_COLS, PRIMARY_METRIC_COLS

        # Deep per-field ref accuracies are diagnostics (computed + emitted, never
        # gated); ref_title_acc / ref_year_acc remain the primary ref-field signals.
        for col in ("ref_author_acc", "ref_journal_acc", "ref_volume_acc", "ref_pages_acc"):
            assert col in DIAGNOSTIC_METRIC_COLS
            assert col not in PRIMARY_METRIC_COLS
        for col in ("ref_title_acc", "ref_year_acc", "ref_doi_recall"):
            assert col in PRIMARY_METRIC_COLS


# ============================================================================
# Artifact provenance + attrition accounting (2026-07-24 validation audit)
# ============================================================================


class TestArtifactProvenance:
    """H1/H2 — an artifact nobody can tie to a build or a metric definition."""

    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"doi": "10.1/a", "file_name": "a.pdf", "title_soft": 1.0},
                {"doi": "10.1/b", "file_name": "b.pdf", "title_soft": 0.0},
            ]
        )

    def test_metrics_version_is_recorded(self, tmp_path):
        import json as _json

        from evaluation.evaluate import METRICS_VERSION, save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        assert _json.loads(out.read_text())["metrics_version"] == METRICS_VERSION

    def test_metrics_version_is_four(self):
        """Bumped whenever a metric definition moves, so numbers from different
        versions can never be silently compared.

        v4 (campaign triage 2026-08-01) redefined `pass_rate` to count a
        front-matter abstention as a failure and stopped re-deriving a
        PREDICTION's abstract from section text. Both change published numbers,
        so the constant — and the changelog beside it — must move with them.
        """
        from evaluation.evaluate import METRICS_VERSION

        assert METRICS_VERSION == 4

    def test_records_bibr_commit(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        commit = _json.loads(out.read_text())["bibr_commit"]
        assert commit is None or (isinstance(commit, str) and len(commit) == 40)

    def test_records_generated_at_utc(self, tmp_path):
        import json as _json
        from datetime import datetime

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        stamp = _json.loads(out.read_text())["generated_at"]
        assert stamp.endswith("Z")
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))

    def test_records_predictions_dir_and_tree_digest(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        pred = tmp_path / "preds"
        pred.mkdir()
        (pred / "a.json").write_text('{"x": 1}')
        out = tmp_path / "eval.json"
        save_results(self._df(), out, predictions_dir=pred)
        data = _json.loads(out.read_text())
        assert data["predictions_dir"] == str(pred)
        assert len(data["predictions_tree_sha256"]) == 64

    def test_predictions_dir_null_when_not_supplied(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        data = _json.loads(out.read_text())
        assert data["predictions_dir"] is None
        assert data["predictions_tree_sha256"] is None


class TestPredictionsTreeDigest:
    def _dir(self, tmp_path) -> Path:
        pred = tmp_path / "preds"
        pred.mkdir()
        (pred / "a.json").write_text('{"a": 1}')
        (pred / "b.json").write_text('{"b": 2}')
        return pred

    def test_digest_is_stable(self, tmp_path):
        from evaluation.evaluate import predictions_tree_sha256

        pred = self._dir(tmp_path)
        assert predictions_tree_sha256(pred) == predictions_tree_sha256(pred)

    def test_digest_changes_when_content_changes(self, tmp_path):
        from evaluation.evaluate import predictions_tree_sha256

        pred = self._dir(tmp_path)
        before = predictions_tree_sha256(pred)
        (pred / "b.json").write_text('{"b": 3}')
        assert predictions_tree_sha256(pred) != before

    def test_digest_changes_when_a_file_is_dropped(self, tmp_path):
        from evaluation.evaluate import predictions_tree_sha256

        pred = self._dir(tmp_path)
        before = predictions_tree_sha256(pred)
        (pred / "b.json").unlink()
        assert predictions_tree_sha256(pred) != before

    def test_digest_covers_only_the_files_actually_scored(self, tmp_path):
        """validation_report.json is skipped by the loader, so it must not count."""
        from evaluation.evaluate import predictions_tree_sha256

        pred = self._dir(tmp_path)
        before = predictions_tree_sha256(pred)
        (pred / "validation_report.json").write_text('{"noise": true}')
        assert predictions_tree_sha256(pred) == before

    def test_digest_honours_the_ids_subsample(self, tmp_path):
        from evaluation.evaluate import predictions_tree_sha256

        pred = self._dir(tmp_path)
        assert predictions_tree_sha256(pred, ids={"a"}) != predictions_tree_sha256(pred)


class TestExpectedIds:
    def test_named_set_requires_an_explicit_registry(self):
        from evaluation.evaluate import load_expected_ids

        with pytest.raises(ValueError, match="--evaluation-sets"):
            load_expected_ids("unconfigured-set")

    def test_loads_a_json_list_file(self, tmp_path):
        import json as _json

        from evaluation.evaluate import load_expected_ids

        p = tmp_path / "ids.json"
        p.write_text(_json.dumps(["x", "y"]))
        assert load_expected_ids(str(p)) == {"x", "y"}

    def test_loads_a_json_object_with_members(self, tmp_path):
        import json as _json

        from evaluation.evaluate import load_expected_ids

        p = tmp_path / "ids.json"
        p.write_text(_json.dumps({"members": ["x", "y", "z"]}))
        assert load_expected_ids(str(p)) == {"x", "y", "z"}

    def test_resolves_a_named_evaluation_set(self, tmp_path):
        import json as _json

        from evaluation.evaluate import load_expected_ids

        registry = tmp_path / "evaluation_sets.json"
        registry.write_text(
            _json.dumps(
                {
                    "schema": 1,
                    "sets": [
                        {"set_id": "tiny_set", "members": ["p1", "p2"]},
                        {"set_id": "other", "members": ["q1"]},
                    ],
                }
            )
        )
        assert load_expected_ids("tiny_set", registry_path=registry) == {"p1", "p2"}

    def test_unknown_set_name_names_the_available_sets(self, tmp_path):
        import json as _json

        from evaluation.evaluate import load_expected_ids

        registry = tmp_path / "evaluation_sets.json"
        registry.write_text(_json.dumps({"sets": [{"set_id": "tiny_set", "members": ["p1"]}]}))
        with pytest.raises(ValueError, match="tiny_set"):
            load_expected_ids("nope", registry_path=registry)


class TestMissingPapers:
    """H3 — a paper that crashed upstream must not simply vanish."""

    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "doi": "10.1/a",
                    "file_name": "a.pdf",
                    "abstained": False,
                    "title_soft": 1.0,
                    "doi_match": 1.0,
                    "authors_fullname_f1": 1.0,
                    "ref_matching_f1": 1.0,
                    "abstract_rouge_l": 1.0,
                },
                {
                    "doi": "10.1/b",
                    "file_name": "b.pdf",
                    "abstained": False,
                    "title_soft": 1.0,
                    "doi_match": 1.0,
                    "authors_fullname_f1": 1.0,
                    "ref_matching_f1": 1.0,
                    "abstract_rouge_l": 1.0,
                },
            ]
        )

    def test_missing_rows_score_zero_on_every_floor_metric(self):
        from evaluation.evaluate import append_missing_papers

        df = append_missing_papers(self._df(), ["c"])
        row = df[df["missing"]].iloc[0]
        for col in FLOORS:
            assert row[col] == 0.0

    def test_missing_rows_are_none_on_non_floor_metrics(self):
        from evaluation.evaluate import append_missing_papers

        df = append_missing_papers(self._df(), ["c"])
        row = df[df["missing"]].iloc[0]
        assert pd.isna(row["abstract_rouge_l"])

    def test_missing_rows_are_flagged_and_present_rows_are_not(self):
        from evaluation.evaluate import append_missing_papers

        df = append_missing_papers(self._df(), ["c"])
        assert list(df["missing"]) == [False, False, True]

    def test_missing_rows_are_not_abstentions(self):
        from evaluation.evaluate import append_missing_papers

        df = append_missing_papers(self._df(), ["c"])
        assert not bool(df[df["missing"]].iloc[0]["abstained"])

    def test_no_missing_ids_is_a_no_op_on_values(self):
        from evaluation.evaluate import append_missing_papers

        df = append_missing_papers(self._df(), [])
        assert len(df) == 2
        assert list(df["missing"]) == [False, False]

    def test_save_results_counts_attempted_evaluated_and_missing(self, tmp_path):
        import json as _json

        from evaluation.evaluate import append_missing_papers, save_results

        out = tmp_path / "eval.json"
        save_results(append_missing_papers(self._df(), ["c"]), out)
        data = _json.loads(out.read_text())
        assert data["papers_attempted"] == 3
        assert data["papers_evaluated"] == 2
        assert data["papers_missing"] == 1
        assert data["missing_ids"] == ["c"]

    def test_missing_papers_drag_the_gated_mean_down(self, tmp_path):
        import json as _json

        from evaluation.evaluate import append_missing_papers, save_results

        out = tmp_path / "eval.json"
        save_results(append_missing_papers(self._df(), ["c"]), out)
        data = _json.loads(out.read_text())
        # 2 papers at 1.0 + 1 crashed paper at 0.0 — not 1.0 over the survivors.
        assert data["metrics"]["title_soft"]["mean"] == round(2 / 3, 4)
        assert data["pass_rate"] == 2 / 3

    def test_default_run_reports_zero_missing(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        data = _json.loads(out.read_text())
        assert data["papers_attempted"] == 2
        assert data["papers_evaluated"] == 2
        assert data["papers_missing"] == 0
        assert data["missing_ids"] == []
        assert data["metrics"]["title_soft"]["mean"] == 1.0


class TestUnmatchedVisibility:
    """L7 — the silent `continue` on an unmatched prediction must be countable."""

    def test_evaluate_json_exports_reports_unmatched_ids(self, tmp_path):
        import json as _json

        from evaluation.evaluate import evaluate_json_exports

        pred = tmp_path / "preds"
        pred.mkdir()
        (pred / "ghost.json").write_text(
            _json.dumps({"info": {"title": "T", "doi": "10.9/ghost", "file_name": "ghost.pdf"}})
        )
        gt = pd.DataFrame(
            [{"doi": "10.1/a", "file_name": "a.pdf", "title": "T", "authors": [], "references": []}]
        )
        report: dict = {}
        evaluate_json_exports(pred, gt, report=report)
        assert report["unmatched_ids"] == ["ghost"]

    def test_save_results_emits_unmatched_fields(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(
            pd.DataFrame([{"doi": "10.1/a", "file_name": "a.pdf", "title_soft": 1.0}]),
            out,
            unmatched_ids=["ghost"],
        )
        data = _json.loads(out.read_text())
        assert data["papers_unmatched"] == 1
        assert data["unmatched_ids"] == ["ghost"]

    def test_unmatched_defaults_to_zero(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(pd.DataFrame([{"file_name": "a.pdf", "title_soft": 1.0}]), out)
        data = _json.loads(out.read_text())
        assert data["papers_unmatched"] == 0
        assert data["unmatched_ids"] == []


class TestRefCoverageAndMicroAverages:
    """A7/M7/M8 — reference metrics must declare their coverage and pool honestly."""

    def _df(self) -> pd.DataFrame:
        # Paper 1: 10 gold refs, 4 carry pages, 4 matched, 1 correct (0.25).
        # Paper 2: 10 gold refs, 2 carry pages, 2 matched, 2 correct (1.0).
        # Macro mean = 0.625; micro = 3/6 = 0.5; gold coverage = 6/20 = 0.3.
        return pd.DataFrame(
            [
                {
                    "file_name": "a.pdf",
                    "ref_pages_acc": 0.25,
                    "ref_doi_recall": 0.5,
                    "ref_field_counts": {
                        "gold_refs": 10,
                        "pages_gold": 4,
                        "pages_matched": 4,
                        "doi_gold": 8,
                        "doi_matched": 8,
                    },
                },
                {
                    "file_name": "b.pdf",
                    "ref_pages_acc": 1.0,
                    "ref_doi_recall": 1.0,
                    "ref_field_counts": {
                        "gold_refs": 10,
                        "pages_gold": 2,
                        "pages_matched": 2,
                        "doi_gold": 2,
                        "doi_matched": 2,
                    },
                },
            ]
        )

    def test_gold_field_coverage_is_reported_for_pages(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        data = _json.loads(out.read_text())
        assert data["metrics"]["ref_pages_acc"]["gold_field_coverage"] == 0.3

    def test_gold_field_coverage_is_reported_for_ref_doi_recall(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        data = _json.loads(out.read_text())
        assert data["metrics"]["ref_doi_recall"]["gold_field_coverage"] == 0.5

    def test_macro_mean_is_unchanged(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        data = _json.loads(out.read_text())
        assert data["metrics"]["ref_pages_acc"]["mean"] == 0.625

    def test_micro_mean_pools_over_matched_pairs(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        block = _json.loads(out.read_text())["metrics"]["ref_pages_acc"]
        assert block["micro_mean"] == 0.5
        assert block["micro_correct"] == 3
        assert block["micro_denominator"] == 6
        assert block["micro_denominator_kind"] == "matched_pairs"

    def test_ref_doi_recall_micro_pools_over_gold_dois(self, tmp_path):
        """The macro is recall over *gold* DOIs, so the micro must be too."""
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        block = _json.loads(out.read_text())["metrics"]["ref_doi_recall"]
        assert block["micro_correct"] == 6  # 0.5*8 + 1.0*2
        assert block["micro_denominator"] == 10
        assert block["micro_denominator_kind"] == "gold_refs_with_field"
        assert block["micro_mean"] == 0.6

    def test_no_counts_column_leaves_the_annotations_absent(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(pd.DataFrame([{"file_name": "a.pdf", "ref_pages_acc": 1.0}]), out)
        block = _json.loads(out.read_text())["metrics"]["ref_pages_acc"]
        assert "micro_mean" not in block
        assert block["mean"] == 1.0

    def test_ref_field_counts_are_not_summarised_as_a_metric(self, tmp_path):
        import json as _json

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(self._df(), out)
        assert "ref_field_counts" not in _json.loads(out.read_text())["metrics"]


class TestRefFieldCounts:
    """The per-paper denominators micro-averaging pools over."""

    def test_counts_gold_and_matched_per_field(self):
        from evaluation.evaluate import ref_field_counts

        gold = [
            {"title": "Alpha study", "year": "2020", "first_page": "1", "last_page": "9"},
            {"title": "Beta study", "year": "2021"},
        ]
        extracted = [{"title": "Alpha study", "year": "2020", "first_page": "1", "last_page": "9"}]
        counts = ref_field_counts(extracted, gold)
        assert counts["gold_refs"] == 2
        assert counts["pages_gold"] == 1
        assert counts["pages_matched"] == 1
        assert counts["title_gold"] == 2
        assert counts["title_matched"] == 1

    def test_counts_are_zero_without_extraction(self):
        from evaluation.evaluate import ref_field_counts

        counts = ref_field_counts([], [{"title": "Alpha study", "year": "2020"}])
        assert counts["gold_refs"] == 1
        assert counts["title_gold"] == 1
        assert counts["title_matched"] == 0

    def test_score_paper_emits_ref_field_counts(self):
        from evaluation.evaluate import score_paper

        scores = score_paper(
            {"title": "T", "doi": "", "abstract": "", "authors": [], "references": []},
            {"title": "T", "doi": "", "abstract": "", "authors": [], "references": []},
        )
        assert "ref_field_counts" in scores


class TestAbstentionPenalizedMeans:
    """L5 — the gate report must show what abstaining bought."""

    def test_mean_incl_abstained_uses_the_pre_suppression_value(self, tmp_path):
        import json as _json

        from evaluation.evaluate import _apply_abstentions, save_results

        df = _apply_abstentions(
            pd.DataFrame(
                [
                    {"file_name": "a.pdf", "abstained": True, "title_soft": 0.0},
                    {"file_name": "b.pdf", "abstained": False, "title_soft": 1.0},
                    {"file_name": "c.pdf", "abstained": False, "title_soft": 1.0},
                ]
            )
        )
        out = tmp_path / "eval.json"
        save_results(df, out)
        block = _json.loads(out.read_text())["metrics"]["title_soft"]
        assert block["mean"] == 1.0
        assert block["mean_incl_abstained"] == round(2 / 3, 4)

    def test_shadow_columns_do_not_leak_into_per_paper(self, tmp_path):
        import json as _json

        from evaluation.evaluate import _apply_abstentions, save_results

        df = _apply_abstentions(
            pd.DataFrame(
                [
                    {"file_name": "a.pdf", "abstained": True, "title_soft": 0.0},
                    {"file_name": "b.pdf", "abstained": False, "title_soft": 1.0},
                ]
            )
        )
        out = tmp_path / "eval.json"
        save_results(df, out)
        rows = _json.loads(out.read_text())["per_paper"]
        assert all(not k.endswith("__preabstention") for r in rows for k in r)

    def test_unsuppressed_metric_has_no_companion(self, tmp_path):
        import json as _json

        from evaluation.evaluate import _apply_abstentions, save_results

        df = _apply_abstentions(
            pd.DataFrame(
                [
                    {"file_name": "a.pdf", "abstained": True, "ref_matching_f1": 0.2},
                    {"file_name": "b.pdf", "abstained": False, "ref_matching_f1": 1.0},
                ]
            )
        )
        out = tmp_path / "eval.json"
        save_results(df, out)
        block = _json.loads(out.read_text())["metrics"]["ref_matching_f1"]
        assert "mean_incl_abstained" not in block


class TestExplicitEvaluationInputs:
    @staticmethod
    def _run(*args):
        return subprocess.run(  # noqa: S603
            [sys.executable, "-m", "evaluation.evaluate", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_metadata_cli_requires_gold_paths(self, tmp_path):
        result = self._run("--results-dir", tmp_path)
        assert result.returncode == 2
        assert "requires --gold-dirs" in result.stderr

    def test_sections_cli_requires_section_gold_paths(self, tmp_path):
        result = self._run("--results-dir", tmp_path, "--sections")
        assert result.returncode == 2
        assert "requires --section-gold-dirs" in result.stderr

    def test_scores_only_supplied_files_and_accounts_for_missing_predictions(self, tmp_path):
        predictions = tmp_path / "predictions"
        gold = tmp_path / "gold"
        predictions.mkdir()
        gold.mkdir()
        paper = {
            "paper_id": "sample",
            "info": {
                "doi": "10.1234/example",
                "file_name": "sample.pdf",
                "title": "Synthetic Study of Paper Boats",
                "abstract": "We compare folded paper boats in a synthetic example.",
            },
            "author": [{"family": "Example", "given": "Alex"}],
            "bib": [],
        }
        for directory in (predictions, gold):
            (directory / "sample.json").write_text(json.dumps(paper))
        expected = tmp_path / "expected.json"
        expected.write_text(json.dumps(["sample", "missing"]))
        output = tmp_path / "scores.json"
        result = self._run(
            "--results-dir",
            predictions,
            "--gold-dir",
            gold,
            "--expected-ids",
            expected,
            "--output",
            output,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(output.read_text())
        assert report["papers_evaluated"] == 1
        assert report["missing_ids"] == ["missing"]
        assert report["pass_rate_n"] == 2
        assert report["metrics"]["title_soft"]["mean"] == 0.5
