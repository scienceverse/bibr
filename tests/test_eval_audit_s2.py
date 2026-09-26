"""Regression tests for audit action S2 (batch/evaluation findings, 2026-09-24 audit).

Covers batch-eval-16 (dead metric code removed), batch-eval-18 (shared
reference matching) and batch-eval-19 (rapidfuzz LCS). batch-eval-12 is
tested in tests/batch/test_report.py; batch-eval-20 in
TestDirtyTreeProvenance below.
"""

from __future__ import annotations

import importlib.util


class TestDeadMetricCodeRemoved:
    """batch-eval-16: unreferenced helpers are gone; used ones are untouched."""

    def test_unreferenced_validation_metric_helpers_are_removed(self):
        import evaluation.validation_metrics as vm

        for name in (
            "keywords_f1",  # opposite-contract duplicate of evaluate.keywords_f1
            "keywords_fuzzy_f1",
            "authors_count_ratio",
            "authors_order_score",
        ):
            assert not hasattr(vm, name), f"dead helper still present: {name}"

    def test_performance_recorder_module_is_removed(self):
        try:
            spec = importlib.util.find_spec("bibr.metrics.performance")
        except ModuleNotFoundError:
            spec = None  # parent package is gone too: fully removed
        assert spec is None

    def test_title_soft_containment_no_longer_promises_an_aggregate(self):
        from evaluation.validation_metrics import title_soft_containment

        assert "title_soft_containment_n" not in (title_soft_containment.__doc__ or "")

    def test_used_validation_metrics_survive(self):
        # Guard against over-deletion: everything the harness imports stays.
        import evaluation.validation_metrics as vm

        for name in (
            "abstract_ned",
            "abstract_rouge_l",
            "authors_family_f1",
            "authors_fullname_f1",
            "doi_match",
            "first_author_match",
            "normalize_doi",
            "paper_passes_floors",
            "pass_rate",
            "ref_field_scores",
            "ref_matching_f1",
            "references_count_ratio",
            "title_soft_containment",
            "title_soft_match",
            "_get_ref_container",
            "_get_ref_pages",
            "_get_ref_title",
            "_get_ref_volume",
            "_get_ref_year",
            "_greedy_match_pairs",
            "_ref_similarity",
            "_ref_surname_tokens",
        ):
            assert hasattr(vm, name), f"used metric helper missing: {name}"

    def test_harness_keywords_f1_is_evaluate_version(self):
        # The surviving keywords_f1 is the harness one (None for empty gold),
        # not the removed validation_metrics duplicate (1.0 for both-empty).
        from evaluation.evaluate import keywords_f1

        assert keywords_f1([], []) is None


def _synthetic_refs():
    return (
        [
            {
                "title": "Deep learning models",
                "year": "2020",
                "doi": "10.5555/a.1",
                "authors": "Smith J",
                "container": "Nature",
                "volume": "12",
                "first_page": "10",
            },
            {"title": "Protein folding advances", "year": "2021", "authors": "Doe A"},
            {"unstructured": "a citation string with no fields at all"},
        ],
        [
            {
                "title": "Deep learning models",
                "year": "2020",
                "doi": "10.5555/A.1",
                "authors": "Smith J",
                "container": "Nature",
                "volume": "12",
                "first_page": "10",
                "last_page": "20",
            },
            {"title": "Protein folding advances", "year": "2021"},
            {"title": "An unrelated paper", "year": "2019", "doi": "10.5555/b.9"},
        ],
    )


class TestSharedReferenceMatching:
    """batch-eval-18: one matching pass feeds all three ref metrics."""

    def test_shared_match_equals_separate_calls(self):
        from evaluation.evaluate import ref_field_counts
        from evaluation.validation_metrics import (
            match_references,
            ref_field_scores,
            ref_matching_f1,
        )

        ext, gold = _synthetic_refs()
        match = match_references(ext, gold)
        assert ref_matching_f1(ext, gold, match=match) == ref_matching_f1(ext, gold)
        assert ref_field_scores(ext, gold, match=match) == ref_field_scores(ext, gold)
        assert ref_field_counts(ext, gold, match=match) == ref_field_counts(ext, gold)

    def test_match_pairs_drive_all_three_metrics(self):
        # The failure scenario was denominators disagreeing with accuracies
        # after a one-sided predicate edit. There is only one predicate table
        # now, and the per-pair loop gates on its output
        # (``match.gold_has_field``) instead of re-deriving presence inline:
        # counts and denominators come from the same match object.
        from evaluation.evaluate import ref_field_counts
        from evaluation.validation_metrics import match_references, ref_field_scores

        ext, gold = _synthetic_refs()
        match = match_references(ext, gold)
        counts = ref_field_counts(ext, gold, match=match)
        scores = ref_field_scores(ext, gold, match=match)
        assert counts["gold_refs"] == len(gold) == match.n_gold
        assert len(match.pairs) <= min(len(ext), len(gold))
        for field in ("title", "year", "doi", "author", "journal", "volume", "pages"):
            presence = match.gold_has_field[field]
            assert counts[f"{field}_gold"] == sum(presence)
            assert counts[f"{field}_matched"] == sum(presence[gi] for gi, _ in match.pairs)
        # Every accuracy with a denominator divides by matched pairs carrying
        # the field — the same matched count pooled above.
        assert scores["ref_title_acc"] is not None
        assert counts["title_matched"] >= 1

    def test_field_scores_pin_exact_values(self):
        # Hard values on the synthetic pairs: any predicate edit — in the
        # table or in the loop — moves a score or a count below.
        from evaluation.evaluate import ref_field_counts
        from evaluation.validation_metrics import match_references, ref_field_scores

        ext, gold = _synthetic_refs()
        match = match_references(ext, gold)
        assert match.pairs == ((0, 0), (1, 1))
        assert ref_field_scores(ext, gold, match=match) == {
            "ref_title_acc": 1.0,
            "ref_year_acc": 1.0,
            "ref_doi_recall": 0.5,
            "ref_author_acc": 1.0,
            "ref_journal_acc": 1.0,
            "ref_volume_acc": 1.0,
            # The gold carries a last_page the extraction lacks.
            "ref_pages_acc": 0.0,
        }
        counts = ref_field_counts(ext, gold, match=match)
        assert counts["title_matched"] == 2
        assert counts["year_matched"] == 2
        assert counts["doi_matched"] == 1
        assert counts["author_matched"] == 1
        assert counts["journal_matched"] == 1
        assert counts["volume_matched"] == 1
        assert counts["pages_matched"] == 1

    def test_field_scores_gate_on_the_shared_presence_table(self, monkeypatch):
        # A predicate edit must move the accuracy denominators together with
        # the pooled counts: the per-pair loop gates on
        # ``match.gold_has_field`` instead of re-deriving presence inline.
        # Invert the DOI predicate (present exactly where the gold has no
        # DOI): the matched G1 then counts as carrying a DOI per the table,
        # but its empty value can never compare equal, so ref_doi_recall is
        # 0.0. Inline ``if gt_doi:`` gating would still score the matched
        # G0 DOI and report 1.0.
        import evaluation.validation_metrics as vm
        from evaluation.evaluate import ref_field_counts

        ext, gold = _synthetic_refs()
        monkeypatch.setitem(
            vm._REF_GOLD_FIELD_GETTERS,
            "doi",
            lambda ref: (
                "" if vm.normalize_doi(ref.get("doi") or ref.get("DOI") or "") else "10.0/inverted"
            ),
        )
        match = vm.match_references(ext, gold)
        assert match.pairs == ((0, 0), (1, 1))
        counts = ref_field_counts(ext, gold, match=match)
        assert counts["doi_gold"] == 1
        assert counts["doi_matched"] == 1
        assert vm.ref_field_scores(ext, gold, match=match)["ref_doi_recall"] == 0.0

    def test_empty_sides_match_nothing(self):
        from evaluation.evaluate import ref_field_counts
        from evaluation.validation_metrics import match_references, ref_matching_f1

        assert match_references([], []).pairs == ()
        assert ref_matching_f1([], []) == 1.0
        assert ref_matching_f1([], [{"title": "x"}]) == 0.0
        counts = ref_field_counts([{"title": "x"}], [])
        assert counts["gold_refs"] == 0
        assert all(v == 0 for k, v in counts.items() if k != "gold_refs")


class TestRapidfuzzLcs:
    """batch-eval-19: bit-parallel LCS returns the same lengths."""

    def test_lcs_matches_reference_values(self):
        import random

        from rapidfuzz.distance import LCSseq

        from evaluation.validation_metrics import _lcs_length

        random.seed(20260926)
        vocab = ["the", "model", "data", "α", "x", "of", "and", "neural"]
        cases = [
            ([], []),
            (["a"], []),
            ([], ["b"]),
            (["a"], ["a"]),
            (["a", "a", "a"], ["a", "a"]),
            (["a", "b", "a", "b"], ["b", "a", "b", "a"]),
        ]
        for _ in range(200):
            n, m = random.randint(0, 25), random.randint(0, 25)  # noqa: S311
            cases.append(
                (
                    [random.choice(vocab) for _ in range(n)],  # noqa: S311
                    [random.choice(vocab) for _ in range(m)],  # noqa: S311
                )
            )
        for x, y in cases:
            assert _lcs_length(x, y) == int(LCSseq.similarity(x, y)) == _dp_lcs(x, y)

    def test_abstract_rouge_l_unchanged_by_speedup(self):
        # Spot values pin the metric while the engine underneath changes.
        from evaluation.validation_metrics import abstract_rouge_l

        assert abstract_rouge_l("the cat sat", "the cat sat") == 1.0
        assert abstract_rouge_l("", "nonempty gold") == 0.0
        assert abstract_rouge_l("x", "") is None
        assert abstract_rouge_l("the cat sat on the mat", "the cat sat") == 2 / 3


def _dp_lcs(x, y):
    """Textbook LCS table: independent oracle for the test above."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _init_git_repo(path, monkeypatch):
    """A deterministic git checkout: no reliance on the surrounding repo state."""
    import subprocess

    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("HOME", str(path))
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }

    def git(*args):
        import os

        full_env = dict(os.environ, **env)
        subprocess.run(["git", *args], cwd=path, env=full_env, check=True, capture_output=True)  # noqa: S603, S607

    git("init", "-q")
    (path / "tracked.txt").write_text("v1")
    git("add", "tracked.txt")
    git("commit", "-qm", "init")
    return git


class TestDirtyTreeProvenance:
    """batch-eval-20: artifacts tell a patched worktree from a clean checkout."""

    def test_provenance_helpers_exist(self):
        from evaluation import evaluate

        assert callable(evaluate.bibr_dirty)
        assert callable(evaluate.eval_code_sha256)

    def test_bibr_dirty_false_on_clean_checkout(self, tmp_path, monkeypatch):
        from evaluation.evaluate import bibr_dirty

        _init_git_repo(tmp_path, monkeypatch)
        assert bibr_dirty(tmp_path) is False

    def test_bibr_dirty_true_with_uncommitted_change(self, tmp_path, monkeypatch):
        from evaluation.evaluate import bibr_dirty

        _init_git_repo(tmp_path, monkeypatch)
        (tmp_path / "tracked.txt").write_text("v2-uncommitted")
        assert bibr_dirty(tmp_path) is True

    def test_bibr_dirty_ignores_untracked_files(self, tmp_path, monkeypatch):
        # Untracked files cannot change scoring; only tracked status matters.
        from evaluation.evaluate import bibr_dirty

        _init_git_repo(tmp_path, monkeypatch)
        (tmp_path / "notes-scratch.txt").write_text("not scoring code")
        assert bibr_dirty(tmp_path) is False

    def test_bibr_dirty_none_outside_a_repo(self, tmp_path):
        from evaluation.evaluate import bibr_dirty

        assert bibr_dirty(tmp_path) is None

    def test_eval_code_sha256_tracks_metric_sources(self, tmp_path):
        import hashlib

        from evaluation.evaluate import eval_code_sha256

        evo = tmp_path / "evaluation"
        evo.mkdir()
        (evo / "a.py").write_text("X = 1\n")
        (evo / "b.py").write_text("Y = 2\n")
        first = eval_code_sha256(tmp_path)
        assert first is not None
        (evo / "a.py").write_text("X = 1  # uncommitted metric tweak\n")
        assert eval_code_sha256(tmp_path) != first
        # Same inputs hash the same way twice.
        assert eval_code_sha256(tmp_path) == eval_code_sha256(tmp_path)
        assert len(first) == len(hashlib.sha256().hexdigest())

    def test_eval_code_sha256_none_without_sources(self, tmp_path):
        from evaluation.evaluate import eval_code_sha256

        assert eval_code_sha256(tmp_path) is None

    def test_save_results_records_dirty_provenance(self, tmp_path):
        import json as _json

        import pandas as pd

        from evaluation.evaluate import save_results

        out = tmp_path / "eval.json"
        save_results(
            pd.DataFrame([{"doi": "10.1/a", "file_name": "a.pdf", "title_soft": 1.0}]),
            out,
        )
        data = _json.loads(out.read_text())
        assert "bibr_dirty" in data
        assert "eval_code_sha256" in data
        # Provenance-only: no metric block changes shape.
        assert data["metrics"]["title_soft"]["mean"] == 1.0
