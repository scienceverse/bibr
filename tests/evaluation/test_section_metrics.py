from evaluation import section_metrics as sm


class TestTokenize:
    def test_lowercases_splits_and_drops_empties(self):
        assert sm.tokenize("Hello,  World!") == ["hello", "world"]

    def test_keeps_numbers(self):
        assert sm.tokenize("p < 0.05 (N=42)") == ["p", "0", "05", "n", "42"]

    def test_strips_diacritics(self):
        assert sm.tokenize("Méthode") == ["methode"]

    def test_empty_string_is_empty_list(self):
        assert sm.tokenize("") == []


class TestUnigramRecall:
    def test_full_recall(self):
        assert sm.unigram_recall(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_partial_recall_is_multiset(self):
        # ref has "a" twice; pred has it once -> 2 of 3 ref tokens covered
        assert sm.unigram_recall(["a"], ["a", "a", "b"]) == 1 / 3

    def test_none_when_ref_empty(self):
        assert sm.unigram_recall(["a"], []) is None

    def test_zero_when_pred_empty(self):
        assert sm.unigram_recall([], ["a", "b"]) == 0.0


class TestUnigramPrecision:
    def test_precision_is_multiset(self):
        assert sm.unigram_precision(["a", "a", "x"], ["a"]) == 1 / 3

    def test_none_when_pred_empty(self):
        assert sm.unigram_precision([], ["a"]) is None


class TestShingleRecall:
    def test_full_shingle_recall(self):
        toks = ["the", "quick", "brown", "fox", "jumps", "over"]
        assert sm.shingle_recall(toks, toks) == 1.0

    def test_none_when_ref_below_n(self):
        assert sm.shingle_recall(["a", "b"], ["a", "b", "c", "d"]) is None

    def test_dropped_paragraph_lowers_shingles_more_than_unigrams(self):
        # ref = two paragraphs; pred drops the second contiguous block.
        ref = ("alpha beta gamma delta epsilon " * 2 + "zeta eta theta iota kappa").split()
        pred = ("alpha beta gamma delta epsilon " * 2).split()
        u = sm.unigram_recall(pred, ref)
        s = sm.shingle_recall(pred, ref)
        assert s < u  # contiguous drop hurts shingles more than the bag-of-words

    def test_reading_order_invariance_unigram_vs_shingle(self):
        ref = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        scrambled = ["f", "g", "h", "i", "j", "a", "b", "c", "d", "e"]  # two blocks swapped
        assert sm.unigram_recall(scrambled, ref) == 1.0  # order-invariant
        assert sm.shingle_recall(scrambled, ref) < 1.0  # boundary shingles lost

    def test_zero_when_pred_below_n(self):
        assert sm.shingle_recall(["a", "b"], ["a", "b", "c", "d", "e", "f"]) == 0.0


class TestPresence:
    def test_present_above_tau(self):
        assert sm.section_present(0.8, ref_n_tokens=100) is True

    def test_absent_below_tau(self):
        assert sm.section_present(0.2, ref_n_tokens=100) is False

    def test_none_for_trivially_short_ref(self):
        assert sm.section_present(0.0, ref_n_tokens=5) is None

    def test_none_recall_counts_as_absent(self):
        assert sm.section_present(None, ref_n_tokens=100) is False


class TestScoreSection:
    def test_identical_text_scores_perfect(self):
        text = "the participants completed a battery of cognitive tasks " * 5
        out = sm.score_section(text, text)
        assert out["unigram_recall"] == 1.0
        assert out["shingle_recall"] == 1.0
        assert out["rouge_l"] == 1.0
        assert out["present"] is True
        assert out["ref_tokens"] > 0

    def test_dropped_text_flags_low_recall(self):
        ref = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 2 + "lambda mu nu"
        pred = "alpha beta gamma"
        out = sm.score_section(pred, ref)
        assert out["unigram_recall"] < 0.5
        assert out["present"] is False

    def test_empty_prediction(self):
        out = sm.score_section("", "alpha beta gamma delta epsilon")
        assert out["unigram_recall"] == 0.0
        assert out["unigram_precision"] is None  # no predicted tokens

    def test_short_ref_shingle_is_none(self):
        out = sm.score_section("a b", "a b c")
        assert out["shingle_recall"] is None

    def test_rouge_l_and_ned_skipped_for_large_sections(self):
        big = " ".join(str(i) for i in range(3500))  # 3500 tokens > cap
        out = sm.score_section(big, big)
        assert out["rouge_l"] is None  # capped for perf (O(m*n) LCS)
        assert out["ned"] is None  # capped for perf (O(m*n) edit distance)
        assert out["unigram_recall"] == 1.0  # O(n) primary alarm still computed
        assert out["shingle_recall"] == 1.0

    def test_ned_computed_for_small_sections(self):
        out = sm.score_section("the quick brown fox", "the quick brown fox")
        assert out["ned"] == 1.0  # below the cap, ned still runs


def _row(pid, st, rec, ref_tokens=100, present=True, pages=None):
    return {
        "paper_id": pid,
        "section_type": st,
        "pages": pages or [],
        "ref_tokens": ref_tokens,
        "pred_tokens": ref_tokens,
        "unigram_recall": rec,
        "shingle_recall": rec,
        "unigram_precision": 1.0,
        "rouge_l": rec,
        "ned": rec,
        "present": present,
    }


class TestAggregateByType:
    def test_groups_and_means_per_type(self):
        rows = [_row("p1", "intro", 0.8), _row("p2", "intro", 0.6), _row("p1", "method", 1.0)]
        agg = sm.aggregate_by_type(rows)
        assert agg["intro"]["unigram_recall"]["mean"] == 0.7
        assert agg["intro"]["n_papers"] == 2
        assert agg["method"]["unigram_recall"]["mean"] == 1.0

    def test_presence_rate_excludes_none(self):
        rows = [
            _row("p1", "intro", 0.8, present=True),
            _row("p2", "intro", 0.1, present=False),
            _row("p3", "intro", 0.0, present=None),
        ]
        agg = sm.aggregate_by_type(rows)
        assert agg["intro"]["presence_rate"] == 0.5  # 1 of 2 evaluable
        assert agg["intro"]["n_present_eval"] == 2

    def test_none_metric_excluded_from_mean(self):
        rows = [
            _row("p1", "intro", 0.8),
            {**_row("p2", "intro", 0.4), "shingle_recall": None},
        ]
        agg = sm.aggregate_by_type(rows)
        assert agg["intro"]["shingle_recall"]["n"] == 1

    def test_layout_recall_aggregated_only_where_present(self):
        rows = [
            {**_row("p1", "_total_body", 0.5), "layout_recall": 0.9},
            {**_row("p2", "_total_body", 0.7), "layout_recall": 0.95},
            _row("p1", "intro", 0.8),  # per-type rows carry no layout_recall
        ]
        agg = sm.aggregate_by_type(rows)
        assert agg["_total_body"]["layout_recall"]["mean"] == 0.925
        assert agg["intro"]["layout_recall"]["mean"] is None


class TestDropReport:
    def test_sorted_ascending_by_recall(self):
        rows = [_row("p1", "intro", 0.9), _row("p2", "method", 0.2), _row("p3", "results", 0.5)]
        report = sm.build_drop_report_rows(rows)
        assert [r["unigram_recall"] for r in report] == [0.2, 0.5, 0.9]

    def test_excludes_trivial_and_none(self):
        rows = [
            _row("p1", "intro", 0.9, ref_tokens=10),  # too short
            {**_row("p2", "method", None), "unigram_recall": None},
        ]
        assert sm.build_drop_report_rows(rows) == []


def test_captions_and_footnotes_count_as_predicted_text():
    """Since 12.0 caption and footnote rows have no section; the prediction
    still groups them by the figure, table or footnote that points at them."""
    from evaluation.evaluate import extract_sections_from_json

    export = {
        "section": [{"section_id": 1, "section_type": "results"}],
        "text": [
            {"text_id": 1, "section_id": 1, "text": "Body."},
            {"text_id": 2, "section_id": None, "text": "Figure 1. Plot."},
            {"text_id": 3, "section_id": None, "text": "1 A note."},
            {"text_id": 4, "section_id": None, "text": "Front matter."},
        ],
        "figure": [{"figure_id": 1, "text_id": 2}],
        "table": [],
        "footnote": [{"footnote_id": 1, "text_id": 3}],
    }
    assert extract_sections_from_json(export) == {
        "results": "Body.",
        "figure": "Figure 1. Plot.",
        "footnote": "1 A note.",
    }
