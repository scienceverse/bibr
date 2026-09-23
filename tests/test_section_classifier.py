"""Tests for the section classifier lookup-based classification.

Only tests the fast lookup path (no LLM needed).
LLM fallback tests would be @pytest.mark.slow.
"""

from unittest import mock

import pytest

from bibr.clients.prompts import prompt_text
from bibr.config import GlobalSettings, Settings
from bibr.paper_contents import CanonicalSection
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.structure.section_classifier import _classify_lookup, classify_header


class TestLookupClassification:
    """Tests for the fast lookup-based classification path."""

    def test_exact_match_introduction(self):
        section, score = _classify_lookup("introduction")
        assert section == CanonicalSection.INTRODUCTION
        assert score == 1.0

    def test_exact_match_methods(self):
        section, score = _classify_lookup("methods")
        assert section == CanonicalSection.METHODS
        assert score == 1.0

    def test_exact_match_results(self):
        section, score = _classify_lookup("results")
        assert section == CanonicalSection.RESULTS
        assert score == 1.0

    def test_exact_match_discussion(self):
        section, score = _classify_lookup("discussion")
        assert section == CanonicalSection.DISCUSSION
        assert score == 1.0

    def test_exact_match_conclusion(self):
        section, score = _classify_lookup("conclusion")
        assert section == CanonicalSection.DISCUSSION
        assert score == 1.0

    def test_exact_match_abstract(self):
        section, score = _classify_lookup("abstract")
        assert section == CanonicalSection.ABSTRACT
        assert score == 1.0

    def test_exact_match_references(self):
        section, score = _classify_lookup("references")
        assert section == CanonicalSection.REFERENCES
        assert score == 1.0

    def test_alias_materials_and_methods(self):
        section, score = _classify_lookup("materials and methods")
        assert section == CanonicalSection.METHODS
        assert score == 1.0

    def test_alias_background(self):
        section, score = _classify_lookup("background")
        assert section == CanonicalSection.INTRODUCTION
        assert score == 1.0

    def test_alias_summary(self):
        section, score = _classify_lookup("summary")
        assert section == CanonicalSection.ABSTRACT
        assert score == 1.0

    def test_substring_match(self):
        """Substring matching returns 0.95 confidence."""
        section, score = _classify_lookup("materials and methods section")
        assert section == CanonicalSection.METHODS
        assert score == 0.95

    def test_substring_match_experimental(self):
        section, score = _classify_lookup("experimental setup and procedures")
        assert section == CanonicalSection.METHODS
        assert score == 0.95

    def test_unknown_header(self):
        section, score = _classify_lookup("some random section title")
        assert section == CanonicalSection.UNKNOWN
        assert score == 0.0

    def test_empty_string(self):
        section, score = _classify_lookup("")
        assert section == CanonicalSection.UNKNOWN
        assert score == 0.0


class TestWordBoundaryMatching:
    """Tests that substring matching uses word boundaries to prevent false positives."""

    def test_preferences_not_matched_as_references(self):
        """'preferences' contains 'references' as a substring but must not match."""
        section, score = _classify_lookup("general preferences and relationship context effect")
        assert section != CanonicalSection.REFERENCES

    def test_flipped_preferences_not_matched_as_references(self):
        section, score = _classify_lookup("flipped preferences")
        assert section != CanonicalSection.REFERENCES

    def test_literature_search_not_matched_as_references(self):
        """'literature search' must not match REFERENCES (it's a methods subsection)."""
        section, score = _classify_lookup("literature search")
        assert section != CanonicalSection.REFERENCES

    def test_actual_references_still_matches(self):
        section, score = _classify_lookup("references")
        assert section == CanonicalSection.REFERENCES
        assert score == 1.0

    def test_references_in_longer_header(self):
        """'references and notes' should still match via word-boundary substring."""
        section, score = _classify_lookup("references and notes")
        assert section == CanonicalSection.REFERENCES
        assert score == 0.95

    def test_literature_review_matches_introduction(self):
        """'literature review' is an exact alias for INTRODUCTION."""
        section, score = _classify_lookup("literature review")
        assert section == CanonicalSection.INTRODUCTION
        assert score == 1.0

    def test_literature_cited_matches_references(self):
        section, score = _classify_lookup("literature cited")
        assert section == CanonicalSection.REFERENCES
        assert score == 1.0


class TestClassifyWithNormalization:
    """Tests for classify_header() which normalizes input before lookup."""

    def test_numbered_heading(self):
        """classify_header() normalizes '1. Introduction' -> 'introduction'."""
        section, score = classify_header("1. Introduction")
        assert section == CanonicalSection.INTRODUCTION
        assert score == 1.0

    def test_dotted_numbering(self):
        section, score = classify_header("3.1 Results and Discussion")
        assert section == CanonicalSection.RESULTS
        assert score == 1.0  # "results and discussion" is now an exact alias

    def test_unknown_returns_unknown(self):
        """When lookup fails, classify_header returns UNKNOWN (no LLM in sync mode)."""
        section, score = classify_header("our novel computational framework")
        assert section == CanonicalSection.UNKNOWN


class TestExpandedAliases:
    """Tests for the expanded alias set added in the performance optimization."""

    def test_ampersand_materials_methods(self):
        section, score = _classify_lookup("materials & methods")
        assert section == CanonicalSection.METHODS
        assert score == 1.0

    def test_ampersand_results_discussion(self):
        section, score = _classify_lookup("results & discussion")
        assert section == CanonicalSection.RESULTS
        assert score == 1.0

    def test_related_work(self):
        section, score = _classify_lookup("related work")
        assert section == CanonicalSection.INTRODUCTION
        assert score == 1.0

    def test_literature_review(self):
        section, score = _classify_lookup("literature review")
        assert section == CanonicalSection.INTRODUCTION
        assert score == 1.0

    def test_evaluation(self):
        section, score = _classify_lookup("evaluation")
        assert section == CanonicalSection.RESULTS
        assert score == 1.0

    def test_experiments(self):
        section, score = _classify_lookup("experiments")
        assert section == CanonicalSection.RESULTS
        assert score == 1.0

    def test_competing_interests(self):
        section, score = _classify_lookup("competing interests")
        assert section == CanonicalSection.COI
        assert score == 1.0

    def test_data_availability(self):
        section, score = _classify_lookup("data availability")
        assert section == CanonicalSection.OPEN_DATA
        assert score == 1.0

    def test_author_contributions(self):
        section, score = _classify_lookup("author contributions")
        assert section == CanonicalSection.AUTHOR_CONTRIBUTIONS
        assert score == 1.0

    def test_ethics_statement(self):
        section, score = _classify_lookup("ethics statement")
        assert section == CanonicalSection.ETHICS
        assert score == 1.0

    def test_supplementary_information(self):
        section, score = _classify_lookup("supplementary information")
        assert section == CanonicalSection.ENDNOTE
        assert score == 1.0

    def test_supplementary_materials(self):
        section, score = _classify_lookup("supplementary materials")
        assert section == CanonicalSection.ENDNOTE
        assert score == 1.0

    def test_extended_data(self):
        section, score = _classify_lookup("extended data")
        assert section == CanonicalSection.ENDNOTE
        assert score == 1.0

    def test_conclusions_and_future_work(self):
        section, score = _classify_lookup("conclusions and future work")
        assert section == CanonicalSection.ENDNOTE
        assert score == 1.0

    def test_funding(self):
        section, score = _classify_lookup("funding")
        assert section == CanonicalSection.FUNDING
        assert score == 1.0

    def test_proposed_method(self):
        section, score = _classify_lookup("proposed method")
        assert section == CanonicalSection.METHODS
        assert score == 1.0

    def test_participants(self):
        section, score = _classify_lookup("participants")
        assert section == CanonicalSection.METHODS
        assert score == 1.0

    def test_statistical_analysis(self):
        section, score = _classify_lookup("statistical analysis")
        assert section == CanonicalSection.METHODS
        assert score == 1.0

    def test_results_and_discussion(self):
        section, score = _classify_lookup("results and discussion")
        assert section == CanonicalSection.RESULTS
        assert score == 1.0

    def test_numbered_related_work_via_classify(self):
        """classify_header() normalizes '2. Related Work' -> 'related work'."""
        section, score = classify_header("2. Related Work")
        assert section == CanonicalSection.INTRODUCTION
        assert score == 1.0


class TestBatchClassification:
    """Tests for classify_headers_batch (sync, lookup-only)."""

    def test_all_known_headers(self):
        from bibr.structure.section_classifier import classify_headers_batch

        results = classify_headers_batch(["Introduction", "Methods", "Results"])
        assert len(results) == 3
        assert results[0][0] == CanonicalSection.INTRODUCTION
        assert results[1][0] == CanonicalSection.METHODS
        assert results[2][0] == CanonicalSection.RESULTS

    def test_empty_input(self):
        from bibr.structure.section_classifier import classify_headers_batch

        results = classify_headers_batch([])
        assert results == []

    def test_preserves_order(self):
        from bibr.structure.section_classifier import classify_headers_batch

        results = classify_headers_batch(["Funding", "Abstract", "References"])
        assert results[0][0] == CanonicalSection.FUNDING
        assert results[1][0] == CanonicalSection.ABSTRACT
        assert results[2][0] == CanonicalSection.REFERENCES

    def test_numbered_headers_normalized(self):
        from bibr.structure.section_classifier import classify_headers_batch

        results = classify_headers_batch(["1. Introduction", "2. Methods", "3. Results"])
        assert results[0][0] == CanonicalSection.INTRODUCTION
        assert results[1][0] == CanonicalSection.METHODS
        assert results[2][0] == CanonicalSection.RESULTS


class TestTrainedClassifierBranch:
    """Tests for the trained-model branch in classify_headers_batch_async."""

    async def test_classify_uses_trained_model_when_configured(self, monkeypatch):
        """When ml.section_classifier_model_id is set, classifier uses the trained
        model and skips the LLM fallback path."""
        from bibr.structure import section_classifier

        called = {"trained": 0, "llm": 0}

        async def fake_trained(pairs):
            called["trained"] += 1
            return [(CanonicalSection.METHODS, 0.99, True) for _ in pairs]

        async def fake_llm(*args, **kwargs):
            called["llm"] += 1
            return []

        monkeypatch.setattr(section_classifier, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(section_classifier, "_classify_llm_batch", fake_llm)
        monkeypatch.setattr(
            Settings.ml,
            "section_classifier_model_id",
            "fake-repo",
        )
        # Prevent any accidental snapshot_download call from a stale cache.
        monkeypatch.setattr(section_classifier, "_model_cache", object())

        headers = ["Some Weird Header"]
        body_snippets = ["body..."]
        results = await section_classifier.classify_headers_batch_async(
            headers, body_snippets=body_snippets
        )

        assert called["trained"] == 1
        assert called["llm"] == 0
        assert len(results) == 1
        # 4-tuple return shape: (canonical, score, is_top_level | None, source)
        canon, score, is_top, source = results[0]
        assert canon == CanonicalSection.METHODS
        assert score == 0.99
        assert is_top is True
        assert source == "model"

    async def test_degraded_managed_classifier_falls_back_to_llm(self, monkeypatch):
        from bibr.structure import section_classifier

        async def no_trained_result(_items):
            return []

        async def llm_result(headers, **_kwargs):
            return [(CanonicalSection.METHODS, 0.9) for _ in headers]

        async def configured_model():
            return object()

        monkeypatch.setattr(section_classifier, "_classify_trained_batch", no_trained_result)
        monkeypatch.setattr(section_classifier, "_classify_llm_batch", llm_result)
        monkeypatch.setattr(section_classifier, "_get_trained_model_async", configured_model)
        monkeypatch.setattr(Settings.ml, "section_classifier_llm_escalation", True)

        result = await section_classifier.classify_headers_batch_async(["Unfamiliar heading"])

        assert result == [(CanonicalSection.METHODS, 0.9, None, "llm")]

    async def test_classify_falls_back_to_llm_when_model_id_unset(self, monkeypatch):
        """When ml.section_classifier_model_id is None, the LLM path runs."""
        from bibr.structure import section_classifier

        called = {"trained": 0, "llm": 0}

        async def fake_trained(pairs):
            called["trained"] += 1
            return [(CanonicalSection.METHODS, 0.99, True) for _ in pairs]

        async def fake_llm(header_texts, body_snippets=None, llm_client=None):
            called["llm"] += 1
            return [(CanonicalSection.RESULTS, 0.9) for _ in header_texts]

        monkeypatch.setattr(section_classifier, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(section_classifier, "_classify_llm_batch", fake_llm)
        monkeypatch.setattr(Settings.ml, "section_classifier_model_id", None)

        results = await section_classifier.classify_headers_batch_async(["Some Weird Header"])

        assert called["trained"] == 0
        assert called["llm"] == 1
        canon, score, is_top, source = results[0]
        assert canon == CanonicalSection.RESULTS
        assert is_top is None  # LLM path has no is_top_level signal
        assert source == "llm"

    async def test_low_confidence_collapses_to_unknown(self, monkeypatch):
        """Trained predictions below ``section_classifier_min_confidence``
        must collapse to UNKNOWN with ``is_top_level=None``."""
        pytest.importorskip("torch")
        from bibr.structure import section_classifier
        from bibr.structure.section_classifier_model import SectionPrediction

        class FakeModel:
            def classify_batch(self, pairs, max_length=256):  # noqa: ARG002
                return [
                    SectionPrediction(
                        canonical_type=CanonicalSection.METHODS,
                        is_top_level=True,
                        score=0.30,
                    )
                ]

        monkeypatch.setattr(section_classifier, "_get_trained_model", lambda: FakeModel())
        monkeypatch.setattr(
            Settings.ml,
            "section_classifier_model_id",
            "fake-repo",
        )
        monkeypatch.setattr(
            Settings.ml,
            "section_classifier_min_confidence",
            0.5,
        )

        results = await section_classifier.classify_headers_batch_async(
            ["Mysterious Header"], body_snippets=["body..."]
        )

        canon, score, is_top, source = results[0]
        assert canon == CanonicalSection.UNKNOWN
        assert is_top is None
        # Score preserved for downstream debugging/logging.
        assert score == 0.30
        assert source is None  # UNKNOWN carries no classification source

    async def test_high_confidence_passes_threshold(self, monkeypatch):
        """At-or-above threshold, the trained prediction is kept verbatim."""
        pytest.importorskip("torch")
        from bibr.structure import section_classifier
        from bibr.structure.section_classifier_model import SectionPrediction

        class FakeModel:
            def classify_batch(self, pairs, max_length=256):  # noqa: ARG002
                return [
                    SectionPrediction(
                        canonical_type=CanonicalSection.RESULTS,
                        is_top_level=False,
                        score=0.92,
                    )
                ]

        monkeypatch.setattr(section_classifier, "_get_trained_model", lambda: FakeModel())
        monkeypatch.setattr(
            Settings.ml,
            "section_classifier_model_id",
            "fake-repo",
        )
        monkeypatch.setattr(
            Settings.ml,
            "section_classifier_min_confidence",
            0.5,
        )

        results = await section_classifier.classify_headers_batch_async(
            ["Some Weird Header"], body_snippets=["body..."]
        )

        canon, score, is_top, source = results[0]
        assert canon == CanonicalSection.RESULTS
        assert is_top is False
        assert score == 0.92
        assert source == "model"

    async def test_lookup_hits_skip_both_paths(self, monkeypatch):
        """Known aliases must bypass both LLM and trained-model calls."""
        from bibr.structure import section_classifier

        called = {"trained": 0, "llm": 0}

        async def fake_trained(pairs):
            called["trained"] += 1
            return []

        async def fake_llm(*args, **kwargs):
            called["llm"] += 1
            return []

        monkeypatch.setattr(section_classifier, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(section_classifier, "_classify_llm_batch", fake_llm)
        monkeypatch.setattr(
            Settings.ml,
            "section_classifier_model_id",
            "fake-repo",
        )

        results = await section_classifier.classify_headers_batch_async(
            ["Introduction", "Methods", "References"]
        )

        assert called["trained"] == 0
        assert called["llm"] == 0
        assert results[0][0] == CanonicalSection.INTRODUCTION
        assert results[1][0] == CanonicalSection.METHODS
        assert results[2][0] == CanonicalSection.REFERENCES
        # Lookup hits emit None for is_top_level (no signal).
        assert all(r[2] is None for r in results)
        # Exact alias hits are tagged as such in the classification source.
        assert all(r[3] == "exact_alias" for r in results)


class TestLLMKeySanitization:
    """Schema tolerance: reasoning models sometimes emit decorated JSON keys
    (e.g. '<header'). `_sanitize_llm_keys` strips such decoration before
    pydantic validation."""

    def test_stray_leading_bracket_stripped(self):
        from bibr.structure.section_classifier import _sanitize_llm_keys

        out = _sanitize_llm_keys({"<header": "comparison of methods", "section_type": "results"})
        assert out == {"header": "comparison of methods", "section_type": "results"}

    def test_trailing_bracket_stripped(self):
        from bibr.structure.section_classifier import _sanitize_llm_keys

        out = _sanitize_llm_keys({"header>": "intro", "section_type>": "intro"})
        assert out == {"header": "intro", "section_type": "intro"}

    def test_quoted_key_stripped(self):
        from bibr.structure.section_classifier import _sanitize_llm_keys

        out = _sanitize_llm_keys({'"header"': "abstract"})
        assert out == {"header": "abstract"}

    def test_clean_keys_unchanged(self):
        from bibr.structure.section_classifier import _sanitize_llm_keys

        out = _sanitize_llm_keys({"header": "abstract", "section_type": "abstract"})
        assert out == {"header": "abstract", "section_type": "abstract"}

    def test_non_dict_passthrough(self):
        from bibr.structure.section_classifier import _sanitize_llm_keys

        assert _sanitize_llm_keys("not a dict") == "not a dict"
        assert _sanitize_llm_keys([1, 2, 3]) == [1, 2, 3]

    def test_key_becomes_empty_falls_back_to_original(self):
        """If stripping leaves an empty string, keep the original key so
        pydantic's own error message points at the real culprit."""
        from bibr.structure.section_classifier import _sanitize_llm_keys

        out = _sanitize_llm_keys({"<>": "value"})
        assert out == {"<>": "value"}


# ── Async batch / LLM fallback tests ───────────────────────────────────


def _stub_llm(classifications):
    """Build a stub LLMClient whose invoke_structured returns the provided list."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    result_obj = SimpleNamespace(
        classifications=[SimpleNamespace(header=h, section_type=t) for (h, t) in classifications]
    )
    client = mock.MagicMock()
    client.limiter = mock.MagicMock()
    client.limiter.acquire = AsyncMock(return_value=None)
    client.invoke_structured = AsyncMock(return_value=result_obj)
    client.close = AsyncMock(return_value=None)
    return client


class TestClassifyHeadersBatchAsync:
    async def test_section_llm_receives_task_output_cap(self):
        from bibr.structure.section_classifier import classify_headers_batch_async

        client = _stub_llm([("mystery", "method")])
        await classify_headers_batch_async(["mystery"], llm_client=client)
        assert client.invoke_structured.await_args.kwargs["max_tokens"] == 4096

    async def test_section_llm_prompt_has_explicit_native_roles(self):
        from bibr.structure.section_classifier import _classify_llm_batch

        client = _stub_llm([("mystery", "method")])
        await _classify_llm_batch(["mystery"], llm_client=client)

        messages = client.invoke_structured.await_args.args[1]
        parts = messages[0]["content"]
        assert [item["nuextract_role"] for item in parts] == ["instructions", "document"]
        assert "mystery" not in parts[0]["text"]
        assert "mystery" in parts[1]["text"]
        assert "above" not in parts[0]["text"].casefold()
        assert "below" not in parts[0]["text"].casefold()

    async def test_known_headers_skip_llm(self):
        from bibr.structure.section_classifier import classify_headers_batch_async

        client = _stub_llm([])
        results = await classify_headers_batch_async(
            ["Introduction", "Methods", "References"], llm_client=client
        )
        # All three matched via lookup — LLM not called
        client.invoke_structured.assert_not_called()
        assert results[0][0] == CanonicalSection.INTRODUCTION
        assert results[1][0] == CanonicalSection.METHODS
        assert results[2][0] == CanonicalSection.REFERENCES

    async def test_unknown_headers_routed_to_llm(self):
        from bibr.structure.section_classifier import classify_headers_batch_async

        client = _stub_llm([("our novel framework", "method")])
        results = await classify_headers_batch_async(
            ["Introduction", "Our Novel Framework"], llm_client=client
        )
        client.invoke_structured.assert_awaited_once()
        assert results[0][0] == CanonicalSection.INTRODUCTION
        assert results[1][0] == CanonicalSection.METHODS

    async def test_duplicate_unknowns_deduped(self):
        from bibr.structure.section_classifier import classify_headers_batch_async

        client = _stub_llm([("xyz framework", "method")])
        results = await classify_headers_batch_async(
            ["xyz framework", "xyz framework", "xyz framework"], llm_client=client
        )
        client.invoke_structured.assert_awaited_once()
        # All three get the same classification
        assert all(r[0] == CanonicalSection.METHODS for r in results)

    async def test_empty_normalized_text_is_unknown_no_llm_call(self):
        from bibr.structure.section_classifier import classify_headers_batch_async

        client = _stub_llm([])
        results = await classify_headers_batch_async(["123", "..."], llm_client=client)
        client.invoke_structured.assert_not_called()
        assert all(r[0] == CanonicalSection.UNKNOWN for r in results)

    async def test_llm_failure_falls_back_to_unknown(self):
        from unittest.mock import AsyncMock

        from bibr.structure.section_classifier import classify_headers_batch_async

        client = mock.MagicMock()
        client.limiter = mock.MagicMock()
        client.limiter.acquire = AsyncMock(return_value=None)
        client.invoke_structured = AsyncMock(side_effect=RuntimeError("LLM down"))
        client.close = AsyncMock(return_value=None)

        results = await classify_headers_batch_async(
            ["abstract", "weird unknown thing"], llm_client=client
        )
        assert results[0][0] == CanonicalSection.ABSTRACT
        assert results[1][0] == CanonicalSection.UNKNOWN

    async def test_llm_invalid_section_value_falls_back(self):
        """LLM returns a section_type not in the valid set → unknown."""
        from bibr.structure.section_classifier import classify_headers_batch_async

        client = _stub_llm([("weird thing", "made_up_value")])
        results = await classify_headers_batch_async(["weird thing"], llm_client=client)
        assert results[0][0] == CanonicalSection.UNKNOWN

    async def test_body_snippets_passed_to_llm(self):
        """When body snippets are provided, they should be included in the prompt."""
        from bibr.structure.section_classifier import classify_headers_batch_async

        client = _stub_llm([("results", "results")])
        await classify_headers_batch_async(
            ["weirdly named section"],
            body_snippets=["This section reports the experimental results of..."],
            llm_client=client,
        )
        # Inspect the prompt that was sent
        call = client.invoke_structured.await_args
        messages = call.args[1]
        prompt = prompt_text(messages[0]["content"])
        assert "experimental results" in prompt
        assert "opening text" in prompt

    async def test_body_snippet_truncated_at_500_chars(self, monkeypatch):
        import bibr.structure.section_classifier as sc

        captured = {}

        class FakeLimiter:
            async def acquire(self):
                return None

        class FakeClient:
            limiter = FakeLimiter()

            async def invoke_structured(self, model, messages, system, label=None, max_tokens=None):
                captured["prompt"] = prompt_text(messages[0]["content"])

                class R:
                    classifications = []

                return R()

            async def close(self):
                return None

        await sc._classify_llm_batch(
            ["mystery header"], body_snippets=["x" * 1000], llm_client=FakeClient()
        )
        # 500 chars of body (plus the "..." suffix), not the old 200 cap.
        assert "x" * 500 in captured["prompt"]
        assert "x" * 501 not in captured["prompt"]


class TestTrainedModelLoadGuard:
    """First load must be single-flight (no double-load under concurrency)
    and must never run on the event-loop thread (it downloads + loads
    weights — seconds of blocking)."""

    @staticmethod
    def _patch_loader(monkeypatch, record):
        import threading
        import time

        pytest.importorskip("torch")
        from bibr.structure import section_classifier
        from bibr.structure.section_classifier_model import SectionClassifierModel

        monkeypatch.setattr(section_classifier, "_model_cache", None)
        monkeypatch.setattr(Settings.ml, "section_classifier_model_id", "fake/model")

        def fake_from_pretrained(repo_id, revision="main", device=None):
            record.append(threading.current_thread())
            time.sleep(0.05)  # widen the race window
            return object()

        monkeypatch.setattr(
            SectionClassifierModel, "from_pretrained", staticmethod(fake_from_pretrained)
        )
        return section_classifier

    def test_concurrent_first_loads_are_single_flight(self, monkeypatch):
        import concurrent.futures

        record = []
        sc = self._patch_loader(monkeypatch, record)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda _: sc._get_trained_model(), range(4)))

        assert len(record) == 1, f"model loaded {len(record)} times"
        assert all(r is results[0] for r in results)

    async def test_async_path_loads_off_the_event_loop(self, monkeypatch):
        import threading

        record = []
        sc = self._patch_loader(monkeypatch, record)

        model = await sc._get_trained_model_async()

        assert model is not None
        assert record, "loader was never invoked"
        assert record[0] is not threading.current_thread(), (
            "model load ran on the event-loop thread"
        )


async def test_trained_batch_uses_managed_classifier_resource(monkeypatch):
    # The trained classifier path is an ``ml``-extra feature; a core install
    # only ever reaches the alias/LLM fallbacks covered elsewhere in this file.
    # exc_type: the module re-raises a plain ImportError (not the
    # ModuleNotFoundError importorskip defaults to) when torch is absent.
    pytest.importorskip(
        "bibr.structure.section_classifier_model",
        reason="needs the 'ml' extra",
        exc_type=ImportError,
    )

    import bibr.structure.section_classifier as sc
    from bibr.paper_contents import CanonicalSection
    from bibr.structure.section_classifier_model import SectionPrediction

    class Managed:
        async def classify_sections(self, items):
            return [SectionPrediction(CanonicalSection.METHODS, True, 0.91) for _ in items]

    settings = GlobalSettings()
    settings.ml.section_classifier_min_confidence = 0.5
    result = await sc._classify_trained_batch(
        [sc.HeaderContext("Methods", "body")],
        classifier_resources=Managed(),
        settings=settings,
    )
    assert result == [(CanonicalSection.METHODS, 0.91, True)]


async def test_trained_inference_failure_falls_back_and_warns(monkeypatch):
    import bibr.structure.section_classifier as sc

    async def fake_get_model(*_args):
        return object()

    async def fail_trained(items):  # noqa: ARG001
        raise RuntimeError("tokenizer exploded on private document text")

    settings = GlobalSettings()
    settings.ml.section_classifier_llm_escalation = False
    warnings: list[ProcessingWarning] = []
    monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)
    monkeypatch.setattr(sc, "_classify_trained_batch", fail_trained)

    results = await sc.classify_headers_batch_async(
        ["unfamiliar section"],
        settings=settings,
        degradation_warnings=warnings,
    )

    assert results == [(CanonicalSection.UNKNOWN, 0.0, None, None)]
    # Recorded once although both the error and the empty result fell back.
    assert [w.code for w in warnings] == [WarningCode.SECTION_CLASSIFIER_DEGRADED]
    assert "private document text" not in warnings[0].message


async def test_configured_but_unloadable_model_warns(monkeypatch):
    """A core install (no torch) or a failed download must not look like a healthy run."""
    import bibr.structure.section_classifier as sc

    async def no_model(*_args):
        return None

    settings = GlobalSettings()
    settings.ml.section_classifier_model_id = "scienceverse/bibr-section-classifier"
    settings.ml.section_classifier_llm_escalation = False
    warnings: list[ProcessingWarning] = []
    monkeypatch.setattr(sc, "_get_trained_model_async", no_model)
    results = await sc.classify_headers_batch_async(
        ["unfamiliar section"],
        settings=settings,
        degradation_warnings=warnings,
    )
    assert results == [(CanonicalSection.UNKNOWN, 0.0, None, None)]
    assert [w.code for w in warnings] == [WarningCode.SECTION_CLASSIFIER_DEGRADED]


async def test_unconfigured_model_is_not_reported_as_degraded(monkeypatch):
    import bibr.structure.section_classifier as sc

    async def no_model(*_args):
        return None

    settings = GlobalSettings()
    settings.ml.section_classifier_model_id = None
    settings.ml.section_classifier_llm_escalation = False
    warnings: list[ProcessingWarning] = []
    monkeypatch.setattr(sc, "_get_trained_model_async", no_model)
    await sc.classify_headers_batch_async(
        ["unfamiliar section"],
        settings=settings,
        degradation_warnings=warnings,
    )
    assert warnings == []


async def test_degraded_serve_classifier_resource_warns():
    """serve: a classifier that failed to load answers None; the export must say so."""
    import bibr.structure.section_classifier as sc

    class Degraded:
        async def classify_sections(self, items):  # noqa: ARG002
            return None

    settings = GlobalSettings()
    settings.ml.section_classifier_model_id = "scienceverse/bibr-section-classifier"
    settings.ml.section_classifier_llm_escalation = False
    warnings: list[ProcessingWarning] = []
    results = await sc.classify_headers_batch_async(
        ["unfamiliar section"],
        classifier_resources=Degraded(),
        settings=settings,
        degradation_warnings=warnings,
    )
    assert results == [(CanonicalSection.UNKNOWN, 0.0, None, None)]
    assert [w.code for w in warnings] == [WarningCode.SECTION_CLASSIFIER_DEGRADED]


class TestLlmEscalation:
    """UNKNOWN results from the trained model escalate to the LLM."""

    @pytest.fixture(autouse=True)
    def _reset_model_cache(self, monkeypatch):
        import bibr.structure.section_classifier as sc

        monkeypatch.setattr(sc, "_model_cache", None)
        yield
        monkeypatch.setattr(sc, "_model_cache", None)

    async def test_unknown_trained_results_escalate_to_llm(self, monkeypatch):
        import bibr.structure.section_classifier as sc
        from bibr.paper_contents import CanonicalSection

        async def fake_trained(pairs):
            # Low-confidence collapse: UNKNOWN with is_top None
            return [(CanonicalSection.UNKNOWN, 0.3, None) for _ in pairs]

        llm_calls = []

        async def fake_llm(texts, body_snippets=None, llm_client=None, **kw):
            llm_calls.append(list(texts))
            return [(CanonicalSection.ETHICS, 0.85) for _ in texts]

        monkeypatch.setattr(sc, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(sc, "_classify_llm_batch", fake_llm)

        async def fake_get_model():
            return object()  # non-None → trained branch taken

        monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)

        results = await sc.classify_headers_batch_async(["patient consent and irb"])
        assert llm_calls == [["patient consent and irb"]]
        assert results[0] == (CanonicalSection.ETHICS, 0.85, None, "llm")

    async def test_confident_trained_results_do_not_escalate(self, monkeypatch):
        import bibr.structure.section_classifier as sc
        from bibr.paper_contents import CanonicalSection

        async def fake_trained(pairs):
            return [(CanonicalSection.METHODS, 0.92, True) for _ in pairs]

        async def fake_llm(texts, **kw):
            raise AssertionError("LLM must not be called for confident results")

        async def fake_get_model():
            return object()

        monkeypatch.setattr(sc, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(sc, "_classify_llm_batch", fake_llm)
        monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)

        results = await sc.classify_headers_batch_async(["some unusual data header"])
        assert results[0] == (CanonicalSection.METHODS, 0.92, True, "model")

    async def test_escalation_disabled_by_setting(self, monkeypatch):
        import bibr.structure.section_classifier as sc
        from bibr.config import Settings
        from bibr.paper_contents import CanonicalSection

        monkeypatch.setattr(Settings.ml, "section_classifier_llm_escalation", False)

        async def fake_trained(pairs):
            return [(CanonicalSection.UNKNOWN, 0.3, None) for _ in pairs]

        async def fake_llm(texts, **kw):
            raise AssertionError("escalation disabled — LLM must not be called")

        async def fake_get_model():
            return object()

        monkeypatch.setattr(sc, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(sc, "_classify_llm_batch", fake_llm)
        monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)

        results = await sc.classify_headers_batch_async(["mystery header"])
        assert results[0] == (CanonicalSection.UNKNOWN, 0.3, None, None)

    async def test_llm_unknown_keeps_trained_result(self, monkeypatch):
        import bibr.structure.section_classifier as sc
        from bibr.paper_contents import CanonicalSection

        async def fake_trained(pairs):
            return [(CanonicalSection.UNKNOWN, 0.4, None) for _ in pairs]

        async def fake_llm(texts, **kw):
            return [(CanonicalSection.UNKNOWN, 0.0) for _ in texts]

        async def fake_get_model():
            return object()

        monkeypatch.setattr(sc, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(sc, "_classify_llm_batch", fake_llm)
        monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)

        results = await sc.classify_headers_batch_async(["mystery header"])
        assert results[0] == (CanonicalSection.UNKNOWN, 0.4, None, None)


class TestContextPropagation:
    """The trained-model path must see each header's document context
    (position + neighbors), and headers with identical text but distinct
    context must not collapse onto one prediction."""

    @pytest.fixture(autouse=True)
    def _reset_model_cache(self, monkeypatch):
        import bibr.structure.section_classifier as sc

        monkeypatch.setattr(sc, "_model_cache", None)
        yield
        monkeypatch.setattr(sc, "_model_cache", None)

    async def test_trained_model_receives_neighbor_context(self, monkeypatch):
        import bibr.structure.section_classifier as sc
        from bibr.paper_contents import CanonicalSection

        captured = {}

        async def fake_trained(items):
            captured["items"] = items
            return [(CanonicalSection.METHODS, 0.9, True) for _ in items]

        async def fake_get_model():
            return object()

        monkeypatch.setattr(sc, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)

        headers = ["Zqx One", "Zqx Two", "Zqx Three"]  # no alias hits
        await sc.classify_headers_batch_async(
            headers, body_snippets=["b1", "b2", "b3"], llm_client=None
        )
        items = captured["items"]
        assert items[0].prev_heading == "" and items[0].next_heading == "Zqx Two"
        assert items[1].prev_heading == "Zqx One" and items[1].next_heading == "Zqx Three"
        assert items[1].relative_position == 0.5

    async def test_repeated_header_distinct_contexts_not_collapsed(self, monkeypatch):
        """Two identical headers in different document positions (e.g. two
        "Participants" sections belonging to different studies) must reach
        the trained model as two distinct items, and each index's prediction
        must come from its own item, not a collapsed shared one."""
        import bibr.structure.section_classifier as sc
        from bibr.paper_contents import CanonicalSection

        captured = {}

        async def fake_trained(items):
            captured["items"] = items
            # Distinguish the two "Zqx Section" items by their prev_heading
            # so the test can verify per-index results are honored.
            out = []
            for item in items:
                if item.heading == "zqx section" and item.prev_heading == "Introduction":
                    out.append((CanonicalSection.METHODS, 0.9, True))
                elif item.heading == "zqx section" and item.prev_heading == "Discussion":
                    out.append((CanonicalSection.RESULTS, 0.9, True))
                else:
                    out.append((CanonicalSection.UNKNOWN, 0.0, None))
            return out

        async def fake_get_model():
            return object()

        monkeypatch.setattr(sc, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)

        # "Introduction"/"Discussion"/"References" are trusted aliases and
        # bypass the trained model entirely — only the two "Zqx Section"
        # headers (identical text, different neighbors) reach the pool.
        headers = ["Introduction", "Zqx Section", "Discussion", "Zqx Section", "References"]
        results = await sc.classify_headers_batch_async(headers)

        items = captured["items"]
        assert len(items) == 2
        assert results[1][0] == CanonicalSection.METHODS
        assert results[3][0] == CanonicalSection.RESULTS


class TestSubstringDemotion:
    """Weak substring alias hits are priors, not trusted classifications."""

    @pytest.fixture(autouse=True)
    def _reset_model_cache(self, monkeypatch):
        import bibr.structure.section_classifier as sc

        monkeypatch.setattr(sc, "_model_cache", None)
        yield
        monkeypatch.setattr(sc, "_model_cache", None)

    def test_full_lookup_exact_is_trusted(self):
        from bibr.structure.section_classifier import _classify_lookup_full

        section, score, trusted = _classify_lookup_full("introduction")
        assert section == CanonicalSection.INTRODUCTION
        assert score == 1.0
        assert trusted is True

    def test_full_lookup_high_coverage_substring_is_trusted(self):
        from bibr.structure.section_classifier import _classify_lookup_full

        # alias "materials and methods" covers 21/29 chars ≈ 0.72
        section, score, trusted = _classify_lookup_full("materials and methods section")
        assert section == CanonicalSection.METHODS
        assert trusted is True

    def test_full_lookup_low_coverage_substring_is_untrusted(self):
        from bibr.structure.section_classifier import _classify_lookup_full

        # alias "limitations" covers 11/32 chars ≈ 0.34
        section, score, trusted = _classify_lookup_full("limitations of existing theories")
        assert section == CanonicalSection.DISCUSSION
        assert trusted is False

    def test_sync_lookup_behavior_unchanged(self):
        # no_llm path contract: substring hits still returned at 0.95
        section, score = _classify_lookup("limitations of existing theories")
        assert section == CanonicalSection.DISCUSSION
        assert score == 0.95

    async def test_untrusted_hit_routed_to_model_model_wins(self, monkeypatch):
        import bibr.structure.section_classifier as sc
        from bibr.paper_contents import CanonicalSection

        async def fake_trained(pairs):
            return [(CanonicalSection.INTRODUCTION, 0.9, True) for _ in pairs]

        async def fake_get_model():
            return object()

        monkeypatch.setattr(sc, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)

        results = await sc.classify_headers_batch_async(["limitations of existing theories"])
        assert results[0] == (CanonicalSection.INTRODUCTION, 0.9, True, "model")

    async def test_untrusted_hit_falls_back_to_alias_prior_when_model_unknown(self, monkeypatch):
        import bibr.structure.section_classifier as sc
        from bibr.config import Settings
        from bibr.paper_contents import CanonicalSection

        # Disable escalation so the model's UNKNOWN survives to the fallback.
        monkeypatch.setattr(Settings.ml, "section_classifier_llm_escalation", False)

        async def fake_trained(pairs):
            return [(CanonicalSection.UNKNOWN, 0.2, None) for _ in pairs]

        async def fake_get_model():
            return object()

        monkeypatch.setattr(sc, "_classify_trained_batch", fake_trained)
        monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)

        results = await sc.classify_headers_batch_async(["limitations of existing theories"])
        assert results[0] == (CanonicalSection.DISCUSSION, 0.6, None, "alias_prior")

    async def test_trusted_hits_still_skip_model(self, monkeypatch):
        import bibr.structure.section_classifier as sc

        async def fail_trained(pairs):
            raise AssertionError("trusted lookup hits must not reach the model")

        async def fake_get_model():
            return object()

        monkeypatch.setattr(sc, "_classify_trained_batch", fail_trained)
        monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)

        results = await sc.classify_headers_batch_async(["introduction", "methods"])
        assert results[0][0] == CanonicalSection.INTRODUCTION
        assert results[1][0] == CanonicalSection.METHODS
        # Exact alias hits are tagged "exact_alias".
        assert results[0][3] == "exact_alias"
        assert results[1][3] == "exact_alias"

    async def test_trusted_substring_hit_tagged_substring_alias(self, monkeypatch):
        import bibr.structure.section_classifier as sc

        async def fail_trained(pairs):
            raise AssertionError("trusted substring hits must not reach the model")

        async def fake_get_model():
            return object()

        monkeypatch.setattr(sc, "_classify_trained_batch", fail_trained)
        monkeypatch.setattr(sc, "_get_trained_model_async", fake_get_model)

        # "materials and methods" covers most of the header → trusted substring.
        results = await sc.classify_headers_batch_async(["Materials and Methods Section"])
        assert results[0][0] == CanonicalSection.METHODS
        assert results[0][1] == 0.95
        assert results[0][3] == "substring_alias"
