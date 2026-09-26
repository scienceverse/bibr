"""Tests for the lazy, thread-safe paper-classifier orchestration entry.

Mirrors section_classifier._get_trained_model_async. No real HF model — a fake
PaperClassifierModel is injected via the module-level cache.
"""

from __future__ import annotations

import pytest

from bibr.config import GlobalSettings, Settings
from bibr.structure import paper_classifier as pc


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.setattr(pc, "_paper_model_cache", None)
    yield
    monkeypatch.setattr(pc, "_paper_model_cache", None)


class _FakePrediction:
    def __init__(self):
        self.oecd_l1 = "Social Sciences"
        self.oecd_l1_score = 0.91
        self.oecd_l2 = "Psychology and Cognitive Sciences"
        self.oecd_l2_score = 0.77
        self.paper_type = "empirical"
        self.paper_type_score = 0.88


class _FakeModel:
    def __init__(self):
        self.calls = []

    def classify_batch(self, items):
        self.calls.append(list(items))
        return [_FakePrediction() for _ in items]


async def test_returns_none_when_model_id_unset(monkeypatch):
    monkeypatch.setattr(Settings.ml, "paper_classifier_model_id", None)
    result = await pc.classify_paper_async("Some title", "Some abstract")
    assert result is None


async def test_get_model_async_returns_none_when_unset(monkeypatch):
    monkeypatch.setattr(Settings.ml, "paper_classifier_model_id", None)
    assert await pc._get_paper_model_async() is None


async def test_classify_returns_full_tuple_when_configured(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(Settings.ml, "paper_classifier_model_id", "fake/repo")
    monkeypatch.setattr(pc, "_paper_model_cache", fake)

    result = await pc.classify_paper_async("My Title", "My abstract")

    assert result is not None
    l1, l1s, l2, l2s, pt, pts = result
    assert l1 == "Social Sciences"
    assert l1s == pytest.approx(0.91)
    assert l2 == "Psychology and Cognitive Sciences"
    assert l2s == pytest.approx(0.77)
    assert pt == "empirical"
    assert pts == pytest.approx(0.88)
    # The (title, abstract) pair was forwarded verbatim.
    assert fake.calls == [[("My Title", "My abstract")]]


async def test_sentinel_false_short_circuits(monkeypatch):
    """After a first miss the cache holds sentinel False; no reload attempt."""
    monkeypatch.setattr(Settings.ml, "paper_classifier_model_id", None)
    # Prime the cache the way _get_paper_model does on a miss.
    monkeypatch.setattr(pc, "_paper_model_cache", False)
    assert await pc._get_paper_model_async() is None
    assert await pc.classify_paper_async("t", "a") is None


async def test_managed_resource_is_used_instead_of_lazy_singleton(monkeypatch):
    fake = _FakeModel()

    class Managed:
        async def classify_paper(self, item):
            return fake.classify_batch([item])[0]

    monkeypatch.setattr(pc, "_paper_model_cache", None)
    result = await pc.classify_paper_async(
        "Managed",
        "batch",
        classifier_resources=Managed(),
        settings=GlobalSettings(),
    )
    assert result is not None
    assert result[0] == "Social Sciences"
    assert fake.calls == [[("Managed", "batch")]]


async def test_empty_title_and_abstract_returns_none_without_running_model(monkeypatch):
    """structure-sections-classifiers-11: empty title+abstract carries no
    signal, so the model must not run — None sends the caller down the LLM
    path with the full classification text."""
    fake = _FakeModel()
    monkeypatch.setattr(Settings.ml, "paper_classifier_model_id", "fake/repo")
    monkeypatch.setattr(pc, "_paper_model_cache", fake)

    assert await pc.classify_paper_async("", "") is None
    assert await pc.classify_paper_async("   ", "  ") is None
    assert fake.calls == []


async def test_empty_input_short_circuits_managed_resource(monkeypatch):
    """The empty-input guard applies before the managed-resource path too."""

    class Managed:
        async def classify_paper(self, item):  # pragma: no cover - must not run
            raise AssertionError("model must not run on empty input")

    assert (
        await pc.classify_paper_async(
            "", "", classifier_resources=Managed(), settings=GlobalSettings()
        )
        is None
    )


async def test_nonempty_title_with_empty_abstract_still_runs_model(monkeypatch):
    """Guard: the empty-input rule only fires when _build_input_text is empty —
    a real title with no abstract still goes to the model."""
    fake = _FakeModel()
    monkeypatch.setattr(Settings.ml, "paper_classifier_model_id", "fake/repo")
    monkeypatch.setattr(pc, "_paper_model_cache", fake)

    assert await pc.classify_paper_async("Some title", "") is not None
    assert fake.calls == [[("Some title", "")]]
