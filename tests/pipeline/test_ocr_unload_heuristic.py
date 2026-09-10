"""The per-chunk OCR teardown heuristic (_should_unload_ocr_after_chunk).

balanced mode used to tear down the OCR engine after every chunk, reloading
weights (vllm-mlx 30-60s) on each chunk of a batch. The only balanced consumer
of that freed VRAM is a *local* LLM server; with a cloud/remote LLM (the
default) the unload is pure thrash, so OCR now stays resident across chunks.
"""

import pytest

from bibr.config import Settings
from bibr.pipeline.stages.ocr import _should_unload_ocr_after_chunk


@pytest.fixture(autouse=True)
def _override_auto(monkeypatch):
    # Exercise the heuristic itself unless a test overrides the knob.
    monkeypatch.setattr(Settings.ocr, "unload_between_chunks", "auto", raising=False)


def test_balanced_cloud_keeps_ocr_loaded():
    assert _should_unload_ocr_after_chunk("balanced", "cloud") is False


def test_balanced_local_unloads_ocr():
    assert _should_unload_ocr_after_chunk("balanced", "local") is True


def test_balanced_resolved_local_backends_unload_ocr():
    # The CLI resolves --llm local to a concrete backend before RunConfig,
    # so the heuristic must match the resolved names, not just the alias.
    assert _should_unload_ocr_after_chunk("balanced", "vllm") is True
    assert _should_unload_ocr_after_chunk("balanced", "vllm-mlx") is True


def test_aggressive_always_unloads():
    assert _should_unload_ocr_after_chunk("aggressive", "cloud") is True
    assert _should_unload_ocr_after_chunk("aggressive", "local") is True


def test_keep_all_never_unloads():
    assert _should_unload_ocr_after_chunk("keep_all", "cloud") is False
    assert _should_unload_ocr_after_chunk("keep_all", "local") is False


def test_override_always_forces_unload(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "unload_between_chunks", "always", raising=False)
    assert _should_unload_ocr_after_chunk("balanced", "cloud") is True
    assert _should_unload_ocr_after_chunk("keep_all", "local") is True


def test_override_never_keeps_loaded(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "unload_between_chunks", "never", raising=False)
    assert _should_unload_ocr_after_chunk("aggressive", "local") is False
    assert _should_unload_ocr_after_chunk("balanced", "local") is False
