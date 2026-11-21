"""Concurrency, batching, and length-sorting regressions for CPU ONNX and NumPy GBM models."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from bibr.utils.locks import LOCAL_INFERENCE_LOCK


def _lock_free_from_other_thread() -> bool:
    result: dict[str, bool] = {}

    def probe() -> None:
        acquired = LOCAL_INFERENCE_LOCK.acquire(blocking=False)
        result["acquired"] = acquired
        if acquired:
            LOCAL_INFERENCE_LOCK.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return result["acquired"]


class _FakeOrtSession:
    def __init__(self, *providers: str) -> None:
        self._providers = list(providers)

    def get_providers(self) -> list[str]:
        return self._providers


def test_classifier_resources_forward_unlocks_cpu_onnx_and_locks_cuda() -> None:
    from bibr.pipeline.classifier_resources import _locked_forward

    seen: dict[str, bool] = {}

    class _CpuModel:
        session = _FakeOrtSession("CPUExecutionProvider")

        def classify_batch(self, items):
            seen["cpu_free"] = _lock_free_from_other_thread()
            return list(items)

    class _CudaModel:
        session = _FakeOrtSession("CUDAExecutionProvider", "CPUExecutionProvider")

        def classify_batch(self, items):
            seen["cuda_free"] = _lock_free_from_other_thread()
            return list(items)

    assert _locked_forward(_CpuModel(), ["a"]) == ["a"]
    assert _locked_forward(_CudaModel(), ["b"]) == ["b"]
    assert seen == {"cpu_free": True, "cuda_free": False}


async def test_section_and_paper_classifiers_unlock_cpu_onnx(monkeypatch) -> None:
    from bibr.structure import paper_classifier as pc_mod
    from bibr.structure import section_classifier as sc_mod

    seen: dict[str, bool] = {}

    class _CpuSection:
        session = _FakeOrtSession("CPUExecutionProvider")

        def classify_batch(self, items):
            seen["sec_free"] = _lock_free_from_other_thread()
            return []

    class _CpuPaper:
        session = _FakeOrtSession("CPUExecutionProvider")

        def classify_batch(self, items):
            seen["paper_free"] = _lock_free_from_other_thread()
            return []

    monkeypatch.setattr(sc_mod, "_model_cache", _CpuSection())
    monkeypatch.setattr(pc_mod, "_paper_model_cache", _CpuPaper())

    assert await sc_mod._classify_trained_batch([object()]) == []
    assert await pc_mod.classify_paper_async("Title", "Abstract") is None
    assert seen == {"sec_free": True, "paper_free": True}


def test_pure_numpy_geom_segmenter_unlocks_inference_lock(monkeypatch) -> None:
    from bibr.extract import ref_extractor as re_mod
    from bibr.paper_contents import PaperContents
    from bibr.utils.gbm_bundle import TreeEnsemble

    seen: dict[str, bool] = {}
    tiny_tree = TreeEnsemble(
        {
            "baseline": np.array([0.0], dtype=np.float64),
            "feature_idx": np.array([0], dtype=np.int64),
            "threshold": np.array([0.0], dtype=np.float64),
            "missing_go_to_left": np.array([True]),
            "left": np.array([0], dtype=np.int64),
            "right": np.array([0], dtype=np.int64),
            "is_leaf": np.array([True]),
            "value": np.array([0.0], dtype=np.float64),
            "tree_offsets": np.array([0, 1], dtype=np.int64),
            "tree_class": np.array([0], dtype=np.int64),
        },
        classes=[0, 1],
        loss="binomial",
        n_features=1,
    )

    class _NumpyGeom:
        model = tiny_tree

        def segment_spans(self, ref_text, lines):
            seen["free"] = _lock_free_from_other_thread()
            return [(0, len(ref_text))], 1.0, 1, 1

    monkeypatch.setattr(re_mod, "_get_geom_segmenter", lambda *_a: _NumpyGeom())
    contents = mock.Mock(spec=PaperContents)
    contents.region_summaries = []
    contents.processing_warnings = []
    contents.ref_line_geometry = [object()]
    ext = re_mod.ReferenceExtractor(contents, llm_client=mock.Mock())
    ext._segment_geom("Smith, J. (2020). Title. Journal, 1, 1-2.")
    assert seen == {"free": True}


def test_section_classifier_length_sorted_multi_chunk_matches_input_order() -> None:
    from bibr.structure.section_classifier_common import HeaderContext, _coerce_header_context
    from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel
    from tests import test_onnx_runtime_core as core

    model = OnnxSectionClassifierModel(core._bundle("section"), "cpu")
    # Build > 64 items with strongly alternating lengths to force multiple length-sorted chunks.
    items = []
    for i in range(80):
        base = _coerce_header_context(core.CONTEXTS[i % len(core.CONTEXTS)])
        body = (base.body + " ") * (8 if i % 3 == 0 else 1)
        items.append(
            HeaderContext(
                heading=base.heading,
                body=body,
                relative_position=base.relative_position,
                prev_heading=base.prev_heading,
                next_heading=base.next_heading,
            )
        )
    batched = model.classify_batch(items, max_length=32)
    singles = [model.classify_batch([item], max_length=32)[0] for item in items]
    assert len(batched) == len(singles)
    for got, want in zip(batched, singles, strict=True):
        assert got.canonical_type == want.canonical_type
        assert got.is_top_level == want.is_top_level
        assert got.score == pytest.approx(want.score, abs=1e-5)


def test_text_segmenter_batched_windows_match_batch1(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    from bibr.extract.text_segmenter import OnnxTextSegmenter
    from tests.extract.test_text_segmenter import _bundle

    # 3 full windows + 1 short tail
    bdir = _bundle(tmp_path / "b", window=8, stride=6, slope=(0.0, 0.05, -0.02))
    seg = OnnxTextSegmenter(bdir)
    text = "@ a b c d e @ f g h i j @ k l m n o @ p q r s t u v"

    tags_batched, probs_batched, offsets_batched, n_batched = seg.tag(text)

    # Compare against forced batch-1 path.
    seg._dynamic_batch = False
    tags_seq, probs_seq, offsets_seq, n_seq = seg.tag(text)

    assert n_batched == n_seq and n_batched >= 3
    assert offsets_batched == offsets_seq
    np.testing.assert_array_equal(tags_batched, tags_seq)
    np.testing.assert_allclose(probs_batched, probs_seq, atol=1e-6)


def test_statement_tagger_probe_false_skips_probe_and_runs_concurrently(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    from bibr.statement_tagger.runtime import StatementTagger
    from tests.statement_tagger.bundle import write_bundle

    # Bad n_out=3 graph fails when probe=True, opens when probe=False (default bundle fast path).
    bad = write_bundle(tmp_path / "bad", n_out=3)
    with pytest.raises(ValueError):
        StatementTagger(bad, probe=True)
    StatementTagger(bad, probe=False)

    # Good bundle runs concurrently across threads with identical results.
    good = write_bundle(tmp_path / "good")
    tagger = StatementTagger(good, threads=1)
    blocks = ["Funding", "Supported by NSF grant 123.", "No conflicts of interest."] * 8
    ref = tagger.probabilities(blocks)

    results: list[np.ndarray] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(tagger.probabilities(blocks))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 4
    for out in results:
        np.testing.assert_allclose(out, ref, atol=1e-6)


def test_geom_tagger_probe_flag(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    from bibr.exceptions import ConfigurationError
    from bibr.extract.geom_tagger import GeomLineTagger
    from tests.extract.test_geom_tagger import _bundle

    good = _bundle(tmp_path / "good")
    tagger = GeomLineTagger(good, probe=False)
    assert tagger.threshold >= 0.0
    # Manifest/shape still checked even with probe=False.
    (good / "gate.json").write_text(json.dumps({"threshold": "bad"}))
    with pytest.raises(ConfigurationError):
        GeomLineTagger(good, probe=False)
