"""PP-DocLayoutV4 postprocessing in numpy: quads, reading-order decode, reference parity.

The reference is ``PPDocLayoutV4ImageProcessor.post_process_object_detection``
from huggingface/transformers#48387. Until a transformers release ships it, the
recorded fixture below is what pins bibr's port to it; the live comparison
runs whenever the installed transformers has the class.

Regenerate the fixture with a transformers that has PP-DocLayoutV4::

    python tests/test_layout_v4_decode.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bibr.exceptions import ConfigurationError
from bibr.layout_base import _check_label_map
from bibr.layout_onnx import decode_detections_v4, decode_reading_order, quad_corners
from bibr.layout_utils import _CORRECT_ID2LABEL

FIXTURE = Path(__file__).parent / "fixtures" / "layout" / "pp_doclayout_v4_reference_decode.json"
SEEDS = range(16)


def v4_case(seed: int):
    """Deterministic raw V4 outputs (legacy ``RandomState``: stable across numpy versions).

    Covers one-box pages, single-label and 25-label heads, sparse and dense
    successor graphs (dense ones are full of cycles), exact Borda ties and the
    three thresholds bibr and the reference use.
    """
    rng = np.random.RandomState(seed)
    batch = 1 + seed % 2
    queries = (1, 2, 6, 12, 24, 30)[seed % 6]
    classes = 25 if seed % 3 else 4
    logits = rng.normal(0, 2.5, size=(batch, queries, classes)).astype(np.float32)
    boxes = rng.uniform(0.2, 0.8, size=(batch, queries, 10)).astype(np.float32)
    relative = rng.normal(0, 3, size=(batch, queries, queries)).astype(np.float32)
    relative = ((relative - relative.transpose(0, 2, 1)) / 2).astype(np.float32)
    if seed % 4 == 0:
        relative = np.round(relative)
    density = (0.05, 0.15, 0.5)[seed % 3]
    positive = rng.uniform(size=(batch, queries, queries)) < density
    successor = np.where(
        positive,
        rng.uniform(0.1, 5, size=(batch, queries, queries)),
        -rng.uniform(0.1, 5, size=(batch, queries, queries)),
    ).astype(np.float32)
    diagonal = np.arange(queries)
    successor[:, diagonal, diagonal] = -1e4
    sizes = [(int(rng.randint(500, 3000)), int(rng.randint(400, 2500))) for _ in range(batch)]
    threshold = (0.3, 0.5, 0.05)[seed % 3]
    return logits, boxes, relative, successor, sizes, threshold


def _as_record(result: dict) -> dict:
    return {
        "labels": [int(v) for v in result["labels"]],
        "order_seq": [int(v) for v in result["order_seq"]],
        "scores": [round(float(v), 6) for v in result["scores"]],
        "boxes": [[round(float(v), 2) for v in box] for box in result["boxes"]],
    }


# -- reading order ------------------------------------------------------------


def _no_edges(n: int) -> np.ndarray:
    return np.full((n, n), -3.0, dtype=np.float32)


def test_successor_chain_sets_the_order():
    successor = _no_edges(4)
    for i, j in ((2, 0), (0, 3), (3, 1)):
        successor[i, j] = 2.0
    ranks = decode_reading_order(np.zeros((4, 4), np.float32), successor)
    assert ranks.tolist() == [1, 3, 0, 2]


def test_a_cycle_loses_its_weakest_edge():
    successor = _no_edges(3)
    successor[0, 1], successor[1, 2], successor[2, 0] = 3.0, 2.0, 0.5
    ranks = decode_reading_order(np.zeros((3, 3), np.float32), successor)
    assert ranks.tolist() == [0, 1, 2]
    successor[1, 2] = 0.1  # now 1 -> 2 is the weakest: 2, 0, 1
    assert decode_reading_order(np.zeros((3, 3), np.float32), successor).tolist() == [1, 2, 0]


def test_components_follow_their_mean_relative_vote():
    """Two unlinked chains: the one the relative head reads first comes first."""
    successor = _no_edges(4)
    successor[0, 1] = successor[2, 3] = 2.0
    relative = np.zeros((4, 4), np.float32)
    for early in (2, 3):
        for late in (0, 1):
            relative[early, late], relative[late, early] = 5.0, -5.0
    assert decode_reading_order(relative, successor).tolist() == [2, 3, 0, 1]


def test_borda_ties_go_to_the_smaller_index():
    successor = _no_edges(3)
    successor[0, 2] = successor[1, 2] = 2.0  # 0 and 1 both precede 2
    assert decode_reading_order(np.zeros((3, 3), np.float32), successor).tolist() == [0, 1, 2]
    relative = np.zeros((3, 3), np.float32)
    relative[1, 0], relative[0, 1] = 2.0, -2.0  # 1 reads before 0
    assert decode_reading_order(relative, successor).tolist() == [1, 0, 2]


def test_nan_relative_scores_do_not_fail_the_page():
    """fp16 overflow can turn the relative head into NaN; the reference raises there."""
    successor = _no_edges(3)
    successor[0, 2] = successor[1, 2] = 2.0
    relative = np.full((3, 3), np.nan, dtype=np.float32)
    assert decode_reading_order(relative, successor).tolist() == [0, 1, 2]


def test_trivial_pages():
    assert decode_reading_order(np.zeros((0, 0), np.float32), np.zeros((0, 0))).tolist() == []
    assert decode_reading_order(np.zeros((1, 1), np.float32), np.zeros((1, 1))).tolist() == [0]


# -- quads and the decode -----------------------------------------------------


def test_quad_corners_are_center_plus_shifted_offsets():
    quad = np.array([0.5, 0.5, 0.4, 0.4, 0.6, 0.45, 0.6, 0.6, 0.4, 0.55], dtype=np.float32)
    corners = quad_corners(quad[None])[0]
    np.testing.assert_allclose(
        corners, [[0.4, 0.4], [0.6, 0.45], [0.6, 0.6], [0.4, 0.55]], atol=1e-6
    )
    with pytest.raises(ValueError, match="10 coordinates"):
        quad_corners(np.zeros((1, 4), np.float32))


def test_boxes_enclose_the_quad_and_a_repeated_query_keeps_one_rank():
    # Three queries, so the flat top-k (k = queries) has room for query 0
    # under two labels plus query 1; query 2 stays below the threshold.
    logits = np.full((1, 3, 3), -9.0, dtype=np.float32)
    logits[0, 0, 0] = logits[0, 0, 2] = 3.0
    logits[0, 1, 1] = 2.0
    boxes = np.full((1, 3, 10), 0.5, dtype=np.float32)
    boxes[0, 0] = [0.5, 0.5, 0.4, 0.4, 0.6, 0.45, 0.6, 0.6, 0.4, 0.55]
    boxes[0, 1] = [0.3, 0.8, 0.45, 0.45, 0.55, 0.45, 0.55, 0.55, 0.45, 0.55]
    successor = _no_edges(3)
    successor[1, 0] = 4.0  # query 1 directly precedes query 0
    (result,) = decode_detections_v4(
        logits, boxes, np.zeros((1, 3, 3), np.float32), successor[None], [(1000, 500)], 0.5
    )
    assert result["order_seq"].tolist() == [0, 1, 1]
    assert result["labels"].tolist()[0] == 1
    assert sorted(result["labels"].tolist()[1:]) == [0, 2]
    np.testing.assert_allclose(result["boxes"][0], [125, 750, 175, 850], atol=1e-3)
    np.testing.assert_allclose(result["boxes"][1], [200, 400, 300, 600], atol=1e-3)
    np.testing.assert_allclose(result["boxes"][1], result["boxes"][2])
    assert result["polygon_points"].shape == (3, 4, 2)


def test_nothing_above_threshold_is_an_empty_page():
    logits = np.full((1, 3, 2), -9.0, dtype=np.float32)
    (result,) = decode_detections_v4(
        logits,
        np.full((1, 3, 10), 0.5, np.float32),
        np.zeros((1, 3, 3), np.float32),
        np.zeros((1, 3, 3), np.float32),
        [(100, 100)],
        0.3,
    )
    assert all(len(v) == 0 for v in result.values())
    assert result["boxes"].shape == (0, 4)


@pytest.mark.parametrize("seed", SEEDS)
def test_decode_matches_the_recorded_reference(seed):
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"][str(seed)]
    logits, boxes, relative, successor, sizes, threshold = v4_case(seed)
    ours = decode_detections_v4(logits, boxes, relative, successor, sizes, threshold)
    assert len(ours) == len(expected)
    for got, want in zip(ours, expected, strict=True):
        assert got["order_seq"].tolist() == want["order_seq"]
        assert got["labels"].tolist() == want["labels"]
        np.testing.assert_allclose(got["scores"], want["scores"], atol=2e-6)
        np.testing.assert_allclose(
            got["boxes"].reshape(-1, 4), want["boxes"] or np.zeros((0, 4)), atol=0.01
        )


def test_decode_matches_the_live_reference():
    """Same comparison against the installed processor, when there is one."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    proc_cls = getattr(transformers, "PPDocLayoutV4ImageProcessor", None)
    if proc_cls is None:
        pytest.skip("transformers without PPDocLayoutV4")
    from tests import onnx_fixtures as fx

    proc = proc_cls()
    for seed in range(40):
        logits, boxes, relative, successor, sizes, threshold = v4_case(seed)
        reference = proc.post_process_object_detection(
            fx.fake_hf_v4_outputs(logits, boxes, relative, successor),
            threshold=threshold,
            target_sizes=sizes,
        )
        ours = decode_detections_v4(logits, boxes, relative, successor, sizes, threshold)
        for ref, got in zip(reference, ours, strict=True):
            np.testing.assert_array_equal(got["order_seq"], ref["order_seq"].numpy())
            np.testing.assert_array_equal(got["labels"], ref["labels"].numpy())
            np.testing.assert_allclose(got["scores"], ref["scores"].numpy(), atol=1e-6)
            np.testing.assert_allclose(got["boxes"], ref["boxes"].numpy(), atol=1e-3)
            np.testing.assert_allclose(
                got["polygon_points"], ref["polygon_points"].numpy(), atol=1e-3
            )


# -- label map ------------------------------------------------------------------


def test_label_map_check_accepts_bibrs_labels_and_refuses_others():
    _check_label_map(None, "a V3 bundle")
    _check_label_map({str(k): v for k, v in _CORRECT_ID2LABEL.items()}, "a V4 manifest")
    swapped = dict(_CORRECT_ID2LABEL)
    swapped[18], swapped[19] = swapped[19], swapped[18]
    with pytest.raises(ConfigurationError, match=r"ids \[18, 19\]"):
        _check_label_map(swapped, "a reordered checkpoint")
    with pytest.raises(ConfigurationError):
        _check_label_map({**_CORRECT_ID2LABEL, 25: "stamp"}, "an extended checkpoint")


def _record() -> None:
    """Rewrite the fixture from the installed transformers' V4 processor."""
    import transformers

    from tests import onnx_fixtures as fx

    proc = transformers.PPDocLayoutV4ImageProcessor()
    cases = {}
    for seed in SEEDS:
        logits, boxes, relative, successor, sizes, threshold = v4_case(seed)
        reference = proc.post_process_object_detection(
            fx.fake_hf_v4_outputs(logits, boxes, relative, successor),
            threshold=threshold,
            target_sizes=sizes,
        )
        cases[str(seed)] = [
            _as_record({k: v.numpy() for k, v in result.items()}) for result in reference
        ]
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": "PPDocLayoutV4ImageProcessor.post_process_object_detection from "
        f"transformers {transformers.__version__} (huggingface/transformers#48387)",
        "inputs": "tests/test_layout_v4_decode.py::v4_case(seed)",
        "cases": cases,
    }
    FIXTURE.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")


if __name__ == "__main__":
    _record()
