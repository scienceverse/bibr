"""Pin the vectorised containment filter against the legacy triple-loop
semantics. Boxes are NumPy arrays whose columns are
``[class_id, score, x1, y1, x2, y2]``."""

import numpy as np


def _legacy_filter_containment(boxes, id2label):
    """The pre-vectorised reference implementation, kept here as the
    correctness oracle."""
    from bibr.layout_utils import (
        _MERGE_BBOXES_MODE,
        _PRESERVE_LABELS,
        _is_contained,
    )

    n = len(boxes)
    keep = np.ones(n, dtype=bool)
    unique_ids = {int(boxes[i, 0]) for i in range(n)}
    for cat_id in unique_ids:
        cat_name = id2label.get(cat_id, "")
        mode = _MERGE_BBOXES_MODE.get(cat_name, "large")
        if mode == "union":
            continue
        contains_other = np.zeros(n, dtype=int)
        contained_by_other = np.zeros(n, dtype=int)
        for i in range(n):
            box_label = id2label.get(int(boxes[i, 0]), "")
            if box_label in _PRESERVE_LABELS:
                continue
            for j in range(n):
                if i == j:
                    continue
                if mode == "large" and int(boxes[j, 0]) == cat_id:
                    if _is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
                elif (
                    mode == "small"
                    and int(boxes[i, 0]) == cat_id
                    and _is_contained(boxes[i], boxes[j])
                ):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
        if mode == "large":
            keep &= contained_by_other == 0
        elif mode == "small":
            keep &= (contains_other == 0) | (contained_by_other == 1)
    return keep


def test_filter_containment_matches_legacy_random():
    """Vectorised version must produce identical results to the legacy loop
    across a range of random layouts."""
    from bibr.layout_utils import _filter_containment

    rng = np.random.default_rng(seed=12345)
    id2label = {0: "text", 1: "table", 2: "figure", 3: "title"}
    for _ in range(20):
        n = int(rng.integers(0, 30))
        if n == 0:
            boxes = np.empty((0, 6))
        else:
            classes = rng.integers(0, 4, size=n)
            scores = rng.uniform(0.5, 1.0, size=n)
            x1 = rng.uniform(0, 800, size=n)
            y1 = rng.uniform(0, 1000, size=n)
            w = rng.uniform(20, 400, size=n)
            h = rng.uniform(20, 400, size=n)
            boxes = np.stack([classes, scores, x1, y1, x1 + w, y1 + h], axis=1)
        legacy = _legacy_filter_containment(boxes, id2label)
        new = _filter_containment(boxes, id2label)
        np.testing.assert_array_equal(legacy, new)


def test_filter_containment_empty():
    from bibr.layout_utils import _filter_containment

    keep = _filter_containment(np.empty((0, 6)), {0: "text"})
    assert keep.shape == (0,)


def test_filter_containment_drops_contained_text():
    """A small text box fully inside another text box of the same class is
    dropped under ``large`` mode."""
    from bibr.layout_utils import _filter_containment

    boxes = np.array(
        [
            [0, 0.9, 0, 0, 100, 100],  # outer text
            [0, 0.8, 10, 10, 50, 50],  # inner text — should drop
        ]
    )
    keep = _filter_containment(boxes, {0: "text"})
    assert keep.tolist() == [True, False]
