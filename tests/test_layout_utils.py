"""Characterization tests for layout bbox utilities.

These functions currently live in bibr/serve/deployments/layout.py but
will move to bibr/layout_utils.py. The tests are written against the
NEW import path so they fail until the move is done.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_iou_disjoint_zero():
    from bibr.layout_utils import _iou

    a = np.array([0, 0, 10, 10])
    b = np.array([20, 20, 30, 30])
    assert _iou(a, b) == 0


def test_iou_identical_one():
    from bibr.layout_utils import _iou

    a = np.array([0, 0, 10, 10])
    assert _iou(a, a) == pytest.approx(1.0, abs=0.01)


def test_iou_partial():
    from bibr.layout_utils import _iou

    a = np.array([0, 0, 10, 10])
    b = np.array([5, 5, 15, 15])
    # _iou uses +1 adjustments: inter = 6*6 = 36; area = 11*11 = 121;
    # union = 121 + 121 - 36 = 206; IoU = 36/206 ~ 0.175.
    assert 0.10 < _iou(a, b) < 0.25


def test_nms_drops_overlapping_same_class():
    """Same-class boxes with high overlap merge; cross-class disjoint boxes survive."""
    from bibr.layout_utils import _nms

    boxes = np.array(
        [
            [0, 0.9, 0, 0, 10, 10],  # cls 0, score 0.9
            [0, 0.5, 1, 1, 11, 11],  # cls 0, large overlap with box 0 → drop
            [1, 0.6, 50, 50, 60, 60],  # cls 1, disjoint → kept
        ]
    )
    keep = _nms(boxes, iou_same=0.5, iou_diff=0.99)
    assert 0 in keep
    assert 1 not in keep
    assert 2 in keep


def test_compute_read_order_top_to_bottom_left_to_right():
    from bibr.layout_utils import _compute_read_order

    boxes = np.array(
        [
            [0, 0.9, 100, 200, 200, 250],  # row 2
            [0, 0.9, 0, 0, 100, 50],  # row 0, col 0
            [0, 0.9, 200, 0, 300, 50],  # row 0, col 2
            [0, 0.9, 100, 0, 200, 50],  # row 0, col 1
        ]
    )
    order = _compute_read_order(boxes)
    # Box 1 (top-left) ranks first; box 0 (bottom row) ranks last.
    assert order[1] == 0
    assert order[0] == 3


def test_filter_containment_drops_inner_text_inside_text():
    from bibr.layout_utils import _filter_containment

    # cls 22 = text (mode "large"). Box B fully inside A → drop B.
    boxes = np.array(
        [
            [22, 0.9, 0, 0, 100, 100],
            [22, 0.9, 10, 10, 50, 50],  # contained
        ],
        dtype=float,
    )
    id2label = {22: "text"}
    keep = _filter_containment(boxes, id2label)
    assert keep[0]  # large kept
    assert not keep[1]  # smaller dropped


def test_label_task_mapping_present():
    """LABEL_TASK_MAPPING is exposed via the new module."""
    from bibr.layout_utils import LABEL_TASK_MAPPING

    assert LABEL_TASK_MAPPING["table"] == ["table"]
    assert "text" in LABEL_TASK_MAPPING


def test_correct_id2label_is_25_classes():
    from bibr.layout_utils import _CORRECT_ID2LABEL

    assert len(_CORRECT_ID2LABEL) == 25
    assert _CORRECT_ID2LABEL[22] == "text"


def test_filter_containment_collapses_coincident_reference_pair():
    """A lone reference co-emitted as reference(18)+reference_content(19) over
    the same extent must NOT mutually annihilate — keep the higher-scored
    reference_content, drop the reference."""
    from bibr.layout_utils import _filter_containment

    boxes = np.array(
        [
            [18, 0.65, 100, 100, 800, 200],  # reference (low conf)
            [19, 0.96, 100, 100, 800, 200],  # reference_content (high conf), coincident
        ],
        dtype=float,
    )
    id2label = {18: "reference", 19: "reference_content"}
    keep = _filter_containment(boxes, id2label)
    assert not keep[0]  # reference dropped
    assert keep[1]  # reference_content survives


def test_filter_containment_keeps_nested_reference_and_preserved_container():
    """A small-mode reference cannot mutually remove a preserved container."""
    from bibr.layout_utils import _filter_containment

    boxes = np.array(
        [
            [14, 0.90, 0, 0, 100, 100],  # image is always preserved
            [18, 0.95, 10, 10, 90, 90],  # reference is small-mode
        ],
        dtype=float,
    )

    keep = _filter_containment(boxes, {14: "image", 18: "reference"})

    assert keep.all()


def test_filter_containment_keeps_healthy_reference_hierarchy():
    """A real reference envelope over many reference_content items is NOT mutual
    (items sit inside the envelope but the envelope does not sit inside any
    item) → untouched."""
    from bibr.layout_utils import _filter_containment

    boxes = np.array(
        [
            [18, 0.90, 100, 100, 800, 900],  # reference envelope
            [19, 0.95, 100, 100, 800, 300],  # item 1
            [19, 0.95, 100, 300, 800, 600],  # item 2
            [19, 0.95, 100, 600, 800, 900],  # item 3
        ],
        dtype=float,
    )
    id2label = {18: "reference", 19: "reference_content"}
    keep = _filter_containment(boxes, id2label)
    assert keep.all()  # envelope + all items retained (unchanged behavior)


def test_filter_containment_collapses_generic_crossclass_pair():
    """Any coincident cross-class pair collapses to the higher-scored box."""
    from bibr.layout_utils import _filter_containment

    boxes = np.array(
        [
            [22, 0.70, 100, 100, 800, 200],  # text (lower)
            [0, 0.90, 100, 100, 800, 200],  # abstract (higher), coincident
        ],
        dtype=float,
    )
    id2label = {22: "text", 0: "abstract"}
    keep = _filter_containment(boxes, id2label)
    assert not keep[0]  # text dropped
    assert keep[1]  # abstract kept


def test_filter_containment_collapse_never_drops_preserve_label():
    """When the lower-scored box of a coincident pair is a preserve label, the
    non-preserve box is dropped instead."""
    from bibr.layout_utils import _filter_containment

    boxes = np.array(
        [
            [14, 0.60, 100, 100, 800, 200],  # image (preserve, lower score)
            [22, 0.90, 100, 100, 800, 200],  # text (higher score), coincident
        ],
        dtype=float,
    )
    id2label = {14: "image", 22: "text"}
    keep = _filter_containment(boxes, id2label)
    assert keep[0]  # image (preserve) kept
    assert not keep[1]  # text dropped


# ---------------------------------------------------------------------------
# Docling-rulebook overlap resolver (_resolve_overlaps_rulebook)
# ---------------------------------------------------------------------------

_RB_ID2LABEL = {
    14: "image",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    22: "text",
}


def test_rulebook_coincident_crossclass_pair_exactly_one_survivor():
    """A coincident cross-class pair is one entity: exactly one survivor
    (the higher-scored box), never zero."""
    from bibr.layout_utils import _resolve_overlaps_rulebook

    boxes = np.array(
        [
            [18, 0.65, 100, 100, 800, 200],  # reference (low conf)
            [19, 0.96, 100, 100, 800, 200],  # reference_content (high conf), coincident
        ],
        dtype=float,
    )
    keep, out = _resolve_overlaps_rulebook(boxes, _RB_ID2LABEL)
    assert keep.sum() == 1
    assert keep[1] and not keep[0]
    assert out[1, 2:6].tolist() == [100, 100, 800, 200]  # coincident union = same box


def test_rulebook_large_mode_winner_absorbs_nested_loser_to_union():
    """A "large"-mode winner absorbs a removed loser's bbox: no page area
    (hence no OCR text) may be lost."""
    from bibr.layout_utils import _resolve_overlaps_rulebook

    boxes = np.array(
        [
            [22, 0.95, 200, 200, 400, 400],  # inner text, higher score → winner
            [22, 0.60, 100, 100, 500, 500],  # outer text, absorbed
        ],
        dtype=float,
    )
    keep, out = _resolve_overlaps_rulebook(boxes, _RB_ID2LABEL)
    assert keep.tolist() == [True, False]
    assert out[0, 2:6].tolist() == [100, 100, 500, 500]  # expanded to union


def test_rulebook_small_mode_winner_keeps_tight_bbox():
    """A "small"-mode (reference) winner never expands: the coincident loser is
    dropped but the winner's tight bbox is untouched."""
    from bibr.layout_utils import _resolve_overlaps_rulebook

    boxes = np.array(
        [
            [18, 0.95, 100, 100, 800, 200],  # reference, higher score → winner
            [22, 0.60, 100, 100, 800, 200],  # text, coincident → dropped
        ],
        dtype=float,
    )
    keep, out = _resolve_overlaps_rulebook(boxes, _RB_ID2LABEL)
    assert keep.tolist() == [True, False]
    assert out[0, 2:6].tolist() == [100, 100, 800, 200]  # tight bbox unchanged


def test_small_mode_keeps_nested_reference_without_annihilating_both_boxes():
    """Legacy containment keeps the reference; rulebook also preserves the text area."""
    from bibr.layout_utils import _filter_containment, _resolve_overlaps_rulebook

    boxes = np.array(
        [
            [18, 0.95, 300, 200, 700, 300],  # reference item (winner)
            [22, 0.90, 100, 100, 900, 500],  # text container, <80% covered by winner
        ],
        dtype=float,
    )
    assert _filter_containment(boxes, _RB_ID2LABEL).tolist() == [True, False]
    keep, out = _resolve_overlaps_rulebook(boxes, _RB_ID2LABEL)
    assert keep.tolist() == [True, True]
    np.testing.assert_array_equal(out, boxes)  # no bbox mutated


def test_rulebook_preserve_label_never_loses_to_higher_score():
    """A preserve-label box wins its group even against a higher-scored
    non-preserve box."""
    from bibr.layout_utils import _resolve_overlaps_rulebook

    boxes = np.array(
        [
            [14, 0.60, 100, 100, 800, 200],  # image (preserve, lower score)
            [22, 0.90, 100, 100, 800, 200],  # text (higher score), coincident
        ],
        dtype=float,
    )
    keep, out = _resolve_overlaps_rulebook(boxes, _RB_ID2LABEL)
    assert keep.tolist() == [True, False]
    assert out[0, 2:6].tolist() == [100, 100, 800, 200]


def test_rulebook_transitive_chain_resolved_via_union_find():
    """A~B and B~C overlap but A and C are disjoint: union-find still groups all
    three, one winner survives and absorbs the whole chain's extent."""
    from bibr.layout_utils import _resolve_overlaps_rulebook

    boxes = np.array(
        [
            [22, 0.90, 0, 0, 100, 100],  # A: coincident-ish with B (80% mutual)
            [22, 0.50, 0, 0, 125, 100],  # B: contains A and C
            [22, 0.95, 100, 0, 125, 100],  # C: inside B, disjoint from A → winner
        ],
        dtype=float,
    )
    keep, out = _resolve_overlaps_rulebook(boxes, _RB_ID2LABEL)
    assert keep.tolist() == [False, False, True]
    assert out[2, 2:6].tolist() == [0, 0, 125, 100]  # union of the whole chain


def test_rulebook_keeps_healthy_reference_hierarchy():
    """reference_content items inside a reference envelope create no mode
    conflict: no edges, everything survives untouched (the naive Docling port
    would collapse the group to one box)."""
    from bibr.layout_utils import _resolve_overlaps_rulebook

    boxes = np.array(
        [
            [18, 0.90, 100, 100, 800, 900],  # reference envelope
            [19, 0.95, 100, 100, 800, 300],  # item 1
            [19, 0.95, 100, 300, 800, 600],  # item 2
            [19, 0.95, 100, 600, 800, 900],  # item 3
        ],
        dtype=float,
    )
    keep, out = _resolve_overlaps_rulebook(boxes, _RB_ID2LABEL)
    assert keep.all()
    np.testing.assert_array_equal(out, boxes)


def test_rulebook_unrelated_leftovers_are_not_cross_merged():
    """Two texts grouped only through a shared "small" winner (a reference in
    their overlap strip) must not absorb each other once the winner is
    resolved: leftovers are regrouped by their own connectivity."""
    from bibr.layout_utils import _resolve_overlaps_rulebook

    boxes = np.array(
        [
            [18, 0.99, 400, 400, 500, 500],  # reference, inside both texts → winner
            [22, 0.90, 0, 0, 500, 1000],  # text A (left column)
            [22, 0.80, 400, 0, 1000, 1000],  # text B (right column), no A~B edge
        ],
        dtype=float,
    )
    keep, out = _resolve_overlaps_rulebook(boxes, _RB_ID2LABEL)
    # Winner keeps tight; both under-covered texts survive, unmerged.
    assert keep.all()
    np.testing.assert_array_equal(out, boxes)


def test_rulebook_random_invariants_vs_legacy():
    """Differential fuzz: the rulebook never returns zero regions when legacy
    returns >= 1 (in fact never zero for non-empty input), preserved boxes
    always survive, and every dropped box is >= 80% covered by some survivor's
    final bbox — no page area silently lost."""
    from bibr.layout_utils import _PRESERVE_LABELS, _filter_containment, _resolve_overlaps_rulebook

    rng = np.random.default_rng(seed=20260704)
    class_ids = np.array([14, 17, 18, 19, 22])
    for _ in range(40):
        n = int(rng.integers(1, 25))
        classes = rng.choice(class_ids, size=n)
        scores = rng.uniform(0.3, 1.0, size=n)
        x1 = rng.uniform(0, 800, size=n)
        y1 = rng.uniform(0, 800, size=n)
        w = rng.uniform(20, 500, size=n)
        h = rng.uniform(20, 500, size=n)
        boxes = np.stack([classes, scores, x1, y1, x1 + w, y1 + h], axis=1)

        legacy = _filter_containment(boxes, _RB_ID2LABEL)
        keep, out = _resolve_overlaps_rulebook(boxes, _RB_ID2LABEL)

        assert keep.sum() >= 1  # never zero on non-empty input
        assert keep.sum() >= min(1, int(legacy.sum()))
        preserved = np.array([_RB_ID2LABEL[int(c)] in _PRESERVE_LABELS for c in classes])
        assert keep[preserved].all()  # preserve labels never dropped
        # Survivors may only grow.
        assert (out[keep, 2] <= boxes[keep, 2]).all()
        assert (out[keep, 3] <= boxes[keep, 3]).all()
        assert (out[keep, 4] >= boxes[keep, 4]).all()
        assert (out[keep, 5] >= boxes[keep, 5]).all()
        # Every dropped box is >= 80% covered by some survivor's final bbox.
        for i in np.where(~keep)[0]:
            bx1, by1, bx2, by2 = boxes[i, 2:6]
            area = (bx2 - bx1) * (by2 - by1)
            covered = False
            for j in np.where(keep)[0]:
                ix1 = max(bx1, out[j, 2])
                iy1 = max(by1, out[j, 3])
                ix2 = min(bx2, out[j, 4])
                iy2 = min(by2, out[j, 5])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter / area >= 0.8:
                    covered = True
                    break
            assert covered, f"dropped box {i} not covered by any survivor"


def test_rulebook_empty_input():
    from bibr.layout_utils import _resolve_overlaps_rulebook

    keep, out = _resolve_overlaps_rulebook(np.empty((0, 6)), _RB_ID2LABEL)
    assert keep.shape == (0,)
    assert out.shape == (0, 6)


# ---------------------------------------------------------------------------
# layout_base wiring: Settings.layout.overlap_resolver selects the resolver
# ---------------------------------------------------------------------------


def _bare_detector(settings=None):
    """BaseLayoutDetector instance without __init__ (no model, no torch)."""
    from bibr.config import GlobalSettings
    from bibr.layout_base import BaseLayoutDetector
    from bibr.layout_utils import _CORRECT_ID2LABEL

    det = BaseLayoutDetector.__new__(BaseLayoutDetector)
    det.threshold = 0.3
    det._id2label = _CORRECT_ID2LABEL
    det._settings = settings or GlobalSettings()
    return det


_ANNIHILATION_RESULT = {
    # reference nested in text: legacy containment mutually annihilates the pair.
    "scores": np.array([0.90, 0.95]),
    "labels": np.array([22, 18]),
    "boxes": np.array([[100, 100, 900, 500], [300, 200, 700, 300]], dtype=float),
}


def test_postprocess_legacy_default_keeps_nested_reference():
    """The default legacy resolver must not annihilate a nested reference pair."""
    from bibr.config import Settings

    assert Settings.layout.overlap_resolver == "legacy"
    det = _bare_detector()
    regions = det._postprocess(dict(_ANNIHILATION_RESULT), 1000, 1000)
    assert [region["label"] for region in regions] == ["reference"]


def test_postprocess_rulebook_selected_by_setting():
    """With LAYOUT_OVERLAP_RESOLVER=rulebook the same page keeps both regions."""
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.layout.overlap_resolver = "rulebook"
    det = _bare_detector(settings)
    regions = det._postprocess(dict(_ANNIHILATION_RESULT), 1000, 1000)
    labels = sorted(r["label"] for r in regions)
    assert labels == ["reference", "text"]


# ---------------------------------------------------------------------------
# Column-aware read-order fallback (_compute_read_order_rb)
# ---------------------------------------------------------------------------

# Two-column page with interleaved y-centers: xy-lexsort zigzags across the
# gutter; the rb order must finish the left column before the right one.
_RB_TWO_COL = np.array(
    [
        [22, 0.9, 50, 100, 450, 180],  # 0: left top
        [22, 0.9, 550, 140, 950, 220],  # 1: right top
        [22, 0.9, 50, 200, 450, 280],  # 2: left bottom
        [22, 0.9, 550, 240, 950, 320],  # 3: right bottom
    ],
    dtype=float,
)

# The same two columns under a full-width title, rows deliberately shuffled.
_RB_TITLED_TWO_COL = np.array(
    [
        [6, 0.95, 50, 20, 950, 80],  # 0: doc_title spanning both columns
        [22, 0.9, 550, 140, 950, 220],  # 1: right top
        [22, 0.9, 50, 100, 450, 180],  # 2: left top
        [22, 0.9, 550, 240, 950, 320],  # 3: right bottom
        [22, 0.9, 50, 200, 450, 280],  # 4: left bottom
    ],
    dtype=float,
)


def _reading_sequence(ranks: list[int]) -> list[int]:
    """Invert rank-per-box into the sequence of box indices as read."""
    seq = [0] * len(ranks)
    for idx, rank in enumerate(ranks):
        seq[rank] = idx
    return seq


def test_read_order_rb_two_column_page_column_major():
    """xy-lexsort interleaves the columns; rb reads the left column fully first."""
    from bibr.layout_utils import _compute_read_order, _compute_read_order_rb

    assert _reading_sequence(_compute_read_order(_RB_TWO_COL)) == [0, 1, 2, 3]
    assert _reading_sequence(_compute_read_order_rb(_RB_TWO_COL)) == [0, 2, 1, 3]


def test_read_order_rb_single_column_matches_xy():
    """On a single-column page rb must reproduce the xy order exactly."""
    from bibr.layout_utils import _compute_read_order, _compute_read_order_rb

    boxes = np.array(
        [
            [22, 0.9, 100, 400, 500, 500],
            [17, 0.9, 100, 100, 400, 160],
            [22, 0.9, 100, 180, 500, 380],
            [22, 0.9, 100, 520, 500, 700],
        ],
        dtype=float,
    )
    assert _compute_read_order_rb(boxes) == _compute_read_order(boxes)


def test_read_order_rb_full_width_title_before_columns():
    """Above-title first, then the whole left column, then the right column."""
    from bibr.layout_utils import _compute_read_order_rb

    seq = _reading_sequence(_compute_read_order_rb(_RB_TITLED_TWO_COL))
    assert seq == [0, 2, 4, 1, 3]


def test_read_order_rb_full_width_table_after_columns():
    """A full-width element below two columns comes after BOTH columns (the
    upward-climb keeps the table waiting until the right column is read)."""
    from bibr.layout_utils import _compute_read_order_rb

    boxes = np.array(
        [
            [22, 0.9, 50, 100, 450, 180],  # 0: left top
            [22, 0.9, 50, 200, 450, 280],  # 1: left bottom
            [22, 0.9, 550, 140, 950, 220],  # 2: right top
            [22, 0.9, 550, 240, 950, 320],  # 3: right bottom
            [21, 0.9, 50, 700, 950, 900],  # 4: full-width table below both columns
        ],
        dtype=float,
    )
    seq = _reading_sequence(_compute_read_order_rb(boxes))
    assert seq == [0, 1, 2, 3, 4]


def test_read_order_rb_empty_and_single_box():
    from bibr.layout_utils import _compute_read_order_rb

    assert _compute_read_order_rb(np.empty((0, 6))) == []
    single = np.array([[22, 0.9, 100, 100, 500, 200]], dtype=float)
    assert _compute_read_order_rb(single) == [0]


def test_read_order_rb_deterministic():
    from bibr.layout_utils import _compute_read_order_rb

    first = _compute_read_order_rb(_RB_TITLED_TWO_COL)
    for _ in range(3):
        assert _compute_read_order_rb(_RB_TITLED_TWO_COL.copy()) == first


def test_read_order_rb_random_boxes_yield_a_permutation():
    """Fuzz: ranks are always a permutation of range(n) (contract parity with
    _compute_read_order) and repeated calls agree."""
    from bibr.layout_utils import _compute_read_order_rb

    rng = np.random.default_rng(seed=20260704)
    for _ in range(20):
        n = int(rng.integers(1, 30))
        x1 = rng.uniform(0, 900, size=n)
        y1 = rng.uniform(0, 900, size=n)
        boxes = np.stack(
            [
                np.full(n, 22.0),
                np.full(n, 0.9),
                x1,
                y1,
                x1 + rng.uniform(10, 400, size=n),
                y1 + rng.uniform(10, 200, size=n),
            ],
            axis=1,
        )
        ranks = _compute_read_order_rb(boxes)
        assert sorted(ranks) == list(range(n))
        assert _compute_read_order_rb(boxes.copy()) == ranks


# ---------------------------------------------------------------------------
# layout_base wiring: Settings.layout.read_order_fallback selects the fallback
# ---------------------------------------------------------------------------

_TWO_COL_RESULT = {
    # Two-column page (interleaved y), no order_seq → the fallback path runs.
    "scores": np.array([0.9, 0.9, 0.9, 0.9]),
    "labels": np.array([22, 22, 22, 22]),
    "boxes": np.array(
        [
            [50, 100, 450, 180],
            [550, 140, 950, 220],
            [50, 200, 450, 280],
            [550, 240, 950, 320],
        ],
        dtype=float,
    ),
}


def test_postprocess_read_order_fallback_xy_default():
    """Default fallback stays "xy": the two-column page interleaves
    (characterizes that default behavior did not change)."""
    from bibr.config import Settings

    assert Settings.layout.read_order_fallback == "xy"
    det = _bare_detector()
    regions = det._postprocess(dict(_TWO_COL_RESULT), 1000, 1000)
    assert [r["bbox_2d"][0] for r in regions] == [50, 550, 50, 550]


def test_postprocess_read_order_fallback_rb_selected_by_setting():
    """With LAYOUT_READ_ORDER_FALLBACK=rb the same page reads column-major."""
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.layout.read_order_fallback = "rb"
    det = _bare_detector(settings)
    regions = det._postprocess(dict(_TWO_COL_RESULT), 1000, 1000)
    assert [r["bbox_2d"][0] for r in regions] == [50, 50, 550, 550]
