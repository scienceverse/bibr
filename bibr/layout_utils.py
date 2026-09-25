"""Shared layout-detection utilities.

Bbox math, NMS, containment filtering, read-order computation, and
constant tables used by both ``bibr.serve.deployments.layout`` (serve
worker) and ``bibr.local.layout`` (local pipeline). Pure functions —
no model loading, no torch dependency.
"""

from __future__ import annotations

import functools

import numpy as np

from bibr.config import GlobalSettings

_DEFAULT_SETTINGS = GlobalSettings()

# ---------------------------------------------------------------------------
# Label constants — sourced from PaddlePaddle/PP-DocLayoutV3 inference.yml
# ---------------------------------------------------------------------------

# Correct id2label mapping from original PaddlePaddle PP-DocLayoutV3 inference.yml.
# The HuggingFace config.json has a broken mapping that collapses 5 distinct labels
# (display_formula, footer_image, header_image, inline_formula, vertical_text)
# into duplicates (formula, footer, header, formula, text).
# This override restores the correct 25-class labels.
_CORRECT_ID2LABEL: dict[int, str] = {
    0: "abstract",
    1: "algorithm",
    2: "aside_text",
    3: "chart",
    4: "content",
    5: "display_formula",
    6: "doc_title",
    7: "figure_title",
    8: "footer",
    9: "footer_image",
    10: "footnote",
    11: "formula_number",
    12: "header",
    13: "header_image",
    14: "image",
    15: "inline_formula",
    16: "number",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    20: "seal",
    21: "table",
    22: "text",
    23: "vertical_text",
    24: "vision_footnote",
}

# Map layout labels to OCR task types.
# Aligned with vendored SDK config.yaml label_task_mapping.
LABEL_TASK_MAPPING: dict[str, list[str]] = {
    "text": [
        "abstract",
        "algorithm",
        "content",
        "doc_title",
        "figure_title",
        "paragraph_title",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
        "seal",
        "formula_number",
        # OCR'd so PDFParser can record header/footer metadata,
        # create PaperFootnote objects, and detect reference sections.
        # Matches bibr's active layout-to-OCR task contract.
        "header",
        "footer",
        "footnote",
        "reference",
    ],
    "table": ["table"],
    "formula": ["display_formula", "inline_formula", "formula"],
    "skip": ["chart", "image"],
    "abandon": [
        "number",
        "aside_text",
        "footer_image",
        "header_image",
    ],
}

# Invert: label_name -> task_type
_LABEL_TO_TASK: dict[str, str] = {}
for task_type, labels in LABEL_TASK_MAPPING.items():
    for label in labels:
        _LABEL_TO_TASK[label] = task_type

# Labels that should be treated as figure/table/chart captions
_CAPTION_LABELS: set[str] = {"figure_title"}

DEFAULT_THRESHOLD = _DEFAULT_SETTINGS.layout.detection_threshold
_MAX_BATCH_SIZE = _DEFAULT_SETTINGS.layout.batch_size

# Effective page batch for layout inference on CPU-only devices. A CPU batch
# of 8 buys no throughput (audit-measured ~275 ms/page at batch 1 vs
# ~312 ms/page at batch 8) but grows the ORT CPU arena to several GB, so CPU
# inference runs one page at a time unless the operator set
# ``LAYOUT_BATCH_SIZE`` explicitly.
_CPU_LAYOUT_BATCH_SIZE = 1


def effective_layout_batch_size(settings, device_type: str | None = None) -> int:
    """Page batch size for layout inference given the resolved device.

    An explicitly configured ``LAYOUT_BATCH_SIZE`` always wins. Otherwise CPU
    resolves to :data:`_CPU_LAYOUT_BATCH_SIZE` and anything else (CUDA, MPS,
    or an unknown device) keeps the configured default of 8.
    """
    layout = settings.layout
    if "batch_size" in layout.model_fields_set:
        return layout.batch_size
    if device_type is not None and str(device_type).lower() == "cpu":
        return _CPU_LAYOUT_BATCH_SIZE
    return layout.batch_size


# NMS thresholds (from vendored SDK config.yaml → layout.layout_nms settings).
_NMS_IOU_SAME = _DEFAULT_SETTINGS.layout.nms_iou_same
_NMS_IOU_DIFF = _DEFAULT_SETTINGS.layout.nms_iou_diff

# Per-class containment merge mode (from config.yaml → layout.layout_merge_bboxes_mode).
# Most classes use "large" (drop the smaller box when it is contained within
# a larger one). "reference" uses "small" (keep individual reference items
# that sit inside a reference container).
_MERGE_BBOXES_MODE: dict[str, str] = {
    "abstract": "large",
    "algorithm": "large",
    "aside_text": "large",
    "chart": "large",
    "content": "large",
    "display_formula": "large",
    "doc_title": "large",
    "figure_title": "large",
    "footer": "large",
    "footer_image": "large",
    "footnote": "large",
    "formula": "large",
    "formula_number": "large",
    "header": "large",
    "header_image": "large",
    "image": "large",
    "inline_formula": "large",
    "number": "large",
    "paragraph_title": "large",
    "reference": "small",
    "reference_content": "large",
    "seal": "large",
    "table": "large",
    "text": "large",
    "vertical_text": "large",
    "vision_footnote": "large",
}

# Labels whose boxes should never be removed by containment filtering.
_PRESERVE_LABELS: set[str] = {"image", "seal", "chart", "figure_title"}

# Label name for "image" — used by large-image filter.
_IMAGE_LABEL: str = "image"

# Area thresholds for filtering spurious full-page "image" detections.
# Ported from vendored SDK ``apply_layout_postprocess``.
_LARGE_IMAGE_AREA_LANDSCAPE = _DEFAULT_SETTINGS.layout.large_image_area_landscape
_LARGE_IMAGE_AREA_PORTRAIT = _DEFAULT_SETTINGS.layout.large_image_area_portrait


# ---------------------------------------------------------------------------
# Bbox utilities
# ---------------------------------------------------------------------------


def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU of two boxes [x1, y1, x2, y2]."""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    inter = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return float(inter / (area1 + area2 - inter))


def _nms(boxes: np.ndarray, iou_same: float, iou_diff: float) -> list[int]:
    """NMS with separate thresholds for same-class and cross-class overlaps.

    ``boxes`` has shape (N, 6+) with columns [label_id, score, x1, y1, x2, y2, ...].
    Returns indices of kept boxes.
    """
    from collections import deque

    scores = boxes[:, 1]
    indices = deque(np.argsort(scores)[::-1])
    selected: list[int] = []

    while indices:
        cur = indices.popleft()
        selected.append(cur)
        cur_cls = boxes[cur, 0]
        cur_coords = boxes[cur, 2:6]

        remaining: deque[int] = deque()
        for i in indices:
            other_cls = boxes[i, 0]
            other_coords = boxes[i, 2:6]
            threshold = iou_same if cur_cls == other_cls else iou_diff
            if _iou(cur_coords, other_coords) < threshold:
                remaining.append(i)
        indices = remaining

    return selected


def _is_contained(box_a: np.ndarray, box_b: np.ndarray) -> bool:
    """Check if box_a is contained within box_b (>=80% of box_a's area overlaps).

    Each box is [label_id, score, x1, y1, x2, y2, ...].
    """
    x1, y1, x2, y2 = box_a[2], box_a[3], box_a[4], box_a[5]
    area_a = (x2 - x1) * (y2 - y1)
    if area_a <= 0:
        return False

    xi1 = max(x1, box_b[2])
    yi1 = max(y1, box_b[3])
    xi2 = min(x2, box_b[4])
    yi2 = min(y2, box_b[5])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    return bool((inter / area_a) >= 0.8)


def _filter_containment(boxes: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    """Return a boolean keep-mask after containment filtering.

    Mirrors the vendored SDK's per-category ``check_containment`` logic:
    each category's mode is evaluated independently with fresh containment
    arrays, and the results are AND'd into a single keep mask.

    - ``"large"``: when box[i] is contained within a box[j] of this class,
      drop box[i] (remove the smaller contained box).
    - ``"small"``: when box[i] IS this class and is contained in box[j],
      drop box[j] (remove the larger container, keep the small item).

    Uses ``_MERGE_BBOXES_MODE`` and ``_PRESERVE_LABELS`` module constants.

    Implemented as a single vectorised pairwise containment matrix plus
    per-category masking — the previous nested-Python triple loop was
    O(k · n²) but with a Python tight loop body, costing ~5–10 ms per page
    on dense layouts. The math is identical (verified by the unit tests).
    """
    n = len(boxes)
    if n == 0:
        return np.ones(0, dtype=bool)

    classes = boxes[:, 0].astype(int)
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    x2 = boxes[:, 4]
    y2 = boxes[:, 5]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    # Pairwise intersection: inter[i, j] = area of overlap of boxes i and j.
    xi1 = np.maximum(x1[:, None], x1[None, :])
    yi1 = np.maximum(y1[:, None], y1[None, :])
    xi2 = np.minimum(x2[:, None], x2[None, :])
    yi2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(0.0, xi2 - xi1) * np.maximum(0.0, yi2 - yi1)

    # containment[i, j] = box i is contained in box j (>= 80% of i inside j).
    safe_areas = np.where(areas > 0, areas, 1.0)
    cov = inter / safe_areas[:, None]
    cov[areas == 0] = 0.0
    np.fill_diagonal(cov, 0.0)
    containment = cov >= 0.8

    is_preserved = np.array(
        [id2label.get(int(c), "") in _PRESERVE_LABELS for c in classes],
        dtype=bool,
    )
    is_small_mode = np.array(
        [_MERGE_BBOXES_MODE.get(id2label.get(int(c), ""), "large") == "small" for c in classes],
        dtype=bool,
    )

    # Collapse coincident cross-class pairs before the per-category loop.
    # PP-DocLayoutV3 co-emits a coincident reference/reference_content pair over
    # a lone reference; the "small"/"large" modes below would mutually annihilate
    # it. Any two cross-class boxes that each cover >= 80% of the other are one
    # entity — keep the higher-scored box (never a _PRESERVE_LABELS box), drop
    # the other, then run the loop on the survivors. Generic by design: the
    # 625-page census found reference/reference_content is the only pair this
    # ever fires on, so it is a no-op everywhere else.
    mutual = containment & containment.T
    cross = classes[:, None] != classes[None, :]
    pairs = np.triu(mutual & cross, 1)
    if pairs.any():
        scores = boxes[:, 1]
        collapsed = np.zeros(n, dtype=bool)
        for i, j in zip(*np.where(pairs), strict=True):
            if collapsed[i] or collapsed[j]:
                continue
            pi, pj = bool(is_preserved[i]), bool(is_preserved[j])
            if pi and pj:
                continue  # never drop two preserve boxes
            if pi:
                loser = j
            elif pj:
                loser = i
            else:
                loser = i if scores[i] < scores[j] else j
            collapsed[loser] = True
        if collapsed.any():
            sub_keep = _filter_containment(boxes[~collapsed], id2label)
            keep = np.zeros(n, dtype=bool)
            keep[np.where(~collapsed)[0][sub_keep]] = True
            return keep

    keep = np.ones(n, dtype=bool)
    for cat_id in np.unique(classes):
        cat_name = id2label.get(int(cat_id), "")
        mode = _MERGE_BBOXES_MODE.get(cat_name, "large")
        if mode == "union":
            continue

        if mode == "large":
            # "i contained in j" where i is a non-preserved, large-mode box
            # and j is class C. A small-mode item gets priority to displace
            # its container, so the two per-class passes cannot delete both.
            mask = (
                containment
                & (classes == cat_id)[None, :]
                & ~is_preserved[:, None]
                & ~is_small_mode[:, None]
            )
            keep &= ~mask.any(axis=1)
        elif mode == "small":
            # "i contained in j" where i is class C and non-preserved. A
            # preserve-labelled container is never removed by a small item.
            valid_i = (classes == cat_id) & ~is_preserved
            mask = containment & valid_i[:, None] & ~is_preserved[None, :]
            contained_by_other = mask.any(axis=1)
            contains_other = mask.any(axis=0)
            keep &= ~contains_other | contained_by_other

    return keep


# ---------------------------------------------------------------------------
# Docling-rulebook overlap resolver (selected by LAYOUT_OVERLAP_RESOLVER)
# ---------------------------------------------------------------------------

# Thresholds ported from docling LayoutPostprocessor._remove_overlapping_clusters
# (overlap_threshold / containment_threshold, both 0.8); ">=" matches bibr's
# _is_contained convention.
_RULEBOOK_CONTAINMENT = 0.8
_RULEBOOK_COINCIDENT_IOU = 0.8


class _UnionFind:
    """Union-Find over 0..n-1 with path compression + union by rank (Docling port)."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1


def _connected_components(indices: list[int], edges: np.ndarray) -> list[list[int]]:
    """Connected components of ``indices`` under the boolean ``edges`` matrix."""
    uf = _UnionFind(len(indices))
    sub = edges[np.ix_(indices, indices)]
    for a, b in zip(*np.where(np.triu(sub, 1)), strict=True):
        uf.union(int(a), int(b))
    comps: dict[int, list[int]] = {}
    for a, idx in enumerate(indices):
        comps.setdefault(uf.find(a), []).append(idx)
    return list(comps.values())


def _resolve_overlaps_rulebook(
    boxes: np.ndarray, id2label: dict[int, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Docling-rulebook overlap resolution: group, keep best, absorb losers.

    Port of docling ``LayoutPostprocessor._remove_overlapping_clusters``
    semantics onto bibr's flat region arrays. ``boxes`` has shape (N, 6+) with
    columns [label_id, score, x1, y1, x2, y2, ...]. Returns ``(keep_mask,
    boxes_out)``, both index-aligned with the input so callers can subset
    parallel arrays (e.g. model reading order) with the same mask;
    ``boxes_out`` is a copy in which winners' bboxes may have been expanded.

    Rulebook (vs the legacy per-category ``_filter_containment``):

    - Overlapping boxes are GROUPED via union-find and each group keeps its
      best member — two overlapping regions can never both be deleted (the
      legacy filter can mutually annihilate e.g. a "small"-mode reference
      nested in a "large"-mode text).
    - Two distinct edge kinds (Docling's coincident-vs-nesting split):
      IoU >= 0.8 marks *coincident duplicates* (near-identical boxes,
      typically one entity re-emitted under two classes) and always
      conflicts; containment >= 0.8 marks *nesting*, which only conflicts
      when the per-class merge modes say so ("large" containers eat their
      contents; "small" contents displace their container). A nested pair
      whose modes create no conflict — reference_content items inside a
      reference envelope — forms no edge and survives untouched.
    - Winner: ``_PRESERVE_LABELS`` first, then score, then area. Preserved
      boxes are never dropped (no edge between two preserved boxes, and a
      preserved box in a group is kept even when it is not the winner).
    - Absorption maps Docling's ``best.cells.extend(...)`` onto bboxes
      (region text does not exist yet at this stage — a vision model OCRs
      surviving crops later): a non-"small" winner expands to the union with
      every absorbed loser so no page area (hence no text) is lost; a
      "small" winner keeps its tight bbox and may only drop losers already
      >= 80% covered by it — under-covered losers survive, regrouped by
      their own connectivity so boxes related only through the winner never
      merge with each other.
    """
    n = len(boxes)
    out = boxes.copy()
    if n == 0:
        return np.ones(0, dtype=bool), out

    classes = boxes[:, 0].astype(int)
    scores = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    x2 = boxes[:, 4]
    y2 = boxes[:, 5]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    xi1 = np.maximum(x1[:, None], x1[None, :])
    yi1 = np.maximum(y1[:, None], y1[None, :])
    xi2 = np.minimum(x2[:, None], x2[None, :])
    yi2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(0.0, xi2 - xi1) * np.maximum(0.0, yi2 - yi1)

    # cov[i, j] = fraction of box i covered by box j; degenerate boxes cover
    # nothing and are covered by nothing (they form no edges, so they survive).
    safe_areas = np.where(areas > 0, areas, 1.0)
    cov = inter / safe_areas[:, None]
    cov[areas == 0] = 0.0
    np.fill_diagonal(cov, 0.0)

    union_area = safe_areas[:, None] + safe_areas[None, :] - inter
    iou = inter / np.maximum(union_area, 1.0)
    np.fill_diagonal(iou, 0.0)

    labels = [id2label.get(int(c), "") for c in classes]
    preserved = np.array([lbl in _PRESERVE_LABELS for lbl in labels], dtype=bool)
    modes = [_MERGE_BBOXES_MODE.get(lbl, "large") for lbl in labels]
    mode_large = np.array([m == "large" for m in modes], dtype=bool)
    mode_small = np.array([m == "small" for m in modes], dtype=bool)

    # Coincident duplicates conflict regardless of merge modes; two preserved
    # boxes are both kept, so no edge between them.
    coincident = (iou >= _RULEBOOK_COINCIDENT_IOU) & ~(preserved[:, None] & preserved[None, :])
    # Nesting conflicts per merge mode: nesting[i, j] means "i inside j" and
    # either the container's class eats contents ("large") or the contained
    # box's class displaces containers ("small"). Preserved inner boxes are
    # untouchable, mirroring the legacy filter's ~is_preserved guard.
    contained = cov >= _RULEBOOK_CONTAINMENT
    nesting = contained & ~preserved[:, None] & (mode_large[None, :] | mode_small[:, None])
    edges = coincident | nesting | nesting.T
    np.fill_diagonal(edges, False)

    keep = np.zeros(n, dtype=bool)
    stack = _connected_components(list(range(n)), edges)
    while stack:
        members = stack.pop()
        if len(members) == 1:
            keep[members[0]] = True
            continue
        winner = max(members, key=lambda k: (preserved[k], scores[k], areas[k], -k))
        keep[winner] = True
        tight = mode_small[winner]
        leftovers: list[int] = []
        for m in members:
            if m == winner:
                continue
            if preserved[m]:
                keep[m] = True
            elif not tight:
                # Absorb: expand the winner to the union so no area is lost.
                out[winner, 2] = min(out[winner, 2], boxes[m, 2])
                out[winner, 3] = min(out[winner, 3], boxes[m, 3])
                out[winner, 4] = max(out[winner, 4], boxes[m, 4])
                out[winner, 5] = max(out[winner, 5], boxes[m, 5])
            elif cov[m, winner] < _RULEBOOK_CONTAINMENT:
                # A "small" winner keeps its tight bbox; dropping an
                # under-covered loser would lose page area — re-resolve it.
                leftovers.append(m)
        if leftovers:
            stack.extend(_connected_components(leftovers, edges))
    return keep, out


def _compute_read_order(boxes: np.ndarray) -> list[int]:
    """Compute reading order by sorting boxes top-to-bottom, left-to-right.

    RT-DETR doesn't output explicit read_order, so we approximate it from
    bounding box positions.

    ``boxes`` has shape (N, 6+) with columns [label_id, score, x1, y1, x2, y2, ...].
    Returns list of read_order values (rank of each box in reading order).
    """
    n = len(boxes)
    if n == 0:
        return []
    y_centers = (boxes[:, 3] + boxes[:, 5]) / 2  # (y1 + y2) / 2
    x_centers = (boxes[:, 2] + boxes[:, 4]) / 2  # (x1 + x2) / 2
    order = np.lexsort((x_centers, y_centers))
    read_orders = [0] * n
    for rank, idx in enumerate(order):
        read_orders[idx] = rank
    return read_orders


# ---------------------------------------------------------------------------
# Column-aware read-order fallback (selected by LAYOUT_READ_ORDER_FALLBACK)
# ---------------------------------------------------------------------------

# Ported from docling-ibm-models ReadingOrderPredictor (reading_order_rb.py).
# Three stages: (1) up/down adjacency — "i directly above j with x-overlap and
# nothing vertically between them"; (2) horizontal dilation of each box toward
# its first up/down neighbour (capped per side at 0.15 x page width) so ragged
# column members grow to a shared x-span, then adjacency is recomputed on the
# dilated spans — this is where column membership emerges without ever letting
# a column box grow across the gutter toward a full-width neighbour; (3) a DFS
# over the resulting DAG that climbs to the highest unread ancestor before
# emitting, which is what holds spanning content back until every column
# feeding it has been read.
#
# Deliberately dropped from the donor:
# - separate furniture ordering (page headers/footers ordered apart from the
#   body): all region classes are ordered uniformly here — bibr's downstream
#   LABEL_TASK_MAPPING already decides how furniture is treated;
# - the l2r/r2l same-line merge maps: dead code in the donor (guarded by a
#   literal ``False``), never populated;
# - caption/footnote attachment and the hyphen text-merge post-passes: not
#   reading order;
# - the R-tree index: candidate pruning only — pages carry ~10-10^2 regions,
#   pairwise numpy masks suffice;
# - multi-page grouping: bibr computes read order per page.
#
# Deviations for determinism/correctness: neighbours are collected in
# ascending box-index order (the donor leans on unspecified R-tree iteration
# order); comparator ties get a top-to-bottom then left-to-right break (the
# donor leaves them unordered) so single-column pages reproduce the xy order;
# the donor's revert-dilation-on-collision guard is honoured as written intent
# (its implementation mutates the box before testing, making the guard a no-op
# there); the DFS scan offset advances monotonically instead of resetting to a
# slice-relative index (skipped children are still reached through the emitted
# ancestor's descendants).

_RB_DILATION_CAP = 0.15 * 1000.0  # per-side cap in 0-1000 page-width units
_RB_EPS = 1.0e-3  # donor PageElement.eps: touching edges count as above/below


def _rb_up_down_maps(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
) -> tuple[list[list[int]], list[list[int]]]:
    """Up/down adjacency: i -> j when i is strictly above j with x-overlap and
    no third box both x-overlaps either end and sits strictly between them
    (donor ``_init_ud_maps`` + ``_has_sequence_interruption``)."""
    n = len(x1)
    ho = (x1[:, None] < x2[None, :]) & (x1[None, :] < x2[:, None])
    np.fill_diagonal(ho, False)
    above = (y2[:, None] - _RB_EPS) < y1[None, :]
    up: list[list[int]] = [[] for _ in range(n)]
    dn: list[list[int]] = [[] for _ in range(n)]
    for i, j in zip(*np.where(above & ho), strict=True):
        between = (ho[:, i] | ho[:, j]) & above[i, :] & above[:, j]
        between[i] = between[j] = False
        if not between.any():
            dn[int(i)].append(int(j))
            up[int(j)].append(int(i))
    return up, dn


def _rb_dilate_spans(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    up: list[list[int]],
    dn: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Widen each x-span toward its first up/down neighbour (donor
    ``_do_horizontal_dilation``). A box is skipped entirely when either side
    would grow past the cap — that is what stops column members from dilating
    across the gutter toward a full-width neighbour — and a widened span that
    would collide with any other original box is reverted."""
    dx1, dx2 = x1.copy(), x2.copy()
    for i in range(len(x1)):
        lo, hi = x1[i], x2[i]
        grown = False
        capped = False
        for neighbours in (up[i], dn[i]):
            if not neighbours:
                continue
            nb = neighbours[0]
            lo_d, hi_d = min(lo, x1[nb]), max(hi, x2[nb])
            if (lo - lo_d) > _RB_DILATION_CAP or (hi_d - hi) > _RB_DILATION_CAP:
                capped = True
                break
            lo, hi = lo_d, hi_d
            grown = True
        if capped or not grown:
            continue
        collide = (x1 < hi) & (lo < x2) & (y1 < y2[i]) & (y1[i] < y2)
        collide[i] = False
        if not collide.any():
            dx1[i], dx2[i] = lo, hi
    return dx1, dx2


def _compute_read_order_rb(boxes: np.ndarray) -> list[int]:
    """Column-aware reading order (rule-based Docling port).

    Same contract as ``_compute_read_order``: ``boxes`` has shape (N, 6+) with
    columns [label_id, score, x1, y1, x2, y2, ...] in 0-1000 normalised page
    space; returns the rank of each box in reading order.
    """
    n = len(boxes)
    if n == 0:
        return []
    x1 = boxes[:, 2].astype(np.float64)
    y1 = boxes[:, 3].astype(np.float64)
    x2 = boxes[:, 4].astype(np.float64)
    y2 = boxes[:, 5].astype(np.float64)

    up, dn = _rb_up_down_maps(x1, y1, x2, y2)
    dx1, dx2 = _rb_dilate_spans(x1, y1, x2, y2, up, dn)
    up, dn = _rb_up_down_maps(dx1, y1, dx2, y2)

    # Donor comparator, on the ORIGINAL spans (dilation only shapes the DAG):
    # x-overlapping boxes order by bottom edge (higher first), disjoint ones
    # left-to-right; ties break top-to-bottom, left-to-right, then index.
    def _before(a: int, b: int) -> int:
        if x1[a] < x2[b] and x1[b] < x2[a]:
            if y2[a] != y2[b]:
                return -1 if y2[a] < y2[b] else 1
        elif x1[a] != x1[b]:
            return -1 if x1[a] < x1[b] else 1
        key_a, key_b = (y1[a], x1[a], a), (y1[b], x1[b], b)
        return -1 if key_a < key_b else 1

    key = functools.cmp_to_key(_before)
    heads = sorted((i for i in range(n) if not up[i]), key=key)
    dn = [sorted(children, key=key) for children in dn]

    visited = [False] * n
    order: list[int] = []

    def _climb(i: int) -> int:
        # Highest not-yet-read ancestor: reading resumes wherever the up-chain
        # re-enters unread territory (donor ``_depth_first_search_upwards``).
        k = i
        while True:
            parent = next((p for p in up[k] if not visited[p]), None)
            if parent is None:
                return k
            k = parent

    for head in heads:
        if visited[head]:
            continue
        order.append(head)
        visited[head] = True
        stack: list[tuple[list[int], int]] = [(dn[head], 0)]
        while stack:
            children, offset = stack[-1]
            advanced = False
            for pos in range(offset, len(children)):
                k = _climb(children[pos])
                if not visited[k]:
                    order.append(k)
                    visited[k] = True
                    stack[-1] = (children, pos + 1)
                    stack.append((dn[k], 0))
                    advanced = True
                    break
            if not advanced:
                stack.pop()

    if len(order) != n:  # unreachable for a DAG; never emit duplicate ranks
        order.extend(i for i in range(n) if not visited[i])

    ranks = [0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks
