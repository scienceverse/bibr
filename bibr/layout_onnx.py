"""ONNX Runtime backend for the PP-DocLayout layout detector (torch-free).

Two architectures share the backend; the bundle manifest's ``architecture``
picks one (bundles without the key predate PP-DocLayoutV4 and are V3).

**PP-DocLayoutV3** — replicates the parts of transformers'
``PPDocLayoutV3ImageProcessor`` that bibr's detector uses, in numpy:

- **preprocessing** — bicubic resize to the model's fixed input size with
  ``antialias=False`` (PyTorch's ``upsample_bicubic2d``: A = -0.75, half-pixel
  source index, 4-tap window clamped at the border), rounded back to uint8 as
  torchvision does for integer tensors, then rescaled by 1/255 (the model's
  mean is 0 and std is 1);
- **postprocessing** — sigmoid scores, flat top-*k* over queries × classes,
  ``cxcywh`` → ``xyxy`` in original pixels, the reading-order sequence from the
  order head, thresholding and sorting by order.

The mask head is not exported (bibr never reads the polygons; the HF polygon
path is what needs OpenCV), so the graph outputs are ``logits``,
``pred_boxes`` and ``order_logits``.

**PP-DocLayoutV4** — replicates ``PPDocLayoutV4ImageProcessor`` as of
huggingface/transformers#48387 (head ``c831093b``, the revision this port was
checked against):

- **preprocessing** — rescale to ``[0, 1]`` *first*, resize in float with the
  same bicubic kernel (no rounding between the passes), and clip the bicubic
  overshoot back to ``[0, 1]``;
- **postprocessing** — the box head regresses a quadrilateral
  (``[cx, cy, dx1, dy1, …, dx4, dy4]``, offsets shifted by +0.5) whose enclosing
  rectangle becomes ``boxes``; the reading order is decoded from two pairwise
  heads, ``relative_order_logits`` ("is *i* read before *j*?") and
  ``successor_order_logits`` ("does *j* directly follow *i*?"), by
  :func:`decode_reading_order`, over the boxes that pass the threshold.

The graph outputs are ``logits``, ``pred_boxes``, ``relative_order_logits``
and ``successor_order_logits``. Either way the result dicts feed
:meth:`bibr.layout_base.BaseLayoutDetector._postprocess` unchanged.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bibr.exceptions import ConfigurationError
from bibr.utils.ml_runtime import ONNX_MODEL, read_onnx_manifest

logger = logging.getLogger(__name__)

PP_DOCLAYOUT_V3 = "PPDocLayoutV3ForObjectDetection"
PP_DOCLAYOUT_V4 = "PPDocLayoutV4ForObjectDetection"

_GRAPH_OUTPUTS: dict[str, tuple[str, ...]] = {
    PP_DOCLAYOUT_V3: ("logits", "pred_boxes", "order_logits"),
    PP_DOCLAYOUT_V4: ("logits", "pred_boxes", "relative_order_logits", "successor_order_logits"),
}

# PyTorch's bicubic coefficient (torch/csrc/api/... UpSampleBicubic2d: A = -0.75).
_CUBIC_A = -0.75

# Column scaling is the slow pass (strided gathers): split it into row blocks
# across at most this many threads. Each block runs the exact per-tap loop,
# so threaded output is bit-identical to the serial one.
_RESIZE_THREADS = min(8, os.cpu_count() or 1)
_RESIZE_ROWS_PER_THREAD = 256


def _cubic_coefficients(t: np.ndarray) -> np.ndarray:
    """Four bicubic tap weights for fractional offsets ``t`` in [0, 1).

    Mirrors ``get_cubic_upsample_coefficients`` in PyTorch's upsample kernels.
    """
    a = _CUBIC_A

    def conv1(x):  # |x| <= 1
        return ((a + 2) * x - (a + 3)) * x * x + 1

    def conv2(x):  # 1 < |x| < 2
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

    return np.stack([conv2(t + 1.0), conv1(t), conv1(1.0 - t), conv2(2.0 - t)], axis=-1)


def _resize_plan(
    in_size: int, out_size: int, *, float_kernel: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Clamped tap indices ``(out, 4)`` and float32 weights ``(out, 4)`` for one axis.

    ``float_kernel`` follows ATen's float kernel, which computes the scale, the
    source index and the tap weights in float32, the index as one fused
    multiply-add. At 2,339 source pixels (A4 at 200 dpi) one float32 step of
    the index is 2.4e-4, so float64 arithmetic, or a separate multiply and
    subtract, drifts by ~2e-4; this plan stays within ~1e-6 of torch. The
    uint8 plan keeps the float64 arithmetic its V3 parity was measured with.
    """
    if float_kernel:
        scale = np.float32(in_size) / np.float32(out_size)
        dst = np.arange(out_size, dtype=np.float32) + np.float32(0.5)
        # fma(scale, dst, -0.5): the float64 product of two float32 values is
        # exact, so subtracting there and rounding once reproduces it.
        src = (np.float64(scale) * dst.astype(np.float64) - 0.5).astype(np.float32)
        base = np.floor(src)
        t = src - base
        rest = np.float32(1.0) - t  # ATen evaluates the far taps at (1 - t) and (1 - t) + 1
        a = np.float32(_CUBIC_A)
        weights = np.stack(
            [
                _cubic_far(t + np.float32(1.0), a),
                _cubic_near(t, a),
                _cubic_near(rest, a),
                _cubic_far(rest + np.float32(1.0), a),
            ],
            axis=-1,
        )
    else:
        scale = in_size / out_size
        dst = np.arange(out_size, dtype=np.float64)
        src = (dst + 0.5) * scale - 0.5  # align_corners=False, cubic: no clamp at 0
        base = np.floor(src)
        t = src - base
        weights = _cubic_coefficients(t)
    idx = base[:, None].astype(np.int64) + np.array([-1, 0, 1, 2], dtype=np.int64)[None, :]
    idx = np.clip(idx, 0, in_size - 1)
    return idx, weights.astype(np.float32)


def _cubic_near(x: np.ndarray, a: np.float32) -> np.ndarray:
    """``cubic_convolution1`` (``|x| <= 1``) in the dtype of ``x``."""
    return ((a + np.float32(2)) * x - (a + np.float32(3))) * x * x + np.float32(1)


def _cubic_far(x: np.ndarray, a: np.float32) -> np.ndarray:
    """``cubic_convolution2`` (``1 < |x| < 2``) in the dtype of ``x``."""
    return ((a * x - np.float32(5) * a) * x + np.float32(8) * a) * x - np.float32(4) * a


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255)


def _bicubic_passes(src: np.ndarray, out_h: int, out_w: int, *, round_between: bool) -> np.ndarray:
    """Separable bicubic over a float32 ``(C, H, W)`` array, x pass then y pass.

    ``round_between`` is the uint8 kernel (saturate after each pass, float64
    plan); without it this is the float kernel (float32 plan, no rounding).
    """
    c, h, w = src.shape
    ix, wx = _resize_plan(w, out_w, float_kernel=not round_between)
    iy, wy = _resize_plan(h, out_h, float_kernel=not round_between)
    tmp = _x_pass(src, out_w, ix, wx)
    if round_between:
        tmp = _to_uint8(tmp)
    out = np.zeros((c, out_h, out_w), dtype=np.float32)
    for k in range(4):
        out += tmp[:, iy[:, k], :] * wy[None, :, k, None]
    return out


def _x_pass(src: np.ndarray, out_w: int, ix: np.ndarray, wx: np.ndarray) -> np.ndarray:
    """Column scaling, row blocks across threads when the image is tall.

    Blocks write disjoint row slices and each runs the same per-tap loop the
    serial code ran, so the result is bit-identical — threads only overlap
    the strided gathers, which is where the time goes.
    """
    c, h, _w = src.shape
    workers = min(_RESIZE_THREADS, max(1, h // _RESIZE_ROWS_PER_THREAD))
    tmp = np.empty((c, h, out_w), dtype=np.float32)
    bounds = [round(h * i / workers) for i in range(workers + 1)]

    def block(b: int) -> None:
        lo, hi = bounds[b], bounds[b + 1]
        part = src[:, lo:hi, :]
        acc = np.zeros((c, hi - lo, out_w), dtype=np.float32)
        for k in range(4):
            acc += part[:, :, ix[:, k]] * wx[None, None, :, k]
        tmp[:, lo:hi, :] = acc

    if workers == 1:
        block(0)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="bibr-resize"
        ) as pool:
            list(pool.map(block, range(workers)))
    return tmp


def resize_bicubic_no_antialias(chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a ``(C, H, W)`` uint8 array like ``torchvision.resize(antialias=False)``.

    Interpolates along x then y, in the order of PyTorch's CPU kernel.

    The pass boundary matters. `transformers`' fast image processor resizes the
    *uint8* tensor, and for bicubic on CPU torchvision hands uint8 straight to
    ``interpolate``, which runs ATen's separable fixed-point kernel: each pass
    saturates back to uint8 before the next one begins. Bicubic overshoots at a
    sharp edge — and a scanned page is nothing but sharp edges — so carrying the
    overshoot through to the second pass in float, as a naive float32
    implementation does, moves pixels by up to 20 levels and changes which
    regions the detector finds. Rounding and clamping between the passes brings
    it back to within one level of the torch path.
    """
    src = chw.astype(np.float32)
    if chw.shape[1:] != (out_h, out_w):
        src = _bicubic_passes(src, out_h, out_w, round_between=True)
    return _to_uint8(src).astype(np.uint8)


def resize_bicubic_float(chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a float ``(C, H, W)`` array like ``torchvision.resize(antialias=False)``.

    The float counterpart of :func:`resize_bicubic_no_antialias`: a float
    tensor runs ATen's float kernel, which neither rounds nor clamps, so the
    overshoot survives both passes. PP-DocLayoutV4's processor resizes the
    rescaled float image and clips afterwards.
    """
    src = chw.astype(np.float32, copy=False)
    if chw.shape[1:] == (out_h, out_w):
        return src.copy()
    return _bicubic_passes(src, out_h, out_w, round_between=False)


def preprocess_images(
    images: list[Image.Image],
    *,
    size: tuple[int, int],
    rescale_factor: float,
    image_mean: list[float],
    image_std: list[float],
    rescale_before_resize: bool = False,
) -> np.ndarray:
    """PIL images → ``(B, 3, H, W)`` float32 ``pixel_values``.

    ``rescale_before_resize`` selects PP-DocLayoutV4's order (rescale, float
    resize, clip to the rescaled uint8 range) over V3's (uint8 resize, then
    rescale). V4's reference resizes uint8 with ``cv2.resize``, which rounds
    once; resizing an integer tensor rounds twice, and the drift is enough to
    permute the predicted reading order.
    """
    out_h, out_w = size
    batch = np.empty((len(images), 3, out_h, out_w), dtype=np.float32)
    mean = np.asarray(image_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(image_std, dtype=np.float32)[:, None, None]
    factor = np.float32(rescale_factor)
    upper = np.float32(255.0 * rescale_factor)
    for i, image in enumerate(images):
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        chw = np.asarray(rgb, dtype=np.uint8).transpose(2, 0, 1)
        if rescale_before_resize:
            resized = resize_bicubic_float(chw.astype(np.float32) * factor, out_h, out_w)
            batch[i] = (np.clip(resized, np.float32(0.0), upper) - mean) / std
        else:
            resized = resize_bicubic_no_antialias(chw, out_h, out_w)
            batch[i] = (resized.astype(np.float32) * factor - mean) / std
    return batch


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-free logistic — the order head emits ±1e4 mask values."""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def order_sequences(order_logits: np.ndarray) -> np.ndarray:
    """Reading-order rank per query from the order head, ``(B, Q)``.

    Mirrors ``PPDocLayoutV3ImageProcessor._get_order_seqs``: each query
    collects votes from the pairwise "before" scores, and its rank is its
    position in the ascending vote order.
    """
    scores = _sigmoid(order_logits.astype(np.float32))
    batch, n, _ = scores.shape
    votes = np.triu(scores, 1).sum(axis=1) + np.tril(1.0 - scores.transpose(0, 2, 1), -1).sum(
        axis=1
    )
    pointers = np.argsort(votes, axis=1, kind="stable")
    seq = np.empty_like(pointers)
    ranks = np.arange(n, dtype=pointers.dtype)
    for b in range(batch):
        seq[b, pointers[b]] = ranks
    return seq


def _top_detections(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat top-*k* (k = queries) over queries × classes: ``(scores, labels, queries)``."""
    batch, num_queries, num_classes = logits.shape
    scores_all = _sigmoid(logits.astype(np.float32)).reshape(batch, -1)
    k = num_queries
    # torch.topk(sorted=True): the k best, descending. argpartition + stable
    # sort of the selected block reproduces it up to exact score ties (torch
    # orders those by its own kernel) and NaN scores (torch ranks NaN first,
    # this ranks it last), neither of which a trained head's float logits
    # produce in practice.
    part = np.argpartition(-scores_all, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores_all, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    index = np.take_along_axis(part, order, axis=1)
    scores = np.take_along_axis(scores_all, index, axis=1)
    return scores, index % num_classes, index // num_classes


def decode_detections(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    order_logits: np.ndarray,
    orig_sizes: list[tuple[int, int]],
    threshold: float,
) -> list[dict[str, np.ndarray]]:
    """PP-DocLayoutV3 graph outputs → per-image ``{scores, labels, boxes, order_seq}``.

    Mirrors ``post_process_object_detection`` minus the polygon masks: boxes
    are ``xyxy`` in original pixels, rows are thresholded and sorted by the
    model's reading order.
    """
    scores, labels, queries = _top_detections(logits)
    centers = pred_boxes[..., :2]
    dims = pred_boxes[..., 2:]
    boxes = np.concatenate([centers - 0.5 * dims, centers + 0.5 * dims], axis=-1)
    seq = order_sequences(order_logits)

    results: list[dict[str, np.ndarray]] = []
    for b in range(len(orig_sizes)):
        height, width = orig_sizes[b]
        scale = np.asarray([width, height, width, height], dtype=np.float32)
        b_boxes = boxes[b][queries[b]] * scale
        b_seq = seq[b][queries[b]]
        keep = scores[b] >= threshold
        kept_seq = b_seq[keep]
        sort_idx = np.argsort(kept_seq, kind="stable")
        results.append(
            {
                "scores": scores[b][keep][sort_idx],
                "labels": labels[b][keep][sort_idx],
                "boxes": b_boxes[keep][sort_idx],
                "order_seq": kept_seq[sort_idx],
            }
        )
    return results


# -- PP-DocLayoutV4 ------------------------------------------------------------
#
# The graph helpers below replace the ``scipy.sparse.csgraph`` calls of the
# reference processor with plain Python: pages keep tens of boxes, and scipy is
# not a core dependency. Only the *partition* into components matters to the
# decode (cycle removal compares component labels for equality; the undirected
# components are numbered by their smallest node, as scipy numbers them), so the
# results are identical.


def _strong_components(num_nodes: int, edges) -> list[int]:
    """Strongly connected component label per node (iterative Tarjan)."""
    graph: list[list[int]] = [[] for _ in range(num_nodes)]
    for i, j in edges:
        graph[i].append(j)
    index = [-1] * num_nodes
    low = [0] * num_nodes
    labels = [-1] * num_nodes
    on_stack = [False] * num_nodes
    stack: list[int] = []
    counter = 0
    component = 0
    for root in range(num_nodes):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            node, child = work[-1]
            if child < len(graph[node]):
                work[-1] = (node, child + 1)
                succ = graph[node][child]
                if index[succ] == -1:
                    index[succ] = low[succ] = counter
                    counter += 1
                    stack.append(succ)
                    on_stack[succ] = True
                    work.append((succ, 0))
                elif on_stack[succ]:
                    low[node] = min(low[node], index[succ])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    labels[member] = component
                    if member == node:
                        break
                component += 1
    return labels


def _remove_cycles(num_nodes: int, edges: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    """Drop the lowest-confidence edge of every cyclic component until the graph is a DAG.

    A strongly connected component of more than one node is exactly a set of
    nodes on a common cycle, so every edge inside one is a cycle edge and
    dropping the weakest always breaks at least one cycle.
    """
    edges = dict(edges)
    while True:
        labels = _strong_components(num_nodes, edges)
        weakest: dict[int, tuple[int, int]] = {}
        for edge, confidence in edges.items():
            component = labels[edge[0]]
            if component != labels[edge[1]]:
                continue
            # Ties keep the first edge in insertion order (deterministic decode).
            if component not in weakest or confidence < edges[weakest[component]]:
                weakest[component] = edge
        if not weakest:
            return list(edges)
        for edge in weakest.values():
            del edges[edge]


def _weak_components(num_nodes: int, edges) -> list[list[int]]:
    """Connected components of the undirected view, numbered by their smallest node."""
    neighbours: list[list[int]] = [[] for _ in range(num_nodes)]
    for i, j in edges:
        neighbours[i].append(j)
        neighbours[j].append(i)
    labels = [-1] * num_nodes
    count = 0
    for start in range(num_nodes):
        if labels[start] != -1:
            continue
        labels[start] = count
        frontier = [start]
        while frontier:
            node = frontier.pop()
            for other in neighbours[node]:
                if labels[other] == -1:
                    labels[other] = count
                    frontier.append(other)
        count += 1
    components: list[list[int]] = [[] for _ in range(count)]
    for node, label in enumerate(labels):
        components[label].append(node)
    return components


def _topological_sort(
    num_nodes: int, edges: list[tuple[int, int]], relative_scores: np.ndarray
) -> list[int]:
    """Topological order of a DAG, ties broken by a Borda count of the relative scores.

    ``relative_scores[i, j]`` high means *i* is likely read before *j*; equal
    counts go to the smaller index.
    """
    in_degree = [0] * num_nodes
    graph: defaultdict[int, list[int]] = defaultdict(list)
    for i, j in edges:
        graph[i].append(j)
        in_degree[j] += 1

    candidates = [node for node in range(num_nodes) if in_degree[node] == 0]
    order: list[int] = []
    while candidates:
        if len(candidates) == 1:
            best = candidates[0]
        else:
            best, best_score = None, -1.0
            for candidate in candidates:
                score = sum(
                    relative_scores[candidate][other] for other in candidates if other != candidate
                )
                if score > best_score or (
                    score == best_score and (best is None or candidate < best)
                ):
                    best_score = score
                    best = candidate
            if best is None:
                # Every count is NaN (fp16 overflow in the order head). The
                # reference raises here; take the smallest index instead of
                # failing the whole layout batch. Finite scores never get here.
                best = min(candidates)
        candidates.remove(best)
        order.append(best)
        for neighbour in graph[best]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                candidates.append(neighbour)

    # A cycle that survived cycle removal would strand nodes; append them.
    if len(order) < num_nodes:
        placed = set(order)
        order.extend(node for node in range(num_nodes) if node not in placed)
    return order


def decode_reading_order(relative_logits: np.ndarray, successor_logits: np.ndarray) -> np.ndarray:
    """0-based reading-order rank per box from PP-DocLayoutV4's two order heads.

    Mirrors ``PPDocLayoutV4ImageProcessor._decode_reading_order``: positive
    successor logits form a "directly precedes" graph, which is made acyclic,
    split into connected components and sorted topologically; the relative
    order scores break ties inside a component and order the components by
    their mean vote. Both matrices are restricted to the kept boxes.
    """
    num_boxes = successor_logits.shape[0]
    if num_boxes <= 1:
        return np.zeros(num_boxes, dtype=np.int64)

    rows, cols = np.nonzero(successor_logits > 0)  # row-major, as the reference's nested loops
    edges = {
        (int(i), int(j)): float(successor_logits[i, j])
        for i, j in zip(rows, cols, strict=True)
        if i != j
    }
    dag_edges = _remove_cycles(num_boxes, edges)
    components = _weak_components(num_boxes, dag_edges)

    # The successor head masks its diagonal with -1e4; exp(-|x|) never overflows.
    magnitude = np.exp(-np.abs(relative_logits))
    relative_scores = np.where(
        relative_logits >= 0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude)
    )
    np.fill_diagonal(relative_scores, 0.0)
    node_votes = relative_scores.sum(axis=0)

    component_orders: list[list[int]] = []
    for component in components:
        if len(component) == 1:
            component_orders.append(list(component))
            continue
        nodes = sorted(component)
        local_index = {node: local for local, node in enumerate(nodes)}
        local_edges = [
            (local_index[i], local_index[j])
            for i, j in dag_edges
            if i in local_index and j in local_index
        ]
        local_scores = relative_scores[np.ix_(nodes, nodes)]
        local_order = _topological_sort(len(nodes), local_edges, local_scores)
        component_orders.append([nodes[local] for local in local_order])

    # Earliest-reading component first, by mean relative-order vote.
    component_orders.sort(key=lambda component: float(np.mean(node_votes[component])))

    ranks = np.zeros(num_boxes, dtype=np.int64)
    for rank, node in enumerate(node for component in component_orders for node in component):
        ranks[node] = rank
    return ranks


def quad_corners(pred_boxes: np.ndarray) -> np.ndarray:
    """``(..., 10)`` quad parameters → ``(..., 4, 2)`` normalized corners (TL, TR, BR, BL)."""
    if pred_boxes.shape[-1] != 10:
        raise ValueError(f"PP-DocLayoutV4 boxes have 10 coordinates, got {pred_boxes.shape[-1]}")
    centers = pred_boxes[..., None, :2]
    offsets = pred_boxes[..., 2:].reshape(*pred_boxes.shape[:-1], 4, 2) - np.float32(0.5)
    return centers + offsets


def decode_detections_v4(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    relative_order_logits: np.ndarray,
    successor_order_logits: np.ndarray,
    orig_sizes: list[tuple[int, int]],
    threshold: float,
) -> list[dict[str, np.ndarray]]:
    """PP-DocLayoutV4 graph outputs → per-image ``{scores, labels, boxes, polygon_points, order_seq}``.

    Mirrors ``PPDocLayoutV4ImageProcessor.post_process_object_detection``:
    ``boxes`` is the ``xyxy`` rectangle enclosing each quad in original
    pixels, and rows are thresholded and sorted by the decoded reading order.
    A query kept under several labels shares one rank.
    """
    scores, labels, queries = _top_detections(logits)
    corners = quad_corners(pred_boxes.astype(np.float32))

    results: list[dict[str, np.ndarray]] = []
    for b in range(len(orig_sizes)):
        height, width = orig_sizes[b]
        scale = np.asarray([width, height], dtype=np.float32)
        keep = scores[b] >= threshold
        b_queries = queries[b][keep]
        b_corners = corners[b][b_queries] * scale
        unique_queries, inverse = np.unique(b_queries, return_inverse=True)
        submatrix = np.ix_(unique_queries, unique_queries)
        ranks = decode_reading_order(
            relative_order_logits[b].astype(np.float32)[submatrix],
            successor_order_logits[b].astype(np.float32)[submatrix],
        )
        order_seq = ranks[inverse.reshape(-1)]
        sort_idx = np.argsort(order_seq, kind="stable")
        sorted_corners = b_corners[sort_idx]
        results.append(
            {
                "scores": scores[b][keep][sort_idx],
                "labels": labels[b][keep][sort_idx],
                "boxes": np.concatenate(
                    [sorted_corners.min(axis=-2), sorted_corners.max(axis=-2)], axis=-1
                ),
                "polygon_points": sorted_corners,
                "order_seq": order_seq[sort_idx],
            }
        )
    return results


def architecture_label(architecture: str) -> str:
    """``PPDocLayoutV4ForObjectDetection`` → ``PP-DocLayoutV4`` (for logs and errors)."""
    for version in ("V4", "V3"):
        if f"PPDocLayout{version}" in architecture:
            return f"PP-DocLayout{version}"
    return architecture


class OnnxLayoutBackend:
    """ORT session + numpy pre/post-processing behind ``BaseLayoutDetector``."""

    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        device: str | None = None,
        threshold: float,
    ) -> None:
        from bibr.utils.onnx_providers import create_session

        self.bundle_dir = Path(bundle_dir)
        self.manifest = read_onnx_manifest(self.bundle_dir)
        self.architecture = self.manifest.get("architecture", PP_DOCLAYOUT_V3)
        if self.architecture not in _GRAPH_OUTPUTS:
            raise ConfigurationError(
                f"ONNX layout bundle {self.bundle_dir} declares architecture "
                f"{self.architecture!r}; bibr supports {sorted(_GRAPH_OUTPUTS)}"
            )
        self.output_names = _GRAPH_OUTPUTS[self.architecture]
        pre = self.manifest.get("preprocessing", {})
        size = pre.get("size", {"height": 800, "width": 800})
        self.size = (int(size["height"]), int(size["width"]))
        self.rescale_factor = float(pre.get("rescale_factor", 1.0 / 255.0))
        self.image_mean = [float(v) for v in pre.get("image_mean", [0.0, 0.0, 0.0])]
        self.image_std = [float(v) for v in pre.get("image_std", [1.0, 1.0, 1.0])]
        self.rescale_before_resize = bool(
            pre.get("rescale_before_resize", self.architecture == PP_DOCLAYOUT_V4)
        )
        self.threshold = threshold
        model_path = self.bundle_dir / self.manifest.get("model_file", ONNX_MODEL)
        self.session, self.device = create_session(
            model_path,
            device=device,
            model_name=f"layout ({architecture_label(self.architecture)} ONNX)",
        )
        self._input_name = self.session.get_inputs()[0].name
        outputs = {o.name for o in self.session.get_outputs()}
        for required in self.output_names:
            if required not in outputs:
                raise ValueError(
                    f"ONNX layout bundle {self.bundle_dir} lacks output {required!r} "
                    f"(has {sorted(outputs)})"
                )
        logger.info(
            "ONNX layout session ready (bundle=%s, model=%s, device=%s, input=%dx%d)",
            self.bundle_dir,
            architecture_label(self.architecture),
            self.device,
            self.size[0],
            self.size[1],
        )

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def close(self) -> None:
        self.session = None

    def forward(self, pixel_values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Raw graph outputs, in :attr:`output_names` order."""
        outs = self.session.run(list(self.output_names), {self._input_name: pixel_values})
        return tuple(outs)

    def run(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """Detect on PIL page images; one HF-shaped result dict per image."""
        if not images:
            return []
        orig_sizes = [(img.height, img.width) for img in images]
        pixel_values = preprocess_images(
            images,
            size=self.size,
            rescale_factor=self.rescale_factor,
            image_mean=self.image_mean,
            image_std=self.image_std,
            rescale_before_resize=self.rescale_before_resize,
        )
        outputs = self.forward(pixel_values)
        if self.architecture == PP_DOCLAYOUT_V4:
            return decode_detections_v4(*outputs, orig_sizes, self.threshold)
        return decode_detections(*outputs, orig_sizes, self.threshold)
