"""ONNX Runtime backend for the PP-DocLayoutV3 layout detector (torch-free).

Replicates the parts of transformers' ``PPDocLayoutV3ImageProcessor`` that
bibr's detector uses, in numpy:

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
``pred_boxes`` and ``order_logits``. The result dicts feed
:meth:`bibr.layout_base.BaseLayoutDetector._postprocess` unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bibr.utils.ml_runtime import ONNX_MODEL, read_onnx_manifest

logger = logging.getLogger(__name__)

# PyTorch's bicubic coefficient (torch/csrc/api/... UpSampleBicubic2d: A = -0.75).
_CUBIC_A = -0.75


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


def _resize_plan(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Clamped tap indices ``(out, 4)`` and float32 weights ``(out, 4)`` for one axis."""
    scale = in_size / out_size
    dst = np.arange(out_size, dtype=np.float64)
    src = (dst + 0.5) * scale - 0.5  # align_corners=False, cubic: no clamp at 0
    base = np.floor(src)
    t = src - base
    idx = base[:, None].astype(np.int64) + np.array([-1, 0, 1, 2], dtype=np.int64)[None, :]
    idx = np.clip(idx, 0, in_size - 1)
    return idx, _cubic_coefficients(t).astype(np.float32)


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255)


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
    c, h, w = chw.shape
    src = chw.astype(np.float32)
    if (h, w) != (out_h, out_w):
        ix, wx = _resize_plan(w, out_w)
        iy, wy = _resize_plan(h, out_h)
        # x pass: (C, H, out_w)
        tmp = np.zeros((c, h, out_w), dtype=np.float32)
        for k in range(4):
            tmp += src[:, :, ix[:, k]] * wx[None, None, :, k]
        tmp = _to_uint8(tmp)
        # y pass: (C, out_h, out_w)
        out = np.zeros((c, out_h, out_w), dtype=np.float32)
        for k in range(4):
            out += tmp[:, iy[:, k], :] * wy[None, :, k, None]
        src = out
    return _to_uint8(src).astype(np.uint8)


def preprocess_images(
    images: list[Image.Image],
    *,
    size: tuple[int, int],
    rescale_factor: float,
    image_mean: list[float],
    image_std: list[float],
) -> np.ndarray:
    """PIL images → ``(B, 3, H, W)`` float32 ``pixel_values``."""
    out_h, out_w = size
    batch = np.empty((len(images), 3, out_h, out_w), dtype=np.float32)
    mean = np.asarray(image_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(image_std, dtype=np.float32)[:, None, None]
    for i, image in enumerate(images):
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        chw = np.asarray(rgb, dtype=np.uint8).transpose(2, 0, 1)
        resized = resize_bicubic_no_antialias(chw, out_h, out_w)
        batch[i] = (resized.astype(np.float32) * np.float32(rescale_factor) - mean) / std
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


def decode_detections(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    order_logits: np.ndarray,
    orig_sizes: list[tuple[int, int]],
    threshold: float,
) -> list[dict[str, np.ndarray]]:
    """Raw graph outputs → per-image ``{scores, labels, boxes, order_seq}``.

    Mirrors ``post_process_object_detection`` minus the polygon masks: boxes
    are ``xyxy`` in original pixels, rows are thresholded and sorted by the
    model's reading order.
    """
    batch, num_queries, num_classes = logits.shape
    scores_all = _sigmoid(logits.astype(np.float32)).reshape(batch, -1)
    k = num_queries
    # torch.topk(sorted=True): the k best, descending. argpartition + stable
    # sort of the selected block reproduces it (ties are measure-zero here).
    part = np.argpartition(-scores_all, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores_all, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    index = np.take_along_axis(part, order, axis=1)
    scores = np.take_along_axis(scores_all, index, axis=1)
    labels = index % num_classes
    queries = index // num_classes

    centers = pred_boxes[..., :2]
    dims = pred_boxes[..., 2:]
    boxes = np.concatenate([centers - 0.5 * dims, centers + 0.5 * dims], axis=-1)
    seq = order_sequences(order_logits)

    results: list[dict[str, np.ndarray]] = []
    for b in range(batch):
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
        pre = self.manifest.get("preprocessing", {})
        size = pre.get("size", {"height": 800, "width": 800})
        self.size = (int(size["height"]), int(size["width"]))
        self.rescale_factor = float(pre.get("rescale_factor", 1.0 / 255.0))
        self.image_mean = [float(v) for v in pre.get("image_mean", [0.0, 0.0, 0.0])]
        self.image_std = [float(v) for v in pre.get("image_std", [1.0, 1.0, 1.0])]
        self.threshold = threshold
        model_path = self.bundle_dir / self.manifest.get("model_file", ONNX_MODEL)
        self.session, self.device = create_session(
            model_path, device=device, model_name="layout (PP-DocLayoutV3 ONNX)"
        )
        self._input_name = self.session.get_inputs()[0].name
        outputs = {o.name for o in self.session.get_outputs()}
        for required in ("logits", "pred_boxes", "order_logits"):
            if required not in outputs:
                raise ValueError(
                    f"ONNX layout bundle {self.bundle_dir} lacks output {required!r} "
                    f"(has {sorted(outputs)})"
                )
        logger.info(
            "ONNX layout session ready (bundle=%s, device=%s, input=%dx%d)",
            self.bundle_dir,
            self.device,
            self.size[0],
            self.size[1],
        )

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def close(self) -> None:
        self.session = None

    def forward(self, pixel_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outs = self.session.run(
            ["logits", "pred_boxes", "order_logits"], {self._input_name: pixel_values}
        )
        return outs[0], outs[1], outs[2]

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
        )
        logits, boxes, order_logits = self.forward(pixel_values)
        return decode_detections(logits, boxes, order_logits, orig_sizes, self.threshold)
