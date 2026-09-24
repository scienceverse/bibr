"""Shared PP-DocLayout layout-detector core (PP-DocLayoutV3, or V4 when selected).

``BaseLayoutDetector`` owns everything the local and serve variants have in
common: model loading, device resolution, the fixed-size padded PyTorch
forward pass, and detection postprocessing (threshold → NMS → large-image
filter → containment → read order). Variants layer lifecycle on top —
``bibr.local.layout`` adds explicit ``unload()`` for sequential GPU phases,
``bibr.serve.deployments.layout`` adds torch.compile + the GpuBatcher.

Two runtimes sit behind the same class (``bibr/utils/ml_runtime.py``): the
ONNX Runtime backend (:mod:`bibr.layout_onnx`, core install) and the original
torch/transformers path (``torch`` extra). ``_runtime`` records which one a
detector holds; ``_detect_images`` dispatches on it.

The checkpoint is ``LAYOUT_MODEL_ID`` at ``LAYOUT_MODEL_REVISION`` for torch and
the bundle manifest's ``architecture`` for ONNX. PP-DocLayoutV4 keeps V3's 25
labels and region contract; it regresses quadrilaterals (the enclosing
rectangle becomes ``bbox_2d``) and decodes reading order from two order heads
(:func:`bibr.layout_onnx.decode_detections_v4`, shared by both runtimes).
"""

import logging
import time
from types import SimpleNamespace

import numpy as np

from bibr.config import GlobalSettings, snapshot_settings
from bibr.layout_utils import (
    _CORRECT_ID2LABEL,
    _LABEL_TO_TASK,
    _compute_read_order,
    _compute_read_order_rb,
    _filter_containment,
    _nms,
    _resolve_overlaps_rulebook,
)

logger = logging.getLogger(__name__)

_IMAGE_LABEL = "image"

# Heading labels whose silent loss in overlap resolution costs a section
# downstream — worth an INFO trace (see the lead-reference containment RCA:
# a coincident pair once annihilated a real region with no log to show for it).
_HEADING_BOX_LABELS = frozenset({"doc_title", "paragraph_title"})


def _log_dropped_heading_boxes(
    boxes: "np.ndarray", keep_mask: "np.ndarray", id2label: dict[int, str]
) -> None:
    """Log heading boxes about to be dropped by overlap resolution.

    ``boxes`` rows are ``[label_id, score, x1, y1, x2, y2, ...]`` and
    ``keep_mask`` indexes them; body-text drops are routine dedup and stay
    silent.
    """
    for i in np.where(~keep_mask)[0]:
        label = id2label.get(int(boxes[i, 0]), "")
        if label in _HEADING_BOX_LABELS:
            x1, y1, x2, y2 = (int(v) for v in boxes[i, 2:6])
            logger.info(
                "Overlap resolution dropped a %s box at [%d, %d, %d, %d] — "
                "heading text may be lost",
                label,
                x1,
                y1,
                x2,
                y2,
            )


# Throttle window for handing VRAM back to a co-located OCR server.
_EMPTY_CACHE_MIN_INTERVAL_SECONDS = 30.0
_last_empty_cache_time = 0.0


def _maybe_empty_cache() -> None:
    """Throttled ``torch.cuda.empty_cache()`` (at most once per 30s).

    With ``expandable_segments`` the caching allocator returns freed physical
    pages to the driver, so this hands VRAM back to a sibling GPU process (the
    co-located OCR server). But a per-forward-batch sweep held the GIL — stalling
    the async event loop under concurrency — and defeated the CUDA caching
    allocator (every ≤4-page batch re-freed and re-mmapped segments). Throttling
    keeps the handback while amortizing the cost, mirroring
    ``ExportStage._maybe_gc_collect``.
    """
    global _last_empty_cache_time
    now = time.monotonic()
    if now - _last_empty_cache_time < _EMPTY_CACHE_MIN_INTERVAL_SECONDS:
        return
    _last_empty_cache_time = now
    import torch

    torch.cuda.empty_cache()


def _is_oom_error(exc: BaseException) -> bool:
    """True for a torch CUDA OOM or an ORT arena allocation failure."""
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "bfcarena" in message
        or "failed to allocate memory" in message
    )


def _check_label_map(id2label, source: str) -> None:
    """Refuse a checkpoint whose class list is not the 25 labels bibr maps.

    Every downstream rule keys on label *names* through ``_CORRECT_ID2LABEL``;
    a checkpoint with a reordered or extended class list would silently swap
    region types. ``None`` (V3 bundles and checkpoints) skips the check.
    """
    if id2label is None:
        return
    labels = {int(k): str(v) for k, v in dict(id2label).items()}
    if labels != _CORRECT_ID2LABEL:
        from bibr.exceptions import ConfigurationError

        changed = sorted(
            i
            for i in set(labels) | set(_CORRECT_ID2LABEL)
            if labels.get(i) != _CORRECT_ID2LABEL.get(i)
        )
        raise ConfigurationError(
            f"{source} has a different layout label list than bibr maps "
            f"(ids {changed[:8]} differ); update bibr.layout_utils._CORRECT_ID2LABEL "
            "and the label rules before using it."
        )


def resolve_layout_runtime(settings: GlobalSettings):
    """Apply the ``ML_RUNTIME`` rule to the layout model.

    The ONNX bundle lives in ``LAYOUT_ONNX_MODEL_ID`` at ``LAYOUT_ONNX_REVISION``
    (a bibr-owned repo or a local directory), since the torch weights are in
    a third-party repo bibr cannot add files to. The bundle's manifest names
    the architecture it was exported from, so it need not match
    ``LAYOUT_MODEL_ID``.
    """
    from bibr.utils.ml_runtime import find_onnx_bundle, hub_bundle_hint, resolve_runtime

    model_id = settings.layout.onnx_model_id
    revision = settings.layout.onnx_revision
    return resolve_runtime(
        "layout detector (PP-DocLayout)",
        settings=settings,
        bundle=lambda: find_onnx_bundle(model_id, revision, label="layout detector"),
        bundle_hint=hub_bundle_hint("LAYOUT_ONNX_MODEL_ID", model_id, revision),
    )


class BaseLayoutDetector:
    """PP-DocLayout wrapper: loading, inference, and postprocessing.

    Subclasses set ``_variant`` (used in device reporting) and implement
    ``_install_model(model)`` to take ownership of the loaded model (compile
    it, stash it, build batchers, ...), plus their own ``detect_batch``.
    """

    _variant = "base"

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float | None = None,
        device: str | None = None,
        settings: GlobalSettings | None = None,
    ):
        import os

        # Must be set before importing torch / first CUDA allocation.
        # With expandable_segments, the caching allocator uses virtual memory
        # mappings (cuMemMap/cuMemUnmap) instead of cudaMalloc, so
        # empty_cache() truly returns physical pages to the driver — making
        # them available to sibling GPU processes (e.g. the OCR server).
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        from bibr.utils.hf_cache import disable_hf_cache_symlinks_on_windows

        disable_hf_cache_symlinks_on_windows()
        self._settings = settings if settings is not None else snapshot_settings()
        self.threshold = (
            threshold if threshold is not None else self._settings.layout.detection_threshold
        )
        # Override the broken HF config id2label with the correct mapping.
        self._id2label = _CORRECT_ID2LABEL
        # Set True by variants that wrap the model in torch.compile — gates the
        # fixed-size batch padding (only compiled graphs recompile on new shapes).
        self._compiled: bool = False
        self._model = None
        self._image_processor = None

        # Pre-allocate a dummy image for batch padding (avoids per-call PIL
        # allocation).
        from PIL import Image

        self._pad_image = Image.new("RGB", (640, 480))

        runtime, bundle_dir = resolve_layout_runtime(self._settings)
        self._runtime = runtime
        if runtime == "onnx":
            self._init_onnx(bundle_dir, device)
        else:
            self._init_torch(model_id or self._settings.layout.model_id, device)

        logger.info("LayoutDetector ready (runtime=%s, device=%s)", self._runtime, self._device)
        from bibr.utils.device import report_device

        report_device(
            f"LayoutDetector ({self._variant}, {self._runtime})",
            self._device.type,
            gpu_capable=True,
        )

    def _init_onnx(self, bundle_dir, device: str | None) -> None:
        """Open the ONNX bundle; ``self._model`` is the ORT backend."""
        from bibr.layout_onnx import PP_DOCLAYOUT_V4, OnnxLayoutBackend

        backend = OnnxLayoutBackend(bundle_dir, device=device, threshold=self.threshold)
        id2label = backend.manifest.get("id2label")
        if id2label is None and backend.architecture == PP_DOCLAYOUT_V4:
            from bibr.exceptions import ConfigurationError

            raise ConfigurationError(
                f"ONNX layout bundle {bundle_dir} is PP-DocLayoutV4 but declares no id2label; "
                "re-export it with scripts/export_onnx_layout.py, which records the label list."
            )
        _check_label_map(id2label, f"ONNX layout bundle {bundle_dir}")
        source = (backend.manifest.get("source") or {}).get("repo_id")
        if source and source != self._settings.layout.model_id:
            # Under ONNX the bundle alone decides the model; LAYOUT_MODEL_ID and
            # LAYOUT_MODEL_REVISION only pin the torch runtime.
            logger.warning(
                "LAYOUT_MODEL_ID=%s is not used: the ONNX layout runtime loads %s, exported "
                "from %s. Set LAYOUT_ONNX_MODEL_ID/LAYOUT_ONNX_REVISION to change its model, "
                "or ML_RUNTIME=torch to load LAYOUT_MODEL_ID.",
                self._settings.layout.model_id,
                bundle_dir,
                source,
            )
        # ``.type`` is what the variants read (warmup, unload, cache handback).
        self._device = SimpleNamespace(type=backend.device)
        self._install_model(backend)

    def _init_torch(self, model_id: str, device: str | None) -> None:
        """Load the transformers model; ``self._model`` is the torch module."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as e:
            from bibr.utils.ml_extra import TORCH_EXTRA_HINT

            raise ImportError(
                "PDF processing, including cloud OCR, needs the layout detector. Its ONNX "
                "bundle was not selected (LAYOUT_ONNX_MODEL_ID / ML_RUNTIME), so it requires "
                f"the 'torch' extra: {TORCH_EXTRA_HINT}. The core install remains sufficient "
                "for native DOCX, JATS, HTML, and ePub inputs."
            ) from e

        if device is None:
            from bibr.utils.device import detect_torch_device

            device = detect_torch_device()
        self._device = torch.device(device)

        revision = self._settings.layout.model_revision
        self._image_processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModelForObjectDetection.from_pretrained(model_id, revision=revision)
        config = getattr(model, "config", None)
        if getattr(config, "model_type", None) == "pp_doclayout_v4":
            # V3's hub config collapses five labels (see _CORRECT_ID2LABEL), so
            # only V4, whose converter writes the real list, can be checked.
            _check_label_map(config.id2label, f"{model_id}@{revision}")
        if self._device.type != "cpu":
            model = model.to(self._device)
        if self._device.type == "cuda":
            # Callers may hand us an explicit "cuda" device, bypassing
            # detect_torch_device() — so enable the perf knobs here too.
            from bibr.utils.device import configure_cuda_perf

            configure_cuda_perf()
        model.eval()
        self._install_model(model)

    def _install_model(self, model) -> None:
        """Take ownership of the loaded, eval-mode model."""
        raise NotImplementedError

    def _detect_images(self, images: list) -> list[list[dict]]:
        """Run detection on a batch of PIL images (orig sizes from image dims)."""
        orig_sizes = [(img.height, img.width) for img in images]  # (h, w) for post_process
        try:
            if getattr(self, "_runtime", "torch") == "onnx":
                return self._detect_onnx(images, orig_sizes)
            return self._detect_pytorch(images, orig_sizes)
        except Exception as exc:
            if not _is_oom_error(exc) or len(images) <= 1:
                raise

            # Compiled inference pads every call back to the configured maximum.
            # Disable that padding before retrying or the smaller logical batch
            # would consume exactly the same memory and OOM again.
            if self._compiled:
                self._compiled = False
                logger.warning("Layout OOM disabled fixed-size compiled padding for retries")
            half = max(1, len(images) // 2)
            logger.warning(
                "Layout OOM with batch size %d, retrying as %d + %d",
                len(images),
                half,
                len(images) - half,
            )

        # Retry after leaving the exception handler: Python then clears the
        # traceback (which may retain failed-forward tensors and their VRAM).
        return self._detect_images(images[:half]) + self._detect_images(images[half:])

    def _detect_onnx(self, pil_images: list, orig_sizes: list[tuple[int, int]]) -> list[list[dict]]:
        """ONNX Runtime inference path: numpy pre/post-processing around one session run."""
        results = self._model.run(pil_images)
        return [
            self._postprocess(result, orig_w, orig_h)
            for result, (orig_h, orig_w) in zip(results, orig_sizes, strict=True)
        ]

    def _detect_pytorch(
        self, pil_images: list, orig_sizes: list[tuple[int, int]]
    ) -> list[list[dict]]:
        """PyTorch inference path for layout detection."""
        import torch

        real_count = len(pil_images)
        # Pad to a fixed batch size ONLY under torch.compile, so the compiled
        # graph never recompiles for a new batch shape. In the default eager
        # path padding would waste up to (_MAX_BATCH_SIZE - 1) forward passes on
        # dummy images.
        if self._compiled:
            pad_count = self._settings.layout.batch_size - real_count
            if pad_count > 0:
                pil_images = pil_images + [self._pad_image] * pad_count

        inputs = self._image_processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self._device.type,
                dtype=torch.float16,
                enabled=self._device.type in ("cuda", "mps"),
            ),
        ):
            outputs = self._model(**inputs)

        # Free input tensors immediately
        del inputs

        if getattr(outputs, "successor_order_logits", None) is not None:
            # PP-DocLayoutV4: bibr's numpy decode, which the ONNX path shares,
            # instead of the processor's (that one needs scipy for the reading
            # order graph). Only the real images are decoded.
            from bibr.layout_onnx import decode_detections_v4

            arrays = [
                getattr(outputs, name)[:real_count].float().cpu().numpy()
                for name in (
                    "logits",
                    "pred_boxes",
                    "relative_order_logits",
                    "successor_order_logits",
                )
            ]
            del outputs
            results = decode_detections_v4(*arrays, orig_sizes[:real_count], self.threshold)
        else:
            # Post-process only real images — slice model outputs to real_count
            # before HF post-processing to avoid decoding boxes for dummy padding.
            real_outputs = type(outputs)(**{k: v[:real_count] for k, v in outputs.items()})
            del outputs

            target_sizes = torch.tensor(orig_sizes[:real_count], dtype=torch.float32)
            results = self._image_processor.post_process_object_detection(
                real_outputs, threshold=self.threshold, target_sizes=target_sizes
            )
            del real_outputs, target_sizes

        all_results = []
        for result, (orig_h, orig_w) in zip(results, orig_sizes[:real_count], strict=True):
            regions = self._postprocess(result, orig_w, orig_h)
            all_results.append(regions)

        # Break reference cycles between HF post-process result tensors and
        # model output graph, then (throttled) release CUDA blocks to the driver.
        del results
        if self._device.type == "cuda":
            _maybe_empty_cache()

        return all_results

    def _postprocess(self, result: dict, orig_width: int, orig_height: int) -> list[dict]:
        """Convert post_process_object_detection output to region dicts.

        ``result`` has keys ``scores``, ``labels``, ``boxes`` ([x1,y1,x2,y2]
        in original pixel coords), and optionally ``order_seq`` (model-
        predicted reading order). After thresholding and normalisation to the
        0-1000 scale, cross-class NMS, the full-page-image filter, and
        overlap resolution suppress duplicate detections. Overlap resolution
        is selected by ``LAYOUT_OVERLAP_RESOLVER``: "legacy" (default) is the
        containment filter ported from the vendored SDK
        ``apply_layout_postprocess``; "rulebook" is the Docling-derived
        union-find resolver. Reading order uses the model's ``order_seq``
        when available (Global Pointer Mechanism), falling back to the
        spatial ordering selected by ``LAYOUT_READ_ORDER_FALLBACK``: "xy"
        (default, y-then-x lexsort) or "rb" (column-aware Docling-derived
        dilation + adjacency ordering).
        """
        raw_scores = result["scores"]
        scores = raw_scores.cpu().numpy() if hasattr(raw_scores, "cpu") else np.asarray(raw_scores)
        raw_labels = result["labels"]
        labels = raw_labels.cpu().numpy() if hasattr(raw_labels, "cpu") else np.asarray(raw_labels)
        raw_boxes = result["boxes"]
        boxes_px = raw_boxes.cpu().numpy() if hasattr(raw_boxes, "cpu") else np.asarray(raw_boxes)

        raw_order = result.get("order_seq")
        if raw_order is not None:
            order_seq = (
                raw_order.cpu().numpy() if hasattr(raw_order, "cpu") else np.asarray(raw_order)
            )
        else:
            order_seq = None

        # -- Pass 1: threshold and normalise coordinates --
        valid_rows = []
        valid_order = []  # parallel: model order for each valid row
        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.threshold:
                continue
            label_id = int(labels[i])
            if label_id not in self._id2label:
                continue

            # Normalise from original pixel space to 0-1000 scale
            x1 = max(0, min(int(float(boxes_px[i, 0]) * 1000.0 / orig_width), 1000))
            y1 = max(0, min(int(float(boxes_px[i, 1]) * 1000.0 / orig_height), 1000))
            x2 = max(0, min(int(float(boxes_px[i, 2]) * 1000.0 / orig_width), 1000))
            y2 = max(0, min(int(float(boxes_px[i, 3]) * 1000.0 / orig_height), 1000))

            # Skip degenerate boxes
            if x1 >= x2 or y1 >= y2:
                continue

            valid_rows.append([label_id, score, x1, y1, x2, y2])
            valid_order.append(int(order_seq[i]) if order_seq is not None else -1)

        if not valid_rows:
            return []

        boxes = np.array(valid_rows, dtype=np.float64)
        model_order = np.array(valid_order, dtype=np.int64)

        # -- Pass 2: NMS (cross-class + same-class) --
        kept_indices = _nms(
            boxes,
            iou_same=self._settings.layout.nms_iou_same,
            iou_diff=self._settings.layout.nms_iou_diff,
        )
        boxes = boxes[kept_indices]
        model_order = model_order[kept_indices]

        n_before_nms = len(valid_rows)
        n_after_nms = len(boxes)

        # -- Pass 2b: filter spurious full-page "image" detections --
        # A page-spanning "image" box is almost always a false positive.
        if len(boxes) > 1:
            is_landscape = orig_width > orig_height
            area_thresh = (
                self._settings.layout.large_image_area_landscape
                if is_landscape
                else self._settings.layout.large_image_area_portrait
            )
            # Coordinates are 0-1000 normalised; total area = 1_000_000.
            total_area = 1_000_000.0
            keep = []
            for i in range(len(boxes)):
                label_name = self._id2label.get(int(boxes[i, 0]), "")
                if label_name == _IMAGE_LABEL:
                    bx1, by1, bx2, by2 = boxes[i, 2], boxes[i, 3], boxes[i, 4], boxes[i, 5]
                    box_area = max(0, bx2 - bx1) * max(0, by2 - by1)
                    if box_area > area_thresh * total_area:
                        continue  # drop this oversized image box
                keep.append(i)
            if len(keep) < len(boxes):
                boxes = boxes[keep]
                model_order = model_order[keep]

        # -- Pass 3: overlap resolution --
        # "rulebook" = Docling-derived union-find resolver (keep-best, absorb
        # losers' bboxes); "legacy" (default) = per-category containment filter.
        if len(boxes) > 1:
            if self._settings.layout.overlap_resolver == "rulebook":
                keep_mask, boxes = _resolve_overlaps_rulebook(boxes, self._id2label)
                _log_dropped_heading_boxes(boxes, keep_mask, self._id2label)
                boxes = boxes[keep_mask]
                model_order = model_order[keep_mask]
            else:
                keep_mask = _filter_containment(boxes, self._id2label)
                if not keep_mask.all():
                    _log_dropped_heading_boxes(boxes, keep_mask, self._id2label)
                    boxes = boxes[keep_mask]
                    model_order = model_order[keep_mask]

        n_after_contain = len(boxes)
        if n_before_nms != n_after_contain:
            logger.debug(
                "NMS/containment: %d -> %d -> %d regions (threshold/NMS/containment)",
                n_before_nms,
                n_after_nms,
                n_after_contain,
            )

        # -- Determine read order --
        has_model_order = model_order[0] >= 0 if len(model_order) > 0 else False
        if has_model_order:
            read_orders = list(model_order)
        elif self._settings.layout.read_order_fallback == "rb":
            read_orders = _compute_read_order_rb(boxes)
        else:
            read_orders = _compute_read_order(boxes)

        # -- Build region dicts --
        regions = []
        for idx in range(len(boxes)):
            label_id = int(boxes[idx, 0])
            label = self._id2label.get(label_id, f"unknown_{label_id}")
            task_type = _LABEL_TO_TASK.get(label, "text")
            regions.append(
                {
                    "index": idx,
                    "label": label,
                    "task_type": task_type,
                    "score": float(boxes[idx, 1]),
                    "bbox_2d": [
                        int(boxes[idx, 2]),
                        int(boxes[idx, 3]),
                        int(boxes[idx, 4]),
                        int(boxes[idx, 5]),
                    ],
                    "read_order": int(read_orders[idx]),
                }
            )

        regions.sort(key=lambda r: r["read_order"])  # type: ignore[arg-type,return-value]

        # Re-index after sorting
        for idx, r in enumerate(regions):
            r["index"] = idx

        return regions
