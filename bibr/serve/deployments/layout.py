"""
LayoutDetector — serve variant of the shared PP-DocLayoutV3 core
(:class:`bibr.layout_base.BaseLayoutDetector`).

On GPU, optionally wraps the model with ``torch.compile(mode="default")``
when ``LAYOUT_TORCH_COMPILE`` is set (otherwise eager), and routes all
forward passes through a shared :class:`bibr.serve.batching.GpuBatcher` so
pages from concurrent requests coalesce into fuller padded batches while a
single GPU thread bounds peak VRAM to one batch.
"""

import logging

from bibr.layout_base import BaseLayoutDetector
from bibr.layout_utils import (
    _CORRECT_ID2LABEL,
    _IMAGE_LABEL,
    _LABEL_TO_TASK,
    _LARGE_IMAGE_AREA_LANDSCAPE,
    _LARGE_IMAGE_AREA_PORTRAIT,
    _MAX_BATCH_SIZE,
    _MERGE_BBOXES_MODE,
    _NMS_IOU_DIFF,
    _NMS_IOU_SAME,
    _PRESERVE_LABELS,
    DEFAULT_THRESHOLD,
    LABEL_TASK_MAPPING,
    _compute_read_order,
    _filter_containment,
    _iou,
    _is_contained,
    _nms,
)

logger = logging.getLogger(__name__)

# Labels that should be treated as figure/table/chart captions
_CAPTION_LABELS = {"figure_title"}

__all__ = [
    "DEFAULT_THRESHOLD",
    "LABEL_TASK_MAPPING",
    "LayoutDetector",
    "_CAPTION_LABELS",
    "_CORRECT_ID2LABEL",
    "_IMAGE_LABEL",
    "_LABEL_TO_TASK",
    "_LARGE_IMAGE_AREA_LANDSCAPE",
    "_LARGE_IMAGE_AREA_PORTRAIT",
    "_MAX_BATCH_SIZE",
    "_MERGE_BBOXES_MODE",
    "_NMS_IOU_DIFF",
    "_NMS_IOU_SAME",
    "_PRESERVE_LABELS",
    "_compute_read_order",
    "_filter_containment",
    "_iou",
    "_is_contained",
    "_nms",
]


class LayoutDetector(BaseLayoutDetector):
    """PP-DocLayoutV3 layout detection model wrapper (serve variant)."""

    _variant = "serve"

    def _install_model(self, model) -> None:
        if self._runtime == "onnx":
            # ORT owns graph optimisation; torch.compile does not apply. The
            # warmup still matters on CUDA (arena growth, cuDNN algorithm
            # search inside the EP).
            self._model = model
            if self._device.type == "cuda":
                self._run_warmup()
        else:
            if self._settings.layout.torch_compile and self._device.type == "cuda":
                model = self._compile_model(model)
            else:
                logger.info("torch.compile disabled for layout (LAYOUT_TORCH_COMPILE=false)")
            self._model = model
            # Warm up on GPU whether compiled or eager: the first real forward
            # otherwise pays cuDNN autotuning + CUDA allocator growth + GPU clock
            # ramp, which surfaces as an inflated layout time on the first paper(s)
            # after a (re)start. Running it here moves that cost into setup(), inside
            # the healthcheck start_period, so real requests hit a warm model.
            if self._device.type == "cuda":
                self._run_warmup()

        # Single GPU thread + a single batching collector serialize every layout
        # forward, so peak VRAM is bounded to one batch no matter how many
        # requests are in flight; the batcher also coalesces concurrent
        # requests' pages into fuller padded batches. See bibr/serve/batching.py.
        from concurrent.futures import ThreadPoolExecutor

        self._gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="layout-gpu")
        self._batcher = self._build_batcher()

    def _build_batcher(self):
        """Construct the per-detector micro-batcher over ``_detect_images``."""
        from bibr.serve.batching import GpuBatcher

        timeout = max(0.0, self._settings.layout.batch_timeout_ms / 1000.0)
        return GpuBatcher(
            self._detect_images,
            max_batch_size=self._settings.layout.batch_size,
            batch_timeout=timeout,
            executor=getattr(self, "_gpu_executor", None),
            name="layout",
        )

    async def aclose(self) -> None:
        """Release the batcher + GPU executor (best-effort worker teardown)."""
        await self._batcher.close()
        executor = getattr(self, "_gpu_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

    @property
    def loaded(self) -> bool:
        # Serve worker constructs eagerly and never unloads — duck-types
        # against the local LayoutDetector for ResourceManager checks.
        return True

    def _compile_model(self, model):
        """Wrap the model in ``torch.compile(mode="default")`` for Inductor fusion.

        Returns the eager model unchanged on failure.
        """
        import torch

        try:
            model = torch.compile(model, mode="default")
            # Compiled graphs recompile on new input shapes, so enable fixed-size
            # batch padding (see BaseLayoutDetector._detect_pytorch).
            self._compiled = True
            logger.info("torch.compile(default) applied to layout model")
        except Exception:
            logger.warning("torch.compile not available, using eager mode", exc_info=True)
        return model

    def _run_warmup(self):
        """Run one forward pass to warm the GPU before real traffic.

        Primes cuDNN autotuning, grows the CUDA caching allocator, and ramps GPU
        clocks out of idle — costs the first real forward would otherwise pay.
        Runs for both the eager and torch.compile paths (compile additionally
        needs it to trigger Inductor autotuning). Best-effort: never fatal.
        """
        from PIL import Image

        if self._runtime == "onnx":
            try:
                self._model.run([Image.new("RGB", (640, 480))] * self._settings.layout.batch_size)
                logger.info("Layout model warmup complete (onnxruntime)")
            except Exception:
                logger.warning("Layout model warmup failed", exc_info=True)
            return

        import torch

        try:
            dummy = Image.new("RGB", (640, 480))
            inputs = self._image_processor(
                images=[dummy] * self._settings.layout.batch_size,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.inference_mode():
                self._model(**inputs)
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("Layout model warmup complete (autotuning finished)")
        except Exception:
            logger.warning("Layout model warmup failed", exc_info=True)

    async def detect_batch(self, images: list) -> list[list[dict]]:
        """Detect layout on a batch of PIL images via the shared GpuBatcher.

        Pages are submitted individually so that pages from *concurrent requests*
        coalesce into the same forward pass; the single collector + single-thread
        GPU executor serialize those passes, bounding peak VRAM to one batch.
        Under torch.compile each fn call pads to _MAX_BATCH_SIZE (see
        ``_detect_pytorch``) so the graph never recompiles. Result order matches
        ``images``.
        """
        if not images:
            return []

        import asyncio

        return await asyncio.gather(*(self._batcher.submit(img) for img in images))
