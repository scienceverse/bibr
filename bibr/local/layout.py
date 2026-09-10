"""Standalone layout detector for local pipeline.

Local-pipeline variant of the shared PP-DocLayoutV3 core
(:class:`bibr.layout_base.BaseLayoutDetector`): supports explicit
``unload()`` for sequential GPU model loading across pipeline phases and
runs inference in the default thread executor.
"""

import gc
import logging

from bibr.layout.registry import register
from bibr.layout_base import BaseLayoutDetector
from bibr.layout_utils import _LABEL_TO_TASK, LABEL_TASK_MAPPING

logger = logging.getLogger(__name__)

# Re-export for local pipeline use
__all__ = ["LayoutDetector", "LABEL_TASK_MAPPING", "_LABEL_TO_TASK"]


@register
class LayoutDetector(BaseLayoutDetector):
    """Standalone PP-DocLayoutV3 layout detector.

    Local-pipeline variant: supports explicit ``unload()`` for VRAM
    recovery and auto-detects device (cuda → mps → cpu).
    """

    _variant = "local"
    name = "local"

    def __init__(self, *args, settings=None, **kwargs):
        super().__init__(*args, settings=settings, **kwargs)

    def _install_model(self, model) -> None:
        self._model = model
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def unload(self):
        """Release model and free GPU memory."""
        if not self._loaded:
            return

        if self._runtime == "onnx":
            # Dropping the session releases the ORT arena; nothing torch-side.
            self._model.close()
            self._model = None
            self._loaded = False
            gc.collect()
            logger.info("LayoutDetector unloaded (runtime=onnx, device=%s)", self._device.type)
            return

        import torch

        del self._model
        del self._image_processor
        self._model = None
        self._image_processor = None
        self._loaded = False

        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("LayoutDetector unloaded (device=%s)", self._device)

    async def detect_batch(self, images: list) -> list[list[dict]]:
        """Detect layout regions in a batch of PIL page images.

        Runs sync inference in a thread executor to avoid blocking the event loop.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._detect_sync, images)

    def _detect_sync(self, images: list) -> list[list[dict]]:
        """Synchronous batch detection in model-sized chunks."""
        if not images:
            return []

        batch_size = self._settings.layout.batch_size
        all_results: list[list[dict]] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            all_results.extend(self._detect_images(chunk))
        return all_results
