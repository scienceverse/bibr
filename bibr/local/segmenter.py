"""Standalone sentence segmenter for local pipeline.

Local-pipeline variant of the shared wtpsplit-lite SaT core
(:class:`bibr.segmenter_base.BaseSentenceSegmenter`): adds an explicit
``unload()`` for memory recovery across sequential pipeline phases and
serializes concurrent callers through a per-loop asyncio lock. Tiny model
(~0.1 GB) — typically stays resident.
"""

import logging

from bibr.segmenter.registry import register
from bibr.segmenter_base import BaseSentenceSegmenter

logger = logging.getLogger(__name__)


@register
class SentenceSegmenter(BaseSentenceSegmenter):
    """Standalone sentence segmenter using wtpsplit-lite SaT.

    Local-pipeline variant: supports explicit ``unload()`` for memory
    recovery and auto-detects GPU (CUDA) when available.
    """

    _variant = "local"
    name = "local"

    def __init__(
        self,
        model_name: str | None = None,
        use_gpu: bool | None = None,
        threshold: float | None = None,
        settings=None,
    ):
        # Serializes access to the non-thread-safe model; lazily bound to the
        # running loop in ``segment_batch`` (see ``_inference_lock``).
        self._infer_lock = None
        self._infer_lock_loop = None
        super().__init__(
            model_name=model_name,
            use_gpu=use_gpu,
            threshold=threshold,
            settings=settings,
        )
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def unload(self):
        """Release model and free memory."""
        if not self._loaded:
            return
        del self.model
        self.model = None
        self._loaded = False
        logger.info("SentenceSegmenter unloaded")

    def _inference_lock(self):
        """Return a per-event-loop ``asyncio.Lock`` serializing model access.

        Rebinds if the running loop changed (an ``asyncio.Lock`` binds to the
        loop that first awaits it), so reuse across loops — e.g. in tests — is
        safe. Construction is race-free: this method has no ``await``, so it
        runs atomically within the single-threaded event loop.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        lock = getattr(self, "_infer_lock", None)
        if lock is None or getattr(self, "_infer_lock_loop", None) is not loop:
            lock = asyncio.Lock()
            self._infer_lock = lock
            self._infer_lock_loop = loop
        return lock

    async def segment_batch(self, texts: list[str]) -> list[list[str]]:
        """Segment a batch of texts into sentences.

        Runs the blocking ONNX inference in a thread executor so it does
        not stall the pipeline's event loop. The model is not thread-safe and
        may be shared across concurrent callers, so entry is serialized.
        """
        if not texts:
            return []
        import asyncio

        loop = asyncio.get_running_loop()
        async with self._inference_lock():
            return await loop.run_in_executor(None, self._split_many, texts)
