"""
SentenceSegmenter — serve variant of the shared wtpsplit-lite SaT core
(:class:`bibr.segmenter_base.BaseSentenceSegmenter`).

Dispatches to CUDAExecutionProvider when a GPU is available (the default),
or CPUExecutionProvider otherwise. ``use_gpu`` defaults to ``None``
(auto-detect); pass ``False`` to force CPU on VRAM-constrained boxes that
co-locate the OCR server. CPU inference is *not* cheap on large inputs —
segmenting a 137k-char paper took ~197s on CPU vs ~0.7s on GPU — so the
accelerator is used whenever present. Concurrent requests coalesce through
a shared :class:`bibr.serve.batching.GpuBatcher`.
"""

import logging

from bibr.segmenter_base import BaseSentenceSegmenter

logger = logging.getLogger(__name__)


class SentenceSegmenter(BaseSentenceSegmenter):
    """Sentence segmentation via wtpsplit-lite ONNX (serve variant)."""

    _variant = "serve"

    def __init__(
        self,
        model_name: str | None = None,
        use_gpu: bool | None = None,
        threshold: float | None = None,
        settings=None,
    ):
        # Concurrent ``segment_batch`` callers route texts through a GpuBatcher
        # (lazily built in ``_get_batcher``): a single collector + single-thread
        # executor serialize access to the non-thread-safe wtpsplit/ONNX model
        # and coalesce concurrent requests' texts into one call.
        self._batcher = None
        self._gpu_executor = None
        super().__init__(
            model_name=model_name,
            use_gpu=use_gpu,
            threshold=threshold,
            settings=settings,
        )

    @property
    def loaded(self) -> bool:
        # Serve worker constructs eagerly and never unloads — duck-types
        # against the local SentenceSegmenter for ResourceManager checks.
        return True

    def _get_batcher(self):
        """Lazily build the micro-batcher over ``_split_many``.

        Built on first use (not in ``__init__``) so it binds nothing to a loop
        until a request submits; rebuilt only if absent. No ``await`` here, so
        it runs atomically on the single-threaded event loop.
        """
        batcher = getattr(self, "_batcher", None)
        if batcher is None:
            from concurrent.futures import ThreadPoolExecutor

            from bibr.serve.batching import GpuBatcher

            self._gpu_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="segmenter-gpu"
            )
            timeout = max(0.0, self._settings.SEGMENTER_BATCH_TIMEOUT_MS / 1000.0)
            batcher = GpuBatcher(
                self._split_many,
                max_batch_size=self._settings.SEGMENTER_SUB_BATCH_SIZE,
                batch_timeout=timeout,
                executor=self._gpu_executor,
                name="segmenter",
            )
            self._batcher = batcher
        return batcher

    async def segment_batch(self, texts: list[str]) -> list[list[str]]:
        """Segment a pre-collected batch of texts in one call.

        Each text is submitted to the shared GpuBatcher; the single collector +
        single-thread executor serialize entry into the non-thread-safe
        wtpsplit/ONNX model (shared across concurrent serve requests and files
        in one parse chunk) while coalescing concurrent texts into fuller calls.
        Result order matches ``texts``.
        """
        if not texts:
            return []
        import asyncio

        batcher = self._get_batcher()
        return await asyncio.gather(*(batcher.submit(t) for t in texts))

    async def aclose(self) -> None:
        """Release the batcher + GPU executor (best-effort worker teardown)."""
        batcher = getattr(self, "_batcher", None)
        if batcher is not None:
            await batcher.close()
        executor = getattr(self, "_gpu_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
