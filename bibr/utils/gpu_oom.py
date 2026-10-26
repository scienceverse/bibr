"""Recognise GPU out-of-memory failures in bibr's own models, and split around them.

bibr's ONNX sessions (layout, SaT, the NER reference parser, the section
classifier) share the card with the managed vLLM servers. When it is nearly
full, ONNX Runtime fails a run in more than one way: the CUDA arena reports
``BFCArena::AllocateRawInternal ... Failed to allocate memory``, and a
cuDNN convolution's own allocation surfaced once as a bare ``CudaCall`` error
with ``ERRTYPE = cudaError`` and no description at all, in the same
millisecond as arena failures in two other sessions. Each model retries such a
failure with smaller batches (``split_on_gpu_oom``) and, for a single input,
on the CPU, instead of failing the paper or dropping to a weaker tier.

Real CUDA faults are not retried: ORT describes them (``CUDA failure 700: an
illegal memory access was encountered``), and a bare ``cudaError`` counts as
out-of-memory only while the card is nearly full or another out-of-memory
failure was just seen in this process.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

#: Message fragments (lower case) of an allocation failure: torch's and
#: CUDA's "out of memory" (``CUDA failure 2: out of memory``), ORT's arena,
#: and cuDNN's and cuBLAS's allocation statuses.
_OOM_MARKERS = (
    "out of memory",
    "bfcarena",
    "failed to allocate memory",
    "cudaerrormemoryallocation",
    "cuda_error_out_of_memory",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
)

#: A bare ORT ``CudaCall`` failure counts as out-of-memory while the card has
#: less than this free, or within ``_RECENT_OOM_S`` of a recognised one.
NEARLY_FULL_GB = 2.0
_RECENT_OOM_S = 10.0

_last_oom_at: float | None = None
_last_oom_lock = threading.Lock()


def _note_oom() -> None:
    global _last_oom_at
    with _last_oom_lock:
        _last_oom_at = time.monotonic()


def _recent_oom() -> bool:
    with _last_oom_lock:
        return _last_oom_at is not None and time.monotonic() - _last_oom_at <= _RECENT_OOM_S


def _bare_cuda_error(message: str) -> bool:
    """An ORT ``CudaCall`` failure with ``ERRTYPE = cudaError`` and no description."""
    return "errtype = cudaerror" in message and "cuda failure" not in message


def _card_nearly_full() -> bool:
    """Whether the managed servers' card has under ``NEARLY_FULL_GB`` free (False if unknown)."""
    from bibr.ocr.registry import managed_gpu_memory

    try:
        reading = managed_gpu_memory()
    except Exception:  # noqa: BLE001 - a probe failure must not mask the original error
        return False
    return reading is not None and reading.free_gb < NEARLY_FULL_GB


def is_gpu_oom_error(exc: BaseException) -> bool:
    """True for a GPU allocation failure that a smaller batch may avoid.

    Recognised from the message (see ``_OOM_MARKERS``); a bare ORT
    ``cudaError`` without a description only while the card is nearly full or
    shortly after another recognised failure. A recognised failure is
    remembered for that second check.
    """
    message = str(exc).lower()
    if any(marker in message for marker in _OOM_MARKERS):
        _note_oom()
        return True
    if _bare_cuda_error(message):
        return _recent_oom() or _card_nearly_full()
    return False


def split_on_gpu_oom(
    items: Sequence[T],
    run: Callable[[Sequence[T]], list[R]],
    *,
    label: str,
    run_on_cpu: Callable[[Sequence[T]], list[R]] | None = None,
) -> list[R]:
    """``run(items)``, halving the batch after a GPU out-of-memory failure.

    A single input that still fails goes to *run_on_cpu* when given, and
    otherwise re-raises. Any other error propagates unchanged. Each split and
    CPU fallback is logged as a warning naming *label*.
    """
    try:
        return run(items)
    except Exception as exc:
        if not is_gpu_oom_error(exc) or (len(items) <= 1 and run_on_cpu is None):
            raise
        reason = type(exc).__name__
    # Retried outside the handler, so the failed run's traceback (and any
    # buffers it holds) is released first.
    if len(items) <= 1:
        logger.warning(
            "%s GPU out of memory on a single input (%s); running it on the CPU", label, reason
        )
        assert run_on_cpu is not None  # noqa: S101 - checked in the handler above
        return run_on_cpu(items)
    half = len(items) // 2
    logger.warning(
        "%s GPU out of memory with batch size %d (%s), retrying as %d + %d",
        label,
        len(items),
        reason,
        half,
        len(items) - half,
    )
    return split_on_gpu_oom(
        items[:half], run, label=label, run_on_cpu=run_on_cpu
    ) + split_on_gpu_oom(items[half:], run, label=label, run_on_cpu=run_on_cpu)
