"""Size the managed vLLM LLM server's GPU share from the memory free at each start.

vLLM's ``--gpu-memory-utilization`` is a share of the card's whole CUDA
total, and vLLM refuses to start unless that much is free. Its real footprint
is larger than the share: the CUDA context (about 0.4 GiB) lies outside it,
and CUDA graph capture, which runs after the KV cache is allocated, briefly
needs more. On an RTX 3090 (23.56 GiB to CUDA) the 4B student at the fixed
0.85 share (20.03 GiB) peaked at 21.20-21.32 GiB during capture with vLLM's
default 256 sequences (capture sizes up to 512), and 0.54 (12.72 GiB) with
128 sequences peaked 0.75 GiB above its share. The capture transient follows
the capture sizes, not the share.

When OCR and the LLM swap the card every chunk, the LLM restarts beside
bibr's own GPU sessions (layout, SaT, the NER parser, the classifiers), which
grew from about 0.5 to 2.3-3.1 GiB over a 100-paper batch on that card. vLLM's
startup check still passed at 0.85, and capture then ran out of memory once
bibr held more than about 2.2 GiB: every paper in that chunk failed. So
unless the user set the share, the managed server's share is worked out from
what is free when it starts (``size_managed_llm_share``), and a start that
cannot meet the model's floor fails with ``ManagedLlmVramError`` before vLLM
is launched.
"""

from __future__ import annotations

import logging
import math
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bibr.exceptions import UpstreamServiceError

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.ocr.registry import GpuMemory

logger = logging.getLogger(__name__)

#: VRAM a vLLM server takes beyond its share at its peak, before any CUDA
#: graph is captured: the CUDA context, about 0.4 GiB on an RTX 3090.
LLM_CONTEXT_RESERVE_GB = 0.5

#: Further VRAM vLLM's CUDA graph capture briefly takes beyond the share at
#: its largest capture size, which vLLM sets to twice ``--max-num-seqs``,
#: capped at 512. Measured peaks above the share on an RTX 3090: 1.17-1.29
#: GiB at 512 (0.85, default 256 sequences), 0.75 GiB at 256 (0.54, 128
#: sequences). Scaled linearly, with the context above: 1.5 and 1.0 GiB.
LLM_CAPTURE_RESERVE_GB_AT_512 = 1.0

#: Room always left beside the LLM's share for bibr's own GPU sessions to
#: grow within a chunk. It is all the room an LLM-only run leaves (no managed
#: PaddleOCR-VL server takes turns with the LLM), so the configured share
#: (0.85) still fits a free 24 GB card.
LLM_ONNX_MIN_HEADROOM_GB = 1.0

#: vLLM's ``--max-num-seqs`` when neither bibr nor the user sets it.
_VLLM_DEFAULT_MAX_NUM_SEQS = 256
_VLLM_MAX_CAPTURE_SIZE = 512

#: Re-reads while free memory is still rising (a server that just stopped
#: may still be releasing its memory), and the wait between them.
_SETTLE_READS = 20
_SETTLE_INTERVAL_S = 0.5
_SETTLE_RISE_GB = 0.25


class ManagedLlmVramError(UpstreamServiceError):
    """Too little free VRAM to start the managed vLLM LLM server at its floor."""

    def __init__(self, message: str) -> None:
        super().__init__("llm", message)


@dataclass(frozen=True)
class LlmShare:
    """A ``--gpu-memory-utilization`` for one start and how it was reached."""

    share: float
    #: ``cap``: the configured share fitted. ``free``: sized down to what is
    #: free. ``floor``: squeezed to the model's floor, leaving bibr only
    #: ``LLM_ONNX_MIN_HEADROOM_GB`` to grow into (also when the share sized
    #: from free memory rounds down to just under the floor). ``unknown``: no
    #: reading, so the configured share as before.
    reason: str
    budget_gb: float | None = None
    total_gb: float | None = None
    free_gb: float | None = None
    own_gb: float | None = None
    reserve_gb: float | None = None
    headroom_gb: float | None = None
    floor_gb: float | None = None

    def describe(self) -> str:
        if self.total_gb is None:
            return f"{self.share:.2f} (no GPU memory reading; the configured share)"
        floor = "none" if self.floor_gb is None else f"{self.floor_gb:.1f} GiB"
        return (
            f"{self.share:.2f} of {self.total_gb:.2f} GiB ({self.reason}): "
            f"{self.free_gb:.2f} GiB free, bibr {self.own_gb:.2f} GiB, "
            f"vLLM overshoot reserve {self.reserve_gb:.2f} GiB, "
            f"bibr headroom {self.headroom_gb:.2f} GiB, floor {floor}"
        )


def _last_flag_value(args: list[str], flag: str) -> str | None:
    value = None
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
    return value


def _server_args(model: str, extra_args: str | None) -> list[str]:
    """The launch args after the model: registry args, then user extra args."""
    from bibr.local.llm_models import registry_server_args

    return [*registry_server_args(model), *shlex.split(extra_args or "")]


def overshoot_reserve_gb(model: str, extra_args: str | None) -> float:
    """VRAM a vLLM server for *model* may take beyond its share while it starts."""
    args = _server_args(model, extra_args)
    if "--enforce-eager" in args:
        return LLM_CONTEXT_RESERVE_GB
    raw = _last_flag_value(args, "--max-num-seqs")
    try:
        seqs = int(raw) if raw is not None else _VLLM_DEFAULT_MAX_NUM_SEQS
    except ValueError:
        seqs = _VLLM_DEFAULT_MAX_NUM_SEQS
    capture = min(2 * max(1, seqs), _VLLM_MAX_CAPTURE_SIZE)
    return LLM_CONTEXT_RESERVE_GB + LLM_CAPTURE_RESERVE_GB_AT_512 * capture / _VLLM_MAX_CAPTURE_SIZE


def managed_llm_floor_gb(model: str) -> float | None:
    """The registry's minimum VRAM for *model* served by vLLM, None if it is not listed.

    It covers the weights, vLLM's profiling activations and the KV cache the
    server needs to start (one Mamba block per sequence for the hybrid
    Qwen3.5 students, and ``--max-model-len`` tokens): 12 GB for the 4B
    student, rounded up from its measured 11.7 GB low-memory peak.
    """
    from bibr.local.llm_models import REGISTRY

    for entry in REGISTRY:
        for variant in entry.variants:
            if variant.platform == "cuda" and variant.runtime == "vllm" and variant.hf_id == model:
                return float(variant.min_vram_gb)
    return None


def llm_share_user_set(settings: GlobalSettings) -> bool:
    """Whether the user chose the LLM server's share, which then always wins.

    ``LLM_LOCAL_MEM_FRACTION`` (env, ``.env`` or an assigned setting) or a
    ``--gpu-memory-utilization`` in ``LLM_VLLM_EXTRA_ARGS``. Read it before
    bibr writes a planned share onto the settings.
    """
    llm = settings.llm
    return "local_mem_fraction" in llm.model_fields_set or "--gpu-memory-utilization" in (
        llm.vllm_extra_args or ""
    )


def size_managed_llm_share(
    reading: GpuMemory | None,
    *,
    cap: float,
    floor_gb: float | None,
    reserve_gb: float,
    swaps_with_ocr: bool,
) -> LlmShare:
    """The share for a managed vLLM LLM server starting now.

    The budget is the configured share (*cap*) of the card or, when less is
    free, what is free less the server's overshoot reserve and room for bibr's
    sessions to grow. That room depends on what else uses the card. When a
    managed PaddleOCR-VL server takes turns with the LLM (*swaps_with_ocr*),
    the LLM restarts every chunk beside bibr's grown sessions, and the room is
    ``MANAGED_VLLM_ONNX_RESERVE_GB`` less what bibr already holds (at least
    ``LLM_ONNX_MIN_HEADROOM_GB``). Otherwise it is ``LLM_ONNX_MIN_HEADROOM_GB``:
    an LLM-only run shares the card with bibr's own sessions alone (what they
    hold is already not free), and shares planned to run side by side already
    leave that reserve out. Below the model's floor, the budget is squeezed to
    the floor while the minimum room still fits; otherwise
    ``ManagedLlmVramError``.
    """
    from bibr.ocr.registry import MANAGED_VLLM_ONNX_RESERVE_GB

    if reading is None or reading.total_gb <= 0:
        return LlmShare(share=cap, reason="unknown")
    total, free, own = reading.total_gb, reading.free_gb, reading.own_gb
    headroom = (
        max(LLM_ONNX_MIN_HEADROOM_GB, MANAGED_VLLM_ONNX_RESERVE_GB - own)
        if swaps_with_ocr
        else LLM_ONNX_MIN_HEADROOM_GB
    )
    cap_gb = cap * total
    floor = None if floor_gb is None else min(floor_gb, cap_gb)
    target = free - reserve_gb - headroom
    details = {
        "total_gb": total,
        "free_gb": free,
        "own_gb": own,
        "reserve_gb": reserve_gb,
        "headroom_gb": headroom,
        "floor_gb": floor_gb,
    }
    if target >= cap_gb:
        return LlmShare(share=cap, reason="cap", budget_gb=cap_gb, **details)
    if target > 0 and (floor is None or target >= floor):
        # Round down so the reserve and headroom stay free. Rounding down can
        # land just under the floor (on an RTX 3090, 12.008 GiB gives 0.50, or
        # 11.78 GiB, below the 4B's 12): such a start takes the floor path.
        share = math.floor(round(target / total * 100, 6)) / 100
        if share > 0 and (floor is None or share * total >= floor):
            return LlmShare(share=share, reason="free", budget_gb=share * total, **details)
    squeezed = free - reserve_gb - LLM_ONNX_MIN_HEADROOM_GB
    if floor is not None and squeezed >= floor:
        # Round up so the model gets its floor; the minimum headroom absorbs it.
        share = min(cap, math.ceil(round(floor / total * 100, 6)) / 100)
        details["headroom_gb"] = LLM_ONNX_MIN_HEADROOM_GB
        return LlmShare(share=share, reason="floor", budget_gb=share * total, **details)
    need = (floor if floor is not None else 0.0) + reserve_gb + LLM_ONNX_MIN_HEADROOM_GB
    raise ManagedLlmVramError(
        f"not enough free GPU memory to start the managed vLLM LLM server: "
        f"{free:.2f} of {total:.2f} GiB free (bibr itself holds {own:.2f} GiB), and it needs "
        f"{need:.2f} GiB (the model's {floor or 0.0:.1f} GiB floor, {reserve_gb:.2f} GiB vLLM "
        f"takes beyond its share while starting, and {LLM_ONNX_MIN_HEADROOM_GB:.1f} GiB for "
        "bibr's GPU sessions). Free GPU memory (other processes), pick a smaller model "
        "(LLM_LOCAL_MODEL), or set LLM_LOCAL_MEM_FRACTION to choose the share yourself"
    )


def settled_gpu_memory(
    probe: Callable[[], GpuMemory | None],
    *,
    wait: Callable[[float], object],
) -> GpuMemory | None:
    """A reading taken once free memory stops rising.

    A managed server that just stopped (PaddleOCR-VL handing the card to the
    LLM) can still be releasing memory when its launcher exits. *wait* sleeps
    (and may raise to stop early).
    """
    reading = probe()
    if reading is None:
        return None
    for _ in range(_SETTLE_READS):
        wait(_SETTLE_INTERVAL_S)
        later = probe()
        if later is None:
            return reading
        rising = later.free_gb > reading.free_gb + _SETTLE_RISE_GB
        reading = later
        if not rising:
            break
    return reading
