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

A server that stays up for the run (no managed PaddleOCR-VL server takes
turns with it: an OCR server at an OCR URL, cloud OCR, GLM-OCR) starts in
the first chunk, before bibr's sessions have grown. Beside an external
PaddleOCR-VL server on an RTX 5090 it took all but 1 GiB of what was free
while bibr held 1.03 GiB; bibr's sessions then ran out of memory at about
3.3 GiB, and 31 and 30 of 48 papers failed in two runs (vLLM 0.32 at 64
sequences, sized by the same rule). So every start leaves bibr's sessions
``MANAGED_VLLM_ONNX_RESERVE_GB`` less what they hold, and only a start that
lasts one chunk (the swap) may be squeezed below that to reach the floor.
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
#: grow within a chunk, however much they already hold. Otherwise the room is
#: ``MANAGED_VLLM_ONNX_RESERVE_GB`` less what they hold, and this is all a
#: start that takes turns with PaddleOCR-VL keeps when it is squeezed to the
#: model's floor (it lasts one chunk).
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
    #: free. ``floor``: rounded up to the model's floor. A start that takes
    #: turns with PaddleOCR-VL may be squeezed there, leaving bibr only
    #: ``LLM_ONNX_MIN_HEADROOM_GB`` to grow into; any start whose share sized
    #: from free memory rounds down to just under the floor gets it too, which
    #: takes at most a hundredth of the card from bibr's room. ``unknown``: no
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
        floor = "none" if self.floor_gb is None else f"{self.floor_gb:.2f} GiB"
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


def _registry_floor_gb(model: str) -> float | None:
    """The registry's vLLM budget floor for *model*, None if it is not listed.

    The variant's ``min_budget_gb``, or its ``min_vram_gb`` card size when it
    has none (the floor before budget floors were measured).
    """
    from bibr.local.llm_models import REGISTRY

    for entry in REGISTRY:
        for variant in entry.variants:
            if variant.platform == "cuda" and variant.runtime == "vllm" and variant.hf_id == model:
                if variant.min_budget_gb is not None:
                    return float(variant.min_budget_gb)
                return float(variant.min_vram_gb)
    return None


# Models whose lower LLM_LOCAL_MIN_VRAM_GB was already reported as ignored: once
# per process each, since every start and every chunk's check reads the floor.
_floor_setting_ignored_for: set[str] = set()


def managed_llm_floor_gb(model: str, settings: GlobalSettings | None = None) -> float | None:
    """The vLLM budget floor for *model*, None if it is not known.

    The registry's budget floor for the model's vLLM variant
    (``LocalLLMVariant.min_budget_gb``): the weights, vLLM's profiling
    activations with an empty compile cache, and the KV cache the server
    needs to start (one Mamba block per sequence for the hybrid Qwen3.5
    students, and ``--max-model-len`` tokens). ``LLM_LOCAL_MIN_VRAM_GB``
    (*settings*) sets the floor of a model the registry does not list, such
    as a local path, and can raise a listed model's floor but not lower it:
    a lower value is ignored with a warning, so a value left over from a
    local build (6.25 for the W4A16 build) never sizes the bf16 4B's starts
    below what it needs. This matches ``LLM_LOCAL_MEM_FRACTION``, the way to
    choose a share yourself.
    """
    floor = _registry_floor_gb(model)
    setting = settings.llm.local_min_vram_gb if settings is not None else None
    if setting is None:
        return floor
    if floor is None or setting >= floor:
        return float(setting)
    if model not in _floor_setting_ignored_for:
        _floor_setting_ignored_for.add(model)
        logger.warning(
            "LLM_LOCAL_MIN_VRAM_GB=%g is ignored for %s: bibr's model registry lists that "
            "model with a minimum of %g GiB, and the setting can raise a listed model's "
            "minimum but not lower it. Unset it, or set LLM_LOCAL_MEM_FRACTION to choose "
            "the share yourself.",
            setting,
            model,
            floor,
        )
    return floor


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


def _room_text(room: float, own: float, *, stays_up: bool = False) -> str:
    """The refusal's words for the room a start leaves bibr's GPU sessions."""
    from bibr.ocr.registry import MANAGED_VLLM_ONNX_RESERVE_GB

    text = f"{room:.2f} GiB for bibr's GPU sessions"
    if room > LLM_ONNX_MIN_HEADROOM_GB:
        text += f" ({MANAGED_VLLM_ONNX_RESERVE_GB:.1f} GiB less the {own:.2f} GiB they hold"
        text += ", since the server stays up for the run)" if stays_up else ")"
    return text


def _model_that_would_start(budget_now: float) -> tuple[str, float] | None:
    """The registry's vLLM model with the lowest floor, if that floor fits *budget_now*.

    *budget_now* is what a refused start could have had beside the same
    overshoot reserve and room for bibr, which is below its own floor, so a
    model that fits it has a lower one. The first one listed when several
    share the lowest floor. A 10-12 GB card beside cloud OCR gets none: the
    W4A16 build does not fit there, and the 2B's floor is not below it.
    """
    from bibr.local.llm_models import REGISTRY

    lowest: tuple[str, float] | None = None
    for entry in REGISTRY:
        for variant in entry.variants:
            candidate = _registry_floor_gb(variant.hf_id)  # None unless a vLLM build
            if candidate is not None and (lowest is None or candidate < lowest[1]):
                lowest = (variant.hf_id, candidate)
    if lowest is None or lowest[1] > budget_now:
        return None
    return lowest


def _refusal_advice(budget_now: float) -> str:
    """What still runs a refused start: a smaller model only where one would start."""
    smaller = _model_that_would_start(budget_now)
    pick = (
        f"pick a model with a lower floor (LLM_LOCAL_MODEL={smaller[0]}, {smaller[1]:.2f} GiB), "
        if smaller is not None
        else ""
    )
    return (
        f"Free GPU memory (other processes), {pick}set LLM_LOCAL_MEM_FRACTION to choose the "
        "share yourself, or run the LLM on llama.cpp (--llm llama-cpp)"
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
    sessions to grow: ``MANAGED_VLLM_ONNX_RESERVE_GB`` less what bibr already
    holds, and at least ``LLM_ONNX_MIN_HEADROOM_GB``. What bibr holds is
    already not free, so as its sessions grow the budget stays the same.
    Below the model's floor, a start that takes turns with a managed
    PaddleOCR-VL server (*swaps_with_ocr*) is squeezed to the floor while
    ``LLM_ONNX_MIN_HEADROOM_GB`` still fits: it restarts next chunk beside
    bibr's grown sessions and is sized again. Any other server stays up for
    the run and often starts in the first chunk, before bibr's sessions have
    grown, so it is never squeezed below the full room. Otherwise
    ``ManagedLlmVramError``.
    """
    from bibr.ocr.registry import MANAGED_VLLM_ONNX_RESERVE_GB

    if reading is None or reading.total_gb <= 0:
        return LlmShare(share=cap, reason="unknown")
    total, free, own = reading.total_gb, reading.free_gb, reading.own_gb
    headroom = max(LLM_ONNX_MIN_HEADROOM_GB, MANAGED_VLLM_ONNX_RESERVE_GB - own)
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
    # Only a start that lasts one chunk may give up bibr's room down to the
    # minimum; a server that stays up keeps it all beside the floor itself.
    room = LLM_ONNX_MIN_HEADROOM_GB if swaps_with_ocr else headroom
    if floor is not None and free - reserve_gb - room >= floor:
        # Round up so the model gets its floor; the room absorbs the rounding.
        share = min(cap, math.ceil(round(floor / total * 100, 6)) / 100)
        details["headroom_gb"] = min(room, free - reserve_gb - share * total)
        return LlmShare(share=share, reason="floor", budget_gb=share * total, **details)
    if floor is None:
        # Without a floor the smallest share, one hundredth of the card, would
        # do, but only beside the full room for bibr: the squeeze to
        # LLM_ONNX_MIN_HEADROOM_GB is for a model's floor alone. Free memory
        # just above the reserve and room rounds down to a share of 0, so the
        # need counts that hundredth too.
        smallest = 0.01 * total
        need = smallest + reserve_gb + headroom
        needs = (
            f"{need:.2f} GiB (the model has no floor, so the smallest share, one hundredth of "
            f"the card or {smallest:.2f} GiB; {reserve_gb:.2f} GiB vLLM takes beyond its share "
            f"while starting; and {_room_text(headroom, own)})"
        )
    else:
        need = floor + reserve_gb + room
        needs = (
            f"{need:.2f} GiB (the model's {floor:.2f} GiB floor, {reserve_gb:.2f} GiB vLLM "
            f"takes beyond its share while starting, and "
            f"{_room_text(room, own, stays_up=not swaps_with_ocr)})"
        )
    raise ManagedLlmVramError(
        f"not enough free GPU memory to start the managed vLLM LLM server: "
        f"{free:.2f} of {total:.2f} GiB free (bibr itself holds {own:.2f} GiB), and it needs "
        f"{needs}. {_refusal_advice(free - reserve_gb - room)}"
    )


def managed_llm_model(settings: GlobalSettings, model: str | None = None) -> str:
    """The model a managed vLLM LLM server for *settings* serves.

    *model*, else ``LLM_LOCAL_MODEL``, else the registry's default vLLM model.
    """
    from bibr.local.llm_models import default_local_model

    return model or settings.llm.local_model or default_local_model("vllm")


def managed_llm_share(
    reading: GpuMemory | None,
    settings: GlobalSettings,
    model: str,
    *,
    cap: float,
    swaps_with_ocr: bool,
) -> LlmShare:
    """``size_managed_llm_share`` with *model*'s floor and overshoot reserve.

    The one place a start (``VllmLlmServer``) and the check before a chunk's
    OCR (``ResourceManager.llm_start_refusal``) get the floor and the reserve,
    so the two always agree. Raises ``ManagedLlmVramError`` as that does.
    """
    return size_managed_llm_share(
        reading,
        cap=cap,
        floor_gb=managed_llm_floor_gb(model, settings),
        reserve_gb=overshoot_reserve_gb(model, settings.llm.vllm_extra_args),
        swaps_with_ocr=swaps_with_ocr,
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
