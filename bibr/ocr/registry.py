"""Registry of OCR backend implementations.

Each concrete backend class applies ``@register`` at import time. Callers
(e.g. ``ResourceManager``) use ``create(name, **kwargs)`` to construct an
instance without knowing the concrete class.

Extra keyword arguments not consumed by the backend are silently ignored
(backends declare ``**_kw: object`` on their constructors).
"""

from __future__ import annotations

import functools
import logging
import math
import os
import platform
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from bibr.ocr.backend import OcrBackend

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.ocr.profiles import OcrProfileName

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, type[OcrBackend]] = {}


@dataclass(frozen=True)
class OcrBackendCandidate:
    """One concrete OCR runtime that may satisfy an automatic request."""

    backend: str
    model: str
    profile: OcrProfileName


# VRAM the managed PaddleOCR-VL vLLM server is given: its ~2.1 GiB of weights
# and non-torch memory, the vision encoder's ~0.6 GiB profiling peak at 16,384
# batched tokens, and about 2.2 GiB of KV cache (~130,000 tokens; its peak use
# in 20-paper runs on a 24 GB card was under 16,000). vLLM's
# ``--gpu-memory-utilization`` is a share of the whole card that must be free
# when the server starts, so the launcher turns this into a share of the card
# (``paddle_vllm_gpu_memory_utilization``) instead of claiming most of it.
PADDLE_VLLM_GPU_MEMORY_GB = 5.0

# The largest share the managed PaddleOCR-VL server is launched with, and its
# share when the card it starts on cannot be identified.
PADDLE_VLLM_GPU_MEMORY_UTILIZATION = 0.92

# VRAM kept outside both managed vLLM servers' shares (the LLM and
# PaddleOCR-VL) whenever they may be up at the same time. A server that starts
# beside the other needs room for bibr's own GPU sessions (layout, SaT, the
# NER parser, the classifiers: 2.4-2.8 GiB when an RTX 5090's later chunks
# restarted OCR, and layout grows to about 5 GiB when it has room) plus its own
# ~0.5 GiB CUDA context, or it never starts. Below about 4 GiB of room layout
# also halves its batch after CUDA OOM (78 times per 20 papers on a 24 GB card,
# none with 6 GiB).
MANAGED_VLLM_ONNX_RESERVE_GB = 6.0

# PaddleOCR-VL through the managed vLLM server needs an NVIDIA GPU with room
# for ``PADDLE_VLLM_GPU_MEMORY_GB``. Below this floor bibr's documented Linux
# contract ("NVIDIA GPU <= 8 GB uses llama.cpp") routes OCR to ``glm-llama``.
PADDLE_VLLM_MIN_VRAM_GB = 8.0
#: GPU memory a managed GLM-OCR llama.cpp server (``glm-llama``, the Q8_0 GGUF
#: with its vision projector, an 8,192-token context and one slot) holds while
#: it OCRs: at most 2,206 MiB on an RTX 3090 over a chunk of three papers
#: (llama.cpp b11282), rounded up to a quarter GiB. A start beside a managed
#: llama.cpp LLM server stops that server first when less than this plus
#: bibr's minimum room is free (``ResourceManager._llm_server_blocks_ocr``).
GLM_LLAMA_OCR_VRAM_GB = 2.25


@functools.cache
def _cuda_vram_gb() -> float | None:
    """Total VRAM (GiB) of the first NVIDIA GPU, or ``None`` when none is usable.

    Probed once per process: hardware does not change under a running
    pipeline, and the ``nvidia-smi`` subprocess behind it is not free. Tests
    patch this name to pin a platform.
    """
    from bibr.local.llm_models import detect_hardware

    platform_key, memory_gb = detect_hardware()
    if platform_key == "cuda":
        return memory_gb
    return None


_PROBE = object()


@functools.cache
def _nvidia_gpu_inventory() -> tuple[tuple[str, float], ...] | None:
    """``(uuid, total GiB)`` of each NVIDIA GPU in nvidia-smi's (PCI bus) order, or None."""
    import subprocess

    try:
        result = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    gpus = []
    for line in result.stdout.strip().splitlines():
        uuid, _, total = line.partition(",")
        try:
            gpus.append((uuid.strip(), float(total) / 1024.0))  # MiB -> GiB
        except ValueError:
            return None
    return tuple(gpus) or None


def _visible_gpu_indices(
    uuids: tuple[str, ...], visible_devices: str | None, device_order: str | None
) -> tuple[int, ...] | None:
    """Indices into ``uuids`` of the cards that may be CUDA's device 0, or None.

    ``uuids`` is nvidia-smi's list, which is in PCI bus order. CUDA numbers the
    cards fastest first unless ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, and
    ``CUDA_VISIBLE_DEVICES`` picks by that number or by UUID (a unique prefix
    is enough), so without PCI order a number leaves every card a candidate.
    None when the choice names no card, a MIG slice or something unparsable.
    """
    if not uuids:
        return None
    pci_order = (device_order or "").strip().upper() == "PCI_BUS_ID"
    every = tuple(range(len(uuids)))
    if visible_devices is None:
        return every[:1] if pci_order else every
    first = visible_devices.split(",")[0].strip()
    if first.startswith("GPU-"):
        matches = tuple(i for i, uuid in enumerate(uuids) if uuid.startswith(first))
        return matches if len(matches) == 1 else None
    if first.isdigit():
        index = int(first)
        if index >= len(uuids):
            return None
        return (index,) if pci_order else every
    return None


def _visible_gpu_vram_gb(
    gpus: tuple[tuple[str, float], ...] | None,
    visible_devices: str | None,
    device_order: str | None,
) -> float | None:
    """Total VRAM (GiB) of the card CUDA numbers 0, where a vLLM server starts.

    ``gpus`` is nvidia-smi's list (see ``_visible_gpu_indices`` for how the
    card is picked). None when the choice still spans cards of different
    sizes, or when it names no card, a MIG slice or something unparsable.
    """
    if not gpus:
        return None
    indices = _visible_gpu_indices(tuple(uuid for uuid, _ in gpus), visible_devices, device_order)
    if not indices:
        return None
    sizes = {gpus[i][1] for i in indices}
    return sizes.pop() if len(sizes) == 1 else None


@functools.cache
def _managed_gpu_vram_gb() -> float | None:
    """Total VRAM (GiB) of the GPU the managed vLLM servers start on, or None.

    That is CUDA's device 0 under this process's ``CUDA_VISIBLE_DEVICES``
    (the servers inherit it), not nvidia-smi's GPU 0 that ``_cuda_vram_gb``
    reports. None when no NVIDIA GPU is visible or the card cannot be told
    apart from cards of other sizes: the PaddleOCR-VL share then stays at
    ``PADDLE_VLLM_GPU_MEMORY_UTILIZATION`` and the two servers never count as
    fitting together. Probed once per process; tests patch this name.
    """
    return _visible_gpu_vram_gb(
        _nvidia_gpu_inventory(),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("CUDA_DEVICE_ORDER"),
    )


@dataclass(frozen=True)
class GpuMemory:
    """Memory of the card the managed vLLM servers start on, in GiB, read just now."""

    #: What CUDA reports as the card's total (nvidia-smi's ``memory.total``
    #: less ``memory.reserved``): the amount vLLM's ``--gpu-memory-utilization``
    #: is a share of. 23.56 GiB on an RTX 3090.
    total_gb: float
    free_gb: float
    #: This process's own use of the card (bibr's ONNX sessions and CUDA
    #: context); 0 when nvidia-smi does not list it.
    own_gb: float


def _nvidia_smi_rows(query: str, fields: str) -> list[list[str]] | None:
    """nvidia-smi ``--query-<query>=<fields>`` rows, split on commas, or None."""
    import subprocess

    try:
        result = subprocess.run(  # noqa: S603
            ["nvidia-smi", f"--query-{query}={fields}", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return [
        [cell.strip() for cell in line.split(",")] for line in result.stdout.strip().splitlines()
    ]


def _mib_to_gib(value: str) -> float | None:
    try:
        return float(value) / 1024.0
    except ValueError:
        return None


def _own_gpu_memory_gb(uuid: str) -> float:
    """GiB this process holds on the card ``uuid`` (0 when nvidia-smi cannot say)."""
    rows = _nvidia_smi_rows("compute-apps", "pid,gpu_uuid,used_memory") or []
    pid = str(os.getpid())
    used = 0.0
    for row in rows:
        if len(row) == 3 and row[0] == pid and row[1] == uuid:
            used += _mib_to_gib(row[2]) or 0.0
    return used


def managed_gpu_memory() -> GpuMemory | None:
    """Total, free and own memory of the managed vLLM servers' card, or None.

    Read now, not cached: the managed LLM server sizes its share from it at
    every start, and bibr's own GPU sessions grow during a batch. The card is
    CUDA's device 0 under ``CUDA_VISIBLE_DEVICES``, picked as for
    ``_managed_gpu_vram_gb``; when several cards of one size may be device 0,
    the one with the least free memory. None when no NVIDIA GPU is visible,
    the candidates differ in size, or nvidia-smi fails. Drivers without
    ``memory.reserved`` count the whole card as CUDA's total, which only makes
    the shares computed from it smaller. Tests patch this name.
    """
    rows = _nvidia_smi_rows("gpu", "uuid,memory.total,memory.reserved,memory.free")
    if rows is not None and all(len(row) == 4 for row in rows):
        cards = [(row[0], row[1], row[2], row[3]) for row in rows]
    else:
        rows = _nvidia_smi_rows("gpu", "uuid,memory.total,memory.free")
        if rows is None or not all(len(row) == 3 for row in rows):
            return None
        cards = [(row[0], row[1], "0", row[2]) for row in rows]
    parsed = []
    for uuid, total, reserved, free in cards:
        total_gb, free_gb = _mib_to_gib(total), _mib_to_gib(free)
        if total_gb is None or free_gb is None:
            return None
        parsed.append((uuid, total_gb, _mib_to_gib(reserved) or 0.0, free_gb))
    indices = _visible_gpu_indices(
        tuple(card[0] for card in parsed),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("CUDA_DEVICE_ORDER"),
    )
    if not indices or len({parsed[i][1] for i in indices}) != 1:
        return None
    uuid, total_gb, reserved_gb, free_gb = min((parsed[i] for i in indices), key=lambda c: c[3])
    return GpuMemory(
        total_gb=total_gb - reserved_gb, free_gb=free_gb, own_gb=_own_gpu_memory_gb(uuid)
    )


def paddle_vllm_unavailable_reason(*, vram_gb: float | None | object = _PROBE) -> str | None:
    """Why the managed ``paddle-vllm`` runtime cannot start here, or ``None``.

    Consulted by the automatic ``paddle`` chain (so a CPU-only Linux box never
    spends its startup budget bootstrapping vLLM), by the managed launcher
    itself, and by the ``bibr chew`` preflight.
    """
    from bibr.local.llm_models import fits_card

    vram = _cuda_vram_gb() if vram_gb is _PROBE else vram_gb
    if not isinstance(vram, int | float):
        return "no NVIDIA GPU detected (paddle-vllm needs CUDA)"
    # The registry's card-size rule, so a card that counts as 8 GB for the
    # LLM (an RTX 4060 reports 8,188 MiB) also runs PaddleOCR-VL on vLLM.
    if not fits_card(PADDLE_VLLM_MIN_VRAM_GB, vram):
        return (
            f"the GPU has {vram:.0f} GB of VRAM and paddle-vllm needs at least "
            f"{PADDLE_VLLM_MIN_VRAM_GB:.0f} GB (llama.cpp is the low-VRAM OCR path)"
        )
    return None


def paddle_vllm_gpu_memory_utilization(*, vram_gb: float | None | object = _PROBE) -> float:
    """``--gpu-memory-utilization`` for the managed PaddleOCR-VL vLLM server.

    ``PADDLE_VLLM_GPU_MEMORY_GB`` as a share of the card the server starts on,
    rounded up to the next hundredth and capped at
    ``PADDLE_VLLM_GPU_MEMORY_UTILIZATION`` (also the answer when that card
    cannot be identified): 0.16 on a 32 GB card, 0.21 on 24 GB, 0.32 on 16 GB.

    vLLM refuses to start unless that share of the whole card is free. bibr
    keeps its layout, SaT and NER ONNX sessions on the GPU between chunks, so
    a fixed 0.92 left 8% of the card for them and for vLLM's own CUDA context:
    on an RTX 5090 the second chunk's restart found 28.52 of the 28.93 GiB it
    asked for, and every later paper failed.
    """
    vram = _managed_gpu_vram_gb() if vram_gb is _PROBE else vram_gb
    if not isinstance(vram, int | float) or vram <= 0:
        return PADDLE_VLLM_GPU_MEMORY_UTILIZATION
    # round() first, so float noise (7 / 35 * 100 = 20.000000000000004) cannot
    # push the share up a hundredth.
    share = math.ceil(round(PADDLE_VLLM_GPU_MEMORY_GB / vram * 100, 6)) / 100
    return min(PADDLE_VLLM_GPU_MEMORY_UTILIZATION, share)


def managed_vllm_onnx_reserve_share(*, vram_gb: float | None | object = _PROBE) -> float:
    """``MANAGED_VLLM_ONNX_RESERVE_GB`` as a share of the managed servers' card (0 if unknown)."""
    vram = _managed_gpu_vram_gb() if vram_gb is _PROBE else vram_gb
    if not isinstance(vram, int | float) or vram <= 0:
        return 0.0
    return MANAGED_VLLM_ONNX_RESERVE_GB / vram


def managed_vllm_shares_fit(
    llm_share: float, ocr_share: float, *, vram_gb: float | None | object = _PROBE
) -> bool:
    """Whether the managed vLLM LLM and PaddleOCR-VL servers can be up at once.

    Both shares plus ``MANAGED_VLLM_ONNX_RESERVE_GB`` must fit on the card the
    servers start on; on a card that cannot be identified they never do.
    Shares summing to at most 1 are not enough: at 0.85 + 0.15 of a 48 GB
    card, the server starting second would find bibr's own GPU sessions inside
    its share and never start.
    """
    vram = _managed_gpu_vram_gb() if vram_gb is _PROBE else vram_gb
    if not isinstance(vram, int | float) or vram <= 0:
        return False
    reserve = managed_vllm_onnx_reserve_share(vram_gb=vram)
    # The tolerance absorbs float noise in shares that fit exactly
    # (0.54 + 0.21 + 0.25 on 24 GB).
    return llm_share + ocr_share + reserve <= 1.0 + 1e-9


def default_backend() -> str:
    """Pick the preferred local OCR runtime supported by the host platform."""
    from bibr.config import snapshot_settings

    # The automatic ``paddle`` chain skips runtimes the host cannot run (no
    # NVIDIA GPU with room → no managed vLLM), so its head — not a fixed
    # platform guess — is the default that can actually start here.
    return resolve_backend_candidates("paddle", snapshot_settings())[0].backend


def default_glm_backend() -> str:
    """Pick a concrete GLM runtime without entering the Paddle selector."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "glm-rapid-mlx"
    return "glm-llama"


def _candidate_model(backend: str, settings: GlobalSettings) -> str:
    from bibr.ocr.profiles import resolve_served_model

    # One table (bibr/ocr/profiles.py) for startup candidates and static
    # identity alike. serve-http resolves its family from the explicit
    # profile: with OCR_PROFILE=paddle the server is a Paddle endpoint and
    # must be asked for the Paddle served alias, not 'glm-ocr'.
    if backend == "serve-http":
        return resolve_served_model(
            requested_backend="serve-http",
            concrete_backend="serve-http",
            settings=settings,
            explicit_profile=settings.ocr.profile,
        )
    return resolve_served_model(
        requested_backend=backend,
        concrete_backend=backend,
        settings=settings,
        explicit_profile=None,
    )


def _candidate(backend: str, settings: GlobalSettings) -> OcrBackendCandidate:
    return OcrBackendCandidate(
        backend=backend,
        model=_candidate_model(backend, settings),
        profile="paddle" if backend.startswith("paddle") else "glm",
    )


def resolve_backend_candidates(
    name: str | None, settings: GlobalSettings
) -> tuple[OcrBackendCandidate, ...]:
    """Resolve an OCR request to concrete, deterministic startup candidates.

    Only the ``paddle`` automatic selector has more than one candidate.
    Explicit concrete backends are deliberately single-candidate requests so a
    user-selected runtime is never silently replaced by another implementation.
    """
    if name is None:
        ocr_opts = settings.ocr
        name = ocr_opts.backend if "backend" in ocr_opts.model_fields_set else "paddle"
    if name in ("http", "glm-http"):
        name = "glm-http"
    elif name in ("glm", "glm-local"):
        name = default_glm_backend()

    if name != "paddle":
        return (_candidate(name, settings),)
    return tuple(_candidate(backend, settings) for backend in automatic_backend_names())


def automatic_backend_names() -> tuple[str, ...]:
    """The concrete runtimes the automatic ``paddle`` selector tries here, in order."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return ("paddle-rapid-mlx", "paddle-mlx-vlm", "glm-rapid-mlx", "glm-llama")
    if sys.platform == "win32":
        return ("glm-llama",)
    if sys.platform.startswith("linux") and platform.machine().lower() in {"x86_64", "amd64"}:
        # The managed vLLM runtime is only a candidate on hardware that can
        # run it. Otherwise a CPU-only or small-GPU box spent the whole 900 s
        # startup budget bootstrapping vLLM before falling through to llama.cpp.
        vllm_blocker = paddle_vllm_unavailable_reason()
        if vllm_blocker is None:
            return ("paddle-vllm", "glm-llama")
        logger.debug("paddle-vllm skipped in the automatic OCR chain: %s", vllm_blocker)
    return ("glm-llama",)


#: The cloud vision-LLM OCR backends. They call their provider's API (set
#: ``OCR_VISION_BASE_URL`` to move it), never an OCR server, and hold no local
#: weights. The one definition every OCR module imports.
CLOUD_VISION_OCR_BACKENDS = frozenset({"gemini", "openai", "anthropic"})

#: Requests an ``ocr_url`` leaves alone: the two HTTP clients already address
#: the URL, and the cloud vision backends ignore it.
_URL_KEEPS_BACKEND = CLOUD_VISION_OCR_BACKENDS | {"paddle-http", "serve-http"}


def url_request(explicit: str | None, configured: str | None) -> str | None:
    """Return the OCR backend request an ``ocr_url`` is resolved from.

    ``explicit`` is the caller's backend (``--ocr``, ``ocr_backend=``,
    ``RunConfig.ocr_backend``) and is the request whenever it is set.
    Without it the configured ``OCR_BACKEND`` is the request, unless it names
    a cloud vision provider: a server URL the caller named outranks a
    configured cloud default, so page images never go to a cloud provider
    because of a ``.env`` setting. ``None`` means no request (the default
    ``paddle`` selector). Every entry point calls this before
    :func:`resolve_url_backend`.
    """
    if explicit:
        return explicit
    if configured in CLOUD_VISION_OCR_BACKENDS:
        return None
    return configured


@overload
def resolve_url_backend(requested: str | None, ocr_url: str) -> str: ...


@overload
def resolve_url_backend(requested: str, ocr_url: str | None) -> str: ...


@overload
def resolve_url_backend(requested: str | None, ocr_url: str | None) -> str | None: ...


def resolve_url_backend(requested: str | None, ocr_url: str | None) -> str | None:
    """Apply an explicit OCR server URL to an OCR backend request.

    The one rule behind ``bibr chew --ocr-url``, ``LocalPipeline(ocr_url=...)``
    (and so ``bibr.chew``), ``ResourceManager`` and the OCR runtime identity,
    so they all start, cache and report the same backend. ``requested`` comes
    from :func:`url_request`. An OCR URL means Paddle, bibr's default OCR:

    - without a URL the request is returned unchanged;
    - ``paddle-http``, ``serve-http`` and the cloud vision backends
      (:data:`CLOUD_VISION_OCR_BACKENDS`) are kept;
    - a GLM request (``glm``, any ``glm-*`` name, or the legacy ``http``
      alias of ``glm-http``) becomes ``glm-http``;
    - anything else, including the ``paddle`` selector and no request at
      all, becomes ``paddle-http``.

    Idempotent: a resolved backend resolves to itself.
    """
    if not ocr_url:
        return requested
    if requested in _URL_KEEPS_BACKEND:
        return requested
    if requested is not None and (requested == "http" or requested.startswith("glm")):
        return "glm-http"
    return "paddle-http"


def resolve_backend_name(name: str | None, settings: GlobalSettings | None = None) -> str:
    """Expand OCR backend aliases to concrete registry names.

    The single source of truth for alias/platform resolution — used by the
    CLI, ``LocalPipeline``, and the demo, so ``bibr.chew(ocr="glm")``
    behaves exactly like ``bibr chew --ocr glm``.

    ``None`` falls back to an explicitly-set ``OCR_BACKEND`` setting, else the
    platform default.
    """
    from bibr.config import snapshot_settings

    effective = settings if settings is not None else snapshot_settings()
    return resolve_backend_candidates(name, effective)[0].backend


def register(cls: type[OcrBackend]) -> type[OcrBackend]:
    """Class decorator that registers an `OcrBackend` under ``cls.name``."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must declare a class-level `name` attribute")
    if name in _BACKENDS and _BACKENDS[name] is not cls:
        raise ValueError(f"OCR backend {name!r} already registered to {_BACKENDS[name].__name__}")
    _BACKENDS[name] = cls
    return cls


def create(name: str, **kwargs: object) -> OcrBackend:
    """Instantiate a registered backend by name."""
    cls = _BACKENDS.get(name)
    if cls is None:
        known = sorted(_BACKENDS)
        raise ValueError(f"Unknown OCR backend: {name!r}. Known: {known}")
    return cls(**kwargs)


def known_backends() -> list[str]:
    """Return a sorted list of registered backend names (for diagnostics)."""
    return sorted(_BACKENDS)
