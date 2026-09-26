"""Local single-machine pipeline orchestrator.

Runs the full bibr paper processing pipeline on a single machine
with sequential GPU model loading to avoid VRAM conflicts.

Key optimisation: OCR engine startup (30-90 s) runs in a background
thread *during* layout detection, hiding most of the latency.  SGLang
spawns child processes with their own CUDA contexts, and vllm-mlx runs
as a managed subprocess — both coexist safely with the PyTorch layout
model in the main process.

Batch mode: ``process_chunk()`` processes multiple files through shared
pipeline stages.  Each model loads once per chunk and loops over files
within its stage.  HTTP OCR fires concurrent requests across files;
local OCR stays sequential.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, cast

from bibr.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.pipeline.context import (
        ConsolidationMode,
        MemoryMode,
        RefParseStrategy,
        RefSegStrategy,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_system_memory_gb() -> float:
    """Detect total system RAM in GB on Windows, macOS, and Linux."""
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
                ctypes.byref(status)
            ):
                return float(status.total_physical) / (1024**3)
        except (AttributeError, OSError, ValueError):
            pass
    # macOS: sysctl
    try:
        import subprocess

        result = subprocess.run(  # noqa: S603
            ["sysctl", "-n", "hw.memsize"],  # noqa: S607
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024**3)
    except Exception:  # noqa: S110
        pass
    # Linux: os.sysconf
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024**3)
    except (ValueError, OSError):
        pass
    return 16.0  # safe default


# Managed local LLM backends — bibr launches and owns the server process for
# each of these (vllm on Linux/CUDA, MLX runtimes on Apple Silicon). Used to gate
# OCR VRAM teardown, doctor checks, preflight, and --no-llm suppression.
LOCAL_LLM_BACKENDS = frozenset({"vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster"})

# Every accepted ``--llm`` / ``LLM_BACKEND`` value once the ``local`` alias is
# resolved: cloud plus the managed local backends.
_VALID_LLM_BACKENDS = frozenset({"cloud"}) | LOCAL_LLM_BACKENDS


def resolve_llm_backend(raw: str) -> str:
    """Resolve the ``local`` LLM-backend alias to a concrete backend.

    ``local`` auto-picks rapid-mlx (falling back to vllm-mlx when the
    rapid-mlx executable is missing) on Apple Silicon, llama.cpp on Windows
    and VRAM-constrained CUDA cards, and vLLM on larger Linux CUDA systems.
    Concrete names pass through unchanged. Any value outside the documented
    backend set raises ``InputValidationError`` rather than silently falling
    through to cloud. Shared by the CLI and ``LocalPipeline`` so the library
    API accepts everything ``bibr chew --llm`` does.
    """
    if raw == "local":
        import platform
        import sys

        is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
        if is_mac_arm:
            # Prefer the supported Apple Silicon runtime, falling back when it is unavailable.
            from bibr.local.rapid_mlx import rapid_mlx_unavailable_reason

            if rapid_mlx_unavailable_reason() is None:
                return "rapid-mlx"
            return "vllm-mlx"
        if sys.platform == "win32":
            return "llama-cpp"
        from bibr.local.llm_models import cuda_llm_backend_for, detect_hardware

        platform_key, memory_gb = detect_hardware()
        if platform_key == "cuda":
            return cuda_llm_backend_for(memory_gb)
        return "vllm"

    if raw not in _VALID_LLM_BACKENDS:
        from bibr.exceptions import InputValidationError

        raise InputValidationError(
            f"Unknown LLM backend {raw!r}. Valid values: cloud, local, vllm, vllm-mlx, "
            "rapid-mlx, llama-cpp, llmster."
        )
    return raw


def _auto_memory_mode(
    platform_key: str | None, accel_gb: float | None, system_ram_gb: float
) -> MemoryMode:
    """Pure hardware → memory-mode rule (shared by runtime default and setup).

    ``accel_gb`` is discrete-GPU VRAM for ``cuda`` and unified memory for
    ``mlx``. On Apple Silicon the OCR and local-LLM servers never coexist
    within a chunk (``_should_unload_ocr_after_chunk`` hands unified memory
    from one to the other), and the current MLX models are small (GLM-OCR
    8-bit ≈1.5 GB, LLM ≤5 GB), so 16 GB Macs run ``balanced`` fine. Compared
    to ``aggressive`` that saves a 30-60 s OCR server reload per chunk and
    keeps layout+segmenter resident; only ≤8 GB machines still need
    ``aggressive`` (caught by the ``system_ram_gb`` rule below, since
    ``accel_gb`` is unified memory on mlx).
    """
    if platform_key == "cuda" and accel_gb is not None and accel_gb <= 8:
        return "aggressive"
    if system_ram_gb <= 8:
        return "aggressive"
    return "balanced"


def _default_memory_mode(settings: GlobalSettings | None = None) -> MemoryMode:
    """Default memory mode: ``PIPELINE_MEMORY_MODE`` if set, else RAM auto-detect."""
    from bibr.config import snapshot_settings

    effective = settings if settings is not None else snapshot_settings()
    configured = effective.pipeline.memory_mode
    if configured:
        return cast("MemoryMode", configured)
    from bibr.local.llm_models import detect_hardware

    platform_key, memory_gb = detect_hardware()
    return _auto_memory_mode(platform_key, memory_gb, _get_system_memory_gb())


def _auto_batch_size(memory_mode: str) -> int:
    """Auto-select chunk size based on memory mode.

    Returns number of files per chunk for batch processing.
    """
    if memory_mode == "aggressive":
        return 3
    elif memory_mode == "keep_all":
        return 20
    return 8  # balanced default


class LocalPipeline(Pipeline):
    """Single-machine pipeline orchestrator.

    Processes PDF/DOCX/JATS-XML/HTML/ePub files through the full bibr pipeline
    using sequential GPU model loading. Memory management is controlled by ``memory_mode``:

    - ``"aggressive"``: Load/unload each model per phase (8 GB machines)
    - ``"balanced"``: Keep layout + segmenter loaded; OCR engine stays
      resident across chunks unless a local LLM server needs the VRAM
    - ``"keep_all"``: Keep everything loaded (24+ GB GPU or cloud LLM only)
    - ``None`` (default): ``PIPELINE_MEMORY_MODE`` if set, else auto-detect
      from system RAM (≤8 GB → aggressive, else balanced)

    For batch processing, use ``process_chunk()`` which loads each model once
    per chunk and loops over files within each stage.
    """

    def __init__(
        self,
        memory_mode: MemoryMode | None = None,
        ocr_backend: str | None = None,
        ocr_url: str | None = None,
        ocr_model: str | None = None,
        ocr_profile: str | None = None,
        device: str | None = None,
        crossref: bool | None = None,
        equations: bool = True,
        start_page: int | None = None,
        end_page: int | None = None,
        llm_backend: str | None = None,
        no_llm: bool = False,
        figure_images: bool | None = None,
        include_regions: bool = False,
        include_region_meta: bool = False,
        consolidate: ConsolidationMode | None = None,
        ref_seg_strategy: RefSegStrategy | None = None,
        ref_parse_strategy: RefParseStrategy | None = None,
        settings: GlobalSettings | None = None,
    ):
        from bibr.config import snapshot_settings

        # Alias/platform resolution ("glm", None → OCR_BACKEND or
        # platform default) — same table as the CLI, so the library API
        # accepts everything `bibr chew --ocr` does.
        from bibr.ocr.registry import resolve_backend_name
        from bibr.pipeline.context import RunConfig
        from bibr.pipeline.enricher import CrossrefEnricher, RorEnricher
        from bibr.pipeline.plans import build_stage_plan
        from bibr.pipeline.resources import ResourceManager

        settings_snapshot = snapshot_settings(settings)

        # ``paddle`` is the automatic startup selector, not a concrete
        # backend: ResourceManager must retain it to try its ordered runtime
        # candidates and record the one that actually starts. Other aliases
        # still resolve eagerly for compatibility with concrete callers.
        configured_ocr = settings_snapshot.ocr
        configured_backend = (
            configured_ocr.backend if "backend" in configured_ocr.model_fields_set else None
        )
        requested_backend = ocr_backend or configured_backend
        if ocr_url and requested_backend not in {"paddle-http", "serve-http"}:
            # A supplied URL is the established explicit remote-GLM contract;
            # resolve it before the local automatic selector can attach Paddle
            # identity or cache provenance to that request.
            ocr_backend = "glm-http"
        else:
            automatic_paddle = ocr_backend == "paddle" or (
                ocr_backend is None
                and (configured_ocr.backend == "paddle" or configured_backend is None)
            )
            if automatic_paddle:
                ocr_backend = "paddle"
            else:
                ocr_backend = resolve_backend_name(ocr_backend, settings=settings_snapshot)
        # None → LLM_BACKEND setting (default "cloud"); explicit values pass through.
        llm_backend = resolve_llm_backend(llm_backend or settings_snapshot.llm.backend)
        # None → PIPELINE_MEMORY_MODE, else RAM auto-detect — same resolution
        # ladder as the CLI, so the library API behaves identically.
        if memory_mode is None:
            memory_mode = _default_memory_mode(settings_snapshot)

        # --no-llm implies no Crossref and no equation extraction, and no
        # managed local LLM server (which would never be called — forcing
        # "cloud" makes LlmServerStage a no-op regardless of LLM_BACKEND).
        # ``crossref`` is tri-state: None follows CROSSREF_ENRICH (off by
        # default), True/False force it for this pipeline.
        if no_llm:
            crossref = False
            equations = False
            llm_backend = "cloud"

        config = RunConfig(
            memory_mode=memory_mode,
            ocr_backend=ocr_backend,
            ocr_url=ocr_url,
            ocr_model=ocr_model,
            ocr_profile=ocr_profile,
            device=device,
            crossref=crossref,
            equations=equations,
            start_page=start_page,
            end_page=end_page,
            llm_backend=llm_backend,
            no_llm=no_llm,
            include_figures=figure_images,
            include_regions=include_regions,
            include_region_meta=include_region_meta,
            consolidate=consolidate,
            ref_seg_strategy=ref_seg_strategy,
            ref_parse_strategy=ref_parse_strategy,
        )
        resources = ResourceManager(
            memory_mode=memory_mode,
            ocr_backend=ocr_backend,
            ocr_url=ocr_url,
            ocr_model=ocr_model,
            ocr_profile=ocr_profile,
            device=device,
            managed_vllm_fraction=(
                settings_snapshot.llm.local_mem_fraction
                if llm_backend in ("local", "vllm")
                else 0.0
            ),
            settings=settings_snapshot,
        )

        # refs=off produces an empty bib table — Crossref enrichment (which
        # works on references) would be a per-paper no-op; skip the enricher
        # so the enrich stage disappears rather than silently idling.
        refs_off = (
            ref_parse_strategy or settings_snapshot.REF_PARSE_STRATEGY or ""
        ).lower() == "off"

        enrichers = []
        if config.enrichment_enabled(settings_snapshot) and not refs_off:
            enrichers.append(CrossrefEnricher(settings=settings_snapshot))
            if settings_snapshot.ror.enrich:
                enrichers.append(RorEnricher(settings=settings_snapshot))

        # Cloud LLM has no OCR→LLM VRAM handoff, so each OCR window's back
        # half (parse → extract → enrich → export) streams under subsequent
        # windows' OCR instead of waiting at each stage barrier. Managed
        # local backends keep the barrier (LlmServerStage needs the OCR
        # teardown handoff); aggressive mode would thrash the segmenter.
        stream_backhalf = (
            llm_backend == "cloud"
            and memory_mode != "aggressive"
            and settings_snapshot.pipeline.stream_backhalf
        )
        stages = build_stage_plan(
            mode="local", stream_backhalf=stream_backhalf, enrichers=enrichers
        )
        super().__init__(
            stages=stages,
            resources=resources,
            config=config,
            settings=settings_snapshot,
        )

        # Back-compat attribute access for callers inspecting the pipeline.
        self.memory_mode = memory_mode
        self.ocr_backend = ocr_backend
        self.ocr_url = ocr_url
        self.ocr_model = ocr_model
        self.ocr_profile = ocr_profile
        self.device = device
        self.crossref = crossref
        self.equations = equations
        self.start_page = start_page
        self.end_page = end_page
        self.llm_backend = llm_backend
        self.no_llm = no_llm

    def llm_usage_snapshot(self) -> dict[str, dict[str, int]]:
        """Cumulative LLM token usage by model, captured before pipeline teardown.

        Returns an empty dict if no LLM client was instantiated (e.g. ``--no-llm``)
        or if usage tracking is disabled (``LLM_TRACK_USAGE=false``).
        """
        client = self._resources.llm_client
        return {model: dict(counts) for model, counts in client.usage.items()}
