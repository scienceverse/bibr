"""Resolve ``chew`` CLI args into a concrete pipeline run configuration."""

import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from bibr.local.cli.parser import normalize_ocr_backend
from bibr.utils.pages import parse_pages as _parse_pages


@dataclass
class ResolvedRunConfig:
    """Pipeline configuration derived from parsed ``chew`` arguments.

    Built by :func:`resolve_run_config` — plain data, no I/O.
    """

    ocr_backend: str
    memory_mode: str
    llm_backend: str
    ocr_url: str | None = None
    ocr_model: str | None = None
    ocr_profile: str | None = None
    device: str | None = None
    crossref: bool | None = None
    """Tri-state enrichment switch: ``True`` (``--crossref``), ``False``
    (``--no-crossref``), ``None`` follows ``CROSSREF_ENRICH`` (off by default)."""
    equations: bool = True
    no_llm: bool = False
    figure_images: bool | None = None
    include_regions: bool = False
    include_region_meta: bool = False
    start_page: int | None = None
    end_page: int | None = None
    consolidate: str | None = None
    chunk_size: int = 1
    ref_seg: str | None = None
    refs: str | None = None
    ocr_summary_model: str | None = None
    mode: str = "full"
    """The run mode: ``fast`` makes no LLM call, ``full`` uses the configured LLM
    (``--no-llm`` is a ``full`` run that skips the LLM stages, see ``skips_llm``)."""
    ocr_summary_backend: str | None = None
    """The first concrete OCR candidate (``paddle`` expands to an ordered chain)."""
    ocr_requested: bool = True
    """Whether the user chose the OCR route (``--ocr``, ``--ocr-url`` or ``OCR_BACKEND``).
    A fast run starts the multi-GB vLLM bootstrap only when this is set."""

    @property
    def skips_llm(self) -> bool:
        """Whether the run makes no LLM call: fast mode, or ``--no-llm``."""
        return self.no_llm or self.mode == "fast"

    @property
    def vllm_bootstrap(self) -> bool:
        """Whether a missing vLLM may be bootstrapped: not in a fast run nobody chose OCR for."""
        return self.mode != "fast" or self.ocr_requested

    @property
    def uses_llm(self) -> bool:
        """Whether the run calls the configured LLM (a full run without ``--no-llm``)."""
        return not self.skips_llm

    def enrichment_enabled(self) -> bool:
        """Whether this run enriches references (mirrors ``RunConfig.enrichment_enabled``).

        ``--no-llm`` forces enrichment off, an explicit ``--crossref`` /
        ``--no-crossref`` wins next, and otherwise the ``CROSSREF_ENRICH``
        setting decides. ``refs=off`` leaves nothing to enrich — Crossref
        works on references — so it also reads as off.
        """
        if self.no_llm or self.refs == "off":
            return False
        if self.crossref is not None:
            return self.crossref
        from bibr.config import Settings

        return bool(Settings.crossref.enrich)

    def active_stages(self, files: Iterable[Path] | None = None) -> list[str]:
        """Progress stages for known input formats, or every format before discovery."""
        from bibr.pipeline.progress import STAGES, stages_for_files

        stages = list(STAGES) if files is None else stages_for_files(STAGES, files)
        if not self.enrichment_enabled():
            stages.remove("enrich")
        return stages


def resolve_run_config(args) -> ResolvedRunConfig:
    """Derive the pipeline run configuration from parsed ``chew`` args.

    Reads (but never mutates) ``Settings`` for the OCR backend fallback;
    flag/preset overrides are applied separately by ``_apply_runtime_settings``.
    Raises ``ValueError`` describing the offending option (``--pages``,
    ``--ocr-model``/``--ocr-profile``).
    """
    # Determine OCR backend. ``--ocr-url`` goes through the rule LocalPipeline
    # and the OCR runtime identity share: Paddle unless ``--ocr`` (or a
    # configured OCR_BACKEND) asks for GLM. An explicit ``--ocr`` cloud vision
    # backend is kept; a configured cloud OCR_BACKEND never outranks the URL.
    ocr_backend = args.ocr
    if args.ocr_url:
        from bibr.config import Settings
        from bibr.ocr.registry import resolve_url_backend, url_request

        ocr_backend = resolve_url_backend(url_request(args.ocr, Settings.ocr.backend), args.ocr_url)
    elif ocr_backend == "paddle":
        # ``paddle`` is a startup selector, not a concrete backend. Keep it
        # intact for ResourceManager's transactional fallback chain.
        pass
    elif ocr_backend is None:
        from bibr.config import Settings

        if Settings.ocr.backend == "paddle":
            ocr_backend = "paddle"
        else:
            ocr_backend = normalize_ocr_backend(Settings.ocr.backend)
    else:
        ocr_backend = normalize_ocr_backend(ocr_backend)

    # --llm unset (None) falls back to the LLM_BACKEND setting (default "cloud").
    from bibr.config import Settings

    mode = _resolve_run_mode(args, Settings)
    llm_backend = _resolve_llm_backend(args.llm or Settings.llm.backend)
    if mode == "fast":
        # A fast run starts no LLM server and calls no provider, so it runs with
        # the cloud backend's memory, OCR-chain and streaming semantics.
        llm_backend = "cloud"
    # The LLM server the run starts: none under --no-llm or in a fast run.
    server_llm_backend = "cloud" if getattr(args, "no_llm", False) else llm_backend

    # Determine memory mode: on a CUDA card it depends on whether a managed
    # local LLM server shares the card.
    memory_mode = args.memory
    if memory_mode is None:
        from bibr.local.pipeline import _default_memory_mode

        memory_mode = _default_memory_mode(llm_backend=server_llm_backend)

    # Parse page range
    start_page, end_page = None, None
    if args.pages:
        start_page, end_page = _parse_pages(args.pages)

    # Determine chunk size for batch processing
    from bibr.local.pipeline import _auto_batch_size

    chunk_size = args.batch_size if args.batch_size > 0 else _auto_batch_size(memory_mode)

    from bibr.ocr.registry import bootstrap_kwargs, resolve_backend_candidates

    explicit_profile = getattr(args, "ocr_profile", None)
    configured_profile = getattr(Settings.ocr, "profile", None)
    if args.ocr_model and explicit_profile is None and configured_profile is None:
        model_name = args.ocr_model.lower()
        if "paddle" not in model_name and "glm" not in model_name:
            raise ValueError("Custom --ocr-model aliases require --ocr-profile {paddle,glm}.")

    ocr_requested = _ocr_requested(args, Settings)
    summary_candidate = resolve_backend_candidates(
        ocr_backend,
        Settings,
        llm_backend=server_llm_backend,
        **bootstrap_kwargs(mode != "fast" or ocr_requested),
    )[0]
    return ResolvedRunConfig(
        ocr_backend=ocr_backend,
        memory_mode=memory_mode,
        llm_backend=llm_backend,
        ocr_url=args.ocr_url,
        ocr_model=args.ocr_model,
        ocr_profile=explicit_profile or configured_profile,
        device=args.device,
        crossref=_resolve_crossref_flag(args),
        equations=not args.no_equations,
        no_llm=args.no_llm,
        figure_images=args.figure_images or None,
        include_regions=args.regions,
        include_region_meta=args.region_meta,
        start_page=start_page,
        end_page=end_page,
        consolidate=args.consolidate,
        chunk_size=chunk_size,
        ref_seg=getattr(args, "ref_seg", None) or None,
        refs=getattr(args, "refs", None) or None,
        ocr_summary_model=summary_candidate.model,
        mode=mode,
        ocr_summary_backend=summary_candidate.backend,
        ocr_requested=ocr_requested,
    )


def _llm_only_flags(args) -> list[str]:
    """The per-run flags in ``args`` that can only run with an LLM, as typed."""
    from bibr.run_mode import LLM_PARSE_STRATEGIES

    flags: list[str] = []
    if getattr(args, "llm", None) is not None:
        flags.append(f"--llm {args.llm}")
    refs = getattr(args, "refs", None)
    if refs in LLM_PARSE_STRATEGIES:
        flags.append(f"--refs {refs}")
    if getattr(args, "ref_seg", None) == "llm":
        flags.append("--ref-seg llm")
    return flags


def _resolve_run_mode(args, settings) -> str:
    """The mode this run takes: ``--fast`` / ``--full``, else ``PIPELINE_MODE``, else auto.

    See :func:`bibr.run_mode.resolve_run_mode` for how ``--no-llm`` and the
    LLM-only options (``--llm``, ``--refs llm``, ``--ref-seg llm``) take part.
    """
    from bibr.run_mode import resolve_run_mode

    explicit = (
        "fast" if getattr(args, "fast", False) else "full" if getattr(args, "full", False) else None
    )
    return resolve_run_mode(
        settings,
        explicit=explicit,
        llm_backend=getattr(args, "llm", None),
        llm_flags=_llm_only_flags(args),
        no_llm=getattr(args, "no_llm", False),
    )


def _ocr_requested(args, settings) -> bool:
    """Whether the user chose the OCR route: ``--ocr``, ``--ocr-url`` or a set ``OCR_BACKEND``."""
    return bool(
        getattr(args, "ocr", None)
        or getattr(args, "ocr_url", None)
        or "backend" in settings.ocr.model_fields_set
    )


def _resolve_crossref_flag(args) -> bool | None:
    """``--crossref`` → True, ``--no-crossref`` → False, neither → None (setting decides)."""
    if getattr(args, "crossref", False):
        return True
    if getattr(args, "no_crossref", False):
        return False
    return None


def _apply_runtime_settings(args) -> None:
    """Apply preset config + per-flag overrides to the global Settings.

    Mutates ``bibr.config.Settings`` in place. Called once at the start
    of ``_run_process`` before pipeline construction so the pipeline
    sees the final values.
    """
    from rich.console import Console
    from rich.markup import escape

    import bibr.config

    if args.preset:
        from bibr.presets import PresetManager

        # Mirror the same dir-resolution logic the ``preset`` subcommand uses
        # so ``BIBR_PRESETS_DIR`` is honored consistently across both call sites.
        presets_dir_str = os.environ.get("BIBR_PRESETS_DIR", "").strip()
        mgr = (
            PresetManager(presets_dir=Path(presets_dir_str)) if presets_dir_str else PresetManager()
        )
        try:
            unknown = mgr.apply_to_settings(args.preset, bibr.config.Settings)
            if unknown:
                Console(stderr=True).print(
                    f"[yellow]![/yellow] Preset [cyan]{args.preset}[/cyan] has "
                    f"{len(unknown)} unknown setting(s) (skipped): "
                    + escape(", ".join(sorted(unknown)))
                )
        except FileNotFoundError:
            console = Console(stderr=True)
            console.print(f"[red]✗[/red] Preset [cyan]{args.preset}[/cyan] not found.")
            names = mgr.list_presets()
            console.print("  Available: " + (", ".join(names) or "(none)"))
            sys.exit(1)

    if args.llm_provider:
        bibr.config.Settings.llm.provider = args.llm_provider
        bibr.config.Settings.llm.apply_provider_default_model()
    if args.llm_model:
        bibr.config.Settings.llm.model = args.llm_model
    if args.no_equations:
        bibr.config.Settings.EQUATION_EXTRACTION = False
    # --refs / --ref-seg are per-run options: they ride the pipeline's
    # RunConfig (see the LocalPipeline construction) instead of mutating
    # the process-global Settings.


def _resolve_llm_backend(raw: str) -> str:
    """Resolve the ``--llm`` value to a concrete backend.

    Thin wrapper over :func:`bibr.local.pipeline.resolve_llm_backend` — the
    library-level single source of truth, shared with ``LocalPipeline``.
    """
    from bibr.local.pipeline import resolve_llm_backend

    return resolve_llm_backend(raw)


def _preflight_local_backend(backend: str) -> str | None:
    """Return an error message if a managed local LLM backend can't run here.

    Fail fast before OCR: unsupported hardware (no NVIDIA GPU, not Apple
    Silicon) or a missing launcher would otherwise crash/hang deep in the run.
    Returns ``None`` when the backend is ready to launch; a slow llama.cpp
    build is logged as a warning, not refused.
    """
    problem = _local_llm_backend_blocker(backend)
    if problem is None and backend == "llama-cpp":
        from bibr.local.llama_cpp import (
            cuda_steering_hint,
            find_llama_server,
            install_hint,
            probe_backend_kind,
            probe_gpu_backend,
        )

        prefix = find_llama_server()
        if prefix is None:
            return None
        # Soft check only — CPU builds are usable but painfully slow; surface
        # that at doctor time rather than hard-failing preflight (users may be
        # smoke-testing install order before swapping in a CUDA binary).
        gpu = probe_gpu_backend(prefix)
        if gpu is False:
            import logging

            logging.getLogger(__name__).warning(
                "llama.cpp appears to be a CPU-only build; OCR/LLM will be very slow. %s",
                install_hint(),
            )
        else:
            # Not CPU-only: the Vulkan build runs but is slower than CUDA on
            # NVIDIA for bibr's prefill-heavy workload — nudge toward CUDA.
            steer = cuda_steering_hint(probe_backend_kind(prefix))
            if steer is not None:
                import logging

                logging.getLogger(__name__).warning("%s", steer)
    return problem


def _local_llm_backend_blocker(backend: str) -> str | None:
    """Why a managed local LLM backend cannot start here, or ``None``.

    The hard half of :func:`_preflight_local_backend`, without its llama.cpp
    speed warnings: ``bibr doctor`` reports those itself, and takes its
    pass/fail verdict from here so it refuses exactly what ``bibr chew`` refuses.
    """
    import importlib.util
    import shutil

    from bibr.local.llm_models import detect_hardware

    if backend == "llmster":
        if shutil.which("lms") is not None:
            return None
        return (
            "llmster backend selected but LM Studio's `lms` CLI is not installed. "
            "Install it manually with: curl -fsSL https://lmstudio.ai/install.sh | bash"
        )

    if backend == "llama-cpp":
        from bibr.local.llama_cpp import find_llama_server, install_hint

        if find_llama_server() is None:
            return (
                "llama.cpp backend selected but no server executable was found. " + install_hint()
            )
        return None

    platform_key, _ = detect_hardware()
    if platform_key is None:
        return (
            "--llm local needs an NVIDIA GPU (vLLM) or Apple Silicon "
            "(vllm-mlx); neither was detected. Use --llm cloud (hosted API) or run "
            "a local Ollama server (LLM_PROVIDER=ollama, LLM_BACKEND=cloud)."
        )
    if backend == "vllm" and (
        importlib.util.find_spec("vllm") is None and shutil.which("uv") is None
    ):
        return (
            "vllm backend selected but neither the 'vllm' package nor 'uv' is "
            "available. Install with: uv sync --extra vllm, or install uv "
            "(curl -LsSf https://astral.sh/uv/install.sh | sh)."
        )
    if backend == "vllm-mlx":
        from bibr.local.vllm_mlx_runtime import vllm_mlx_unavailable_reason

        unavailable_reason = vllm_mlx_unavailable_reason()
        if unavailable_reason is not None:
            return (
                "vllm-mlx backend selected but not installed or incomplete "
                f"({unavailable_reason}). Install with: "
                "uv sync --extra local-mlx  (or uv pip install vllm-mlx)."
            )
    if backend == "rapid-mlx":
        from bibr.local.rapid_mlx import rapid_mlx_unavailable_reason

        unavailable_reason = rapid_mlx_unavailable_reason()
        if unavailable_reason is not None:
            return (
                "rapid-mlx backend selected but the launcher is not available "
                f"({unavailable_reason}). Install with: pip install 'rapid-mlx[guided]'."
            )
    return None


def _ocr_candidate_unavailable_reason(backend: str) -> str | None:
    """Cheap, start-nothing check that a managed local OCR runtime can launch.

    Returns ``None`` for backends this function does not vet (HTTP and cloud
    backends validate themselves at startup) and for runtimes whose launcher
    and hardware are present.
    """
    if backend == "paddle-vllm":
        import importlib.util
        import shutil

        from bibr.ocr.registry import paddle_vllm_unavailable_reason

        reason = paddle_vllm_unavailable_reason()
        if reason is not None:
            return reason
        if importlib.util.find_spec("vllm") is None and shutil.which("uv") is None:
            return (
                "vllm is not installed and uv is unavailable to bootstrap it "
                "(install with: uv sync --extra vllm)"
            )
        return None
    if backend == "glm-llama":
        from bibr.local.llama_cpp import find_llama_server, missing_binary_message

        if find_llama_server() is None:
            return missing_binary_message()
        return None
    if backend in {"paddle-rapid-mlx", "glm-rapid-mlx"}:
        from bibr.local.rapid_mlx import rapid_mlx_unavailable_reason

        reason = rapid_mlx_unavailable_reason()
        if reason is not None:
            return f"rapid-mlx launcher not available ({reason}); pip install 'rapid-mlx[guided]'"
        return None
    if backend == "glm-mlx":
        return "this backend is disabled (vllm-mlx produced corrupted text); use glm-rapid-mlx"
    return None


def _preflight_ocr_runtime(config: ResolvedRunConfig) -> str | None:
    """Fail fast when no local OCR runtime can start for the PDFs in this run.

    The transactional startup chain would reach the same verdict — but only
    after the layout model has loaded and, for ``paddle-vllm``, after a
    multi-GB vLLM bootstrap. Explicit ``--ocr-url`` runs and cloud/HTTP
    backends are not vetted here; their own startup paths report failures.
    """
    if config.ocr_url:
        return None
    from bibr.ocr.registry import bootstrap_kwargs

    return _ocr_runtime_blocker(
        config.ocr_backend,
        llm_backend=_run_llm_backend(config),
        **bootstrap_kwargs(getattr(config, "vllm_bootstrap", True)),
    )


def _run_llm_backend(config: ResolvedRunConfig) -> str:
    """The LLM backend whose server the run starts: ``cloud`` under ``--no-llm`` or fast."""
    # ``getattr``: the preflights also take any object shaped like the config.
    skips = config.no_llm or getattr(config, "mode", "full") == "fast"
    return "cloud" if skips else config.llm_backend


def _ocr_runtime_blocker(
    ocr_backend: str, *, llm_backend: str | None = None, vllm_bootstrap: bool = True
) -> str | None:
    """Why no candidate of ``ocr_backend``'s startup chain can start here, or ``None``.

    Shared by the ``bibr chew``/``bibr batch`` preflight and ``bibr doctor``.
    ``llm_backend`` is the run's resolved LLM backend and ``vllm_bootstrap``
    whether a missing vLLM may be bootstrapped (see
    ``bibr.ocr.registry.automatic_backend_names``). When the chain has nothing
    else, the error names the paddle-vllm bootstrap it left out and how to opt in.
    """
    from bibr.config import Settings
    from bibr.ocr.registry import (
        VLLM_BOOTSTRAP_NOT_REQUESTED,
        VLLM_BOOTSTRAP_OPT_IN,
        automatic_backend_names,
        bootstrap_kwargs,
        resolve_backend_candidates,
    )

    candidates = resolve_backend_candidates(
        ocr_backend, Settings, llm_backend=llm_backend, **bootstrap_kwargs(vllm_bootstrap)
    )
    blockers: list[str] = []
    for candidate in candidates:
        reason = _ocr_candidate_unavailable_reason(candidate.backend)
        if reason is None:
            return None
        blockers.append(f"{candidate.backend}: {reason}")
    if (
        ocr_backend == "paddle"
        and not vllm_bootstrap
        and "paddle-vllm" in automatic_backend_names(llm_backend=llm_backend)
        and "paddle-vllm" not in {c.backend for c in candidates}
    ):
        blockers.insert(0, f"paddle-vllm: {VLLM_BOOTSTRAP_NOT_REQUESTED}; {VLLM_BOOTSTRAP_OPT_IN}")
    if not blockers:
        return None
    alternatives = (
        "Alternatives: point --ocr-url at an external OCR server, or use a cloud vision "
        "backend (--ocr gemini|openai|anthropic)."
    )
    if len(blockers) == 1 and ocr_backend != "paddle":
        return f"OCR backend cannot start here — {blockers[0]}. {alternatives}"
    listed = "\n".join(f"  - {blocker}" for blocker in blockers)
    return (
        f"No local OCR runtime can start on this machine for PDF input:\n{listed}\n{alternatives}"
    )


def _preflight_opencv() -> tuple[str, str] | None:
    """``(problem, repair command)`` when PDF input needs opencv and it is unusable.

    opencv is only on the torch layout path (transformers' image processor
    imports cv2); a core install runs layout through ONNX Runtime, where the
    crop and post-processing are Pillow/numpy, so a missing cv2 is not a
    reason to refuse the PDF. ``bibr chew`` and ``bibr batch`` both ask here,
    so they refuse the same runs with the same repair.
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return None
    # Looked up through the package root at call time so that
    # ``monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", ...)``
    # (used by existing tests) takes effect.
    from bibr.local.cli import _opencv_unavailable_reason

    reason = _opencv_unavailable_reason()
    if reason is None:
        return None
    repair = (
        "uv sync --extra torch"
        if "not installed" in reason
        else "uv pip install --reinstall opencv-python-headless"
    )
    return f"Layout/OCR image runtime unavailable: {reason}", repair


def _managed_llm_model(backend: str, settings) -> str:
    """The model a managed backend will serve — mirrors each server's own resolution."""
    if backend == "llmster":
        return settings.llm.llmster_model or settings.llm.llmster_model_id
    if settings.llm.local_model:
        return settings.llm.local_model
    if backend == "rapid-mlx":
        return settings.llm.rapid_mlx_model
    from bibr.local.llm_models import default_local_model

    return default_local_model(backend)


def _managed_llm_weight_repo(backend: str, settings) -> str | None:
    if backend == "llmster":
        return None
    return _managed_llm_model(backend, settings)
