"""Shared ONNX Runtime execution provider configuration.

Builds the provider chain for ONNX Runtime sessions. All model deployments
call ``get_ort_providers()`` (or ``create_session()``) instead of constructing
providers ad-hoc.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

GPU_ONNX_RUNTIME_DOCS = "https://bibr.org/getting-started/install/#gpu-onnx-runtime"

_preload_lock = threading.Lock()
_cuda_libraries_preloaded = False


def _preload_cuda_libraries(ort) -> None:
    """Load the CUDA and cuDNN libraries the CUDA provider needs, once per process.

    ``onnxruntime-gpu[cuda,cudnn]`` installs CUDA and cuDNN as ``nvidia-*``
    wheels, but ORT's CUDA provider library does not look in their
    directories. Unless something loaded them first (a PyTorch built for the
    same CUDA major, imported earlier, or ``LD_LIBRARY_PATH``), the provider
    fails to load and ORT runs the session on CPU. ``onnxruntime.preload_dlls()`` (ORT >= 1.21) loads
    them. It reports failures with ``print``; those go to the log instead of
    stdout, next to the fallback warning in :func:`session_device`.
    """
    global _cuda_libraries_preloaded
    with _preload_lock:
        if _cuda_libraries_preloaded:
            return
        _cuda_libraries_preloaded = True
        preload = getattr(ort, "preload_dlls", None)
        if preload is None:
            return
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                preload()
        except Exception as exc:  # noqa: BLE001 — a failed preload must not break ORT setup
            logger.warning("onnxruntime.preload_dlls() failed: %s", exc)
        lines = [line.strip() for line in printed.getvalue().splitlines() if line.strip()]
        if lines:
            logger.warning("onnxruntime.preload_dlls(): %s", " ".join(lines))


def cuda_provider_available() -> bool:
    """True when onnxruntime exposes ``CUDAExecutionProvider`` in this process.

    The auto-detect signal for ONNX models (sentence segmenter): if the GPU
    execution provider is present, default to using it. Degrades to ``False``
    when onnxruntime is missing or fails to import — never raises.
    """
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001 — absence/breakage just means "no CUDA EP"
        return False


def onnxruntime_gpu_reinstall_command(
    version: str, *, uv: str | None = "uv", python: str | None = None
) -> list[str]:
    """The command that makes ``onnxruntime-gpu`` ``version`` the build that loads.

    The core ``onnxruntime`` dependency and the ``gpu`` extra's
    ``onnxruntime-gpu`` are separate distributions that write the same
    ``onnxruntime/`` directory, so the build that loads is whichever wheel's
    files were written last. Reinstalling ``onnxruntime-gpu`` rewrites all of
    them from the GPU wheel. ``onnxruntime`` stays installed: ``uv run``
    reinstalls a missing one, and its files would replace the GPU build's.
    ``uv=None`` gives the pip form, for environments pip manages.
    """
    python = python or sys.executable
    if uv:
        return [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "--reinstall-package",
            "onnxruntime-gpu",
            f"onnxruntime-gpu[cuda,cudnn]=={version}",
        ]
    return [
        python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        f"onnxruntime-gpu=={version}",
    ]


_gpu_build_check_lock = threading.Lock()
_gpu_build_shadowed: bool | None = None


def _cpu_build_shadows_gpu(available: set[str]) -> bool:
    """True when ``onnxruntime-gpu`` is installed but the CPU build is the one loaded.

    ``available`` is ``ort.get_available_providers()``, and only the GPU
    build's binary registers ``CUDAExecutionProvider``. Without it, both
    distributions installed means the CPU wheel's files won: installing both
    in one step races, and reinstalling ``onnxruntime`` later writes its files
    over the GPU build's. Every ONNX model then runs on CPU. The loaded build
    cannot change within a process, so this is checked once; finding it logs
    a warning with the command that repairs it.
    """
    global _gpu_build_shadowed
    with _gpu_build_check_lock:
        if _gpu_build_shadowed is None:
            builds = None if "CUDAExecutionProvider" in available else _installed_builds()
            _gpu_build_shadowed = builds is not None
            if builds is not None:
                import shlex

                cpu_version, gpu_version, installer = builds
                command = onnxruntime_gpu_reinstall_command(
                    gpu_version, uv="uv" if installer == "uv" else None
                )
                logger.warning(
                    "onnxruntime %s and onnxruntime-gpu %s are both installed, and the CPU "
                    "build is the one loaded: the two share the onnxruntime/ directory and "
                    "the CPU wheel's files are the ones on disk, so ONNX models run on CPU. "
                    "To load the GPU build, reinstall it so its files are written last: %s",
                    cpu_version,
                    gpu_version,
                    shlex.join(command),
                )
        return _gpu_build_shadowed


def _installed_builds() -> tuple[str, str, str] | None:
    """``(onnxruntime version, onnxruntime-gpu version, onnxruntime-gpu installer)``
    when both distributions are installed, else ``None``."""
    from importlib import metadata

    try:
        cpu_version = metadata.version("onnxruntime")
        gpu = metadata.distribution("onnxruntime-gpu")
        return cpu_version, gpu.version, (gpu.read_text("INSTALLER") or "").strip()
    except metadata.PackageNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 — unreadable metadata must not break ORT setup
        logger.debug("Could not read the installed onnxruntime distributions: %s", exc)
        return None


def _installed_onnxruntime_version() -> str | None:
    """The ``onnxruntime`` distribution's version, or ``None`` when its metadata is missing."""
    from importlib import metadata

    try:
        return metadata.version("onnxruntime")
    except Exception:  # noqa: BLE001 — not installed, or unreadable metadata
        return None


def onnxruntime_repair_command() -> list[str]:
    """The command that rewrites the ``onnxruntime`` distribution's files.

    It pins the installed version when metadata still has one. The uv form
    names this interpreter, so it works in a uv project and in a plain venv
    that ``uv pip install`` filled alike; the pip form is for environments pip
    manages. Either one writes the CPU build's files (see
    :func:`onnxruntime_repair_hint`).
    """
    from importlib import metadata

    try:
        dist = metadata.distribution("onnxruntime")
        installer = (dist.read_text("INSTALLER") or "").strip()
        requirement = f"onnxruntime=={dist.version}"
    except Exception:  # noqa: BLE001 — missing or unreadable metadata: uv, unpinned
        installer, requirement = "uv", "onnxruntime"
    if installer == "pip":
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            requirement,
        ]
    return [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall-package",
        "onnxruntime",
        requirement,
    ]


def onnxruntime_repair_hint(*, lines: bool = False) -> str:
    """The repair command, then the GPU step to repeat after it.

    One line for an error message, or two lines for a ``bibr doctor`` hint.
    """
    import shlex

    command = f"Reinstall it with: {shlex.join(onnxruntime_repair_command())}"
    gpu = (
        "a GPU install, repeat the onnxruntime-gpu reinstall afterwards, since this "
        f"writes the CPU build: {GPU_ONNX_RUNTIME_DOCS}"
    )
    return f"{command}\nOn {gpu}" if lines else f"{command}; on {gpu}"


def import_onnxruntime(feature: str = "ONNX Runtime inference"):
    """Import ``onnxruntime``, raising a fixable error when it is missing or broken.

    ``onnxruntime`` and ``onnxruntime-gpu`` write the same ``onnxruntime/``
    directory, so uninstalling one deletes files the other still needs. A
    ``uv sync`` without ``--extra gpu`` after a GPU install does exactly that,
    and ``onnxruntime`` stays installed per its metadata. When ``import
    onnxruntime`` then finds only a namespace package, without
    ``get_available_providers``, this raises :class:`ConfigurationError` with
    the repair command instead of an ``AttributeError`` somewhere later. When
    the import itself fails while the metadata is there (the whole directory
    gone, a CPU/GPU file mix, a missing native library), the ``ImportError``
    gives the same repair: ``pip install onnxruntime`` would do nothing, since
    the requirement is already satisfied.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        version = _installed_onnxruntime_version()
        if version is None:
            from bibr.utils.ml_extra import onnxruntime_import_error

            raise onnxruntime_import_error(feature) from e
        raise ImportError(
            f"{feature} requires onnxruntime, but onnxruntime {version} is installed and "
            f"importing it failed: {e}. {onnxruntime_repair_hint()}"
        ) from e

    if not hasattr(ort, "get_available_providers"):
        from bibr.exceptions import ConfigurationError

        raise ConfigurationError(
            f"{feature} requires onnxruntime, but the onnxruntime package is empty: its "
            "files were deleted, typically by uninstalling onnxruntime-gpu (which a "
            "`uv sync` without `--extra gpu` does after a GPU install), since the two "
            f"share the onnxruntime/ directory. {onnxruntime_repair_hint()}"
        )
    return ort


def get_ort_providers(
    *,
    enable_cuda: bool = True,
    model_name: str = "",
    gpu_mem_limit: int | None = None,
    enable_coreml: bool = True,
    cuda_device_id: int | None = None,
) -> list[str | tuple[str, dict]]:
    """Build ONNX Runtime provider list.

    Falls back gracefully: CUDA → CoreML → CPU. Including CUDA first loads
    the CUDA libraries it needs (see :func:`_preload_cuda_libraries`).

    Args:
        enable_cuda: Whether to include CUDAExecutionProvider.
        model_name: Human-readable model name for logging and import errors.
        gpu_mem_limit: Optional cap on the CUDA EP arena size in bytes.
        enable_coreml: Whether to include CoreMLExecutionProvider.
        cuda_device_id: GPU the CUDA provider runs on (ORT's default is 0).

    Returns:
        Ordered list of providers suitable for ``ort.InferenceSession(providers=...)``.

    Raises:
        ConfigurationError: onnxruntime is installed but its files are gone
            (see :func:`import_onnxruntime`).
        ImportError: onnxruntime is missing or fails to import.
    """
    ort = import_onnxruntime(model_name or "ONNX Runtime inference")
    available = set(ort.get_available_providers())
    providers: list[str | tuple[str, dict]] = []

    if enable_cuda and "CUDAExecutionProvider" in available:
        _preload_cuda_libraries(ort)
        cuda_opts: dict[str, str | int | bool] = {
            "arena_extend_strategy": "kSameAsRequested",
            "do_copy_in_default_stream": True,
        }
        if gpu_mem_limit is not None:
            cuda_opts["gpu_mem_limit"] = gpu_mem_limit
        if cuda_device_id is not None:
            cuda_opts["device_id"] = cuda_device_id
        providers.append(("CUDAExecutionProvider", cuda_opts))

    if enable_coreml and "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")

    providers.append("CPUExecutionProvider")

    # Checked whether or not this chain asks for CUDA: the segmenter asks for
    # CPU when cuda_provider_available() finds no CUDA EP, and a shadowed GPU
    # build is one reason it finds none.
    shadowed = _cpu_build_shadows_gpu(available)
    if enable_cuda and "CUDAExecutionProvider" not in available and not shadowed:
        # Only consult a torch that is *already* loaded: the ONNX path must not
        # import torch just to phrase this warning (and a core install has none).
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_available():
                logger.warning(
                    "CUDA GPU detected but onnxruntime-gpu is not installed — "
                    "%s will run on CPU. Install the gpu extra as described at %s",
                    model_name or "model",
                    GPU_ONNX_RUNTIME_DOCS,
                )
        except Exception as exc:  # noqa: BLE001 — a broken torch must not break ORT setup
            logger.debug("torch CUDA probe failed while building ORT providers: %s", exc)

    return providers


def selected_device(providers: list[str | tuple[str, dict]]) -> str:
    """``"cuda"`` when the CUDA provider heads the chain, else ``"cpu"``."""
    for provider in providers:
        name = provider[0] if isinstance(provider, tuple) else provider
        if name == "CUDAExecutionProvider":
            return "cuda"
    return "cpu"


def session_device(
    session, requested: list[str | tuple[str, dict]], *, model_name: str = ""
) -> str:
    """The device ``session`` actually runs on: ``"cuda"`` or ``"cpu"``.

    Read from the session's own providers, not the ``requested`` chain: ORT
    drops a provider that fails to start (a CUDA library it cannot load, no
    visible GPU, a driver too old for its CUDA) and runs the session on the
    next one without raising. Losing requested CUDA that way logs a warning.
    """
    device = selected_device(session.get_providers())
    if device != "cuda" and selected_device(requested) == "cuda":
        logger.warning(
            "%s requested CUDA, but onnxruntime could not start its CUDA execution "
            "provider and runs it on CPU. onnxruntime's own error names the cause, "
            "usually a CUDA or cuDNN library it cannot load (install "
            "'onnxruntime-gpu[cuda,cudnn]') or no visible GPU.",
            model_name or "ONNX model",
        )
    return device


_ARENA_SHRINKAGE = "memory.enable_memory_arena_shrinkage"


class _ArenaShrinkingSession:
    """A CUDA session whose every run frees the arena memory it no longer uses.

    Everything but ``run`` passes through to the wrapped ``InferenceSession``.
    """

    def __init__(self, session, run_options) -> None:
        self._session = session
        self._run_options = run_options

    def run(self, output_names, input_feed, run_options=None):
        return self._session.run(output_names, input_feed, run_options or self._run_options)

    def __getattr__(self, name: str):
        return getattr(self._session, name)


def shrink_arena_after_runs(session):
    """Make each run of a CUDA ``session`` give the memory it no longer uses back to CUDA.

    ORT's CUDA arena keeps every block it allocates, and bibr grows it by
    exactly the size a run asks for (``kSameAsRequested``). Input shapes change
    from call to call (page batches, reference and header lengths), so new
    shapes keep asking for blocks that no free one fits, and over a batch of
    papers the arenas grow until the GPU is full. A ``gpu_mem_limit`` only turns
    that into an earlier allocation failure. With
    ``memory.enable_memory_arena_shrinkage`` each run ends by freeing the
    arena regions nothing uses any more, so the arena holds the weights and
    what the runs in flight need. A session that is not on CUDA comes back
    unchanged.
    """
    if selected_device(session.get_providers()) != "cuda":
        return session
    import onnxruntime as ort

    cuda_options = session.get_provider_options().get("CUDAExecutionProvider", {})
    run_options = ort.RunOptions()
    run_options.add_run_config_entry(_ARENA_SHRINKAGE, f"gpu:{cuda_options.get('device_id', '0')}")
    return _ArenaShrinkingSession(session, run_options)


def enable_cuda_for(device: str | None) -> bool:
    """Map a torch-style device request onto the CUDA provider switch.

    ``None`` means auto (use CUDA when the provider exists); ``cpu`` forces the
    CPU provider; anything else (``cuda``, ``cuda:1``, ``mps``) allows CUDA and
    otherwise falls through the provider chain.
    """
    if device is None:
        return True
    return str(device).split(":", 1)[0].strip().lower() != "cpu"


def cuda_device_id_for(device: str | None) -> int | None:
    """The GPU index in a ``cuda:N`` request, else ``None`` (ORT's default GPU)."""
    kind, _, index = str(device or "").partition(":")
    if kind.strip().lower() != "cuda" or not index.strip().isdigit():
        return None
    return int(index)


def cpu_ort_session_options():
    """``SessionOptions`` with ORT's CPU memory arena disabled, if available.

    The CPU arena keeps its peak allocation for the life of the session
    (audit-measured: layout ~5 GB at batch 8, SaT ~2.8 GB after one paper),
    so CPU-only sessions for those models opt out and return freed blocks to
    the OS. Returns ``None`` when onnxruntime cannot be imported, so callers
    that build ``ort_kwargs`` for wtpsplit's ``SaT`` can skip the option.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    return options


def create_session(
    model_path: str | Path,
    *,
    device: str | None = None,
    model_name: str = "",
    gpu_mem_limit: int | None = None,
    disable_cpu_arena: bool = False,
):
    """Open an ``InferenceSession`` on ``model_path`` and report its device.

    Returns ``(session, device)`` where ``device`` is ``"cuda"`` or ``"cpu"``,
    the one the session got (see :func:`session_device`). A CUDA session comes
    back wrapped by :func:`shrink_arena_after_runs`. Graph optimisations are
    left at ORT's default (all), which is what the wtpsplit segmenter already
    runs with.

    ``disable_cpu_arena`` opts a CPU-only session out of ORT's CPU memory
    arena (see :func:`cpu_ort_session_options`). It is a no-op for sessions
    whose provider chain includes CUDA: the CUDA EP has its own arena
    (``kSameAsRequested`` plus per-run shrinkage) and disabling the CPU arena
    there is unverified. Only pass it for models where the arena cost was
    measured (layout, SaT), not blindly for every model.
    """
    ort = import_onnxruntime(model_name or "ONNX Runtime inference")
    allow_accelerators = enable_cuda_for(device)
    providers = get_ort_providers(
        enable_cuda=allow_accelerators,
        model_name=model_name,
        gpu_mem_limit=gpu_mem_limit,
        # ``cpu`` means the CPU provider: CoreML may compute in FP16, which
        # breaks the parity a CPU run is asked for.
        enable_coreml=allow_accelerators,
        cuda_device_id=cuda_device_id_for(device),
    )
    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only; ORT's warnings are noisy at load
    if disable_cpu_arena and selected_device(providers) != "cuda":
        options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    device = session_device(session, providers, model_name=model_name)
    return shrink_arena_after_runs(session), device
