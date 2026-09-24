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


def onnxruntime_repair_command() -> list[str]:
    """The command that rewrites the ``onnxruntime`` distribution's files.

    The uv form resyncs the project; the pip form is for environments pip
    manages, pinned to the installed version when metadata still has one.
    """
    from importlib import metadata

    try:
        dist = metadata.distribution("onnxruntime")
        installer = (dist.read_text("INSTALLER") or "").strip()
        requirement = f"onnxruntime=={dist.version}"
    except Exception:  # noqa: BLE001 — missing or unreadable metadata: assume a uv project
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
    return ["uv", "sync", "--reinstall-package", "onnxruntime"]


def import_onnxruntime(feature: str = "ONNX Runtime inference"):
    """Import ``onnxruntime``, raising a fixable error when it is missing or hollow.

    ``onnxruntime`` and ``onnxruntime-gpu`` write the same ``onnxruntime/``
    directory, so uninstalling one deletes files the other still needs. A
    ``uv sync`` without ``--extra gpu`` after a GPU install does exactly that:
    ``onnxruntime`` stays installed per its metadata, ``import onnxruntime``
    finds only a namespace package, and ``get_available_providers`` is missing.
    That raises :class:`ConfigurationError` with the repair command instead of
    an ``AttributeError`` somewhere later.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        from bibr.utils.ml_extra import onnxruntime_import_error

        raise onnxruntime_import_error(feature) from e

    if not hasattr(ort, "get_available_providers"):
        import shlex

        from bibr.exceptions import ConfigurationError

        raise ConfigurationError(
            f"{feature} requires onnxruntime, but the onnxruntime package is empty: its "
            "files were deleted, typically by uninstalling onnxruntime-gpu (which a "
            "`uv sync` without `--extra gpu` does after a GPU install), since the two "
            "share the onnxruntime/ directory. Reinstall it with: "
            f"{shlex.join(onnxruntime_repair_command())}"
        )
    return ort


def get_ort_providers(
    *,
    enable_cuda: bool = True,
    model_name: str = "",
    gpu_mem_limit: int | None = None,
) -> list[str | tuple[str, dict]]:
    """Build ONNX Runtime provider list.

    Falls back gracefully: CUDA → CoreML → CPU. Including CUDA first loads
    the CUDA libraries it needs (see :func:`_preload_cuda_libraries`).

    Args:
        enable_cuda: Whether to include CUDAExecutionProvider.
        model_name: Human-readable model name for logging.
        gpu_mem_limit: Optional cap on the CUDA EP arena size in bytes.

    Returns:
        Ordered list of providers suitable for ``ort.InferenceSession(providers=...)``.

    Raises:
        ConfigurationError: onnxruntime is installed but its files are gone
            (see :func:`import_onnxruntime`).
    """
    ort = import_onnxruntime()
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
        providers.append(("CUDAExecutionProvider", cuda_opts))

    if "CoreMLExecutionProvider" in available:
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
                    "%s will run on CPU. Install the gpu extra as described at "
                    "https://bibr.org/getting-started/install/#gpu-onnx-runtime",
                    model_name or "model",
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


def enable_cuda_for(device: str | None) -> bool:
    """Map a torch-style device request onto the CUDA provider switch.

    ``None`` means auto (use CUDA when the provider exists); ``cpu`` forces the
    CPU provider; anything else (``cuda``, ``cuda:1``, ``mps``) allows CUDA and
    otherwise falls through the provider chain.
    """
    if device is None:
        return True
    return str(device).split(":", 1)[0].strip().lower() != "cpu"


def create_session(
    model_path: str | Path,
    *,
    device: str | None = None,
    model_name: str = "",
    gpu_mem_limit: int | None = None,
):
    """Open an ``InferenceSession`` on ``model_path`` and report its device.

    Returns ``(session, device)`` where ``device`` is ``"cuda"`` or ``"cpu"``,
    the one the session got (see :func:`session_device`). Graph
    optimisations are left at ORT's default (all), which is what the
    wtpsplit segmenter already runs with.
    """
    ort = import_onnxruntime(model_name or "ONNX Runtime inference")
    providers = get_ort_providers(
        enable_cuda=enable_cuda_for(device),
        model_name=model_name,
        gpu_mem_limit=gpu_mem_limit,
    )
    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only; ORT's warnings are noisy at load
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    return session, session_device(session, providers, model_name=model_name)
