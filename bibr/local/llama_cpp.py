"""Managed llama.cpp server shared by native Windows OCR and local LLM paths."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

from bibr.exceptions import UpstreamServiceError
from bibr.local.http_runtime import (
    MANAGED_LOCAL_LLM_RATE_LIMIT_RPM,
    LocalHttpError,
    guard_managed_server_port,
    request_bytes,
)

try:
    import winreg
except ImportError:  # pragma: no cover - only present on Windows
    winreg = None

logger = logging.getLogger(__name__)

_TERM_GRACE_S = 10
# Grace period after a /health 200 before declaring readiness: a sibling bibr
# process may have won a port race and answered our probe while our own bind
# is about to fail. Short enough to be noise against a 30-600 s startup.
_HEALTH_GRACE_S = 1.0
_LLAMA_CPP_DEFAULT_TIMEOUT_SECONDS = 300
_LLAMA_CPP_RELEASES_URL = "https://github.com/ggml-org/llama.cpp/releases"

# Flags applied for every managed server regardless of role. llama.cpp is
# bibr's low-memory path (Windows + ≤8 GB CUDA), so flash-attn + quantized KV
# are on by default. User ``extra_args`` can override any of these
# (see ``_merge_cli_args``). Role- and probe-specific flags are added by
# ``_role_runtime_args``.
_COMMON_RUNTIME_ARGS: tuple[str, ...] = (
    "--n-gpu-layers",
    "999",
    "--flash-attn",
    "on",
    "--cache-type-k",
    "q8_0",
    "--cache-type-v",
    "q8_0",
)

# Canonical option names (longest form) → all aliases llama.cpp accepts.
# Used so user ``-ctk f16`` suppresses the default ``--cache-type-k q8_0``.
_OPTION_ALIASES: dict[str, frozenset[str]] = {
    "--n-gpu-layers": frozenset({"--n-gpu-layers", "-ngl"}),
    "--parallel": frozenset({"--parallel", "-np"}),
    "--flash-attn": frozenset({"--flash-attn", "-fa"}),
    "--cache-type-k": frozenset({"--cache-type-k", "-ctk"}),
    "--cache-type-v": frozenset({"--cache-type-v", "-ctv"}),
    "--ctx-size": frozenset({"--ctx-size", "-c"}),
    "--kv-unified": frozenset({"--kv-unified", "-kvu"}),
    "--no-context-shift": frozenset({"--no-context-shift", "--context-shift"}),
    "--spec-type": frozenset({"--spec-type"}),
    "--no-mmproj": frozenset({"--no-mmproj"}),
    "--host": frozenset({"--host"}),
    "--port": frozenset({"--port"}),
    "--alias": frozenset({"--alias"}),
    "-hf": frozenset({"-hf", "-hfr", "--hf-repo"}),
}

# Cache of ``--help`` text keyed by ``tuple(prefix)`` so repeated launches in
# one process do not re-exec the binary just to re-probe its feature set.
_HELP_CACHE: dict[tuple[str, ...], str] = {}

# Long-option tokens (``--kv-unified`` etc.) as they appear in ``--help``.
_HELP_FLAG_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")


def probe_server_help(prefix: list[str], *, timeout: float = 8.0) -> str:
    """Return combined stdout+stderr of ``[*prefix, "--help"]`` (cached).

    Empty string on ``OSError`` / ``TimeoutExpired`` so callers stay
    conservative and add no optional flags when the probe cannot run.
    """
    key = tuple(prefix)
    cached = _HELP_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        result = subprocess.run(  # noqa: S603
            [*prefix, "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        _HELP_CACHE[key] = ""
        return ""
    combined = f"{result.stdout or ''}\n{result.stderr or ''}"
    _HELP_CACHE[key] = combined
    return combined


def supported_flags(prefix: list[str]) -> frozenset[str]:
    """Return the ``--long-option`` tokens the installed server advertises.

    Empty when the ``--help`` probe fails (conservative: no optional flags).
    """
    return frozenset(_HELP_FLAG_RE.findall(probe_server_help(prefix)))


# llama.cpp logs layer offload like:
#   "load_tensors: offloaded 32/32 layers to GPU"
#   "llama_model_load_from_file: offloaded 28/32 layers to GPU"
_OFFLOAD_RE = re.compile(
    r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+GPU",
    re.IGNORECASE,
)
_GPU_DEVICE_HINTS = (
    "cuda",
    "metal",
    "vulkan",
    "hip",
    "rocm",
    "ggml_cuda",
    "ggml_metal",
    "ggml_vulkan",
    "nvidia",
    "geforce",
    "radeon",
    "apple silicon",
    "mps",
)


def _windows_registry_path_values() -> list[str]:
    """Return User/Machine PATH values that may not be in this process yet."""
    if os.name != "nt" or winreg is None:
        return []

    access = getattr(winreg, "KEY_READ", 0) | getattr(winreg, "KEY_WOW64_64KEY", 0)
    keys = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    values: list[str] = []
    for root, subkey in keys:
        try:
            with winreg.OpenKey(root, subkey, 0, access) as key:
                for value_name in ("Path", "PATH"):
                    try:
                        value, _ = winreg.QueryValueEx(key, value_name)
                    except OSError:
                        continue
                    if value:
                        values.append(os.path.expandvars(str(value)))
                    break
        except OSError:
            continue
    return values


def _windows_augmented_path() -> str:
    sep = ";"
    parts: list[str] = []
    seen: set[str] = set()
    for value in (os.environ.get("PATH", ""), *_windows_registry_path_values()):
        for raw_part in value.split(sep):
            part = raw_part.strip().strip('"')
            if not part:
                continue
            key = part.casefold()
            if key in seen:
                continue
            seen.add(key)
            parts.append(part)
    return sep.join(parts)


def _which_llama_command(command: str) -> str | None:
    found = shutil.which(command)
    if found or os.name != "nt":
        return found

    # `winget install llama.cpp` can update the User PATH while the current
    # PowerShell/Python process still has the old environment. Read the current
    # registry value so users do not have to restart just for bibr to find it.
    search_path = _windows_augmented_path()
    if not search_path:
        return None
    return shutil.which(command, path=search_path)


def find_llama_server() -> list[str] | None:
    """Return the installed llama.cpp server command prefix, if any."""
    standalone = _which_llama_command("llama-server")
    if standalone:
        return [standalone]
    unified = _which_llama_command("llama")
    if unified:
        return [unified, "serve"]
    return None


def install_hint(*, platform_name: str | None = None) -> str:
    """One-line install guidance for the current (or given) platform."""
    plat = platform_name if platform_name is not None else sys.platform
    if plat == "win32" or plat.startswith("win"):
        return "Install with: winget install llama.cpp  (or a CUDA build from the GitHub releases)"
    if plat == "darwin":
        return (
            "Install llama.cpp with GPU support "
            f"(Homebrew: `brew install llama.cpp`, or a release binary from {_LLAMA_CPP_RELEASES_URL}) "
            "and ensure `llama-server` is on PATH"
        )
    # Linux and others: CUDA/Vulkan prebuilt or source build.
    return (
        f"Install a CUDA (or Vulkan) build of llama.cpp from {_LLAMA_CPP_RELEASES_URL} "
        "(extract so `llama-server` is on PATH), or build from source with GGML_CUDA=ON. "
        "CPU-only builds work but are very slow for OCR/LLM"
    )


def missing_binary_message(*, platform_name: str | None = None) -> str:
    """Error text when no llama-server executable is on PATH."""
    return f"llama.cpp is not installed. {install_hint(platform_name=platform_name)}."


def _strip_one_quote_pair(token: str) -> str:
    """Drop one matching pair of surrounding quotes from *token*.

    Non-POSIX ``shlex`` keeps the quote characters inside the token (it does
    not follow Windows CommandLineToArgvW rules), and
    ``subprocess.list2cmdline`` then escapes them — so without this,
    llama-server would receive a path wrapped in literal quote characters.
    Only strips when the token both starts and ends with the same quote;
    ``prefix\"quoted part\"`` style tokens are left alone.
    """
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def split_extra_args(value: str, *, setting: str = "llama_cpp_extra_args") -> list[str]:
    """Split user CLI arguments with shlex using the host platform's rules.

    Pass each flag and its value as separate tokens, quoting values that
    contain spaces: ``--chat-template-file "C:\\path with space\\t.jinja"``.
    That is the only supported form: ``--flag="value with space"`` still
    splits at the space on Windows (non-POSIX shlex), as do JSON values
    with escaped quotes, so use the separate-token form instead.
    """
    posix = os.name != "nt"
    try:
        parts = shlex.split(value, posix=posix)
    except ValueError as exc:
        raise UpstreamServiceError(
            "local inference",
            f"Could not parse {setting}={value!r}: {exc}. Quote paths with spaces, e.g. "
            '--chat-template-file "C:\\models\\my model\\template.jinja".',
        ) from exc
    if not posix:
        parts = [_strip_one_quote_pair(part) for part in parts]
    return parts


def _canonical_option(flag: str) -> str | None:
    """Map a CLI flag to its canonical long form, or None if unknown."""
    for canonical, aliases in _OPTION_ALIASES.items():
        if flag in aliases:
            return canonical
    return None


def _options_present(args: list[str]) -> set[str]:
    """Canonical option names present in *args* (value-bearing and bare flags)."""
    present: set[str] = set()
    for token in args:
        if not token.startswith("-"):
            continue
        # Support ``--flag=value`` forms.
        bare = token.split("=", 1)[0]
        canonical = _canonical_option(bare)
        if canonical is not None:
            present.add(canonical)
        else:
            present.add(bare)
    return present


def _merge_cli_args(base: list[str], extra: list[str]) -> list[str]:
    """Drop base options that the user already set in *extra*, then append *extra*.

    llama.cpp is last-wins for many flags, but stripping defaults keeps the
    launched command readable and avoids contradictory pairs in logs.
    """
    if not extra:
        return list(base)
    user_opts = _options_present(extra)
    merged: list[str] = []
    i = 0
    while i < len(base):
        token = base[i]
        if token.startswith("-"):
            bare = token.split("=", 1)[0]
            canonical = _canonical_option(bare) or bare
            has_inline_value = "=" in token
            # Value-bearing options consume the next token when not ``--flag=val``.
            takes_value = (
                not has_inline_value and i + 1 < len(base) and not base[i + 1].startswith("-")
            )
            if canonical in user_opts:
                i += 1 if not takes_value else 2
                continue
            merged.append(token)
            if takes_value:
                merged.append(base[i + 1])
                i += 2
            else:
                i += 1
        else:
            merged.append(token)
            i += 1
    merged.extend(extra)
    return merged


def _role_runtime_args(role: str, available: frozenset[str]) -> list[str]:
    """Runtime flags for *role*, gating optional features on *available*.

    The startup retry strips every optional flag added here when the stderr
    tail matches an argument-parse error (see ``_ARG_ERROR_MARKERS``), so a
    build whose ``--help`` lists a flag it does not actually accept still
    recovers to a safe launch. Anything else — a crash, an OOM, any message
    outside those markers — fails fast with no retry.
    """
    if role not in ("ocr", "llm"):
        raise ValueError(f"unknown llama.cpp server role {role!r} (expected 'ocr' or 'llm')")
    args = list(_COMMON_RUNTIME_ARGS)
    if role == "ocr":
        # Image encode serializes across slots; extra slots only add queueing.
        args += ["--parallel", "1"]
        return args
    # role == "llm"
    if "--kv-unified" in available:
        # Slots share the full ctx budget, so two slots don't halve context.
        # Accepted tradeoff: the shared context-size budget (default 16384)
        # can still be oversubscribed if two concurrent requests both land
        # near max size (~6k prompt + 4096 completion tokens each); per-task
        # context slices (LLM_PER_TASK_CONTEXT) keep typical requests far
        # smaller than that, and ``--no-context-shift`` below converts any
        # remaining overflow into a hard, retryable error instead of
        # llama.cpp silently dropping the oldest prompt tokens.
        args += ["--parallel", "2", "--kv-unified"]
    else:
        # Without kv-unified, ``--parallel 2`` would halve per-slot context.
        args += ["--parallel", "1"]
    if "--no-context-shift" in available:
        # Context shift silently truncates the oldest prompt tokens on
        # overflow rather than erroring — for a small extraction model that
        # means silently dropped fields. Current llama.cpp builds already
        # default it off; this is insurance for builds where it's toggleable
        # or enabled by default.
        args += ["--no-context-shift"]
    if "--spec-type" in available:
        args += ["--spec-type", "ngram-mod"]
    if "--no-mmproj" in available:
        # The NuExtract3-GGUF HF repo ships a 676 MB BF16 mmproj that ``-hf``
        # auto-loads; the LLM path is text-only, so skip it.
        args += ["--no-mmproj"]
    if "--jinja" in available:
        # NuExtract passes its extraction template through chat-template
        # kwargs; llama.cpp only applies those kwargs when Jinja is enabled.
        args += ["--jinja"]
    return args


def _option_value(args: list[str], canonical: str) -> str | None:
    """Return the value passed for *canonical* (or any alias) in *args*."""
    aliases = _OPTION_ALIASES.get(canonical, frozenset({canonical}))
    for i, token in enumerate(args):
        bare = token.split("=", 1)[0]
        if bare not in aliases:
            continue
        if "=" in token:
            return token.split("=", 1)[1]
        return args[i + 1] if i + 1 < len(args) else None
    return None


def _n_slots_from_argv(argv: list[str]) -> int:
    """Server slot count from the launched argv.

    Single-slot unless the multi-slot LLM args are active
    (``--kv-unified`` plus ``--parallel N``); ``N`` is read as an int so
    ``--parallel 4`` actually yields 4 slots, not 1.
    """
    if "--kv-unified" not in _options_present(argv):
        return 1
    raw = _option_value(argv, "--parallel")
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


# Server-identity flags a user must never override via extra args: bibr polls
# the configured port/alias and would lose (then kill) a server that moved.
# `--host` is deliberately allowed: the base argv binds loopback and a user
# `--host` (e.g. 0.0.0.0, as on main) still answers the loopback health poll.
_IDENTITY_OPTIONS = frozenset({"--port", "--alias", "-hf"})


def _check_no_identity_override(extra: list[str], *, role: str) -> None:
    """Reject extra args that would move the server bibr thinks it started."""
    if role == "llm":
        extra_setting = "LLM_LLAMA_CPP_EXTRA_ARGS"
        port_setting = "LLM_LLAMA_CPP_PORT"
    else:
        extra_setting = "OCR_LLAMA_CPP_EXTRA_ARGS"
        port_setting = "OCR_LLAMA_CPP_PORT"
    for token in extra:
        bare = token.split("=", 1)[0]
        if _canonical_option(bare) in _IDENTITY_OPTIONS:
            raise UpstreamServiceError(
                "local inference",
                f"{extra_setting} must not set {bare}: the managed server's port, "
                f"alias and model are fixed, and bibr polls the configured port — it would "
                f"time out and kill a healthy server that moved. Set {port_setting} to move "
                "the server instead.",
            )


def build_server_argv(
    prefix: list[str],
    *,
    model: str,
    port: int,
    context_size: int,
    extra_args: str = "",
    # Default "ocr" = exactly the pre-rework args (--parallel 1, no optional
    # feature flags), so callers not yet passing a role keep today's
    # semantics; the "llm" role must be opted into explicitly.
    role: str = "ocr",
    available: frozenset[str] | None = None,
) -> list[str]:
    """Assemble the full ``llama-server`` argv including role-based optim defaults.

    *available* is the set of ``--long-option`` flags the installed server
    supports; ``None`` probes it via ``supported_flags(prefix)``. Pass an
    explicit set (e.g. ``frozenset()``) to keep tests subprocess-free.
    """
    if available is None:
        available = supported_flags(prefix)
    setting = "LLM_LLAMA_CPP_EXTRA_ARGS" if role == "llm" else "OCR_LLAMA_CPP_EXTRA_ARGS"
    extra = split_extra_args(extra_args, setting=setting)
    _check_no_identity_override(extra, role=role)
    fixed = [
        *prefix,
        "-hf",
        model,
        "--alias",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        *_role_runtime_args(role, available),
    ]
    return _merge_cli_args(fixed, extra)


def parse_gpu_offload(stderr: str) -> tuple[int, int] | None:
    """Return ``(offloaded, total)`` layers from llama.cpp stderr, if logged."""
    matches = _OFFLOAD_RE.findall(stderr)
    if not matches:
        return None
    # Prefer the last report (final load state).
    offloaded, total = matches[-1]
    return int(offloaded), int(total)


def _text_suggests_gpu_backend(text: str) -> bool | None:
    """Heuristic: True/False when text clearly indicates GPU or CPU-only; else None."""
    if not text or not text.strip():
        return None
    lower = text.lower()
    # Explicit CPU-only markers without any GPU backend.
    gpu_hits = any(h in lower for h in _GPU_DEVICE_HINTS)
    cpu_only_markers = (
        "no devices found" in lower
        or "ggml_cuda_init: failed" in lower
        or "no cuda-capable device" in lower
        or ("cuda error" in lower and "no cuda" in lower)
        or re.search(r"available devices:\s*cpu\s*$", lower, re.MULTILINE) is not None
        # A header followed by a real device table is not an empty list. Match
        # only when whitespace after the header reaches end-of-output.
        or re.search(r"available devices:\s*\Z", lower) is not None
    )
    if cpu_only_markers:
        return False
    if gpu_hits:
        return True
    # ``--list-devices`` with only "CPU" lines.
    if "available devices" in lower or "ggml_backend" in lower:
        lines = [ln.strip() for ln in lower.splitlines() if ln.strip()]
        device_lines = [
            ln
            for ln in lines
            if ln.startswith("cpu") or "device" in ln or any(h in ln for h in _GPU_DEVICE_HINTS)
        ]
        if device_lines and all(
            "cpu" in ln and not any(h in ln for h in _GPU_DEVICE_HINTS) for ln in device_lines
        ):
            return False
    return None


def probe_gpu_backend(
    prefix: list[str] | None = None,
    *,
    timeout: float = 8.0,
) -> bool | None:
    """Return whether the installed llama.cpp build has a GPU backend.

    ``True`` / ``False`` when the probe is conclusive; ``None`` when the binary
    is missing, the probe fails, or the output is ambiguous. Prefer
    ``--list-devices`` (current llama.cpp), then fall back to ``--version``.
    """
    cmd_prefix = find_llama_server() if prefix is None else prefix
    if not cmd_prefix:
        return None

    for probe_args in (["--list-devices"], ["--version"], ["-h"]):
        try:
            result = subprocess.run(  # noqa: S603
                [*cmd_prefix, *probe_args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        combined = f"{result.stdout or ''}\n{result.stderr or ''}"
        verdict = _text_suggests_gpu_backend(combined)
        if verdict is not None:
            return verdict
        # ``--list-devices`` with non-empty GPU-looking device table.
        if probe_args == ["--list-devices"] and combined.strip():
            lower = combined.lower()
            if any(h in lower for h in _GPU_DEVICE_HINTS):
                return True
            # Non-empty list-devices with no GPU keyword → treat as CPU-only.
            if "cpu" in lower:
                return False
    return None


# Backend kinds in precedence order, each mapped to the markers that identify it
# in ``--list-devices`` / ``--version`` output. Device *names* (nvidia, geforce,
# radeon, ...) are deliberately excluded: the Vulkan build lists NVIDIA cards as
# "Vulkan0: NVIDIA GeForce ...", so a name match would misread Vulkan as CUDA —
# exactly the case this probe exists to catch.
_BACKEND_KIND_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cuda", ("ggml_cuda", "cuda")),
    ("hip", ("ggml_hip", "rocm", "hipblas")),
    ("sycl", ("ggml_sycl", "sycl")),
    ("metal", ("ggml_metal", "metal")),
    ("vulkan", ("ggml_vulkan", "vulkan")),
    ("cpu", ("cpu",)),
)


def _classify_backend_kind(text: str) -> str | None:
    """Classify llama.cpp probe output into a backend kind, or None if unknown.

    Precedence cuda > hip > sycl > metal > vulkan > cpu; the first backend whose
    marker appears wins.
    """
    if not text or not text.strip():
        return None
    lower = text.lower()
    for kind, markers in _BACKEND_KIND_MARKERS:
        if any(marker in lower for marker in markers):
            return kind
    return None


# Cache of probe_backend_kind() results keyed by ``tuple(prefix)``, mirroring
# ``_HELP_CACHE``: every non-Vulkan build re-execs ``--list-devices``/
# ``--version`` on each ``LlamaCppServer`` construction otherwise, because
# ``_warn_cuda_steering_once``'s once-flag only latches when a Vulkan+NVIDIA
# hint actually fires.
_BACKEND_KIND_CACHE: dict[tuple[str, ...], str | None] = {}


def probe_backend_kind(
    prefix: list[str] | None = None,
    *,
    timeout: float = 8.0,
) -> str | None:
    """Return the installed llama.cpp build's GPU backend KIND, or None.

    One of ``"cuda"``, ``"hip"``, ``"metal"``, ``"vulkan"``, ``"sycl"`` or
    ``"cpu"``; ``None`` when the binary is missing, the probe fails, or the
    output names no backend. Where :func:`probe_gpu_backend` answers only
    present/absent, this identifies *which* backend so callers can steer
    Vulkan-on-NVIDIA users to a faster CUDA build. Prefers ``--list-devices``,
    then falls back to ``--version``. Result is cached per ``tuple(prefix)``.
    """
    cmd_prefix = find_llama_server() if prefix is None else prefix
    if not cmd_prefix:
        return None

    key = tuple(cmd_prefix)
    if key in _BACKEND_KIND_CACHE:
        return _BACKEND_KIND_CACHE[key]

    kind: str | None = None
    for probe_args in (["--list-devices"], ["--version"]):
        try:
            result = subprocess.run(  # noqa: S603
                [*cmd_prefix, *probe_args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        combined = f"{result.stdout or ''}\n{result.stderr or ''}"
        kind = _classify_backend_kind(combined)
        if kind is not None:
            break
    _BACKEND_KIND_CACHE[key] = kind
    return kind


def cuda_steering_hint(kind: str | None) -> str | None:
    """Steer Vulkan-on-NVIDIA users toward a faster CUDA llama.cpp build.

    Returns an actionable message when *kind* is ``"vulkan"`` and an NVIDIA GPU
    is present (``nvidia-smi`` visible via
    :func:`bibr.local.llm_models._nvidia_vram_gb`); ``None`` otherwise — no
    NVIDIA GPU, or a non-Vulkan build where a CUDA swap would not help.
    """
    if kind != "vulkan":
        return None
    from bibr.local.llm_models import _nvidia_vram_gb

    if _nvidia_vram_gb() is None:
        return None
    return (
        "llama.cpp is the Vulkan build (what `winget install llama.cpp` ships), but an "
        "NVIDIA GPU is present. A CUDA build is ~1.5-2x faster for bibr's prefill-heavy "
        "OCR/LLM. Download llama-<ver>-bin-win-cuda-12.4-x64.zip AND "
        "cudart-llama-bin-win-cuda-12.4-x64.zip from "
        f"{_LLAMA_CPP_RELEASES_URL}, extract both into one folder, and put it on PATH "
        "ahead of the winget install. GPUs newer than the GTX 10-series may use the "
        "cuda-13.x zips instead."
    )


# Fired at most once per process (both server roles share it).
_cuda_steering_warned = False


def _warn_cuda_steering_once(prefix: list[str]) -> None:
    """Log the Vulkan→CUDA steering hint once per process (log-only, never raises).

    A probe failure must never break a server that already started healthily.
    """
    global _cuda_steering_warned
    if _cuda_steering_warned:
        return
    try:
        hint = cuda_steering_hint(probe_backend_kind(prefix))
    except Exception:  # noqa: BLE001 - a steering probe must not fail startup
        return
    if hint is None:
        return
    _cuda_steering_warned = True
    logger.warning("%s", hint)


def log_offload_status(stderr: str) -> None:
    """Log a warning when the model loaded with zero GPU layers."""
    offload = parse_gpu_offload(stderr)
    if offload is None:
        return
    offloaded, total = offload
    if total <= 0:
        return
    if offloaded == 0:
        logger.warning(
            "llama.cpp offloaded 0/%d layers to GPU — inference will run on CPU and "
            "be very slow. Install a CUDA/Metal/Vulkan build of llama.cpp (%s).",
            total,
            _LLAMA_CPP_RELEASES_URL,
        )
    elif offloaded < total:
        logger.info(
            "llama.cpp partial GPU offload: %d/%d layers on GPU (rest on CPU)",
            offloaded,
            total,
        )
    else:
        logger.info("llama.cpp full GPU offload: %d/%d layers", offloaded, total)


# Substrings (lowercased) of a startup failure that blame argument parsing
# rather than model loading — only these justify the conservative-args retry.
# A load failure (e.g. CUDA OOM) would fail identically on retry, so it must
# surface immediately instead of doubling the time to failure.
_ARG_ERROR_MARKERS = (
    "unknown argument",
    "invalid argument",
    "unrecognized argument",
    "unrecognised argument",
    "invalid option",
    "unknown option",
    "parse error",
    "failed to parse",
    # llama.cpp wraps every flag-handler exception as
    # 'error while handling argument "<flag>": <reason>' (common/arg.cpp),
    # e.g. a build whose --help lists --spec-type but rejects its value
    # ('unknown speculative type: ngram-mod').
    "error while handling argument",
)


def _looks_like_arg_error(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _ARG_ERROR_MARKERS)


class LlamaCppServer:
    """Own a llama.cpp OpenAI-compatible server subprocess.

    llama.cpp publishes native Windows CUDA binaries and supports both GGUFs
    used by bibr's low-memory path. A fresh process per phase guarantees that
    the previous model and CUDA context are returned to the OS.
    """

    def __init__(
        self,
        *,
        model: str,
        port: int,
        context_size: int = 8192,
        startup_timeout: int = 600,
        extra_args: str = "",
        # Default "ocr" = exactly the pre-rework args (--parallel 1, no
        # optional feature flags), so callers not yet passing a role — like
        # LlamaCppOcrClient — keep today's semantics; the "llm" role must be
        # opted into explicitly.
        role: str = "ocr",
    ) -> None:
        self._model = model
        self._port = port
        self._role = role
        self._n_slots = 1
        self._process: subprocess.Popen | None = None
        self._stderr_fh = None
        self._stderr_log = None
        # ``--alias model`` (see build_server_argv) makes /v1/models report the
        # requested model string verbatim, so the guard's id match is exact.
        self._reused = guard_managed_server_port(
            role,
            base_url=self.base_url,
            model=model,
            server_label="llama.cpp",
            request_fn=request_bytes,
        )
        if self._reused:
            return

        prefix = find_llama_server()
        if prefix is None:
            raise UpstreamServiceError(
                "local inference",
                missing_binary_message(),
            )

        available = supported_flags(prefix)
        # Build both argvs BEFORE opening the log handle: extra-args parsing
        # (shlex errors, identity-flag overrides) must fail without leaking it.
        cmd = build_server_argv(
            prefix,
            model=model,
            port=port,
            context_size=context_size,
            extra_args=extra_args,
            role=role,
            available=available,
        )
        # Conservative launch = role defaults minus every probed optional
        # feature flag; used as the single startup-retry fallback.
        conservative_cmd = build_server_argv(
            prefix,
            model=model,
            port=port,
            context_size=context_size,
            extra_args=extra_args,
            role=role,
            available=frozenset(),
        )
        has_optional_features = cmd != conservative_cmd

        from bibr.utils.secure_temp import open_subprocess_log

        self._stderr_log, self._stderr_fh = open_subprocess_log("llama", port)

        popen_kwargs: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": self._stderr_fh,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        try:
            launched = self._launch(cmd, popen_kwargs, startup_timeout)
        except RuntimeError as exc:
            # Process EXITED during startup (not a timeout). Retry with the
            # conservative args only when the failure blames argument parsing
            # (an older build rejecting a probed flag) — a load failure such
            # as CUDA OOM would fail identically, so surface it at once.
            if not has_optional_features or not _looks_like_arg_error(str(exc)):
                self.shutdown()
                raise
            logger.warning(
                "llama.cpp exited during startup with optional feature args; "
                "retrying with conservative args. Detail: %s",
                exc,
            )
            try:
                launched = self._launch(conservative_cmd, popen_kwargs, startup_timeout)
            except BaseException:
                self.shutdown()
                raise
        except BaseException:
            # BaseException, not Exception: the child runs in its own session
            # and never sees the terminal's Ctrl-C, so a KeyboardInterrupt out
            # of the health wait must still shut it down (mirrors VllmLlmServer).
            self.shutdown()
            raise
        self._n_slots = _n_slots_from_argv(launched)
        # Server is healthy — nudge Vulkan-on-NVIDIA users toward CUDA (log-only,
        # once per process, both roles). Never blocks or fails startup.
        _warn_cuda_steering_once(prefix)

    def _launch(self, cmd: list[str], popen_kwargs: dict, startup_timeout: int) -> list[str]:
        """Spawn the server and block until healthy; return the launched argv."""
        logger.info("Starting llama.cpp server: %s", " ".join(cmd))
        logger.info("llama.cpp stderr -> %s", self._stderr_log)
        self._process = subprocess.Popen(cmd, **popen_kwargs)  # noqa: S603
        self._wait_until_healthy(startup_timeout)
        return cmd

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def n_slots(self) -> int:
        """Number of server slots in the final launched argv (1 or 2)."""
        return self._n_slots

    @property
    def loaded(self) -> bool:
        return self._reused or (self._process is not None and self._process.poll() is None)

    def _wait_until_healthy(self, timeout: int) -> None:
        assert self._process is not None  # noqa: S101
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                rc = self._process.returncode
                tail = self._read_stderr_tail()
                raise RuntimeError(f"llama.cpp exited during startup (code {rc}): {tail[-1000:]}")
            try:
                status, _reason, _body = request_bytes(f"{self.base_url}/health", timeout=5)
                if status == 200:
                    # A sibling bibr process may have won a port race: our own
                    # bind failed and its server answered this probe. Require
                    # our process to still be alive after a short grace period;
                    # an exit here is a startup failure, not readiness.
                    time.sleep(_HEALTH_GRACE_S)
                    if self._process.poll() is not None:
                        rc = self._process.returncode
                        tail = self._read_stderr_tail()
                        raise RuntimeError(
                            f"llama.cpp exited during startup (code {rc}): {tail[-1000:]}"
                        )
                    # Flush log so offload lines written before the health OK
                    # are visible to parse_gpu_offload.
                    if self._stderr_fh is not None:
                        try:
                            self._stderr_fh.flush()
                        except OSError:
                            pass
                    log_offload_status(self._read_stderr_tail(n=12_000))
                    logger.info("llama.cpp ready (model=%s, port=%d)", self._model, self._port)
                    return
            except (LocalHttpError, json.JSONDecodeError):
                pass
            time.sleep(2)
        raise TimeoutError(f"llama.cpp did not become ready within {timeout}s")

    def _read_stderr_tail(self, n: int = 4000) -> str:
        if self._stderr_log is None:
            return ""
        try:
            return self._stderr_log.read_bytes()[-n:].decode("utf-8", errors="replace")
        except OSError:
            return ""

    def shutdown(self) -> None:
        proc, self._process = self._process, None
        try:
            if proc is not None and proc.poll() is None:
                if os.name == "nt":
                    proc.terminate()
                else:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        proc.terminate()
                try:
                    proc.wait(timeout=_TERM_GRACE_S)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            proc.kill()
                    proc.wait(timeout=5)
        finally:
            if self._stderr_fh is not None:
                self._stderr_fh.close()
                self._stderr_fh = None


class LlamaCppLlmServer:
    """Managed NuExtract GGUF server for Windows and small CUDA GPUs."""

    def __init__(self, settings=None) -> None:
        from bibr.config import snapshot_settings
        from bibr.local.llm_models import default_local_model

        self._settings = settings if settings is not None else snapshot_settings()
        self._server = LlamaCppServer(
            model=self._settings.llm.local_model or default_local_model("llama-cpp"),
            port=self._settings.llm.llama_cpp_port,
            context_size=self._settings.llm.llama_cpp_context_size,
            startup_timeout=self._settings.llm.llama_cpp_startup_timeout,
            extra_args=self._settings.llm.llama_cpp_extra_args,
            role="llm",
        )

    @property
    def base_url(self) -> str:
        return self._server.base_url

    def configure_llm_client(self) -> None:
        self._settings.llm.provider = "openai"
        self._settings.llm.base_url = self.base_url + "/v1"
        self._settings.llm.api_key = "not-needed"
        self._settings.llm.model = self._server.model
        # Concurrency follows the server's slot count (1, or 2 when the probed
        # multi-slot args are active), unless the user pinned it explicitly.
        if "max_concurrency" not in self._settings.llm.model_fields_set:
            self._settings.llm.max_concurrency = self._server.n_slots
        # Keep prompt + completion inside the launched context size. The caps
        # below equal the historical constants at the default 16384-token
        # context (completion ≤ ctx/4); a larger LLM_LLAMA_CPP_CONTEXT_SIZE
        # scales them instead of silently keeping the small-context budget.
        # Explicit user values are never replaced — only warned about when
        # they cannot fit the context.
        ctx = self._settings.llm.llama_cpp_context_size
        caps = (
            ("max_tokens", ctx // 4),
            ("max_input_chars", 24_000 * ctx // 16_384),
            ("ref_seg_window_chars", 12_000 * ctx // 16_384),
        )
        for field, cap in caps:
            current = getattr(self._settings.llm, field)
            if field in self._settings.llm.model_fields_set:
                if current > cap:
                    logger.warning(
                        "Local llama.cpp backend — explicit llm.%s=%s exceeds the %d-token "
                        "context budget (~%s); requests past the context will fail or be cut. "
                        "Raise LLM_LLAMA_CPP_CONTEXT_SIZE or lower the value.",
                        field,
                        current,
                        ctx,
                        cap,
                    )
                continue
            if current > cap:
                setattr(self._settings.llm, field, cap)
                logger.info(
                    "Local llama.cpp backend — capping llm.%s to %d for the %d-token context",
                    field,
                    cap,
                    ctx,
                )
        if "rate_limit_rpm" not in self._settings.llm.model_fields_set:
            # Our own server has no external quota to protect; the cloud 60
            # rpm default would only add dead time between serialized calls.
            self._settings.llm.rate_limit_rpm = MANAGED_LOCAL_LLM_RATE_LIMIT_RPM
            logger.info(
                "Local llama.cpp backend — raising llm.rate_limit_rpm to %d",
                MANAGED_LOCAL_LLM_RATE_LIMIT_RPM,
            )
        if "timeout_seconds" not in self._settings.llm.model_fields_set:
            self._settings.llm.timeout_seconds = _LLAMA_CPP_DEFAULT_TIMEOUT_SECONDS
            logger.info(
                "Local llama.cpp backend — raising llm.timeout_seconds 30s->%ds",
                self._settings.llm.timeout_seconds,
            )

    def shutdown(self) -> None:
        self._server.shutdown()
