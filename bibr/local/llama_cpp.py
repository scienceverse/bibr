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
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from bibr.exceptions import UpstreamServiceError
from bibr.local.http_runtime import (
    MANAGED_LOCAL_LLM_RATE_LIMIT_RPM,
    LocalHttpError,
    bearer_headers,
    check_startup_stop,
    env_with_api_key,
    guard_managed_server_port,
    kill_process_group,
    new_server_api_key,
    own_server_key,
    pause_startup_poll,
    register_server_key,
    request_bytes,
    unregister_server_key,
)

if TYPE_CHECKING:
    from bibr.local.llm_models import PinnedGguf

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
# On a CPU the slowest call can take several minutes: the one measured CPU run
# (bibr's 4B student at Q8_0, 4 threads) read 82 and wrote 9.6 tokens/s, and
# two server slots share the cores. A timeout would discard the whole call.
_LLAMA_CPP_CPU_TIMEOUT_SECONDS = 900
_LLAMA_CPP_RELEASES_URL = "https://github.com/ggml-org/llama.cpp/releases"

# Flags applied for every managed server regardless of role. llama.cpp is
# bibr's low-memory path (Windows + ≤8 GB CUDA), so flash-attn + quantized KV
# are on by default. User ``extra_args`` can override any of these
# (see ``_merge_cli_args``). Role- and probe-specific flags are added by
# ``_role_runtime_args``; a build that sees no GPU drops the layer offload
# (``_GPU_OFFLOAD_ARGS``) and gets a thread count instead.
_GPU_OFFLOAD_ARGS: tuple[str, ...] = ("--n-gpu-layers", "999")
_COMMON_RUNTIME_ARGS: tuple[str, ...] = (
    *_GPU_OFFLOAD_ARGS,
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
    "--n-gpu-layers": frozenset({"--n-gpu-layers", "-ngl", "--gpu-layers"}),
    "--threads": frozenset({"--threads", "-t"}),
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
    "--model-url": frozenset({"--model-url", "-mu"}),
    "--model": frozenset({"--model", "-m"}),
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
# ``--list-devices`` of a build that sees no GPU (lowercased): the header
# followed by the "(none)" entry current builds print, or by nothing at all.
_EMPTY_DEVICE_TABLE_RE = re.compile(r"available devices:\s*(?:\(none\)|\Z)")
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


def cpu_install_hint(*, platform_name: str | None = None) -> str:
    """Install guidance for a machine with no GPU, where llama.cpp runs on the CPU."""
    plat = platform_name if platform_name is not None else sys.platform
    if plat == "win32" or plat.startswith("win"):
        # Not ``winget install llama.cpp``: winget ships the Vulkan build,
        # which offloads to an integrated GPU and so skips the CPU defaults.
        return (
            f"Extract llama-<ver>-bin-win-cpu-x64.zip (the CPU build) from "
            f"{_LLAMA_CPP_RELEASES_URL} and put the folder on PATH"
        )
    if plat == "darwin":
        return (
            "Install with: brew install llama.cpp, or extract llama-<ver>-bin-macos-x64.tar.gz "
            f"from {_LLAMA_CPP_RELEASES_URL} and put `llama-server` on PATH"
        )
    return (
        f"Extract llama-<ver>-bin-ubuntu-x64.tar.gz (the CPU build) from {_LLAMA_CPP_RELEASES_URL} "
        "and put `llama-server` on PATH, or install with Homebrew: brew install llama.cpp"
    )


def missing_binary_message(*, platform_name: str | None = None, cpu: bool = False) -> str:
    """Error text when no llama-server executable is on PATH.

    ``cpu`` gives the install guidance for a machine with no GPU.
    """
    hint = cpu_install_hint if cpu else install_hint
    return f"llama.cpp is not installed. {hint(platform_name=platform_name)}."


def _cpu_list(text: str) -> set[int]:
    """CPU numbers of a sysfs cpulist such as ``0-3,8-11``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def _linux_physical_cores(
    cpus: Iterable[int], *, sysfs: Path = Path("/sys/devices/system/cpu")
) -> int | None:
    """Distinct (package, core) pairs among *cpus*, or None when sysfs cannot tell."""
    cores: set[tuple[str, str]] = set()
    for cpu in cpus:
        topology = sysfs / f"cpu{cpu}" / "topology"
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            return None
        cores.add((package, core))
    return len(cores) or None


def physical_core_count(
    *,
    sysfs: Path = Path("/sys/devices/system/cpu"),
    performance_cores: Path = Path("/sys/devices/cpu_core/cpus"),
) -> int | None:
    """Physical cores this process may run on, or None when that is unknown.

    The thread count for llama.cpp on a CPU. llama.cpp's own default counts
    every physical core of the machine, so a run confined to fewer CPUs (a
    container, ``taskset``, a cgroup cpuset) would oversubscribe them. On
    hybrid Intel CPUs only the performance cores count, as in llama.cpp:
    threads that wait on the slower efficiency cores slow every step. None
    on Windows and wherever the topology cannot be read; llama.cpp then
    picks the count itself.
    """
    if sys.platform.startswith("linux"):
        try:
            allowed = set(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            return None
        try:
            performance = _cpu_list(performance_cores.read_text()) & allowed
        except (OSError, ValueError):
            performance = set()
        return _linux_physical_cores(sorted(performance or allowed), sysfs=sysfs)
    if sys.platform == "darwin":
        for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            try:
                result = subprocess.run(  # noqa: S603
                    ["sysctl", "-n", key],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return int(result.stdout.strip()) or None
    return None


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


def _role_runtime_args(
    role: str,
    available: frozenset[str],
    *,
    cpu_only: bool = False,
    cpu_threads: int | None = None,
) -> list[str]:
    """Runtime flags for *role*, gating optional features on *available*.

    The startup retry strips every optional flag added here when the stderr
    tail matches an argument-parse error (see ``_ARG_ERROR_MARKERS``), so a
    build whose ``--help`` lists a flag it does not actually accept still
    recovers to a safe launch. Anything else — a crash, an OOM, any message
    outside those markers — fails fast with no retry.

    *cpu_only* is a build that sees no GPU (:func:`runs_on_cpu`): it gets no
    layer offload and, when *cpu_threads* is known, that many threads.
    """
    if role not in ("ocr", "llm"):
        raise ValueError(f"unknown llama.cpp server role {role!r} (expected 'ocr' or 'llm')")
    args = list(_COMMON_RUNTIME_ARGS)
    if cpu_only:
        args = args[len(_GPU_OFFLOAD_ARGS) :]
        if cpu_threads:
            args += ["--threads", str(cpu_threads)]
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
        # GGUF repos of vision-language models (Qwen3.5, NuExtract 3) can ship
        # an mmproj that ``-hf`` auto-loads; the LLM path is text-only, so
        # skip it.
        args += ["--no-mmproj"]
    if "--jinja" in available:
        # Chat-template kwargs (thinking off, NuExtract's extraction template)
        # only apply when Jinja is enabled.
        args += ["--jinja"]
        if "--chat-template-kwargs" in available:
            # bibr's Qwen3.5 students were trained and scored with thinking
            # off; templates without the switch ignore it.
            args += ["--chat-template-kwargs", '{"enable_thinking": false}']
    # With the default ``--reasoning auto``, llama-server passes
    # enable_thinking=true itself to any template that supports thinking,
    # whatever the template's own default; turn reasoning off outright.
    if "--reasoning" in available:
        args += ["--reasoning", "off"]
    elif "--reasoning-budget" in available:
        args += ["--reasoning-budget", "0"]
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
# The model flags are identity too: the server answers to the registry name
# (``--alias``) whatever file it loads, and a user ``-m`` next to bibr's own
# would become llama.cpp's download target, overwriting that file.
_MODEL_OPTIONS = frozenset({"-hf", "--model-url", "--model"})
_IDENTITY_OPTIONS = frozenset({"--port", "--alias"}) | _MODEL_OPTIONS


def _check_no_identity_override(extra: list[str], *, role: str) -> None:
    """Reject extra args that would move the server bibr thinks it started."""
    if role == "llm":
        extra_setting = "LLM_LLAMA_CPP_EXTRA_ARGS"
        port_setting = "LLM_LLAMA_CPP_PORT"
        model_setting = "LLM_LOCAL_MODEL"
    else:
        extra_setting = "OCR_LLAMA_CPP_EXTRA_ARGS"
        port_setting = "OCR_LLAMA_CPP_PORT"
        model_setting = "OCR_LLAMA_CPP_MODEL"
    for token in extra:
        bare = token.split("=", 1)[0]
        canonical = _canonical_option(bare)
        if canonical in _MODEL_OPTIONS:
            raise UpstreamServiceError(
                "local inference",
                f"{extra_setting} must not set {bare}: bibr chooses the model the managed "
                f"server loads and asks for it by name. Set {model_setting} to another "
                "Hugging Face GGUF (repo:quant) instead.",
            )
        if canonical in _IDENTITY_OPTIONS:
            raise UpstreamServiceError(
                "local inference",
                f"{extra_setting} must not set {bare}: the managed server's port, "
                f"alias and model are fixed, and bibr polls the configured port — it would "
                f"time out and kill a healthy server that moved. Set {port_setting} to move "
                "the server instead.",
            )


def _validated_extra_args(extra_args: str, *, role: str) -> list[str]:
    """*extra_args* split into tokens, refusing any that move the server."""
    setting = "LLM_LLAMA_CPP_EXTRA_ARGS" if role == "llm" else "OCR_LLAMA_CPP_EXTRA_ARGS"
    extra = split_extra_args(extra_args, setting=setting)
    _check_no_identity_override(extra, role=role)
    return extra


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
    model_path: str | None = None,
    cpu_only: bool = False,
    cpu_threads: int | None = None,
) -> list[str]:
    """Assemble the full ``llama-server`` argv including role-based optim defaults.

    *available* is the set of ``--long-option`` flags the installed server
    supports; ``None`` probes it via ``supported_flags(prefix)``. Pass an
    explicit set (e.g. ``frozenset()``) to keep tests subprocess-free.
    *model_path* loads that local GGUF file (``-m``, see
    :func:`download_pinned_gguf`) instead of ``-hf model``; the server still
    answers to *model*. Every launch names its model with one of the two:
    without either, llama-server starts as a router that loads nothing.
    *cpu_only* and *cpu_threads*: see :func:`_role_runtime_args`.
    """
    if available is None:
        available = supported_flags(prefix)
    extra = _validated_extra_args(extra_args, role=role)
    source = ["-m", model_path] if model_path else ["-hf", model]
    fixed = [
        *prefix,
        *source,
        "--alias",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        *_role_runtime_args(role, available, cpu_only=cpu_only, cpu_threads=cpu_threads),
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
        # An empty device table. A header followed by a real device table is
        # not one: only "(none)" or the end of the output may follow it.
        or _EMPTY_DEVICE_TABLE_RE.search(lower) is not None
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
    marker appears wins. An empty device table with no backend named (what a
    current CPU-only build prints) is ``"cpu"``.
    """
    if not text or not text.strip():
        return None
    lower = text.lower()
    for kind, markers in _BACKEND_KIND_MARKERS:
        if any(marker in lower for marker in markers):
            return kind
    if _EMPTY_DEVICE_TABLE_RE.search(lower) is not None:
        return "cpu"
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


def runs_on_cpu(prefix: list[str] | None = None) -> bool:
    """Whether the installed llama.cpp build sees no GPU, so models run on the CPU.

    From the cached :func:`probe_backend_kind`; False when no build is
    installed or the probe cannot tell, which keeps the GPU launch flags.
    """
    try:
        return probe_backend_kind(prefix) == "cpu"
    except Exception:  # noqa: BLE001 - a probe must never fail a launch
        return False


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


def download_pinned_gguf(
    gguf: PinnedGguf,
    *,
    timeout: float,
    stop_event: threading.Event | None = None,
    timeout_setting: str = "LLM_LLAMA_CPP_STARTUP_TIMEOUT",
) -> str:
    """Local path of a pinned GGUF file, fetched into the Hugging Face cache.

    llama-server is then started with ``-m <path>``. It cannot fetch the file
    itself from a pinned URL: given only ``--model-url``, it starts as a
    router that loads no model. ``huggingface_hub`` keeps the file in the
    cache ``bibr chew --dry-run`` reads and honours ``HF_ENDPOINT``,
    ``HF_TOKEN`` and offline mode. A cached file returns at once. Otherwise
    the download runs in a worker thread, so a stop request or *timeout*
    ends the wait; the worker carries on in the background, and the next
    run resumes a partial download.
    """
    from bibr.utils.hf_cache import hf_hub_download_no_symlink

    request = {"repo_id": gguf.repo_id, "filename": gguf.filename, "revision": gguf.revision}
    try:
        return hf_hub_download_no_symlink(**request, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - not in the cache yet: download it below
        logger.debug("%s is not in the Hugging Face cache: %s", gguf.filename, exc)
    logger.info(
        "Downloading %s from %s (commit %s)", gguf.filename, gguf.repo_id, gguf.revision[:12]
    )
    outcome: dict[str, object] = {}

    def fetch() -> None:
        try:
            outcome["path"] = hf_hub_download_no_symlink(**request)
        except BaseException as exc:  # noqa: BLE001 - handed to the waiting thread
            outcome["error"] = exc

    worker = threading.Thread(target=fetch, name="bibr-gguf-download", daemon=True)
    worker.start()
    deadline = time.monotonic() + timeout
    while worker.is_alive():
        check_startup_stop(stop_event, "llama.cpp")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Downloading {gguf.filename} from {gguf.repo_id} did not finish within "
                f"{timeout:.0f}s. Raise {timeout_setting}, or fetch it first with: hf download "
                f"{gguf.repo_id} {gguf.filename} --revision {gguf.revision}"
            )
        worker.join(timeout=0.5)
    error = outcome.get("error")
    if isinstance(error, BaseException):
        raise UpstreamServiceError(
            "local inference",
            f"Could not download {gguf.filename} from {gguf.repo_id} at commit "
            f"{gguf.revision[:12]}: {error}",
        ) from error
    return str(outcome["path"])


def _served_model_ids(body: bytes) -> set[str] | None:
    """Model ids and aliases in a ``/v1/models`` body, or None when it is not that list."""
    try:
        data = json.loads(body.decode("utf-8")).get("data")
    except (ValueError, AttributeError):
        return None
    if not isinstance(data, list):
        return None
    served: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        if "id" in item:
            served.add(str(item["id"]))
        aliases = item.get("aliases")
        if isinstance(aliases, list):
            served.update(str(alias) for alias in aliases)
    return served


class LlamaCppServer:
    """Own a llama.cpp OpenAI-compatible server subprocess.

    llama.cpp publishes native Windows CUDA binaries and supports both GGUFs
    used by bibr's low-memory path. A fresh process per phase guarantees that
    the previous model and CUDA context are returned to the OS.
    """

    # Set by the owner to stop the startup wait early (see ResourceManager).
    _stop_event: threading.Event | None = None
    # Per-launch key of the server this process started, which a sibling
    # pipeline reusing it shares; another process's listener has none.
    api_key = ""
    # Whether the installed build sees no GPU (see ``cpu_only``).
    _cpu_only = False

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
        stop_event: threading.Event | None = None,
        gguf: PinnedGguf | None = None,
        cpu_only: bool | None = None,
    ) -> None:
        """*gguf*: load that file at its pinned commit rather than ``-hf model``,
        which follows the repo's main branch; a download
        (:func:`download_pinned_gguf`) gets *startup_timeout* of its own.
        *cpu_only*: whether the build runs on the CPU; ``None`` probes it
        (:func:`runs_on_cpu`)."""
        self._stop_event = stop_event
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
            # Another pipeline in this process may have started it.
            self.api_key = own_server_key(self.base_url)
            return

        prefix = find_llama_server()
        if prefix is None:
            raise UpstreamServiceError(
                "local inference",
                missing_binary_message(),
            )

        available = supported_flags(prefix)
        self._cpu_only = runs_on_cpu(prefix) if cpu_only is None else cpu_only
        cpu_threads = physical_core_count() if self._cpu_only else None
        if self._cpu_only:
            logger.info(
                "llama.cpp sees no GPU: %s runs on the CPU (%s threads)",
                model,
                cpu_threads or "llama.cpp's default",
            )
        # Check the extra args before a download and before opening the log
        # handle: shlex errors and identity-flag overrides must fail at once,
        # without leaking it.
        _validated_extra_args(extra_args, role=role)
        model_path = None
        if gguf is not None:
            model_path = download_pinned_gguf(
                gguf,
                timeout=startup_timeout,
                stop_event=stop_event,
                timeout_setting=(
                    "LLM_LLAMA_CPP_STARTUP_TIMEOUT"
                    if role == "llm"
                    else "OCR_LLAMA_CPP_STARTUP_TIMEOUT"
                ),
            )
        launch_args = {
            "model": model,
            "port": port,
            "context_size": context_size,
            "extra_args": extra_args,
            "role": role,
            "model_path": model_path,
            "cpu_only": self._cpu_only,
            "cpu_threads": cpu_threads,
        }
        cmd = build_server_argv(prefix, available=available, **launch_args)
        # Conservative launch = role defaults minus every probed optional
        # feature flag; used as the single startup-retry fallback.
        conservative_cmd = build_server_argv(prefix, available=frozenset(), **launch_args)
        has_optional_features = cmd != conservative_cmd

        from bibr.utils.secure_temp import open_subprocess_log

        self._stderr_log, self._stderr_fh = open_subprocess_log("llama", port)

        # /health stays public, so the readiness poll needs no key.
        self.api_key = new_server_api_key()
        popen_kwargs: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": self._stderr_fh,
            "env": env_with_api_key("LLAMA_API_KEY", self.api_key),
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
        register_server_key(self.base_url, self.api_key)
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
    def cpu_only(self) -> bool:
        """Whether the server was launched for a build that sees no GPU."""
        return self._cpu_only

    @property
    def loaded(self) -> bool:
        return self._reused or (self._process is not None and self._process.poll() is None)

    def _wait_until_healthy(self, timeout: int) -> None:
        assert self._process is not None  # noqa: S101
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            check_startup_stop(self._stop_event, "llama.cpp")
            if self._process.poll() is not None:
                rc = self._process.returncode
                tail = self._read_stderr_tail()
                kill_process_group(self._process.pid)
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
                        kill_process_group(self._process.pid)
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
                    self._check_serves_model()
                    if not self._cpu_only:
                        # A CPU build offloads nothing by design.
                        log_offload_status(self._read_stderr_tail(n=12_000))
                    logger.info("llama.cpp ready (model=%s, port=%d)", self._model, self._port)
                    return
            except (LocalHttpError, json.JSONDecodeError):
                pass
            pause_startup_poll(2, self._stop_event, "llama.cpp")
        raise TimeoutError(f"llama.cpp did not become ready within {timeout}s")

    def _check_serves_model(self) -> None:
        """Fail a healthy server whose ``/v1/models`` lists models but not ours.

        llama-server given no model starts as a router: ``/health`` answers 200
        at once while every request for the model fails. A ``/v1/models`` that
        cannot be read or parsed proves nothing and is let through.
        """
        assert self._process is not None  # noqa: S101
        try:
            status, _reason, body = request_bytes(
                f"{self.base_url}/v1/models", headers=bearer_headers(self.api_key), timeout=5
            )
        except LocalHttpError:
            return
        served = _served_model_ids(body) if status == 200 else None
        if served is None or self._model in served:
            return
        kill_process_group(self._process.pid)
        listed = ", ".join(sorted(served)) or "none"
        raise RuntimeError(
            f"llama.cpp started but does not serve {self._model!r} (its /v1/models lists: "
            f"{listed}), so every request would fail. llama-server loads no model when "
            "none is named; check the model arguments in its command line above."
        )

    def _read_stderr_tail(self, n: int = 4000) -> str:
        if self._stderr_log is None:
            return ""
        try:
            return self._stderr_log.read_bytes()[-n:].decode("utf-8", errors="replace")
        except OSError:
            return ""

    def shutdown(self) -> None:
        proc, self._process = self._process, None
        if proc is not None and self.api_key:
            unregister_server_key(self.base_url, self.api_key)
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
    """Managed GGUF extraction-model server for Windows, small CUDA GPUs and CPUs."""

    # Whether the installed llama.cpp build sees no GPU; set per launch.
    _cpu_only = False

    def __init__(self, settings=None, *, stop_event: threading.Event | None = None) -> None:
        from bibr.config import snapshot_settings
        from bibr.local.llm_models import default_local_model, pinned_gguf

        self._settings = settings if settings is not None else snapshot_settings()
        self._cpu_only = runs_on_cpu()
        model = self._settings.llm.local_model or default_local_model(
            "llama-cpp", cpu=self._cpu_only
        )
        self._server = LlamaCppServer(
            model=model,
            port=self._settings.llm.llama_cpp_port,
            context_size=self._settings.llm.llama_cpp_context_size,
            startup_timeout=self._settings.llm.llama_cpp_startup_timeout,
            extra_args=self._settings.llm.llama_cpp_extra_args,
            role="llm",
            stop_event=stop_event,
            gguf=pinned_gguf(model),
            cpu_only=self._cpu_only,
        )

    @property
    def base_url(self) -> str:
        return self._server.base_url

    def configure_llm_client(self) -> None:
        self._settings.llm.provider = "openai"
        self._settings.llm.base_url = self.base_url + "/v1"
        self._settings.llm.api_key = self._server.api_key or "not-needed"
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
            self._settings.llm.timeout_seconds = (
                _LLAMA_CPP_CPU_TIMEOUT_SECONDS
                if self._cpu_only
                else _LLAMA_CPP_DEFAULT_TIMEOUT_SECONDS
            )
            logger.info(
                "Local llama.cpp backend — raising llm.timeout_seconds 30s->%ds",
                self._settings.llm.timeout_seconds,
            )
        if self._cpu_only and "EQUATION_LLM_FALLBACK" not in self._settings.model_fields_set:
            # Most equation LLM calls find nothing, and on a CPU each one
            # holds a server slot the front-matter calls are waiting for. The
            # regex equations need no LLM and stay.
            self._settings.EQUATION_LLM_FALLBACK = False
            logger.info(
                "Local llama.cpp backend on the CPU — equation LLM fallback off, regex "
                "equations kept (set EQUATION_LLM_FALLBACK=true to keep it)"
            )

    def shutdown(self) -> None:
        self._server.shutdown()
