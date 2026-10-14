"""Cross-platform process plumbing for the managed llama.cpp runtime."""

import contextlib
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Imported before the conftest stubs replace them.
from bibr.local.llama_cpp import download_pinned_gguf as _real_download_pinned_gguf
from bibr.local.llama_cpp import runs_on_cpu as _real_runs_on_cpu


@contextlib.contextmanager
def _unset_field(settings_section, field_name):
    was_explicit = field_name in settings_section.model_fields_set
    settings_section.model_fields_set.discard(field_name)
    try:
        yield
    finally:
        if was_explicit:
            settings_section.model_fields_set.add(field_name)


@contextlib.contextmanager
def _set_field_explicit(settings_section, field_name):
    was_explicit = field_name in settings_section.model_fields_set
    settings_section.model_fields_set.add(field_name)
    try:
        yield
    finally:
        if not was_explicit:
            settings_section.model_fields_set.discard(field_name)


def test_find_llama_server_supports_unified_cli(monkeypatch):
    from bibr.local import llama_cpp

    monkeypatch.setattr(
        llama_cpp.shutil,
        "which",
        lambda name, **_kwargs: "C:\\llama.exe" if name == "llama" else None,
    )
    assert llama_cpp.find_llama_server() == ["C:\\llama.exe", "serve"]


def test_find_llama_server_uses_updated_windows_user_path(monkeypatch):
    """winget updates User PATH before the current PowerShell sees it."""
    from bibr.local import llama_cpp

    user_bin = "C:\\Users\\dlakens\\AppData\\Local\\Microsoft\\WindowsApps"

    class FakeKey:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeWinreg:
        HKEY_CURRENT_USER = object()
        HKEY_LOCAL_MACHINE = object()
        KEY_READ = 0
        KEY_WOW64_64KEY = 0

        def OpenKey(self, root, subkey, reserved=0, access=0):
            if root is self.HKEY_CURRENT_USER and subkey == "Environment":
                return FakeKey(user_bin)
            if root is self.HKEY_LOCAL_MACHINE:
                return FakeKey("")
            raise OSError

        def QueryValueEx(self, key, name):
            if name == "Path":
                return key.value, 1
            raise OSError

    def fake_which(name, path=None):
        if name == "llama-server" and path and user_bin in path:
            return f"{user_bin}\\llama-server.exe"
        return None

    monkeypatch.setattr(llama_cpp.os, "name", "nt", raising=False)
    monkeypatch.setenv("PATH", "C:\\Python312")
    monkeypatch.setattr(llama_cpp, "winreg", FakeWinreg(), raising=False)
    monkeypatch.setattr(llama_cpp.shutil, "which", fake_which)

    assert llama_cpp.find_llama_server() == [f"{user_bin}\\llama-server.exe"]


def test_install_hint_is_platform_specific():
    from bibr.local.llama_cpp import install_hint, missing_binary_message

    assert "winget" in install_hint(platform_name="win32")
    assert "GGML_CUDA" in install_hint(platform_name="linux")
    assert "brew" in install_hint(platform_name="darwin")
    assert "winget" in missing_binary_message(platform_name="win32")
    assert "releases" in missing_binary_message(platform_name="linux")


def test_common_runtime_args_include_flash_attn_and_kv_quant():
    from bibr.local.llama_cpp import _role_runtime_args

    args = _role_runtime_args("llm", frozenset())
    assert args[args.index("--flash-attn") + 1] == "on"
    assert args[args.index("--cache-type-k") + 1] == "q8_0"
    assert args[args.index("--cache-type-v") + 1] == "q8_0"
    assert args[args.index("--n-gpu-layers") + 1] == "999"
    # No kv-unified support -> single slot (never --parallel 2 without kv-unified).
    assert args[args.index("--parallel") + 1] == "1"
    assert "--kv-unified" not in args
    assert "--spec-type" not in args
    assert "--no-mmproj" not in args
    assert "--no-context-shift" not in args


def test_role_runtime_args_ocr_is_always_single_slot():
    from bibr.local.llama_cpp import _role_runtime_args

    # Even with every optional flag advertised, OCR stays single-slot and adds
    # none of the LLM-only speculation / mmproj / context-shift flags.
    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj", "--no-context-shift"})
    args = _role_runtime_args("ocr", full)
    assert args[args.index("--parallel") + 1] == "1"
    assert "--kv-unified" not in args
    assert "--spec-type" not in args
    assert "--no-mmproj" not in args
    assert "--no-context-shift" not in args


def test_role_runtime_args_llm_full_features():
    from bibr.local.llama_cpp import _role_runtime_args

    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj", "--no-context-shift"})
    args = _role_runtime_args("llm", full)
    assert args[args.index("--parallel") + 1] == "2"
    assert "--kv-unified" in args
    assert args[args.index("--spec-type") + 1] == "ngram-mod"
    assert "--no-mmproj" in args
    assert "--no-context-shift" in args


def test_role_runtime_args_llm_omits_no_context_shift_when_unsupported():
    """The probed server doesn't advertise ``--no-context-shift`` -> omitted,
    same conservatism as every other optional flag."""
    from bibr.local.llama_cpp import _role_runtime_args

    args = _role_runtime_args("llm", frozenset({"--kv-unified", "--spec-type", "--no-mmproj"}))
    assert "--no-context-shift" not in args


def test_user_extra_args_suppress_no_context_shift():
    """A user who wants context shift back on can override it via extra_args."""
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/model:Q4",
        port=8770,
        context_size=16384,
        extra_args="--context-shift",
        role="llm",
        available=frozenset({"--kv-unified", "--no-context-shift"}),
    )
    assert "--no-context-shift" not in cmd
    assert "--context-shift" in cmd


def test_build_server_argv_includes_optim_defaults():
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/model:Q4",
        port=8770,
        context_size=4096,
        available=frozenset(),
    )
    assert cmd[:4] == ["llama-server", "-hf", "org/model:Q4", "--alias"]
    assert "--flash-attn" in cmd
    assert "q8_0" in cmd
    assert cmd[cmd.index("--ctx-size") + 1] == "4096"
    assert cmd[cmd.index("--port") + 1] == "8770"


def test_build_server_argv_llm_role_adds_probed_features():
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/model:Q4",
        port=8770,
        context_size=16384,
        role="llm",
        available=frozenset({"--kv-unified", "--spec-type", "--no-mmproj"}),
    )
    assert cmd[cmd.index("--parallel") + 1] == "2"
    assert "--kv-unified" in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "ngram-mod"
    assert "--no-mmproj" in cmd


def test_build_server_argv_llm_role_enables_jinja_when_supported():
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/model:Q4",
        port=8770,
        context_size=16384,
        role="llm",
        available=frozenset({"--jinja"}),
    )

    assert "--jinja" in cmd
    assert "--chat-template-kwargs" not in cmd  # only when the build supports it


def test_build_server_argv_llm_role_turns_thinking_off_when_supported():
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/model:Q4",
        port=8770,
        context_size=16384,
        role="llm",
        available=frozenset({"--jinja", "--chat-template-kwargs"}),
    )

    assert cmd[cmd.index("--chat-template-kwargs") + 1] == '{"enable_thinking": false}'


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (frozenset({"--reasoning", "--reasoning-budget"}), ["--reasoning", "off"]),
        (frozenset({"--reasoning-budget"}), ["--reasoning-budget", "0"]),
    ],
)
def test_build_server_argv_llm_role_turns_reasoning_off(available, expected):
    """llama-server's ``--reasoning auto`` forces thinking on, whatever the template says."""
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/model:Q4",
        port=8770,
        context_size=16384,
        role="llm",
        available=available,
    )

    i = cmd.index(expected[0])
    assert cmd[i : i + 2] == expected


def test_build_server_argv_ocr_role_leaves_reasoning_alone():
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/ocr:Q8",
        port=8771,
        context_size=8192,
        role="ocr",
        available=frozenset({"--reasoning", "--reasoning-budget"}),
    )

    assert "--reasoning" not in cmd
    assert "--reasoning-budget" not in cmd


def test_build_server_argv_user_chat_template_kwargs_replace_the_default():
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/model:Q4",
        port=8770,
        context_size=16384,
        extra_args="--chat-template-kwargs '{\"enable_thinking\": true}'",
        role="llm",
        available=frozenset({"--jinja", "--chat-template-kwargs"}),
    )

    assert cmd.count("--chat-template-kwargs") == 1
    assert cmd[cmd.index("--chat-template-kwargs") + 1] == '{"enable_thinking": true}'


def test_build_server_argv_ocr_role_single_slot():
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/ocr:Q8",
        port=8771,
        context_size=8192,
        role="ocr",
        available=frozenset({"--kv-unified", "--spec-type", "--no-mmproj"}),
    )
    assert cmd[cmd.index("--parallel") + 1] == "1"
    assert "--kv-unified" not in cmd
    assert "--no-mmproj" not in cmd


def test_build_server_argv_default_role_is_ocr():
    """Callers not passing a role keep the pre-rework semantics (conservative)."""
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/ocr:Q8",
        port=8771,
        context_size=8192,
        available=frozenset({"--kv-unified", "--spec-type", "--no-mmproj"}),
    )
    assert cmd[cmd.index("--parallel") + 1] == "1"
    assert "--kv-unified" not in cmd
    assert "--spec-type" not in cmd
    assert "--no-mmproj" not in cmd


def test_invalid_role_raises_value_error():
    from bibr.local.llama_cpp import _role_runtime_args, build_server_argv

    with pytest.raises(ValueError, match="role"):
        _role_runtime_args("bogus", frozenset())
    with pytest.raises(ValueError, match="role"):
        build_server_argv(
            ["llama-server"],
            model="org/model:Q4",
            port=8770,
            context_size=8192,
            role="bogus",
            available=frozenset(),
        )


def test_user_extra_args_override_defaults():
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/model:Q4",
        port=8770,
        context_size=8192,
        extra_args="-ctk f16 -ctv f16 -fa off -ngl 20",
        available=frozenset(),
    )
    # Defaults for overridden flags must be gone (not just last-wins clutter).
    assert "--cache-type-k" not in cmd
    assert "--cache-type-v" not in cmd
    assert "--flash-attn" not in cmd
    assert "--n-gpu-layers" not in cmd
    assert cmd[cmd.index("-ctk") + 1] == "f16"
    assert cmd[cmd.index("-ctv") + 1] == "f16"
    assert cmd[cmd.index("-fa") + 1] == "off"
    assert cmd[cmd.index("-ngl") + 1] == "20"
    # Untouched default still present.
    assert cmd[cmd.index("--parallel") + 1] == "1"


def test_user_extra_args_suppress_probed_feature_flags():
    from bibr.local.llama_cpp import build_server_argv

    cmd = build_server_argv(
        ["llama-server"],
        model="org/model:Q4",
        port=8770,
        context_size=16384,
        # User pins a single slot and disables speculation; both defaults drop.
        extra_args="-np 1 --spec-type none",
        role="llm",
        available=frozenset({"--kv-unified", "--spec-type", "--no-mmproj"}),
    )
    # --parallel default (2) suppressed by user -np; only user's value remains.
    assert "--parallel" not in cmd
    assert cmd[cmd.index("-np") + 1] == "1"
    # spec-type default suppressed; user's value wins (single occurrence).
    assert cmd.count("--spec-type") == 1
    assert cmd[cmd.index("--spec-type") + 1] == "none"


def test_parse_gpu_offload_reads_last_report():
    from bibr.local.llama_cpp import parse_gpu_offload

    text = (
        "load_tensors: offloaded 10/32 layers to GPU\n"
        "llama_model_load: offloaded 32/32 layers to GPU\n"
    )
    assert parse_gpu_offload(text) == (32, 32)
    assert parse_gpu_offload("no offload here") is None


def test_log_offload_status_warns_on_zero_layers(caplog):
    from bibr.local.llama_cpp import log_offload_status

    with caplog.at_level(logging.WARNING):
        log_offload_status("load_tensors: offloaded 0/28 layers to GPU")
    assert any("0/28" in r.message for r in caplog.records)


def test_probe_gpu_backend_detects_cuda(monkeypatch):
    from bibr.local import llama_cpp

    def fake_run(cmd, **_kwargs):
        out = MagicMock()
        out.stdout = "Available devices:\n  CUDA0: NVIDIA GeForce GTX 1060\n"
        out.stderr = ""
        return out

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    assert llama_cpp.probe_gpu_backend(["llama-server"]) is True


def test_probe_gpu_backend_detects_cpu_only(monkeypatch):
    from bibr.local import llama_cpp

    def fake_run(cmd, **_kwargs):
        out = MagicMock()
        out.stdout = "Available devices:\nCPU\n"
        out.stderr = ""
        return out

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    assert llama_cpp.probe_gpu_backend(["llama-server"]) is False


@pytest.mark.parametrize(
    "diagnostic",
    [
        "ggml_cuda_init: failed: no devices found",
        "CUDA error: no CUDA-capable device is detected",
    ],
)
def test_probe_gpu_backend_prefers_explicit_cuda_failure(monkeypatch, diagnostic):
    from bibr.local import llama_cpp

    def fake_run(cmd, **_kwargs):
        out = MagicMock()
        out.stdout = ""
        out.stderr = diagnostic
        return out

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)

    assert llama_cpp.probe_gpu_backend(["llama-server"]) is False


def test_probe_gpu_backend_missing_binary_returns_none(monkeypatch):
    from bibr.local import llama_cpp

    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: None)
    assert llama_cpp.probe_gpu_backend(None) is None
    assert llama_cpp.probe_gpu_backend([]) is None


_CUDA_DEVICE_TABLE = (
    "ggml_cuda_init: found 1 CUDA devices:\n"
    "  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes\n"
    "Available devices:\n"
    "  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 24000 MiB free)\n"
    "  CPU\n"
)
# The Vulkan build lists NVIDIA cards by their device NAME — a naive name match
# (nvidia/geforce) would misread this as a CUDA build.
_VULKAN_ON_NVIDIA_DEVICE_TABLE = (
    "ggml_vulkan: Found 1 Vulkan devices:\n"
    "ggml_vulkan: 0 = NVIDIA GeForce RTX 4090 (NVIDIA) | ...\n"
    "Available devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 24000 MiB free)\n"
    "  CPU\n"
)
_CPU_ONLY_DEVICE_TABLE = "Available devices:\n  CPU\n"


def _stub_probe_output(monkeypatch, *, stdout="", stderr=""):
    from bibr.local import llama_cpp

    def fake_run(cmd, **_kwargs):
        out = MagicMock()
        out.stdout = stdout
        out.stderr = stderr
        return out

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)


def test_probe_backend_kind_classifies_cuda(monkeypatch):
    from bibr.local import llama_cpp

    llama_cpp._BACKEND_KIND_CACHE.clear()
    _stub_probe_output(monkeypatch, stdout=_CUDA_DEVICE_TABLE)
    assert llama_cpp.probe_backend_kind(["llama-server"]) == "cuda"


def test_probe_backend_kind_classifies_vulkan_on_nvidia(monkeypatch):
    """The Vulkan build on an NVIDIA card must classify as vulkan, not cuda."""
    from bibr.local import llama_cpp

    llama_cpp._BACKEND_KIND_CACHE.clear()
    _stub_probe_output(monkeypatch, stdout=_VULKAN_ON_NVIDIA_DEVICE_TABLE)
    assert llama_cpp.probe_backend_kind(["llama-server"]) == "vulkan"


def test_probe_backend_kind_classifies_cpu_only(monkeypatch):
    from bibr.local import llama_cpp

    llama_cpp._BACKEND_KIND_CACHE.clear()
    _stub_probe_output(monkeypatch, stdout=_CPU_ONLY_DEVICE_TABLE)
    assert llama_cpp.probe_backend_kind(["llama-server"]) == "cpu"


@pytest.mark.parametrize("payload", ["", "   \n  ", "totally unrelated banner text"])
def test_probe_backend_kind_garbage_returns_none(monkeypatch, payload):
    from bibr.local import llama_cpp

    llama_cpp._BACKEND_KIND_CACHE.clear()
    _stub_probe_output(monkeypatch, stdout=payload)
    assert llama_cpp.probe_backend_kind(["llama-server"]) is None


def test_probe_backend_kind_missing_binary_returns_none(monkeypatch):
    from bibr.local import llama_cpp

    llama_cpp._BACKEND_KIND_CACHE.clear()
    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: None)
    assert llama_cpp.probe_backend_kind(None) is None
    assert llama_cpp.probe_backend_kind([]) is None


def test_probe_backend_kind_falls_back_to_version(monkeypatch):
    """A binary that has no --list-devices still classifies via --version."""
    from bibr.local import llama_cpp

    llama_cpp._BACKEND_KIND_CACHE.clear()

    def fake_run(cmd, **_kwargs):
        out = MagicMock()
        if cmd[-1] == "--list-devices":
            out.stdout = ""
            out.stderr = "error: unknown argument: --list-devices"
        else:  # --version
            out.stdout = ""
            out.stderr = "version: 1234 (abcdef)\nbuilt with ... for x86_64 (Vulkan)\n"
        return out

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    assert llama_cpp.probe_backend_kind(["llama-server"]) == "vulkan"


def test_probe_backend_kind_caches_per_prefix(monkeypatch):
    """Repeated calls with the same prefix must not re-exec the subprocess."""
    from bibr.local import llama_cpp

    llama_cpp._BACKEND_KIND_CACHE.clear()
    calls = {"n": 0}

    def fake_run(cmd, **_kwargs):
        calls["n"] += 1
        out = MagicMock()
        out.stdout = _CUDA_DEVICE_TABLE
        out.stderr = ""
        return out

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    first = llama_cpp.probe_backend_kind(["llama-server"])
    second = llama_cpp.probe_backend_kind(["llama-server"])
    assert first == second == "cuda"
    assert calls["n"] == 1  # cached, binary exec'd once


def test_cuda_steering_hint_vulkan_with_nvidia(monkeypatch):
    from bibr.local import llama_cpp

    monkeypatch.setattr("bibr.local.llm_models._nvidia_vram_gb", lambda: 24.0)
    hint = llama_cpp.cuda_steering_hint("vulkan")
    assert hint is not None
    assert "Vulkan build" in hint
    assert "cuda-12.4" in hint
    assert "cudart-llama-bin-win-cuda-12.4-x64.zip" in hint
    assert "llama-" in hint and "bin-win-cuda-12.4-x64.zip" in hint
    assert "github.com/ggml-org/llama.cpp/releases" in hint
    assert "GTX 10" in hint  # newer cards may use cuda-13.x


def test_cuda_steering_hint_vulkan_without_nvidia(monkeypatch):
    from bibr.local import llama_cpp

    monkeypatch.setattr("bibr.local.llm_models._nvidia_vram_gb", lambda: None)
    assert llama_cpp.cuda_steering_hint("vulkan") is None


def test_cuda_steering_hint_cuda_build_returns_none(monkeypatch):
    from bibr.local import llama_cpp

    # Even with an NVIDIA GPU, a CUDA build needs no steering.
    monkeypatch.setattr("bibr.local.llm_models._nvidia_vram_gb", lambda: 24.0)
    assert llama_cpp.cuda_steering_hint("cuda") is None


@pytest.mark.parametrize("kind", [None, "cpu", "metal"])
def test_cuda_steering_hint_non_vulkan_returns_none(monkeypatch, kind):
    from bibr.local import llama_cpp

    monkeypatch.setattr("bibr.local.llm_models._nvidia_vram_gb", lambda: 24.0)
    assert llama_cpp.cuda_steering_hint(kind) is None


def test_server_startup_warns_cuda_steering_once(monkeypatch, caplog):
    from bibr.local import llama_cpp

    monkeypatch.setattr(llama_cpp, "_cuda_steering_warned", False, raising=False)
    monkeypatch.setattr("bibr.local.llm_models._nvidia_vram_gb", lambda: 24.0)
    _healthy_stub_server(
        monkeypatch,
        available=frozenset(),
        wait_side_effect=lambda self, t: None,
        backend_kind="vulkan",
    )

    with caplog.at_level(logging.WARNING):
        s1 = llama_cpp.LlamaCppServer(model="org/ocr:Q8", port=8771, role="ocr")
        s2 = llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")

    steering = [r for r in caplog.records if "Vulkan build" in r.getMessage()]
    assert len(steering) == 1  # fires once across both server roles
    for srv in (s1, s2):
        srv._process = None
        srv.shutdown()


def test_server_startup_no_steering_warning_for_cuda_build(monkeypatch, caplog):
    from bibr.local import llama_cpp

    monkeypatch.setattr(llama_cpp, "_cuda_steering_warned", False, raising=False)
    monkeypatch.setattr("bibr.local.llm_models._nvidia_vram_gb", lambda: 24.0)
    _healthy_stub_server(
        monkeypatch,
        available=frozenset(),
        wait_side_effect=lambda self, t: None,
        backend_kind="cuda",
    )

    with caplog.at_level(logging.WARNING):
        srv = llama_cpp.LlamaCppServer(model="org/ocr:Q8", port=8771, role="ocr")

    assert not any("Vulkan build" in r.getMessage() for r in caplog.records)
    srv._process = None
    srv.shutdown()


def test_server_startup_steering_probe_never_breaks_startup(monkeypatch):
    """A crashing steering probe must not fail server construction."""
    from bibr.local import llama_cpp

    monkeypatch.setattr(llama_cpp, "_cuda_steering_warned", False, raising=False)

    def boom(*_a, **_k):
        raise RuntimeError("probe blew up")

    _healthy_stub_server(
        monkeypatch,
        available=frozenset(),
        wait_side_effect=lambda self, t: None,
    )
    monkeypatch.setattr(llama_cpp, "probe_backend_kind", boom)

    srv = llama_cpp.LlamaCppServer(model="org/ocr:Q8", port=8771, role="ocr")
    assert srv is not None
    srv._process = None
    srv.shutdown()


def test_repeated_server_construction_probes_backend_kind_once(monkeypatch):
    """``probe_backend_kind`` must not re-exec the binary on every
    ``LlamaCppServer`` construction: the CUDA-steering warn-once flag only
    latches when a Vulkan+NVIDIA hint actually fires, so without its own
    cache a non-Vulkan build re-probes every time. Exercise the real
    (un-stubbed) ``probe_backend_kind`` across two constructions and count
    the underlying subprocess calls."""
    from bibr.local import llama_cpp

    llama_cpp._BACKEND_KIND_CACHE.clear()
    llama_cpp._HELP_CACHE.clear()
    monkeypatch.setattr(llama_cpp, "_cuda_steering_warned", False, raising=False)

    probe_calls = {"n": 0}

    def fake_run(cmd, **_kwargs):
        out = MagicMock()
        if cmd[-1] == "--help":
            out.stdout = ""
            out.stderr = ""
        else:
            probe_calls["n"] += 1
            out.stdout = _CPU_ONLY_DEVICE_TABLE
            out.stderr = ""
        return out

    def fake_popen(cmd, **_kwargs):
        proc = MagicMock()
        proc.poll.return_value = 0
        return proc

    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: ["llama-server"])
    monkeypatch.setattr(llama_cpp, "os", types.SimpleNamespace(name="posix"))
    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_cpp.LlamaCppServer, "_wait_until_healthy", lambda self, t: None)

    s1 = llama_cpp.LlamaCppServer(model="org/ocr:Q8", port=8771, role="ocr")
    s2 = llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")

    assert probe_calls["n"] == 1  # cached after the first construction
    for srv in (s1, s2):
        srv._process = None
        srv.shutdown()


def test_server_uses_windows_process_group(monkeypatch):
    from bibr.local import llama_cpp

    proc = MagicMock()
    proc.poll.return_value = None
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: ["llama.exe", "serve"])
    monkeypatch.setattr(llama_cpp, "supported_flags", lambda prefix: frozenset())
    monkeypatch.setattr(llama_cpp, "probe_backend_kind", lambda *a, **k: "cpu")
    monkeypatch.setattr(llama_cpp, "os", types.SimpleNamespace(name="nt"))
    monkeypatch.setattr(llama_cpp.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_cpp.LlamaCppServer, "_wait_until_healthy", lambda self, timeout: None)

    server = llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770)
    assert captured["kwargs"]["creationflags"] == 512
    assert "start_new_session" not in captured["kwargs"]
    assert captured["cmd"][:4] == ["llama.exe", "serve", "-hf", "org/model:Q4"]
    assert "--flash-attn" in captured["cmd"]
    assert "q8_0" in captured["cmd"]
    server._process = None
    server.shutdown()


def test_server_missing_binary_uses_platform_hint(monkeypatch):
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import llama_cpp

    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: None)
    try:
        llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770)
        raise AssertionError("expected UpstreamServiceError")
    except UpstreamServiceError as exc:
        assert "llama.cpp" in str(exc).lower()


def test_llm_server_configures_serial_openai_client(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llama_cpp

    fake = MagicMock(base_url="http://127.0.0.1:8770", model="org/model:Q4", n_slots=1)
    monkeypatch.setattr(llama_cpp, "LlamaCppServer", MagicMock(return_value=fake))
    monkeypatch.setattr(Settings.llm, "local_model", "org/model:Q4")
    server = llama_cpp.LlamaCppLlmServer()
    with _unset_field(Settings.llm, "max_concurrency"):
        server.configure_llm_client()
        assert server._settings.llm.max_concurrency == 1
    assert server._settings.llm.provider == "openai"
    assert server._settings.llm.base_url == "http://127.0.0.1:8770/v1"
    assert server._settings.llm.model == "org/model:Q4"
    assert server._settings.llm.max_tokens <= 4096
    assert server._settings.llm.max_input_chars <= 24_000


def test_llm_server_defaults_to_the_gguf_variant_when_local_model_is_unset(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llama_cpp

    fake = MagicMock(base_url="http://127.0.0.1:8770", model="x", n_slots=1)
    ctor = MagicMock(return_value=fake)
    monkeypatch.setattr(llama_cpp, "LlamaCppServer", ctor)
    monkeypatch.setattr(Settings.llm, "local_model", None)
    llama_cpp.LlamaCppLlmServer()
    assert ctor.call_args.kwargs["model"] == "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"


def test_llm_server_concurrency_follows_slot_count(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llama_cpp

    fake = MagicMock(base_url="http://127.0.0.1:8770", model="org/model:Q4", n_slots=2)
    monkeypatch.setattr(llama_cpp, "LlamaCppServer", MagicMock(return_value=fake))
    monkeypatch.setattr(Settings.llm, "local_model", "org/model:Q4")
    with _unset_field(Settings.llm, "max_concurrency"):
        server = llama_cpp.LlamaCppLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.max_concurrency == 2


def test_llm_server_respects_explicit_max_concurrency(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llama_cpp

    fake = MagicMock(base_url="http://127.0.0.1:8770", model="org/model:Q4", n_slots=2)
    monkeypatch.setattr(llama_cpp, "LlamaCppServer", MagicMock(return_value=fake))
    monkeypatch.setattr(Settings.llm, "local_model", "org/model:Q4")
    monkeypatch.setattr(Settings.llm, "max_concurrency", 1)
    server = llama_cpp.LlamaCppLlmServer()
    with _set_field_explicit(Settings.llm, "max_concurrency"):
        server.configure_llm_client()
        assert server._settings.llm.max_concurrency == 1


def test_llm_server_raises_default_timeout_for_slow_local_generation(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llama_cpp

    fake = MagicMock(base_url="http://127.0.0.1:8770", model="org/model:Q4", n_slots=1)
    monkeypatch.setattr(llama_cpp, "LlamaCppServer", MagicMock(return_value=fake))
    monkeypatch.setattr(Settings.llm, "local_model", "org/model:Q4")
    monkeypatch.setattr(Settings.llm, "timeout_seconds", 30)

    with _unset_field(Settings.llm, "timeout_seconds"):
        server = llama_cpp.LlamaCppLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.timeout_seconds == 300


def test_llm_server_respects_explicit_timeout(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llama_cpp

    fake = MagicMock(base_url="http://127.0.0.1:8770", model="org/model:Q4", n_slots=1)
    monkeypatch.setattr(llama_cpp, "LlamaCppServer", MagicMock(return_value=fake))
    monkeypatch.setattr(Settings.llm, "local_model", "org/model:Q4")
    monkeypatch.setattr(Settings.llm, "timeout_seconds", 120)

    with _set_field_explicit(Settings.llm, "timeout_seconds"):
        server = llama_cpp.LlamaCppLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.timeout_seconds == 120


_FAKE_HELP = """
usage: llama-server [options]
  -c, --ctx-size N
  -np, --parallel N
      --kv-unified
      --spec-type TYPE
      --no-mmproj
"""


def test_supported_flags_parses_long_options(monkeypatch):
    from bibr.local import llama_cpp

    llama_cpp._HELP_CACHE.clear()

    def fake_run(cmd, **_kwargs):
        assert cmd[-1] == "--help"
        out = MagicMock()
        out.stdout = _FAKE_HELP
        out.stderr = ""
        return out

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    flags = llama_cpp.supported_flags(["llama-server"])
    assert "--kv-unified" in flags
    assert "--spec-type" in flags
    assert "--no-mmproj" in flags
    assert "--parallel" in flags


def test_supported_flags_empty_on_probe_failure(monkeypatch):
    from bibr.local import llama_cpp

    llama_cpp._HELP_CACHE.clear()

    def boom(cmd, **_kwargs):
        raise OSError("no such binary")

    monkeypatch.setattr(llama_cpp.subprocess, "run", boom)
    # Empty help -> conservative: no optional flags advertised.
    assert llama_cpp.supported_flags(["missing-server"]) == frozenset()


def test_probe_server_help_caches_per_prefix(monkeypatch):
    from bibr.local import llama_cpp

    llama_cpp._HELP_CACHE.clear()
    calls = {"n": 0}

    def fake_run(cmd, **_kwargs):
        calls["n"] += 1
        out = MagicMock()
        out.stdout = _FAKE_HELP
        out.stderr = ""
        return out

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    first = llama_cpp.probe_server_help(["llama-server"])
    second = llama_cpp.probe_server_help(["llama-server"])
    assert first == second
    assert calls["n"] == 1  # cached, binary exec'd once


def _healthy_stub_server(monkeypatch, *, available, wait_side_effect, backend_kind="cpu"):
    """Wire LlamaCppServer so it launches without a real subprocess/probe."""
    from bibr.local import llama_cpp

    launched: list[list[str]] = []

    def fake_popen(cmd, **_kwargs):
        launched.append(list(cmd))
        proc = MagicMock()
        # Report "already exited" so shutdown()'s kill path is a no-op; the
        # health check is stubbed via _wait_until_healthy, not poll().
        proc.poll.return_value = 0
        return proc

    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: ["llama-server"])
    monkeypatch.setattr(llama_cpp, "supported_flags", lambda prefix: available)
    # Keep the startup steering probe subprocess-free (default: CPU-ish build,
    # so no Vulkan-on-NVIDIA warning fires unless a test opts into it).
    monkeypatch.setattr(llama_cpp, "probe_backend_kind", lambda *a, **k: backend_kind)
    monkeypatch.setattr(llama_cpp, "os", types.SimpleNamespace(name="posix"))
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        llama_cpp.LlamaCppServer,
        "_wait_until_healthy",
        wait_side_effect,
    )
    return launched


def test_startup_retry_falls_back_to_conservative_args(monkeypatch, caplog):
    from bibr.local import llama_cpp

    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj", "--no-context-shift"})
    attempts = {"n": 0}

    def wait(self, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError(
                "llama.cpp exited during startup (code 1): error: unknown argument '--spec-type'"
            )
        return None

    launched = _healthy_stub_server(monkeypatch, available=full, wait_side_effect=wait)

    with caplog.at_level(logging.WARNING):
        server = llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")

    assert attempts["n"] == 2  # first fails, retry succeeds
    assert len(launched) == 2
    # First attempt carried the optional feature flags...
    assert "--kv-unified" in launched[0]
    assert "--spec-type" in launched[0]
    assert "--no-context-shift" in launched[0]
    # ...the retry stripped every one of them.
    assert "--kv-unified" not in launched[1]
    assert "--spec-type" not in launched[1]
    assert "--no-mmproj" not in launched[1]
    assert "--no-context-shift" not in launched[1]
    assert launched[1][launched[1].index("--parallel") + 1] == "1"
    assert server.n_slots == 1
    assert any("conservative" in r.message.lower() for r in caplog.records)
    server._process = None
    server.shutdown()


def test_startup_retry_on_value_rejection(monkeypatch, caplog):
    """A rejected flag VALUE also retries: llama.cpp wraps handler errors as
    'error while handling argument "<flag>": <reason>' (common/arg.cpp), e.g.
    --spec-type ngram-mod on a build from before ngram-mod existed."""
    from bibr.local import llama_cpp

    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj", "--no-context-shift"})
    attempts = {"n": 0}

    def wait(self, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError(
                "llama.cpp exited during startup (code 1): "
                'error while handling argument "--spec-type": '
                "unknown speculative type: ngram-mod\n\nusage:\n"
                "--spec-type [none|ngram-cache|ngram-simple]\n"
            )
        return None

    launched = _healthy_stub_server(monkeypatch, available=full, wait_side_effect=wait)

    with caplog.at_level(logging.WARNING):
        server = llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")

    assert attempts["n"] == 2  # first fails, retry succeeds
    assert len(launched) == 2
    assert "--spec-type" in launched[0]
    assert "--spec-type" not in launched[1]
    assert any("conservative" in r.message.lower() for r in caplog.records)
    server._process = None
    server.shutdown()


def test_startup_no_retry_on_timeout(monkeypatch):
    from bibr.local import llama_cpp

    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj"})

    def wait(self, timeout):
        raise TimeoutError("llama.cpp did not become ready within 600s")

    launched = _healthy_stub_server(monkeypatch, available=full, wait_side_effect=wait)

    with pytest.raises(TimeoutError):
        llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")
    # Timeout must not trigger the conservative retry.
    assert len(launched) == 1


def test_startup_no_retry_without_optional_args(monkeypatch):
    from bibr.local import llama_cpp

    def wait(self, timeout):
        raise RuntimeError("llama.cpp exited during startup (code 1): boom")

    # Empty feature set -> no optional args added -> nothing to fall back to.
    launched = _healthy_stub_server(monkeypatch, available=frozenset(), wait_side_effect=wait)

    with pytest.raises(RuntimeError):
        llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")
    assert len(launched) == 1


def test_n_slots_two_when_multi_slot_active(monkeypatch):
    from bibr.local import llama_cpp

    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj"})
    _healthy_stub_server(monkeypatch, available=full, wait_side_effect=lambda self, t: None)

    server = llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")
    assert server.n_slots == 2
    server._process = None
    server.shutdown()


def test_n_slots_one_for_ocr_role(monkeypatch):
    from bibr.local import llama_cpp

    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj"})
    _healthy_stub_server(monkeypatch, available=full, wait_side_effect=lambda self, t: None)

    server = llama_cpp.LlamaCppServer(model="org/ocr:Q8", port=8771, role="ocr")
    assert server.n_slots == 1
    server._process = None
    server.shutdown()


def test_server_default_role_is_ocr(monkeypatch):
    """No-role construction (e.g. LlamaCppOcrClient) keeps pre-rework behavior."""
    from bibr.local import llama_cpp

    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj"})
    launched = _healthy_stub_server(
        monkeypatch, available=full, wait_side_effect=lambda self, t: None
    )

    server = llama_cpp.LlamaCppServer(model="org/ocr:Q8", port=8771)
    argv = launched[0]
    assert argv[argv.index("--parallel") + 1] == "1"
    assert "--kv-unified" not in argv
    assert "--spec-type" not in argv
    assert "--no-mmproj" not in argv
    assert server.n_slots == 1
    server._process = None
    server.shutdown()


def test_llm_server_launches_with_llm_role(monkeypatch):
    """LlamaCppLlmServer must opt into the llm-role feature args explicitly."""
    from bibr.config import Settings
    from bibr.local import llama_cpp

    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj"})
    launched = _healthy_stub_server(
        monkeypatch, available=full, wait_side_effect=lambda self, t: None
    )
    monkeypatch.setattr(Settings.llm, "local_model", "org/model:Q4")

    server = llama_cpp.LlamaCppLlmServer()
    argv = launched[0]
    assert argv[argv.index("--parallel") + 1] == "2"
    assert "--kv-unified" in argv
    assert argv[argv.index("--spec-type") + 1] == "ngram-mod"
    assert "--no-mmproj" in argv
    assert server._server.n_slots == 2
    server._server._process = None
    server.shutdown()


class _FakeLlamaServer:
    """Stand-in for ``LlamaCppServer`` — no subprocess, no HTTP."""

    base_url = "http://localhost:8771"
    api_key = ""

    @property
    def loaded(self):
        return True


def _stub_ocr_client_deps(
    monkeypatch, tmp_path, server_kwargs: dict | None = None, http_kwargs=None
):
    """Patch ``LlamaCppServer`` and ``HttpOcrClient`` so ``LlamaCppOcrClient()``
    never touches a subprocess or the network, capturing each call's kwargs.

    Also returns a fresh ``GlobalSettings()`` built from the current env (call
    ``monkeypatch.setenv`` / ``delenv`` *before* this helper). Passing it to the
    client reproduces the real runtime
    precondition — ``compute_ocr_concurrency`` has already auto-tuned the
    concurrency fields, poisoning ``model_fields_set`` — while keeping the
    client's clamp mutations off the process-wide singleton.
    """
    from bibr.config import GlobalSettings
    from bibr.local import llama_cpp
    from bibr.local import ocr as ocr_mod

    def fake_server(**kwargs):
        if server_kwargs is not None:
            server_kwargs.update(kwargs)
        return _FakeLlamaServer()

    class FakeHttpClient:
        def __init__(self, **kwargs):
            if http_kwargs is not None:
                http_kwargs.update(kwargs)

    monkeypatch.setattr(llama_cpp, "LlamaCppServer", fake_server)
    monkeypatch.setattr(ocr_mod, "HttpOcrClient", FakeHttpClient)
    # GlobalSettings() reads OCR_* via the CWD-then-home dotenv fallback chain
    # (bibr.config._default_env_files); monkeypatch.delenv/setenv cannot
    # neutralize a dotenv-sourced value, so a dev machine with OCR_* in .env
    # would fail this construction. Point both dotenv sources at empty dirs,
    # matching TestEnvFileFallbackChain's hermetic pattern in test_config.py.
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    fresh_settings = GlobalSettings()
    return ocr_mod, fresh_settings


def test_ocr_client_passes_ocr_role_explicitly(monkeypatch, tmp_path):
    """LlamaCppOcrClient must opt into role="ocr" at the call site (Task 1 interface)."""
    server_kwargs: dict = {}
    ocr_mod, _settings = _stub_ocr_client_deps(monkeypatch, tmp_path, server_kwargs=server_kwargs)

    ocr_mod.LlamaCppOcrClient(settings=_settings)

    assert server_kwargs["role"] == "ocr"


def test_ocr_client_caps_max_tokens_to_half_context(monkeypatch, tmp_path):
    """Today the client passes no max_tokens, asking for 16384 against an 8192
    context. Cap it to half the configured context (mirrors VllmMlxOcrClient)."""
    http_kwargs: dict = {}
    ocr_mod, settings = _stub_ocr_client_deps(monkeypatch, tmp_path, http_kwargs=http_kwargs)
    settings.ocr.llama_cpp_context_size = 2048

    ocr_mod.LlamaCppOcrClient(settings=settings)

    assert http_kwargs["max_tokens"] == 1024


def test_ocr_client_caps_max_tokens_at_4096_ceiling(monkeypatch, tmp_path):
    """Half of a large context must still not exceed the 4096 ceiling."""
    http_kwargs: dict = {}
    ocr_mod, settings = _stub_ocr_client_deps(monkeypatch, tmp_path, http_kwargs=http_kwargs)
    settings.ocr.llama_cpp_context_size = 16384

    ocr_mod.LlamaCppOcrClient(settings=settings)

    assert http_kwargs["max_tokens"] == 4096


def test_ocr_client_clamps_concurrency_to_single_slot_at_runtime(monkeypatch, tmp_path, caplog):
    """The managed llama.cpp OCR server serves exactly one slot: with neither
    concurrency env var set, the client must clamp both knobs to 1 — under the
    REAL runtime precondition where ``compute_ocr_concurrency`` has already
    assigned both fields (pydantic v2 adds assigned fields to
    ``model_fields_set``, so membership alone cannot gate the clamp)."""
    monkeypatch.delenv("OCR_MAX_CONCURRENT_REGIONS", raising=False)
    monkeypatch.delenv("OCR_CONCURRENT_REGIONS_PER_FILE", raising=False)
    ocr_mod, settings = _stub_ocr_client_deps(monkeypatch, tmp_path)

    # Real-world poisoning: the auto-tune validator's own assignment puts the
    # field in model_fields_set even though the user never set it.
    assert "max_concurrent_regions" in settings.ocr.model_fields_set
    assert settings.ocr.user_set_concurrency == frozenset()
    # Platform-independent precondition: Apple Silicon now auto-tunes both knobs
    # to 1 (rapid-mlx prefill serializes), while the Linux path yields 16*gpus.
    # Force both >1 WITHOUT marking them user-set (the private snapshot was taken
    # empty at construction), so the clamp has real work to prove everywhere.
    settings.ocr.max_concurrent_regions = 8
    settings.ocr.concurrent_regions_per_file = 8

    with caplog.at_level(logging.INFO, logger="bibr.local.ocr"):
        ocr_mod.LlamaCppOcrClient(settings=settings)
        assert settings.ocr.concurrent_regions_per_file == 1
        assert settings.ocr.max_concurrent_regions == 1
        # Reconstruction is idempotent and must not re-log the clamp.
        ocr_mod.LlamaCppOcrClient(settings=settings)

    info_records = [r for r in caplog.records if r.name == "bibr.local.ocr"]
    assert len(info_records) == 1


def test_ocr_client_respects_env_explicit_concurrency(monkeypatch, tmp_path):
    """Explicit user env settings win over the single-slot clamp; the field the
    user did NOT set is still clamped."""
    monkeypatch.setenv("OCR_MAX_CONCURRENT_REGIONS", "4")
    monkeypatch.delenv("OCR_CONCURRENT_REGIONS_PER_FILE", raising=False)
    ocr_mod, settings = _stub_ocr_client_deps(monkeypatch, tmp_path)

    ocr_mod.LlamaCppOcrClient(settings=settings)

    assert settings.ocr.max_concurrent_regions == 4
    assert settings.ocr.concurrent_regions_per_file == 1


# ---------------------------------------------------------------------------
# local-runtimes sweep: explicit client tuning, quotes, stderr tail, retry,
# identity flags, Ctrl-C cleanup, health-race grace (findings 4, 6, 7, 10, 12, 14, 15)
# ---------------------------------------------------------------------------


def _llm_server_with_isolated_settings(**env):
    """LlamaCppLlmServer with a fresh GlobalSettings (no global mutation)."""
    import os

    from bibr.config import GlobalSettings
    from bibr.local import llama_cpp

    old = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update({key: str(value) for key, value in env.items()})
        settings = GlobalSettings()
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    server = object.__new__(llama_cpp.LlamaCppLlmServer)
    server._settings = settings
    server._server = MagicMock(base_url="http://127.0.0.1:8770", model="m", n_slots=2)
    return server


def test_llm_server_respects_explicit_token_caps():
    """Explicit LLM_MAX_TOKENS / LLM_MAX_INPUT_CHARS survive configure (10)."""
    server = _llm_server_with_isolated_settings(
        LLM_LLAMA_CPP_CONTEXT_SIZE=65536,
        LLM_MAX_TOKENS=16384,
        LLM_MAX_INPUT_CHARS=100000,
    )
    assert "max_tokens" in server._settings.llm.model_fields_set
    server.configure_llm_client()
    assert server._settings.llm.max_tokens == 16384
    assert server._settings.llm.max_input_chars == 100000


def test_llm_server_caps_scale_with_context_size_when_unset(caplog):
    """Unset caps derive from the launched context size, not fixed constants (10)."""
    server = _llm_server_with_isolated_settings(LLM_LLAMA_CPP_CONTEXT_SIZE=65536)
    for field in ("max_tokens", "max_input_chars", "ref_seg_window_chars"):
        server._settings.llm.model_fields_set.discard(field)
    with caplog.at_level(logging.INFO, logger="bibr.local.llama_cpp"):
        server.configure_llm_client()
    assert server._settings.llm.max_tokens == 16384  # 65536 // 4
    assert server._settings.llm.max_input_chars == 96_000
    # Default 16_000 is already under the 48_000 budget — left alone.
    assert server._settings.llm.ref_seg_window_chars == 16_000


def test_llm_server_default_context_keeps_historical_caps():
    """At the default 16384-token context the derived caps equal the old constants."""
    server = _llm_server_with_isolated_settings()
    server._settings.llm.model_fields_set.discard("max_tokens")
    server._settings.llm.model_fields_set.discard("max_input_chars")
    server._settings.llm.model_fields_set.discard("ref_seg_window_chars")
    server.configure_llm_client()
    assert server._settings.llm.max_tokens == 4096
    assert server._settings.llm.max_input_chars == 24_000
    assert server._settings.llm.ref_seg_window_chars == 12_000


def test_llm_server_warns_on_explicit_value_over_context(caplog):
    """An explicit value that cannot fit the context logs a warning (10)."""
    server = _llm_server_with_isolated_settings(
        LLM_LLAMA_CPP_CONTEXT_SIZE=8192, LLM_MAX_TOKENS=16384
    )
    with caplog.at_level(logging.WARNING, logger="bibr.local.llama_cpp"):
        server.configure_llm_client()
    assert server._settings.llm.max_tokens == 16384  # kept, not replaced
    assert any("max_tokens" in r.message for r in caplog.records)


def test_llm_server_raises_rate_limit_rpm_when_unset():
    """The managed server auto-raises the cloud 60 rpm cap like vllm/vllm-mlx (12)."""
    server = _llm_server_with_isolated_settings()
    server._settings.llm.model_fields_set.discard("rate_limit_rpm")
    server.configure_llm_client()
    assert server._settings.llm.rate_limit_rpm == 600


def test_llm_server_respects_explicit_rate_limit_rpm():
    """An explicit LLM_RATE_LIMIT_RPM suppresses the managed auto-raise (12)."""
    server = _llm_server_with_isolated_settings(LLM_RATE_LIMIT_RPM=30)
    server.configure_llm_client()
    assert server._settings.llm.rate_limit_rpm == 30


def test_split_extra_args_strips_windows_quotes(monkeypatch):
    """Windows-branch tokens lose one surrounding quote pair (6)."""
    import types as _types

    from bibr.local import llama_cpp

    monkeypatch.setattr(llama_cpp, "os", _types.SimpleNamespace(name="nt"))
    parts = llama_cpp.split_extra_args(
        '--chat-template-file "C:\\Users\\Jane Doe\\nx.jinja" -ctk f16'
    )
    assert parts == ["--chat-template-file", "C:\\Users\\Jane Doe\\nx.jinja", "-ctk", "f16"]
    # Tokens without surrounding quotes, and partial quoting, are untouched.
    assert llama_cpp.split_extra_args("--port 8770")[1] == "8770"
    assert llama_cpp.split_extra_args('--flag ab"cd')[1] == 'ab"cd'


def test_split_extra_args_posix_branch_keeps_quotes_stripped(monkeypatch):
    """POSIX shlex already strips quotes — no double processing (6 guard)."""
    import types as _types

    from bibr.local import llama_cpp

    monkeypatch.setattr(llama_cpp, "os", _types.SimpleNamespace(name="posix"))
    parts = llama_cpp.split_extra_args('--chat-template-file "/tmp/my model/t.jinja"')
    assert parts == ["--chat-template-file", "/tmp/my model/t.jinja"]


def test_startup_error_reports_tail_end(monkeypatch, tmp_path):
    """The exit error keeps the LAST 1000 chars (the actual error line), not the first (7)."""
    from bibr.local import llama_cpp

    log = tmp_path / "llama.log"
    log.write_bytes(
        b"\n".join(
            f"load_tensors: layer {i:3d} assigned to device CUDA0".encode() for i in range(120)
        )
        + b"\nllama_model_load: error loading model: cudaMalloc failed: out of memory\n"
    )

    proc = MagicMock()
    proc.poll.return_value = 1
    proc.returncode = 1
    server = object.__new__(llama_cpp.LlamaCppServer)
    server._process = proc
    server._port = 8770
    server._model = "m"
    server._stderr_fh = None
    server._stderr_log = log
    server._reused = False

    with pytest.raises(RuntimeError) as excinfo:
        llama_cpp.LlamaCppServer._wait_until_healthy(server, timeout=30)
    assert "out of memory" in str(excinfo.value)


def test_startup_no_retry_on_load_failure(monkeypatch):
    """An OOM exit with optional feature args must NOT trigger the conservative retry (7)."""
    from bibr.local import llama_cpp

    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj"})

    def wait(self, timeout):
        raise RuntimeError(
            "llama.cpp exited during startup (code 1): "
            "llama_model_load: error loading model: cudaMalloc failed: out of memory"
        )

    launched = _healthy_stub_server(monkeypatch, available=full, wait_side_effect=wait)

    with pytest.raises(RuntimeError, match="out of memory"):
        llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")
    assert len(launched) == 1


@pytest.mark.parametrize(
    "flag",
    [
        "--port 9000",
        "--alias other",
        "-hf other/repo",
        "-hfr other/repo",
        "-mu https://example.org/other.gguf",
        "--model-url=https://example.org/other.gguf",
        "-m /home/u/models/my-finetune.gguf",
        "--model=/home/u/models/my-finetune.gguf",
    ],
)
def test_build_server_argv_rejects_identity_override(flag):
    """Extra args must not move the server bibr polls — point at the port setting (14)."""
    from bibr.exceptions import UpstreamServiceError
    from bibr.local.llama_cpp import build_server_argv

    # The model flags point at the model setting instead.
    setting = "LLM_LLAMA_CPP_PORT" if flag.startswith(("--port", "--alias")) else "LLM_LOCAL_MODEL"
    with pytest.raises(UpstreamServiceError, match=setting):
        build_server_argv(
            ["llama-server"],
            model="m",
            port=8770,
            context_size=8192,
            extra_args=flag,
            role="llm",
            available=frozenset(),
        )


def test_build_server_argv_rejects_identity_override_ocr_role():
    """The OCR role names its own extra-args and port settings (14)."""
    from bibr.exceptions import UpstreamServiceError
    from bibr.local.llama_cpp import build_server_argv

    with pytest.raises(UpstreamServiceError, match="OCR_LLAMA_CPP_PORT"):
        build_server_argv(
            ["llama-server"],
            model="m",
            port=8771,
            context_size=8192,
            extra_args="--port 9000",
            role="ocr",
            available=frozenset(),
        )


def test_build_server_argv_allows_host_override():
    """`--host` in extra args is honoured as on main: binding 0.0.0.0 still
    answers the loopback health poll, so it is not an identity override."""
    from bibr.local.llama_cpp import build_server_argv

    argv = build_server_argv(
        ["llama-server"],
        model="m",
        port=8770,
        context_size=8192,
        extra_args="--host 0.0.0.0",
        role="llm",
        available=frozenset(),
    )
    assert argv.count("--host") == 1
    assert argv[argv.index("--host") + 1] == "0.0.0.0"  # noqa: S104


def test_build_server_argv_names_setting_on_bad_quote():
    """An unbalanced quote raises UpstreamServiceError naming the setting (14)."""
    from bibr.exceptions import UpstreamServiceError
    from bibr.local.llama_cpp import build_server_argv

    with pytest.raises(UpstreamServiceError, match="LLM_LLAMA_CPP_EXTRA_ARGS"):
        build_server_argv(
            ["llama-server"],
            model="m",
            port=8770,
            context_size=8192,
            extra_args='--chat-template-file "oops',
            role="llm",
            available=frozenset(),
        )


def test_build_server_argv_bad_quote_leaks_no_log_handle(monkeypatch, tmp_path):
    """Extra-args parsing fails before the stderr log handle is opened (11/14)."""
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import llama_cpp

    opened = {"n": 0}

    def fake_open(name, port):
        opened["n"] += 1
        raise AssertionError("must not open a log for a launch that never happens")

    import bibr.utils.secure_temp as secure_temp

    monkeypatch.setattr(secure_temp, "open_subprocess_log", fake_open)
    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: ["llama-server"])
    with pytest.raises(UpstreamServiceError):
        llama_cpp.LlamaCppServer(
            model="m", port=8770, extra_args='--chat-template-file "oops', role="llm"
        )
    assert opened["n"] == 0


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["x", "--parallel", "4", "--kv-unified"], 4),
        (["x", "--parallel", "2", "--kv-unified"], 2),
        (["x", "--parallel", "1", "--kv-unified"], 1),
        (["x", "-np", "4", "--kv-unified"], 4),
        (["x", "--parallel=4", "--kv-unified"], 4),
        (["x", "--parallel", "4"], 1),  # no kv-unified: per-slot context halves
        (["x", "--kv-unified"], 1),  # no --parallel value
        (["x", "--parallel", "many", "--kv-unified"], 1),  # garbage value
    ],
)
def test_n_slots_reads_parallel_count(argv, expected):
    """--parallel N yields N slots; garbage or missing values stay single-slot (14)."""
    from bibr.local.llama_cpp import _n_slots_from_argv

    assert _n_slots_from_argv(argv) == expected


def test_ctrl_c_during_startup_shuts_down_server(monkeypatch):
    """KeyboardInterrupt out of the health wait still kills the child (4)."""
    import subprocess as real_subprocess

    from bibr.local import llama_cpp

    kids = []
    real_popen = real_subprocess.Popen

    def fake_popen(cmd, **kwargs):
        proc = real_popen(  # noqa: S603 — `sleep` test double, not a model
            ["sleep", "120"],
            stdout=real_subprocess.DEVNULL,
            stderr=real_subprocess.DEVNULL,
            start_new_session=True,
        )
        kids.append(proc)
        return proc

    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: ["llama-server"])
    monkeypatch.setattr(llama_cpp, "supported_flags", lambda prefix: frozenset())
    monkeypatch.setattr(llama_cpp, "probe_backend_kind", lambda *a, **k: "cpu")
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", fake_popen)

    def ki_wait(self, timeout):
        raise KeyboardInterrupt()

    monkeypatch.setattr(llama_cpp.LlamaCppServer, "_wait_until_healthy", ki_wait)

    with pytest.raises(KeyboardInterrupt):
        llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")
    assert kids and kids[0].poll() is not None


def test_health_200_from_foreign_process_is_not_readiness():
    """A /health 200 while our own process exits is a startup failure, not ready (15)."""
    import http.server
    import socketserver
    import subprocess as real_subprocess
    import sys
    import threading

    from bibr.local import llama_cpp

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        # Our own process: bind fails, exits 1 after 0.3 s (loser of a port race).
        loser = real_subprocess.Popen(
            [sys.executable, "-c", "import time, sys; time.sleep(0.3); sys.exit(1)"],
            stdout=real_subprocess.DEVNULL,
            stderr=real_subprocess.DEVNULL,
        )
        server = object.__new__(llama_cpp.LlamaCppServer)
        server._process = loser
        server._port = port
        server._model = "m"
        server._stderr_fh = None
        server._stderr_log = None
        server._reused = False
        try:
            with pytest.raises(RuntimeError, match="exited during startup"):
                llama_cpp.LlamaCppServer._wait_until_healthy(server, timeout=30)
        finally:
            loser.wait()
            httpd.shutdown()
        assert loser.returncode == 1


def test_health_200_with_live_process_is_readiness(monkeypatch):
    """The grace recheck must not reject a genuinely healthy server (15 guard)."""
    import http.server
    import socketserver
    import subprocess as real_subprocess
    import threading

    from bibr.local import http_runtime, llama_cpp

    # tests/local/conftest.py refuses every probe; this test serves a real one.
    monkeypatch.setattr(llama_cpp, "request_bytes", http_runtime.request_bytes)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        proc = real_subprocess.Popen(
            ["sleep", "30"],  # noqa: S603, S607 — `sleep` test double, not a model
            stdout=real_subprocess.DEVNULL,
            stderr=real_subprocess.DEVNULL,
        )
        server = object.__new__(llama_cpp.LlamaCppServer)
        server._process = proc
        server._port = port
        server._model = "m"
        server._stderr_fh = None
        server._stderr_log = None
        server._reused = False
        try:
            llama_cpp.LlamaCppServer._wait_until_healthy(server, timeout=30)
        finally:
            proc.terminate()
            proc.wait()
            httpd.shutdown()


# ---------------------------------------------------------------------------
# CPU builds: no layer offload, a thread count, the pinned GGUF and the
# slower-call defaults.
# ---------------------------------------------------------------------------


def test_role_runtime_args_cpu_drops_offload_and_sets_threads():
    from bibr.local.llama_cpp import _role_runtime_args

    gpu = _role_runtime_args("llm", frozenset())
    cpu = _role_runtime_args("llm", frozenset(), cpu_only=True, cpu_threads=6)
    assert "--n-gpu-layers" not in cpu
    assert cpu[cpu.index("--threads") + 1] == "6"
    assert cpu[cpu.index("--flash-attn") + 1] == "on"
    assert cpu[cpu.index("--cache-type-k") + 1] == "q8_0"
    # Everything but the offload pair and the thread count is the GPU launch.
    assert [a for a in gpu if a not in ("--n-gpu-layers", "999")] == [
        a for a in cpu if a not in ("--threads", "6")
    ]
    # Unknown core count: llama.cpp picks its own.
    unknown = _role_runtime_args("ocr", frozenset(), cpu_only=True, cpu_threads=None)
    assert "--threads" not in unknown
    assert "--n-gpu-layers" not in unknown


@pytest.mark.parametrize("user", ["-t 2", "--threads 2", "--threads=2"])
def test_build_server_argv_user_threads_replace_the_core_count(user):
    from bibr.local.llama_cpp import build_server_argv

    argv = build_server_argv(
        ["llama-server"],
        model="m",
        port=8770,
        context_size=8192,
        extra_args=user,
        role="llm",
        available=frozenset(),
        cpu_only=True,
        cpu_threads=8,
    )
    assert "8" not in argv
    assert argv[-len(user.split()) :] == user.split()


def test_build_server_argv_user_gpu_layers_survive_a_cpu_build():
    """A user ``-ngl`` is passed through: the CPU launch only drops bibr's default."""
    from bibr.local.llama_cpp import build_server_argv

    argv = build_server_argv(
        ["llama-server"],
        model="m",
        port=8770,
        context_size=8192,
        extra_args="-ngl 10",
        role="llm",
        available=frozenset(),
        cpu_only=True,
        cpu_threads=4,
    )
    assert "999" not in argv
    assert argv[argv.index("-ngl") + 1] == "10"


def test_build_server_argv_loads_the_pinned_file_by_path():
    from bibr.local.llama_cpp import build_server_argv

    path = "/cache/models--org--m-GGUF/snapshots/abc123/m-Q4_K_M.gguf"
    argv = build_server_argv(
        ["llama-server"],
        model="org/m-GGUF:Q4_K_M",
        port=8770,
        context_size=8192,
        role="llm",
        available=frozenset(),
        model_path=path,
    )
    assert argv[1:3] == ["-m", path]
    assert "-hf" not in argv
    assert "--model-url" not in argv
    # bibr still asks the server for the model by its registry name.
    assert argv[argv.index("--alias") + 1] == "org/m-GGUF:Q4_K_M"


@pytest.mark.parametrize("model_path", [None, "/cache/m-Q4_K_M.gguf"])
@pytest.mark.parametrize("role", ["llm", "ocr"])
@pytest.mark.parametrize("cpu_only", [False, True])
def test_build_server_argv_always_names_a_model_to_load(model_path, role, cpu_only):
    """Without -m or -hf, llama-server starts as a router that loads nothing."""
    from bibr.local.llama_cpp import build_server_argv

    full = frozenset({"--kv-unified", "--spec-type", "--no-mmproj", "--no-context-shift"})
    for available in (full, frozenset()):
        argv = build_server_argv(
            ["llama-server"],
            model="org/m-GGUF:Q4_K_M",
            port=8770,
            context_size=8192,
            extra_args="-t 2 --temp 0",
            role=role,
            available=available,
            model_path=model_path,
            cpu_only=cpu_only,
            cpu_threads=4,
        )
        sources = [flag for flag in argv if flag in ("-m", "-hf")]
        assert sources == (["-m"] if model_path else ["-hf"])
        assert argv[argv.index(sources[0]) + 1] == (model_path or "org/m-GGUF:Q4_K_M")
        assert "--model-url" not in argv


def test_empty_device_table_is_a_cpu_build(monkeypatch):
    """Current CPU-only builds print "(none)" under the device header."""
    from bibr.local import llama_cpp

    for table in ("Available devices:\n  (none)\n", "Available devices:\n"):
        llama_cpp._BACKEND_KIND_CACHE.clear()
        _stub_probe_output(monkeypatch, stdout=table)
        assert llama_cpp.probe_backend_kind(["llama-server"]) == "cpu"
        assert llama_cpp.probe_gpu_backend(["llama-server"]) is False
    llama_cpp._BACKEND_KIND_CACHE.clear()


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("cpu", True), ("cuda", False), ("vulkan", False), (None, False)],
)
def test_runs_on_cpu_follows_the_backend_kind(monkeypatch, kind, expected):
    from bibr.local import llama_cpp

    monkeypatch.setattr(llama_cpp, "probe_backend_kind", lambda *a, **k: kind)
    assert _real_runs_on_cpu(["llama-server"]) is expected


def test_runs_on_cpu_never_raises(monkeypatch):
    from bibr.local import llama_cpp

    def boom(*_a, **_k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(llama_cpp, "probe_backend_kind", boom)
    assert _real_runs_on_cpu(["llama-server"]) is False


def _fake_cpu_sysfs(root: Path, cores: dict[int, tuple[int, int]]) -> Path:
    """sysfs ``cpuN/topology`` files for {cpu: (package, core_id)}."""
    for cpu, (package, core) in cores.items():
        topology = root / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(f"{package}\n")
        (topology / "core_id").write_text(f"{core}\n")
    return root


def test_physical_core_count_counts_cores_not_smt_threads(monkeypatch, tmp_path):
    from bibr.local import llama_cpp

    # 4 cores x 2 SMT threads: cpu N and N+4 share a core.
    sysfs = _fake_cpu_sysfs(tmp_path / "cpu", {n: (0, n % 4) for n in range(8)})
    monkeypatch.setattr(llama_cpp.sys, "platform", "linux")
    monkeypatch.setattr(llama_cpp.os, "sched_getaffinity", lambda _pid: set(range(8)))
    absent = tmp_path / "no-hybrid"
    assert llama_cpp.physical_core_count(sysfs=sysfs, performance_cores=absent) == 4
    # Confined to cpus 0-1 and their SMT siblings 4-5: two cores.
    monkeypatch.setattr(llama_cpp.os, "sched_getaffinity", lambda _pid: {0, 1, 4, 5})
    assert llama_cpp.physical_core_count(sysfs=sysfs, performance_cores=absent) == 2


def test_physical_core_count_uses_performance_cores_on_hybrid_cpus(monkeypatch, tmp_path):
    from bibr.local import llama_cpp

    # cpus 0-3: two P-cores with SMT; cpus 4-7: four E-cores.
    layout = {0: (0, 0), 1: (0, 0), 2: (0, 4), 3: (0, 4)}
    layout.update({cpu: (0, 8 + cpu) for cpu in range(4, 8)})
    sysfs = _fake_cpu_sysfs(tmp_path / "cpu", layout)
    p_cores = tmp_path / "cpu_core_cpus"
    p_cores.write_text("0-3\n")
    monkeypatch.setattr(llama_cpp.sys, "platform", "linux")
    monkeypatch.setattr(llama_cpp.os, "sched_getaffinity", lambda _pid: set(range(8)))
    assert llama_cpp.physical_core_count(sysfs=sysfs, performance_cores=p_cores) == 2
    # Pinned to E-cores only: those are all it has.
    monkeypatch.setattr(llama_cpp.os, "sched_getaffinity", lambda _pid: {6, 7})
    assert llama_cpp.physical_core_count(sysfs=sysfs, performance_cores=p_cores) == 2


def test_physical_core_count_unknown_topology_is_none(monkeypatch, tmp_path):
    from bibr.local import llama_cpp

    monkeypatch.setattr(llama_cpp.sys, "platform", "linux")
    monkeypatch.setattr(llama_cpp.os, "sched_getaffinity", lambda _pid: {0, 1})
    empty = tmp_path / "cpu"
    empty.mkdir()
    assert llama_cpp.physical_core_count(sysfs=empty, performance_cores=tmp_path / "x") is None
    monkeypatch.setattr(llama_cpp.sys, "platform", "win32")
    assert llama_cpp.physical_core_count() is None


def test_cpu_list_parses_ranges():
    from bibr.local.llama_cpp import _cpu_list

    assert _cpu_list("0-3,8-9,12\n") == {0, 1, 2, 3, 8, 9, 12}
    assert _cpu_list("") == set()


def test_cpu_install_hint_points_at_the_cpu_builds():
    from bibr.local.llama_cpp import cpu_install_hint, missing_binary_message

    windows = cpu_install_hint(platform_name="win32")
    assert "win-cpu-x64" in windows
    # winget installs the Vulkan build, which offloads to an integrated GPU.
    assert "winget" not in windows
    assert "brew install llama.cpp" in cpu_install_hint(platform_name="darwin")
    linux = cpu_install_hint(platform_name="linux")
    assert "ubuntu-x64" in linux
    assert "GGML_CUDA" not in linux
    assert "ubuntu-x64" in missing_binary_message(platform_name="linux", cpu=True)
    assert "GGML_CUDA" in missing_binary_message(platform_name="linux")


def test_cpu_build_launches_without_offload_and_with_threads(monkeypatch, caplog):
    from bibr.local import llama_cpp

    launched = _healthy_stub_server(
        monkeypatch, available=frozenset(), wait_side_effect=lambda self, t: None
    )
    monkeypatch.setattr(llama_cpp, "runs_on_cpu", _real_runs_on_cpu)
    monkeypatch.setattr(llama_cpp, "physical_core_count", lambda: 6)
    offload_logged = MagicMock()
    monkeypatch.setattr(llama_cpp, "log_offload_status", offload_logged)

    with caplog.at_level(logging.INFO, logger="bibr.local.llama_cpp"):
        server = llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")
    argv = launched[0]
    assert server.cpu_only is True
    assert "--n-gpu-layers" not in argv
    assert argv[argv.index("--threads") + 1] == "6"
    assert "runs on the CPU (6 threads)" in caplog.text
    server._process = None
    server.shutdown()


def test_gpu_build_keeps_the_offload(monkeypatch):
    from bibr.local import llama_cpp

    launched = _healthy_stub_server(
        monkeypatch,
        available=frozenset(),
        wait_side_effect=lambda self, t: None,
        backend_kind="cuda",
    )
    monkeypatch.setattr(llama_cpp, "runs_on_cpu", _real_runs_on_cpu)
    server = llama_cpp.LlamaCppServer(model="org/model:Q4", port=8770, role="llm")
    argv = launched[0]
    assert server.cpu_only is False
    assert argv[argv.index("--n-gpu-layers") + 1] == "999"
    assert "--threads" not in argv
    server._process = None
    server.shutdown()


def test_llm_server_on_a_cpu_build_uses_the_fastest_pinned_build(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llama_cpp
    from bibr.local.llm_models import pinned_gguf

    fake = MagicMock(base_url="http://127.0.0.1:8770", model="x", n_slots=1)
    ctor = MagicMock(return_value=fake)
    monkeypatch.setattr(llama_cpp, "LlamaCppServer", ctor)
    monkeypatch.setattr(llama_cpp, "runs_on_cpu", lambda prefix=None: True)
    monkeypatch.setattr(Settings.llm, "local_model", None)
    llama_cpp.LlamaCppLlmServer()
    model = "scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M"
    assert ctor.call_args.kwargs["model"] == model
    assert ctor.call_args.kwargs["cpu_only"] is True
    assert ctor.call_args.kwargs["gguf"] == pinned_gguf(model)
    assert ctor.call_args.kwargs["gguf"].filename == "bibr-extract-qwen35-2b-Q4_K_M.gguf"


def test_llm_server_pins_no_file_for_a_model_outside_the_registry(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llama_cpp

    fake = MagicMock(base_url="http://127.0.0.1:8770", model="x", n_slots=1)
    ctor = MagicMock(return_value=fake)
    monkeypatch.setattr(llama_cpp, "LlamaCppServer", ctor)
    monkeypatch.setattr(Settings.llm, "local_model", "someone/else-GGUF:Q4_K_M")
    llama_cpp.LlamaCppLlmServer()
    assert ctor.call_args.kwargs["gguf"] is None
    assert ctor.call_args.kwargs["cpu_only"] is False


def test_llm_server_on_a_cpu_build_raises_timeout_and_skips_the_equation_llm(caplog):
    server = _llm_server_with_isolated_settings()
    server._cpu_only = True
    with caplog.at_level(logging.INFO, logger="bibr.local.llama_cpp"):
        server.configure_llm_client()
    assert server._settings.llm.timeout_seconds == 900
    # Only the LLM fallback: the regex equations need no LLM and stay.
    assert server._settings.EQUATION_LLM_FALLBACK is False
    assert server._settings.EQUATION_EXTRACTION is True
    assert "EQUATION_LLM_FALLBACK=true" in caplog.text


def test_llm_server_on_a_cpu_build_respects_explicit_settings():
    server = _llm_server_with_isolated_settings(
        EQUATION_LLM_FALLBACK="true", LLM_TIMEOUT_SECONDS=120
    )
    server._cpu_only = True
    server.configure_llm_client()
    assert server._settings.llm.timeout_seconds == 120
    assert server._settings.EQUATION_LLM_FALLBACK is True


def test_llm_server_on_a_gpu_build_keeps_equations():
    server = _llm_server_with_isolated_settings()
    server.configure_llm_client()
    assert server._settings.llm.timeout_seconds == 300
    assert server._settings.EQUATION_EXTRACTION is True
    assert server._settings.EQUATION_LLM_FALLBACK is True


# ---------------------------------------------------------------------------
# Pinned GGUF builds: fetched into the Hugging Face cache and loaded with -m,
# and a server that loads no model is never reported ready.
# ---------------------------------------------------------------------------


def _pinned():
    from bibr.local.llm_models import PinnedGguf

    return PinnedGguf("org/m-GGUF", "m-Q4_K_M.gguf", "a" * 40)


def test_pinned_build_launches_with_the_downloaded_file(monkeypatch):
    import threading

    from bibr.local import llama_cpp

    launched = _healthy_stub_server(
        monkeypatch, available=frozenset(), wait_side_effect=lambda self, t: None
    )
    download = MagicMock(return_value="/hf/snapshots/aaa/m-Q4_K_M.gguf")
    monkeypatch.setattr(llama_cpp, "download_pinned_gguf", download)
    stop = threading.Event()

    server = llama_cpp.LlamaCppServer(
        model="org/m-GGUF:Q4_K_M",
        port=8770,
        role="llm",
        gguf=_pinned(),
        startup_timeout=123,
        stop_event=stop,
    )
    argv = launched[0]
    assert argv[argv.index("-m") + 1] == "/hf/snapshots/aaa/m-Q4_K_M.gguf"
    assert "-hf" not in argv and "--model-url" not in argv
    assert argv[argv.index("--alias") + 1] == "org/m-GGUF:Q4_K_M"
    download.assert_called_once()
    assert download.call_args.args == (_pinned(),)
    assert download.call_args.kwargs["timeout"] == 123
    assert download.call_args.kwargs["stop_event"] is stop
    assert download.call_args.kwargs["timeout_setting"] == "LLM_LLAMA_CPP_STARTUP_TIMEOUT"
    server._process = None
    server.shutdown()


def test_pinned_build_downloads_nothing_before_the_checks_pass(monkeypatch):
    """No binary, a reused server or bad extra args: refused before any download."""
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import llama_cpp

    download = MagicMock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr(llama_cpp, "download_pinned_gguf", download)

    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: None)
    with pytest.raises(UpstreamServiceError, match="not installed"):
        llama_cpp.LlamaCppServer(model="org/m-GGUF:Q4_K_M", port=8770, role="llm", gguf=_pinned())

    _healthy_stub_server(monkeypatch, available=frozenset(), wait_side_effect=lambda s, t: None)
    with pytest.raises(UpstreamServiceError, match="LLM_LOCAL_MODEL"):
        llama_cpp.LlamaCppServer(
            model="org/m-GGUF:Q4_K_M",
            port=8770,
            role="llm",
            gguf=_pinned(),
            extra_args="-m /home/u/custom.gguf",
        )

    monkeypatch.setattr(llama_cpp, "guard_managed_server_port", lambda *a, **k: True)
    reused = llama_cpp.LlamaCppServer(
        model="org/m-GGUF:Q4_K_M", port=8770, role="llm", gguf=_pinned()
    )
    assert reused.loaded
    download.assert_not_called()


def _fake_hub(monkeypatch, *, cached=None, fetch=None):
    """Stub ``hf_hub_download_no_symlink``: *cached* for cache lookups, *fetch* otherwise."""
    calls: list[dict] = []

    def download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            if cached is None:
                raise FileNotFoundError("not cached")
            return cached
        return fetch(**kwargs)

    monkeypatch.setattr("bibr.utils.hf_cache.hf_hub_download_no_symlink", download)
    return calls


def test_download_pinned_gguf_returns_a_cached_file_without_downloading(monkeypatch):
    calls = _fake_hub(monkeypatch, cached="/hf/m.gguf", fetch=None)
    assert _real_download_pinned_gguf(_pinned(), timeout=5) == "/hf/m.gguf"
    assert calls == [
        {
            "repo_id": "org/m-GGUF",
            "filename": "m-Q4_K_M.gguf",
            "revision": "a" * 40,
            "local_files_only": True,
        }
    ]


def test_download_pinned_gguf_fetches_the_file_at_its_commit(monkeypatch, caplog):
    calls = _fake_hub(monkeypatch, fetch=lambda **_k: "/hf/fresh.gguf")
    with caplog.at_level(logging.INFO, logger="bibr.local.llama_cpp"):
        assert _real_download_pinned_gguf(_pinned(), timeout=5) == "/hf/fresh.gguf"
    assert calls[-1] == {"repo_id": "org/m-GGUF", "filename": "m-Q4_K_M.gguf", "revision": "a" * 40}
    assert "Downloading m-Q4_K_M.gguf from org/m-GGUF" in caplog.text


def test_download_pinned_gguf_failure_names_the_file(monkeypatch):
    from bibr.exceptions import UpstreamServiceError

    def offline(**_k):
        raise OSError("offline mode is enabled")

    _fake_hub(monkeypatch, fetch=offline)
    with pytest.raises(UpstreamServiceError, match="m-Q4_K_M.gguf from org/m-GGUF") as exc:
        _real_download_pinned_gguf(_pinned(), timeout=5)
    assert "offline mode is enabled" in str(exc.value)


def test_download_pinned_gguf_stops_waiting_on_timeout_and_on_stop(monkeypatch):
    import threading

    from bibr.local.http_runtime import StartupCancelled

    release = threading.Event()

    def slow(**_k):
        release.wait(10)
        return "/hf/late.gguf"

    _fake_hub(monkeypatch, fetch=slow)
    try:
        with pytest.raises(TimeoutError, match="OCR_LLAMA_CPP_STARTUP_TIMEOUT") as exc:
            _real_download_pinned_gguf(
                _pinned(), timeout=0.2, timeout_setting="OCR_LLAMA_CPP_STARTUP_TIMEOUT"
            )
        assert f"hf download org/m-GGUF m-Q4_K_M.gguf --revision {'a' * 40}" in str(exc.value)

        stop = threading.Event()
        stop.set()
        with pytest.raises(StartupCancelled):
            _real_download_pinned_gguf(_pinned(), timeout=30, stop_event=stop)
    finally:
        release.set()


class _ModelsHandler:
    """``/health`` 200, ``/v1/models`` answering *models_body*."""

    @staticmethod
    def make(models_body: bytes):
        import http.server

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(models_body if self.path == "/v1/models" else b'{"status":"ok"}')

            def log_message(self, *args):
                pass

        return Handler


def _wait_against(monkeypatch, models_body: bytes):
    """Run the real health wait against a stub server; return (error, child process)."""
    import socketserver
    import subprocess as real_subprocess
    import threading

    from bibr.local import http_runtime, llama_cpp

    monkeypatch.setattr(llama_cpp, "request_bytes", http_runtime.request_bytes)
    with socketserver.TCPServer(("127.0.0.1", 0), _ModelsHandler.make(models_body)) as httpd:
        thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        proc = real_subprocess.Popen(
            ["sleep", "30"],  # noqa: S603, S607 — `sleep` test double, not a model
            stdout=real_subprocess.DEVNULL,
            stderr=real_subprocess.DEVNULL,
            start_new_session=True,
        )
        server = object.__new__(llama_cpp.LlamaCppServer)
        server._process = proc
        server._port = httpd.server_address[1]
        server._model = "org/m-GGUF:Q4_K_M"
        server._stderr_fh = None
        server._stderr_log = None
        server._reused = False
        server._cpu_only = True
        error = None
        try:
            llama_cpp.LlamaCppServer._wait_until_healthy(server, timeout=30)
        except RuntimeError as exc:
            error = exc
        finally:
            httpd.shutdown()
        try:
            proc.wait(timeout=5)
        except real_subprocess.TimeoutExpired:
            pass
        return error, proc


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_a_server_that_loads_no_model_is_not_ready(monkeypatch):
    """llama-server with no model is a router: /health 200, no models listed."""
    error, proc = _wait_against(monkeypatch, b'{"object": "list", "data": []}')
    try:
        assert error is not None
        assert "does not serve 'org/m-GGUF:Q4_K_M'" in str(error)
        assert "lists: none" in str(error)
        # No conservative retry: this is not an argument error.
        from bibr.local.llama_cpp import _looks_like_arg_error

        assert not _looks_like_arg_error(str(error))
        assert proc.poll() is not None  # killed, not left running
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.parametrize(
    "body",
    [
        b'{"object": "list", "data": [{"id": "org/m-GGUF:Q4_K_M"}]}',
        b'{"data": [{"id": "m-Q4_K_M.gguf", "aliases": ["org/m-GGUF:Q4_K_M"]}]}',
        # Not a model list: proves nothing, so /health decides as before.
        b"ok",
    ],
)
def test_a_server_serving_the_model_is_ready(monkeypatch, body):
    error, proc = _wait_against(monkeypatch, body)
    try:
        assert error is None
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait()
