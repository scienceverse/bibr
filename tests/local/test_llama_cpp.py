"""Cross-platform process plumbing for the managed llama.cpp runtime."""

import contextlib
import logging
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


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
        lambda name: "C:\\llama.exe" if name == "llama" else None,
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
    assert ctor.call_args.kwargs["model"] == "numind/NuExtract3-GGUF:Q4_K_M"


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
            raise RuntimeError("llama.cpp exited during startup (code 1): bad --spec-type")
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
