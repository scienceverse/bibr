"""The llama.cpp server checks behind ``bibr setup``'s 10–11 GB llama.cpp pair.

``probe_build_number`` reads the build from ``--version``, and
``cuda_llama_server_for_setup`` qualifies a server only when its device list
shows a working CUDA backend and its build is at least the one every GGUF-tier
run used (b11282).
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from bibr.local import llama_cpp

_SERVER = ["/opt/llama.cpp/llama-server"]
_CUDA_DEVICES = (
    "ggml_cuda_init: found 1 CUDA devices:\n"
    "  Device 0: NVIDIA GeForce RTX 3080, compute capability 8.6, VMM: yes\n"
    "Available devices:\n"
    "  CUDA0: NVIDIA GeForce RTX 3080 (10240 MiB, 9668 MiB free)\n"
)
_CUDA_INIT_FAILED = (
    "ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient "
    "for CUDA runtime version\n"
)
_HIP_DEVICES = "Available devices:\n  ROCm0: AMD Radeon RX 7900 XTX (24560 MiB, 24000 MiB free)\n"
# A HIP build logs through ggml_cuda_init, so its kind reads as "cuda".
_HIP_THROUGH_GGML_CUDA = (
    "ggml_cuda_init: found 1 ROCm devices:\n"
    "  Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no\n" + _HIP_DEVICES
)
# A build whose CUDA backend library failed to load names CUDA only in its path.
_CUDA_LIBRARY_NOT_LOADED = (
    "Available devices:\n"
    "load_backend: failed to load backend from /usr/lib/llama.cpp/libggml-cuda.so\n"
)
_VULKAN_DEVICES = (
    "ggml_vulkan: Found 1 Vulkan devices:\nAvailable devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 3080 (10240 MiB, 9668 MiB free)\n"
)


@pytest.fixture(autouse=True)
def _fresh_probe_caches():
    caches = (
        llama_cpp._PROBE_RUN_CACHE,
        llama_cpp._BACKEND_KIND_CACHE,
    )
    saved = [dict(cache) for cache in caches]
    for cache in caches:
        cache.clear()
    yield
    for cache, before in zip(caches, saved, strict=True):
        cache.clear()
        cache.update(before)


def _server(monkeypatch, list_devices, version, *, prefix=_SERVER, on="stderr"):
    """A server on PATH printing these outputs; an exception class is raised instead."""
    calls: list[list[str]] = []

    def run(cmd, **_kwargs):
        calls.append(list(cmd))
        out = list_devices if cmd[-1] == "--list-devices" else version
        if out is subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(cmd, 8.0)
        if isinstance(out, type) and issubclass(out, BaseException):
            raise out("cannot run")
        return SimpleNamespace(
            stdout=out if on == "stdout" else "",
            stderr=out if on == "stderr" else "",
            returncode=0,
        )

    monkeypatch.setattr(
        llama_cpp, "find_llama_server", lambda: None if prefix is None else list(prefix)
    )
    monkeypatch.setattr(llama_cpp.subprocess, "run", run)
    return calls


def _version(build) -> str:
    return f"version: {build} (1a2b3c4)\nbuilt with cc (GCC) 14.2.1 for x86_64-pc-linux-gnu\n"


# --- probe_build_number ------------------------------------------------------


@pytest.mark.parametrize("on", ["stdout", "stderr"])
def test_the_build_number_comes_from_version(monkeypatch, on):
    calls = _server(monkeypatch, "", "version: 11282 (abc1234)\n", on=on)
    assert llama_cpp.probe_build_number(_SERVER) == 11282
    assert calls == [[*_SERVER, "--version"]]


def test_the_build_number_follows_backend_log_lines(monkeypatch):
    _server(monkeypatch, "", _CUDA_DEVICES + _version(11400))
    assert llama_cpp.probe_build_number(_SERVER) == 11400


def test_the_build_number_is_the_line_that_starts_with_version(monkeypatch):
    _server(monkeypatch, "", "CUDA driver version: 12040\n" + _version(11282))
    assert llama_cpp.probe_build_number(_SERVER) == 11282


@pytest.mark.parametrize(
    "output",
    [
        "",
        "version: unknown\n",
        "garbage\n",
        # llama.cpp built without git metadata reports build 0.
        "version: 0 (unknown)\n",
        # The CUDA failure text says "version" too, with no number after a colon.
        _CUDA_INIT_FAILED,
        # A "version: N" inside a line is not the build.
        "CUDA driver version: 12040\n",
    ],
)
def test_no_build_number_is_none(monkeypatch, output):
    _server(monkeypatch, "", output)
    assert llama_cpp.probe_build_number(_SERVER) is None


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired, OSError])
def test_a_version_run_that_fails_is_none(monkeypatch, error):
    _server(monkeypatch, "", error)
    assert llama_cpp.probe_build_number(_SERVER) is None


def test_no_server_has_no_build_number(monkeypatch):
    calls = _server(monkeypatch, "", _version(11282), prefix=None)
    assert llama_cpp.probe_build_number() is None
    assert calls == []


def test_the_build_number_is_cached_per_server(monkeypatch):
    calls = _server(monkeypatch, "", _version(11282))
    assert llama_cpp.probe_build_number(_SERVER) == 11282
    assert llama_cpp.probe_build_number(_SERVER) == 11282
    assert llama_cpp.probe_build_number(["/other/llama-server"]) == 11282
    assert calls == [[*_SERVER, "--version"], ["/other/llama-server", "--version"]]


def test_a_unified_llama_binary_reports_its_build_through_serve(monkeypatch):
    prefix = ["/usr/local/bin/llama", "serve"]
    calls = _server(monkeypatch, "", _version(11282), prefix=prefix)
    assert llama_cpp.probe_build_number() == 11282
    assert calls == [["/usr/local/bin/llama", "serve", "--version"]]


# --- cuda_llama_server_for_setup ---------------------------------------------


def test_a_cuda_server_of_the_measured_build_qualifies(monkeypatch):
    _server(monkeypatch, _CUDA_DEVICES, _version(llama_cpp.LLAMA_CPP_MIN_MEASURED_BUILD))
    assert llama_cpp.cuda_llama_server_for_setup() == (_SERVER, None)


def test_a_newer_build_qualifies(monkeypatch):
    _server(monkeypatch, _CUDA_DEVICES, _version(12000))
    assert llama_cpp.cuda_llama_server_for_setup() == (_SERVER, None)


def test_no_server_is_none_without_a_reason(monkeypatch):
    calls = _server(monkeypatch, _CUDA_DEVICES, _version(11282), prefix=None)
    assert llama_cpp.cuda_llama_server_for_setup() == (None, None)
    assert calls == []


@pytest.mark.parametrize(
    ("list_devices", "version", "reason"),
    [
        (_CUDA_DEVICES, _version(11281), "build 11281 below 11282"),
        (_CUDA_DEVICES, "version: unknown\n", "build unreadable"),
        (_CUDA_DEVICES, _version(0), "build unreadable"),
        (_CUDA_DEVICES, subprocess.TimeoutExpired, "probe timed out"),
        # A CUDA build on a driver older than its CUDA: kind "cuda", no GPU.
        (
            _CUDA_INIT_FAILED + "Available devices:\n",
            _CUDA_INIT_FAILED + _version(11282),
            "CUDA could not start in this build (driver older than the build's CUDA?)",
        ),
        ("Available devices:\n", _version(11282), "Vulkan/HIP/CPU build"),
        ("Available devices:\n  CPU\n", _version(11282), "Vulkan/HIP/CPU build"),
        (_VULKAN_DEVICES, _version(11282), "Vulkan/HIP/CPU build"),
        (_HIP_DEVICES, _version(11282), "Vulkan/HIP/CPU build"),
        (_HIP_THROUGH_GGML_CUDA, _version(11282), "Vulkan/HIP/CPU build"),
        # One probe timed out and the other names another backend: that backend.
        (subprocess.TimeoutExpired, _VULKAN_DEVICES + _version(11282), "Vulkan/HIP/CPU build"),
        (_VULKAN_DEVICES, subprocess.TimeoutExpired, "Vulkan/HIP/CPU build"),
        # CUDA is named, but no CUDA device.
        (
            _CUDA_LIBRARY_NOT_LOADED,
            _version(11282),
            "CUDA could not start in this build (driver older than the build's CUDA?)",
        ),
        (
            subprocess.TimeoutExpired,
            "ggml_cuda_init: found 0 CUDA devices:\n" + _version(11282),
            "CUDA could not start in this build (driver older than the build's CUDA?)",
        ),
        # An empty device list decides, as it does for probe_gpu_backend, even
        # when --version counts a CUDA device.
        (
            "Available devices:\n",
            _CUDA_DEVICES + _version(11282),
            "CUDA could not start in this build (driver older than the build's CUDA?)",
        ),
        (subprocess.TimeoutExpired, subprocess.TimeoutExpired, "probe timed out"),
        (OSError, OSError, "backend unreadable"),
        ("unrelated banner\n", "unrelated banner\n", "backend unreadable"),
    ],
)
def test_a_server_that_does_not_qualify_says_why(monkeypatch, list_devices, version, reason):
    _server(monkeypatch, list_devices, version)
    assert llama_cpp.cuda_llama_server_for_setup() == (None, reason)


def test_the_device_list_can_time_out_when_version_names_cuda(monkeypatch):
    _server(monkeypatch, subprocess.TimeoutExpired, _CUDA_DEVICES + _version(11282))
    assert llama_cpp.cuda_llama_server_for_setup() == (_SERVER, None)


@pytest.mark.parametrize(
    ("list_devices", "version"),
    [
        # The device line alone, as a build that logs no ggml_cuda_init lines prints it.
        (
            "Available devices:\n  CUDA0: NVIDIA GeForce RTX 3080 (10240 MiB, 9668 MiB free)\n",
            _version(11282),
        ),
        # The device count alone, in --version output.
        (
            subprocess.TimeoutExpired,
            "ggml_cuda_init: found 1 CUDA devices:\n"
            "  Device 0: NVIDIA GeForce RTX 3080, compute capability 8.6, VMM: yes\n"
            + _version(11282),
        ),
    ],
)
def test_either_sign_of_a_cuda_device_is_enough(monkeypatch, list_devices, version):
    _server(monkeypatch, list_devices, version)
    assert llama_cpp.cuda_llama_server_for_setup() == (_SERVER, None)


def test_one_list_devices_and_one_version_run_answer_every_check(monkeypatch):
    calls = _server(monkeypatch, _CUDA_DEVICES, _version(11282))
    assert llama_cpp.cuda_llama_server_for_setup() == (_SERVER, None)
    assert llama_cpp.cuda_llama_server_for_setup() == (_SERVER, None)
    assert llama_cpp.probe_build_number(_SERVER) == 11282
    assert llama_cpp.probe_backend_kind(_SERVER) == "cuda"
    assert calls == [[*_SERVER, "--list-devices"], [*_SERVER, "--version"]]


def test_a_rejection_is_cached_too(monkeypatch):
    calls = _server(monkeypatch, subprocess.TimeoutExpired, subprocess.TimeoutExpired)
    assert llama_cpp.cuda_llama_server_for_setup() == (None, "probe timed out")
    assert llama_cpp.cuda_llama_server_for_setup() == (None, "probe timed out")
    assert len(calls) == 2


def test_the_setup_check_agrees_with_the_runtime_probes(monkeypatch):
    """The CUDA-init failure reads as kind "cuda" but no GPU to the runtime's
    own probes, which is why setup asks for both."""
    _server(monkeypatch, _CUDA_INIT_FAILED + "Available devices:\n", _CUDA_INIT_FAILED)
    assert llama_cpp.probe_backend_kind(_SERVER) == "cuda"
    assert llama_cpp.probe_gpu_backend(_SERVER) is False


def test_a_unified_llama_binary_qualifies_as_llama_serve(monkeypatch):
    prefix = ["/usr/local/bin/llama", "serve"]
    calls = _server(monkeypatch, _CUDA_DEVICES, _version(11282), prefix=prefix)
    assert llama_cpp.cuda_llama_server_for_setup() == (prefix, None)
    assert calls == [[*prefix, "--list-devices"], [*prefix, "--version"]]
