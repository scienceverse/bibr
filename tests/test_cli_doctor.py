"""Tests for ``bibr doctor`` check helpers."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest


class _Recorder:
    """Collects ok/warn/fail calls the way ``_run_doctor`` closures would."""

    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def ok(self, msg: str) -> None:
        self.calls.append(("ok", msg, ""))

    def warn(self, msg: str, hint: str = "") -> None:
        self.calls.append(("warn", msg, hint))

    def fail(self, msg: str, hint: str = "") -> None:
        self.calls.append(("fail", msg, hint))


def _run_check_device(rec: _Recorder) -> None:
    from bibr.local.cli import _check_device

    _check_device(rec.ok, rec.warn, rec.fail)


def test_check_device_warns_on_incompatible_gpu():
    """Present-but-unusable GPU must be a yellow warn with the reason, not a green ok."""
    pytest.importorskip("torch")
    rec = _Recorder()
    reason = "NVIDIA GeForce GTX 1060 6GB is sm_61; this PyTorch build supports sm_75+"
    with (
        mock.patch("bibr.utils.device.cuda_incompatibility", return_value=reason),
        mock.patch("bibr.utils.device.detect_torch_device", return_value="cpu"),
    ):
        _run_check_device(rec)

    assert len(rec.calls) == 1
    status, msg, hint = rec.calls[0]
    assert status == "warn"
    assert msg.startswith("Device: cpu")  # reports the device actually selected
    assert "GPU present but unusable" in msg
    assert reason in msg  # says exactly why the GPU was skipped
    assert "glm-llama" in hint  # points at llama.cpp path for older cards
    assert "PyTorch" in msg or "falls back" in hint.lower() or "CPU" in hint


def test_check_device_ok_on_compatible_gpu():
    pytest.importorskip("torch")
    rec = _Recorder()
    with (
        mock.patch("bibr.utils.device.cuda_incompatibility", return_value=None),
        mock.patch("bibr.utils.device.detect_torch_device", return_value="cuda"),
        mock.patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"),
    ):
        _run_check_device(rec)

    assert rec.calls == [("ok", "Device: cuda (NVIDIA RTX 4090)", "")]


def test_check_device_ok_on_plain_cpu():
    """No GPU at all is still a green ok — CPU is a supported configuration."""
    pytest.importorskip("torch")
    rec = _Recorder()
    with (
        mock.patch("bibr.utils.device.cuda_incompatibility", return_value=None),
        mock.patch("bibr.utils.device.detect_torch_device", return_value="cpu"),
    ):
        _run_check_device(rec)

    assert rec.calls == [("ok", "Device: cpu", "")]


def test_opencv_runtime_reason_reports_incomplete_cv2(monkeypatch):
    from bibr.local.cli import _opencv_unavailable_reason

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())

    reason = _opencv_unavailable_reason()

    assert reason == "cv2 module is incomplete (missing 'resize')"


def test_probe_ocr_url_rejects_non_http_scheme(monkeypatch):
    """Doctor must not hand file:// or other local schemes to the HTTP probe."""
    import http.client

    from bibr.local.cli import _probe_ocr_url

    connector = mock.Mock(side_effect=AssertionError("HTTP connection must not be opened"))
    monkeypatch.setattr(http.client, "HTTPConnection", connector)
    monkeypatch.setattr(http.client, "HTTPSConnection", connector)

    assert _probe_ocr_url("file:///etc/passwd") is False


def test_check_llm_local_backend_vllm_ok_with_uv():
    """A managed local backend reports the backend + model, not an API-key check."""
    import importlib.util
    import shutil

    from bibr.local.cli import _check_llm_local_backend

    rec = _Recorder()
    with (
        mock.patch.object(importlib.util, "find_spec", return_value=None),
        mock.patch.object(shutil, "which", return_value="/usr/bin/uv"),
    ):
        _check_llm_local_backend("vllm", "org/model", rec.ok, rec.fail)
    assert rec.calls[0][0] == "ok"
    assert "vllm" in rec.calls[0][1]
    assert "org/model" in rec.calls[0][1]


def test_check_llm_local_backend_vllm_uv_runner_is_honest_about_python_314(monkeypatch):
    """No vllm package on 3.14: the extra installs nothing, so say what will happen."""
    import importlib.util
    import shutil
    import sys

    from bibr.local.cli import _check_llm_local_backend

    rec = _Recorder()
    monkeypatch.setattr(sys, "version_info", (3, 14, 0, "final", 0))
    with (
        mock.patch.object(importlib.util, "find_spec", return_value=None),
        mock.patch.object(shutil, "which", return_value="/usr/bin/uv"),
    ):
        _check_llm_local_backend("vllm", "org/model", rec.ok, rec.fail)
    assert rec.calls[0][0] == "ok"
    assert "several GB" in rec.calls[0][1]
    assert "Python 3.13" in rec.calls[0][1]
    assert "installs nothing" in rec.calls[0][1]


def test_check_llm_local_backend_vllm_missing_launcher():
    """No vllm package and no uv → red fail with an install hint (no API key)."""
    import importlib.util
    import shutil

    from bibr.local.cli import _check_llm_local_backend

    rec = _Recorder()
    with (
        mock.patch.object(importlib.util, "find_spec", return_value=None),
        mock.patch.object(shutil, "which", return_value=None),
    ):
        _check_llm_local_backend("vllm", "org/model", rec.ok, rec.fail)
    assert rec.calls[0][0] == "fail"
    assert "vllm" in rec.calls[0][1]
    assert rec.calls[0][2]  # non-empty hint


def test_check_llm_local_backend_vllm_mlx_missing():
    import importlib.util

    from bibr.local.cli import _check_llm_local_backend

    rec = _Recorder()
    with mock.patch.object(importlib.util, "find_spec", return_value=None):
        _check_llm_local_backend("vllm-mlx", "org/model", rec.ok, rec.fail)
    assert rec.calls[0][0] == "fail"
    assert "vllm-mlx" in rec.calls[0][1]


def test_check_llm_local_backend_vllm_mlx_stale_namespace_fails(monkeypatch):
    """A leftover ``vllm_mlx`` directory is not a runnable vllm-mlx install."""
    import importlib.metadata
    import importlib.util

    from bibr.local.cli import _check_llm_local_backend

    def fake_find_spec(name):
        if name == "vllm_mlx":
            return object()
        if name == "vllm_mlx.server":
            return None
        return None

    def fake_version(name):
        if name == "vllm-mlx":
            raise importlib.metadata.PackageNotFoundError(name)
        return "0"

    rec = _Recorder()
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    _check_llm_local_backend("vllm-mlx", "org/model", rec.ok, rec.fail)

    assert rec.calls[0][0] == "fail"
    assert "vllm-mlx" in rec.calls[0][1]


def test_check_ocr_backend_glm_mlx_stale_namespace_fails(monkeypatch):
    """The OCR doctor check must reject incomplete vllm-mlx installs too."""
    import importlib.metadata
    import importlib.util

    import bibr.config
    from bibr.local.cli import _check_ocr_backend

    def fake_find_spec(name):
        if name == "vllm_mlx":
            return object()
        if name == "vllm_mlx.server":
            return None
        return None

    def fake_version(name):
        if name == "vllm-mlx":
            raise importlib.metadata.PackageNotFoundError(name)
        return "0"

    rec = _Recorder()
    monkeypatch.setattr(bibr.config.Settings.ocr, "backend", "glm-mlx")
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    assert rec.calls[0][0] == "fail"
    assert "vllm-mlx" in rec.calls[0][1]


def _pin_apple_silicon_paddle_chain(monkeypatch) -> None:
    """Make the ``paddle`` automatic chain resolve as it does on Apple Silicon.

    ``resolve_backend_candidates`` branches on the host platform, so without
    this the Rapid-MLX assertions below only hold on darwin/arm64 and fail on a
    Linux CI runner (which resolves ``paddle-vllm`` first). Mirrors what
    ``test_check_automatic_linux_paddle_uses_vllm_package_runner_cache_status``
    does for the Linux chain.
    """
    from bibr.ocr.registry import OcrBackendCandidate

    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates",
        lambda backend, settings: (
            OcrBackendCandidate("paddle-rapid-mlx", "PaddlePaddle/PaddleOCR-VL-1.6", "paddle"),
            OcrBackendCandidate("paddle-mlx-vlm", "PaddlePaddle/PaddleOCR-VL-1.6", "paddle"),
            OcrBackendCandidate("glm-rapid-mlx", "THUDM/GLM-OCR", "glm"),
            OcrBackendCandidate("glm-llama", "THUDM/GLM-OCR", "glm"),
        ),
    )


def test_check_ocr_backend_paddle_rapid_candidate_is_unverified_not_ready(monkeypatch):
    """An executable alone cannot prove Paddle image OCR works through Rapid-MLX."""
    from bibr.config import Settings
    from bibr.local.cli import _check_ocr_backend

    rec = _Recorder()
    monkeypatch.setattr(Settings.ocr, "backend", "paddle")
    _pin_apple_silicon_paddle_chain(monkeypatch)
    monkeypatch.setattr("bibr.local.rapid_mlx.rapid_mlx_unavailable_reason", lambda: None)
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    assert rec.calls[0][0] == "warn"
    assert "capability unverified" in rec.calls[0][1]
    assert "startup OCR smoke" in rec.calls[0][2]
    assert "MLX-VLM fallback" in rec.calls[0][2]


def test_check_ocr_backend_paddle_reports_rapid_executable_failure_separately(monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _check_ocr_backend

    rec = _Recorder()
    monkeypatch.setattr(Settings.ocr, "backend", "paddle")
    _pin_apple_silicon_paddle_chain(monkeypatch)
    monkeypatch.setattr(
        "bibr.local.rapid_mlx.rapid_mlx_unavailable_reason",
        lambda: "'rapid-mlx' was not found or is not executable",
    )
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    assert rec.calls[0][0] == "warn"
    assert "executable unavailable" in rec.calls[0][1]
    assert "MLX-VLM fallback" in rec.calls[0][2]


def test_check_explicit_paddle_rapid_mlx_never_claims_ready_without_smoke(monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _check_ocr_backend

    rec = _Recorder()
    monkeypatch.setattr(Settings.ocr, "backend", "paddle-rapid-mlx")
    monkeypatch.setattr("bibr.local.rapid_mlx.rapid_mlx_unavailable_reason", lambda: None)
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    assert rec.calls[0][0] == "warn"
    assert "capability unverified" in rec.calls[0][1]
    assert "startup OCR smoke" in rec.calls[0][2]


def test_check_explicit_paddle_http_probes_the_configured_remote_url(monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _check_ocr_backend

    rec = _Recorder()
    monkeypatch.setattr(Settings.ocr, "backend", "paddle-http")
    monkeypatch.setattr(Settings, "OCR_BASE_URL", "http://ocr.example", raising=False)
    monkeypatch.setattr("bibr.local.cli._probe_ocr_url", lambda url: False)

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    assert rec.calls[0][0] == "warn"
    assert "Paddle OCR" in rec.calls[0][1]
    assert "unreachable" in rec.calls[0][1]


def test_check_explicit_paddle_mlx_vlm_reports_launch_and_cache_state(monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _check_ocr_backend

    rec = _Recorder()
    monkeypatch.setattr(Settings.ocr, "backend", "paddle-mlx-vlm")
    monkeypatch.setattr(
        "bibr.local.cli.doctor._paddle_mlx_vlm_status",
        lambda model: (False, "launch unavailable; model cache not checked"),
    )

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    assert rec.calls[0][0] == "fail"
    assert "launch unavailable" in rec.calls[0][1]
    assert "capability unverified" in rec.calls[0][1]


def test_check_explicit_paddle_vllm_reports_uv_runner_and_cache_without_green(monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _check_ocr_backend

    rec = _Recorder()
    monkeypatch.setattr(Settings.ocr, "backend", "paddle-vllm")
    monkeypatch.setattr(
        "bibr.local.cli.doctor._paddle_vllm_status",
        lambda model: (True, "launch: uv isolated vllm runner; model cache: cached"),
    )

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    assert rec.calls[0][0] == "warn"
    assert "uv isolated vllm runner" in rec.calls[0][1]
    assert "model cache: cached" in rec.calls[0][1]
    assert "capability unverified" in rec.calls[0][1]


def test_check_automatic_linux_paddle_uses_vllm_package_runner_cache_status(monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _check_ocr_backend
    from bibr.ocr.registry import OcrBackendCandidate

    rec = _Recorder()
    monkeypatch.setattr(Settings.ocr, "backend", "paddle")
    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates",
        lambda backend, settings: (
            OcrBackendCandidate("paddle-vllm", "PaddlePaddle/PaddleOCR-VL-1.6", "paddle"),
            OcrBackendCandidate("glm-llama", "THUDM/GLM-OCR", "glm"),
        ),
    )
    monkeypatch.setattr(
        "bibr.local.cli.doctor._paddle_vllm_status",
        lambda model: (False, "launch unavailable; package absent; model cache not checked"),
    )

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    assert rec.calls[0][0] == "warn"
    assert "paddle-vllm" in rec.calls[0][1]
    assert "package absent" in rec.calls[0][1]
    assert "model cache not checked" in rec.calls[0][1]


def test_preflight_local_backend_vllm_mlx_stale_namespace_fails(monkeypatch):
    """The chew preflight should stop before pipeline startup in this state."""
    import importlib.metadata
    import importlib.util

    from bibr.local.cli import _preflight_local_backend

    def fake_find_spec(name):
        if name == "vllm_mlx":
            return object()
        if name == "vllm_mlx.server":
            return None
        return None

    def fake_version(name):
        if name == "vllm-mlx":
            raise importlib.metadata.PackageNotFoundError(name)
        return "0"

    monkeypatch.setattr("bibr.local.llm_models.detect_hardware", lambda: ("mlx", 32.0))
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    msg = _preflight_local_backend("vllm-mlx")

    assert msg is not None
    assert "vllm-mlx backend selected" in msg


def test_check_llm_local_backend_llama_cpp():
    from bibr.local.cli import _check_llm_local_backend

    rec = _Recorder()
    with (
        mock.patch("bibr.local.llama_cpp.find_llama_server", return_value=["llama", "serve"]),
        mock.patch("bibr.local.llama_cpp.probe_gpu_backend", return_value=None),
    ):
        _check_llm_local_backend("llama-cpp", "org/model:Q4", rec.ok, rec.fail)
    assert rec.calls == [
        ("ok", "LLM backend: llama-cpp (managed local server, model=org/model:Q4)", "")
    ]


def test_check_llm_local_backend_llama_cpp_gpu():
    from bibr.local.cli import _check_llm_local_backend

    rec = _Recorder()
    with (
        mock.patch("bibr.local.llama_cpp.find_llama_server", return_value=["llama-server"]),
        mock.patch("bibr.local.llama_cpp.probe_gpu_backend", return_value=True),
        # A CUDA build needs no steering, so the plain GPU line is reported.
        mock.patch("bibr.local.llama_cpp.probe_backend_kind", return_value="cuda"),
    ):
        _check_llm_local_backend("llama-cpp", "org/model:Q4", rec.ok, rec.fail)
    assert rec.calls == [
        ("ok", "LLM backend: llama-cpp (GPU, managed local server, model=org/model:Q4)", "")
    ]


def test_check_llm_local_backend_llama_cpp_vulkan_on_nvidia_steers():
    """A Vulkan build on an NVIDIA GPU must surface the CUDA steering hint."""
    from bibr.local.cli import _check_llm_local_backend

    rec = _Recorder()
    with (
        mock.patch("bibr.local.llama_cpp.find_llama_server", return_value=["llama-server"]),
        mock.patch("bibr.local.llama_cpp.probe_gpu_backend", return_value=True),
        mock.patch("bibr.local.llama_cpp.probe_backend_kind", return_value="vulkan"),
        mock.patch("bibr.local.llama_cpp.cuda_steering_hint", return_value="STEER-CUDA-LLM"),
    ):
        _check_llm_local_backend("llama-cpp", "org/model:Q4", rec.ok, rec.fail)
    assert rec.calls[0][0] == "ok"
    assert "STEER-CUDA-LLM" in rec.calls[0][1]
    assert "org/model:Q4" in rec.calls[0][1]


def test_check_ocr_backend_glm_llama_vulkan_on_nvidia_warns(monkeypatch):
    """The OCR doctor check surfaces the steering hint as a warning line."""
    import bibr.config
    from bibr.local.cli import _check_ocr_backend

    rec = _Recorder()
    monkeypatch.setattr(bibr.config.Settings.ocr, "backend", "glm-llama")
    # Reach the soft-warn branch (a real broken opencv install must not mask it).
    # ``doctor`` calls its own module-level helper, so patching the re-export on
    # ``bibr.local.cli`` (which is what the chew preflight resolves) is a no-op
    # here — on a torch-free install cv2 is genuinely absent and this test hit
    # the opencv hard-fail instead of the branch under test.
    monkeypatch.setattr("bibr.local.cli.doctor._opencv_unavailable_reason", lambda: None)
    with (
        mock.patch("bibr.local.llama_cpp.find_llama_server", return_value=["llama-server"]),
        mock.patch("bibr.local.llama_cpp.probe_gpu_backend", return_value=True),
        mock.patch("bibr.local.llama_cpp.probe_backend_kind", return_value="vulkan"),
        mock.patch("bibr.local.llama_cpp.cuda_steering_hint", return_value="STEER-CUDA-OCR"),
    ):
        _check_ocr_backend(rec.ok, rec.warn, rec.fail)
    assert rec.calls[0][0] == "warn"
    assert "vulkan" in rec.calls[0][1].lower()
    assert "STEER-CUDA-OCR" in rec.calls[0][2]


def test_preflight_llama_cpp_vulkan_on_nvidia_warns(monkeypatch, caplog):
    """Chew preflight logs the steering hint for a Vulkan-on-NVIDIA build."""
    import logging

    from bibr.local.cli import _preflight_local_backend

    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: ["llama-server"])
    monkeypatch.setattr("bibr.local.llama_cpp.probe_gpu_backend", lambda prefix: True)
    monkeypatch.setattr("bibr.local.llama_cpp.probe_backend_kind", lambda prefix: "vulkan")
    monkeypatch.setattr(
        "bibr.local.llama_cpp.cuda_steering_hint",
        lambda kind: "STEER-CUDA-PREFLIGHT" if kind == "vulkan" else None,
    )
    with caplog.at_level(logging.WARNING):
        msg = _preflight_local_backend("llama-cpp")

    assert msg is None  # steering is a soft warning, not a preflight blocker
    assert any("STEER-CUDA-PREFLIGHT" in r.getMessage() for r in caplog.records)


def test_preflight_llama_cpp_cpu_only_still_warns_without_steering(monkeypatch, caplog):
    """CPU-only detection is unchanged; no steering hint is probed/emitted."""
    import logging

    from bibr.local.cli import _preflight_local_backend

    steering_calls = {"n": 0}

    def _steering(kind):
        steering_calls["n"] += 1
        return None

    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: ["llama-server"])
    monkeypatch.setattr("bibr.local.llama_cpp.probe_gpu_backend", lambda prefix: False)
    monkeypatch.setattr("bibr.local.llama_cpp.cuda_steering_hint", _steering)
    with caplog.at_level(logging.WARNING):
        msg = _preflight_local_backend("llama-cpp")

    assert msg is None
    assert any("CPU-only build" in r.getMessage() for r in caplog.records)
    # CPU-only path must not probe the backend kind for steering.
    assert steering_calls["n"] == 0


def test_check_llm_local_backend_llama_cpp_missing_uses_install_hint():
    from bibr.local.cli import _check_llm_local_backend

    rec = _Recorder()
    with (
        mock.patch("bibr.local.llama_cpp.find_llama_server", return_value=None),
        mock.patch(
            "bibr.local.llama_cpp.install_hint",
            return_value="Install a CUDA build from releases",
        ),
    ):
        _check_llm_local_backend("llama-cpp", "org/model:Q4", rec.ok, rec.fail)
    assert rec.calls[0][0] == "fail"
    assert "launcher not available" in rec.calls[0][1]
    assert "CUDA build" in rec.calls[0][2]


def test_check_ref_strategies_llm_ok(monkeypatch):
    """Explicit llm/llm strategies need no ML deps and report ok."""
    import bibr.config

    monkeypatch.setattr(bibr.config.Settings, "REF_SEG_STRATEGY", "llm")
    monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "llm")
    from bibr.local.cli import _check_ref_strategies

    rec = _Recorder()
    _check_ref_strategies(rec.ok, rec.fail)
    assert rec.calls == [("ok", "References: seg=llm, parse=llm", "")]


def test_check_ref_strategies_geom_without_sklearn_fails(monkeypatch):
    """The geom default without the ml extra reports a soft fail (cascades to LLM)."""
    import importlib.util

    import bibr.config

    monkeypatch.setattr(bibr.config.Settings, "REF_SEG_STRATEGY", "geom")
    monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "llm")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    from bibr.local.cli import _check_ref_strategies

    rec = _Recorder()
    _check_ref_strategies(rec.ok, rec.fail)
    assert rec.calls[0][0] == "fail"
    assert "cascade to LLM" in rec.calls[0][1]
    assert "Reinstall bibr" in rec.calls[0][2]


def test_check_ref_strategies_ner_without_torch_is_ok_on_onnx(monkeypatch):
    """A core install parses references on ONNX Runtime — not a failure."""
    import importlib.util

    import bibr.config

    monkeypatch.setattr(bibr.config.Settings, "REF_SEG_STRATEGY", None)
    monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "ner")
    # conftest pins ML_RUNTIME=torch so nothing reaches the Hub; this is the
    # default-install case.
    monkeypatch.setattr(bibr.config.Settings.ml, "runtime", "auto")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    from bibr.local.cli import _check_ref_strategies

    rec = _Recorder()
    _check_ref_strategies(rec.ok, rec.fail)
    assert rec.calls[0][0] == "ok"
    assert "ONNX Runtime" in rec.calls[0][1]


def test_check_ref_strategies_ner_fails_when_torch_runtime_is_forced(monkeypatch):
    """ML_RUNTIME=torch with no torch is a real misconfiguration."""
    import importlib.util

    import bibr.config

    monkeypatch.setattr(bibr.config.Settings, "REF_SEG_STRATEGY", None)
    monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "ner")
    monkeypatch.setattr(bibr.config.Settings.ml, "runtime", "torch")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    from bibr.local.cli import _check_ref_strategies

    rec = _Recorder()
    _check_ref_strategies(rec.ok, rec.fail)
    assert rec.calls[0][0] == "fail"
    assert "ML_RUNTIME=torch" in rec.calls[0][1]
    assert "uv sync --extra torch" in rec.calls[0][2]


def test_check_ref_strategies_crf_segmenter_without_torch_fails(monkeypatch):
    """The CRF segmenter has no ONNX export, so it still needs the extra."""
    import importlib.util

    import bibr.config

    monkeypatch.setattr(bibr.config.Settings, "REF_SEG_STRATEGY", "crf")
    monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "ner")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    from bibr.local.cli import _check_ref_strategies

    rec = _Recorder()
    _check_ref_strategies(rec.ok, rec.fail)
    assert rec.calls[0][0] == "fail"
    assert "torch-only" in rec.calls[0][1]


def test_check_ref_strategies_off_reports_disabled(monkeypatch):
    """parse=off disables reference extraction — no ML deps required, always ok."""
    import importlib.util

    import bibr.config

    monkeypatch.setattr(bibr.config.Settings, "REF_SEG_STRATEGY", None)
    monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "off")
    # Even with no ML deps installed the check must pass.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    from bibr.local.cli import _check_ref_strategies

    rec = _Recorder()
    _check_ref_strategies(rec.ok, rec.fail)
    assert rec.calls[0][0] == "ok"
    assert "off" in rec.calls[0][1]


def test_run_doctor_exits_1_when_a_check_fails(monkeypatch):
    """``bibr doctor`` must exit 1 if any check reports a hard failure.

    Forces the LLM-provider check to fail (no API key resolvable) without
    hitting the network: ``provider="openai"`` with ``api_key=None`` fails
    the provider check (component 3) and, since ``api_key`` is falsy, also
    skips straight to the "LLM connection: skipped" fail (component 4) —
    no real API call is made either way.
    """
    import bibr.config
    from bibr.local.cli import _run_doctor

    monkeypatch.setattr(bibr.config.Settings.llm, "provider", "openai")
    monkeypatch.setattr(bibr.config.Settings.llm, "api_key", None)

    with pytest.raises(SystemExit) as exc_info:
        _run_doctor()
    assert exc_info.value.code == 1


def test_check_ref_strategies_off_with_geom_seg_still_ok(monkeypatch):
    """parse=off short-circuits the geom-seg dep check too — seg never runs."""
    import importlib.util

    import bibr.config

    monkeypatch.setattr(bibr.config.Settings, "REF_SEG_STRATEGY", "geom")
    monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "off")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    from bibr.local.cli import _check_ref_strategies

    rec = _Recorder()
    _check_ref_strategies(rec.ok, rec.fail)
    assert rec.calls[0][0] == "ok"
    assert "off" in rec.calls[0][1]


# --- LLM connection check redaction (audit M2) --------------------------------


def _run_check_llm_connection(rec, *, provider="google", model="gemini-x", api_key, base_url=None):
    import io

    from rich.console import Console

    from bibr.local.cli.doctor import _check_llm_connection

    console = Console(file=io.StringIO())
    _check_llm_connection(provider, model, api_key, base_url, console, rec.ok, rec.warn, rec.fail)


def test_llm_connection_failure_redacts_api_key():
    """A transient SDK error whose text embeds ``?key=<API_KEY>`` must not leak the
    key into doctor output (which users paste into GitHub issues)."""
    key = "AIzaSyA1234567890abcdefGHIJKLMNOPqrstuv"
    boom = RuntimeError(
        f"403 Forbidden: GET https://generativelanguage.googleapis.com/v1?key={key}"
    )

    class _RaisingClient:
        def create(self, *a, **k):
            raise boom

    rec = _Recorder()
    with mock.patch("instructor.from_provider", return_value=_RaisingClient()):
        _run_check_llm_connection(rec, api_key=key)

    assert len(rec.calls) == 1
    status, msg, _hint = rec.calls[0]
    assert status == "fail"
    assert "LLM connection failed" in msg
    assert key not in msg, "raw API key leaked into doctor output"
    assert key[:12] not in msg


def test_llm_connection_ok_reports_success():
    class _OkClient:
        def create(self, *a, **k):
            return SimpleNamespace(reply="OK")

    rec = _Recorder()
    with mock.patch("instructor.from_provider", return_value=_OkClient()):
        _run_check_llm_connection(rec, api_key="AIzaSyAsomethinglongenough12345")

    assert rec.calls == [("ok", "LLM connection OK", "")]


# --- the whole doctor run, with the device and OCR checks stubbed ------------


class _Completion:
    """What the fake Instructor client returns; awaitable, like the async client's result."""

    reply = "OK"

    def __await__(self):
        async def _result():
            return self

        return _result().__await__()


class _FakeInstructor:
    """Stands in for ``instructor.from_provider``: records clients built and requests sent."""

    def __init__(self):
        self.clients: list[tuple[str, dict]] = []
        self.requests: list[dict] = []

    def __call__(self, model, **kwargs):
        kwargs.pop("api_key", None)
        self.clients.append((model, kwargs))
        return self

    def create(self, **kwargs):
        self.requests.append(
            {
                k: v
                for k, v in kwargs.items()
                if k not in {"response_model", "messages", "max_retries"}
            }
        )
        return _Completion()


def _run_doctor_lines(monkeypatch) -> tuple[list[str], int, _FakeInstructor]:
    """Run ``bibr doctor`` offline; return its output lines, exit code and fake client."""
    import io

    import rich.console

    import bibr.local.cli.doctor as doctor

    out = io.StringIO()
    real_console = rich.console.Console
    monkeypatch.setattr(
        rich.console,
        "Console",
        lambda *a, **k: real_console(file=out, width=400, color_system=None),
    )
    monkeypatch.setattr(doctor, "_check_device", lambda ok, warn, fail: ok("Device: stub"))
    monkeypatch.setattr(
        doctor, "_check_ocr_backend", lambda ok, warn, fail: ok("OCR backend: stub")
    )
    fake = _FakeInstructor()
    monkeypatch.setattr("instructor.from_provider", fake)
    code = 0
    try:
        doctor._run_doctor()
    except SystemExit as exc:
        code = exc.code
    return [line.strip() for line in out.getvalue().splitlines() if line.strip()], code, fake


def test_doctor_names_the_home_env_file(monkeypatch, tmp_path):
    """Settings read ~/.bibr/.env, so a project directory without .env is not a failure."""
    from pathlib import Path

    home = tmp_path / "home"
    (home / ".bibr").mkdir(parents=True)
    env_file = home / ".bibr" / ".env"
    env_file.write_text("LLM_PROVIDER=google\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("BIBR_ENV_FILE", raising=False)
    monkeypatch.delenv("BIBR_DISABLE_DOTENV", raising=False)

    lines, code, _fake = _run_doctor_lines(monkeypatch)

    assert [line for line in lines if ".env" in line] == [f"✓ .env: {env_file.resolve()}"]
    assert code == 0


def test_doctor_without_env_file_warns_and_passes(monkeypatch):
    """Configuration from the environment alone is legitimate (containers, CI)."""
    monkeypatch.setenv("BIBR_ENV_FILE", "")

    lines, code, _fake = _run_doctor_lines(monkeypatch)

    assert "! No .env file found; settings come from the environment and defaults" in lines
    assert code == 0


def test_doctor_without_uv_warns_and_passes(monkeypatch):
    """``pip install bibr`` is a documented setup; only the uv-managed runners need uv."""
    import shutil

    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda name, *a, **k: None if name == "uv" else real_which(name, *a, **k)
    )

    lines, code, _fake = _run_doctor_lines(monkeypatch)

    assert "! uv not on PATH" in lines
    assert code == 0
