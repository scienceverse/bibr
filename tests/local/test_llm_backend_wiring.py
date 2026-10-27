"""--llm local resolution and the managed-vllm resources branch."""

import threading
from unittest.mock import ANY, MagicMock, patch

import pytest

from bibr.exceptions import InputValidationError
from bibr.local.pipeline import LOCAL_LLM_BACKENDS, resolve_llm_backend


def test_local_resolves_to_vllm_on_linux():
    with (
        patch("platform.system", return_value="Linux"),
        patch("sys.platform", "linux"),
        patch("bibr.local.llm_models.detect_hardware", return_value=("cuda", 24.0)),
    ):
        assert resolve_llm_backend("local") == "vllm"


def test_local_resolves_to_rapid_mlx_on_mac_arm_when_installed():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch("bibr.local.rapid_mlx.rapid_mlx_unavailable_reason", return_value=None),
    ):
        assert resolve_llm_backend("local") == "rapid-mlx"


def test_local_uses_pipeline_settings_for_rapid_mlx_availability():
    """The injected pipeline settings pick the executable, not the globals (16).

    `rapid_mlx.executable` set on the pipeline snapshot must drive the
    Apple-Silicon `local` choice even when the process-global Settings point
    elsewhere.
    """
    from bibr.config import GlobalSettings

    pipeline_settings = GlobalSettings()
    pipeline_settings.rapid_mlx.executable = "python3"  # resolvable on PATH
    global_settings = GlobalSettings()
    global_settings.rapid_mlx.executable = "definitely-not-on-path-bibr"

    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch("bibr.local.rapid_mlx.snapshot_settings", return_value=global_settings),
    ):
        assert resolve_llm_backend("local", settings=pipeline_settings) == "rapid-mlx"
        assert resolve_llm_backend("local") == "vllm-mlx"


def test_local_falls_back_to_vllm_mlx_on_mac_arm_without_rapid_mlx():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch(
            "bibr.local.rapid_mlx.rapid_mlx_unavailable_reason",
            return_value="'rapid-mlx' was not found or is not executable",
        ),
    ):
        assert resolve_llm_backend("local") == "vllm-mlx"


def test_local_resolves_to_llama_cpp_on_windows():
    with patch("sys.platform", "win32"), patch("platform.system", return_value="Windows"):
        assert resolve_llm_backend("local") == "llama-cpp"


def test_local_resolves_to_llama_cpp_when_no_vllm_variant_fits():
    """A 7 GB card cannot hold the smallest recommended vLLM model (the 2B student, 8 GB)."""
    with (
        patch("platform.system", return_value="Linux"),
        patch("sys.platform", "linux"),
        patch("bibr.local.llm_models.detect_hardware", return_value=("cuda", 7.0)),
    ):
        assert resolve_llm_backend("local") == "llama-cpp"


def test_local_resolves_to_vllm_once_a_student_fits():
    with (
        patch("platform.system", return_value="Linux"),
        patch("sys.platform", "linux"),
        patch("bibr.local.llm_models.detect_hardware", return_value=("cuda", 10.0)),
    ):
        assert resolve_llm_backend("local") == "vllm"


def test_local_resolves_to_llama_cpp_on_an_8_gb_card():
    # Every managed vLLM start was refused on an emulated 8 GB card with a desktop.
    with (
        patch("platform.system", return_value="Linux"),
        patch("sys.platform", "linux"),
        patch("bibr.local.llm_models.detect_hardware", return_value=("cuda", 8.0)),
    ):
        assert resolve_llm_backend("local") == "llama-cpp"


def test_local_resolves_to_llama_cpp_on_small_cuda_gpu():
    with (
        patch("platform.system", return_value="Linux"),
        patch("sys.platform", "linux"),
        patch("bibr.local.llm_models.detect_hardware", return_value=("cuda", 6.0)),
    ):
        assert resolve_llm_backend("local") == "llama-cpp"


def test_concrete_names_pass_through():
    for name in ("cloud", "vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster"):
        assert resolve_llm_backend(name) == name


def test_resolve_llm_backend_rejects_unknown():
    with pytest.raises(InputValidationError, match="sglang"):
        resolve_llm_backend("sglang")


def test_resolve_llm_backend_rejects_typo():
    with pytest.raises(InputValidationError, match="vllm-mxl"):
        resolve_llm_backend("vllm-mxl")


def test_local_llm_backends_lists_managed_servers():
    assert (
        frozenset({"vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster"}) == LOCAL_LLM_BACKENDS
    )


async def test_start_llm_server_vllm_branch(monkeypatch):
    import bibr.local.vllm_llm as vllm_llm
    from bibr.config import GlobalSettings
    from bibr.pipeline.resources import ResourceManager

    instance = MagicMock()
    factory = MagicMock(return_value=instance)
    monkeypatch.setattr(vllm_llm, "VllmLlmServer", factory)
    custom = GlobalSettings()
    resources = ResourceManager(settings=custom, ocr_backend="paddle-vllm")
    await resources.start_llm_server(backend="vllm")
    assert resources._llm_server is instance
    # The share is sized from the VRAM free at start (the user set none), and
    # the default 0.85 cannot share the card with PaddleOCR-VL: they swap.
    factory.assert_called_once_with(
        settings=custom, stop_event=ANY, share_from_free_memory=True, swaps_with_ocr=True
    )
    assert isinstance(factory.call_args.kwargs["stop_event"], threading.Event)
    instance.configure_llm_client.assert_called_once()


async def test_start_llm_server_llama_cpp_branch(monkeypatch):
    import bibr.local.llama_cpp as llama_cpp
    from bibr.pipeline.resources import ResourceManager

    instance = MagicMock()
    monkeypatch.setattr(llama_cpp, "LlamaCppLlmServer", MagicMock(return_value=instance))
    resources = ResourceManager()
    await resources.start_llm_server(backend="llama-cpp")
    assert resources._llm_server is instance
    instance.configure_llm_client.assert_called_once()


async def test_start_llm_server_llmster_branch(monkeypatch):
    import bibr.local.llmster as llmster
    from bibr.pipeline.resources import ResourceManager

    instance = MagicMock()
    monkeypatch.setattr(llmster, "LlmsterLlmServer", MagicMock(return_value=instance))
    resources = ResourceManager()
    await resources.start_llm_server(backend="llmster")
    assert resources._llm_server is instance
    instance.configure_llm_client.assert_called_once()


def test_llm_backend_setting_default():
    from bibr.config import LlmOptions

    assert LlmOptions(_env_file=None).backend == "cloud"


def test_llmster_settings_defaults():
    from bibr.config import GlobalSettings

    llm = GlobalSettings().llm
    assert llm.llmster_model == ""
    assert llm.llmster_model_id == "bibr-local"
    assert llm.llmster_port == 1234
    assert llm.llmster_context_length == 32768


# --- item 5: fail-fast preflight for --llm local ---


def test_preflight_local_backend_no_hardware_names_alternatives():
    from bibr.local import cli

    with patch("bibr.local.llm_models.detect_hardware", return_value=(None, None)):
        err = cli._preflight_local_backend("vllm")
    assert err is not None
    assert "ollama" in err.lower()
    assert "cloud" in err.lower()


def test_preflight_local_backend_vllm_ok_with_uv(monkeypatch):
    import importlib.util
    import shutil

    from bibr.local import cli

    with (
        patch("bibr.local.llm_models.detect_hardware", return_value=("cuda", 24.0)),
        patch.object(importlib.util, "find_spec", return_value=None),
        patch.object(shutil, "which", return_value="/usr/bin/uv"),
    ):
        assert cli._preflight_local_backend("vllm") is None


def test_preflight_local_backend_vllm_missing_launcher(monkeypatch):
    import importlib.util
    import shutil

    from bibr.local import cli

    with (
        patch("bibr.local.llm_models.detect_hardware", return_value=("cuda", 24.0)),
        patch.object(importlib.util, "find_spec", return_value=None),
        patch.object(shutil, "which", return_value=None),
    ):
        err = cli._preflight_local_backend("vllm")
    assert err is not None
    assert "vllm" in err.lower()


def test_preflight_local_backend_vllm_mlx_missing(monkeypatch):
    import importlib.util

    from bibr.local import cli

    with (
        patch("bibr.local.llm_models.detect_hardware", return_value=("mlx", 32.0)),
        patch.object(importlib.util, "find_spec", return_value=None),
    ):
        err = cli._preflight_local_backend("vllm-mlx")
    assert err is not None
    assert "vllm-mlx" in err.lower()


def test_preflight_llmster_only_requires_lms_cli():
    import shutil

    from bibr.local import cli

    with (
        patch("bibr.local.llm_models.detect_hardware", return_value=(None, None)),
        patch.object(shutil, "which", return_value="/usr/local/bin/lms"),
    ):
        assert cli._preflight_local_backend("llmster") is None


def test_preflight_llmster_missing_cli_has_install_hint():
    import shutil

    from bibr.local import cli

    with patch.object(shutil, "which", return_value=None):
        err = cli._preflight_local_backend("llmster")
    assert err is not None
    assert "lmstudio.ai/install.sh" in err


def test_preflight_llama_cpp_does_not_require_detected_accelerator():
    from bibr.local import cli

    with (
        patch("bibr.local.llm_models.detect_hardware", return_value=(None, None)),
        patch("bibr.local.llama_cpp.find_llama_server", return_value=["llama-server"]),
        patch("bibr.local.llama_cpp.probe_gpu_backend", return_value=True),
    ):
        assert cli._preflight_local_backend("llama-cpp") is None


def test_preflight_llama_cpp_missing_launcher_has_platform_hint():
    from bibr.local import cli

    with (
        patch("bibr.local.llm_models.detect_hardware", return_value=(None, None)),
        patch("bibr.local.llama_cpp.find_llama_server", return_value=None),
        patch(
            "bibr.local.llama_cpp.install_hint",
            return_value="Install with: winget install llama.cpp",
        ),
    ):
        err = cli._preflight_local_backend("llama-cpp")

    assert err is not None
    assert "winget install llama.cpp" in err


def test_preflight_llama_cpp_cpu_only_still_passes():
    """CPU builds are slow but valid; preflight must not hard-fail."""
    from bibr.local import cli

    with (
        patch("bibr.local.llama_cpp.find_llama_server", return_value=["llama-server"]),
        patch("bibr.local.llama_cpp.probe_gpu_backend", return_value=False),
    ):
        assert cli._preflight_local_backend("llama-cpp") is None


@pytest.mark.parametrize(
    ("llm_backend", "expected"),
    [
        # Beside a llama.cpp LLM the automatic chain runs GLM-OCR on llama.cpp:
        # no managed vLLM server starts, so the classifiers keep the card.
        ("llama-cpp", 0.0),
        # Beside a cloud LLM the same chain starts PaddleOCR-VL on vLLM, which
        # claims 0.21 of a 24 GB card.
        ("cloud", 0.21),
    ],
)
def test_the_classifiers_leave_room_only_for_the_ocr_the_run_starts(
    monkeypatch, llm_backend, expected
):
    """``_managed_gpu_fraction`` resolves the automatic ``paddle`` chain beside
    the run's LLM backend. On a Linux 24 GB Ampere card the chain resolved
    without it lists paddle-vllm, so a llama.cpp run would keep its classifiers
    off PaddleOCR-VL's share for a server it never starts."""
    from bibr.config import GlobalSettings
    from bibr.local.pipeline import _managed_gpu_fraction
    from bibr.ocr import registry

    monkeypatch.setattr(registry.sys, "platform", "linux")
    monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(registry, "_cuda_vram_gb", lambda: 24.0)
    monkeypatch.setattr(registry, "_nvidia_compute_capability", lambda: 8.6)
    fraction = _managed_gpu_fraction(llm_backend, "paddle", GlobalSettings())
    assert fraction == pytest.approx(expected)
