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
        patch("bibr.local.llm_models.detect_hardware", return_value=("cuda", 8.0)),
    ):
        assert resolve_llm_backend("local") == "vllm"


def test_local_resolves_to_llama_cpp_on_small_cuda_gpu():
    with (
        patch("platform.system", return_value="Linux"),
        patch("sys.platform", "linux"),
        patch("bibr.local.llm_models.detect_hardware", return_value=("cuda", 6.0)),
    ):
        assert resolve_llm_backend("local") == "llama-cpp"


@pytest.mark.parametrize(
    ("llama_server", "expected"), [(["llama-server"], "llama-cpp"), (None, "vllm")]
)
def test_local_without_a_gpu_runs_llama_cpp_on_the_cpu_when_installed(llama_server, expected):
    """No GPU: llama.cpp when installed; otherwise vllm, whose preflight names the options."""
    with (
        patch("platform.system", return_value="Linux"),
        patch("sys.platform", "linux"),
        patch("bibr.local.llm_models.detect_hardware", return_value=(None, None)),
        patch("bibr.local.llama_cpp.find_llama_server", return_value=llama_server),
    ):
        assert resolve_llm_backend("local") == expected


def test_local_on_a_gpu_does_not_look_for_llama_server():
    with (
        patch("platform.system", return_value="Linux"),
        patch("sys.platform", "linux"),
        patch("bibr.local.llm_models.detect_hardware", return_value=("cuda", 24.0)),
        patch("bibr.local.llama_cpp.find_llama_server", return_value=["llama-server"]) as find,
    ):
        assert resolve_llm_backend("local") == "vllm"
    find.assert_not_called()


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
    resources = ResourceManager(settings=custom)
    await resources.start_llm_server(backend="vllm")
    assert resources._llm_server is instance
    factory.assert_called_once_with(settings=custom, stop_event=ANY)
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

    with (
        patch("bibr.local.llm_models.detect_hardware", return_value=(None, None)),
        patch("bibr.local.llama_cpp.cpu_install_hint", return_value="CPU-HINT"),
    ):
        err = cli._preflight_local_backend("vllm")
    assert err is not None
    assert "ollama" in err.lower()
    assert "cloud" in err.lower()
    # The CPU route: install llama.cpp and --llm local picks it.
    assert "CPU-HINT" in err
    assert "--llm local" in err


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


@pytest.mark.parametrize(
    ("hardware", "hint"), [(("cuda", 6.0), "GPU-BUILD-HINT"), ((None, None), "CPU-BUILD-HINT")]
)
def test_preflight_llama_cpp_missing_launcher_has_platform_hint(hardware, hint):
    """A machine with no GPU is pointed at the CPU builds, not at a CUDA build."""
    from bibr.local import cli

    with (
        patch("bibr.local.llm_models.detect_hardware", return_value=hardware),
        patch("bibr.local.llama_cpp.find_llama_server", return_value=None),
        patch("bibr.local.llama_cpp.install_hint", return_value="GPU-BUILD-HINT"),
        patch("bibr.local.llama_cpp.cpu_install_hint", return_value="CPU-BUILD-HINT"),
    ):
        err = cli._preflight_local_backend("llama-cpp")

    assert err is not None
    assert err.endswith(hint)


@pytest.mark.parametrize("hardware", [("cuda", 8.0), (None, None)])
def test_preflight_llama_cpp_cpu_only_still_passes(hardware):
    """CPU builds are slow but valid; preflight must not hard-fail."""
    from bibr.local import cli

    with (
        patch("bibr.local.llm_models.detect_hardware", return_value=hardware),
        patch("bibr.local.llama_cpp.find_llama_server", return_value=["llama-server"]),
        patch("bibr.local.llama_cpp.probe_gpu_backend", return_value=False),
    ):
        assert cli._preflight_local_backend("llama-cpp") is None


def test_preflight_llama_cpp_on_a_machine_without_a_gpu_expects_the_cpu(caplog):
    """No GPU to switch to: say what the CPU costs instead of advising a CUDA build."""
    import logging

    from bibr.local import cli

    with (
        patch("bibr.local.llm_models.detect_hardware", return_value=(None, None)),
        patch("bibr.local.llama_cpp.find_llama_server", return_value=["llama-server"]),
        patch("bibr.local.llama_cpp.probe_gpu_backend", return_value=False),
        caplog.at_level(logging.INFO),
    ):
        assert cli._preflight_local_backend("llama-cpp") is None
    assert "minutes per paper" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_managed_llm_model_follows_the_llama_cpp_build():
    """dry-run and doctor name the model the server will load on a CPU build."""
    from bibr.config import GlobalSettings
    from bibr.local.cli.run_config import _managed_llm_model

    settings = GlobalSettings()
    settings.llm.local_model = None
    with patch("bibr.local.llama_cpp.runs_on_cpu", return_value=True):
        assert _managed_llm_model("llama-cpp", settings) == (
            "scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M"
        )
    with patch("bibr.local.llama_cpp.runs_on_cpu", return_value=False):
        assert _managed_llm_model("llama-cpp", settings) == (
            "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"
        )
