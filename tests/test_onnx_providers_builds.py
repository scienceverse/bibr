"""Tests for bibr.utils.onnx_providers — onnxruntime and onnxruntime-gpu installed together.

Both distributions write the same ``onnxruntime/`` directory, so the build that
loads is whichever wrote its files last, and uninstalling one leaves the other
empty. These tests mock ``sys.modules['onnxruntime']`` for the module that
loads and ``importlib.metadata`` for the distributions that are installed. The
``bibr doctor`` line that reports the loaded build is tested here too.
"""

import logging
import re
import shlex
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.markup import escape

from bibr.exceptions import ConfigurationError
from bibr.utils import onnx_providers

ROOT = Path(__file__).parents[1]
LOGGER = "bibr.utils.onnx_providers"


@pytest.fixture(autouse=True)
def _fresh_build_check(monkeypatch):
    """The build check runs once per process; give each test a fresh one."""
    monkeypatch.setattr(onnx_providers, "_gpu_build_shadowed", None)


def _loaded_build(*, gpu: bool, version: str = "1.26.0"):
    """A mock onnxruntime module: the GPU build registers CUDAExecutionProvider."""
    mock_ort = MagicMock()
    mock_ort.__version__ = version
    mock_ort.get_available_providers.return_value = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    )
    return mock_ort


def _installed(monkeypatch, *, cpu="1.27.0", gpu="1.26.0", installer="uv", cpu_installer="uv"):
    """Stub importlib.metadata so exactly the given distributions are installed."""
    dists = {}
    if cpu:
        dists["onnxruntime"] = SimpleNamespace(
            version=cpu,
            read_text=lambda name: f"{cpu_installer}\n" if name == "INSTALLER" else None,
        )
    if gpu:
        dists["onnxruntime-gpu"] = SimpleNamespace(
            version=gpu,
            read_text=lambda name: f"{installer}\n" if name == "INSTALLER" else None,
        )

    def distribution(name):
        if name not in dists:
            raise metadata.PackageNotFoundError(name)
        return dists[name]

    monkeypatch.setattr(metadata, "distribution", distribution)
    monkeypatch.setattr(metadata, "version", lambda name: distribution(name).version)


def _build_warnings(caplog):
    return [r.getMessage() for r in caplog.records if "are both installed" in r.getMessage()]


def _providers(*, gpu_loaded: bool, **kwargs):
    with patch.dict("sys.modules", {"onnxruntime": _loaded_build(gpu=gpu_loaded)}):
        return onnx_providers.get_ort_providers(**kwargs)


def test_cpu_build_loaded_over_installed_gpu_build_warns_once(monkeypatch, caplog):
    _installed(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        providers = _providers(gpu_loaded=False)
        _providers(gpu_loaded=False)

    assert providers == ["CPUExecutionProvider"]
    [warning] = _build_warnings(caplog)
    assert "onnxruntime 1.27.0 and onnxruntime-gpu 1.26.0 are both installed" in warning
    assert "the CPU build is the one loaded" in warning
    repair = shlex.join(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--reinstall-package",
            "onnxruntime-gpu",
            "onnxruntime-gpu[cuda,cudnn]==1.26.0",
        ]
    )
    assert warning.endswith(repair)


def test_warns_when_the_chain_does_not_request_cuda(monkeypatch, caplog):
    """The segmenter asks for CPU when the CUDA EP is missing; it still warns."""
    _installed(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _providers(gpu_loaded=False, enable_cuda=False)

    assert len(_build_warnings(caplog)) == 1


def test_pip_managed_environment_gets_the_pip_command(monkeypatch, caplog):
    _installed(monkeypatch, installer="pip")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _providers(gpu_loaded=False)

    [warning] = _build_warnings(caplog)
    repair = shlex.join(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "onnxruntime-gpu==1.26.0",
        ]
    )
    assert warning.endswith(repair)


def test_gpu_build_loaded_with_both_installed_does_not_warn(monkeypatch, caplog):
    """The state the gpu install step leaves: both installed, GPU files on disk."""
    _installed(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        providers = _providers(gpu_loaded=True)

    assert providers[0][0] == "CUDAExecutionProvider"
    assert _build_warnings(caplog) == []


@pytest.mark.parametrize(
    ("cpu", "gpu"),
    [("1.27.0", None), (None, "1.26.0")],
    ids=["onnxruntime-only", "onnxruntime-gpu-only"],
)
def test_one_distribution_installed_does_not_warn(monkeypatch, caplog, cpu, gpu):
    _installed(monkeypatch, cpu=cpu, gpu=gpu)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _providers(gpu_loaded=False)

    assert _build_warnings(caplog) == []


def test_unreadable_metadata_does_not_break_the_chain(monkeypatch, caplog):
    def broken(name):
        raise OSError("unreadable dist-info")

    monkeypatch.setattr(metadata, "version", broken)
    monkeypatch.setattr(metadata, "distribution", broken)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        providers = _providers(gpu_loaded=False)

    assert providers == ["CPUExecutionProvider"]
    assert _build_warnings(caplog) == []


def _torch_with_gpu():
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    return torch


def test_shadowed_gpu_build_is_not_reported_as_missing(monkeypatch, caplog):
    """With torch seeing a GPU, the missing-onnxruntime-gpu hint would be false here."""
    _installed(monkeypatch)
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER),
        patch.dict("sys.modules", {"torch": _torch_with_gpu()}),
    ):
        _providers(gpu_loaded=False, model_name="layout")

    assert len(_build_warnings(caplog)) == 1
    assert "is not installed" not in caplog.text


def test_missing_gpu_build_with_torch_gpu_points_at_the_gpu_extra(monkeypatch, caplog):
    _installed(monkeypatch, gpu=None)
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER),
        patch.dict("sys.modules", {"torch": _torch_with_gpu()}),
    ):
        _providers(gpu_loaded=False, model_name="layout")

    assert "onnxruntime-gpu is not installed — layout will run on CPU" in caplog.text
    assert "https://bibr.org/getting-started/install/#gpu-onnx-runtime" in caplog.text


def test_reinstall_command_forms():
    assert onnx_providers.onnxruntime_gpu_reinstall_command(
        "1.26.0", uv="/bin/uv", python="/venv/bin/python"
    ) == [
        "/bin/uv",
        "pip",
        "install",
        "--python",
        "/venv/bin/python",
        "--reinstall-package",
        "onnxruntime-gpu",
        "onnxruntime-gpu[cuda,cudnn]==1.26.0",
    ]
    assert onnx_providers.onnxruntime_gpu_reinstall_command(
        "1.26.0", uv=None, python="/venv/bin/python"
    ) == [
        "/venv/bin/python",
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        "onnxruntime-gpu==1.26.0",
    ]


@pytest.mark.parametrize(
    "path", ["pyproject.toml", "docs/getting-started/install.md", "docs/tester-guide.md"]
)
def test_documented_reinstall_pins_the_locked_gpu_build(path):
    """An unpinned reinstall upgrades onnxruntime-gpu past the lock (to a CUDA 13 build)."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    [locked] = {p["version"] for p in lock["package"] if p["name"] == "onnxruntime-gpu"}
    text = (ROOT / path).read_text(encoding="utf-8")

    assert "--reinstall-package onnxruntime-gpu" in text
    assert set(re.findall(r'"onnxruntime-gpu\[cuda,cudnn\]==([^"]+)"', text)) == {locked}


# --- An installed but empty onnxruntime --------------------------------------


def _hollow_module():
    """What ``import onnxruntime`` gives once the shared files are deleted: a
    namespace package with none of ONNX Runtime's attributes."""
    module = ModuleType("onnxruntime")
    module.__path__ = []
    return module


_UV_REPAIR = "uv sync --reinstall-package onnxruntime"


def test_hollow_module_raises_configuration_error_naming_the_repair(monkeypatch):
    _installed(monkeypatch, gpu=None)
    with (
        patch.dict("sys.modules", {"onnxruntime": _hollow_module()}),
        pytest.raises(ConfigurationError) as excinfo,
    ):
        onnx_providers.get_ort_providers(model_name="layout")

    assert "the onnxruntime package is empty" in str(excinfo.value)
    assert str(excinfo.value).endswith(_UV_REPAIR)


def test_create_session_on_a_hollow_module_names_the_model_and_repair(monkeypatch):
    _installed(monkeypatch, gpu=None)
    with (
        patch.dict("sys.modules", {"onnxruntime": _hollow_module()}),
        pytest.raises(ConfigurationError) as excinfo,
    ):
        onnx_providers.create_session("model.onnx", model_name="section classifier")

    assert str(excinfo.value).startswith("section classifier requires onnxruntime")
    assert str(excinfo.value).endswith(_UV_REPAIR)


def test_hollow_pip_install_gets_the_pinned_pip_repair(monkeypatch):
    _installed(monkeypatch, gpu=None, cpu_installer="pip")
    with (
        patch.dict("sys.modules", {"onnxruntime": _hollow_module()}),
        pytest.raises(ConfigurationError) as excinfo,
    ):
        onnx_providers.get_ort_providers()

    repair = shlex.join(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
         "onnxruntime==1.27.0"]
    )  # fmt: skip
    assert str(excinfo.value).endswith(repair)


def test_hollow_module_without_metadata_falls_back_to_the_uv_repair(monkeypatch):
    _installed(monkeypatch, cpu=None, gpu=None)
    assert onnx_providers.onnxruntime_repair_command() == _UV_REPAIR.split()


def test_hollow_module_is_no_cuda_provider_not_a_crash():
    with patch.dict("sys.modules", {"onnxruntime": _hollow_module()}):
        assert onnx_providers.cuda_provider_available() is False


# --- bibr doctor --------------------------------------------------------------


def _doctor_onnx_line(*, module) -> tuple[str, str, str]:
    from bibr.local.cli import _check_onnx_runtime

    calls = []
    with patch.dict("sys.modules", {"onnxruntime": module}):
        _check_onnx_runtime(
            lambda msg: calls.append(("ok", msg, "")),
            lambda msg, hint="": calls.append(("warn", msg, hint)),
            lambda msg, hint="": calls.append(("fail", msg, hint)),
        )
    [call] = calls
    return call


def test_doctor_reports_the_gpu_build(monkeypatch):
    _installed(monkeypatch)
    line = _doctor_onnx_line(module=_loaded_build(gpu=True))
    assert line == ("ok", "ONNX Runtime: GPU build 1.26.0", "")


def test_doctor_reports_the_cpu_build_of_a_core_install(monkeypatch):
    _installed(monkeypatch, gpu=None)
    line = _doctor_onnx_line(module=_loaded_build(gpu=False, version="1.27.0"))
    assert line == ("ok", "ONNX Runtime: CPU build 1.27.0", "")


def test_doctor_warns_when_the_cpu_build_shadows_the_gpu_build(monkeypatch, caplog):
    _installed(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        status, msg, hint = _doctor_onnx_line(module=_loaded_build(gpu=False, version="1.27.0"))

    assert status == "warn"
    assert msg.startswith("ONNX Runtime: CPU build 1.27.0 loaded")
    assert "onnxruntime-gpu 1.26.0 is installed" in msg
    repair = shlex.join(onnx_providers.onnxruntime_gpu_reinstall_command("1.26.0"))
    assert hint.endswith(escape(repair))  # Rich markup; rendering is tested below
    assert _build_warnings(caplog) == []  # the doctor line, not the log, says it


def test_doctor_fails_on_a_hollow_module_with_the_repair(monkeypatch):
    _installed(monkeypatch)
    status, msg, hint = _doctor_onnx_line(module=_hollow_module())

    assert status == "fail"
    assert "its files are missing" in msg
    assert hint.endswith(_UV_REPAIR)


def test_doctor_device_line_on_a_core_install_leaves_build_problems_to_the_onnx_line(
    monkeypatch, caplog
):
    """Without torch the device line asks ONNX Runtime; it must neither crash on
    a hollow module nor log the shadowed-build warning the ONNX line reports."""
    from bibr.local.cli import _check_device

    calls = []
    for module in (_hollow_module(), _loaded_build(gpu=False)):
        _installed(monkeypatch)
        with (
            caplog.at_level(logging.WARNING, logger=LOGGER),
            patch.dict("sys.modules", {"onnxruntime": module, "torch": None}),
        ):
            _check_device(
                lambda msg: calls.append(("ok", msg)),
                lambda msg, hint="": calls.append(("warn", msg)),
                lambda msg, hint="": calls.append(("fail", msg)),
            )

    assert calls == [("ok", "Device: cpu (ONNX Runtime; torch not installed)")] * 2
    assert _build_warnings(caplog) == []


def test_doctor_hint_keeps_the_extras_through_rich_markup(monkeypatch):
    """Rich reads ``[cuda,cudnn]`` as a style tag and drops it unless escaped."""
    from io import StringIO

    from rich.console import Console

    from bibr.local.cli import ui

    _installed(monkeypatch)
    _, _, hint = _doctor_onnx_line(module=_loaded_build(gpu=False))
    out = StringIO()
    ui.warn(Console(file=out, width=400), "ONNX Runtime", hint=hint)

    assert "'onnxruntime-gpu[cuda,cudnn]==1.26.0'" in out.getvalue()
