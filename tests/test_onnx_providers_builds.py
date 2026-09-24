"""Tests for bibr.utils.onnx_providers — onnxruntime and onnxruntime-gpu installed together.

Both distributions write the same ``onnxruntime/`` directory. These tests mock
``sys.modules['onnxruntime']`` for the build that loads and
``importlib.metadata`` for the distributions that are installed.
"""

import logging
import re
import shlex
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bibr.utils import onnx_providers

ROOT = Path(__file__).parents[1]
LOGGER = "bibr.utils.onnx_providers"


@pytest.fixture(autouse=True)
def _fresh_build_check(monkeypatch):
    """The build check runs once per process; give each test a fresh one."""
    monkeypatch.setattr(onnx_providers, "_gpu_build_shadowed", None)


def _loaded_build(*, gpu: bool):
    """A mock onnxruntime module: the GPU build registers CUDAExecutionProvider."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    )
    return mock_ort


def _installed(monkeypatch, *, cpu="1.27.0", gpu="1.26.0", installer="uv"):
    """Stub importlib.metadata so exactly the given distributions are installed."""
    dists = {}
    if cpu:
        dists["onnxruntime"] = SimpleNamespace(version=cpu, read_text=lambda name: "uv\n")
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
