import tomllib
from pathlib import Path


def _pyproject() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text())


def test_vllm_extra_is_limited_to_linux_cuda_python_range():
    optional = _pyproject()["project"]["optional-dependencies"]

    assert optional["vllm"] == [
        "vllm==0.27.0; sys_platform == 'linux' and platform_machine == 'x86_64' "
        "and python_version < '3.14'",
        "openai>=2.54.0,<3; sys_platform == 'linux' and platform_machine == 'x86_64' "
        "and python_version < '3.14'",
    ]


def test_all_extra_does_not_include_platform_specific_runtimes():
    optional = _pyproject()["project"]["optional-dependencies"]

    assert optional["all"] == ["bibr[batch,cache,demo,mcp,torch]"]


def test_ml_extra_is_an_alias_for_torch():
    """`ml` is what every existing install, Dockerfile and setup plan asks for."""
    optional = _pyproject()["project"]["optional-dependencies"]

    assert optional["ml"] == ["bibr[torch]"]


def test_core_carries_the_onnx_runtime_and_no_torch():
    """The core install must be able to serve bibr's own models, torch-free."""
    project = _pyproject()["project"]
    core = {
        req.split(";")[0].split(">")[0].split("=")[0].split("[")[0].strip()
        for req in project["dependencies"]
    }

    assert {"onnxruntime", "tokenizers", "huggingface-hub", "scikit-learn", "joblib"} <= core
    assert not core & {
        "torch",
        "torchvision",
        "transformers",
        "opencv-python-headless",
        "pytorch-crf",
    }
