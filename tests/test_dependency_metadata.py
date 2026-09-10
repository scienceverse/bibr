import tomllib
from pathlib import Path


def _pyproject() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text())


def test_vllm_extra_is_limited_to_linux_cuda_python_range():
    optional = _pyproject()["project"]["optional-dependencies"]

    assert optional["vllm"] == [
        "vllm==0.25.1; sys_platform == 'linux' and platform_machine == 'x86_64' "
        "and python_version < '3.14'"
    ]


def test_all_extra_does_not_include_platform_specific_runtimes():
    optional = _pyproject()["project"]["optional-dependencies"]

    assert optional["all"] == ["bibr[batch,cache,demo,mcp,ml]"]
