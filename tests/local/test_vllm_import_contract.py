"""Exercise optional vLLM imports with the real installed SDK dependency set."""

import importlib.util
import subprocess
import sys

import pytest


@pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="optional vLLM extra")
def test_managed_vllm_tool_parser_imports_without_loading_a_model():
    # Package metadata alone missed vLLM 0.27's dependency on NamespaceTool.
    # A subprocess contains its heavyweight imports without requiring a GPU.
    result = subprocess.run(
        [sys.executable, "-c", "import vllm.tool_parsers.utils"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
