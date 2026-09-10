"""Slow smoke: launch the managed vLLM server against a tiny model.

Requires the vllm extra + an NVIDIA GPU; skipped everywhere else.
"""

import importlib.util

import pytest

pytestmark = pytest.mark.slow

vllm_missing = importlib.util.find_spec("vllm") is None


@pytest.mark.skipif(vllm_missing, reason="vllm extra not installed")
def test_vllm_server_launch_and_shutdown():
    from bibr.local.llm_models import detect_hardware
    from bibr.local.vllm_llm import VllmLlmServer

    platform_key, _ = detect_hardware()
    if platform_key != "cuda":
        pytest.skip("no NVIDIA GPU")

    server = VllmLlmServer(model="Qwen/Qwen2.5-0.5B-Instruct", port=18999)
    try:
        assert server.base_url == "http://localhost:18999"
        from bibr.local.http_runtime import request_bytes

        status, _reason, _body = request_bytes(server.base_url + "/health", timeout=5)
        assert status == 200
    finally:
        server.shutdown()
