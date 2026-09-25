"""Managed-server tests must not depend on bibr's production ports being free.

The managed runtimes (rapid-mlx on :8772/:8773, vllm-mlx on :8767, vllm-llm,
llama.cpp) pre-flight their port with a real TCP probe
(``bibr.local.http_runtime._port_is_held``) and a real ``/v1/models`` HTTP
probe before spawning. Tests stub the subprocess but (mostly) never these
probes, so a leftover developer server — or a parallel suite run — flips
them from "spawn normally" to "port occupied" and fails the run, while a
listener that accepts but never answers stalls each test on its multi-second
HTTP timeout (x-tests-4). Force both probes off for every test under
``tests/local/``: the TCP probe returns False and each managed module's
``request_bytes`` binding raises ``LocalHttpError``, exactly what a free
port produces. Tests that stub ``request_bytes`` themselves keep their canned
responses (their patch applies after this one). The guard's own behavior
tests (``test_server_port_guard.py``) and the HTTP runtime tests
(``test_http_runtime.py``) bind ephemeral ports and exercise the real
probes, so they opt out.
"""

from __future__ import annotations

import pytest

_MANAGED_MODULES = (
    "bibr.local.rapid_mlx",
    "bibr.local.vllm_llm",
    "bibr.local.llama_cpp",
    "bibr.local.ocr",
    "bibr.local.vllm_ocr",
    "bibr.local.mlx_vlm_ocr",
)

_UNSTUBBED = ("test_server_port_guard", "test_http_runtime")


@pytest.fixture(autouse=True)
def _force_managed_port_probe_off(monkeypatch, request):
    if request.node.module.__name__.split(".")[-1] in _UNSTUBBED:
        yield
        return
    monkeypatch.setattr("bibr.local.http_runtime._port_is_held", lambda base_url: False)

    def _refused(url, **kwargs):
        from bibr.local.http_runtime import LocalHttpError

        raise LocalHttpError(f"connection refused (test hermetic stub for {url})")

    for module in _MANAGED_MODULES:
        monkeypatch.setattr(f"{module}.request_bytes", _refused)
    yield
