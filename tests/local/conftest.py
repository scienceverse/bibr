"""Managed-server tests must not depend on bibr's production ports being free.

The managed runtimes (rapid-mlx on :8772/:8773, vllm-mlx on :8767, vllm-llm,
llama.cpp) pre-flight their port with a real TCP probe
(``bibr.local.http_runtime._port_is_held``) and a real ``/v1/models`` HTTP
probe before spawning. Tests stub the subprocess but (mostly) never these
probes, so a leftover developer server — or a parallel suite run — flips
them from "spawn normally" to "port occupied" and fails the run, while a
listener that accepts but never answers stalls each test on its multi-second
HTTP timeout (x-tests-4). Force both probes off for every test under
``tests/local/`` — except the opt-in ``slow`` tests, which launch real
servers by design and must see the real probes: the TCP probe returns False
and each managed module's ``request_bytes`` binding raises
``LocalHttpError``, exactly what a free port produces. Tests that stub
``request_bytes`` themselves keep their canned responses (their patch applies
after this one). The guard's own behavior tests
(``test_server_port_guard.py``) and the HTTP runtime tests
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
def _fresh_own_server_keys(monkeypatch):
    # A server a test "starts" registers its per-launch key process-wide; a
    # later test on the same port would otherwise share that server.
    monkeypatch.setattr("bibr.local.http_runtime._OWN_SERVER_KEYS", {})


@pytest.fixture(autouse=True)
def _llama_cpp_build_sees_a_gpu(monkeypatch, request):
    # The managed llama.cpp servers ask the installed build whether it sees a
    # GPU (``runs_on_cpu``), which would run the host's llama-server and make
    # the launch flags depend on the machine. Default to a GPU build; the CPU
    # tests opt in by patching it back.
    if request.node.get_closest_marker("slow") is not None:
        return
    monkeypatch.setattr("bibr.local.llama_cpp.runs_on_cpu", lambda prefix=None: False)


@pytest.fixture(autouse=True)
def _pinned_gguf_is_cached(monkeypatch, request):
    # A launch of one of bibr's own GGUF builds fetches the file from the Hub
    # first (``download_pinned_gguf``); unit tests stand in a cached path.
    # The download tests use the real function, imported before this stub.
    if request.node.get_closest_marker("slow") is not None:
        return
    monkeypatch.setattr(
        "bibr.local.llama_cpp.download_pinned_gguf",
        lambda gguf, **_kwargs: f"/hf-cache/{gguf.repo_id}/{gguf.revision}/{gguf.filename}",
    )


@pytest.fixture(autouse=True)
def _force_managed_port_probe_off(monkeypatch, request):
    if request.node.module.__name__.split(".")[-1] in _UNSTUBBED:
        yield
        return
    if request.node.get_closest_marker("slow") is not None:
        # Opt-in slow tests launch real managed servers by design (the vLLM
        # integration test polls a real /health endpoint), so stubbing the
        # probes would make the server never look ready. They see the real
        # TCP probe and the real request_bytes.
        yield
        return
    monkeypatch.setattr("bibr.local.http_runtime._port_is_held", lambda base_url: False)

    def _refused(url, **kwargs):
        from bibr.local.http_runtime import LocalHttpError

        raise LocalHttpError(f"connection refused (test hermetic stub for {url})")

    for module in _MANAGED_MODULES:
        monkeypatch.setattr(f"{module}.request_bytes", _refused)
    yield
