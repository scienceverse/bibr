import importlib
import importlib.util
import json
import pathlib
import stat
import subprocess
import sys
from unittest import mock

import pytest

_real_find_spec = importlib.util.find_spec


@pytest.fixture(autouse=True)
def vllm_mlx_importable(monkeypatch):
    """Pretend ``vllm_mlx`` is installed. These tests exercise subprocess
    plumbing with ``Popen`` mocked, so they must also run on CI (linux),
    where the darwin-arm64-only ``local-mlx`` extra is never installed and
    the constructor's pre-flight check would raise first."""
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "vllm_mlx" else _real_find_spec(name, *a, **k),
    )
    monkeypatch.setattr(
        "bibr.local.vllm_mlx_runtime.vllm_mlx_unavailable_reason",
        lambda: None,
    )


def test_vllm_mlx_routes_stderr_to_file_not_pipe(monkeypatch):
    """Server constructor must not pass stderr=PIPE — that path deadlocks
    on Apple Silicon's ~16 KB pipe buffer once the server emits enough
    request logs (vllm-mlx logs ~5 lines per OCR request via uvicorn +
    FastAPI). When the buffer fills and nothing in the parent reads it,
    the request-handler thread blocks inside a write(2) syscall and the
    next chat completion never returns.

    Fix: write stderr to a regular file so logs accumulate without
    blocking, and remain available for post-mortem inspection.
    """
    import tempfile

    from bibr.local import ocr as mod

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["kwargs"] = kwargs
        # Make the construct loop fail fast so we don't actually wait for /health.
        proc = mock.MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 1
        proc.stderr = None  # stderr now redirected to a file, not a PIPE
        return proc

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="exited during startup"):
        mod.VllmMlxServer(model="x", port=8766)

    assert captured["kwargs"]["stdout"] == subprocess.DEVNULL

    stderr = captured["kwargs"]["stderr"]
    assert stderr != subprocess.PIPE, "stderr=PIPE deadlocks the server"
    # File-like handle in the system tempdir, named after the port.
    assert hasattr(stderr, "fileno")
    log_path = pathlib.Path(stderr.name).resolve()
    expected_dir = pathlib.Path(tempfile.gettempdir()).resolve()
    assert log_path.parent == expected_dir
    assert "vllm-mlx" in log_path.name and "8766" in log_path.name
    if sys.platform != "win32":  # Windows uses ACLs, not POSIX mode bits.
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("multimodal", "continuous_batching", "expected", "unexpected"),
    [
        (True, False, "--mllm", "--continuous-batching"),
        (False, True, "--continuous-batching", "--mllm"),
    ],
)
def test_vllm_mlx_selects_engine_and_batching_mode(
    monkeypatch, multimodal, continuous_batching, expected, unexpected
):
    """OCR uses mlx-vlm; the metadata LLM uses batched mlx-lm."""
    from bibr.local import ocr as mod

    captured = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        proc = mock.MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 1
        return proc

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="exited during startup"):
        mod.VllmMlxServer(
            model="x",
            port=8766,
            multimodal=multimodal,
            continuous_batching=continuous_batching,
        )

    assert expected in captured["cmd"]
    assert unexpected not in captured["cmd"]


def test_vllm_mlx_server_rejects_stale_namespace_before_spawn(monkeypatch):
    """A leftover namespace package must not pass the vllm-mlx preflight.

    Broken reconciles can leave ``site-packages/vllm_mlx`` behind after the
    actual distribution has gone away. ``find_spec("vllm_mlx")`` is true in
    that state, but there is no runnable ``vllm_mlx.server`` module.
    """
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import ocr as mod

    monkeypatch.setattr(
        "bibr.local.vllm_mlx_runtime.vllm_mlx_unavailable_reason",
        lambda: "vllm_mlx.server was not found",
    )
    monkeypatch.setattr(
        mod.subprocess,
        "Popen",
        mock.Mock(side_effect=AssertionError("should not spawn broken vllm-mlx")),
    )

    with pytest.raises(UpstreamServiceError, match="vllm-mlx"):
        mod.VllmMlxServer(model="x", port=8766)

    mod.subprocess.Popen.assert_not_called()


def test_vllm_mlx_launcher_imports_without_legacy_mamba_cache(monkeypatch):
    """The shim should tolerate vllm-mlx versions without mamba_cache internals."""
    monkeypatch.delitem(sys.modules, "bibr.local._vllm_mlx_server", raising=False)

    module = importlib.import_module("bibr.local._vllm_mlx_server")

    assert module is not None


def test_vllm_mlx_ocr_client_construction_is_disabled():
    """glm-mlx (vllm-mlx --mllm OCR) is permanently disabled — NUL-corrupted
    text plus an uncapped vision-cache leak; see test_backend_resolution.py.
    """
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import ocr as mod

    with pytest.raises(UpstreamServiceError, match="glm-rapid-mlx"):
        mod.VllmMlxOcrClient()


def test_vllm_mlx_timeout_path_reraises_timeout_even_if_shutdown_fails(monkeypatch):
    """The original TimeoutError must surface even when cleanup raises."""
    from bibr.config import Settings
    from bibr.local import ocr as mod

    monkeypatch.setattr(Settings.vllm_mlx, "startup_timeout", 0)  # force timeout

    fake_proc = mock.MagicMock()
    fake_proc.poll.return_value = None  # still running

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: fake_proc)

    def boom_shutdown(self):
        raise RuntimeError("cleanup boom")

    monkeypatch.setattr(mod.VllmMlxServer, "shutdown", boom_shutdown)

    with pytest.raises(TimeoutError):
        mod.VllmMlxServer(model="x", port=8766)


def test_vllm_mlx_waits_for_model_loaded_then_warms_up(monkeypatch):
    """Server must wait for model_loaded=true and POST a warmup before returning."""
    from bibr.local import ocr as mod

    fake_proc = mock.MagicMock()
    fake_proc.poll.return_value = None  # still running
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: fake_proc)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    bodies = [
        {"model_loaded": False},
        {"model_loaded": False},
        {"model_loaded": True},
    ]
    requests_seen: list[str] = []

    def fake_request_bytes(url, **_kwargs):
        requests_seen.append(url)
        if "/health" in url:
            return 200, "OK", json.dumps(bodies.pop(0)).encode("utf-8")
        return 200, "OK", b'{"choices":[{"message":{"content":""}}]}'

    monkeypatch.setattr(mod, "request_bytes", fake_request_bytes)

    server = mod.VllmMlxServer(model="m", port=8766)

    health_calls = [u for u in requests_seen if "/health" in u]
    warmup_calls = [u for u in requests_seen if "chat/completions" in u]
    assert len(health_calls) == 3, "should poll until model_loaded=true"
    assert len(warmup_calls) == 1, "warmup must run exactly once after readiness"
    assert server.loaded


@pytest.mark.parametrize("multimodal", [True, False])
def test_vllm_mlx_warmup_payload_matches_engine_mode(monkeypatch, multimodal):
    """A text-only (multimodal=False) server must warm up with a text-only
    payload, never an image. Sending an image to a server started without
    ``--mllm`` (no vision encoder loaded) previously hung the warmup request
    for the full ``warmup_timeout`` (300s) instead of failing fast or
    succeeding quickly — every ``--llm local`` run paid that tax before any
    real inference happened.
    """
    from bibr.local import ocr as mod

    fake_proc = mock.MagicMock()
    fake_proc.poll.return_value = None
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: fake_proc)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    captured_payload = {}

    def fake_request_bytes(url, *, body=None, **_kwargs):
        if url.endswith("/v1/models"):
            # Pre-spawn port guard probe — nothing is listening yet.
            raise mod.LocalHttpError("connection refused")
        if "/health" in url:
            return 200, "OK", json.dumps({"model_loaded": True}).encode("utf-8")
        assert body is not None
        captured_payload["body"] = json.loads(body.decode("utf-8"))
        return 200, "OK", b'{"choices":[{"message":{"content":""}}]}'

    monkeypatch.setattr(mod, "request_bytes", fake_request_bytes)

    mod.VllmMlxServer(model="m", port=8766, multimodal=multimodal)

    content = captured_payload["body"]["messages"][0]["content"]
    if multimodal:
        assert isinstance(content, list)
        assert any(part.get("type") == "image_url" for part in content)
    else:
        assert isinstance(content, str), "text-only server must not receive an image payload"


def test_startup_keyboard_interrupt_shuts_down_child(monkeypatch):
    """Ctrl-C during the health wait still shuts down the child (11/13)."""
    from bibr.local import ocr as mod
    from bibr.local.http_runtime import LocalHttpError

    proc = mock.MagicMock()
    proc.poll.return_value = None
    popens = []
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: popens.append((a, k)) or proc)

    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            # Pre-spawn port guard probe: nothing listening yet.
            raise LocalHttpError("connection refused")
        raise KeyboardInterrupt

    monkeypatch.setattr(mod, "request_bytes", fake_request)

    with pytest.raises(KeyboardInterrupt):
        mod.VllmMlxServer(model="x", port=8765)
    assert len(popens) == 1  # the interrupt hit the health wait, not the guard
    proc.terminate.assert_called_once()


def test_startup_error_reports_tail_end(tmp_path):
    """The exit error keeps the LAST 500 chars (the OOM line), not the first (7)."""
    from bibr.config import GlobalSettings
    from bibr.local import ocr as mod

    process = mock.MagicMock(returncode=1)
    process.poll.return_value = 1
    server = mod.VllmMlxServer.__new__(mod.VllmMlxServer)
    server._settings = GlobalSettings()
    server._model = "x"
    server._port = 8766
    server._process = process
    server._stderr_fh = None
    log = tmp_path / "vllm-mlx.log"
    log.write_bytes(
        b"\n".join(
            f"INFO loading weights shard {i:3d}/200 into unified memory".encode()
            for i in range(200)
        )
        + b"\nRuntimeError: Metal OOM: out of memory mapping weights into MTLBuffer\n"
    )
    server._stderr_log = log

    with pytest.raises(RuntimeError) as excinfo:
        server._wait_until_ready()

    assert "Metal OOM" in str(excinfo.value)
    assert "shard   0/200" not in str(excinfo.value)
