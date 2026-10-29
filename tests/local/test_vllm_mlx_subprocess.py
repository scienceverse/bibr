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


def _vllm_mlx_launch_cmd(monkeypatch, model: str, *, multimodal: bool = False) -> list[str]:
    """The command a vllm-mlx start runs; a text LLM's (``VllmMlxLlmServer``) by default."""
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
        mod.VllmMlxServer(model=model, port=8767, continuous_batching=True, multimodal=multimodal)
    return captured["cmd"]


def test_vllm_mlx_serves_a_pinned_build_from_its_snapshot_under_the_hub_id(monkeypatch):
    """mlx-lm reads a repository's main: the pinned build is downloaded first
    (bibr.local.hub_pins; stubbed by tests/conftest.py) and served by path,
    under the Hub id the requests and the port guard use."""
    model = "scienceverse/bibr-extract-qwen35-4b-mlx-8bit"

    cmd = _vllm_mlx_launch_cmd(monkeypatch, model)

    assert cmd[cmd.index("--model") + 1] == (
        "/nonexistent-hf-cache/models--scienceverse--bibr-extract-qwen35-4b-mlx-8bit/"
        "snapshots/432b8cd80ea920919ef26c48401473c6d8c8a3e1"
    )
    assert cmd[cmd.index("--served-model-name") + 1] == model
    # The shim judges the snapshot by that Hub id: text, as served by id.
    assert "--mllm" not in cmd
    assert "--no-mllm" not in cmd


def test_vllm_mlx_passes_unpinned_ids_unchanged(monkeypatch):
    cmd = _vllm_mlx_launch_cmd(monkeypatch, "scienceverse/bibr-extract-qwen35-2b-mlx-8bit")

    assert cmd[cmd.index("--model") + 1] == "scienceverse/bibr-extract-qwen35-2b-mlx-8bit"
    assert "--served-model-name" not in cmd
    assert "--no-mllm" not in cmd


def test_vllm_mlx_serves_a_local_model_directory_as_text(monkeypatch, tmp_path):
    """LLM_LOCAL_MODEL=<dir> (such as the 4B's previous MLX build): vllm-mlx
    would judge the directory by its config.json, so the shim is told it is a
    text LLM."""
    cmd = _vllm_mlx_launch_cmd(monkeypatch, str(tmp_path))

    assert cmd[cmd.index("--model") + 1] == str(tmp_path)
    assert "--no-mllm" in cmd
    assert "--served-model-name" not in cmd
    assert "--mllm" not in cmd


def test_vllm_mlx_multimodal_start_keeps_mllm_for_a_local_directory(monkeypatch, tmp_path):
    cmd = _vllm_mlx_launch_cmd(monkeypatch, str(tmp_path), multimodal=True)

    assert "--mllm" in cmd
    assert "--no-mllm" not in cmd


@pytest.mark.parametrize(
    ("argv", "name", "rest"),
    [
        (
            ["prog", "--model", "/snap", "--served-model-name", "org/m"],
            "org/m",
            ["prog", "--model", "/snap"],
        ),
        (["prog", "--served-model-name=org/m", "--port", "1"], "org/m", ["prog", "--port", "1"]),
        (["prog", "--model", "org/m"], None, ["prog", "--model", "org/m"]),
    ],
)
def test_vllm_mlx_shim_takes_the_served_model_name(argv, name, rest):
    from bibr.local import _vllm_mlx_server as shim

    assert shim._pop_served_model_name(argv) == name
    assert argv == rest


def _fake_vllm_mlx_server(monkeypatch):
    """A stand-in ``vllm_mlx.server`` whose main() loads the model as 0.4.x does."""
    import types

    loads: list[tuple[tuple, dict]] = []
    server = types.ModuleType("vllm_mlx.server")

    def load_model(model_name, served_model_name=None, **kwargs):
        loads.append(((model_name,), {"served_model_name": served_model_name, **kwargs}))

    def main():
        server.load_model("/snap", use_batching=True)

    server.load_model = load_model
    server.main = main
    package = types.ModuleType("vllm_mlx")
    package.server = server
    monkeypatch.setitem(sys.modules, "vllm_mlx", package)
    monkeypatch.setitem(sys.modules, "vllm_mlx.server", server)
    return loads


def test_vllm_mlx_shim_serves_the_model_under_the_given_name(monkeypatch):
    from bibr.local import _vllm_mlx_server as shim

    loads = _fake_vllm_mlx_server(monkeypatch)
    argv = ["prog", "--model", "/snap", "--served-model-name", "org/m", "--port", "8767"]

    shim._main(argv)

    assert loads == [(("/snap",), {"served_model_name": "org/m", "use_batching": True})]
    # vllm_mlx.server's own argparse sees no --served-model-name.
    assert argv == ["prog", "--model", "/snap", "--port", "8767"]


def test_vllm_mlx_shim_without_a_served_name_runs_the_server_module(monkeypatch):
    from bibr.local import _vllm_mlx_server as shim

    _fake_vllm_mlx_server(monkeypatch)
    ran = []
    monkeypatch.setattr(
        shim.runpy, "run_module", lambda name, run_name: ran.append((name, run_name))
    )

    shim._main(["prog", "--model", "org/m"])

    assert ran == [("vllm_mlx.server", "__main__")]


# --- vllm-mlx's multimodal detection (is_mllm_model) -------------------------
#
# vllm-mlx is not installed on Linux CI, so these run the shim against stub
# modules laid out as vllm-mlx 0.4.0 / 0.4.1 are: api.utils defines
# is_mllm_model (a local directory judged by its config.json, a Hub id by its
# name), and the engines, the server and the model registry each import their
# own reference to it.

_STUB_API_UTILS = """
import json
from pathlib import Path


def is_mllm_model(model_name):
    config = Path(model_name) / "config.json"
    if Path(model_name).is_dir() and config.is_file():
        return "image_token_id" in json.loads(config.read_text())
    return "Qwen3.5-" in model_name


is_vlm_model = is_mllm_model
"""

_STUB_ENGINE = """
from vllm_mlx.api.utils import is_mllm_model


class Engine:
    def __init__(self, model_name, force_mllm=False):
        self.is_mllm = force_mllm or is_mllm_model(model_name)
"""

_STUB_SERVER = """
import argparse
import sys

from vllm_mlx.api.utils import is_mllm_model
from vllm_mlx.engine import batched, simple

loaded = []


def load_model(model_name, use_batching=False, force_mllm=False, served_model_name=None):
    engine = (batched if use_batching else simple).Engine(model_name, force_mllm=force_mllm)
    loaded.append(
        {
            "model_name": served_model_name or model_name,
            "is_mllm": engine.is_mllm,
            # what /v1/models reports, judged on the model path
            "reported_mllm": force_mllm or is_mllm_model(model_name),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--mllm", action="store_true")
    parser.add_argument("--continuous-batching", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    load_model(args.model, use_batching=args.continuous_batching, force_mllm=args.mllm)
"""

_PINNED_MLX = "scienceverse/bibr-extract-qwen35-4b-mlx-8bit"


def _stub_vllm_mlx(monkeypatch, extra: dict[str, str] | None = None):
    """Install the stub ``vllm_mlx`` package (plus *extra* modules); return its server."""
    import types

    sources = {
        "vllm_mlx": "",
        "vllm_mlx.engine": "",
        "vllm_mlx.api.utils": _STUB_API_UTILS,
        "vllm_mlx.api": "from vllm_mlx.api.utils import is_mllm_model, is_vlm_model\n",
        "vllm_mlx.engine.simple": _STUB_ENGINE,
        "vllm_mlx.engine.batched": _STUB_ENGINE,
        "vllm_mlx.model_registry": "from vllm_mlx.api.utils import is_mllm_model\n",
        "vllm_mlx.server": _STUB_SERVER,
        **(extra or {}),
    }
    for name in sorted(sources, key=lambda n: n.count(".")):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    for name, source in sources.items():
        # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
        exec(compile(source, f"<stub {name}>", "exec"), sys.modules[name].__dict__)  # noqa: S102
    return sys.modules["vllm_mlx.server"]


def _qwen35_build(tmp_path, name: str = "snapshot") -> str:
    """A local MLX build whose config.json has image_token_id, as bibr's Qwen3.5 builds' do."""
    build = tmp_path / name
    build.mkdir()
    (build / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "image_token_id": 7}))
    return str(build)


def _run_shim(monkeypatch, *args: str) -> list[str]:
    """Run the shim as ``python -m`` would; return the argv vllm-mlx's server parsed."""
    from bibr.local import _vllm_mlx_server as shim

    monkeypatch.setattr(sys, "argv", ["prog", *args])
    shim._main(sys.argv)
    return sys.argv


def test_stub_vllm_mlx_loads_a_qwen35_snapshot_as_multimodal(monkeypatch, tmp_path):
    """The failure the shim works around: vllm-mlx alone picks mlx-vlm for the snapshot."""
    server = _stub_vllm_mlx(monkeypatch)
    snapshot = _qwen35_build(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "--model", snapshot])

    server.main()

    assert server.loaded[0]["is_mllm"] is True
    assert sys.modules["vllm_mlx.api.utils"].is_mllm_model(_PINNED_MLX) is False


@pytest.mark.parametrize("batching", [[], ["--continuous-batching"]])
def test_vllm_mlx_shim_judges_a_snapshot_by_its_served_name(monkeypatch, tmp_path, batching):
    server = _stub_vllm_mlx(monkeypatch)
    snapshot = _qwen35_build(tmp_path)

    argv = _run_shim(
        monkeypatch,
        "--model",
        snapshot,
        "--port",
        "8767",
        "--served-model-name",
        _PINNED_MLX,
        *batching,
    )

    assert server.loaded == [{"model_name": _PINNED_MLX, "is_mllm": False, "reported_mllm": False}]
    assert argv == ["prog", "--model", snapshot, "--port", "8767", *batching]


def test_vllm_mlx_shim_judges_by_the_served_name_both_ways(monkeypatch, tmp_path):
    """A served name vllm-mlx reads as multimodal keeps the snapshot multimodal."""
    server = _stub_vllm_mlx(monkeypatch)
    snapshot = _qwen35_build(tmp_path)

    _run_shim(monkeypatch, "--model", snapshot, "--served-model-name", "org/Qwen3.5-9B")

    assert server.loaded[0]["is_mllm"] is True


def test_vllm_mlx_shim_serves_a_local_directory_as_text_with_no_mllm(monkeypatch, tmp_path):
    """The previous revision's MLX build from a local directory (LLM_LOCAL_MODEL=<dir>)."""
    server = _stub_vllm_mlx(monkeypatch)
    build = _qwen35_build(tmp_path, "bibr-4b-mlx-6ab47a4e")

    argv = _run_shim(monkeypatch, "--model", build, "--port", "8767", "--no-mllm")

    assert server.loaded == [{"model_name": build, "is_mllm": False, "reported_mllm": False}]
    # --no-mllm is the shim's: vllm-mlx's own argparse never sees it.
    assert argv == ["prog", "--model", build, "--port", "8767"]


@pytest.mark.parametrize(
    "flags", [["--mllm", "--no-mllm"], ["--served-model-name", _PINNED_MLX, "--mllm"]]
)
def test_vllm_mlx_shim_leaves_detection_alone_with_mllm(monkeypatch, tmp_path, flags):
    """--mllm forces mlx-vlm and wins over --no-mllm (without the shim's flags,
    --mllm alone runs vllm_mlx.server as before)."""
    server = _stub_vllm_mlx(monkeypatch)
    original = sys.modules["vllm_mlx.api.utils"].is_mllm_model
    snapshot = _qwen35_build(tmp_path)

    _run_shim(monkeypatch, "--model", snapshot, *flags)

    assert server.loaded[0]["is_mllm"] is True
    assert sys.modules["vllm_mlx.engine.simple"].is_mllm_model is original


def test_vllm_mlx_shim_keeps_vllm_mlx_answers_for_other_models(monkeypatch, tmp_path):
    from bibr.local import _vllm_mlx_server as shim

    _stub_vllm_mlx(monkeypatch)
    snapshot = _qwen35_build(tmp_path)
    other = _qwen35_build(tmp_path, "other")

    shim._judge_launch_model(snapshot, _PINNED_MLX)

    detect = sys.modules["vllm_mlx.engine.batched"].is_mllm_model
    assert detect(snapshot) is False
    assert detect(str(pathlib.Path(snapshot) / ".." / "snapshot")) is False
    assert detect(other) is True
    assert detect("org/Qwen3.5-9B") is True
    assert detect("org/text-model") is False


def test_vllm_mlx_shim_judges_every_copy_of_is_mllm_model(monkeypatch, tmp_path):
    """Fails when a copy of is_mllm_model in a vllm_mlx module still reads the
    snapshot's config.json: one more module importing it, one defining its
    own, one holding it under another name, or one imported afterwards."""
    import types

    from bibr.local import _vllm_mlx_server as shim

    extra = {
        "vllm_mlx.engine.hybrid": (
            "from pathlib import Path\n\n\n"
            "def is_mllm_model(model_name):\n"
            "    return (Path(model_name) / 'config.json').is_file()\n"
        ),
        "vllm_mlx.engine.alias": "from vllm_mlx.api.utils import is_mllm_model as detect\n",
    }
    _stub_vllm_mlx(monkeypatch, extra)
    snapshot = _qwen35_build(tmp_path)

    replaced = shim._judge_launch_model(snapshot, _PINNED_MLX)
    later = types.ModuleType("vllm_mlx.engine.later")
    monkeypatch.setitem(sys.modules, later.__name__, later)
    # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
    exec("from vllm_mlx.api.utils import is_mllm_model", later.__dict__)  # noqa: S102

    copies = {
        f"{name}.{attr}": value
        for name, module in list(sys.modules.items())
        if name == "vllm_mlx" or name.startswith("vllm_mlx.")
        for attr, value in vars(module).items()
        if getattr(value, "__name__", None) == "is_mllm_model"
    }
    assert {
        f"vllm_mlx.{name}"
        for name in (
            "api.utils.is_mllm_model",
            "api.utils.is_vlm_model",
            "api.is_mllm_model",
            "api.is_vlm_model",
            "engine.simple.is_mllm_model",
            "engine.batched.is_mllm_model",
            "model_registry.is_mllm_model",
            "server.is_mllm_model",
            "engine.hybrid.is_mllm_model",
            "engine.alias.detect",
            "engine.later.is_mllm_model",
        )
    } <= set(copies)
    assert sorted(name for name, detect in copies.items() if detect(snapshot)) == []
    assert set(replaced) == set(copies) - {"vllm_mlx.engine.later.is_mllm_model"}
    # Judging twice wraps nothing twice.
    assert shim._judge_launch_model(snapshot, _PINNED_MLX) == []


def test_vllm_mlx_shim_imports_the_modules_holding_the_detector(monkeypatch, tmp_path):
    """A detector module not imported yet is imported, then judged; one this
    vllm-mlx lacks is skipped, but a missing dependency still fails."""
    from bibr.local import _vllm_mlx_server as shim

    _stub_vllm_mlx(monkeypatch)
    registry = sys.modules["vllm_mlx.model_registry"]
    monkeypatch.delitem(sys.modules, "vllm_mlx.model_registry")
    imported = []
    real_import = importlib.import_module

    def import_module(name):
        imported.append(name)
        if name == "vllm_mlx.model_registry":
            sys.modules[name] = registry
        return real_import(name)

    monkeypatch.setattr(shim.importlib, "import_module", import_module)
    snapshot = _qwen35_build(tmp_path)

    assert "vllm_mlx.model_registry.is_mllm_model" in shim._judge_launch_model(snapshot, None)
    assert imported == list(shim._MLLM_DETECTOR_MODULES)

    def missing(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(shim.importlib, "import_module", missing)
    assert shim._judge_launch_model(snapshot, None) == []

    def missing_dependency(name):
        raise ModuleNotFoundError("No module named 'mlx'", name="mlx")

    monkeypatch.setattr(shim.importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="mlx"):
        shim._judge_launch_model(snapshot, None)
