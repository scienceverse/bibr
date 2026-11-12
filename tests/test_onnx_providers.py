"""Tests for bibr.utils.onnx_providers — provider chain construction."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from bibr.utils import onnx_providers


@pytest.fixture(autouse=True)
def _fresh_cuda_preload(monkeypatch):
    """The CUDA library preload runs once per process; give each test a fresh one."""
    monkeypatch.setattr(onnx_providers, "_cuda_libraries_preloaded", False)


def _cuda_build_ort():
    """A mock onnxruntime whose build includes the CUDA provider (onnxruntime-gpu)."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return mock_ort


def test_cpu_only_provider_chain():
    """When no GPU EPs are available, returns CPUExecutionProvider only."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import get_ort_providers

        providers = get_ort_providers(enable_cuda=False)

    assert providers == ["CPUExecutionProvider"]


def test_cuda_provider_chain():
    """When CUDA EP is available and enabled, it appears before CPU with options."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import get_ort_providers

        providers = get_ort_providers(enable_cuda=True)

    assert len(providers) == 2
    assert isinstance(providers[0], tuple)
    assert providers[0][0] == "CUDAExecutionProvider"
    cuda_opts = providers[0][1]
    assert cuda_opts["arena_extend_strategy"] == "kSameAsRequested"
    assert cuda_opts["do_copy_in_default_stream"] is True
    assert providers[1] == "CPUExecutionProvider"


def test_cuda_with_gpu_mem_limit():
    """gpu_mem_limit is forwarded into CUDA EP options when set."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import get_ort_providers

        providers = get_ort_providers(enable_cuda=True, gpu_mem_limit=1024 * 1024 * 1024)

    assert providers[0][1]["gpu_mem_limit"] == 1024 * 1024 * 1024


def test_coreml_included_only_when_asked_for():
    """CoreMLExecutionProvider exists on macOS, but a chain has it only on request."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import get_ort_providers

        default = get_ort_providers(enable_cuda=False)
        asked_for = get_ort_providers(enable_cuda=False, enable_coreml=True)

    assert default == ["CPUExecutionProvider"]
    assert asked_for == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_cuda_provider_available_true():
    """True when onnxruntime exposes CUDAExecutionProvider in this process."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import cuda_provider_available

        assert cuda_provider_available() is True


def test_cuda_provider_available_false():
    """False when only the CPU EP is available."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import cuda_provider_available

        assert cuda_provider_available() is False


def test_cuda_provider_available_handles_missing_onnxruntime():
    """A missing/broken onnxruntime must degrade to False, not raise."""
    with patch.dict("sys.modules", {"onnxruntime": None}):
        from bibr.utils.onnx_providers import cuda_provider_available

        assert cuda_provider_available() is False


def test_cuda_chain_preloads_cuda_libraries_once():
    """The CUDA provider cannot find onnxruntime-gpu's pip-installed CUDA and
    cuDNN libraries until preload_dlls() loads them; a CUDA chain calls it once."""
    mock_ort = _cuda_build_ort()
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        onnx_providers.get_ort_providers(enable_cuda=True)
        onnx_providers.get_ort_providers(enable_cuda=True)

    mock_ort.preload_dlls.assert_called_once_with()


def test_cpu_chain_skips_the_cuda_preload():
    mock_ort = _cuda_build_ort()
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        onnx_providers.get_ort_providers(enable_cuda=False)

    mock_ort.preload_dlls.assert_not_called()


def test_preload_messages_go_to_the_log_not_stdout(capsys, caplog):
    """preload_dlls() print()s the libraries it could not load."""
    mock_ort = _cuda_build_ort()
    mock_ort.preload_dlls.side_effect = lambda: print(
        "Failed to load libcudnn.so.9: cannot open shared object file\n"
        "Please follow https://onnxruntime.ai/docs/install/#cuda-and-cudnn to install CUDA and CuDNN."
    )
    with (
        caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"),
        patch.dict("sys.modules", {"onnxruntime": mock_ort}),
    ):
        providers = onnx_providers.get_ort_providers(enable_cuda=True)

    assert capsys.readouterr().out == ""
    assert "Failed to load libcudnn.so.9" in caplog.text
    assert providers[0][0] == "CUDAExecutionProvider"


def test_preload_error_does_not_break_the_chain(caplog):
    mock_ort = _cuda_build_ort()
    mock_ort.preload_dlls.side_effect = RuntimeError("Invalid parameter of directory")
    with (
        caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"),
        patch.dict("sys.modules", {"onnxruntime": mock_ort}),
    ):
        providers = onnx_providers.get_ort_providers(enable_cuda=True)

    assert providers[0][0] == "CUDAExecutionProvider"
    assert "preload_dlls() failed: Invalid parameter of directory" in caplog.text


def test_onnxruntime_without_preload_dlls_still_builds_the_chain():
    """onnxruntime < 1.21 has no preload_dlls()."""
    mock_ort = _cuda_build_ort()
    del mock_ort.preload_dlls
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers = onnx_providers.get_ort_providers(enable_cuda=True)

    assert providers[0][0] == "CUDAExecutionProvider"


def _open_session(session_providers, **kwargs):
    mock_ort = _cuda_build_ort()
    mock_ort.InferenceSession.return_value.get_providers.return_value = session_providers
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        return onnx_providers.create_session("model.onnx", model_name="layout", **kwargs)


def test_create_session_reports_cpu_when_the_session_dropped_cuda(caplog):
    """ORT runs the session on CPU when the CUDA provider fails to start;
    create_session reports that device, not the CUDA it requested."""
    with caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"):
        _, device = _open_session(["CPUExecutionProvider"])

    assert device == "cpu"
    assert "layout requested CUDA" in caplog.text


def test_create_session_reports_cuda_when_the_session_has_it(caplog):
    with caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"):
        _, device = _open_session(["CUDAExecutionProvider", "CPUExecutionProvider"])

    assert device == "cuda"
    assert "requested CUDA" not in caplog.text


def test_create_session_on_cpu_request_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"):
        _, device = _open_session(["CPUExecutionProvider"], device="cpu")

    assert device == "cpu"
    assert "requested CUDA" not in caplog.text


def _open_cuda_session(device_id: str = "0"):
    mock_ort = _cuda_build_ort()
    opened = mock_ort.InferenceSession.return_value
    opened.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    opened.get_provider_options.return_value = {
        "CUDAExecutionProvider": {"device_id": device_id},
        "CPUExecutionProvider": {},
    }
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        session, device = onnx_providers.create_session("model.onnx", model_name="ner")
    return mock_ort, opened, session, device


def test_cuda_session_runs_free_unused_arena_memory():
    """ORT's CUDA arena keeps every block it allocates; with input shapes that
    change from call to call it grows until the GPU is full. Every run of a CUDA
    session ends with arena shrinkage, on the session's own device."""
    mock_ort, opened, session, device = _open_cuda_session(device_id="1")

    session.run(["emissions"], {"input_ids": 1})

    assert device == "cuda"
    mock_ort.RunOptions.return_value.add_run_config_entry.assert_called_once_with(
        "memory.enable_memory_arena_shrinkage", "gpu:1"
    )
    opened.run.assert_called_once_with(
        ["emissions"], {"input_ids": 1}, mock_ort.RunOptions.return_value
    )


def test_cuda_session_wrapper_passes_everything_else_through():
    _, opened, session, _ = _open_cuda_session()

    assert session.get_inputs() is opened.get_inputs.return_value
    assert session.get_providers()[0] == "CUDAExecutionProvider"


def test_cpu_session_is_returned_as_opened():
    """Shrinkage is only asked for a CUDA arena; ORT rejects it for a device the
    session has no arena on."""
    mock_ort = _cuda_build_ort()
    opened = mock_ort.InferenceSession.return_value
    opened.get_providers.return_value = ["CPUExecutionProvider"]
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        session, device = onnx_providers.create_session("model.onnx", device="cpu")

    assert device == "cpu"
    assert session is opened
    mock_ort.RunOptions.assert_not_called()


def test_arena_shrinkage_run_option_is_accepted_by_onnxruntime():
    """The config key and value format are onnxruntime's own; a real RunOptions takes them."""
    ort = pytest.importorskip("onnxruntime")
    opened = MagicMock()
    opened.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    opened.get_provider_options.return_value = {"CUDAExecutionProvider": {"device_id": "0"}}

    session = onnx_providers.shrink_arena_after_runs(opened)
    session.run(None, {})

    run_options = opened.run.call_args.args[2]
    assert isinstance(run_options, ort.RunOptions)
    assert run_options.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "gpu:0"


def _cpu_only_ort_with_recording_options():
    """A mock onnxruntime exposing only the CPU provider with a SessionOptions
    stand-in that records attribute writes (a MagicMock would auto-create any
    attribute on read, so absence could not be asserted)."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CPUExecutionProvider"]
    recorded: dict = {}

    class _RecordingOptions:
        def __setattr__(self, name, value):
            recorded[name] = value

        def add_session_config_entry(self, key, value):
            recorded[f"config:{key}"] = value

    mock_ort.SessionOptions = _RecordingOptions
    return mock_ort, recorded


def test_create_session_disables_cpu_arena_for_cpu_only_when_asked():
    """CPU-only sessions pass enable_cpu_mem_arena=False: the CPU arena keeps
    its peak allocation for the session's life (layout ~5 GB, SaT ~2.8 GB)."""
    mock_ort, recorded = _cpu_only_ort_with_recording_options()
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        _, device = onnx_providers.create_session(
            "model.onnx", device="cpu", model_name="layout", disable_cpu_arena=True
        )

    assert device == "cpu"
    assert recorded.get("enable_cpu_mem_arena") is False


def test_create_session_keeps_cpu_arena_by_default():
    """Without the opt-in, CPU sessions keep ORT's default arena behavior."""
    mock_ort, recorded = _cpu_only_ort_with_recording_options()
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        _, device = onnx_providers.create_session("model.onnx", device="cpu", model_name="layout")

    assert device == "cpu"
    assert "enable_cpu_mem_arena" not in recorded


def test_create_session_cpu_arena_opt_in_is_noop_for_cuda():
    """The flag must not change CUDA sessions: the CUDA EP manages its own
    arena and disabling the CPU arena there is unverified."""
    mock_ort = _cuda_build_ort()
    recorded: dict = {}

    class _RecordingOptions:
        def __setattr__(self, name, value):
            recorded[name] = value

        def add_session_config_entry(self, key, value):
            recorded[f"config:{key}"] = value

    mock_ort.SessionOptions = _RecordingOptions
    mock_ort.InferenceSession.return_value.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        _, device = onnx_providers.create_session(
            "model.onnx", device="cuda", model_name="layout", disable_cpu_arena=True
        )

    assert device == "cuda"
    assert "enable_cpu_mem_arena" not in recorded


def test_cpu_ort_session_options_disables_the_arena():
    """The wtpsplit ort_kwargs helper carries enable_cpu_mem_arena=False."""
    ort = pytest.importorskip("onnxruntime")

    options = onnx_providers.cpu_ort_session_options()

    assert isinstance(options, ort.SessionOptions)
    assert options.enable_cpu_mem_arena is False


def test_cpu_ort_session_options_is_none_without_onnxruntime():
    """Without onnxruntime there is nothing to build options for (SaT skips
    ort_kwargs then)."""
    with patch.dict("sys.modules", {"onnxruntime": None}):
        assert onnx_providers.cpu_ort_session_options() is None


_SESSION_ARENA_OOM = (
    "[ONNXRuntimeError] : 1 : FAIL : Exception during initialization: "
    "/onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:358 void* "
    "onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) "
    "Failed to allocate memory for requested buffer of size 384003072"
)


def _ort_whose_sessions_fail(message: str, *, fail_on: str = "CUDAExecutionProvider"):
    """A mock CUDA-build onnxruntime whose sessions with *fail_on* in their chain raise *message*."""
    mock_ort = _cuda_build_ort()
    opened: list[dict] = []

    class _RecordingOptions:
        def __init__(self):
            object.__setattr__(self, "written", {})

        def __setattr__(self, name, value):
            self.written[name] = value

        def add_session_config_entry(self, key, value):
            self.written[f"config:{key}"] = value

    cpu_session = MagicMock()
    cpu_session.get_providers.return_value = ["CPUExecutionProvider"]

    def open_session(path, sess_options=None, providers=None):
        names = [p[0] if isinstance(p, tuple) else p for p in providers]
        opened.append({"providers": names, "options": dict(sess_options.written)})
        if fail_on in names:
            raise RuntimeError(message)
        return cpu_session

    mock_ort.SessionOptions = _RecordingOptions
    mock_ort.InferenceSession.side_effect = open_session
    return mock_ort, opened, cpu_session


@pytest.mark.parametrize("disable_cpu_arena", [True, False])
def test_create_session_opens_on_cpu_when_the_cuda_session_is_out_of_memory(
    caplog, disable_cpu_arena
):
    mock_ort, opened, cpu_session = _ort_whose_sessions_fail(_SESSION_ARENA_OOM)
    with (
        patch.dict("sys.modules", {"onnxruntime": mock_ort}),
        caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"),
    ):
        session, device = onnx_providers.create_session(
            "model.onnx", model_name="layout", disable_cpu_arena=disable_cpu_arena
        )

    assert device == "cpu"
    assert session is cpu_session
    assert [o["providers"] for o in opened] == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    expected = {
        "log_severity_level": 3,
        "intra_op_num_threads": onnx_providers.available_cpu_count(),
        "config:session.intra_op.allow_spinning": "0",
    }
    if disable_cpu_arena:
        expected["enable_cpu_mem_arena"] = False
    assert opened[1]["options"] == expected
    # The CUDA attempt is not sized: only CPU sessions take the thread settings.
    assert "intra_op_num_threads" not in opened[0]["options"]
    assert (
        "layout could not open its CUDA session: GPU out of memory (RuntimeError); "
        "opening it on the CPU"
    ) in caplog.text
    assert "requested CUDA" not in caplog.text


def test_create_session_raises_other_cuda_session_failures():
    mock_ort, opened, _ = _ort_whose_sessions_fail(
        "[ONNXRuntimeError] : 1 : FAIL : CUDA failure 100: no CUDA-capable device is detected"
    )
    with (
        patch.dict("sys.modules", {"onnxruntime": mock_ort}),
        pytest.raises(RuntimeError, match="no CUDA-capable device"),
    ):
        onnx_providers.create_session("model.onnx", model_name="layout")
    assert len(opened) == 1


def test_create_session_raises_a_cpu_sessions_arena_failure():
    """ORT's CPU arena words its failures like the CUDA one; that is no GPU shortage."""
    mock_ort, opened, _ = _ort_whose_sessions_fail(
        _SESSION_ARENA_OOM, fail_on="CPUExecutionProvider"
    )
    with (
        patch.dict("sys.modules", {"onnxruntime": mock_ort}),
        pytest.raises(RuntimeError, match="BFCArena"),
    ):
        onnx_providers.create_session("model.onnx", device="cpu", model_name="layout")
    assert len(opened) == 1


# -- CPU thread sizing --------------------------------------------------------


def _cgroup_tree(tmp_path, *, own="/", files=None):
    """A fake /proc/self/cgroup and /sys/fs/cgroup; *files* maps relative path to content."""
    proc = tmp_path / "proc_cgroup"
    proc.write_text(f"0::{own}\n")
    root = tmp_path / "cg"
    root.mkdir()
    for rel, content in (files or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return str(proc), str(root)


def test_cgroup_cpu_limit_reads_the_container_quota(tmp_path):
    proc, root = _cgroup_tree(tmp_path, files={"cpu.max": "250000 100000\n"})
    assert onnx_providers._cgroup_cpu_limit(proc, root) == 2.5


def test_cgroup_cpu_limit_is_none_when_unlimited(tmp_path):
    proc, root = _cgroup_tree(tmp_path, files={"cpu.max": "max 100000\n"})
    assert onnx_providers._cgroup_cpu_limit(proc, root) is None


def test_cgroup_cpu_limit_takes_the_smallest_quota_up_the_tree(tmp_path):
    proc, root = _cgroup_tree(
        tmp_path,
        own="/a/b",
        files={
            "a/b/cpu.max": "max 100000\n",
            "a/cpu.max": "400000 100000\n",
            "cpu.max": "800000 100000\n",
        },
    )
    assert onnx_providers._cgroup_cpu_limit(proc, root) == 4.0


def test_cgroup_cpu_limit_reads_cgroup_v1(tmp_path):
    proc, root = _cgroup_tree(
        tmp_path,
        files={"cpu/cpu.cfs_quota_us": "150000\n", "cpu/cpu.cfs_period_us": "100000\n"},
    )
    assert onnx_providers._cgroup_cpu_limit(proc, root) == 1.5


def test_cgroup_cpu_limit_v1_unlimited_is_none(tmp_path):
    proc, root = _cgroup_tree(
        tmp_path,
        files={"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"},
    )
    assert onnx_providers._cgroup_cpu_limit(proc, root) is None


def test_cgroup_cpu_limit_is_none_without_cgroup_files(tmp_path):
    assert (
        onnx_providers._cgroup_cpu_limit(str(tmp_path / "missing"), str(tmp_path / "none")) is None
    )


def test_available_cpu_count_follows_the_affinity_mask(monkeypatch):
    monkeypatch.setattr(onnx_providers.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(
        onnx_providers.os, "sched_getaffinity", lambda _pid: {4, 5, 6, 7}, raising=False
    )
    monkeypatch.setattr(onnx_providers, "_cgroup_cpu_limit", lambda: None)
    assert onnx_providers.available_cpu_count() == 4


def test_available_cpu_count_follows_the_cgroup_quota_rounded_up(monkeypatch):
    monkeypatch.setattr(onnx_providers.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(
        onnx_providers.os, "sched_getaffinity", lambda _pid: set(range(16)), raising=False
    )
    monkeypatch.setattr(onnx_providers, "_cgroup_cpu_limit", lambda: 2.5)
    assert onnx_providers.available_cpu_count() == 3
    monkeypatch.setattr(onnx_providers, "_cgroup_cpu_limit", lambda: 0.25)
    assert onnx_providers.available_cpu_count() == 1


def test_available_cpu_count_without_affinity_support(monkeypatch):
    """macOS has no sched_getaffinity: the machine's CPU count stands."""
    monkeypatch.setattr(onnx_providers.os, "cpu_count", lambda: 8)
    monkeypatch.delattr(onnx_providers.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(onnx_providers, "_cgroup_cpu_limit", lambda: None)
    assert onnx_providers.available_cpu_count() == 8


def test_cpu_session_threads_prefers_the_explicit_setting(monkeypatch):
    monkeypatch.setattr(onnx_providers, "available_cpu_count", lambda: 6)
    assert onnx_providers.cpu_session_threads(0) == 6
    assert onnx_providers.cpu_session_threads(3) == 3


class _Options:
    def __init__(self):
        self.entries: dict[str, str] = {}

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


def test_apply_cpu_threading_sizes_the_pool_and_stops_spinning(monkeypatch):
    monkeypatch.setattr(onnx_providers, "available_cpu_count", lambda: 4)
    monkeypatch.setattr(onnx_providers, "_onnx_thread_settings", lambda: (0, False))
    options = _Options()
    onnx_providers.apply_cpu_threading(options)
    assert options.intra_op_num_threads == 4
    assert options.entries == {"session.intra_op.allow_spinning": "0"}


def test_apply_cpu_threading_honours_settings(monkeypatch):
    monkeypatch.setattr(onnx_providers, "available_cpu_count", lambda: 16)
    monkeypatch.setattr(onnx_providers, "_onnx_thread_settings", lambda: (2, True))
    options = _Options()
    onnx_providers.apply_cpu_threading(options)
    assert options.intra_op_num_threads == 2
    assert options.entries == {}


def test_ml_options_read_the_ml_env(monkeypatch):
    from bibr.config import MlOptions

    monkeypatch.setenv("ML_ONNX_THREADS", "3")
    monkeypatch.setenv("ML_ONNX_SPINNING", "true")
    options = MlOptions()
    assert (options.onnx_threads, options.onnx_spinning) == (3, True)
    monkeypatch.delenv("ML_ONNX_THREADS")
    monkeypatch.delenv("ML_ONNX_SPINNING")
    options = MlOptions()
    assert (options.onnx_threads, options.onnx_spinning) == (0, False)


def test_onnx_thread_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("BIBR_DISABLE_DOTENV", "1")
    monkeypatch.setenv("ML_ONNX_THREADS", "5")
    monkeypatch.setenv("ML_ONNX_SPINNING", "true")
    assert onnx_providers._onnx_thread_settings() == (5, True)


def test_onnx_thread_settings_read_the_dotenv_chain(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("ML_ONNX_THREADS=3\nML_ONNX_SPINNING=true\n")
    monkeypatch.setenv("BIBR_ENV_FILE", str(env_file))
    monkeypatch.delenv("BIBR_DISABLE_DOTENV", raising=False)
    monkeypatch.delenv("ML_ONNX_THREADS", raising=False)
    monkeypatch.delenv("ML_ONNX_SPINNING", raising=False)
    assert onnx_providers._onnx_thread_settings() == (3, True)
    monkeypatch.setenv("ML_ONNX_THREADS", "2")  # a real environment variable wins
    assert onnx_providers._onnx_thread_settings() == (2, True)


def test_onnx_thread_settings_default_when_unset_or_invalid(monkeypatch):
    monkeypatch.setenv("BIBR_DISABLE_DOTENV", "1")
    monkeypatch.delenv("ML_ONNX_THREADS", raising=False)
    monkeypatch.delenv("ML_ONNX_SPINNING", raising=False)
    assert onnx_providers._onnx_thread_settings() == (0, False)
    monkeypatch.setenv("ML_ONNX_THREADS", "many")
    assert onnx_providers._onnx_thread_settings() == (0, False)
    monkeypatch.setenv("ML_ONNX_THREADS", "-2")
    assert onnx_providers._onnx_thread_settings() == (0, False)


def test_cpu_session_gets_the_thread_settings_and_cuda_session_does_not():
    mock_ort, recorded = _cpu_only_ort_with_recording_options()
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        onnx_providers.create_session("model.onnx", device="cpu", model_name="layout")
    assert recorded["intra_op_num_threads"] == onnx_providers.available_cpu_count()
    assert recorded["config:session.intra_op.allow_spinning"] == "0"


def test_cuda_session_keeps_ort_thread_defaults():
    mock_ort = _cuda_build_ort()
    recorded: dict = {}

    class _RecordingOptions:
        def __setattr__(self, name, value):
            recorded[name] = value

        def add_session_config_entry(self, key, value):
            recorded[f"config:{key}"] = value

    mock_ort.SessionOptions = _RecordingOptions
    mock_ort.InferenceSession.return_value.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        _, device = onnx_providers.create_session("model.onnx", device="cuda", model_name="layout")
    assert device == "cuda"
    assert "intra_op_num_threads" not in recorded
    assert not any(key.startswith("config:") for key in recorded)


def test_cpu_ort_session_options_carry_the_thread_settings(monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(onnx_providers, "_onnx_thread_settings", lambda: (3, False))

    options = onnx_providers.cpu_ort_session_options()

    assert isinstance(options, ort.SessionOptions)
    assert options.intra_op_num_threads == 3
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
