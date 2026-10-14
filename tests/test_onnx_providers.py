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


def test_coreml_included_when_available():
    """CoreMLExecutionProvider is included when available (macOS)."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import get_ort_providers

        providers = get_ort_providers(enable_cuda=False)

    assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


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


# --- ML_ORT_*_THREADS caps -----------------------------------------------------


def _capped_settings(intra: int = 0, inter: int = 0):
    from bibr.config import GlobalSettings

    return GlobalSettings(ml={"ort_intra_op_threads": intra, "ort_inter_op_threads": inter})


def test_thread_caps_default_to_onnxruntime_own_sizing():
    """Unset, both caps are 0: ORT keeps sizing its pools to the cores."""
    from bibr.config import GlobalSettings

    assert onnx_providers.ort_thread_caps(GlobalSettings()) == (0, 0)


def test_thread_caps_read_the_ml_settings():
    assert onnx_providers.ort_thread_caps(_capped_settings(4, 1)) == (4, 1)


def test_thread_caps_reject_negative_values():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _capped_settings(-1)


def test_create_session_leaves_thread_pools_alone_without_caps():
    """No cap configured: the session options carry no thread attribute, so a
    run opens its sessions exactly as before the setting existed."""
    mock_ort, recorded = _cpu_only_ort_with_recording_options()
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        onnx_providers.create_session("model.onnx", device="cpu", model_name="layout")

    assert "intra_op_num_threads" not in recorded
    assert "inter_op_num_threads" not in recorded


def test_create_session_applies_the_configured_thread_caps(monkeypatch):
    """The caps the bench used to patch into every SessionOptions now come from
    ML_ORT_INTRA_OP_THREADS / ML_ORT_INTER_OP_THREADS."""
    monkeypatch.setattr(onnx_providers, "ort_thread_caps", lambda settings=None: (4, 1))
    mock_ort, recorded = _cpu_only_ort_with_recording_options()
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        onnx_providers.create_session(
            "model.onnx", device="cpu", model_name="layout", disable_cpu_arena=True
        )

    assert recorded["intra_op_num_threads"] == 4
    assert recorded["inter_op_num_threads"] == 1
    assert recorded["enable_cpu_mem_arena"] is False


def test_create_session_sets_only_the_cap_that_is_configured(monkeypatch):
    monkeypatch.setattr(onnx_providers, "ort_thread_caps", lambda settings=None: (2, 0))
    mock_ort, recorded = _cpu_only_ort_with_recording_options()
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        onnx_providers.create_session("model.onnx", device="cpu", model_name="ner")

    assert recorded["intra_op_num_threads"] == 2
    assert "inter_op_num_threads" not in recorded


def test_create_session_reads_the_process_settings(monkeypatch):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.ml, "ort_intra_op_threads", 3)
    mock_ort, recorded = _cpu_only_ort_with_recording_options()
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        onnx_providers.create_session("model.onnx", device="cpu", model_name="ner")

    assert recorded["intra_op_num_threads"] == 3


def test_create_session_takes_the_caps_of_the_settings_it_is_given(monkeypatch):
    """A loader's settings (a pipeline's, a Python caller's) decide the caps,
    not the process settings."""
    from bibr.config import Settings

    monkeypatch.setattr(Settings.ml, "ort_intra_op_threads", 0)
    mock_ort, recorded = _cpu_only_ort_with_recording_options()
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        onnx_providers.create_session(
            "model.onnx", device="cpu", model_name="ner", settings=_capped_settings(2, 1)
        )

    assert (recorded["intra_op_num_threads"], recorded["inter_op_num_threads"]) == (2, 1)


def test_thread_caps_read_the_process_settings_without_copying_them(monkeypatch):
    """Opening a session reads two integers; it does not deep-copy the settings."""
    from bibr.config import Settings

    def no_snapshot(*_args, **_kwargs):
        raise AssertionError("ort_thread_caps copied the settings")

    monkeypatch.setattr("bibr.config.snapshot_settings", no_snapshot)
    monkeypatch.setattr(Settings.ml, "ort_intra_op_threads", 5)

    assert onnx_providers.ort_thread_caps() == (5, 0)


def test_with_thread_caps_returns_options_unchanged_without_caps():
    """No cap: the segmenter's options (or their absence) pass straight through."""
    sentinel = object()
    settings = _capped_settings()

    assert onnx_providers.with_ort_thread_caps(sentinel, settings) is sentinel
    assert onnx_providers.with_ort_thread_caps(None, settings) is None


def test_with_thread_caps_capped_real_session_options():
    """A real SessionOptions takes the caps: the arena-free CPU options keep
    their arena setting, and a CUDA SaT session (no options of its own) gets
    fresh ones carrying only the caps."""
    ort = pytest.importorskip("onnxruntime")
    settings = _capped_settings(2, 1)

    cpu_options = onnx_providers.with_ort_thread_caps(
        onnx_providers.cpu_ort_session_options(), settings
    )
    fresh = onnx_providers.with_ort_thread_caps(None, settings)

    assert cpu_options.enable_cpu_mem_arena is False
    assert (cpu_options.intra_op_num_threads, cpu_options.inter_op_num_threads) == (2, 1)
    assert isinstance(fresh, ort.SessionOptions)
    assert (fresh.intra_op_num_threads, fresh.inter_op_num_threads) == (2, 1)


def test_with_thread_caps_is_none_without_onnxruntime():
    with patch.dict("sys.modules", {"onnxruntime": None}):
        assert onnx_providers.with_ort_thread_caps(None, _capped_settings(2)) is None
