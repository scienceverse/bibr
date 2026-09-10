import os
import subprocess
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _fake_pids_never_reach_killpg(monkeypatch):
    """Keep ``RapidMlxServer.shutdown()`` from signalling a real process group.

    The fake ``Popen`` objects below are plain ``MagicMock``s, so ``proc.pid`` is
    a mock — and ``os.killpg`` coerces it via ``__index__``, which MagicMock
    answers with 1. Unpatched, ``shutdown()`` therefore SIGTERMs process group 1
    and takes down the CI runner mid-suite. See the guard in tests/conftest.py.
    """
    monkeypatch.setattr(os, "killpg", MagicMock(name="os.killpg"), raising=False)


def _capture_server_cmd(monkeypatch, *, model, multimodal, settings):
    """Build a RapidMlxServer with mocked subprocess/health and return the launch cmd."""
    from bibr.local import rapid_mlx as mod

    captured: dict[str, object] = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    monkeypatch.setattr(mod.shutil, "which", lambda exe: "/opt/rapid-mlx/bin/rapid-mlx")
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod, "request_bytes", lambda *_a, **_k: (200, "OK", b"{}"))
    monkeypatch.setattr(mod.RapidMlxServer, "_warmup_model", lambda self: None)

    server = mod.RapidMlxServer(
        model=model,
        served_model_name=model,
        port=8773,
        multimodal=multimodal,
        settings=settings,
    )
    server.shutdown()
    return captured["cmd"]


def test_rapid_mlx_ocr_default_is_8bit():
    from bibr.config import OcrOptions

    assert OcrOptions(_env_file=None).rapid_mlx_model == "mlx-community/GLM-OCR-8bit"


def test_rapid_mlx_server_command_uses_fast_qwen_flags(monkeypatch):
    from bibr.config import Settings
    from bibr.local import rapid_mlx as mod

    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    def fake_request_bytes(url, **kwargs):
        captured.setdefault("requests", []).append((url, kwargs))
        return 200, "OK", b"{}"

    monkeypatch.setattr(mod.shutil, "which", lambda exe: "/opt/rapid-mlx/bin/rapid-mlx")
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod, "request_bytes", fake_request_bytes)
    monkeypatch.setattr(Settings.rapid_mlx, "executable", "rapid-mlx")
    monkeypatch.setattr(Settings.rapid_mlx, "startup_timeout", 1)
    monkeypatch.setattr(Settings.rapid_mlx, "warmup_timeout", 1)

    server = mod.RapidMlxServer(
        model="qwen3.5-4b-4bit",
        served_model_name="qwen3.5-4b-4bit",
        port=8773,
        multimodal=False,
        prefill_step_size=8192,
        max_tokens=8192,
    )

    cmd = captured["cmd"]
    assert cmd[:3] == ["/opt/rapid-mlx/bin/rapid-mlx", "serve", "qwen3.5-4b-4bit"]
    assert "--no-thinking" in cmd
    assert "--no-mllm" in cmd
    assert "--no-tool-call-parser" in cmd
    assert "--no-reasoning-parser" in cmd
    assert "--prefill-step-size" in cmd
    assert cmd[cmd.index("--prefill-step-size") + 1] == "8192"
    assert "--max-tokens" in cmd
    assert cmd[cmd.index("--max-tokens") + 1] == "8192"
    assert "--force-disk-check" in cmd
    assert captured["kwargs"]["stdout"] == subprocess.DEVNULL
    # The warmup POST is the only request carrying a body (other requests are
    # the pre-spawn port-guard probe and health polls).
    warmup = next(k["body"] for _url, k in captured["requests"] if "body" in k).decode("utf-8")
    assert "Return JSON only" not in warmup
    assert '"content": "OK"' in warmup

    server.shutdown()


def test_rapid_mlx_server_passes_external_cache_env(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local import rapid_mlx as mod

    settings = GlobalSettings()
    settings.rapid_mlx.hf_home = "/Volumes/Datasets/huggingface-cache"
    settings.rapid_mlx.hf_hub_cache = "/Volumes/Datasets/huggingface-cache/hub"
    settings.rapid_mlx.home = "/Volumes/Datasets/bibr-runtime-experiments/home"

    captured: dict[str, object] = {}

    def fake_popen(_cmd, **kwargs):
        captured["env"] = kwargs["env"]
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    monkeypatch.setattr(mod.shutil, "which", lambda exe: "/opt/rapid-mlx/bin/rapid-mlx")
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod, "request_bytes", lambda *_a, **_k: (200, "OK", b"{}"))

    server = mod.RapidMlxServer(
        model="qwen3.5-4b-4bit",
        served_model_name="qwen3.5-4b-4bit",
        port=8773,
        multimodal=False,
        settings=settings,
    )

    env = captured["env"]
    assert env["HF_HOME"] == "/Volumes/Datasets/huggingface-cache"
    assert env["HF_HUB_CACHE"] == "/Volumes/Datasets/huggingface-cache/hub"
    assert env["HOME"] == "/Volumes/Datasets/bibr-runtime-experiments/home"

    server.shutdown()


def test_rapid_mlx_ocr_client_uses_glm_ocr_8bit(monkeypatch):
    from bibr.config import Settings
    from bibr.local import rapid_mlx as mod

    captured: dict[str, object] = {}

    class FakeServer:
        base_url = "http://localhost:8772"

        @property
        def loaded(self):
            return True

    class FakeHttpClient:
        def __init__(self, **kwargs):
            captured["http"] = kwargs

        async def shutdown(self):
            return None

    def fake_server(**kwargs):
        captured["server"] = kwargs
        return FakeServer()

    monkeypatch.setattr(Settings.ocr, "rapid_mlx_model", "mlx-community/GLM-OCR-8bit")
    monkeypatch.setattr(Settings.ocr, "rapid_mlx_port", 8772)
    monkeypatch.setattr(Settings.ocr, "rapid_mlx_extra_args", "")
    monkeypatch.setattr(mod, "RapidMlxServer", fake_server)
    monkeypatch.setattr(mod, "HttpOcrClient", FakeHttpClient)

    mod.RapidMlxOcrClient()

    assert captured["server"]["model"] == "mlx-community/GLM-OCR-8bit"
    assert captured["server"]["served_model_name"] == "mlx-community/GLM-OCR-8bit"
    assert captured["server"]["port"] == 8772
    assert captured["server"]["multimodal"] is True
    assert captured["server"]["max_tokens"] == 16384
    assert captured["http"]["model"] == "mlx-community/GLM-OCR-8bit"


def test_paddle_rapid_mlx_runs_a_strict_image_smoke(monkeypatch):
    """Paddle Rapid-MLX is eligible only after a real image OCR probe."""
    from bibr.config import GlobalSettings
    from bibr.local import rapid_mlx as mod

    settings = GlobalSettings()
    captured: dict[str, object] = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    def fake_request(url, **kwargs):
        if "body" in kwargs:
            captured["payload"] = __import__("json").loads(kwargs["body"])
            return (
                200,
                "OK",
                b'{"model":"olragon/PaddleOCR-VL-1.6-8bit",'
                b'"choices":[{"message":{"content":"OCR OK"}}]}',
            )
        return 200, "OK", b"{}"

    monkeypatch.setattr(mod.shutil, "which", lambda _exe: "/opt/rapid-mlx/rapid-mlx")
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod, "request_bytes", fake_request)

    server = mod.RapidMlxServer(
        model="olragon/PaddleOCR-VL-1.6-8bit",
        served_model_name="olragon/PaddleOCR-VL-1.6-8bit",
        port=8775,
        multimodal=True,
        strict_ocr_smoke=True,
        settings=settings,
    )

    assert "--mllm" in captured["cmd"]
    payload = captured["payload"]
    assert payload["model"] == "olragon/PaddleOCR-VL-1.6-8bit"
    assert payload["temperature"] == 0.0
    assert payload["messages"][0]["content"][1]["text"] == "OCR:"
    assert payload["messages"][0]["content"][0]["type"] == "image_url"
    server.shutdown()


def test_paddle_rapid_mlx_reused_listener_must_pass_strict_smoke(monkeypatch):
    """A compatible model alias alone is not evidence that its image path works."""
    from bibr.config import GlobalSettings
    from bibr.local import rapid_mlx as mod

    popen = MagicMock()
    requests: list[object] = []
    monkeypatch.setattr(mod, "guard_managed_server_port", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(mod.subprocess, "Popen", popen)
    monkeypatch.setattr(
        mod,
        "request_bytes",
        lambda _url, **kwargs: (
            requests.append(kwargs["body"])
            or (
                200,
                "OK",
                b'{"model":"olragon/PaddleOCR-VL-1.6-8bit",'
                b'"choices":[{"message":{"content":"OCR OK"}}]}',
            )
        ),
    )

    server = mod.RapidMlxServer(
        model="olragon/PaddleOCR-VL-1.6-8bit",
        served_model_name="olragon/PaddleOCR-VL-1.6-8bit",
        port=8775,
        multimodal=True,
        strict_ocr_smoke=True,
        settings=GlobalSettings(),
    )

    assert server.loaded is True
    assert server._process is None
    popen.assert_not_called()
    assert len(requests) == 1


def test_paddle_rapid_mlx_reuse_rejects_failed_smoke_without_shutdown(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import rapid_mlx as mod

    popen = MagicMock()
    killpg = MagicMock()
    monkeypatch.setattr(mod, "guard_managed_server_port", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(mod.subprocess, "Popen", popen)
    monkeypatch.setattr(mod.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(
        mod,
        "request_bytes",
        lambda _url, **kwargs: (200, "OK", b"{}") if "body" in kwargs else (200, "OK", b"{}"),
    )

    with pytest.raises(UpstreamServiceError, match="smoke"):
        mod.RapidMlxServer(
            model="olragon/PaddleOCR-VL-1.6-8bit",
            served_model_name="olragon/PaddleOCR-VL-1.6-8bit",
            port=8775,
            multimodal=True,
            strict_ocr_smoke=True,
            settings=GlobalSettings(),
        )

    popen.assert_not_called()
    killpg.assert_not_called()


@pytest.mark.parametrize(
    "status, body",
    [
        (503, b"unavailable"),
        (200, b"{}"),
        (200, b'{"model":"olragon/PaddleOCR-VL-1.6-8bit","choices":[]}'),
        (
            200,
            b'{"model":"wrong-model","choices":[{"message":{"content":"OCR OK"}}]}',
        ),
        (
            200,
            b'{"model":"olragon/PaddleOCR-VL-1.6-8bit",'
            b'"choices":[{"message":{"content":"nothing"}}]}',
        ),
    ],
)
def test_paddle_rapid_mlx_rejects_incompatible_smoke(monkeypatch, status, body):
    from bibr.config import GlobalSettings
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import rapid_mlx as mod

    proc = MagicMock()
    proc.poll.return_value = None
    monkeypatch.setattr(mod.shutil, "which", lambda _exe: "/opt/rapid-mlx/rapid-mlx")
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *_args, **_kw: proc)
    monkeypatch.setattr(
        mod,
        "request_bytes",
        lambda _url, **kwargs: (
            (status, "unavailable", body) if "body" in kwargs else (200, "OK", b"{}")
        ),
    )

    with pytest.raises(UpstreamServiceError, match="smoke"):
        mod.RapidMlxServer(
            model="olragon/PaddleOCR-VL-1.6-8bit",
            served_model_name="olragon/PaddleOCR-VL-1.6-8bit",
            port=8775,
            multimodal=True,
            strict_ocr_smoke=True,
            settings=GlobalSettings(),
        )

    assert proc.terminate.called or proc.wait.called


@pytest.mark.asyncio
async def test_paddle_rapid_mlx_client_recycles_after_configured_threshold(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local import rapid_mlx as mod

    settings = GlobalSettings()
    settings.ocr.paddle_rapid_mlx_model = "olragon/PaddleOCR-VL-1.6-8bit"
    settings.ocr.rapid_mlx_recycle_after = 80
    spawns: list[object] = []

    class FakeServer:
        base_url = "http://localhost:8775"

        def __init__(self, **kwargs):
            spawns.append(kwargs)

        @property
        def loaded(self):
            return True

        def shutdown(self):
            return None

    class FakeClient:
        def __init__(self, **_kwargs):
            return None

        async def recognize(self, _image, _prompt):
            return "text"

        async def shutdown(self):
            return None

    monkeypatch.setattr(mod, "RapidMlxServer", FakeServer)
    monkeypatch.setattr(mod, "PaddleHttpOcrClient", FakeClient)
    client = mod.PaddleRapidMlxOcrClient(settings=settings)
    client._request_count = 79

    assert await client.recognize(None, "OCR:") == "text"
    assert len(spawns) == 2
    assert spawns[0]["model"] == "olragon/PaddleOCR-VL-1.6-8bit"
    assert spawns[0]["strict_ocr_smoke"] is True


@pytest.mark.asyncio
async def test_paddle_rapid_mlx_preserves_paddle_profile_for_http_requests(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local import rapid_mlx as mod
    from bibr.ocr.profiles import PADDLE_PROFILE

    server = MagicMock(base_url="http://localhost:8775", loaded=True)
    monkeypatch.setattr(mod, "RapidMlxServer", MagicMock(return_value=server))
    client = mod.PaddleRapidMlxOcrClient(settings=GlobalSettings(), profile=PADDLE_PROFILE)

    payload = client._http_client._build_payload("image-base64", "OCR:")
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 1024
    assert client._http_client._profile is PADDLE_PROFILE
    await client.shutdown()


async def test_rapid_mlx_ocr_client_recycles_server_after_threshold(monkeypatch):
    """Regression test for the vendored rapid-mlx vision-cache leak.

    rapid-mlx's MLLM vision-embedding cache retains every request's pixel
    tensors, evicted by entry count (100) but never by bytes — bibr's
    unique-crop-per-region workload never hits it, so it fills with ~100
    dead entries (~3.4GB) before OCR starts failing silently. The client
    must restart its managed subprocess after `rapid_mlx_recycle_after`
    regions to release it.
    """
    from bibr.config import Settings
    from bibr.local import rapid_mlx as mod

    server_spawns: list[int] = []
    shutdown_calls: list[int] = []
    http_spawns: list[int] = []
    http_shutdowns: list[int] = []

    class FakeServer:
        def __init__(self, **kwargs):
            self.id = len(server_spawns)
            server_spawns.append(self.id)

        base_url = "http://localhost:8772"

        @property
        def loaded(self):
            return True

        def shutdown(self):
            shutdown_calls.append(self.id)

    class FakeHttpClient:
        def __init__(self, **kwargs):
            self.id = len(http_spawns)
            http_spawns.append(self.id)

        async def recognize(self, image, prompt):
            return "text"

        async def shutdown(self):
            http_shutdowns.append(self.id)

    monkeypatch.setattr(Settings.ocr, "rapid_mlx_model", "mlx-community/GLM-OCR-8bit")
    monkeypatch.setattr(Settings.ocr, "rapid_mlx_port", 8772)
    monkeypatch.setattr(Settings.ocr, "rapid_mlx_extra_args", "")
    monkeypatch.setattr(Settings.ocr, "rapid_mlx_recycle_after", 3)
    monkeypatch.setattr(mod, "RapidMlxServer", FakeServer)
    monkeypatch.setattr(mod, "HttpOcrClient", FakeHttpClient)

    client = mod.RapidMlxOcrClient()

    for _ in range(3):
        await client.recognize(None, "Text Recognition:")

    # Threshold hit at request 3 → old server/http client torn down, new ones spawned.
    assert server_spawns == [0, 1]
    assert shutdown_calls == [0]
    assert http_spawns == [0, 1]
    assert http_shutdowns == [0]
    assert client._request_count == 0

    for _ in range(3):
        await client.recognize(None, "Text Recognition:")

    assert server_spawns == [0, 1, 2]
    assert shutdown_calls == [0, 1]


async def test_rapid_mlx_recycle_failure_recovers_on_next_request(monkeypatch):
    from bibr.config import Settings
    from bibr.exceptions import UpstreamServiceError
    from bibr.local import rapid_mlx as mod

    spawn_count = 0

    class FakeServer:
        base_url = "http://localhost:8772"

        def __init__(self, server_id):
            self.server_id = server_id
            self._loaded = True

        @property
        def loaded(self):
            return self._loaded

        def shutdown(self):
            self._loaded = False

    def fake_server(**kwargs):
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 2:
            raise RuntimeError("restart failed")
        return FakeServer(spawn_count)

    class FakeHttpClient:
        def __init__(self, **kwargs):
            self.closed = False

        async def recognize(self, image, prompt):
            assert not self.closed
            return "text"

        async def shutdown(self):
            self.closed = True

    monkeypatch.setattr(Settings.ocr, "rapid_mlx_model", "mlx-community/GLM-OCR-8bit")
    monkeypatch.setattr(Settings.ocr, "rapid_mlx_port", 8772)
    monkeypatch.setattr(Settings.ocr, "rapid_mlx_extra_args", "")
    monkeypatch.setattr(Settings.ocr, "rapid_mlx_recycle_after", 3)
    monkeypatch.setattr(mod, "RapidMlxServer", fake_server)
    monkeypatch.setattr(mod, "HttpOcrClient", FakeHttpClient)

    client = mod.RapidMlxOcrClient()
    client._request_count = 3

    with pytest.raises(UpstreamServiceError, match="restart failed"):
        await client._recycle()

    assert client.loaded is False
    assert client._server is None
    assert client._http_client is None
    assert await client.recognize(None, "Text Recognition:") == "text"
    assert client.loaded is True
    assert spawn_count == 3


def test_rapid_mlx_llm_server_configures_qwen_no_think_defaults(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local import rapid_mlx as mod

    captured: dict[str, object] = {}

    class FakeServer:
        base_url = "http://localhost:8773"

        def shutdown(self):
            return None

    def fake_server(**kwargs):
        captured["server"] = kwargs
        return FakeServer()

    settings = GlobalSettings()
    settings.llm.rapid_mlx_model = "qwen3.5-4b-4bit"
    settings.llm.rapid_mlx_port = 8773
    settings.llm.rapid_mlx_extra_args = ""
    settings.llm.max_tokens = 65536
    settings.llm.timeout_seconds = 30
    settings.llm.max_concurrency = 0
    settings.llm.instructor_mode = "json"
    settings.llm.reasoning_effort = "low"
    settings.llm.reasoning_effort_authors = "low"
    settings.llm.reasoning_effort_citations = "low"
    monkeypatch.setattr(mod, "RapidMlxServer", fake_server)

    for field in ("max_tokens", "timeout_seconds", "max_concurrency"):
        settings.llm.model_fields_set.discard(field)

    server = mod.RapidMlxLlmServer(settings=settings)
    server.configure_llm_client()

    assert captured["server"]["model"] == "qwen3.5-4b-4bit"
    assert captured["server"]["multimodal"] is False
    assert captured["server"]["prefill_step_size"] == 8192
    assert captured["server"]["no_thinking"] is True
    assert settings.llm.provider == "openai"
    assert settings.llm.base_url == "http://localhost:8773/v1"
    assert settings.llm.model == "qwen3.5-4b-4bit"
    assert settings.llm.max_tokens == 8192
    assert settings.llm.timeout_seconds == 300
    assert settings.llm.max_concurrency == 1
    assert settings.llm.instructor_mode == "json"
    assert settings.llm.reasoning_effort is None
    assert settings.llm.reasoning_effort_authors is None
    assert settings.llm.reasoning_effort_citations is None


@pytest.mark.asyncio
async def test_resource_manager_starts_rapid_mlx_llm(monkeypatch):
    import bibr.local.rapid_mlx as rapid_mlx
    from bibr.config import GlobalSettings
    from bibr.pipeline.resources import ResourceManager

    instance = MagicMock()
    factory = MagicMock(return_value=instance)
    monkeypatch.setattr(rapid_mlx, "RapidMlxLlmServer", factory)
    custom = GlobalSettings()
    resources = ResourceManager(settings=custom)

    await resources.start_llm_server(backend="rapid-mlx")

    assert resources._llm_server is instance
    factory.assert_called_once_with(settings=custom)
    instance.configure_llm_client.assert_called_once()


def test_auto_spec_decode_enables_mtp_for_qwen(monkeypatch):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.rapid_mlx.spec_decode = "auto"

    cmd = _capture_server_cmd(
        monkeypatch, model="qwen3.5-4b-4bit", multimodal=False, settings=settings
    )

    assert "--spec-decode" in cmd
    assert cmd[cmd.index("--spec-decode") + 1] == "mtp"
    assert "--no-spec-decode" not in cmd


def test_auto_spec_decode_disabled_for_non_qwen(monkeypatch):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.rapid_mlx.spec_decode = "auto"

    cmd = _capture_server_cmd(
        monkeypatch, model="numind/NuExtract3-mlx-8bits", multimodal=False, settings=settings
    )

    assert "--no-spec-decode" in cmd
    assert "--spec-decode" not in cmd


def test_spec_decode_none_overrides_qwen(monkeypatch):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.rapid_mlx.spec_decode = "none"

    cmd = _capture_server_cmd(
        monkeypatch, model="qwen3.5-4b-4bit", multimodal=False, settings=settings
    )

    assert "--no-spec-decode" in cmd
    assert "--spec-decode" not in cmd


def test_pin_system_prompt_present_by_default_and_absent_when_disabled(monkeypatch):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    cmd = _capture_server_cmd(
        monkeypatch, model="qwen3.5-4b-4bit", multimodal=False, settings=settings
    )
    assert "--pin-system-prompt" in cmd

    settings_off = GlobalSettings()
    settings_off.rapid_mlx.pin_system_prompt = False
    cmd_off = _capture_server_cmd(
        monkeypatch, model="qwen3.5-4b-4bit", multimodal=False, settings=settings_off
    )
    assert "--pin-system-prompt" not in cmd_off


def test_multimodal_server_never_gets_spec_decode(monkeypatch):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.rapid_mlx.spec_decode = "mtp"

    cmd = _capture_server_cmd(
        monkeypatch, model="mlx-community/GLM-OCR-8bit", multimodal=True, settings=settings
    )

    assert "--mllm" in cmd
    assert "--spec-decode" not in cmd
    assert "--no-spec-decode" not in cmd


def test_configure_llm_client_respects_explicit_max_concurrency(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local import rapid_mlx as mod

    class FakeServer:
        base_url = "http://localhost:8773"

        def shutdown(self):
            return None

    monkeypatch.setattr(mod, "RapidMlxServer", lambda **_kw: FakeServer())

    settings = GlobalSettings()
    settings.llm.rapid_mlx_model = "qwen3.5-4b-4bit"
    settings.llm.max_concurrency = 8
    settings.llm.model_fields_set.add("max_concurrency")

    server = mod.RapidMlxLlmServer(settings=settings)
    server.configure_llm_client()

    assert settings.llm.max_concurrency == 8


def test_auto_mtp_falls_back_when_checkpoint_lacks_mtp_layers(monkeypatch):
    """auto mode relaunches without MTP when Rapid-MLX rejects the checkpoint
    (e.g. mlx-community/Qwen3.5-4B-MLX-4bit ships no MTP layers)."""
    from bibr.config import GlobalSettings
    from bibr.local import rapid_mlx as mod

    settings = GlobalSettings()
    settings.rapid_mlx.spec_decode = "auto"

    cmds: list[list[str]] = []

    def fake_popen(cmd, **_kwargs):
        cmds.append(list(cmd))
        proc = MagicMock()
        proc.poll.return_value = 2 if len(cmds) == 1 else None
        proc.returncode = 2
        return proc

    monkeypatch.setattr(mod.shutil, "which", lambda exe: "/opt/rapid-mlx/bin/rapid-mlx")
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod, "request_bytes", lambda *_a, **_k: (200, "OK", b"{}"))
    monkeypatch.setattr(mod.RapidMlxServer, "_warmup_model", lambda self: None)
    monkeypatch.setattr(
        mod.RapidMlxServer,
        "_read_stderr_tail",
        lambda self, n_bytes=8192: (
            "error: --spec-decode mtp requires a Qwen3.5 / Qwen3.6 checkpoint with "
            "mtp_num_hidden_layers >= 1 in config.json."
        ),
    )

    server = mod.RapidMlxServer(
        model="qwen3.5-4b-4bit",
        served_model_name="qwen3.5-4b-4bit",
        port=8773,
        multimodal=False,
        settings=settings,
    )
    server.shutdown()

    assert len(cmds) == 2
    assert cmds[0][cmds[0].index("--spec-decode") + 1] == "mtp"
    assert "--no-spec-decode" in cmds[1]
    assert "--spec-decode" not in cmds[1]


def test_auto_mtp_falls_back_when_error_is_preceded_by_info_noise(monkeypatch):
    """The mtp rejection marker must be detected even when the server prints
    hundreds of chars of INFO logging before the ``error:`` line (regression:
    the startup RuntimeError embedded only the first 500 chars of the stderr
    tail, slicing the marker off and defeating the auto fallback)."""
    from bibr.config import GlobalSettings
    from bibr.local import rapid_mlx as mod

    settings = GlobalSettings()
    settings.rapid_mlx.spec_decode = "auto"

    cmds: list[list[str]] = []

    def fake_popen(cmd, **_kwargs):
        cmds.append(list(cmd))
        proc = MagicMock()
        proc.poll.return_value = 2 if len(cmds) == 1 else None
        proc.returncode = 2
        return proc

    info_noise = "".join(
        f"INFO:rapid_mlx.cli:startup step {i} completed without incident\n" for i in range(12)
    )
    assert len(info_noise) > 500
    stderr_tail = info_noise + (
        "error: --spec-decode mtp requires a Qwen3.5 / Qwen3.6 checkpoint with "
        "mtp_num_hidden_layers >= 1 in config.json."
    )

    monkeypatch.setattr(mod.shutil, "which", lambda exe: "/opt/rapid-mlx/bin/rapid-mlx")
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod, "request_bytes", lambda *_a, **_k: (200, "OK", b"{}"))
    monkeypatch.setattr(mod.RapidMlxServer, "_warmup_model", lambda self: None)
    monkeypatch.setattr(
        mod.RapidMlxServer, "_read_stderr_tail", lambda self, n_bytes=8192: stderr_tail
    )

    server = mod.RapidMlxServer(
        model="qwen3.5-4b-4bit",
        served_model_name="qwen3.5-4b-4bit",
        port=8773,
        multimodal=False,
        settings=settings,
    )
    server.shutdown()

    assert len(cmds) == 2
    assert cmds[0][cmds[0].index("--spec-decode") + 1] == "mtp"
    assert "--no-spec-decode" in cmds[1]
    assert "--spec-decode" not in cmds[1]


def test_explicit_mtp_does_not_fall_back(monkeypatch):
    """spec_decode="mtp" is an explicit operator choice — fail loudly instead
    of silently downgrading."""
    import pytest

    from bibr.config import GlobalSettings
    from bibr.local import rapid_mlx as mod

    settings = GlobalSettings()
    settings.rapid_mlx.spec_decode = "mtp"

    cmds: list[list[str]] = []

    def fake_popen(cmd, **_kwargs):
        cmds.append(list(cmd))
        proc = MagicMock()
        proc.poll.return_value = 2
        proc.returncode = 2
        return proc

    monkeypatch.setattr(mod.shutil, "which", lambda exe: "/opt/rapid-mlx/bin/rapid-mlx")
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod, "request_bytes", lambda *_a, **_k: (200, "OK", b"{}"))
    monkeypatch.setattr(mod.RapidMlxServer, "_warmup_model", lambda self: None)
    monkeypatch.setattr(
        mod.RapidMlxServer,
        "_read_stderr_tail",
        lambda self, n_bytes=8192: "error: --spec-decode mtp requires mtp_num_hidden_layers",
    )

    with pytest.raises(RuntimeError, match="mtp_num_hidden_layers"):
        mod.RapidMlxServer(
            model="qwen3.5-4b-4bit",
            served_model_name="qwen3.5-4b-4bit",
            port=8773,
            multimodal=False,
            settings=settings,
        )

    assert len(cmds) == 1
