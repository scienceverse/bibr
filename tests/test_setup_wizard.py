import sys
from unittest.mock import MagicMock, patch

import pytest

from bibr.setup_wizard import (
    LLM_DEFAULTS,
    RecommendedSetup,
    SetupWizard,
    _available_extras,
    _build_recommended_setup,
    _fetch_models,
    _merge_env,
    _ml_extra_available,
    _select_model,
    _write_env_fresh,
)


@pytest.fixture(autouse=True)
def _ml_extra_installed(monkeypatch):
    """Pin the ml-extra probe so wizard tests take the same path in every venv.

    ``_step_external_services`` installs the ml extra for real whenever torch
    is not importable, so unpinned, the first test to reach that step in a
    core-only venv ran ``uv sync --inexact --extra=ml`` and installed torch
    mid-suite. Tests of that auto-install path pin False and stub the install.
    """
    monkeypatch.setattr("bibr.setup_wizard._ml_extra_available", lambda: True)


@pytest.mark.parametrize("missing", [None, "torch", "cv2"])
def test_ml_extra_available_requires_every_heavy_module(monkeypatch, missing):
    """The real probe, which the fixture above pins for every other test."""
    monkeypatch.setattr(
        "bibr.setup_wizard.importlib.util.find_spec",
        lambda name: None if name == missing else object(),
    )

    assert _ml_extra_available() is (missing is None)


def test_google_recommended_model_is_current_flash_lite():
    assert LLM_DEFAULTS["google"]["model"] == "gemini-3.5-flash-lite"


def test_write_env_fresh(tmp_path):
    env_path = tmp_path / ".env"
    env_vars = {
        "LLM_PROVIDER": "google",
        "LLM_MODEL": "gemini-3.5-flash-lite",
        "GOOGLE_API_KEY": "test-key-123",
        "CROSSREF_API_EMAIL": "user@example.com",
    }

    _write_env_fresh(env_path, env_vars)

    content = env_path.read_text()
    assert "LLM_PROVIDER=google" in content
    assert "LLM_MODEL=gemini-3.5-flash-lite" in content
    assert "GOOGLE_API_KEY=test-key-123" in content
    assert "CROSSREF_API_EMAIL=user@example.com" in content
    assert "# bibr" in content


def test_merge_env_preserves_existing(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# existing comment\nEXISTING_KEY=old_value\nLLM_MODEL=old-model\n",
        encoding="utf-8",
    )

    new_vars = {"LLM_MODEL": "new-model", "NEW_KEY": "new_value"}
    _merge_env(env_path, new_vars)

    content = env_path.read_text()
    # Existing key preserved
    assert "EXISTING_KEY=old_value" in content
    # Overwritten key updated
    assert "LLM_MODEL=new-model" in content
    assert "LLM_MODEL=old-model" not in content
    # New key added
    assert "NEW_KEY=new_value" in content
    # Comment preserved
    assert "# existing comment" in content


def test_fetch_models_openai():
    """_fetch_models for openai returns sorted model IDs, filtering non-chat models."""
    mock_models = [
        MagicMock(id="gpt-4o"),
        MagicMock(id="gpt-4o-mini"),
        MagicMock(id="text-embedding-3-small"),
        MagicMock(id="dall-e-3"),
        MagicMock(id="tts-1"),
        MagicMock(id="whisper-1"),
        MagicMock(id="gpt-5-nano"),
    ]
    mock_client = MagicMock()
    mock_client.models.list.return_value = mock_models

    with patch("openai.OpenAI", return_value=mock_client):
        result = _fetch_models("openai", "sk-test")

    assert result == ["gpt-4o", "gpt-4o-mini", "gpt-5-nano"]


def test_fetch_models_openai_custom_base_url_no_filter():
    """_fetch_models with custom base_url skips filtering (user-controlled server)."""
    mock_models = [
        MagicMock(id="gemma-3-27b"),
        MagicMock(id="my-embedding-model"),
    ]
    mock_client = MagicMock()
    mock_client.models.list.return_value = mock_models

    with patch("openai.OpenAI", return_value=mock_client):
        result = _fetch_models("openai", "sk-test", base_url="http://gpu-server:8080/v1")

    assert result == ["gemma-3-27b", "my-embedding-model"]


def test_fetch_models_anthropic():
    """_fetch_models for anthropic returns all models sorted."""
    import sys

    mock_model_1 = MagicMock()
    mock_model_1.id = "claude-sonnet-4-5-20250929"
    mock_model_2 = MagicMock()
    mock_model_2.id = "claude-haiku-4-5-20251001"

    mock_client = MagicMock()
    mock_client.models.list.return_value = MagicMock(data=[mock_model_1, mock_model_2])

    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
        result = _fetch_models("anthropic", "sk-ant-test")

    mock_anthropic_module.Anthropic.assert_called_once_with(api_key="sk-ant-test", timeout=10.0)
    assert result == ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929"]


def test_fetch_models_google():
    """_fetch_models for google returns models that support generateContent."""
    mock_model_ok = MagicMock()
    mock_model_ok.name = "models/gemini-2.0-flash"
    mock_model_ok.supported_actions = ["generateContent", "countTokens"]
    mock_model_bad = MagicMock()
    mock_model_bad.name = "models/text-embedding-004"
    mock_model_bad.supported_actions = ["embedContent"]

    mock_client = MagicMock()
    mock_client.models.list.return_value = [mock_model_ok, mock_model_bad]

    with patch("google.genai.Client", return_value=mock_client):
        result = _fetch_models("google", "test-key")

    assert result == ["gemini-2.0-flash"]


def test_fetch_models_groq():
    """_fetch_models for groq uses openai SDK with groq base URL."""
    mock_models = [
        MagicMock(id="llama-3.3-70b-versatile"),
        MagicMock(id="whisper-large-v3-turbo"),
    ]
    mock_client = MagicMock()
    mock_client.models.list.return_value = mock_models

    with patch("openai.OpenAI", return_value=mock_client) as mock_cls:
        result = _fetch_models("groq", "gsk-test")

    mock_cls.assert_called_once_with(
        api_key="gsk-test",
        base_url="https://api.groq.com/openai/v1",
        timeout=10.0,
    )
    assert result == ["llama-3.3-70b-versatile"]


def test_fetch_models_ollama():
    """_fetch_models for ollama uses openai SDK with ollama base URL."""
    mock_models = [MagicMock(id="llama3:latest"), MagicMock(id="qwen2:7b")]
    mock_client = MagicMock()
    mock_client.models.list.return_value = mock_models

    with patch("openai.OpenAI", return_value=mock_client) as mock_cls:
        result = _fetch_models("ollama", "", base_url="http://localhost:11434")

    mock_cls.assert_called_once_with(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        timeout=10.0,
    )
    assert result == ["llama3:latest", "qwen2:7b"]


def test_fetch_models_ollama_does_not_double_v1():
    """A base URL typed with /v1 must not list models from /v1/v1."""
    mock_client = MagicMock()
    mock_client.models.list.return_value = [MagicMock(id="llama3:latest")]

    with patch("openai.OpenAI", return_value=mock_client) as mock_cls:
        result = _fetch_models("ollama", "", base_url="http://localhost:11434/v1")

    mock_cls.assert_called_once_with(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        timeout=10.0,
    )
    assert result == ["llama3:latest"]


# --- LLM connection test: the adapter extraction uses ----------------------


class _Completion:
    """The fake client's answer; awaitable, like the async Instructor client's."""

    reply = "OK"

    def __await__(self):
        async def _result():
            return self

        return _result().__await__()


class _FakeInstructor:
    """Stands in for ``instructor.from_provider``; each create() fails or answers in turn."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.clients: list[tuple[str, dict]] = []
        self.requests: list[dict] = []

    def __call__(self, model, **kwargs):
        self.clients.append((model, kwargs))
        return self

    def create(self, **kwargs):
        self.requests.append(
            {
                k: v
                for k, v in kwargs.items()
                if k not in {"response_model", "messages", "max_retries"}
            }
        )
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        return _Completion()


def test_connection_test_sends_ollama_requests_to_the_v1_api(monkeypatch):
    wizard = _recording_wizard()
    wizard.env_vars = {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "gpt-oss:20b",
        "LLM_OLLAMA_BASE_URL": "http://gpu-box:11434",
    }
    fake = _FakeInstructor()
    monkeypatch.setattr("instructor.from_provider", fake)
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: True)

    wizard._offer_llm_connection_test()

    assert fake.clients == [
        ("ollama/gpt-oss:20b", {"async_client": True, "base_url": "http://gpu-box:11434/v1"})
    ]
    assert "Connected — LLM responded: OK" in wizard.console.export_text()


def test_connection_test_retry_for_ollama_asks_for_the_url_not_a_key(monkeypatch):
    """Ollama has no key: a retry must not write '=<answer>' into .env."""
    wizard = _recording_wizard()
    wizard.env_vars = {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "gpt-oss:20b",
        "LLM_OLLAMA_BASE_URL": "http://localhost:11434",
    }
    fake = _FakeInstructor(ConnectionError("connection refused"))
    confirms = []
    prompts = []
    monkeypatch.setattr("instructor.from_provider", fake)
    monkeypatch.setattr(
        "bibr.setup_wizard.Confirm.ask", lambda text, **k: confirms.append(text) or True
    )
    monkeypatch.setattr(
        "bibr.setup_wizard.Prompt.ask",
        lambda text, **k: prompts.append(text) or "http://gpu-box:11434",
    )

    wizard._offer_llm_connection_test()

    assert confirms == ["Test the LLM connection now?", "Retry with a different Ollama base URL?"]
    assert prompts == ["Ollama base URL"]
    assert wizard.env_vars == {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "gpt-oss:20b",
        "LLM_OLLAMA_BASE_URL": "http://gpu-box:11434",
    }
    # The retry tests the URL just typed, not the first one again.
    assert [kw["base_url"] for _model, kw in fake.clients] == [
        "http://localhost:11434/v1",
        "http://gpu-box:11434/v1",
    ]
    assert len(fake.requests) == 2


def test_connection_test_sends_the_google_adapter_request(monkeypatch):
    """The recommended cloud setup tests gemini-3.5-flash-lite, which cannot turn
    thinking off: the test must send the adapter's thinking budget, as chew does.

    It must also test the key just typed. The current settings hold another
    GOOGLE_API_KEY and an LLM_API_KEY, which the Google adapter would send in
    its place; the wizard writes LLM_API_KEY blank, so chew will not send it.
    """
    import bibr.config

    wizard = _recording_wizard()
    wizard.env_vars = {
        "LLM_PROVIDER": "google",
        "LLM_MODEL": "gemini-3.5-flash-lite",
        "GOOGLE_API_KEY": "AIza-typed-key-1234567890",
    }
    monkeypatch.setattr(bibr.config.Settings, "GOOGLE_API_KEY", "AIza-older-key-placeholder")
    monkeypatch.setattr(bibr.config.Settings.llm, "api_key", "sk-older-openai-key-1234")
    fake = _FakeInstructor()
    monkeypatch.setattr("instructor.from_provider", fake)
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: True)

    wizard._offer_llm_connection_test()

    assert fake.clients == [
        (
            "google/gemini-3.5-flash-lite",
            {"async_client": True, "api_key": "AIza-typed-key-1234567890"},
        )
    ]
    assert fake.requests == [
        {
            "generation_config": {"temperature": 0.0, "max_tokens": 4096},
            "thinking_config": {"thinking_budget": 1},
        }
    ]


@pytest.mark.parametrize(
    ("typed_url", "expected_url"),
    [("http://gpu-box:8000/v1", "http://gpu-box:8000/v1"), ("", None)],
)
def test_connection_test_uses_the_typed_openai_key_and_server(monkeypatch, typed_url, expected_url):
    """An older LLM_API_KEY or LLM_BASE_URL must not replace what was typed.

    A blank custom URL means OpenAI itself: the older server is not tested,
    and the wizard writes LLM_BASE_URL blank so chew does not use it either.
    """
    import bibr.config

    wizard = _recording_wizard()
    wizard.env_vars = {
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-5-nano",
        "LLM_API_KEY": "sk-typed-key-1234567890",
    }
    if typed_url:
        wizard.env_vars["LLM_BASE_URL"] = typed_url
    monkeypatch.setattr(bibr.config.Settings.llm, "api_key", "sk-older-key-placeholder")
    monkeypatch.setattr(bibr.config.Settings.llm, "base_url", "http://older-server:8000/v1")
    fake = _FakeInstructor()
    monkeypatch.setattr("instructor.from_provider", fake)
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: True)

    wizard._offer_llm_connection_test()

    [(model, kwargs)] = fake.clients
    assert model == "openai/gpt-5-nano"
    assert kwargs["api_key"] == "sk-typed-key-1234567890"
    assert kwargs.get("base_url") == expected_url


def test_connection_test_stops_on_an_invalid_saved_value(monkeypatch):
    """A bad value in the existing config is not a connection problem: no key retry."""
    from bibr.exceptions import ConfigurationError

    wizard = _recording_wizard()
    wizard.env_vars = {
        "LLM_PROVIDER": "google",
        "LLM_MODEL": "gemini-3.5-flash-lite",
        "GOOGLE_API_KEY": "AIza-typed-key-1234567890",
    }

    def invalid(*a, **k):
        raise ConfigurationError("LLM_MAX_TOKENS=abc is invalid: expected an integer")

    confirms = []
    fake = _FakeInstructor()
    monkeypatch.setattr("bibr.config.snapshot_settings", invalid)
    monkeypatch.setattr("instructor.from_provider", fake)
    monkeypatch.setattr(
        "bibr.setup_wizard.Confirm.ask", lambda text, **k: confirms.append(text) or True
    )

    wizard._offer_llm_connection_test()

    text = wizard.console.export_text()
    assert "Can't test the connection: LLM_MAX_TOKENS=abc is invalid" in text
    assert "`bibr chew` stops on it too" in text
    assert confirms == ["Test the LLM connection now?"]
    assert fake.clients == []


def _settings_from(env_file, monkeypatch):
    """The settings ``bibr chew`` would load from ``env_file`` alone."""
    from bibr.config import GlobalSettings

    for name in ("LLM_PROVIDER", "LLM_BACKEND", "LLM_API_KEY", "LLM_BASE_URL", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BIBR_ENV_FILE", str(env_file))
    return GlobalSettings()


def test_merge_leaves_no_older_llm_key_server_or_backend_in_effect(tmp_path, monkeypatch):
    """Switching to Google must not keep sending an older LLM_API_KEY, or running vLLM.

    The Google adapter prefers LLM_API_KEY to GOOGLE_API_KEY, and a merge keeps
    every key the wizard does not write.
    """
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_PROVIDER=openai\nLLM_API_KEY=sk-older-openai-key\n"
        "LLM_BASE_URL=http://older-server/v1\nLLM_BACKEND=vllm\nLLM_RATE_LIMIT_RPM=7\n",
        encoding="utf-8",
    )
    wizard = _recording_wizard()
    wizard.env_path = env_path
    wizard.env_vars = {
        "LLM_PROVIDER": "google",
        "LLM_MODEL": "gemini-3.5-flash-lite",
        "GOOGLE_API_KEY": "AIza-typed-key-1234567890",
    }
    answers = iter(["merge"])
    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *a, **k: next(answers))
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: False)

    wizard._step_write_env()

    settings = _settings_from(env_path, monkeypatch)
    assert settings.llm.backend == "cloud"
    assert settings.llm.provider == "google"
    assert (settings.llm.api_key or settings.GOOGLE_API_KEY) == "AIza-typed-key-1234567890"
    assert settings.llm.rate_limit_rpm == 7  # other hand-set values survive the merge
    # The answers themselves (and so a saved preset) are unchanged.
    assert "LLM_API_KEY" not in wizard.env_vars


@pytest.mark.parametrize(
    ("answers", "written"),
    [
        (
            {"LLM_PROVIDER": "google", "GOOGLE_API_KEY": "AIza-typed"},
            {"LLM_BACKEND": "cloud", "LLM_API_KEY": ""},
        ),
        (
            {"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-typed"},
            {"LLM_BACKEND": "cloud", "LLM_API_KEY": "sk-typed", "LLM_BASE_URL": ""},
        ),
        (
            {"LLM_PROVIDER": "ollama", "LLM_OLLAMA_BASE_URL": "http://localhost:11434"},
            {"LLM_BACKEND": "cloud"},
        ),
        ({"LLM_BACKEND": "vllm", "LLM_LOCAL_MODEL": "org/model"}, {"LLM_BACKEND": "vllm"}),
    ],
)
def test_fresh_env_pins_the_llm_routing_settings(tmp_path, answers, written):
    """A fresh ./.env overrides ~/.bibr/.env, so it must name the key and server too."""
    from bibr.env_utils import parse_env

    wizard = _recording_wizard()
    wizard.env_path = tmp_path / ".env"
    wizard.env_vars = dict(answers)

    with patch("bibr.setup_wizard.Confirm.ask", return_value=False):
        wizard._step_write_env()

    env = parse_env(wizard.env_path)
    routing = {"LLM_BACKEND", "LLM_API_KEY", "LLM_BASE_URL"}
    assert {k: v for k, v in env.items() if k in routing} == written


def test_fetch_models_returns_empty_on_error():
    """_fetch_models returns empty list when API call fails."""
    with patch("openai.OpenAI", side_effect=Exception("connection refused")):
        result = _fetch_models("openai", "sk-test")

    assert result == []


def test_select_model_single_model():
    """Single model in list should auto-select without prompting."""
    result = _select_model(["gpt-4o"], default="gpt-4o", console=MagicMock())
    assert result == "gpt-4o"


def test_select_model_empty_list():
    """Empty list returns None (caller handles fallback)."""
    result = _select_model([], default="gpt-4o", console=MagicMock())
    assert result is None


def test_select_model_questionary_select():
    """Multiple models should use questionary.select."""
    models = ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929"]

    with patch("questionary.select") as mock_select:
        mock_select.return_value.ask.return_value = "claude-sonnet-4-5-20250929"
        result = _select_model(models, default="claude-haiku-4-5-20251001", console=MagicMock())

    assert result == "claude-sonnet-4-5-20250929"
    mock_select.assert_called_once()
    call_kwargs = mock_select.call_args
    choices = call_kwargs.kwargs.get("choices") or call_kwargs[1].get("choices")
    assert choices[-1] == "Enter model name manually..."
    default_val = call_kwargs.kwargs.get("default") or call_kwargs[1].get("default")
    assert default_val == "claude-haiku-4-5-20251001"


def test_select_model_manual_entry():
    """Choosing 'Enter model name manually...' returns None (caller handles)."""
    models = ["gpt-4o", "gpt-4o-mini"]

    with patch("questionary.select") as mock_select:
        mock_select.return_value.ask.return_value = "Enter model name manually..."
        result = _select_model(models, default="gpt-4o", console=MagicMock())

    assert result is None


def test_select_model_ctrl_c_returns_none():
    """Ctrl+C during questionary returns None."""
    models = ["gpt-4o", "gpt-4o-mini"]

    with patch("questionary.select") as mock_select:
        mock_select.return_value.ask.return_value = None
        result = _select_model(models, default="gpt-4o", console=MagicMock())

    assert result is None


def test_available_extras_windows_steers_to_ml_not_local_cuda(monkeypatch):
    """Native Windows should guide users to ml + glm-llama."""
    monkeypatch.setattr("bibr.setup_wizard.sys.platform", "win32")

    extras = _available_extras()

    assert "ml" in extras
    assert "local" not in extras
    assert "local-cuda" not in extras
    assert "glm-llama" in extras["ml"]


def test_step_extras_failed_uv_sync_stops_setup(monkeypatch):
    """A failed dependency install must stop before later smoke tests."""
    wizard = _recording_wizard()
    monkeypatch.setattr("bibr.setup_wizard.sys.platform", "win32")
    monkeypatch.setattr("bibr.setup_wizard.shutil.which", lambda name: "uv")
    monkeypatch.setattr(
        "bibr.setup_wizard.Confirm.ask",
        lambda prompt, **kwargs: "ml" in prompt,
    )
    result = MagicMock(returncode=1, stderr="No solution found")
    monkeypatch.setattr("bibr.setup_wizard.subprocess.run", MagicMock(return_value=result))

    with pytest.raises(SystemExit):
        wizard._step_extras()

    text = wizard.console.export_text()
    assert "uv sync failed" in text
    assert "No solution found" in text


def _gpu_build_run(
    *, reinstall_returncode=0, reinstall_stderr="", probe_stdout="True\n", probe_stderr=""
):
    """A subprocess.run stub for the gpu step: the install, the reinstall, then the probe."""

    def run(cmd, **kwargs):
        if cmd[:2] == [sys.executable, "-c"]:
            return MagicMock(returncode=0, stdout=probe_stdout, stderr=probe_stderr)
        if "--reinstall-package" in cmd or "--force-reinstall" in cmd:
            return MagicMock(returncode=reinstall_returncode, stdout="", stderr=reinstall_stderr)
        return MagicMock(returncode=0, stdout="", stderr="")

    return MagicMock(side_effect=run)


def test_gpu_extra_reinstalls_the_synced_onnxruntime_gpu_last(monkeypatch, tmp_path):
    """The sync writes onnxruntime and onnxruntime-gpu at once, so either can win;
    the wizard reinstalls the version the sync chose and checks the GPU build loads."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "bibr"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("bibr.setup_wizard.shutil.which", lambda name: "/bin/uv")
    monkeypatch.setattr("bibr.setup_wizard.importlib.metadata.version", lambda name: "1.26.0")
    run = _gpu_build_run()
    monkeypatch.setattr("bibr.setup_wizard.subprocess.run", run)
    wizard = _recording_wizard()
    wizard.selected_extras = {"gpu"}

    wizard._install_selected_extras()

    commands = [call.args[0] for call in run.call_args_list]
    assert commands[0] == ["/bin/uv", "sync", "--inexact", "--extra=gpu"]
    assert commands[1] == [
        "/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall-package",
        "onnxruntime-gpu",
        "onnxruntime-gpu[cuda,cudnn]==1.26.0",
    ]
    assert commands[2][:2] == [sys.executable, "-c"]
    assert "CUDAExecutionProvider" in commands[2][2]
    assert "onnxruntime-gpu 1.26.0 is the onnxruntime build that loads" in (
        wizard.console.export_text()
    )


def test_gpu_build_reinstall_uses_pip_without_uv(monkeypatch):
    monkeypatch.setattr("bibr.setup_wizard.importlib.metadata.version", lambda name: "1.30.0")
    run = _gpu_build_run()
    monkeypatch.setattr("bibr.setup_wizard.subprocess.run", run)
    wizard = _recording_wizard()

    wizard._reinstall_onnxruntime_gpu(None)

    assert run.call_args_list[0].args[0] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        "onnxruntime-gpu==1.30.0",
    ]


def test_gpu_build_reinstall_reports_a_cpu_build_that_still_loads(monkeypatch):
    monkeypatch.setattr("bibr.setup_wizard.importlib.metadata.version", lambda name: "1.26.0")
    monkeypatch.setattr(
        "bibr.setup_wizard.subprocess.run",
        _gpu_build_run(probe_stdout="False\n"),
    )
    wizard = _recording_wizard()

    wizard._reinstall_onnxruntime_gpu("/bin/uv")

    text = wizard.console.export_text()
    assert "onnxruntime-gpu 1.26.0 is not the onnxruntime build that loads" in text
    assert "The CPU build still loads." in text
    assert "--reinstall-package onnxruntime-gpu 'onnxruntime-gpu[cuda,cudnn]==1.26.0'" in text


def test_gpu_build_reinstall_reports_an_import_failure(monkeypatch):
    monkeypatch.setattr("bibr.setup_wizard.importlib.metadata.version", lambda name: "1.26.0")
    monkeypatch.setattr(
        "bibr.setup_wizard.subprocess.run",
        _gpu_build_run(
            probe_stdout="",
            probe_stderr="Traceback (most recent call last):\n"
            "ImportError: libcudart.so.13: cannot open shared object file",
        ),
    )
    wizard = _recording_wizard()

    wizard._reinstall_onnxruntime_gpu("/bin/uv")

    text = wizard.console.export_text()
    assert "ImportError: libcudart.so.13" in text
    assert "Traceback" not in text


def test_gpu_build_reinstall_failure_skips_the_probe(monkeypatch):
    monkeypatch.setattr("bibr.setup_wizard.importlib.metadata.version", lambda name: "1.26.0")
    run = _gpu_build_run(reinstall_returncode=2, reinstall_stderr="error: Failed to download")
    monkeypatch.setattr("bibr.setup_wizard.subprocess.run", run)
    wizard = _recording_wizard()

    wizard._reinstall_onnxruntime_gpu("/bin/uv")

    assert run.call_count == 1
    text = wizard.console.export_text()
    assert "error: Failed to download" in text
    assert "Run: /bin/uv pip install" in text


def test_gpu_build_reinstall_without_onnxruntime_gpu_installed(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def not_installed(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("bibr.setup_wizard.importlib.metadata.version", not_installed)
    run = MagicMock()
    monkeypatch.setattr("bibr.setup_wizard.subprocess.run", run)
    wizard = _recording_wizard()

    wizard._reinstall_onnxruntime_gpu("/bin/uv")

    run.assert_not_called()
    assert "onnxruntime-gpu is not installed" in wizard.console.export_text()


def _install_command_for_extras_for_test(extras, *, cwd, uv_bin):
    from bibr.setup_wizard import _install_command_for_extras

    return _install_command_for_extras(extras, cwd=cwd, uv_bin=uv_bin)


def test_install_command_source_checkout_uses_uv_sync(tmp_path):
    """Inside bibr's own source tree, extras are project extras selected by uv sync."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "bibr"\n', encoding="utf-8")

    command, label = _install_command_for_extras_for_test(
        {"ml", "local"}, cwd=tmp_path, uv_bin="/bin/uv"
    )

    # --inexact keeps packages installed outside these extras (e.g. --extra all).
    assert command == ["/bin/uv", "sync", "--inexact", "--extra=local", "--extra=ml"]
    assert "source checkout" in label


def test_install_command_consumer_uv_project_uses_uv_add(tmp_path):
    """After `uv add bibr`, setup must add extras to the consuming project."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "paper-lab"\n', encoding="utf-8")

    command, label = _install_command_for_extras_for_test(
        {"ml", "local"}, cwd=tmp_path, uv_bin="/bin/uv"
    )

    assert command == ["/bin/uv", "add", "bibr[local,ml]"]
    assert "project dependency" in label


def test_install_command_without_uv_uses_current_python_pip(tmp_path):
    """Plain pip installs should not be forced through uv."""
    command, label = _install_command_for_extras_for_test({"ml"}, cwd=tmp_path, uv_bin=None)

    assert command[:3] == [sys.executable, "-m", "pip"]
    assert command[-1] == "bibr[ml]"
    assert "current Python" in label


def test_step_llm_provider_fetches_models_after_credentials():
    """Step 2 should collect credentials first, then fetch and select model."""
    wizard = SetupWizard()

    def mock_prompt_ask(prompt_text, **kwargs):
        if "Provider" in prompt_text:
            return "openai"
        if "API key" in prompt_text:
            return "sk-test"
        if "base URL" in prompt_text:
            return ""
        if "Model name" in prompt_text:
            return "gpt-4o"
        return kwargs.get("default", "")

    with (
        patch("bibr.setup_wizard.Prompt.ask", side_effect=mock_prompt_ask),
        patch(
            "bibr.setup_wizard._fetch_models",
            return_value=["gpt-4o", "gpt-4o-mini", "gpt-5-nano"],
        ) as mock_fetch,
        patch(
            "bibr.setup_wizard._select_model",
            return_value="gpt-4o",
        ) as mock_select,
    ):
        wizard._step_llm_provider()

    assert wizard.env_vars["LLM_PROVIDER"] == "openai"
    assert wizard.env_vars["LLM_MODEL"] == "gpt-4o"
    assert wizard.env_vars["LLM_API_KEY"] == "sk-test"
    mock_fetch.assert_called_once_with("openai", "sk-test", "")
    mock_select.assert_called_once()


def test_step_llm_provider_fallback_on_fetch_failure():
    """When model fetch returns empty, fall back to manual Prompt.ask."""
    wizard = SetupWizard()

    def mock_prompt_ask(prompt_text, **kwargs):
        if "Provider" in prompt_text:
            return "openai"
        if "API key" in prompt_text:
            return "sk-test"
        if "base URL" in prompt_text:
            return ""
        if "Model name" in prompt_text:
            return "gpt-5-nano"
        return kwargs.get("default", "")

    with (
        patch("bibr.setup_wizard.Prompt.ask", side_effect=mock_prompt_ask),
        patch("bibr.setup_wizard._fetch_models", return_value=[]),
        patch("bibr.setup_wizard._select_model") as mock_select,
    ):
        wizard._step_llm_provider()

    mock_select.assert_not_called()
    assert wizard.env_vars["LLM_MODEL"] == "gpt-5-nano"


def test_step_llm_provider_ollama_flow():
    """Ollama flow: collect base URL, fetch models, set rate limit."""
    wizard = SetupWizard()

    def mock_prompt_ask(prompt_text, **kwargs):
        if "Provider" in prompt_text:
            return "ollama"
        if "Ollama base URL" in prompt_text:
            return "http://localhost:11434"
        if "Model name" in prompt_text:
            return "llama3:latest"
        return kwargs.get("default", "")

    with (
        patch("bibr.setup_wizard.Prompt.ask", side_effect=mock_prompt_ask),
        patch(
            "bibr.setup_wizard._fetch_models",
            return_value=["llama3:latest", "qwen2:7b"],
        ),
        patch("bibr.setup_wizard._select_model", return_value="llama3:latest"),
    ):
        wizard._step_llm_provider()

    assert wizard.env_vars["LLM_PROVIDER"] == "ollama"
    assert wizard.env_vars["LLM_MODEL"] == "llama3:latest"
    assert wizard.env_vars["LLM_OLLAMA_BASE_URL"] == "http://localhost:11434"
    assert wizard.env_vars["LLM_RATE_LIMIT_RPM"] == "10"


def test_step_write_env_offers_preset_save(tmp_path):
    """After writing .env, wizard offers to save as a preset."""
    wizard = SetupWizard()
    wizard.env_path = tmp_path / ".env"
    wizard.env_vars = {"LLM_PROVIDER": "google", "LLM_MODEL": "gemini"}

    with (
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
        patch("bibr.setup_wizard.Prompt.ask", return_value="my-config"),
        patch("bibr.setup_wizard.PresetManager") as MockManager,
    ):
        mock_mgr = MockManager.return_value
        wizard._step_write_env()

    mock_mgr.save.assert_called_once_with("my-config", wizard.env_vars)


def test_easy_setup_accepts_recommendation_writes_env_and_preset(tmp_path, monkeypatch):
    wizard = _recording_wizard()
    wizard.env_path = tmp_path / ".env"
    setup = RecommendedSetup(
        tier="fully_local",
        env={
            "OCR_BACKEND": "glm-llama",
            "LLM_BACKEND": "vllm",
            "LLM_LOCAL_MODEL": "numind/NuExtract3",
            "WTPSPLIT_MODEL": "sat-6l-sm",
            "REF_SEG_STRATEGY": "geom",
            "REF_PARSE_STRATEGY": "ner",
            "PIPELINE_MEMORY_MODE": "balanced",
            "CROSSREF_CONSOLIDATE": "off",
        },
        extras={"ml"},
        preset_name="recommended-local",
        privacy_summary="document contents stay on this machine",
        download_summary="downloads several GB",
        runtime_summary="can be slow",
        required_prompts=("install_extras", "smoke_test"),
    )
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: ("cuda", 24.0))
    monkeypatch.setattr("bibr.local.pipeline._get_system_memory_gb", lambda: 64.0)
    monkeypatch.setattr("bibr.setup_wizard._build_recommended_setup", lambda **k: setup)
    monkeypatch.setattr("bibr.setup_wizard._ml_extra_available", lambda: True)
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: True)
    monkeypatch.setattr(wizard, "_install_selected_extras", lambda *a, **k: None)
    monkeypatch.setattr(wizard, "_test_local_server", lambda: None)
    monkeypatch.setattr(wizard, "_step_smoke_test", lambda *a, **k: None)

    with patch("bibr.setup_wizard.PresetManager") as MockManager:
        wizard.run()

    content = wizard.env_path.read_text()
    assert "OCR_BACKEND=glm-llama" in content
    assert "LLM_BACKEND=vllm" in content
    MockManager.return_value.save.assert_called_once()
    assert MockManager.return_value.save.call_args.args[0] == "recommended-local"
    assert "experimental" in wizard.console.export_text().lower()


def test_easy_setup_decline_then_decline_advanced_exits_clean(monkeypatch):
    """Declining the plan, then declining the --advanced handoff, exits cleanly."""
    wizard = _recording_wizard()
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: ("mlx", 32.0))
    monkeypatch.setattr("bibr.local.pipeline._get_system_memory_gb", lambda: 32.0)
    monkeypatch.setattr(
        "bibr.setup_wizard._build_recommended_setup",
        lambda **k: RecommendedSetup(
            tier="fully_local",
            env={"OCR_BACKEND": "glm-llama"},
            extras=set(),
            preset_name="recommended-local",
            privacy_summary="document contents stay on this machine",
            download_summary="downloads several GB",
            runtime_summary="can be slow",
            required_prompts=(),
        ),
    )
    # Decline "Use this configuration?" and then "Switch to the advanced wizard?".
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: False)

    wizard.run()  # must not raise SystemExit

    assert wizard.env_vars == {}
    assert "bibr setup --advanced" in wizard.console.export_text()


def test_easy_private_server_prompts_for_urls(tmp_path, monkeypatch):
    wizard = _recording_wizard()
    wizard.env_path = tmp_path / ".env"
    setup = RecommendedSetup(
        tier="private_server",
        env={
            "OCR_BACKEND": "glm-http",
            "WTPSPLIT_MODEL": "sat-6l-sm",
            "REF_SEG_STRATEGY": "geom",
            "REF_PARSE_STRATEGY": "ner",
            "PIPELINE_MEMORY_MODE": "aggressive",
            "CROSSREF_CONSOLIDATE": "off",
        },
        extras=set(),
        preset_name="private-server",
        privacy_summary="document contents stay private",
        download_summary="downloads local models",
        runtime_summary="uses your server",
        required_prompts=("private_server_url",),
    )
    prompts = iter(
        ["https://ocr.internal:8080", "https://llm.internal/v1", "local-key", "nuextract"]
    )
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: (None, None))
    monkeypatch.setattr("bibr.local.pipeline._get_system_memory_gb", lambda: 8.0)
    monkeypatch.setattr("bibr.setup_wizard._build_recommended_setup", lambda **k: setup)
    monkeypatch.setattr("bibr.setup_wizard.PresetManager", MagicMock())
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: True)
    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *a, **k: next(prompts))

    wizard.run()

    content = wizard.env_path.read_text()
    assert "OCR_BASE_URL=https://ocr.internal:8080" in content
    assert "LLM_BASE_URL=https://llm.internal/v1" in content
    assert "LLM_API_KEY=local-key" in content
    assert "LLM_MODEL=nuextract" in content


def test_easy_private_server_default_llm_url_stays_private(monkeypatch):
    wizard = _recording_wizard()
    setup = RecommendedSetup(
        tier="private_server",
        env={"OCR_BACKEND": "glm-http"},
        extras=set(),
        preset_name="private-server",
        privacy_summary="document contents stay private",
        download_summary="downloads local models",
        runtime_summary="uses your server",
        required_prompts=("private_server_url",),
    )

    def prompt_default(*args, **kwargs):
        return kwargs.get("default", "")

    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", prompt_default)

    wizard._collect_required_prompts(setup)

    assert wizard.env_vars["LLM_PROVIDER"] == "openai"
    assert wizard.env_vars["LLM_BASE_URL"] == "http://localhost:8000/v1"


def test_step_external_services_llm_choice():
    """Choosing the full-precision LLM parser writes REF_PARSE_STRATEGY."""
    wizard = SetupWizard()
    # Prompt order: crossref email, wtpsplit model, OCR backend, ref parsing
    with (
        patch(
            "bibr.setup_wizard.Prompt.ask",
            side_effect=["", "sat-6l-sm", "glm-rapid-mlx", "llm"],
        ),
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
    ):
        wizard._step_external_services()
    assert wizard.env_vars.get("REF_PARSE_STRATEGY") == "llm"


def test_step_external_services_default_refs_writes_nothing():
    """The default ner strategy must not clutter .env."""
    wizard = SetupWizard()
    with (
        patch(
            "bibr.setup_wizard.Prompt.ask",
            side_effect=["", "sat-6l-sm", "glm-rapid-mlx", "ner"],
        ),
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
    ):
        wizard._step_external_services()
    assert "REF_PARSE_STRATEGY" not in wizard.env_vars


def test_step_external_services_accepts_automatic_paddle_and_explicit_glm():
    wizard = SetupWizard()
    with (
        patch(
            "bibr.setup_wizard.Prompt.ask",
            side_effect=["", "sat-6l-sm", "paddle", "ner"],
        ),
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
    ):
        wizard._step_external_services()
    assert wizard.env_vars["OCR_BACKEND"] == "paddle"

    explicit_glm = SetupWizard()
    with (
        patch(
            "bibr.setup_wizard.Prompt.ask",
            side_effect=["", "sat-6l-sm", "glm-http", "ner"],
        ),
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
    ):
        explicit_glm._step_external_services()
    assert explicit_glm.env_vars["OCR_BACKEND"] == "glm-http"


def test_external_services_auto_installs_ml_for_layout_and_ner(monkeypatch):
    """PDF layout and NER references both explain the required ml extra."""
    wizard = _recording_wizard()
    prompts = iter(["", "sat-6l-sm", "glm-llama", "ner"])
    calls = []
    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: False)
    monkeypatch.setattr("bibr.setup_wizard._ml_extra_available", lambda: False)
    monkeypatch.setattr(wizard, "_install_selected_extras", lambda reason="": calls.append(reason))

    wizard._step_external_services()

    assert "ml" in wizard.selected_extras
    assert calls == ["PDF layout detection and NER reference parsing run fastest with the ml extra"]


@pytest.mark.parametrize(
    "ocr_backend",
    [
        "paddle",
        "paddle-vllm",
        "paddle-rapid-mlx",
        "paddle-mlx-vlm",
        "paddle-http",
        "glm-llama",
        "glm-rapid-mlx",
        "glm-http",
        "gemini",
        "openai",
        "anthropic",
    ],
)
def test_external_services_auto_installs_ml_for_layout_with_llm_references(
    monkeypatch, ocr_backend
):
    """Every advanced PDF/OCR backend installs layout ML even when references use an LLM."""
    wizard = _recording_wizard()
    answers = ["", "sat-6l-sm", ocr_backend, "llm"]
    prompts = iter(answers)
    calls = []
    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: False)
    monkeypatch.setattr("bibr.setup_wizard._ml_extra_available", lambda: False)
    monkeypatch.setattr(wizard, "_install_selected_extras", lambda reason="": calls.append(reason))

    wizard._step_external_services()

    assert "ml" in wizard.selected_extras
    assert calls == ["PDF layout detection runs fastest with the ml extra"]


def test_step_external_services_no_longer_offers_glm_mlx():
    """The disabled vllm-mlx backend must not be a pickable choice (nor ask its
    quantization question): every run following that choice failed at startup."""
    wizard = SetupWizard()
    seen: dict[str, list[str]] = {}

    def fake_ask(prompt, *args, **kwargs):
        if prompt == "OCR backend":
            seen["choices"] = list(kwargs.get("choices") or [])
            return "glm-rapid-mlx"
        if prompt == "OCR quantization":
            raise AssertionError("the glm-mlx quantization prompt must be gone")
        if prompt == "Reference parsing":
            return "ner"
        return kwargs.get("default", "")

    with (
        patch("bibr.setup_wizard.Prompt.ask", side_effect=fake_ask),
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
    ):
        wizard._step_external_services()

    assert "glm-mlx" not in seen["choices"]
    assert "glm-rapid-mlx" in seen["choices"]
    assert wizard.env_vars["OCR_BACKEND"] == "glm-rapid-mlx"
    assert "OCR_VLLM_MLX_MODEL" not in wizard.env_vars


def test_step_external_services_ocr_quant_default_writes_nothing():
    """The default 8-bit quant must not clutter .env, and non-mlx backends skip it."""
    wizard = SetupWizard()
    # glm-http has no quant prompt: email, wtpsplit, OCR backend, ref parsing
    with (
        patch("bibr.setup_wizard.Prompt.ask", side_effect=["", "sat-6l-sm", "glm-http", "ner"]),
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
    ):
        wizard._step_external_services()
    assert "OCR_VLLM_MLX_MODEL" not in wizard.env_vars


def _quiet_wizard():
    from rich.console import Console

    wizard = SetupWizard()
    wizard.console = Console(quiet=True)
    wizard.selected_extras = set()
    return wizard


def test_external_services_writes_enrich_consolidate_and_llm(monkeypatch):
    wizard = _quiet_wizard()
    # Prompt order: crossref email, wtpsplit model, ocr backend, ref parsing
    prompts = iter(["user@example.com", "sat-6l-sm", "glm-http", "llm"])
    # Confirm order: enrichment, consolidate, layout detection
    confirms = iter([True, True, True])
    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: next(confirms))

    wizard._step_external_services()

    assert wizard.env_vars["CROSSREF_ENRICH"] == "true"
    assert wizard.env_vars["CROSSREF_CONSOLIDATE"] == "fill"
    assert wizard.env_vars["REF_PARSE_STRATEGY"] == "llm"


def test_external_services_enrich_declined_skips_consolidate_question(monkeypatch):
    """Enrichment is opt-in; declining it leaves CROSSREF_ENRICH unset (off) and
    never asks about consolidation, which would have nothing to merge."""
    wizard = _quiet_wizard()
    prompts = iter(["", "sat-6l-sm", "glm-http", "ner"])
    questions: list[str] = []

    def confirm(question, *a, **k):
        questions.append(question)
        # enrichment: no; layout detection: yes
        return "enrichment" not in question

    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", confirm)

    wizard._step_external_services()

    assert "CROSSREF_ENRICH" not in wizard.env_vars
    assert "CROSSREF_CONSOLIDATE" not in wizard.env_vars
    assert not any("consolidation" in q for q in questions)
    assert "REF_PARSE_STRATEGY" not in wizard.env_vars


def test_external_services_consolidate_declined(monkeypatch):
    wizard = _quiet_wizard()
    prompts = iter(["", "sat-6l-sm", "glm-http", "ner"])
    # Confirm order: enrichment (yes), consolidate (no), layout detection
    confirms = iter([True, False, True])
    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: next(confirms))

    wizard._step_external_services()

    assert wizard.env_vars["CROSSREF_ENRICH"] == "true"
    assert "CROSSREF_CONSOLIDATE" not in wizard.env_vars
    assert "REF_PARSE_STRATEGY" not in wizard.env_vars


def test_write_env_fresh_includes_consolidate(tmp_path):
    env_path = tmp_path / ".env"
    _write_env_fresh(env_path, {"CROSSREF_CONSOLIDATE": "fill"})
    assert "CROSSREF_CONSOLIDATE=fill" in env_path.read_text()


# ---------------------------------------------------------------------------
# Local LLM path
# ---------------------------------------------------------------------------


def test_recommended_setup_cuda_uses_fully_local_defaults():
    setup = _build_recommended_setup(
        platform_key="cuda",
        accelerator_memory_gb=24.0,
        system_memory_gb=64.0,
        sys_platform="linux",
        machine="x86_64",
    )

    assert setup.tier == "fully_local"
    assert setup.preset_name == "recommended-local"
    assert setup.env["LLM_BACKEND"] == "vllm"
    assert setup.env["OCR_BACKEND"] == "paddle"
    assert setup.env["REF_PARSE_STRATEGY"] == "ner"
    assert setup.env["REF_SEG_STRATEGY"] == "geom"
    assert setup.env["WTPSPLIT_MODEL"] == "sat-6l-sm"
    assert "OCR_ENABLE_LAYOUT" not in setup.env
    assert setup.env["PIPELINE_MEMORY_MODE"] == "balanced"
    assert setup.env["CROSSREF_CONSOLIDATE"] == "off"
    # Linux/CUDA needs the vllm extra (paddle-vllm OCR + the managed local LLM);
    # ``local`` resolves to nothing on Linux and used to leave the first chew
    # to bootstrap vLLM through ``uv tool run``.
    assert {"ml", "vllm"} <= setup.extras
    assert "local" not in setup.extras
    assert "demo" in setup.extras
    assert setup.required_prompts == ("install_extras", "smoke_test")


def test_recommended_setup_low_vram_cuda_keeps_paddle_automatic_ocr():
    setup = _build_recommended_setup(
        platform_key="cuda",
        accelerator_memory_gb=6.0,
        system_memory_gb=16.0,
        sys_platform="linux",
        machine="x86_64",
    )

    assert setup.tier == "fully_local"
    assert setup.env["OCR_BACKEND"] == "paddle"
    assert setup.env["LLM_BACKEND"] == "llama-cpp"
    assert setup.env["LLM_LOCAL_MODEL"] == "numind/NuExtract3-GGUF:Q4_K_M"
    assert "demo" in setup.extras
    assert "ml" in setup.extras
    # At 6 GB the automatic chain runs OCR and the LLM through llama.cpp, so no
    # GPU-runtime extra is installed and the plan says what to install instead.
    assert "vllm" not in setup.extras
    assert "local" not in setup.extras
    assert "local-cuda" not in setup.extras
    assert "llama.cpp" in setup.runtime_summary


def test_recommended_setup_mid_vram_cuda_splits_ocr_and_llm_runtimes():
    """8-11 GB: paddle-vllm OCR fits (8 GB gate) but NuExtract 3's vLLM build (11 GB)
    does not — the old plan picked vLLM anyway and OOMed after OCR."""
    setup = _build_recommended_setup(
        platform_key="cuda",
        accelerator_memory_gb=10.0,
        system_memory_gb=32.0,
        sys_platform="linux",
        machine="x86_64",
    )
    assert setup.env["OCR_BACKEND"] == "paddle"
    assert setup.env["LLM_BACKEND"] == "llama-cpp"
    assert setup.env["LLM_LOCAL_MODEL"] == "numind/NuExtract3-GGUF:Q4_K_M"
    assert "vllm" in setup.extras  # OCR still runs through paddle-vllm
    assert "llama.cpp for the LLM" in setup.runtime_summary
    assert "llama-server" in setup.runtime_summary


def test_recommended_setup_cuda_picks_vllm_once_the_bf16_variant_fits():
    setup = _build_recommended_setup(
        platform_key="cuda",
        accelerator_memory_gb=11.0,
        system_memory_gb=32.0,
        sys_platform="linux",
        machine="x86_64",
    )
    assert setup.env["LLM_BACKEND"] == "vllm"
    assert setup.env["LLM_LOCAL_MODEL"] == "numind/NuExtract3"
    assert "llama-server" not in setup.runtime_summary


def test_recommended_setup_apple_silicon_warns_about_runtime():
    with patch(
        "bibr.local.rapid_mlx.rapid_mlx_unavailable_reason",
        return_value="'rapid-mlx' was not found or is not executable",
    ):
        setup = _build_recommended_setup(
            platform_key="mlx",
            accelerator_memory_gb=16.0,
            system_memory_gb=16.0,
            sys_platform="darwin",
            machine="arm64",
        )

    assert setup.tier == "fully_local"
    assert setup.env["LLM_BACKEND"] == "vllm-mlx"
    assert setup.env["OCR_BACKEND"] == "paddle"
    # A 16 GB Mac resolves to balanced (OCR/LLM never coexist; models are small)
    # per the current _auto_memory_mode rule.
    assert setup.env["PIPELINE_MEMORY_MODE"] == "balanced"
    assert "demo" in setup.extras
    assert "Apple Silicon" in setup.runtime_summary
    assert "slow" in setup.runtime_summary
    # Honest about the real cost: a full extraction is tens of minutes.
    assert "15-30" in setup.runtime_summary


def test_recommended_setup_apple_silicon_prefers_rapid_mlx_when_installed():
    """Benched 2026-07-09 on M4/16GB: rapid-mlx decodes ~3-6x faster than vllm-mlx."""
    with patch("bibr.local.rapid_mlx.rapid_mlx_unavailable_reason", return_value=None):
        setup = _build_recommended_setup(
            platform_key="mlx",
            accelerator_memory_gb=16.0,
            system_memory_gb=16.0,
            sys_platform="darwin",
            machine="arm64",
        )

    assert setup.tier == "fully_local"
    assert setup.env["LLM_BACKEND"] == "rapid-mlx"
    assert setup.env["OCR_BACKEND"] == "paddle"
    # The 15-30+ minute warning is vllm-mlx's; rapid-mlx gets its own honest estimate.
    assert "rapid-mlx" in setup.runtime_summary
    assert "15-30" not in setup.runtime_summary


def test_run_validation_offers_local_server_test_for_rapid_mlx(monkeypatch):
    wizard = _recording_wizard()
    wizard.env_vars["LLM_BACKEND"] = "rapid-mlx"
    calls = []
    monkeypatch.setattr(wizard, "_offer_local_server_test", lambda: calls.append("local"))
    monkeypatch.setattr(wizard, "_offer_llm_connection_test", lambda: calls.append("cloud"))

    setup = _build_recommended_setup(
        platform_key="mlx",
        accelerator_memory_gb=16.0,
        system_memory_gb=16.0,
        sys_platform="darwin",
        machine="arm64",
    )
    wizard._run_validation(setup)
    assert calls == ["local"]


def test_recommended_setup_weak_machine_stays_private_without_cloud_consent():
    setup = _build_recommended_setup(
        platform_key=None,
        accelerator_memory_gb=None,
        system_memory_gb=8.0,
        sys_platform="linux",
        machine="x86_64",
    )

    assert setup.tier == "private_server"
    assert setup.env["OCR_BACKEND"] == "glm-http"
    assert "LLM_PROVIDER" not in setup.env
    assert "demo" in setup.extras
    assert "cloud" not in setup.privacy_summary.lower()
    assert "private_server_url" in setup.required_prompts


def test_recommended_setup_cloud_requires_explicit_consent():
    setup = _build_recommended_setup(
        platform_key=None,
        accelerator_memory_gb=None,
        system_memory_gb=8.0,
        sys_platform="linux",
        machine="x86_64",
        allow_cloud=True,
    )

    assert setup.tier == "cloud_fallback"
    assert setup.env["LLM_PROVIDER"] == "google"
    assert setup.env["OCR_BACKEND"] == "gemini"
    assert "demo" in setup.extras
    assert "cloud_api_key" in setup.required_prompts


def _feed_prompts(monkeypatch, prompts):
    """Stub sequential ``Prompt.ask`` returns; ``Confirm.ask`` -> False."""
    it = iter(prompts)
    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *a, **k: next(it))
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: False)


def _recording_wizard():
    from rich.console import Console

    wizard = SetupWizard()
    wizard.console = Console(record=True, width=100)
    wizard.selected_extras = set()
    return wizard


def test_print_done_points_to_demo_chew_and_help():
    wizard = _recording_wizard()

    wizard._print_done()

    text = wizard.console.export_text()
    assert "Setup complete!" in text
    assert "bibr demo" in text
    assert "bibr chew" in text
    assert "bibr doctor" in text
    assert "bibr -h" in text


def test_print_done_demo_hint_when_install_declined(monkeypatch):
    """Selecting the demo extra but declining the install must not suggest `bibr demo`."""
    monkeypatch.setattr("bibr.setup_wizard.importlib.util.find_spec", lambda _name: None)

    declined = _recording_wizard()
    declined.selected_extras = {"demo", "ml"}
    declined._print_done()
    assert "needs the demo extra" in declined.console.export_text()

    installed = _recording_wizard()
    installed.selected_extras = {"demo", "ml"}
    installed._extras_installed = True
    installed._print_done()
    assert "needs the demo extra" not in installed.console.export_text()


def test_local_provider_writes_registry_selection(monkeypatch):
    """Picking local + the recommended model writes backend, model id, quirk env."""
    monkeypatch.setattr("bibr.setup_wizard.sys.platform", "linux")
    wizard = _recording_wizard()
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: ("cuda", 24.0))
    _feed_prompts(monkeypatch, ["local", "nuextract3", "1"])
    wizard._step_llm_provider()
    assert wizard.env_vars["LLM_BACKEND"] == "vllm"
    assert wizard.env_vars["LLM_LOCAL_MODEL"].count("/") == 1
    assert "LLM_PROVIDER" not in wizard.env_vars


def test_local_provider_cuda_can_select_nuextract_llama_variant(monkeypatch):
    monkeypatch.setattr("bibr.setup_wizard.sys.platform", "linux")
    wizard = _recording_wizard()
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: ("cuda", 24.0))
    _feed_prompts(monkeypatch, ["local", "nuextract3", "2"])
    wizard._step_llm_provider()
    assert wizard.env_vars["LLM_BACKEND"] == "llama-cpp"
    assert wizard.env_vars["LLM_LOCAL_MODEL"] == "numind/NuExtract3-GGUF:Q4_K_M"


def test_local_provider_custom_hf_id(monkeypatch):
    monkeypatch.setattr("bibr.setup_wizard.sys.platform", "linux")
    wizard = _recording_wizard()
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: ("cuda", 24.0))
    _feed_prompts(monkeypatch, ["local", "custom", "my-org/my-model"])
    wizard._step_llm_provider()
    assert wizard.env_vars["LLM_BACKEND"] == "vllm"
    assert wizard.env_vars["LLM_LOCAL_MODEL"] == "my-org/my-model"
    assert "LLM_PROVIDER" not in wizard.env_vars


def test_local_provider_writes_quirk_env(monkeypatch):
    """gemma-4-e4b carries LLM_INSTRUCTOR_MODE=json into the written env."""
    monkeypatch.setattr("bibr.setup_wizard.sys.platform", "linux")
    wizard = _recording_wizard()
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: ("cuda", 24.0))
    _feed_prompts(monkeypatch, ["local", "gemma-4-e4b", "1"])
    wizard._step_llm_provider()
    assert wizard.env_vars["LLM_BACKEND"] == "vllm"
    assert wizard.env_vars["LLM_INSTRUCTOR_MODE"] == "json"


def test_local_provider_mac_uses_vllm_mlx_and_warns_experimental(monkeypatch):
    wizard = _recording_wizard()
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: ("mlx", 32.0))
    monkeypatch.setattr(
        "bibr.local.rapid_mlx.rapid_mlx_unavailable_reason",
        lambda executable=None: "'rapid-mlx' was not found or is not executable",
    )
    _feed_prompts(monkeypatch, ["local", "nuextract3", "1"])
    wizard._step_llm_provider()
    assert wizard.env_vars["LLM_BACKEND"] == "vllm-mlx"
    text = wizard.console.export_text().lower()
    assert "experimental" in text
    assert "hosted model" in text
    assert "cloud api" in text


def test_local_provider_no_hardware_shows_all_variants(monkeypatch):
    monkeypatch.setattr("bibr.setup_wizard.sys.platform", "linux")
    wizard = _recording_wizard()
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: (None, None))
    _feed_prompts(monkeypatch, ["local", "nuextract3", "1"])
    wizard._step_llm_provider()
    # Unknown hardware: default to the cuda variant list, unfiltered.
    assert wizard.env_vars["LLM_BACKEND"] == "vllm"


def test_local_provider_small_vram_selects_gguf_quant(monkeypatch):
    """A 6 GB CUDA card picks the llama.cpp backend + fitting NuExtract3 GGUF quant."""
    wizard = _recording_wizard()
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: ("cuda", 6.0))
    _feed_prompts(monkeypatch, ["local", "nuextract3", "1"])
    wizard._step_llm_provider()
    assert wizard.env_vars["LLM_BACKEND"] == "llama-cpp"
    assert wizard.env_vars["LLM_LOCAL_MODEL"] == "numind/NuExtract3-GGUF:Q4_K_M"
    text = wizard.console.export_text().lower()
    # The 5 GB GGUF quant fits a 6 GB card cleanly — no "doesn't fit" warning.
    assert "no variant fits" not in text


def test_test_connection_local_skips_by_default(monkeypatch):
    """Local backend: opt-in launch test defaults to skip (no server started)."""
    wizard = _recording_wizard()
    wizard.env_vars = {"LLM_BACKEND": "vllm", "LLM_LOCAL_MODEL": "numind/NuExtract3"}
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: False)
    called = []
    monkeypatch.setattr(wizard, "_test_local_server", lambda: called.append(True))
    wizard._step_test_connection()
    assert called == []


def test_test_connection_local_runs_when_confirmed(monkeypatch):
    wizard = _recording_wizard()
    wizard.env_vars = {"LLM_BACKEND": "vllm-mlx", "LLM_LOCAL_MODEL": "numind/NuExtract3-mlx-nvfp4"}
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: True)
    called = []
    monkeypatch.setattr(wizard, "_test_local_server", lambda: called.append(True))
    wizard._step_test_connection()
    assert called == [True]


# ---------------------------------------------------------------------------
# Smoke-test step (6/6)
# ---------------------------------------------------------------------------


def test_smoke_test_declines_by_default():
    wizard = _quiet_wizard()
    with (
        patch("bibr.setup_wizard.Confirm.ask", return_value=False),
        patch("bibr.api.chew") as mock_chew,
    ):
        wizard._step_smoke_test()
    mock_chew.assert_not_called()


def _smoke_export(*, n_authors: int, n_refs: int) -> dict:
    return {
        "paper_id": "sample",
        "schema_version": "12.0",
        "source": {
            "file_name": "sample_paper.pdf",
            "sha256": "5a" * 32,
            "input_format": "pdf",
        },
        "metadata": {
            "title": "The Coefficient of Rodential Efficiency: A Synthetic Benchmark",
            "keywords": [],
            "doi": None,
        },
        "author": [
            {
                "author_id": index + 1,
                "given": f"Author{index + 1}",
                "family": "Example",
                "corresponding": False,
            }
            for index in range(n_authors)
        ],
        "text": [],
        "section": [],
        "url": [],
        "bib": [{"bib_id": index + 1} for index in range(n_refs)],
        "xref": [],
        "figure": [],
        "table": [],
        "eq": [],
        "extraction": {
            "producer": {"name": "bibr", "version": "0.0.0-test"},
            "completed_at": "2026-09-22T10:00:00Z",
        },
    }


def test_smoke_test_success_prints_title_authors_refs():
    from bibr.api import Result

    fake_data = _smoke_export(n_authors=2, n_refs=10)
    wizard = _recording_wizard()
    with (
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
        patch("bibr.api.chew", return_value=Result(fake_data)) as mock_chew,
    ):
        wizard._step_smoke_test()
    assert mock_chew.called
    text = wizard.console.export_text()
    assert "Rodential Efficiency" in text
    assert "Authors: 2" in text
    assert "References: 10" in text
    assert "succeeded" in text.lower()


def test_smoke_test_uses_quick_no_llm_pipeline_options():
    from bibr.api import Result

    fake_data = _smoke_export(n_authors=0, n_refs=0)
    wizard = _recording_wizard()
    with (
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
        patch("bibr.api.chew", return_value=Result(fake_data)) as mock_chew,
    ):
        wizard._step_smoke_test()

    assert mock_chew.call_count == 1
    _, kwargs = mock_chew.call_args
    assert kwargs == {"pages": "1", "no_llm": True, "refs": "off"}


def test_smoke_test_missing_extra_hint():
    """ImportError (missing extra) surfaces the exact `uv sync --extra` line."""
    wizard = _recording_wizard()
    err = ImportError(
        "Layout detection (LayoutDetector) requires the 'ml' extra: "
        "pip install 'bibr[ml]' (or uv sync --extra ml)"
    )
    with (
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
        patch("bibr.api.chew", side_effect=err),
    ):
        wizard._step_smoke_test()  # must not raise
    text = wizard.console.export_text()
    assert "uv sync --extra ml" in text
    assert "bibr doctor" in text


def test_smoke_test_ocr_unreachable_hint():
    """UpstreamServiceError('ocr', ...) surfaces the backend's launch command."""
    from bibr.exceptions import UpstreamServiceError

    wizard = _recording_wizard()
    wizard.env_vars = {"OCR_BACKEND": "glm-http"}
    err = UpstreamServiceError(
        "ocr",
        "GLM-OCR server at http://localhost:8080 is unreachable (ConnectError). "
        "Check the URL and that the server is running.",
    )
    with (
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
        patch("bibr.api.chew", side_effect=err),
    ):
        wizard._step_smoke_test()
    text = wizard.console.export_text()
    assert "sglang.launch_server" in text or "vllm serve" in text
    assert "OCR_BASE_URL" in text
    assert "bibr doctor" in text


def test_smoke_test_auth_error_hint_names_key_env():
    """UpstreamServiceError('LLM', ...) wrapping an auth failure names the env var."""
    from bibr.exceptions import UpstreamServiceError

    wizard = _recording_wizard()
    wizard.env_vars = {"LLM_PROVIDER": "google"}
    err = UpstreamServiceError(
        "LLM",
        "Failed to extract title/keywords",
        Exception("Error code: 401 - {'error': 'invalid API key'}"),
    )
    with (
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
        patch("bibr.api.chew", side_effect=err),
    ):
        wizard._step_smoke_test()
    text = wizard.console.export_text()
    assert "GOOGLE_API_KEY" in text
    assert "bibr doctor" in text


def test_smoke_test_generic_exception_does_not_crash_wizard():
    """An unmapped exception still exits the step cleanly (config already saved)."""
    wizard = _recording_wizard()
    with (
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
        patch("bibr.api.chew", side_effect=RuntimeError("boom")),
    ):
        wizard._step_smoke_test()  # must not raise
    text = wizard.console.export_text()
    assert "already saved" in text.lower()
    assert "bibr doctor" in text


def test_smoke_test_generic_exception_redacts_api_key():
    """A raw exception message that echoes the configured API key must never be
    printed verbatim — SDK errors sometimes echo the request URL/auth header
    back (e.g. Google's ``?key=AIza...``)."""
    wizard = _recording_wizard()
    fake_key = "AIzaFAKEFAKEFAKEFAKEFAKE1234"
    wizard.env_vars = {"LLM_PROVIDER": "google", "GOOGLE_API_KEY": fake_key}
    err = RuntimeError(
        f"POST https://generativelanguage.googleapis.com/v1/models?key={fake_key} failed"
    )
    with (
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
        patch("bibr.api.chew", side_effect=err),
    ):
        wizard._step_smoke_test()  # must not raise
    text = wizard.console.export_text()
    assert fake_key not in text
    assert "***" in text


def test_reload_settings_in_place_updates_future_runtime_snapshots(monkeypatch):
    """Reloading process settings affects new runtimes without installing a
    mutable singleton reference inside runtime client modules."""
    import bibr.clients.llm as llm_mod
    import bibr.config as config_mod
    from bibr.setup_wizard import _reload_settings_in_place

    assert not hasattr(llm_mod, "Settings")

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    try:
        _reload_settings_in_place()

        assert config_mod.Settings.llm.provider == "anthropic"
        client = llm_mod.LLMClient()
        assert client._settings.llm.provider == "anthropic"
        assert client._settings is not config_mod.Settings
    finally:
        # Undo the env var now (inside the test, not at fixture teardown) and
        # reload again so the process-global singleton doesn't leak "anthropic"
        # into later tests in this session.
        monkeypatch.undo()
        _reload_settings_in_place()


def test_main_advanced_runs_advanced_wizard(monkeypatch):
    calls = []
    monkeypatch.setattr("bibr.setup_wizard.sys.argv", ["bibr", "--advanced"])
    monkeypatch.setattr(
        "bibr.setup_wizard.SetupWizard.run_advanced",
        lambda self: calls.append("advanced"),
        raising=False,
    )
    monkeypatch.setattr("bibr.setup_wizard.SetupWizard.run", lambda self: calls.append("easy"))

    from bibr.setup_wizard import main

    main()

    assert calls == ["advanced"]


@pytest.mark.slow
def test_smoke_test_real_extraction_no_llm():
    """Real end-to-end chew() over the packaged sample PDF — no API keys needed.

    Uses ``ocr="glm-llama"`` (llama.cpp backend) and ``no_llm=True`` so the run
    needs no cloud credentials; the llama.cpp GGUF weights are the smallest
    GLM-OCR variant bibr ships a backend for.
    """
    import importlib.resources as resources

    from bibr.api import chew

    resource = resources.files("bibr.data").joinpath("sample_paper.pdf")
    with resources.as_file(resource) as sample_path:
        result = chew(sample_path, ocr="glm-llama", no_llm=True)

    assert result.ok
    assert isinstance(result.data, dict)
    assert result.data.get("metadata") is not None


@pytest.mark.parametrize(
    ("hardware", "ram", "expected"),
    [
        (("cuda", 6.0), 32.0, "aggressive"),  # 6 GB A1000
        (("cuda", 24.0), 64.0, "balanced"),  # 3090/4090
        (("mlx", 16.0), 16.0, "balanced"),  # 16 GB Apple Silicon (OCR/LLM don't coexist)
        (("mlx", 64.0), 64.0, "balanced"),  # Mac Studio
        ((None, None), 4.0, "aggressive"),  # no accelerator, low RAM
    ],
)
def test_step_memory_mode_persists_resolved_mode(monkeypatch, hardware, ram, expected):
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: hardware)
    monkeypatch.setattr("bibr.local.pipeline._get_system_memory_gb", lambda: ram)

    wizard = SetupWizard()
    wizard._step_memory_mode()

    assert wizard.env_vars["PIPELINE_MEMORY_MODE"] == expected


# ---------------------------------------------------------------------------
# New simple-flow behaviour: weak-hardware fork, plan preview, ordering
# ---------------------------------------------------------------------------


def test_weak_hardware_private_server_path_writes_urls(tmp_path, monkeypatch):
    """Weak machine + 'I have a private server' → URL prompts; env gets OCR_BASE_URL."""
    wizard = _recording_wizard()
    wizard.env_path = tmp_path / ".env"
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: (None, None))
    monkeypatch.setattr("bibr.local.pipeline._get_system_memory_gb", lambda: 8.0)
    monkeypatch.setattr("bibr.setup_wizard.PresetManager", MagicMock())

    prompts = iter(["https://ocr.box:8080", "https://llm.box/v1", "local", "nuextract"])

    def confirm(prompt, **k):
        # Accept only "do you have a private server?"; decline install + smoke.
        return "private OCR/LLM server" in prompt

    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", confirm)

    wizard.run()

    content = wizard.env_path.read_text()
    assert "OCR_BACKEND=glm-http" in content
    assert "OCR_BASE_URL=https://ocr.box:8080" in content
    assert "LLM_BASE_URL=https://llm.box/v1" in content
    assert "LLM_PROVIDER=openai" in content
    # Cloud never entered the picture.
    assert "GOOGLE_API_KEY" not in content


def test_weak_hardware_cloud_consent_writes_gemini(tmp_path, monkeypatch):
    """Weak machine + no private server + cloud consent → cloud_fallback env."""
    wizard = _recording_wizard()
    wizard.env_path = tmp_path / ".env"
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: (None, None))
    monkeypatch.setattr("bibr.local.pipeline._get_system_memory_gb", lambda: 8.0)
    monkeypatch.setattr("bibr.setup_wizard.PresetManager", MagicMock())

    def confirm(prompt, **k):
        # Consent only to the cloud (Google Gemini) prompt; decline the private
        # server, install, connection test, and smoke test.
        return "Google Gemini" in prompt

    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", confirm)
    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *a, **k: "test-google-key")

    wizard.run()

    content = wizard.env_path.read_text()
    assert "OCR_BACKEND=gemini" in content
    assert "LLM_PROVIDER=google" in content
    assert "GOOGLE_API_KEY=test-google-key" in content


def test_cloud_consent_declined_offers_advanced_handoff(monkeypatch):
    """Declining cloud consent must offer --advanced, not SystemExit; yes → run_advanced."""
    wizard = _recording_wizard()
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: (None, None))
    monkeypatch.setattr("bibr.local.pipeline._get_system_memory_gb", lambda: 8.0)
    called = []
    monkeypatch.setattr(wizard, "run_advanced", lambda: called.append(True))

    def confirm(prompt, **k):
        # Decline the private-server and cloud prompts; accept only the
        # --advanced handoff so run_advanced() is invoked.
        return "advanced wizard" in prompt

    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", confirm)

    wizard.run()  # must not raise SystemExit

    assert called == [True]
    assert wizard.env_vars == {}


def test_install_failure_after_env_written_keeps_config(tmp_path, monkeypatch):
    """A failed extras install after .env is written must say config is already saved."""
    wizard = _recording_wizard()
    wizard.env_path = tmp_path / ".env"
    setup = RecommendedSetup(
        tier="fully_local",
        env={
            "OCR_BACKEND": "glm-llama",
            "LLM_BACKEND": "vllm",
            "LLM_LOCAL_MODEL": "numind/NuExtract3",
            "PIPELINE_MEMORY_MODE": "balanced",
        },
        extras={"ml"},
        preset_name="recommended-local",
        privacy_summary="stays local",
        download_summary="downloads GB",
        runtime_summary="slow",
        required_prompts=("install_extras", "smoke_test"),
    )
    monkeypatch.setattr("bibr.setup_wizard.detect_hardware", lambda: ("cuda", 24.0))
    monkeypatch.setattr("bibr.local.pipeline._get_system_memory_gb", lambda: 64.0)
    monkeypatch.setattr("bibr.setup_wizard._build_recommended_setup", lambda **k: setup)
    monkeypatch.setattr("bibr.setup_wizard.PresetManager", MagicMock())
    monkeypatch.setattr("bibr.setup_wizard.shutil.which", lambda name: "/bin/uv")
    monkeypatch.setattr(
        "bibr.setup_wizard.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=1, stderr="No solution found", stdout="")),
    )
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: True)

    with pytest.raises(SystemExit):
        wizard.run()

    assert wizard.env_path.exists()  # .env survives the failed install
    text = wizard.console.export_text().lower()
    assert "already saved" in text


def test_smoke_test_confirm_default_simple_vs_advanced(monkeypatch):
    """Simple flow defaults the smoke-test prompt to yes; advanced keeps no."""
    wizard = _quiet_wizard()
    defaults = []

    def rec_confirm(prompt, **k):
        defaults.append(k.get("default"))
        return False  # decline so chew() never runs

    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", rec_confirm)

    wizard._step_smoke_test(header="Test extraction", confirm_default=True)
    assert defaults[-1] is True

    wizard._step_smoke_test()  # advanced defaults
    assert defaults[-1] is False


def test_write_env_fresh_section_placement(tmp_path):
    """REF_SEG_STRATEGY, PIPELINE_MEMORY_MODE, OCR_BASE_URL live under Models."""
    env_path = tmp_path / ".env"
    _write_env_fresh(
        env_path,
        {
            "REF_SEG_STRATEGY": "geom",
            "PIPELINE_MEMORY_MODE": "aggressive",
            "OCR_BASE_URL": "http://localhost:8080",
            "OCR_BACKEND": "glm-http",
        },
    )
    content = env_path.read_text()

    def section_of(key):
        section = None
        for line in content.splitlines():
            if line.startswith("# --- ") and line.endswith(" ---"):
                section = line[len("# --- ") : -len(" ---")]
            elif line.startswith(f"{key}="):
                return section
        return None

    assert section_of("REF_SEG_STRATEGY") == "Models"
    assert section_of("PIPELINE_MEMORY_MODE") == "Models"
    assert section_of("OCR_BASE_URL") == "Models"


def test_help_text_describes_new_flow():
    """Help text must drop the false 'reads existing .env' claim and describe the flow."""
    from bibr.setup_wizard import _HELP_TEXT

    assert "offers them as defaults" not in _HELP_TEXT
    assert "plan preview" in _HELP_TEXT.lower()
    assert "overwrite, merge, or skip" in _HELP_TEXT
    assert "does not take its answers from your existing\n.env" in _HELP_TEXT


def test_plan_preview_shows_paddle_default_and_model(monkeypatch):
    """The plan preview describes Paddle-first OCR and the registry LLM model."""
    wizard = _recording_wizard()
    # Pin the backend: without this the resolved backend depends on whether the
    # host running the tests has rapid-mlx installed.
    with patch(
        "bibr.local.rapid_mlx.rapid_mlx_unavailable_reason",
        return_value="'rapid-mlx' was not found or is not executable",
    ):
        setup = _build_recommended_setup(
            platform_key="mlx",
            accelerator_memory_gb=32.0,
            system_memory_gb=32.0,
            sys_platform="darwin",
            machine="arm64",
        )
    wizard._print_recommendation(setup)
    text = wizard.console.export_text()

    assert "fully local" in text  # lowercase tier name, not "Fully Local"
    assert "paddle" in text
    assert "GLM fallback" in text
    assert "NuExtract" in text
    assert "vllm-mlx" in text


@pytest.mark.parametrize("plat", ["linux", "win32", "darwin"])
def test_available_extras_all_exist_in_pyproject(monkeypatch, plat):
    """Offering an extra pyproject does not define makes ``uv sync`` fail and the
    advanced wizard exit before ``.env`` is written (the ``local-cuda`` regression)."""
    import tomllib
    from pathlib import Path

    monkeypatch.setattr("bibr.setup_wizard.sys.platform", plat)
    monkeypatch.setattr("platform.machine", lambda: "arm64" if plat == "darwin" else "x86_64")
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    defined = set(pyproject["project"]["optional-dependencies"])

    offered = set(_available_extras())

    assert offered <= defined, offered - defined
    assert "local-cuda" not in offered


def test_linux_local_plan_selects_the_vllm_extra():
    from bibr.setup_wizard import _platform_local_extra

    assert _platform_local_extra("linux", "x86_64", ocr_backend="paddle") == {"vllm"}
    assert _platform_local_extra("linux", "x86_64", ocr_backend="glm-llama") == set()
    assert _platform_local_extra("darwin", "arm64", ocr_backend="paddle") == {"local"}
    assert _platform_local_extra("win32", "AMD64", ocr_backend="paddle") == set()
