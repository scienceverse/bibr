"""LLM API keys are not sent over plain HTTP to public hosts (x-security-9)."""

from __future__ import annotations

import asyncio

import pytest

from bibr.config import snapshot_settings

_REFUSAL = "Refusing to send the LLM API key over plain HTTP to the public host 'llm.example.org'"
_OPT_OUT = (
    "Use an https:// URL, or set LLM_ALLOW_INSECURE_HTTP=true if the network path is trusted."
)


def _settings(base_url: str, *, api_key: str | None = "llm-key-placeholder", allow=False):
    settings = snapshot_settings()
    settings.llm.provider = "openai"
    settings.llm.model = "some-model"
    settings.llm.base_url = base_url
    settings.llm.api_key = api_key
    settings.llm.allow_insecure_http = allow
    return settings


def test_openai_provider_refuses_a_public_http_base_url():
    from bibr.clients.providers.openai import OpenAIProvider

    with pytest.raises(ValueError) as exc:
        OpenAIProvider(_settings("http://llm.example.org/v1")).build_client()
    assert str(exc.value).startswith(_REFUSAL)


@pytest.mark.parametrize(
    ("base_url", "api_key", "allow"),
    [
        ("http://127.0.0.1:8001/v1", "llm-key-placeholder", False),  # local vLLM
        ("http://100.113.200.117:30000/v1", "llm-key-placeholder", False),  # tailnet fleet
        ("http://gpu.tail1234.ts.net:30000/v1", "llm-key-placeholder", False),  # MagicDNS
        ("https://llm.example.org/v1", "llm-key-placeholder", False),
        ("http://llm.example.org/v1", None, False),  # no key, nothing to leak
        ("http://llm.example.org/v1", "llm-key-placeholder", True),  # explicit opt-out
    ],
)
def test_openai_provider_keeps_working_where_no_key_crosses_the_internet(base_url, api_key, allow):
    from bibr.clients.providers.openai import OpenAIProvider

    assert OpenAIProvider(_settings(base_url, api_key=api_key, allow=allow)).build_client()


def test_nuextract_backend_refuses_a_public_http_base_url():
    from bibr.clients.nuextract import NuExtractNativeBackend

    backend = NuExtractNativeBackend(_settings("http://llm.example.org/v1"))
    with pytest.raises(ValueError) as exc:
        backend._get_client()
    assert str(exc.value).startswith(_REFUSAL)
    assert NuExtractNativeBackend(_settings("http://10.0.0.7:8000/v1"))._get_client()


def _vision_client(settings):
    from bibr.local.ocr_cloud import CloudOcrClient

    c = CloudOcrClient.__new__(CloudOcrClient)
    c._client = None
    c._provider = "openai"
    c._loaded = True
    c._init_lock = asyncio.Lock()
    c._settings = settings
    return c


def test_vision_ocr_refuses_a_public_http_base_url(monkeypatch):
    pytest.importorskip("instructor")

    built = []
    monkeypatch.setattr(
        "instructor.from_provider", lambda *a, **kw: built.append(kw) or object(), raising=False
    )
    # The LLM and vision calls share one OpenAI-compatible endpoint, so the
    # vision client forwards the LLM key to it.
    settings = _settings("http://vision.example.org/v1")
    settings.ocr_vision.base_url = "http://vision.example.org/v1"

    # OCR_ALLOW_INSECURE_HTTP (on by default in docker-compose) is the OCR
    # server's opt-out: the vision endpoint gets the LLM key, so it stays refused.
    settings.ocr.allow_insecure_http = True
    with pytest.raises(ValueError, match="LLM API key over plain HTTP"):
        asyncio.run(_vision_client(settings)._aget_client())
    assert built == []

    settings.llm.allow_insecure_http = True
    asyncio.run(_vision_client(settings)._aget_client())
    assert built[0]["base_url"] == "http://vision.example.org/v1"
    assert built[0]["api_key"] == "llm-key-placeholder"


def test_vision_ocr_refuses_the_sdk_environment_key_over_plain_http(monkeypatch):
    pytest.importorskip("instructor")

    built = []
    monkeypatch.setattr(
        "instructor.from_provider", lambda *a, **kw: built.append(kw) or object(), raising=False
    )
    # The LLM key belongs to another endpoint, so the vision client passes no
    # key and the OpenAI SDK would send OPENAI_API_KEY from the environment.
    settings = _settings("https://llm.example.org/v1")
    settings.ocr_vision.base_url = "http://vision.example.org/v1"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-placeholder")
    with pytest.raises(ValueError, match="LLM API key over plain HTTP"):
        asyncio.run(_vision_client(settings)._aget_client())
    assert built == []

    # No key anywhere: nothing crosses the network, so the client is built.
    monkeypatch.delenv("OPENAI_API_KEY")
    asyncio.run(_vision_client(settings)._aget_client())
    assert "api_key" not in built[0]


# --- bibr setup and bibr doctor send the key too --------------------------------


def test_setup_wizard_asks_again_for_a_public_http_base_url(monkeypatch):
    from bibr.setup_wizard import SetupWizard

    monkeypatch.delenv("LLM_ALLOW_INSECURE_HTTP", raising=False)
    answers = iter(["http://llm.example.org/v1", "https://llm.example.org/v1"])
    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *_a, **_k: next(answers))
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *_a, **_k: False)
    wizard = SetupWizard()

    assert wizard._ask_llm_base_url("llm-key-placeholder") == "https://llm.example.org/v1"
    assert "LLM_ALLOW_INSECURE_HTTP" not in wizard.env_vars


def test_setup_wizard_records_an_explicit_plain_http_opt_in(monkeypatch):
    from bibr.setup_wizard import SetupWizard

    monkeypatch.delenv("LLM_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.setattr("bibr.setup_wizard.Prompt.ask", lambda *_a, **_k: "http://llm.example.org")
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *_a, **_k: True)
    wizard = SetupWizard()

    assert wizard._ask_llm_base_url("llm-key-placeholder") == "http://llm.example.org"
    assert wizard.env_vars["LLM_ALLOW_INSECURE_HTTP"] == "true"
    assert wizard._allows_insecure_llm_http()


def test_setup_wizard_model_listing_never_sends_the_key_to_a_public_http_host(monkeypatch):
    from bibr import setup_wizard

    sent = []
    monkeypatch.setattr(
        setup_wizard, "_fetch_openai_compat_models", lambda *a, **kw: sent.append(kw) or ["m"]
    )
    url = "http://llm.example.org/v1"
    assert setup_wizard._fetch_models("openai", "llm-key-placeholder", url) == []
    assert sent == []
    assert setup_wizard._fetch_models(
        "openai", "llm-key-placeholder", url, allow_insecure_http=True
    ) == ["m"]


def test_setup_wizard_connection_test_refuses_a_public_http_base_url(monkeypatch):
    pytest.importorskip("instructor")
    from bibr.clients.llm import ping_llm
    from bibr.setup_wizard import _connection_test_settings

    built = []
    monkeypatch.setattr("instructor.from_provider", lambda *a, **kw: built.append(kw))
    answers = {
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "m",
        "LLM_API_KEY": "llm-key-placeholder",
        "LLM_BASE_URL": "http://llm.example.org/v1",
    }
    with pytest.raises(ValueError, match=_REFUSAL):
        ping_llm(_connection_test_settings(answers))
    assert built == []
    # The opt-in the wizard saves reaches the connection test too.
    opted_in = _connection_test_settings({**answers, "LLM_ALLOW_INSECURE_HTTP": "true"})
    assert opted_in.llm.allow_insecure_http is True


def test_doctor_fails_the_connection_check_without_sending_the_key(monkeypatch):
    pytest.importorskip("instructor")
    import io

    from rich.console import Console

    from bibr.config import snapshot_settings
    from bibr.local.cli.doctor import _check_llm_connection

    built = []
    monkeypatch.setattr("instructor.from_provider", lambda *a, **kw: built.append(kw))
    settings = snapshot_settings()
    settings.llm.provider = "openai"
    settings.llm.model = "m"
    settings.llm.api_key = "llm-key-placeholder"
    settings.llm.base_url = "http://llm.example.org/v1"
    settings.llm.allow_insecure_http = False
    failures = []
    _check_llm_connection(
        settings,
        Console(file=io.StringIO()),
        lambda *_a: None,
        lambda msg, **_k: failures.append(msg),
    )
    assert built == []
    assert failures == [f"LLM connection failed: {_REFUSAL}. " + _OPT_OUT]
