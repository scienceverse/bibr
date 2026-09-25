"""LLM API keys are not sent over plain HTTP to public hosts (x-security-9)."""

from __future__ import annotations

import asyncio

import pytest

from bibr.config import snapshot_settings

_REFUSAL = "Refusing to send the LLM API key over plain HTTP to the public host 'llm.example.org'"


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


def test_vision_ocr_refuses_a_public_http_base_url(monkeypatch):
    pytest.importorskip("instructor")
    from bibr.local.ocr_cloud import CloudOcrClient

    built = []
    monkeypatch.setattr(
        "instructor.from_provider", lambda *a, **kw: built.append(kw) or object(), raising=False
    )
    settings = _settings("unused")
    settings.ocr_vision.base_url = "http://vision.example.org/v1"

    def client():
        c = CloudOcrClient.__new__(CloudOcrClient)
        c._client = None
        c._provider = "openai"
        c._loaded = True
        c._init_lock = asyncio.Lock()
        c._settings = settings
        return c

    with pytest.raises(ValueError, match="vision OCR API key over plain HTTP"):
        asyncio.run(client()._aget_client())
    assert built == []

    settings.ocr.allow_insecure_http = True
    asyncio.run(client()._aget_client())
    assert built[0]["base_url"] == "http://vision.example.org/v1"
