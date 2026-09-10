"""Tests for the CLI credential preflight (fail fast before OCR runs)."""

import pytest

from bibr.clients.llm import preflight_credentials
from bibr.config import Settings


def test_preflight_missing_google_key_raises(monkeypatch):
    monkeypatch.setattr(Settings.llm, "provider", "google")
    monkeypatch.setattr(Settings.llm, "api_key", None)
    monkeypatch.setattr(Settings, "GOOGLE_API_KEY", None)
    with pytest.raises(ValueError, match="API key"):
        preflight_credentials()


def test_preflight_google_key_present(monkeypatch):
    monkeypatch.setattr(Settings.llm, "provider", "google")
    monkeypatch.setattr(Settings.llm, "api_key", "test-key")
    preflight_credentials()


def test_preflight_ollama_needs_no_key(monkeypatch):
    monkeypatch.setattr(Settings.llm, "provider", "ollama")
    monkeypatch.setattr(Settings.llm, "api_key", None)
    preflight_credentials()


def test_preflight_openai_missing_key_raises(monkeypatch):
    monkeypatch.setattr(Settings.llm, "provider", "openai")
    monkeypatch.setattr(Settings.llm, "api_key", None)
    monkeypatch.setattr(Settings.llm, "base_url", None)
    with pytest.raises(ValueError, match="API key"):
        preflight_credentials()


def test_preflight_openai_self_hosted_base_url_needs_no_key(monkeypatch):
    """A custom base_url means a self-hosted OpenAI-compatible server (vLLM,
    LM Studio, ...) which typically doesn't enforce real API-key auth."""
    monkeypatch.setattr(Settings.llm, "provider", "openai")
    monkeypatch.setattr(Settings.llm, "api_key", None)
    monkeypatch.setattr(Settings.llm, "base_url", "http://gpu-box:8001/v1")
    monkeypatch.setattr(Settings.llm, "model", "numind/NuExtract3")
    preflight_credentials()
