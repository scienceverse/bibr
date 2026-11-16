"""The vision model a cloud OCR backend calls when ``OCR_VISION_MODEL`` is not set.

The setting's default is a Gemini model. ``--ocr anthropic`` on its own is what
the scanned-PDF message and ``bibr doctor`` suggest, and it used to send that
Gemini model name to Anthropic.
"""

from __future__ import annotations

from types import SimpleNamespace

from bibr.config import DEFAULT_LLM_MODELS, GlobalSettings
from bibr.ocr.profiles import cloud_vision_model, resolve_served_model


def _settings(monkeypatch, **env: str) -> GlobalSettings:
    monkeypatch.delenv("OCR_VISION_MODEL", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return GlobalSettings(_env_file=None)


def test_an_unset_model_follows_the_anthropic_backend(monkeypatch):
    settings = _settings(monkeypatch)

    assert cloud_vision_model("anthropic", settings) == DEFAULT_LLM_MODELS["anthropic"]


def test_an_unset_model_keeps_the_gemini_default_for_the_other_backends(monkeypatch):
    settings = _settings(monkeypatch)

    assert cloud_vision_model("gemini", settings) == "gemini-3-flash-preview"
    assert cloud_vision_model("openai", settings) == "gemini-3-flash-preview"


def test_a_set_model_wins_for_every_backend(monkeypatch):
    settings = _settings(monkeypatch, OCR_VISION_MODEL="my-vision-model")

    for backend in ("anthropic", "gemini", "openai"):
        assert cloud_vision_model(backend, settings) == "my-vision-model"


def test_a_model_assigned_in_code_wins(monkeypatch):
    settings = _settings(monkeypatch)
    settings.ocr_vision.model = "assigned-model"

    assert cloud_vision_model("anthropic", settings) == "assigned-model"


def test_settings_that_do_not_say_what_was_set_are_taken_as_they_are():
    # Test doubles and hand-built settings objects carry no model_fields_set.
    settings = SimpleNamespace(ocr_vision=SimpleNamespace(model="stub-model"))

    assert cloud_vision_model("anthropic", settings) == "stub-model"


def test_the_served_model_a_dry_run_names_follows_the_backend(monkeypatch):
    settings = _settings(monkeypatch)

    served = resolve_served_model(
        requested_backend="anthropic",
        concrete_backend="anthropic",
        settings=settings,
        explicit_profile=None,
    )

    assert served == DEFAULT_LLM_MODELS["anthropic"]
