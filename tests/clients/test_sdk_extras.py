"""The Google and Anthropic SDKs are extras: a run that needs a missing one stops early.

Each check names the setting, the extra and the install command, and never says
"key", so ``bibr doctor`` can tell it apart from a missing credential.
"""

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from bibr.clients import sdk_extras
from bibr.config import GlobalSettings

pytestmark = pytest.mark.real_cloud_sdks

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def no_cloud_sdks(monkeypatch):
    """Every cloud SDK module reads as not installed."""
    real = sdk_extras.module_installed
    cloud = {"google.genai", "jsonref", "anthropic"}
    monkeypatch.setattr(
        sdk_extras, "module_installed", lambda name: False if name in cloud else real(name)
    )


def _never_built(*_args, **_kwargs):
    pytest.fail("the SDK client was built")


def test_module_installed_follows_sys_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "bibr_test_fake_sdk", SimpleNamespace())
    assert sdk_extras.module_installed("bibr_test_fake_sdk")
    monkeypatch.setitem(sys.modules, "bibr_test_fake_sdk", None)
    assert not sdk_extras.module_installed("bibr_test_fake_sdk")
    assert not sdk_extras.module_installed("bibr_no_such_namespace.genai")


@pytest.mark.parametrize(
    ("provider", "extra", "package"),
    [("google", "google", "google-genai"), ("anthropic", "anthropic", "anthropic")],
)
def test_missing_llm_sdk_names_the_setting_extra_and_command(
    no_cloud_sdks, provider, extra, package
):
    reason = sdk_extras.missing_llm_sdk(provider)

    assert reason is not None
    assert f"LLM_PROVIDER={provider}" in reason
    assert package in reason
    assert f'pip install "bibr[{extra}]"' in reason
    assert f"uv sync --extra {extra}" in reason
    assert "key" not in reason.lower()


def test_google_reports_both_of_its_packages(no_cloud_sdks):
    assert sdk_extras.missing_packages("google") == ["google-genai", "jsonref"]
    assert "google-genai and jsonref packages" in sdk_extras.missing_llm_sdk("google")


@pytest.mark.parametrize("provider", ["openai", "groq", "ollama", "", None])
def test_core_providers_need_no_extra(no_cloud_sdks, provider):
    assert sdk_extras.missing_llm_sdk(provider) is None


def test_installed_sdks_are_not_reported(monkeypatch):
    monkeypatch.setattr(sdk_extras, "module_installed", lambda name: True)
    assert sdk_extras.missing_llm_sdk("google") is None
    assert sdk_extras.missing_ocr_sdk("anthropic") is None


@pytest.mark.parametrize(
    ("provider", "env_key"), [("google", "GOOGLE_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")]
)
def test_provider_refuses_before_instructor_builds_a_client(
    no_cloud_sdks, monkeypatch, provider, env_key
):
    from bibr.clients import providers

    monkeypatch.setattr("instructor.from_provider", _never_built)
    settings = GlobalSettings(llm={"provider": provider, "model": "m"}, **{env_key: "k-123"})

    with pytest.raises(ValueError, match=rf"LLM_PROVIDER={provider} needs"):
        providers.get(provider, settings=settings).build_client()


def test_a_missing_key_is_still_reported_first(no_cloud_sdks, monkeypatch):
    """Key before SDK, as before: the tests and users that set no key see the same message."""
    from bibr.clients import providers

    monkeypatch.setattr("instructor.from_provider", _never_built)
    settings = GlobalSettings(llm={"provider": "google", "api_key": None}, GOOGLE_API_KEY=None)

    with pytest.raises(ValueError, match="Google API key required"):
        providers.get("google", settings=settings).build_client()


def test_preflight_credentials_raises_the_hint(no_cloud_sdks, monkeypatch):
    """What ``bibr chew`` and ``bibr.chew()`` call before any OCR work."""
    from bibr.clients.llm import preflight_credentials

    monkeypatch.setattr("instructor.from_provider", _never_built)
    settings = GlobalSettings(llm={"provider": "google"}, GOOGLE_API_KEY="k-123")

    with pytest.raises(ValueError, match=r'pip install "bibr\[google\]"'):
        preflight_credentials(settings)


def test_dry_run_reports_the_missing_extra(no_cloud_sdks, monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli.dry_run import _dry_run_cloud_credential_blocker

    monkeypatch.setattr(Settings.llm, "provider", "anthropic")
    monkeypatch.setattr(Settings, "ANTHROPIC_API_KEY", "k-123")

    blocker = _dry_run_cloud_credential_blocker()

    assert blocker is not None
    assert "LLM_PROVIDER=anthropic needs the anthropic package" in blocker


@pytest.mark.parametrize(("backend", "extra"), [("gemini", "google"), ("anthropic", "anthropic")])
def test_cloud_ocr_backends_report_the_missing_extra(no_cloud_sdks, backend, extra):
    from bibr.ocr.registry import local_runtime_unavailable_reason

    reason = local_runtime_unavailable_reason(backend)

    assert reason is not None
    assert f"OCR_BACKEND={backend}" in reason
    assert f'"bibr[{extra}]"' in reason


def test_openai_vision_ocr_needs_no_extra(no_cloud_sdks):
    from bibr.ocr.registry import local_runtime_unavailable_reason

    assert local_runtime_unavailable_reason("openai") is None


def test_cloud_ocr_client_refuses_at_startup(no_cloud_sdks):
    from bibr.local.ocr_cloud import CloudOcrClient

    with pytest.raises(ValueError, match=r"OCR_BACKEND=gemini needs"):
        CloudOcrClient(provider="gemini", settings=GlobalSettings())


def test_setup_installs_the_extra_a_chosen_provider_needs(no_cloud_sdks, monkeypatch):
    from bibr.setup_wizard import SetupWizard

    wizard = SetupWizard()
    installed = []
    monkeypatch.setattr(
        wizard,
        "_install_selected_extras",
        lambda reason="", **_kw: installed.append((set(wizard.selected_extras), reason)),
    )

    wizard._ensure_sdk_extra(sdk_extras.llm_provider_extra("google"), "LLM_PROVIDER=google")
    wizard._ensure_sdk_extra(sdk_extras.llm_provider_extra("openai"), "LLM_PROVIDER=openai")

    assert installed == [({"google"}, "LLM_PROVIDER=google")]


def test_setup_installs_nothing_when_the_sdk_is_present(monkeypatch):
    from bibr.setup_wizard import SetupWizard

    monkeypatch.setattr(sdk_extras, "module_installed", lambda name: True)
    wizard = SetupWizard()
    monkeypatch.setattr(wizard, "_install_selected_extras", _never_built)

    wizard._ensure_sdk_extra("anthropic", "LLM_PROVIDER=anthropic")

    assert wizard.selected_extras == set()


def test_recommended_cloud_plan_installs_the_anthropic_extra():
    from bibr.setup_wizard import _build_recommended_setup

    setup = _build_recommended_setup(
        platform_key=None,
        accelerator_memory_gb=None,
        system_memory_gb=8.0,
        sys_platform="linux",
        machine="x86_64",
        allow_cloud=True,
    )

    assert setup.tier == "cloud_fallback"
    assert setup.env["LLM_PROVIDER"] == "anthropic"
    assert "anthropic" in setup.extras


def test_pyproject_keeps_the_sdks_out_of_core():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    core = " ".join(project["dependencies"])
    extras = project["optional-dependencies"]

    for package in ("google-genai", "anthropic", "jsonref"):
        assert package not in core
    assert "instructor" in core  # every provider's structured layer, local servers included
    assert any(r.startswith("google-genai") for r in extras["google"])
    assert any(r.startswith("jsonref") for r in extras["google"])
    assert extras["anthropic"] == ["anthropic>=0.40.0,<1"]
    assert extras["batch"] == ["bibr[anthropic]"]
    assert "google" in extras["all"][0] and "anthropic" in extras["all"][0]
