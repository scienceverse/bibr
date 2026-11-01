"""Run mode: which mode a run takes, and what the fast-mode overlay switches off."""

import logging

import pytest
from pydantic import ValidationError

from bibr.clients.llm import preflight_credentials
from bibr.config import GlobalSettings, compute_behavior_fingerprint
from bibr.exceptions import InputValidationError
from bibr.presets import InvalidPresetError, PresetManager
from bibr.run_mode import _log_local_strategy, fast_settings, llm_configured, resolve_mode
from bibr.structure.section_classifier import _min_confidence

_KEY_VARS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "LANGEXTRACT_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "GROQ_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "PIPELINE_MODE",
)


def _settings(
    monkeypatch,
    *,
    provider="google",
    llm_key=None,
    base_url=None,
    google=None,
    anthropic=None,
    groq=None,
    backend="cloud",
    mode="auto",
) -> GlobalSettings:
    """Settings with exactly the credentials named, whatever the test environment holds."""
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    s = GlobalSettings()
    s.llm.provider = provider
    s.llm.api_key = llm_key
    s.llm.base_url = base_url
    s.llm.backend = backend
    s.GOOGLE_API_KEY = google
    s.ANTHROPIC_API_KEY = anthropic
    s.GROQ_API_KEY = groq
    s.pipeline.mode = mode
    return s


# provider, credentials, is an LLM configured
CONFIGURED = [
    ("google", {}, False),
    ("google", {"llm_key": "k"}, True),
    ("google", {"google": "k"}, True),
    ("google", {"anthropic": "k", "groq": "k"}, False),  # another provider's key
    ("openai", {}, False),
    ("openai", {"llm_key": "k"}, True),
    ("openai", {"base_url": "http://localhost:8000/v1"}, True),  # a keyless local server
    ("openai", {"google": "k"}, False),
    ("anthropic", {}, False),
    ("anthropic", {"anthropic": "k"}, True),
    ("anthropic", {"llm_key": "k"}, True),
    ("anthropic", {"google": "k"}, False),
    ("groq", {}, False),
    ("groq", {"groq": "k"}, True),
    ("groq", {"llm_key": "k"}, True),
    ("ollama", {}, True),  # no key to look for
    ("acme", {}, True),  # an adapter registered by another package
]


@pytest.mark.parametrize(("provider", "creds", "expected"), CONFIGURED)
def test_llm_configured_follows_what_the_provider_needs(monkeypatch, provider, creds, expected):
    s = _settings(monkeypatch, provider=provider, **creds)
    assert llm_configured(s) is expected


@pytest.mark.parametrize(
    "backend", ["local", "vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster"]
)
def test_a_managed_local_backend_is_an_llm_without_any_key(monkeypatch, backend):
    assert llm_configured(_settings(monkeypatch, backend=backend)) is True


def test_the_run_backend_overrides_the_setting(monkeypatch):
    s = _settings(monkeypatch, backend="vllm")
    assert llm_configured(s, backend="cloud") is False
    s = _settings(monkeypatch, backend="cloud")
    assert llm_configured(s, backend="local") is True


@pytest.mark.parametrize(("provider", "creds", "expected"), CONFIGURED)
def test_auto_is_fast_without_an_llm_and_full_with_one(monkeypatch, provider, creds, expected):
    s = _settings(monkeypatch, provider=provider, **creds)
    assert resolve_mode(s) == ("full" if expected else "fast")


def test_auto_with_a_local_backend_is_full(monkeypatch):
    assert resolve_mode(_settings(monkeypatch), backend="local") == "full"
    assert resolve_mode(_settings(monkeypatch, backend="vllm")) == "full"


@pytest.mark.parametrize("key", [None, "k"])
@pytest.mark.parametrize("setting", ["auto", "fast", "full"])
@pytest.mark.parametrize("explicit", [None, "auto", "fast", "full"])
def test_explicit_flag_then_setting_then_auto(monkeypatch, explicit, setting, key):
    s = _settings(monkeypatch, google=key, mode=setting)
    chosen = setting if explicit is None else explicit
    expected = ("full" if key else "fast") if chosen == "auto" else chosen
    assert resolve_mode(s, explicit) == expected


def test_explicit_full_without_an_llm_is_left_for_the_credential_check(monkeypatch):
    s = _settings(monkeypatch)
    assert resolve_mode(s, "full") == "full"
    with pytest.raises(ValueError, match="Google API key required"):
        preflight_credentials(s)


def test_an_unknown_mode_is_an_input_error(monkeypatch):
    s = _settings(monkeypatch)
    with pytest.raises(InputValidationError, match="Valid values: auto, fast, full"):
        resolve_mode(s, "turbo")
    s.pipeline.mode = "turbo"  # assigned past the validator
    with pytest.raises(InputValidationError, match="'turbo'"):
        resolve_mode(s)


PLAINTEXT = {"llm_key": "sk-test-key", "base_url": "http://llm.example.org/v1"}


def test_a_refused_plaintext_key_stays_a_full_run_that_the_preflight_stops(monkeypatch):
    """The refusal is a ValueError the credential check raises. Reading it as "no LLM" would
    turn a broken configuration into a fast run that quietly sends nothing."""
    s = _settings(monkeypatch, provider="openai", **PLAINTEXT)
    with pytest.raises(ValueError, match="LLM API key over plain HTTP"):
        preflight_credentials(s)
    assert llm_configured(s) is True
    assert resolve_mode(s) == "full"


def test_a_missing_key_is_the_only_credential_error_that_means_fast(monkeypatch):
    s = _settings(monkeypatch, provider="openai")
    with pytest.raises(ValueError, match="OpenAI API key required"):
        preflight_credentials(s)
    assert resolve_mode(s) == "fast"


@pytest.mark.parametrize(
    ("provider", "creds", "expected"), CONFIGURED + [("openai", PLAINTEXT, True)]
)
def test_resolving_a_mode_builds_no_client(monkeypatch, provider, creds, expected):
    """Nothing in the mode decision reaches a provider, so no credential error is read as fast."""

    def refuse(*args, **kwargs):
        raise AssertionError("the mode decision built an LLM client")

    monkeypatch.setattr("bibr.clients.llm._create_client", refuse)
    monkeypatch.setattr("bibr.clients.providers.get", refuse)
    s = _settings(monkeypatch, provider=provider, **creds)
    assert llm_configured(s) is expected
    assert resolve_mode(s) == ("full" if expected else "fast")


# ---------------------------------------------------------------------------
# The overlay
# ---------------------------------------------------------------------------


def _llm_on(monkeypatch) -> GlobalSettings:
    """Settings with every LLM path the overlay turns off set the other way."""
    s = _settings(monkeypatch, google="k")
    s.ml.front_tagger_mode = "primary"
    s.ml.section_classifier_llm_escalation = True
    s.ml.paper_classifier_llm_escalation = True
    s.IMPLICIT_SECTION_DETECTION = True
    s.REF_SEG_LLM_FALLBACK = True
    s.REF_NER_LLM_FALLBACK = True
    s.llm.merged_core_metadata = True
    return s


def test_the_overlay_turns_every_llm_path_off(monkeypatch):
    fast = fast_settings(_llm_on(monkeypatch))
    assert fast.ml.front_tagger_mode == "only"
    assert fast.ml.section_classifier_llm_escalation is False
    assert fast.ml.paper_classifier_llm_escalation is False
    assert fast.IMPLICIT_SECTION_DETECTION is False
    assert fast.REF_SEG_LLM_FALLBACK is False
    assert fast.REF_NER_LLM_FALLBACK is False
    assert fast.llm.merged_core_metadata is False


def test_the_overlay_is_a_snapshot_and_leaves_the_source_alone(monkeypatch):
    source = _llm_on(monkeypatch)
    before = source.model_dump(mode="json")
    fast = fast_settings(source)
    assert fast is not source
    assert fast.ml is not source.ml
    assert fast.llm is not source.llm
    assert source.model_dump(mode="json") == before
    assert source.ml.front_tagger_mode == "primary"
    assert source.IMPLICIT_SECTION_DETECTION is True
    # Credentials ride along: crossref, OCR and the cache still read their own settings.
    assert fast.GOOGLE_API_KEY == "k"


def test_the_overlay_is_idempotent(monkeypatch):
    once = fast_settings(_llm_on(monkeypatch))
    assert fast_settings(once).model_dump(mode="json") == once.model_dump(mode="json")


@pytest.mark.parametrize(
    ("seg", "parse", "expected_seg", "expected_parse"),
    [
        ("llm", "llm", "geom", "ner"),
        ("llm", "llm-chunked", "geom", "ner"),
        ("geom", "ref-llm", "geom", "ner"),
        ("crf", "ner", "crf", "ner"),
        ("region", "off", "region", "off"),
        (None, None, None, None),
    ],
)
def test_a_setting_that_names_the_llm_for_references_becomes_the_local_strategy(
    monkeypatch, seg, parse, expected_seg, expected_parse
):
    source = _llm_on(monkeypatch)
    source.REF_SEG_STRATEGY = seg
    source.REF_PARSE_STRATEGY = parse
    fast = fast_settings(source)
    assert (expected_seg, expected_parse) == (fast.REF_SEG_STRATEGY, fast.REF_PARSE_STRATEGY)
    assert (seg, parse) == (source.REF_SEG_STRATEGY, source.REF_PARSE_STRATEGY)


def test_the_strategy_swap_is_logged_once_per_value(monkeypatch, caplog):
    _log_local_strategy.cache_clear()
    source = _llm_on(monkeypatch)
    source.REF_SEG_STRATEGY = "llm"
    source.REF_PARSE_STRATEGY = "llm"
    with caplog.at_level(logging.INFO, logger="bibr.run_mode"):
        for _ in range(3):
            fast_settings(source)
    lines = [r.getMessage() for r in caplog.records if r.name == "bibr.run_mode"]
    assert lines == [
        "Fast mode makes no LLM call, so REF_SEG_STRATEGY=llm is run as geom",
        "Fast mode makes no LLM call, so REF_PARSE_STRATEGY=llm is run as ner",
    ]


def test_section_threshold_is_half_in_fast_and_an_explicit_one_still_wins(monkeypatch):
    source = _llm_on(monkeypatch)
    assert _min_confidence(source) == 0.9
    assert _min_confidence(fast_settings(source)) == 0.5

    source.ml.section_classifier_min_confidence = 0.7
    assert _min_confidence(fast_settings(source)) == 0.7
    source.ml.section_classifier_min_confidence = 0.0
    assert _min_confidence(fast_settings(source)) == 0.0


# ---------------------------------------------------------------------------
# The settings themselves
# ---------------------------------------------------------------------------


def test_mode_defaults_to_auto_and_is_case_insensitive(monkeypatch):
    monkeypatch.delenv("PIPELINE_MODE", raising=False)
    assert GlobalSettings().pipeline.mode == "auto"
    monkeypatch.setenv("PIPELINE_MODE", " FAST ")
    assert GlobalSettings().pipeline.mode == "fast"
    monkeypatch.setenv("PIPELINE_MODE", "")  # a blank means the default
    assert GlobalSettings().pipeline.mode == "auto"


def test_mode_rejects_other_values(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "quick")
    with pytest.raises(ValidationError):
        GlobalSettings()


def test_front_tagger_mode_takes_only(monkeypatch):
    monkeypatch.delenv("ML_FRONT_TAGGER_MODE", raising=False)
    assert GlobalSettings().ml.front_tagger_mode == "off"  # full mode downloads nothing
    for value in ("off", "primary", "only"):
        monkeypatch.setenv("ML_FRONT_TAGGER_MODE", value)
        assert GlobalSettings().ml.front_tagger_mode == value
    monkeypatch.setenv("ML_FRONT_TAGGER_MODE", "always")
    with pytest.raises(ValidationError):
        GlobalSettings()


def test_the_mode_is_part_of_the_cache_fingerprint(monkeypatch):
    """A served result differs between fast and full, so it must not share a namespace."""
    auto = _settings(monkeypatch)
    fast = _settings(monkeypatch, mode="fast")
    full = _settings(monkeypatch, mode="full")
    prints = {compute_behavior_fingerprint(s) for s in (auto, fast, full)}
    assert len(prints) == 3


def test_preset_round_trip(tmp_path, monkeypatch):
    monkeypatch.delenv("PIPELINE_MODE", raising=False)
    manager = PresetManager(presets_dir=tmp_path / "presets")
    env = tmp_path / ".env"
    env.write_text("PIPELINE_MODE=fast\nLLM_PROVIDER=google\n", encoding="utf-8")

    manager.save("quick", manager.snapshot_from_env(env))
    assert manager.load("quick")["PIPELINE_MODE"] == "fast"

    other = tmp_path / "other.env"
    other.write_text("PIPELINE_MODE=full\n", encoding="utf-8")
    manager.apply("quick", other)
    assert "PIPELINE_MODE=fast" in other.read_text()
    assert "PIPELINE_MODE=full" not in other.read_text()

    settings = _settings(monkeypatch, mode="full")
    assert manager.apply_to_settings("quick", settings) == []
    assert settings.pipeline.mode == "fast"


def test_a_preset_with_a_bad_mode_is_refused(tmp_path):
    manager = PresetManager(presets_dir=tmp_path / "presets")
    manager.save("bad", {"PIPELINE_MODE": "turbo"})
    with pytest.raises(InvalidPresetError, match="PIPELINE_MODE"):
        manager.apply("bad", tmp_path / ".env")
