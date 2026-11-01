"""Run mode: which mode a run takes, and what the fast-mode overlay switches off."""

import logging

import pytest
from pydantic import ValidationError

from bibr.clients.llm import preflight_credentials
from bibr.config import GlobalSettings, compute_behavior_fingerprint
from bibr.exceptions import InputValidationError
from bibr.presets import InvalidPresetError, PresetManager
from bibr.run_mode import (
    FAST_SKIPS,
    _log_local_strategy,
    fast_settings,
    llm_configured,
    resolve_mode,
    resolve_run_mode,
)
from bibr.structure.section_classifier import _min_confidence

_KEY_VARS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "LANGEXTRACT_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "PIPELINE_MODE",
)


def _settings(
    monkeypatch,
    *,
    provider=None,
    model=None,
    llm_key=None,
    base_url=None,
    google=None,
    anthropic=None,
    groq=None,
    backend="cloud",
    mode="auto",
) -> GlobalSettings:
    """Settings with exactly the credentials and choices named, whatever the test environment holds.

    ``provider``, ``model`` and ``base_url`` are assigned only when given, as a user's
    ``LLM_PROVIDER``, ``LLM_MODEL`` and ``LLM_BASE_URL`` would set them.
    """
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    s = GlobalSettings()
    for name, value in (("provider", provider), ("model", model), ("base_url", base_url)):
        if value is not None:
            setattr(s.llm, name, value)
    s.llm.api_key = llm_key
    s.llm.backend = backend
    s.GOOGLE_API_KEY = google
    s.ANTHROPIC_API_KEY = anthropic
    s.GROQ_API_KEY = groq
    s.pipeline.mode = mode
    return s


# what the settings name and hold, whether an LLM is configured
CONFIGURED = [
    ({}, False),  # a fresh install
    ({"llm_key": "k"}, True),
    ({"google": "k"}, True),
    ({"anthropic": "k", "groq": "k"}, False),  # another provider's key, and Google is the default
    # Naming a provider, a model or an endpoint is choosing an LLM, key or no key.
    ({"provider": "openai"}, True),
    ({"provider": "anthropic"}, True),
    ({"provider": "groq"}, True),
    ({"provider": "ollama"}, True),  # no key to look for
    ({"provider": "acme"}, True),  # an adapter registered by another package
    ({"provider": "openai", "base_url": "http://localhost:8000/v1"}, True),  # a keyless server
    ({"provider": "anthropic", "model": "claude-haiku-5-5"}, True),
    ({"model": "gemini-2.5-flash"}, True),
    ({"base_url": "http://localhost:8000/v1", "model": "m"}, True),
    ({"provider": "openai", "google": "k"}, True),  # a key for a provider that was not chosen
]


@pytest.mark.parametrize(("choices", "expected"), CONFIGURED)
def test_llm_configured_follows_what_the_settings_name_and_hold(monkeypatch, choices, expected):
    assert llm_configured(_settings(monkeypatch, **choices)) is expected


def test_the_default_provider_is_the_one_whose_key_a_fresh_install_lacks():
    assert GlobalSettings().llm.provider == "google"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"LLM_PROVIDER": "anthropic"}, True),
        ({"LLM_MODEL": "claude-haiku-5-5"}, True),
        ({"LLM_BASE_URL": "http://localhost:8000/v1"}, True),
        ({"LLM_PROVIDER": "groq", "LLM_MODEL": "m"}, True),
        # Nothing here names an LLM or holds the default provider's key.
        ({"OPENAI_API_KEY": "sk-x", "ANTHROPIC_API_KEY": "k"}, False),
    ],
)
def test_a_provider_model_or_endpoint_in_the_environment_is_a_configured_llm(
    monkeypatch, env, expected
):
    """The same through the environment, where a user sets them."""
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    s = GlobalSettings()
    assert llm_configured(s) is expected
    assert resolve_mode(s) == ("full" if expected else "fast")


def test_a_setting_filled_in_on_its_own_does_not_count_as_a_choice(monkeypatch):
    """``rate_limit_rpm`` and the like are set for the user; only the three choices count."""
    s = _settings(monkeypatch)
    s.llm.rate_limit_rpm = 10
    s.llm.reasoning_effort = "low"
    assert {"rate_limit_rpm", "reasoning_effort"} <= s.llm.model_fields_set
    assert llm_configured(s) is False


# The configurations that are a hard credential error on a full run. A user who set
# them up meant to call an LLM, so they must run full and reach that error.
BROKEN = [
    ({"provider": "anthropic", "model": "claude-haiku-5-5"}, "Anthropic API key required"),
    ({"provider": "groq"}, "Groq API key required"),
    ({"provider": "openai"}, "OpenAI API key required"),
    ({"base_url": "http://localhost:8000/v1", "model": "m"}, "Google API key required"),
]


@pytest.mark.parametrize(("choices", "message"), BROKEN)
def test_a_broken_llm_configuration_runs_full_and_reaches_the_credential_error(
    monkeypatch, choices, message
):
    s = _settings(monkeypatch, **choices)
    assert resolve_mode(s) == "full"
    with pytest.raises(ValueError, match=message):
        preflight_credentials(s)


def test_a_key_in_a_variable_the_adapter_does_not_read_is_still_a_broken_configuration(
    monkeypatch,
):
    """The openai adapter reads ``LLM_API_KEY`` only, so OPENAI_API_KEY alone leaves it keyless."""
    s = _settings(monkeypatch, provider="openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert resolve_mode(s) == "full"
    with pytest.raises(ValueError, match="OpenAI API key required"):
        preflight_credentials(s)


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


@pytest.mark.parametrize(("choices", "expected"), CONFIGURED)
def test_auto_is_fast_without_an_llm_and_full_with_one(monkeypatch, choices, expected):
    assert resolve_mode(_settings(monkeypatch, **choices)) == ("full" if expected else "fast")


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


PLAINTEXT = {
    "provider": "openai",
    "llm_key": "sk-test-key",
    "base_url": "http://llm.example.org/v1",
}


def test_a_refused_plaintext_key_stays_a_full_run_that_the_preflight_stops(monkeypatch):
    """The refusal is a ValueError the credential check raises. Reading it as "no LLM" would
    turn a broken configuration into a fast run that quietly sends nothing."""
    s = _settings(monkeypatch, **PLAINTEXT)
    with pytest.raises(ValueError, match="LLM API key over plain HTTP"):
        preflight_credentials(s)
    assert llm_configured(s) is True
    assert resolve_mode(s) == "full"


def test_only_a_missing_default_key_is_a_credential_error_that_means_fast(monkeypatch):
    """A fresh install has chosen nothing, so its missing Google key is the absence of an LLM."""
    s = _settings(monkeypatch)
    with pytest.raises(ValueError, match="Google API key required"):
        preflight_credentials(s)
    assert resolve_mode(s) == "fast"


@pytest.mark.parametrize(("choices", "expected"), [*CONFIGURED, (PLAINTEXT, True)])
def test_resolving_a_mode_builds_no_client(monkeypatch, choices, expected):
    """Nothing in the mode decision reaches a provider, so no credential error is read as fast."""

    def refuse(*args, **kwargs):
        raise AssertionError("the mode decision built an LLM client")

    monkeypatch.setattr("bibr.clients.llm._create_client", refuse)
    monkeypatch.setattr("bibr.clients.providers.get", refuse)
    s = _settings(monkeypatch, **choices)
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


# --- resolve_run_mode: what the entry points (bibr chew, bibr batch, bibr.chew) take ----------


@pytest.mark.parametrize("key", [None, "k"])
@pytest.mark.parametrize("explicit", [None, "auto", "fast", "full"])
def test_no_llm_is_structure_only_and_never_fast(monkeypatch, explicit, key):
    """--no-llm stays "full" so it never takes the fast overlay, whatever else is asked or set."""
    s = _settings(monkeypatch, google=key)
    assert resolve_run_mode(s, explicit=explicit, no_llm=True) == "full"


def test_no_llm_stays_full_even_when_the_setting_says_fast(monkeypatch):
    s = _settings(monkeypatch, mode="fast")
    assert resolve_run_mode(s, no_llm=True) == "full"


def test_a_run_with_no_llm_and_no_flag_is_fast(monkeypatch):
    assert resolve_run_mode(_settings(monkeypatch)) == "fast"


def test_a_run_with_a_key_and_no_flag_is_full(monkeypatch):
    assert resolve_run_mode(_settings(monkeypatch, google="k")) == "full"


def test_the_flag_beats_the_setting_in_both_directions(monkeypatch):
    assert (
        resolve_run_mode(_settings(monkeypatch, google="k", mode="full"), explicit="fast") == "fast"
    )
    assert resolve_run_mode(_settings(monkeypatch, mode="fast"), explicit="full") == "full"


def test_an_llm_only_option_under_auto_asks_for_an_llm_so_the_run_is_full(monkeypatch):
    """--refs llm with no key must reach the credential error, not quietly run fast."""
    s = _settings(monkeypatch)
    assert resolve_run_mode(s, llm_flags=["--refs llm"]) == "full"
    assert resolve_run_mode(s, explicit="auto", llm_flags=["--llm vllm"]) == "full"


def test_an_llm_only_option_with_an_explicit_fast_is_refused(monkeypatch):
    s = _settings(monkeypatch, google="k")
    with pytest.raises(ValueError, match=r"--refs llm calls an LLM, which fast mode \(--fast\)"):
        resolve_run_mode(s, explicit="fast", llm_flags=["--refs llm"])


def test_the_refusal_names_how_fast_was_chosen(monkeypatch):
    s = _settings(monkeypatch, mode="fast")
    with pytest.raises(ValueError, match=r"PIPELINE_MODE=fast"):
        resolve_run_mode(s, llm_flags=["--ref-seg llm"])
    with pytest.raises(ValueError, match=r"mode='fast'"):
        resolve_run_mode(s, explicit="fast", llm_flags=["llm='vllm'"], fast_label="mode='fast'")


def test_an_llm_only_option_with_an_explicit_full_runs_full(monkeypatch):
    s = _settings(monkeypatch, mode="fast")
    assert resolve_run_mode(s, explicit="full", llm_flags=["--refs llm"]) == "full"


def test_fast_skips_names_what_fast_mode_leaves_out():
    assert "LLM" in FAST_SKIPS
    assert "sections" in FAST_SKIPS
