"""Regression tests for audit action 16 (settings validation).

Presets and ``config set`` go through the real settings parser
(``validate_env_overrides``), numeric/probability settings are bounded,
choice-like settings are Literals, empty ``.env`` values fall back to the
default, and preset misuse fails cleanly. Each test fails on the pre-fix
tree and passes with the fix; guard tests pin the boundaries the fix must
not cross.
"""

from __future__ import annotations

import pytest

from bibr.presets import InvalidPresetError, PresetManager


@pytest.fixture()
def manager(tmp_path):
    return PresetManager(presets_dir=tmp_path / "presets")


@pytest.fixture()
def clean_env(monkeypatch):
    """Drop every settings-related process variable the test does not set itself."""
    for var in (
        "BIBR_RESOLVER_SOURCES",
        "BIBR_RESOLVER_LIMIT",
        "BIBR_RESOLVER_TIMEOUT",
        "LLM_PROVIDER",
        "LLM_BACKEND",
        "LLM_STRUCTURED_BACKEND",
        "LLM_INSTRUCTOR_MODE",
        "LLM_CHAT_TEMPLATE_KWARGS",
        "LLM_RATE_LIMIT_RPM",
        "LLM_LOCAL_MEM_FRACTION",
        "LLM_MAX_TOKENS",
        "LLM_MODEL",
        "OCR_BACKEND",
        "OCR_LOCAL_GPUS",
        "OCR_MAX_CONCURRENT_FILES",
        "OCR_MIN_SUCCESS_RATE",
        "OCR_VISION_RATE_LIMIT_RPM",
        "CROSSREF_RATE_LIMIT_RPM",
        "CROSSREF_ENRICH_CONCURRENCY",
        "CROSSREF_REQUEST_TIMEOUT",
        "CROSSREF_CONSOLIDATE",
        "CORS_ORIGINS",
        "LAYOUT_DPI",
        "LAYOUT_USE_GPU",
        "PIPELINE_MEMORY_MODE",
        "PIPELINE_MAX_CONCURRENT_POST_PARSE",
        "REF_PARSE_STRATEGY",
        "REF_GEOM_SEG_CASCADE_THRESHOLD",
        "ROR_ENRICH",
        "ROR_REQUESTS_PER_5MIN",
        "MCP_ENABLED",
        "JOBS_KEY_PREFIX",
        "REF_PARSE_MAX_TOKENS",
        "WTPSPLIT_MODEL",
        "SEGMENTER_USE_GPU",
    ):
        monkeypatch.delenv(var, raising=False)


# --- config-1: presets go through the real settings parser ------------------


def test_preset_applies_lists_dicts_and_literals(manager, clean_env):
    """List/dict preset values must arrive parsed; Literal/case validators run."""
    from bibr.config import GlobalSettings

    manager.save(
        "local",
        {
            "BIBR_RESOLVER_SOURCES": "crossref,openalex",
            "LLM_CHAT_TEMPLATE_KWARGS": '{"enable_thinking": false}',
            "PIPELINE_MEMORY_MODE": "balanced",
            "CROSSREF_CONSOLIDATE": "FILL",
        },
    )
    settings = GlobalSettings()
    assert manager.apply_to_settings("local", settings) == []
    assert settings.resolver.sources == ["crossref", "openalex"]
    assert settings.llm.chat_template_kwargs == {"enable_thinking": False}
    assert settings.pipeline.memory_mode == "balanced"
    assert settings.crossref.consolidate == "fill"


def test_preset_reruns_auto_tune_validators(manager, clean_env):
    """LLM_PROVIDER=ollama must auto-lower RPM; OCR_LOCAL_GPUS must scale regions."""
    from bibr.config import GlobalSettings

    manager.save("tuned", {"LLM_PROVIDER": "ollama", "OCR_LOCAL_GPUS": "4"})
    settings = GlobalSettings()
    assert settings.llm.rate_limit_rpm == 60
    manager.apply_to_settings("tuned", settings)
    assert settings.llm.provider == "ollama"
    assert settings.llm.rate_limit_rpm == 10
    assert settings.ocr.local_gpus == 4
    assert settings.ocr.max_concurrent_regions == 16 * settings.ocr.local_gpus


def test_preset_resolves_ror_mcp_and_alias_prefixes(manager, clean_env):
    """ROR_/MCP_ keys and aliases such as OCR_SGLANG_GPUS are known settings."""
    from bibr.config import GlobalSettings

    manager.save(
        "prefixes",
        {"ROR_ENRICH": "false", "MCP_ENABLED": "true", "OCR_SGLANG_GPUS": "2"},
    )
    settings = GlobalSettings()
    assert manager.apply_to_settings("prefixes", settings) == []
    assert settings.ror.enrich is False
    assert settings.mcp.enabled is True
    assert settings.ocr.local_gpus == 2


def test_preset_applies_to_the_settings_proxy(manager, clean_env):
    """The real chew/demo entry points pass the Settings proxy, not a bare model."""
    import bibr.config

    proxy = bibr.config.Settings
    object.__setattr__(proxy, "_instance", None)
    try:
        manager.save("toponly", {"WTPSPLIT_MODEL": "sat-12l-sm", "SEGMENTER_USE_GPU": "false"})
        unknown = manager.apply_to_settings("toponly", proxy)
        assert unknown == []
        assert proxy.WTPSPLIT_MODEL == "sat-12l-sm"
        assert proxy.SEGMENTER_USE_GPU is False
    finally:
        # Drop the mutated instance so later tests rebuild from the environment.
        object.__setattr__(proxy, "_instance", None)


def test_preset_invalid_value_names_the_setting(manager, clean_env):
    """A preset value that fails validation raises InvalidPresetError naming it."""
    from bibr.config import GlobalSettings

    manager.save("bad", {"LAYOUT_DPI": "0"})
    with pytest.raises(InvalidPresetError, match="LAYOUT_DPI"):
        manager.apply_to_settings("bad", GlobalSettings())


def test_preset_apply_preserves_untouched_sections(manager, clean_env):
    """Guard: applying a preset must not clobber unrelated in-memory values."""
    from bibr.config import GlobalSettings

    manager.save("ocronly", {"OCR_BACKEND": "gemini"})
    settings = GlobalSettings()
    settings.llm.model = "custom-in-memory"
    manager.apply_to_settings("ocronly", settings)
    assert settings.ocr.backend == "gemini"
    assert settings.llm.model == "custom-in-memory"


def test_preset_apply_still_reports_unknown_keys(manager, clean_env):
    """Guard: keys matching no setting are still returned, not applied."""
    from bibr.config import GlobalSettings

    manager.save("weird", {"LLM_PROVIDER": "google", "LLM_NONEXISTENT_FIELD": "x"})
    settings = GlobalSettings()
    unknown = manager.apply_to_settings("weird", settings)
    assert unknown == ["LLM_NONEXISTENT_FIELD"]
    assert settings.llm.provider == "google"


# --- config-2: one .env parser ----------------------------------------------


def test_parse_env_matches_the_runtime_parser(tmp_path):
    """Inline comments, `export` prefixes and escapes read as dotenv reads them."""
    from bibr.env_utils import parse_env, read_dotenv

    env_path = tmp_path / ".env"
    env_path.write_text(
        'LLM_PROVIDER=openai  # local vLLM\nexport OCR_BACKEND=glm-http\nLLM_API_KEY="it\\\'s"\n',
        encoding="utf-8",
    )
    assert parse_env(env_path) == {k: v for k, v in read_dotenv(env_path).items() if v is not None}
    assert parse_env(env_path)["LLM_PROVIDER"] == "openai"
    assert parse_env(env_path)["OCR_BACKEND"] == "glm-http"


def test_preset_save_use_roundtrip_with_comments_and_export(
    manager, tmp_path, clean_env, monkeypatch
):
    """save + use on a hand-edited .env must not bake comments into values."""
    from bibr.config import GlobalSettings

    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_PROVIDER=openai  # local vLLM\nexport OCR_BACKEND=glm-http\n",
        encoding="utf-8",
    )
    data = manager.snapshot_from_env(env_path)
    assert data == {"LLM_PROVIDER": "openai", "OCR_BACKEND": "glm-http"}
    manager.save("local", data)
    manager.apply("local", env_path)
    monkeypatch.setenv("BIBR_ENV_FILE", str(env_path))
    fresh = GlobalSettings()
    assert fresh.llm.provider == "openai"
    assert fresh.ocr.backend == "glm-http"


# --- config-3: end-anchored secret filter ------------------------------------


def test_is_secret_key_is_end_anchored():
    from bibr.presets import is_secret_key

    for key in (
        "LLM_MAX_TOKENS",
        "LLM_TITLE_MAX_TOKENS",
        "LLM_AUTHORS_MAX_TOKENS",
        "REF_PARSE_MAX_TOKENS",
        "OCR_VISION_MAX_TOKENS",
        "OCR_GENERATION_MAX_TOKENS",
        "FIG_MAX_TOKENS",
        "JOBS_KEY_PREFIX",
        "CORS_ALLOW_CREDENTIALS",
    ):
        assert not is_secret_key(key), key
    for key in ("LLM_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN", "REDIS_PASSWORD", "CLIENT_SECRET"):
        assert is_secret_key(key), key


def test_preset_snapshot_keeps_tuning_knobs(manager, tmp_path):
    """MAX_TOKENS knobs and JOBS_KEY_PREFIX must survive save; diff must show them."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_PROVIDER=google\n"
        "LLM_MAX_TOKENS=4096\n"
        "REF_PARSE_MAX_TOKENS=2048\n"
        "JOBS_KEY_PREFIX=myjobs\n"
        "LLM_API_KEY=sk-test-key-placeholder\n",
        encoding="utf-8",
    )
    data = manager.snapshot_from_env(env_path)
    assert data == {
        "LLM_PROVIDER": "google",
        "LLM_MAX_TOKENS": "4096",
        "REF_PARSE_MAX_TOKENS": "2048",
        "JOBS_KEY_PREFIX": "myjobs",
    }
    manager.save("small", data)
    changed, only_in_preset, only_in_env = manager.diff_against("small", data)
    assert (changed, only_in_preset, only_in_env) == ({}, {}, {})


# --- config-4: config set validates through the real model -------------------


def test_config_set_accepts_csv_and_json_lists(clean_env):
    from bibr import config_cli
    from bibr.config_introspect import iter_setting_docs

    docs = {doc.env_name: doc for doc in iter_setting_docs()}
    assert config_cli.validate_value(
        docs["CORS_ORIGINS"], "https://a.example,https://b.example"
    ) == [
        "https://a.example",
        "https://b.example",
    ]
    assert config_cli.validate_value(docs["CORS_ORIGINS"], '["https://a.example"]') == [
        "https://a.example"
    ]
    assert config_cli.validate_value(
        docs["LLM_CHAT_TEMPLATE_KWARGS"], '{"enable_thinking": false}'
    ) == {"enable_thinking": False}


def test_config_set_rejects_bad_values_naming_the_setting(clean_env):
    from bibr import config_cli
    from bibr.config_introspect import iter_setting_docs

    docs = {doc.env_name: doc for doc in iter_setting_docs()}
    with pytest.raises(ValueError, match="LAYOUT_DPI"):
        config_cli.validate_value(docs["LAYOUT_DPI"], "0")
    with pytest.raises(ValueError, match="BIBR_BUILD_SHA"):
        config_cli.validate_value(docs["BIBR_BUILD_SHA"], "abc")
    with pytest.raises(ValueError, match="WTPSPLIT_THRESHOLD"):
        config_cli.validate_value(docs["WTPSPLIT_THRESHOLD"], "7")
    with pytest.raises(ValueError, match="SERVE_LOG_LEVEL"):
        config_cli.validate_value(docs["SERVE_LOG_LEVEL"], "verbose")


def test_config_set_accepts_runtime_case_variants(clean_env):
    """Guard: values the runtime normalizes (REF_PARSE_STRATEGY=NER) still validate."""
    from bibr import config_cli
    from bibr.config_introspect import iter_setting_docs

    docs = {doc.env_name: doc for doc in iter_setting_docs()}
    assert config_cli.validate_value(docs["REF_PARSE_STRATEGY"], "NER") == "ner"


def test_config_set_never_leaks_secret_values(clean_env, monkeypatch):
    """Guard: when the environment is otherwise broken, the error must not echo a secret."""
    from bibr import config_cli
    from bibr.config_introspect import iter_setting_docs

    docs = {doc.env_name: doc for doc in iter_setting_docs()}
    # Break an unrelated cross-field rule so validation fails around the candidate.
    monkeypatch.setenv("WTPSPLIT_BLOCK_SIZE", "512")
    monkeypatch.delenv("WTPSPLIT_STRIDE", raising=False)
    with pytest.raises(ValueError) as exc_info:
        config_cli.validate_value(docs["GOOGLE_API_KEY"], "sk-live-secret-candidate")
    assert "sk-live-secret-candidate" not in str(exc_info.value)


# --- config-14: merge_env rewrites every duplicate ---------------------------


def test_merge_env_rewrites_every_duplicate(tmp_path):
    from bibr.env_utils import merge_env, read_dotenv

    env_path = tmp_path / ".env"
    env_path.write_text("LLM_MODEL=old\nOTHER=1\nLLM_MODEL=older-duplicate\n", encoding="utf-8")
    merge_env(env_path, {"LLM_MODEL": "new"})
    assert read_dotenv(env_path).get("LLM_MODEL") == "new"
    assert env_path.read_text(encoding="utf-8").count("LLM_MODEL=") == 1


def test_merge_env_matches_export_prefixed_lines(tmp_path):
    """Guard: an `export KEY=` line merges in place instead of appending a duplicate."""
    from bibr.env_utils import merge_env, read_dotenv

    env_path = tmp_path / ".env"
    env_path.write_text("export LLM_MODEL=old\n", encoding="utf-8")
    merge_env(env_path, {"LLM_MODEL": "new"})
    assert read_dotenv(env_path).get("LLM_MODEL") == "new"
    assert env_path.read_text(encoding="utf-8").count("LLM_MODEL=") == 1


# --- config-10: the resolver JSON form ---------------------------------------


def test_resolver_sources_accept_json_and_csv(clean_env):
    import os

    from bibr.config import GlobalSettings

    os.environ["BIBR_RESOLVER_SOURCES"] = '["crossref","openalex"]'
    try:
        assert GlobalSettings().resolver.sources == ["crossref", "openalex"]
    finally:
        del os.environ["BIBR_RESOLVER_SOURCES"]
    os.environ["BIBR_RESOLVER_SOURCES"] = "crossref,openalex"
    try:
        assert GlobalSettings().resolver.sources == ["crossref", "openalex"]
    finally:
        del os.environ["BIBR_RESOLVER_SOURCES"]


# --- config-5: zero rate/concurrency values fail fast ------------------------


@pytest.mark.parametrize(
    "var",
    [
        "LLM_RATE_LIMIT_RPM",
        "OCR_VISION_RATE_LIMIT_RPM",
        "OCR_MAX_CONCURRENT_FILES",
        "CROSSREF_RATE_LIMIT_RPM",
        "CROSSREF_ENRICH_CONCURRENCY",
        "BIBR_RESOLVER_LIMIT",
        "PIPELINE_MAX_CONCURRENT_POST_PARSE",
    ],
)
def test_zero_semaphore_and_rpm_values_are_rejected(var, monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv(var, "0")
    with pytest.raises(ValidationError):
        GlobalSettings()


@pytest.mark.parametrize(
    "var",
    ["CROSSREF_REQUEST_TIMEOUT", "BIBR_RESOLVER_TIMEOUT", "ROR_REQUEST_TIMEOUT"],
)
def test_zero_timeouts_are_rejected(var, monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv(var, "0")
    with pytest.raises(ValidationError):
        GlobalSettings()


def test_zero_rejection_names_the_setting(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings, _configuration_error

    monkeypatch.setenv("LLM_RATE_LIMIT_RPM", "0")
    with pytest.raises(ValidationError) as exc_info:
        GlobalSettings()
    assert "LLM_RATE_LIMIT_RPM" in str(_configuration_error(exc_info.value))


def test_boundary_values_stay_accepted(monkeypatch):
    """Guard: 1 is a valid RPM/concurrency; 0 stays valid where it means unlimited/off."""
    from bibr.config import GlobalSettings

    monkeypatch.setenv("LLM_RATE_LIMIT_RPM", "1")
    monkeypatch.setenv("OCR_MIN_SUCCESS_RATE", "0")
    monkeypatch.setenv("LLM_LOCAL_MEM_FRACTION", "1")
    settings = GlobalSettings()
    assert settings.llm.rate_limit_rpm == 1
    assert settings.ocr.min_success_rate == 0
    assert settings.llm.local_mem_fraction == 1


# --- config-16: choice-like settings are Literals ----------------------------


def test_unknown_provider_fails_with_choices(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings, _configuration_error

    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValidationError) as exc_info:
        GlobalSettings()
    message = str(_configuration_error(exc_info.value))
    assert "LLM_PROVIDER" in message
    for choice in ("google", "openai", "anthropic", "groq", "ollama"):
        assert choice in message


@pytest.mark.parametrize("provider", ["google", "openai", "anthropic", "groq", "ollama"])
def test_known_providers_load(provider, monkeypatch):
    """Guard: every documented provider still loads."""
    from bibr.config import GlobalSettings

    monkeypatch.setenv("LLM_PROVIDER", provider)
    assert GlobalSettings().llm.provider == provider


def test_json_object_alias_maps_to_json(monkeypatch):
    """LLM_INSTRUCTOR_MODE=json_object (the description's own spelling) means json."""
    from bibr.config import GlobalSettings

    monkeypatch.setenv("LLM_INSTRUCTOR_MODE", "json_object")
    assert GlobalSettings().llm.instructor_mode == "json"


def test_unknown_structured_backend_and_spec_decode_fail(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv("LLM_STRUCTURED_BACKEND", "bogus")
    with pytest.raises(ValidationError):
        GlobalSettings()
    monkeypatch.delenv("LLM_STRUCTURED_BACKEND")
    monkeypatch.setenv("RAPID_MLX_SPEC_DECODE", "bogus")
    with pytest.raises(ValidationError):
        GlobalSettings()


# --- config-9: empty values and the full example -----------------------------


def test_empty_value_falls_back_to_default_and_null_sets_none(monkeypatch):
    from bibr.config import GlobalSettings

    monkeypatch.setenv("LAYOUT_USE_GPU", "")
    monkeypatch.setenv("BIBR_RESOLVER_SOURCES", "")
    assert GlobalSettings().layout.use_gpu is None
    assert GlobalSettings().resolver.sources == ["crossref"]
    monkeypatch.setenv("PIPELINE_MEMORY_MODE", "null")
    assert GlobalSettings().pipeline.memory_mode is None


def test_full_example_every_line_loads_and_matches_default():
    """Uncommenting any `bibr config example --full` line must load unchanged."""
    import os
    import re

    from bibr.config import GlobalSettings
    from bibr.config_cli import render_env_example

    pairs: list[tuple[str, str]] = []
    for line in render_env_example(full=True).splitlines():
        match = re.match(r"^# ([A-Z][A-Z0-9_]*)=(.*)$", line)
        if match:
            pairs.append((match.group(1), match.group(2)))
    assert len(pairs) > 200
    # Compare against defaults with every example key unset, so harness pins
    # (REDIS_URL, REF_PARSE_STRATEGY, ...) cannot masquerade as a behavior
    # change when their example line renders the production default.
    saved = {key: os.environ.pop(key) for key, _ in pairs if key in os.environ}
    try:
        baseline = GlobalSettings().model_dump(mode="json")
        failures: list[str] = []
        for key, value in pairs:
            os.environ[key] = value
            try:
                dump = GlobalSettings().model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{key}={value}: fails to load ({type(exc).__name__})")
            else:
                if dump != baseline:
                    failures.append(f"{key}={value}: silently changes behavior")
            finally:
                del os.environ[key]
    finally:
        os.environ.update(saved)
    assert failures == []


def test_full_example_renders_lists_as_csv_and_nulls_explicitly():
    from bibr.config_cli import render_env_example

    text = render_env_example(full=True)
    assert "# CORS_ALLOW_METHODS=*" in text
    assert "['*']" not in text
    assert "# LLM_API_KEY=null" in text


# --- local-cli-15: preset misuse fails cleanly --------------------------------


def test_invalid_preset_name_is_a_configuration_error():
    from bibr.exceptions import ConfigurationError

    assert issubclass(InvalidPresetError, ConfigurationError)
    manager = PresetManager(presets_dir="/nonexistent-presets-dir")
    with pytest.raises(InvalidPresetError):
        manager.load("my preset")


def test_corrupt_preset_file_names_the_file(manager):
    manager.directory.mkdir(parents=True, exist_ok=True)
    (manager.directory / "corrupt.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(InvalidPresetError, match="corrupt"):
        manager.load("corrupt")


def test_preset_show_bad_name_exits_cleanly(manager, monkeypatch, capsys):
    """`bibr preset show ../x` must exit 1 with a message, not a traceback."""
    import argparse

    from bibr.local.cli.presets import _run_preset

    monkeypatch.setenv("BIBR_PRESETS_DIR", str(manager.directory))
    with pytest.raises(SystemExit) as exc_info:
        _run_preset(argparse.Namespace(preset_command="show", name="../x"))
    assert exc_info.value.code == 1
    out = capsys.readouterr()
    assert "Traceback" not in out.out + out.err


def test_chew_preset_bad_name_raises_configuration_error(manager, monkeypatch):
    """`chew --preset 'my preset'` surfaces as ConfigurationError (clean exit 1)."""
    from argparse import Namespace

    from bibr.exceptions import ConfigurationError
    from bibr.local.cli.run_config import _apply_runtime_settings

    monkeypatch.setenv("BIBR_PRESETS_DIR", str(manager.directory))
    args = Namespace(
        preset="my preset",
        llm_provider=None,
        llm_model=None,
        no_equations=False,
        refs=None,
        ref_seg=None,
    )
    with pytest.raises(ConfigurationError):
        _apply_runtime_settings(args)
