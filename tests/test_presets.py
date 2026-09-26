import os
import subprocess
import sys

import pytest

from bibr.presets import InvalidPresetError, PresetManager


@pytest.fixture()
def manager(tmp_path):
    return PresetManager(presets_dir=tmp_path)


def test_save_and_load(manager):
    data = {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4o"}
    manager.save("my-openai", data)
    loaded = manager.load("my-openai")
    assert loaded == data


def test_save_creates_directory(tmp_path):
    nested = tmp_path / "sub" / "presets"
    mgr = PresetManager(presets_dir=nested)
    mgr.save("test", {"LLM_PROVIDER": "google"})
    assert nested.exists()
    assert mgr.load("test") == {"LLM_PROVIDER": "google"}


def test_list_empty(manager):
    assert manager.list_presets() == []


def test_list_returns_sorted_names(manager):
    manager.save("beta", {"LLM_PROVIDER": "openai"})
    manager.save("alpha", {"LLM_PROVIDER": "google"})
    assert manager.list_presets() == ["alpha", "beta"]


def test_delete(manager):
    manager.save("temp", {"LLM_PROVIDER": "google"})
    assert "temp" in manager.list_presets()
    manager.delete("temp")
    assert "temp" not in manager.list_presets()


def test_delete_nonexistent_raises(manager):
    with pytest.raises(FileNotFoundError):
        manager.delete("ghost")


def test_load_nonexistent_raises(manager):
    with pytest.raises(FileNotFoundError):
        manager.load("ghost")


def test_save_validates_name(manager):
    with pytest.raises(InvalidPresetError):
        manager.save("../escape", {"LLM_PROVIDER": "google"})
    with pytest.raises(InvalidPresetError):
        manager.save("", {"LLM_PROVIDER": "google"})
    with pytest.raises(InvalidPresetError):
        manager.save("has spaces", {"LLM_PROVIDER": "google"})


def test_save_overwrites_existing(manager):
    manager.save("mine", {"LLM_PROVIDER": "google"})
    manager.save("mine", {"LLM_PROVIDER": "openai"})
    assert manager.load("mine") == {"LLM_PROVIDER": "openai"}


def test_exists(manager):
    assert not manager.exists("nope")
    manager.save("yep", {"LLM_PROVIDER": "google"})
    assert manager.exists("yep")


def test_apply_writes_to_env(manager, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=google\nLLM_MODEL=old-model\n", encoding="utf-8")

    manager.save("fast", {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-5-nano"})
    manager.apply("fast", env_path)

    content = env_path.read_text()
    assert "LLM_PROVIDER=openai" in content
    assert "LLM_MODEL=gpt-5-nano" in content
    assert "BIBR_ACTIVE_PRESET=fast" in content


def test_apply_preserves_unrelated_keys(manager, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("CROSSREF_API_EMAIL=me@x.com\nLLM_PROVIDER=google\n", encoding="utf-8")

    manager.save("mini", {"LLM_PROVIDER": "openai"})
    manager.apply("mini", env_path)

    content = env_path.read_text()
    assert "CROSSREF_API_EMAIL=me@x.com" in content
    assert "LLM_PROVIDER=openai" in content


def test_apply_nonexistent_raises(manager, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        manager.apply("ghost", env_path)


def test_get_active_preset(manager, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("BIBR_ACTIVE_PRESET=fast\nLLM_PROVIDER=openai\n", encoding="utf-8")
    assert manager.get_active(env_path) == "fast"


def test_get_active_preset_none(manager, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    assert manager.get_active(env_path) is None


def test_snapshot_from_env(manager, tmp_path):
    """snapshot_from_env should capture all KEY=VALUE lines (except BIBR_ACTIVE_PRESET)."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\nLLM_PROVIDER=google\nLLM_MODEL=gemini\nBIBR_ACTIVE_PRESET=old\n",
        encoding="utf-8",
    )
    data = manager.snapshot_from_env(env_path)
    assert data == {"LLM_PROVIDER": "google", "LLM_MODEL": "gemini"}
    assert "BIBR_ACTIVE_PRESET" not in data


def test_snapshot_from_env_strips_surrounding_quotes(manager, tmp_path):
    """snapshot_from_env must strip quotes so quoted .env values don't leak
    quote characters into the saved preset JSON."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_PROVIDER='google'\nPLAIN=value\nWTPSPLIT_MODEL=\"sat-12l\"\n",
        encoding="utf-8",
    )
    data = manager.snapshot_from_env(env_path)
    assert data == {"LLM_PROVIDER": "google", "PLAIN": "value", "WTPSPLIT_MODEL": "sat-12l"}


def test_snapshot_excludes_secret_keys_by_default(manager, tmp_path):
    """API keys / tokens / passwords must not be captured into a preset.

    Presets are intended to be shareable. Secrets stay in .env.
    """
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_PROVIDER=google\n"
        "GOOGLE_API_KEY=AIzaSy-abc\n"
        "LLM_API_KEY=sk-abc\n"
        "HF_TOKEN=hf_abc\n"
        "REDIS_PASSWORD=hunter2\n"
        "ANTHROPIC_API_KEY=sk-ant-abc\n",
        encoding="utf-8",
    )
    data = manager.snapshot_from_env(env_path)
    assert data == {"LLM_PROVIDER": "google"}


def test_snapshot_keeps_rate_limit_keys(manager, tmp_path):
    """Rate-limit knobs share substrings with secrets; must NOT be filtered."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_PROVIDER=google\nCROSSREF_RATE_LIMIT_RPM=600\nLLM_RATE_LIMIT_RPM=450\n",
        encoding="utf-8",
    )
    data = manager.snapshot_from_env(env_path)
    assert data == {
        "LLM_PROVIDER": "google",
        "CROSSREF_RATE_LIMIT_RPM": "600",
        "LLM_RATE_LIMIT_RPM": "450",
    }


def test_snapshot_include_secrets_opt_in(manager, tmp_path):
    """include_secrets=True keeps API keys (used internally for migration)."""
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=google\nGOOGLE_API_KEY=AIzaSy-abc\n", encoding="utf-8")
    data = manager.snapshot_from_env(env_path, include_secrets=True)
    assert data == {"LLM_PROVIDER": "google", "GOOGLE_API_KEY": "AIzaSy-abc"}


def test_cli_preset_list_empty(tmp_path, monkeypatch):
    """bibr preset list with no presets shows empty message."""
    monkeypatch.setenv("BIBR_PRESETS_DIR", str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "bibr.local.cli", "preset", "list"],
        capture_output=True,
        text=True,
        env={**os.environ, "BIBR_PRESETS_DIR": str(tmp_path)},
    )
    assert result.returncode == 0
    assert "No presets" in result.stdout or "no presets" in result.stdout.lower()


def test_cli_preset_save_and_list(tmp_path, monkeypatch):
    """bibr preset save snapshots .env, then list shows it."""
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=google\nLLM_MODEL=gemini\n", encoding="utf-8")
    presets_dir = tmp_path / "presets"

    result = subprocess.run(
        [sys.executable, "-m", "bibr.local.cli", "preset", "save", "my-google"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env={**os.environ, "BIBR_PRESETS_DIR": str(presets_dir)},
    )
    assert result.returncode == 0

    result = subprocess.run(
        [sys.executable, "-m", "bibr.local.cli", "preset", "list"],
        capture_output=True,
        text=True,
        env={**os.environ, "BIBR_PRESETS_DIR": str(presets_dir)},
    )
    assert "my-google" in result.stdout


def test_redact_value_masks_secrets(manager):
    from bibr.presets import redact_value

    # Long secret: keep first 4 + last 2.
    assert redact_value("GOOGLE_API_KEY", "AIzaSyAbcDefGhiJklMno") == "AIza…no"
    # Short secret: fully masked.
    assert redact_value("HF_TOKEN", "hf_x") == "***"
    # Non-secret short: passthrough.
    assert redact_value("LLM_PROVIDER", "google") == "google"
    # Rate-limit knob is not a secret even though "_LIMIT_" looks key-ish.
    assert redact_value("CROSSREF_RATE_LIMIT_RPM", "600") == "600"


def test_is_secret_key_recognizes_common_shapes():
    from bibr.presets import is_secret_key

    for k in ["GOOGLE_API_KEY", "LLM_API_KEY", "HF_TOKEN", "REDIS_PASSWORD", "CLIENT_SECRET"]:
        assert is_secret_key(k), k
    for k in [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OCR_BACKEND",
        "CROSSREF_RATE_LIMIT_RPM",
        "LLM_RATE_LIMIT_RPM",
    ]:
        assert not is_secret_key(k), k


def test_apply_to_settings_dispatches_to_nested_sections(manager, tmp_path, monkeypatch):
    """apply_to_settings should hit every settings section, not just llm/ocr."""
    from bibr.config import GlobalSettings

    monkeypatch.delenv("CROSSREF_API_EMAIL", raising=False)
    monkeypatch.delenv("CROSSREF_RATE_LIMIT_RPM", raising=False)
    monkeypatch.delenv("OCR_VISION_PROVIDER", raising=False)

    settings = GlobalSettings()
    manager.save(
        "fullsweep",
        {
            "BIBR_RESOLVER_ENRICH": "false",
            "FIG_EXTRACT": "meta",
            "JOBS_MAX_RUNNING": "3",
            "METER_ENABLED": "false",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-5",
            "OCR_BACKEND": "gemini",
            "OCR_VISION_PROVIDER": "anthropic",
            "CROSSREF_API_EMAIL": "test@x.com",
            "CROSSREF_RATE_LIMIT_RPM": "999",
        },
    )
    unknown = manager.apply_to_settings("fullsweep", settings)
    assert unknown == []
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-5"
    assert settings.ocr.backend == "gemini"
    assert settings.ocr_vision.provider == "anthropic"
    assert settings.crossref.api_email == "test@x.com"
    assert settings.crossref.rate_limit_rpm == 999  # int coercion
    assert settings.resolver.enrich is False
    assert settings.fig.extract == "meta"
    assert settings.jobs.max_running == 3
    assert settings.metering.enabled is False


def test_apply_to_settings_falls_back_to_top_level(manager, monkeypatch):
    """Keys like OCR_BASE_URL share a section's prefix but live on the
    top-level GlobalSettings model. apply_to_settings must not report them
    as unknown."""
    from bibr.config import GlobalSettings

    monkeypatch.delenv("OCR_BASE_URL", raising=False)
    monkeypatch.delenv("WTPSPLIT_MODEL", raising=False)

    settings = GlobalSettings()
    manager.save(
        "toplevel",
        {"OCR_BASE_URL": "http://other:9000", "WTPSPLIT_MODEL": "sat-12l-sm"},
    )
    unknown = manager.apply_to_settings("toplevel", settings)
    assert unknown == []
    assert settings.OCR_BASE_URL == "http://other:9000"
    assert settings.WTPSPLIT_MODEL == "sat-12l-sm"


def test_apply_to_settings_reports_unknown_keys(manager):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    manager.save("weird", {"LLM_PROVIDER": "google", "LLM_NONEXISTENT_FIELD": "x"})
    unknown = manager.apply_to_settings("weird", settings)
    assert "LLM_NONEXISTENT_FIELD" in unknown
    assert settings.llm.provider == "google"


def test_deactivate_clears_only_marker(manager, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_PROVIDER=google\nBIBR_ACTIVE_PRESET=mine\nLLM_MODEL=gemini\n",
        encoding="utf-8",
    )
    assert manager.deactivate(env_path) is True
    content = env_path.read_text()
    assert "BIBR_ACTIVE_PRESET" not in content
    assert "LLM_PROVIDER=google" in content
    assert "LLM_MODEL=gemini" in content


def test_deactivate_no_marker_returns_false(manager, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=google\n", encoding="utf-8")
    assert manager.deactivate(env_path) is False


def test_diff_against_categorizes_keys(manager, tmp_path):
    manager.save("ref", {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-5", "EXTRA": "x"})
    env_dict = {
        "LLM_PROVIDER": "google",  # changed
        "LLM_MODEL": "gpt-5",  # same
        "CROSSREF_API_EMAIL": "me@x.com",  # only in env (non-secret)
        "GOOGLE_API_KEY": "AIzaSy",  # only in env, secret → hidden from diff
        "BIBR_ACTIVE_PRESET": "ref",  # ignored
    }
    changed, only_in_preset, only_in_env = manager.diff_against("ref", env_dict)
    assert changed == {"LLM_PROVIDER": ("google", "openai")}
    assert only_in_preset == {"EXTRA": "x"}
    assert only_in_env == {"CROSSREF_API_EMAIL": "me@x.com"}
    assert "GOOGLE_API_KEY" not in only_in_env
    assert "BIBR_ACTIVE_PRESET" not in only_in_env


def test_load_v0_legacy_format(manager, tmp_path):
    """Old presets without schema_version must still load correctly."""
    import json

    manager._dir.mkdir(parents=True, exist_ok=True)  # noqa: SLF001
    legacy = manager._dir / "old.json"  # noqa: SLF001
    legacy.write_text(
        json.dumps({"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"}), encoding="utf-8"
    )
    assert manager.load("old") == {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"}


def test_save_emits_v1_format(manager):
    import json

    manager.save("new", {"LLM_PROVIDER": "google"})
    raw = json.loads(manager._path("new").read_text())  # noqa: SLF001
    assert raw["schema_version"] == 1
    assert raw["settings"] == {"LLM_PROVIDER": "google"}


def test_full_lifecycle(tmp_path):
    """Test save → list → show → use → apply → rm lifecycle."""
    presets_dir = tmp_path / "presets"
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_PROVIDER=google\nLLM_MODEL=gemini\nCROSSREF_API_EMAIL=a@b.com\n",
        encoding="utf-8",
    )

    manager = PresetManager(presets_dir=presets_dir)

    # Save from env snapshot
    data = manager.snapshot_from_env(env_path)
    manager.save("google-default", data)

    # Save a second preset manually
    manager.save(
        "openai-fast",
        {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-5-nano",
            "LLM_API_KEY": "sk-test",
        },
    )

    # List
    assert manager.list_presets() == ["google-default", "openai-fast"]

    # Show
    loaded = manager.load("openai-fast")
    assert loaded["LLM_PROVIDER"] == "openai"
    assert loaded["LLM_MODEL"] == "gpt-5-nano"

    # Apply
    manager.apply("openai-fast", env_path)
    content = env_path.read_text()
    assert "LLM_PROVIDER=openai" in content
    assert "LLM_MODEL=gpt-5-nano" in content
    assert "CROSSREF_API_EMAIL=a@b.com" in content
    assert "BIBR_ACTIVE_PRESET=openai-fast" in content

    # Active
    assert manager.get_active(env_path) == "openai-fast"

    # Delete
    manager.delete("openai-fast")
    assert manager.list_presets() == ["google-default"]

    # Verify google-default still works
    manager.apply("google-default", env_path)
    content = env_path.read_text()
    assert "LLM_PROVIDER=google" in content
    assert "BIBR_ACTIVE_PRESET=google-default" in content


# --- which .env file `bibr preset` works on ----------------------------------


@pytest.fixture()
def home_only_config(tmp_path, monkeypatch):
    """Config in ~/.bibr/.env, run from a project directory that has no .env."""
    from pathlib import Path

    home = tmp_path / "home"
    (home / ".bibr").mkdir(parents=True)
    env_file = home / ".bibr" / ".env"
    env_file.write_text("LLM_PROVIDER=google\nLLM_MODEL=gemini-3.5-flash-lite\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("BIBR_ENV_FILE", raising=False)
    monkeypatch.setenv("BIBR_PRESETS_DIR", str(tmp_path / "presets"))
    return env_file, project


def _preset(command, **kwargs):
    import argparse

    from bibr.local.cli.presets import _run_preset

    _run_preset(argparse.Namespace(preset_command=command, **kwargs))


def test_preset_save_reads_the_home_env_file(home_only_config, tmp_path):
    """Settings come from ~/.bibr/.env here, so that is what a preset snapshots."""
    _preset("save", name="home", force=True)

    assert PresetManager(presets_dir=tmp_path / "presets").load("home") == {
        "LLM_PROVIDER": "google",
        "LLM_MODEL": "gemini-3.5-flash-lite",
    }


def test_preset_use_writes_the_env_file_in_effect(home_only_config, tmp_path):
    env_file, project = home_only_config
    PresetManager(presets_dir=tmp_path / "presets").save("small", {"LLM_MODEL": "gemini-x"})

    _preset("use", name="small")

    assert env_file.read_text(encoding="utf-8") == (
        "LLM_PROVIDER=google\nLLM_MODEL=gemini-x\n\nBIBR_ACTIVE_PRESET=small\n"
    )
    assert not (project / ".env").exists()


def test_preset_works_on_the_project_env_when_both_files_exist(home_only_config, tmp_path):
    """./.env overrides ~/.bibr/.env, so it is the file a preset is applied to."""
    home_env, project = home_only_config
    home_before = home_env.read_text(encoding="utf-8")
    project_env = project / ".env"
    project_env.write_text("LLM_PROVIDER=openai\nLLM_MODEL=gpt-5-nano\n", encoding="utf-8")
    manager = PresetManager(presets_dir=tmp_path / "presets")

    _preset("save", name="project", force=True)
    manager.save("small", {"LLM_MODEL": "gpt-5-mini"})
    _preset("use", name="small")

    assert manager.load("project") == {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-5-nano"}
    assert project_env.read_text(encoding="utf-8") == (
        "LLM_PROVIDER=openai\nLLM_MODEL=gpt-5-mini\n\nBIBR_ACTIVE_PRESET=small\n"
    )
    assert home_env.read_text(encoding="utf-8") == home_before


def test_effective_env_file_follows_the_dotenv_chain(tmp_path, monkeypatch):
    from pathlib import Path

    from bibr.presets import effective_env_file

    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".bibr").mkdir(parents=True)
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("BIBR_ENV_FILE", raising=False)

    # Neither file exists: the one the chain reads last, ./.env.
    assert effective_env_file() == project / ".env"
    (home / ".bibr" / ".env").write_text("LLM_PROVIDER=google\n", encoding="utf-8")
    assert effective_env_file() == home / ".bibr" / ".env"
    (project / ".env").write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    assert effective_env_file() == project / ".env"

    # BIBR_ENV_FILE replaces the chain; an empty value loads no file at all.
    listed = tmp_path / "ci.env"
    monkeypatch.setenv("BIBR_ENV_FILE", str(listed))
    assert effective_env_file() == listed
    monkeypatch.setenv("BIBR_ENV_FILE", "")
    assert effective_env_file() == project / ".env"
