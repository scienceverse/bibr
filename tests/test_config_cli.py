"""Tests for the ``bibr config`` subcommand suite (show/path/set/example).

Redaction is the load-bearing property here: several tests assert a full
secret value never appears anywhere in captured output, not just that a
masked form is present.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from bibr import config_cli
from bibr.config_introspect import iter_setting_docs

_DOCS_BY_NAME = {d.env_name: d for d in iter_setting_docs()}


def _doc(name: str):
    return _DOCS_BY_NAME[name]


@pytest.fixture()
def env_chain(tmp_path, monkeypatch):
    """Exercise the real runtime chain with an isolated CWD and home."""
    cwd_env = tmp_path / "cwd" / ".env"
    home_env = tmp_path / "home" / ".bibr" / ".env"
    cwd_env.parent.mkdir(parents=True, exist_ok=True)
    home_env.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd_env.parent)
    monkeypatch.setattr(Path, "home", lambda: home_env.parent.parent)
    monkeypatch.delenv("BIBR_ENV_FILE", raising=False)
    return cwd_env, home_env


def _clean_env(monkeypatch, *names: str) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# resolve_provenance
# ---------------------------------------------------------------------------


def test_provenance_env_beats_dotenv_beats_home_beats_default(env_chain, monkeypatch):
    cwd_env, home_env = env_chain
    doc = _doc("CROSSREF_API_EMAIL")
    _clean_env(monkeypatch, "CROSSREF_API_EMAIL")

    # Nothing set anywhere -> default.
    prov = config_cli.resolve_provenance(doc)
    assert prov.tier == "default"
    assert prov.value is None

    # Only in ~/.bibr/.env -> dotenv, that path.
    home_env.write_text("CROSSREF_API_EMAIL=home@example.com\n")
    prov = config_cli.resolve_provenance(doc)
    assert prov.tier == "dotenv"
    assert prov.value == "home@example.com"
    assert prov.path == str(home_env)

    # Also in CWD ./.env -> CWD wins over home.
    cwd_env.write_text("CROSSREF_API_EMAIL=cwd@example.com\n")
    prov = config_cli.resolve_provenance(doc)
    assert prov.tier == "dotenv"
    assert prov.value == "cwd@example.com"
    assert prov.path == str(cwd_env)

    # Process env beats both dotenv files.
    monkeypatch.setenv("CROSSREF_API_EMAIL", "env@example.com")
    prov = config_cli.resolve_provenance(doc)
    assert prov.tier == "env"
    assert prov.value == "env@example.com"
    assert prov.path is None


def test_provenance_checks_aliases(env_chain, monkeypatch):
    """GOOGLE_API_KEY's alias GEMINI_API_KEY must also resolve provenance."""
    _clean_env(monkeypatch, "GOOGLE_API_KEY", "GEMINI_API_KEY", "LANGEXTRACT_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-1234567890abcdef")
    doc = _doc("GOOGLE_API_KEY")
    prov = config_cli.resolve_provenance(doc)
    assert prov.tier == "env"
    assert prov.value == "sk-1234567890abcdef"


def test_dotenv_override_matches_runtime_and_set_target(env_chain, monkeypatch, capsys):
    from bibr.config import LlmOptions

    cwd_env, home_env = env_chain
    _clean_env(monkeypatch, "LLM_MODEL")
    cwd_env.write_text("LLM_MODEL=ignored-cwd\n")
    home_env.write_text("LLM_MODEL=ignored-home\n")
    first = cwd_env.parent / "first.env"
    last = cwd_env.parent / "last.env"
    first.write_text("LLM_MODEL=first-model\n")
    last.write_text("LLM_MODEL=last-model\n")
    monkeypatch.setenv("BIBR_ENV_FILE", os.pathsep.join(map(str, (first, last))))

    assert LlmOptions().model == "last-model"
    assert config_cli.resolve_provenance(_doc("LLM_MODEL")) == ("last-model", "dotenv", str(last))
    assert config_cli.run_config_command(_args(config_command="path")) == 0
    out = capsys.readouterr().out
    assert str(first) in out and str(last) in out
    assert str(cwd_env) not in out and str(home_env) not in out

    assert (
        config_cli.run_config_command(
            _args(config_command="set", key="LLM_MODEL", value="updated-model")
        )
        == 0
    )
    assert LlmOptions().model == "updated-model"
    assert "updated-model" in last.read_text()
    assert cwd_env.read_text() == "LLM_MODEL=ignored-cwd\n"


def test_disabled_dotenv_does_not_read_or_write_files(env_chain, monkeypatch, capsys):
    from bibr.config import LlmOptions

    cwd_env, _ = env_chain
    _clean_env(monkeypatch, "LLM_MODEL")
    cwd_env.write_text("LLM_MODEL=ignored-model\n")
    monkeypatch.setenv("BIBR_ENV_FILE", "")

    assert LlmOptions().model == LlmOptions.model_fields["model"].default
    assert config_cli.resolve_provenance(_doc("LLM_MODEL")).tier == "default"
    assert config_cli.run_config_command(_args(config_command="path")) == 0
    assert "disabled" in capsys.readouterr().out
    assert (
        config_cli.run_config_command(
            _args(config_command="set", key="LLM_MODEL", value="unused-model")
        )
        == 2
    )
    assert "disabled" in capsys.readouterr().out
    assert cwd_env.read_text() == "LLM_MODEL=ignored-model\n"


def test_provenance_uses_literal_dotenv_values_and_case_insensitive_keys(env_chain, monkeypatch):
    from bibr.config import LlmOptions

    cwd_env, _ = env_chain
    _clean_env(monkeypatch, "LLM_MODEL")
    monkeypatch.setenv("BIBR_AUDIT_MODEL", "expanded-model")
    cwd_env.write_text('llm_model="${BIBR_AUDIT_MODEL}"\n')
    assert config_cli.resolve_provenance(_doc("LLM_MODEL")).value == LlmOptions().model
    assert LlmOptions().model == "${BIBR_AUDIT_MODEL}"

    monkeypatch.setenv("llm_model", "environment-model")
    assert config_cli.resolve_provenance(_doc("LLM_MODEL")).value == LlmOptions().model
    assert LlmOptions().model == "environment-model"


def test_provenance_merges_files_before_resolving_aliases(env_chain, monkeypatch):
    from bibr.config import OcrOptions

    cwd_env, home_env = env_chain
    _clean_env(monkeypatch, "OCR_LOCAL_GPUS", "OCR_SGLANG_GPUS")
    home_env.write_text("OCR_LOCAL_GPUS=2\n")
    cwd_env.write_text("OCR_SGLANG_GPUS=3\n")
    prov = config_cli.resolve_provenance(_doc("OCR_LOCAL_GPUS"))
    assert int(prov.value) == OcrOptions().local_gpus == 2
    assert prov.path == str(home_env)


# ---------------------------------------------------------------------------
# format_value (redaction)
# ---------------------------------------------------------------------------


def test_format_value_redacts_secret():
    doc = _doc("LLM_API_KEY")
    assert doc.is_secret
    out = config_cli.format_value(doc, "sk-1234567890abcdef")
    assert out == "sk-1…cdef"
    assert "sk-1234567890abcdef" not in out


def test_format_value_fully_masks_short_secret():
    doc = _doc("LLM_API_KEY")
    out = config_cli.format_value(doc, "short1")
    assert "short1" not in out


def test_format_value_passes_through_non_secret():
    doc = _doc("CROSSREF_API_EMAIL")
    assert not doc.is_secret
    assert config_cli.format_value(doc, "person@example.com") == "person@example.com"


def test_llm_max_tokens_is_not_secret():
    """End-anchored regex: _TOKENS (plural) must not be treated as a secret."""
    doc = _doc("LLM_MAX_TOKENS")
    assert not doc.is_secret
    assert config_cli.format_value(doc, "65536") == "65536"


def test_llm_task_max_tokens_is_not_secret():
    doc = _doc("LLM_AUTHORS_MAX_TOKENS")
    assert not doc.is_secret
    assert config_cli.format_value(doc, "8192") == "8192"


# ---------------------------------------------------------------------------
# suggest_key
# ---------------------------------------------------------------------------


def test_suggest_key_close_match():
    names = [d.env_name for d in iter_setting_docs()]
    assert config_cli.suggest_key("CROSSREF_API_EMIAL", names) == "CROSSREF_API_EMAIL"


def test_suggest_key_no_match():
    assert config_cli.suggest_key("TOTALLY_UNRELATED_GARBAGE_XYZ", ["LLM_PROVIDER"]) is None


# ---------------------------------------------------------------------------
# validate_value
# ---------------------------------------------------------------------------


def test_validate_value_int_rejects_non_numeric():
    doc = _doc("CROSSREF_RATE_LIMIT_RPM")
    with pytest.raises(ValueError, match="valid integer"):
        config_cli.validate_value(doc, "abc")


def test_validate_value_int_accepts_numeric():
    doc = _doc("CROSSREF_RATE_LIMIT_RPM")
    assert config_cli.validate_value(doc, "42") == 42


def test_validate_value_literal_lists_choices():
    doc = _doc("REF_SEG_STRATEGY")
    with pytest.raises(ValueError) as exc_info:
        config_cli.validate_value(doc, "bogus")
    message = str(exc_info.value)
    for choice in ("llm", "crf", "geom", "region"):
        assert choice in message


def test_validate_value_literal_accepts_valid_choice():
    doc = _doc("REF_SEG_STRATEGY")
    assert config_cli.validate_value(doc, "geom") == "geom"


# ---------------------------------------------------------------------------
# render_env_example
# ---------------------------------------------------------------------------


def test_render_env_example_minimal_has_four_keys():
    text = config_cli.render_env_example(full=False)
    for key in ("LLM_PROVIDER", "GOOGLE_API_KEY", "CROSSREF_API_EMAIL", "OCR_BACKEND"):
        assert key in text
    # Minimal template must not blow up into the full settings surface.
    assert "LLM_RATE_LIMIT_RPM" not in text
    assert "REDIS_PASSWORD" not in text


def test_render_env_example_minimal_never_prints_fake_secret_value():
    text = config_cli.render_env_example(full=False)
    assert "GOOGLE_API_KEY=" in text
    assert "GOOGLE_API_KEY=your-google-api-key-here" not in text
    # Bare key=, nothing trailing on that line.
    line = next(line for line in text.splitlines() if line.startswith("GOOGLE_API_KEY="))
    assert line == "GOOGLE_API_KEY="


def test_render_env_example_full_covers_every_setting_grouped_by_section():
    text = config_cli.render_env_example(full=True)
    assert "OCR_BACKEND" in text
    assert "LLM_RATE_LIMIT_RPM" in text
    assert "CROSSREF_API_EMAIL" in text
    # Every setting line is commented out.
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            pytest.fail(f"non-commented setting line in --full output: {line!r}")


def test_render_env_example_full_secret_defaults_are_bare():
    text = config_cli.render_env_example(full=True)
    assert "# LLM_API_KEY=" in text
    assert "# LLM_API_KEY=None" not in text


# ---------------------------------------------------------------------------
# run_config_command: show
# ---------------------------------------------------------------------------


def _args(**kwargs):
    return argparse.Namespace(**kwargs)


def test_show_default_filters_to_non_default_only(env_chain, monkeypatch, capsys):
    cwd_env, _home_env = env_chain
    _clean_env(monkeypatch, "CROSSREF_API_EMAIL", "CROSSREF_RATE_LIMIT_RPM")
    cwd_env.write_text("CROSSREF_API_EMAIL=someone@example.com\n")

    exit_code = config_cli.run_config_command(
        _args(config_command="show", sources=False, all=False)
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "CROSSREF_API_EMAIL" in out
    # A setting at its default with no override anywhere must not appear.
    assert "CROSSREF_RATE_LIMIT_RPM" not in out


def test_show_all_includes_defaults(env_chain, monkeypatch, capsys):
    _clean_env(monkeypatch, "CROSSREF_API_EMAIL", "CROSSREF_RATE_LIMIT_RPM")

    exit_code = config_cli.run_config_command(_args(config_command="show", sources=False, all=True))
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "CROSSREF_RATE_LIMIT_RPM" in out
    assert "CROSSREF_API_EMAIL" in out


def test_show_sources_reports_env_vs_dotenv_vs_default(env_chain, monkeypatch, capsys):
    cwd_env, _home_env = env_chain
    _clean_env(monkeypatch, "CROSSREF_API_EMAIL")
    cwd_env.write_text("CROSSREF_API_EMAIL=someone@example.com\n")

    exit_code = config_cli.run_config_command(_args(config_command="show", sources=True, all=False))
    out = capsys.readouterr().out
    assert exit_code == 0
    assert str(cwd_env) in out


def test_show_never_leaks_full_secret_value(env_chain, monkeypatch, capsys):
    _clean_env(monkeypatch, "LLM_API_KEY")
    monkeypatch.setenv("LLM_API_KEY", "sk-1234567890abcdef")

    exit_code = config_cli.run_config_command(_args(config_command="show", sources=True, all=True))
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "sk-1234567890abcdef" not in out
    assert "sk-1" in out  # masked prefix still visible


# ---------------------------------------------------------------------------
# run_config_command: path
# ---------------------------------------------------------------------------


def test_path_reports_both_states_when_neither_exists(env_chain, capsys):
    cwd_env, home_env = env_chain
    exit_code = config_cli.run_config_command(_args(config_command="path"))
    out = capsys.readouterr().out
    assert exit_code == 0
    assert str(cwd_env) in out
    assert str(home_env) in out
    assert "none found" in out
    assert "bibr config set" in out


def test_path_reports_existing_file(env_chain, capsys):
    cwd_env, home_env = env_chain
    cwd_env.write_text("FOO=bar\n")
    exit_code = config_cli.run_config_command(_args(config_command="path"))
    out = capsys.readouterr().out
    assert exit_code == 0
    assert str(cwd_env) in out
    assert str(home_env) in out
    assert "none found" not in out


# ---------------------------------------------------------------------------
# run_config_command: set
# ---------------------------------------------------------------------------


def test_set_round_trip_preserves_unrelated_lines(env_chain, capsys):
    cwd_env, _home_env = env_chain
    cwd_env.write_text("# a comment\nFOO=bar\n\n# another section\nBAZ=qux\n")

    exit_code = config_cli.run_config_command(
        _args(config_command="set", key="crossref_api_email", value="me@example.com")
    )
    assert exit_code == 0

    text = cwd_env.read_text()
    assert "# a comment" in text
    assert "FOO=bar" in text
    assert "# another section" in text
    assert "BAZ=qux" in text
    assert "CROSSREF_API_EMAIL" in text

    from dotenv import dotenv_values

    values = dotenv_values(cwd_env)
    assert values["CROSSREF_API_EMAIL"] == "me@example.com"
    assert values["FOO"] == "bar"
    assert values["BAZ"] == "qux"


def test_set_writes_to_home_env_when_cwd_env_absent(env_chain):
    cwd_env, home_env = env_chain
    home_env.write_text("EXISTING=1\n")

    exit_code = config_cli.run_config_command(
        _args(config_command="set", key="CROSSREF_API_EMAIL", value="me@example.com")
    )
    assert exit_code == 0
    assert not cwd_env.exists()

    from dotenv import dotenv_values

    values = dotenv_values(home_env)
    assert values["CROSSREF_API_EMAIL"] == "me@example.com"
    assert values["EXISTING"] == "1"


def test_set_creates_cwd_env_when_neither_exists(env_chain):
    cwd_env, home_env = env_chain

    exit_code = config_cli.run_config_command(
        _args(config_command="set", key="CROSSREF_API_EMAIL", value="me@example.com")
    )
    assert exit_code == 0
    assert cwd_env.exists()
    assert not home_env.exists()


def test_set_unknown_key_suggests_close_match(env_chain, capsys):
    exit_code = config_cli.run_config_command(
        _args(config_command="set", key="CROSSREF_API_EMIAL", value="me@example.com")
    )
    out = capsys.readouterr().out
    assert exit_code == 2
    assert "did you mean" in out.lower()
    assert "CROSSREF_API_EMAIL" in out


def test_set_unknown_key_with_no_close_match_has_no_suggestion(env_chain, capsys):
    exit_code = config_cli.run_config_command(
        _args(config_command="set", key="TOTALLY_MADE_UP_XYZZY_PLUGH", value="1")
    )
    out = capsys.readouterr().out
    assert exit_code == 2
    assert "did you mean" not in out.lower()


def test_set_literal_validation_error_lists_choices(env_chain, capsys):
    exit_code = config_cli.run_config_command(
        _args(config_command="set", key="REF_SEG_STRATEGY", value="bogus")
    )
    out = capsys.readouterr().out
    assert exit_code == 2
    for choice in ("llm", "crf", "geom", "region"):
        assert choice in out


def test_set_type_validation_int_field_rejects_non_numeric(env_chain, capsys):
    exit_code = config_cli.run_config_command(
        _args(config_command="set", key="CROSSREF_RATE_LIMIT_RPM", value="abc")
    )
    out = capsys.readouterr().out
    assert exit_code == 2
    assert "integer" in out.lower()


def test_set_case_insensitive_key(env_chain):
    cwd_env, _home_env = env_chain
    exit_code = config_cli.run_config_command(
        _args(config_command="set", key="crossref_api_email", value="me@example.com")
    )
    assert exit_code == 0
    from dotenv import dotenv_values

    assert dotenv_values(cwd_env)["CROSSREF_API_EMAIL"] == "me@example.com"


def test_set_confirmation_redacts_secret_value(env_chain, capsys):
    exit_code = config_cli.run_config_command(
        _args(config_command="set", key="LLM_API_KEY", value="sk-1234567890abcdef")
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "sk-1234567890abcdef" not in out
    assert "sk-1" in out


# ---------------------------------------------------------------------------
# run_config_command: example
# ---------------------------------------------------------------------------


def test_example_command_minimal(capsys):
    exit_code = config_cli.run_config_command(_args(config_command="example", full=False))
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "LLM_PROVIDER" in out
    assert "GOOGLE_API_KEY" in out
    assert "CROSSREF_API_EMAIL" in out
    assert "OCR_BACKEND" in out


def test_example_command_full(capsys):
    exit_code = config_cli.run_config_command(_args(config_command="example", full=True))
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "OCR_BACKEND" in out


# ---------------------------------------------------------------------------
# run_config_command: no subcommand
# ---------------------------------------------------------------------------


def test_no_subcommand_prints_parser_help_and_exits_nonzero():
    from bibr.local.cli import _build_parser

    parser = _build_parser()
    config_parser = None
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            config_parser = action.choices.get("config")
            break
    assert config_parser is not None

    exit_code = config_cli.run_config_command(_args(config_command=None), parser=config_parser)
    assert exit_code != 0
