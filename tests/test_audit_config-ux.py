"""Config tools, presets, setup, doctor and the demo (audit group config-ux).

The configuration tools must read and write the ``.env`` the runtime reads,
never print a credential, and the demo must not run open to the network by
accident.
"""

from __future__ import annotations

import argparse
import copy
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bibr import config_cli
from bibr.config_introspect import iter_setting_docs

_DOCS_BY_NAME = {d.env_name: d for d in iter_setting_docs()}

# Windows reports 0o666/0o777 modes, and creating a symlink needs a privilege.
_POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits and symlinks")


@pytest.fixture()
def env_chain(tmp_path, monkeypatch):
    """The default ``./.env`` + ``~/.bibr/.env`` chain, with an isolated CWD and home."""
    cwd_env = tmp_path / "cwd" / ".env"
    home_env = tmp_path / "home" / ".bibr" / ".env"
    cwd_env.parent.mkdir(parents=True)
    home_env.parent.mkdir(parents=True)
    monkeypatch.chdir(cwd_env.parent)
    monkeypatch.setattr(Path, "home", lambda: home_env.parent.parent)
    # conftest pins BIBR_ENV_FILE="" suite-wide; these tests need the chain.
    monkeypatch.delenv("BIBR_ENV_FILE", raising=False)
    monkeypatch.delenv("BIBR_DISABLE_DOTENV", raising=False)
    return cwd_env, home_env


def _config(**kwargs) -> int:
    return config_cli.run_config_command(argparse.Namespace(**kwargs))


def _preset(capsys, command: str, **kwargs) -> tuple[int, str]:
    """Run ``bibr preset <command>``; return (exit code, output)."""
    from bibr.local.cli.presets import _run_preset

    args = argparse.Namespace(preset_command=command, force=True, yes=True, **kwargs)
    code = 0
    try:
        _run_preset(args)
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr().out


# ---------------------------------------------------------------------------
# BIBR_DISABLE_DOTENV / empty BIBR_ENV_FILE reach the config tools
# ---------------------------------------------------------------------------


def test_disabled_dotenv_is_not_reported_as_a_source(env_chain, monkeypatch, capsys):
    from bibr.config import LlmOptions

    cwd_env, _ = env_chain
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cwd_env.write_text("LLM_MODEL=from-dotenv\n")
    monkeypatch.setenv("BIBR_DISABLE_DOTENV", "1")

    assert LlmOptions().model != "from-dotenv"
    assert config_cli.resolve_provenance(_DOCS_BY_NAME["LLM_MODEL"]).tier == "default"
    assert _config(config_command="show", sources=True, all=False) == 0
    assert "from-dotenv" not in capsys.readouterr().out

    assert _config(config_command="path") == 0
    out = capsys.readouterr().out
    assert "disabled by BIBR_DISABLE_DOTENV" in out
    assert str(cwd_env) not in out


def test_config_set_refuses_while_dotenv_is_disabled(env_chain, monkeypatch, capsys):
    cwd_env, _ = env_chain
    cwd_env.write_text("LLM_MODEL=from-dotenv\n")
    monkeypatch.setenv("BIBR_DISABLE_DOTENV", "1")

    assert _config(config_command="set", key="LLM_MODEL", value="unused") == 2

    out = capsys.readouterr().out
    assert "BIBR_DISABLE_DOTENV" in out
    assert "✓" not in out
    assert cwd_env.read_text() == "LLM_MODEL=from-dotenv\n"


@pytest.mark.parametrize(
    ("variable", "value"), [("BIBR_DISABLE_DOTENV", "1"), ("BIBR_ENV_FILE", "")]
)
@pytest.mark.parametrize("command", ["use", "save", "deactivate", "diff"])
def test_preset_commands_refuse_while_dotenv_is_disabled(
    env_chain, monkeypatch, tmp_path, capsys, command, variable, value
):
    from bibr.presets import PresetManager

    cwd_env, _ = env_chain
    original = "LLM_MODEL=from-dotenv\nBIBR_ACTIVE_PRESET=old\n"
    cwd_env.write_text(original)
    presets = PresetManager(tmp_path / "presets")
    presets.save("fast", {"LLM_MODEL": "preset-model"})
    monkeypatch.setenv("BIBR_PRESETS_DIR", str(presets.directory))
    monkeypatch.setenv(variable, value)

    code, out = _preset(capsys, command, name="other" if command == "save" else "fast")

    assert code == 1
    assert f"disabled by {variable}" in out
    assert cwd_env.read_text() == original
    assert presets.list_presets() == ["fast"]


def test_effective_env_file_is_none_while_dotenv_is_disabled(env_chain, monkeypatch):
    from bibr.presets import effective_env_file

    cwd_env, _ = env_chain
    cwd_env.write_text("LLM_MODEL=x\n")
    assert effective_env_file() == cwd_env

    monkeypatch.setenv("BIBR_DISABLE_DOTENV", "1")
    assert effective_env_file() is None


def test_setup_refuses_while_dotenv_is_disabled(env_chain, monkeypatch, capsys):
    from bibr.setup_wizard import main

    ran = []
    monkeypatch.setattr("bibr.setup_wizard.SetupWizard.run", lambda self: ran.append(self))
    monkeypatch.setattr("bibr.setup_wizard.sys.argv", ["bibr"])
    monkeypatch.setenv("BIBR_DISABLE_DOTENV", "1")

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    assert "BIBR_DISABLE_DOTENV" in capsys.readouterr().out
    assert ran == []
    assert not env_chain[0].exists()


# ---------------------------------------------------------------------------
# bibr setup writes the .env bibr reads, and names exported settings
# ---------------------------------------------------------------------------


def _wizard():
    from rich.console import Console

    from bibr.setup_wizard import SetupWizard

    wizard = SetupWizard()
    wizard.console = Console(record=True, width=200)
    return wizard


def test_setup_writes_the_last_file_bibr_env_file_lists(tmp_path, monkeypatch):
    from bibr.config import LlmOptions

    first, last = tmp_path / "first.env", tmp_path / "last.env"
    monkeypatch.setenv("BIBR_ENV_FILE", os.pathsep.join(map(str, (first, last))))
    monkeypatch.delenv("LLM_MODEL", raising=False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    wizard = _wizard()
    wizard.env_vars = {"LLM_MODEL": "wizard-model"}
    with patch("bibr.setup_wizard.Confirm.ask", return_value=False):
        wizard._step_write_env()

    assert wizard.env_path == last
    assert "LLM_MODEL=wizard-model" in last.read_text()
    assert not (elsewhere / ".env").exists()
    # What the reload and smoke test then read is what the wizard wrote.
    assert LlmOptions().model == "wizard-model"


def test_setup_still_writes_cwd_env_by_default(env_chain):
    cwd_env, _ = env_chain

    assert _wizard().env_path == cwd_env


def _exported_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "shell exports" in line]


def test_setup_names_settings_the_shell_exports_with_other_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BIBR_ENV_FILE", str(tmp_path / ".env"))
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("LLM_API_KEY", "sk-old-shell-key-123456")
    monkeypatch.setenv("LLM_BACKEND", "cloud")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    wizard = _wizard()
    wizard.env_vars = {
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-5-nano",
        "LLM_API_KEY": "sk-typed-key-1234567890",
    }
    with patch("bibr.setup_wizard.Confirm.ask", return_value=False):
        wizard._step_write_env()

    text = wizard.console.export_text()
    [line] = _exported_lines(text)
    assert "LLM_API_KEY" in line and "LLM_PROVIDER" in line
    # Same value as exported, and not exported at all: nothing to warn about.
    assert "LLM_BACKEND" not in line and "LLM_MODEL" not in line
    assert "sk-old-shell-key-123456" not in text
    assert "sk-typed-key-1234567890" not in text


def test_connection_test_warns_that_exported_settings_win_at_runtime(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://old-server:8000/v1")
    monkeypatch.setattr("bibr.setup_wizard.Confirm.ask", lambda *a, **k: True)
    monkeypatch.setattr("bibr.clients.llm.ping_llm", lambda settings: "OK")

    wizard = _wizard()
    wizard.env_vars = {
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-5-nano",
        "LLM_API_KEY": "sk-typed-key-1234567890",
        "LLM_BASE_URL": "http://new-server:8000/v1",
    }
    wizard._offer_llm_connection_test()

    [line] = _exported_lines(wizard.console.export_text())
    assert "LLM_BASE_URL" in line


def _google_setup(tmp_path, monkeypatch, **exported: str):
    """Write a google setup with *exported* as the only key spellings in the shell."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BIBR_ENV_FILE", str(tmp_path / ".env"))
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "LANGEXTRACT_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in exported.items():
        monkeypatch.setenv(name, value)

    wizard = _wizard()
    wizard.env_vars = {"LLM_PROVIDER": "google", "GOOGLE_API_KEY": "AIzaTYPEDtypedTYPED12345"}
    with patch("bibr.setup_wizard.Confirm.ask", return_value=False):
        wizard._step_write_env()
    return wizard.console.export_text()


@pytest.mark.parametrize("alias", ["GEMINI_API_KEY", "LANGEXTRACT_API_KEY"])
def test_setup_names_an_exported_alias_of_a_setting_it_writes(tmp_path, monkeypatch, alias):
    from bibr.config import GlobalSettings

    text = _google_setup(tmp_path, monkeypatch, **{alias: "AIzaSHELLshellSHELL12345"})

    # The settings read the exported alias over GOOGLE_API_KEY in .env.
    assert GlobalSettings().GOOGLE_API_KEY == "AIzaSHELLshellSHELL12345"
    [line] = _exported_lines(text)
    assert alias in line and "GOOGLE_API_KEY" not in line
    assert "AIzaSHELL" not in text


def test_setup_ignores_an_alias_the_exported_setting_outranks(tmp_path, monkeypatch):
    from bibr.config import GlobalSettings

    text = _google_setup(
        tmp_path,
        monkeypatch,
        GOOGLE_API_KEY="AIzaTYPEDtypedTYPED12345",
        GEMINI_API_KEY="AIzaSHELLshellSHELL12345",
    )

    assert GlobalSettings().GOOGLE_API_KEY == "AIzaTYPEDtypedTYPED12345"
    assert _exported_lines(text) == []


def test_setup_refuses_a_bibr_env_file_in_a_missing_folder(tmp_path, monkeypatch, capsys):
    from bibr.setup_wizard import main

    ran = []
    monkeypatch.setattr("bibr.setup_wizard.SetupWizard.run", lambda self: ran.append(self))
    monkeypatch.setattr("bibr.setup_wizard.sys.argv", ["bibr"])
    monkeypatch.delenv("BIBR_DISABLE_DOTENV", raising=False)
    monkeypatch.setenv("BIBR_ENV_FILE", str(tmp_path / "missing" / "app.env"))

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    assert "BIBR_ENV_FILE" in capsys.readouterr().out
    # Refused before the interview, not after it.
    assert ran == []


# ---------------------------------------------------------------------------
# The setup wizard masks every configured secret
# ---------------------------------------------------------------------------


def test_smoke_test_failure_masks_every_configured_secret(monkeypatch):
    google_key = "AIzaFAKEFAKEFAKEFAKEFAKE1234"
    ocr_key = "ocr-secret-0123456789abcdef"
    anthropic_key = "sk-ant-typed-key-0123456789abcdef"
    monkeypatch.setenv("GOOGLE_API_KEY", google_key)
    monkeypatch.setenv("OCR_API_KEY", ocr_key)
    monkeypatch.setattr("bibr.setup_wizard._reload_settings_in_place", lambda: None)
    wizard = _wizard()
    wizard.env_vars = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": anthropic_key}
    err = RuntimeError(
        f"OCR at https://ocr.example/v1?key={ocr_key} said 401 for {ocr_key}; "
        f"vision fallback https://g.example/v1/models?key={google_key}; sent {anthropic_key}"
    )
    with (
        patch("bibr.setup_wizard.Confirm.ask", return_value=True),
        patch("bibr.api.chew", side_effect=err),
    ):
        wizard._step_smoke_test()

    text = wizard.console.export_text()
    for secret in (google_key, ocr_key, anthropic_key):
        assert secret not in text
        assert secret[:12] not in text
    assert "***" in text


# ---------------------------------------------------------------------------
# bibr doctor masks URL credentials
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.lines: list[str] = []

    def ok(self, msg: str) -> None:
        self.lines.append(msg)

    def warn(self, msg: str, hint: str = "") -> None:
        self.lines.append(f"{msg} {hint}")

    def fail(self, msg: str, hint: str = "") -> None:
        self.lines.append(f"{msg} {hint}")


_URL_WITH_SECRETS = "https://bob:hunter2pw@ocr.example:8443/v1?key=sekrit-query-key"


@pytest.mark.parametrize("backend", ["paddle-http", "glm-http"])
@pytest.mark.parametrize("reachable", [True, False])
def test_doctor_masks_the_ocr_url_credentials(monkeypatch, backend, reachable):
    from bibr.config import Settings
    from bibr.local.cli.doctor import _check_ocr_backend

    monkeypatch.setattr(Settings.ocr, "backend", backend)
    monkeypatch.setattr(Settings, "OCR_BASE_URL", _URL_WITH_SECRETS, raising=False)
    monkeypatch.setattr("bibr.local.cli.doctor._probe_ocr_url", lambda url: reachable)
    rec = _Recorder()

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    [line] = rec.lines
    assert "ocr.example:8443" in line
    assert "hunter2pw" not in line and "sekrit-query-key" not in line


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_doctor_llm_hint_masks_the_server_url_credentials(provider):
    from bibr.local.cli.doctor import _llm_connection_hint

    llm = SimpleNamespace(
        provider=provider,
        model="m",
        ollama_base_url=_URL_WITH_SECRETS,
        base_url=_URL_WITH_SECRETS,
    )

    hint = _llm_connection_hint(SimpleNamespace(llm=llm))

    assert "ocr.example:8443" in hint
    assert "hunter2pw" not in hint and "sekrit-query-key" not in hint


def test_doctor_masks_the_redis_url_query_password(monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli.doctor import _check_redis

    monkeypatch.setattr(Settings.redis, "url", "rediss://cache.example:6380/0?password=hunter2pw")
    monkeypatch.setattr(Settings.redis, "password", "hunter2pw")
    rec = _Recorder()

    _check_redis(rec.ok, rec.warn)

    [line] = rec.lines
    assert "cache.example:6380" in line and "hunter2pw" not in line


# ---------------------------------------------------------------------------
# Presets: URL credentials, file mode, endpoint changes, markers, directory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value", "hidden"),
    [
        ("REDIS_URL", "redis://:hunter2pw@redis:6379/0", "hunter2pw"),
        ("JOBS_REDIS_URL", "redis://jobs:hunter2pw@redis:6379/1", "hunter2pw"),
        ("CROSSREF_CACHE_REDIS_URL", "rediss://u:hunter2pw@cache:6380", "hunter2pw"),
        ("LLM_BASE_URL", "https://llm.example/v1?key=sekrit-query-key", "sekrit-query-key"),
    ],
)
def test_preset_display_and_snapshot_keep_url_credentials_out(tmp_path, name, value, hidden):
    from bibr.presets import PresetManager, redact_value

    assert hidden not in redact_value(name, value)

    env_path = tmp_path / ".env"
    env_path.write_text(f"LLM_MODEL=gemini\n{name}={value}\n", encoding="utf-8")
    assert PresetManager(tmp_path / "presets").snapshot_from_env(env_path) == {
        "LLM_MODEL": "gemini"
    }


def test_preset_snapshot_keeps_urls_without_credentials(tmp_path):
    from bibr.presets import PresetManager, redact_value

    env_path = tmp_path / ".env"
    env_path.write_text("REDIS_URL=redis://redis:6379/0\n", encoding="utf-8")

    assert PresetManager(tmp_path).snapshot_from_env(env_path) == {
        "REDIS_URL": "redis://redis:6379/0"
    }
    assert redact_value("REDIS_URL", "redis://redis:6379/0") == "redis://redis:6379/0"


def test_preset_cli_masks_url_credentials_and_names_what_save_left_out(
    tmp_path, monkeypatch, capsys
):
    from bibr.presets import PresetManager

    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_MODEL=gemini\nREDIS_URL=redis://:hunter2pw@redis:6379/0\n", encoding="utf-8"
    )
    presets = PresetManager(tmp_path / "presets")
    presets.save("shared", {"LLM_BASE_URL": "https://h.example/v1?key=sekrit-query-key"})
    monkeypatch.setenv("BIBR_ENV_FILE", str(env_path))
    monkeypatch.setenv("BIBR_PRESETS_DIR", str(presets.directory))

    code, out = _preset(capsys, "save", name="mine")
    assert code == 0
    assert "Left out REDIS_URL" in out
    assert "hunter2pw" not in presets.directory.joinpath("mine.json").read_text()

    for command in ("show", "diff"):
        code, out = _preset(capsys, command, name="shared")
        assert code == 0
        assert "LLM_BASE_URL" in out
        assert "sekrit-query-key" not in out and "hunter2pw" not in out


@_POSIX
def test_preset_files_are_owner_only(tmp_path):
    from bibr.presets import PresetManager

    old_umask = os.umask(0o022)
    try:
        path = PresetManager(tmp_path / "presets").save("p", {"LLM_MODEL": "m"})
    finally:
        os.umask(old_umask)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_preset_use_names_the_endpoint_settings_it_changes(tmp_path, monkeypatch, capsys):
    from bibr.presets import PresetManager

    env_path = tmp_path / ".env"
    env_path.write_text("LLM_MODEL=old\nLLM_BASE_URL=https://mine.example/v1\n")
    presets = PresetManager(tmp_path / "presets")
    presets.save(
        "shared",
        {
            "LLM_MODEL": "new",
            "LLM_BASE_URL": "https://theirs.example/v1",
            "LLM_LLAMA_CPP_EXTRA_ARGS": "--verbose",
        },
    )
    presets.save("same", {"LLM_MODEL": "newer", "LLM_BASE_URL": "https://theirs.example/v1"})
    monkeypatch.setenv("BIBR_ENV_FILE", str(env_path))
    monkeypatch.setenv("BIBR_PRESETS_DIR", str(presets.directory))

    code, out = _preset(capsys, "use", name="shared")
    assert code == 0
    assert "changed LLM_BASE_URL, LLM_LLAMA_CPP_EXTRA_ARGS" in " ".join(out.split())

    # The endpoint is already what the preset says: no notice.
    code, out = _preset(capsys, "use", name="same")
    assert code == 0
    assert "changed" not in out


def test_endpoint_changes_cover_urls_server_args_and_executables():
    from bibr.presets import endpoint_changes

    preset = {
        "OCR_BASE_URL": "http://elsewhere:8002",
        "RAPID_MLX_EXECUTABLE": "/tmp/x",
        "LLM_LLMSTER_LOAD_ARGS": "--gpu max",
        "LLM_MODEL": "m",
        "LLM_MAX_TOKENS": "100",
        "BIBR_ACTIVE_PRESET": "p",
    }

    assert endpoint_changes(preset, {"OCR_BASE_URL": "http://elsewhere:8002"}) == [
        "LLM_LLMSTER_LOAD_ARGS",
        "RAPID_MLX_EXECUTABLE",
    ]


def test_active_preset_marker_with_export_prefix(tmp_path):
    from bibr.presets import PresetManager

    manager = PresetManager(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_MODEL=m\nexport BIBR_ACTIVE_PRESET=fast\n", encoding="utf-8")

    assert manager.get_active(env_path) == "fast"
    assert manager.deactivate(env_path) is True
    assert env_path.read_text(encoding="utf-8") == "LLM_MODEL=m\n"
    assert manager.get_active(env_path) is None


def test_preset_manager_default_honours_bibr_presets_dir(tmp_path, monkeypatch):
    from bibr import presets
    from bibr.presets import PresetManager

    monkeypatch.setenv("BIBR_PRESETS_DIR", str(tmp_path / "presets"))
    assert PresetManager().directory == tmp_path / "presets"

    # A blank value keeps the default instead of the cwd.
    monkeypatch.setenv("BIBR_PRESETS_DIR", " ")
    assert PresetManager().directory == presets._DEFAULT_DIR


def test_setup_saves_presets_into_bibr_presets_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BIBR_PRESETS_DIR", str(tmp_path / "presets"))
    wizard = _wizard()
    wizard.env_vars = {
        "LLM_MODEL": "m",
        "GOOGLE_API_KEY": "AIzaFAKEFAKEFAKEFAKEFAKE1234",
        "OCR_BASE_URL": "https://u:hunter2pw@ocr.example",
    }

    wizard._save_preset("from-setup")

    saved = tmp_path / "presets" / "from-setup.json"
    assert saved.is_file()
    text = saved.read_text()
    assert '"LLM_MODEL": "m"' in text
    assert "AIza" not in text and "hunter2pw" not in text
    # Like ``bibr preset save``, it names the URL it left out.
    [note] = [line for line in wizard.console.export_text().splitlines() if "Left out" in line]
    assert "OCR_BASE_URL" in note and "GOOGLE_API_KEY" not in note


# ---------------------------------------------------------------------------
# bibr config set writes through a symlinked .env
# ---------------------------------------------------------------------------


@_POSIX
def test_config_set_writes_through_a_symlinked_env(env_chain, monkeypatch, tmp_path):
    from bibr.config import LlmOptions

    cwd_env, _ = env_chain
    real = tmp_path / "dotfiles" / "bibr.env"
    real.parent.mkdir()
    real.write_text("LLM_PROVIDER=google\n")
    cwd_env.symlink_to(real)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert _config(config_command="set", key="LLM_MODEL", value="linked-model") == 0

    assert cwd_env.is_symlink()
    assert "LLM_MODEL" in real.read_text() and "linked-model" in real.read_text()
    assert LlmOptions().model == "linked-model"


# ---------------------------------------------------------------------------
# bibr demo: authentication, flags and the "(current .env)" preset
# ---------------------------------------------------------------------------


@pytest.fixture()
def demo_modules(monkeypatch):
    gr = pytest.importorskip("gradio")
    if not hasattr(gr, "Blocks"):
        pytest.skip("gradio not fully installed")
    import bibr.demo.local_app as local_app
    import bibr.demo.server as server

    for name in ("GRADIO_PASSWORD", "DEMO_CACHE_TTL_SECONDS", "DEMO_MAX_FILE_SIZE_MB"):
        monkeypatch.delenv(name, raising=False)
    return local_app, server


def _run_demo(monkeypatch, local_app, server, *argv):
    """Run ``bibr demo``; return the launch kwargs, or None when it never launched."""
    launched: dict = {}

    class _Demo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    monkeypatch.setattr(local_app, "create_local_demo", lambda **_: _Demo())
    monkeypatch.setattr(sys, "argv", ["bibr demo", *argv])
    server.main()
    return launched


_ALL_INTERFACES = "0.0.0.0"  # noqa: S104 - the bind under test


@pytest.mark.parametrize(
    "argv", [["--share"], ["--host", _ALL_INTERFACES], ["--host", "::"], ["--host", "192.168.1.5"]]
)
def test_demo_refuses_to_run_exposed_without_a_password(demo_modules, monkeypatch, capsys, argv):
    local_app, server = demo_modules

    with pytest.raises(SystemExit) as exc:
        _run_demo(monkeypatch, local_app, server, *argv)

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "GRADIO_PASSWORD" in err and "--allow-unauthenticated" in err


def test_demo_runs_exposed_with_a_password_or_explicit_opt_out(demo_modules, monkeypatch):
    local_app, server = demo_modules

    launched = _run_demo(monkeypatch, local_app, server, "--share", "--allow-unauthenticated")
    assert launched["share"] is True
    assert "auth" not in launched

    monkeypatch.setenv("GRADIO_PASSWORD", "long-random-password")
    launched = _run_demo(monkeypatch, local_app, server, "--host", _ALL_INTERFACES)
    assert launched["server_name"] == _ALL_INTERFACES
    assert launched["auth"] == ("demo", "long-random-password")


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_demo_on_loopback_needs_no_password(demo_modules, monkeypatch, host):
    local_app, server = demo_modules

    assert _run_demo(monkeypatch, local_app, server, "--host", host)["server_name"] == host


def test_demo_rejects_an_unknown_log_level(demo_modules, capsys):
    _, server = demo_modules

    with pytest.raises(SystemExit) as exc:
        server.build_parser().parse_args(["--log-level", "verbose"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert server.build_parser().parse_args(["--log-level", "DEBUG"]).log_level == "debug"


@pytest.mark.parametrize("level", ["WARN", "fatal", "notset"])
def test_demo_keeps_accepting_the_logging_aliases(demo_modules, monkeypatch, level):
    local_app, server = demo_modules

    assert _run_demo(monkeypatch, local_app, server, "--log-level", level)


def test_demo_upload_limit_follows_env_at_start(demo_modules, monkeypatch):
    local_app, server = demo_modules
    monkeypatch.setenv("DEMO_MAX_FILE_SIZE_MB", "25")

    assert _run_demo(monkeypatch, local_app, server)["max_file_size"] == "25mb"


_BAD_UPLOAD_LIMITS = ["10MB", "", "0", "-5", "1.5"]


@pytest.mark.parametrize("value", _BAD_UPLOAD_LIMITS)
def test_demo_exits_with_a_message_on_an_invalid_upload_limit(
    demo_modules, monkeypatch, capsys, value
):
    local_app, server = demo_modules
    monkeypatch.setenv("DEMO_MAX_FILE_SIZE_MB", value)

    with pytest.raises(SystemExit) as exc:
        _run_demo(monkeypatch, local_app, server)

    assert exc.value.code == 1
    assert "DEMO_MAX_FILE_SIZE_MB must be a positive whole number" in capsys.readouterr().err


@pytest.mark.parametrize("value", _BAD_UPLOAD_LIMITS)
def test_demo_refuses_an_invalid_upload_limit_before_building_pipeline(
    demo_modules, monkeypatch, value
):
    local_app, _ = demo_modules
    monkeypatch.setenv("DEMO_MAX_FILE_SIZE_MB", value)
    built: list = []
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", lambda **kw: built.append(kw))

    with pytest.raises(ValueError, match="DEMO_MAX_FILE_SIZE_MB"):
        local_app.create_local_demo(ocr_backend="glm-mlx")

    assert built == []


@pytest.fixture()
def restore_settings():
    import bibr.config

    saved = copy.deepcopy(vars(bibr.config.Settings))
    yield bibr.config.Settings
    vars(bibr.config.Settings).clear()
    vars(bibr.config.Settings).update(saved)


async def test_demo_current_env_choice_undoes_the_preset(
    demo_modules, monkeypatch, tmp_path, restore_settings
):
    from bibr.presets import PresetManager

    local_app, _ = demo_modules
    settings = restore_settings
    original_model = settings.llm.model
    presets = PresetManager(tmp_path / "presets")
    presets.save("swap", {"LLM_MODEL": "preset-model"})
    monkeypatch.setenv("BIBR_PRESETS_DIR", str(presets.directory))

    built: list[SimpleNamespace] = []

    def pipeline_factory(**kwargs):
        built.append(SimpleNamespace(aclose=AsyncMock(), llm_backend=kwargs["llm_backend"]))
        return built[-1]

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", pipeline_factory)
    demo = local_app.create_local_demo(ocr_backend="glm-mlx", presets_enabled=True)
    switch = next(f.fn for f in demo.fns.values() if f.fn.__name__ == "_switch_preset")

    status = await switch("swap")
    assert settings.llm.model == "preset-model"
    assert "preset-model" in status
    assert len(built) == 2

    status = await switch("(current .env)")

    # The settings and the pipeline go back to the .env setup, not only the status.
    assert settings.llm.model == original_model
    assert len(built) == 3
    built[1].aclose.assert_awaited_once()
    assert original_model in status and "preset-model" not in status
