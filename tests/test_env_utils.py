import os
import stat

import pytest

from bibr.env_utils import merge_env, parse_env
from bibr.setup_wizard import _write_env_fresh


def test_parse_env(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\nLLM_PROVIDER=google\nLLM_MODEL=gemini\n\nEMPTY=\n",
        encoding="utf-8",
    )
    result = parse_env(env_path)
    assert result == {"LLM_PROVIDER": "google", "LLM_MODEL": "gemini", "EMPTY": ""}


def test_merge_env_preserves_and_overwrites(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# header\nOLD_KEY=old\nLLM_MODEL=old-model\n",
        encoding="utf-8",
    )
    merge_env(env_path, {"LLM_MODEL": "new-model", "NEW_KEY": "new"})
    content = env_path.read_text()
    assert "OLD_KEY=old" in content
    assert "LLM_MODEL=new-model" in content
    assert "LLM_MODEL=old-model" not in content
    assert "NEW_KEY=new" in content
    assert "# header" in content


def test_merge_env_preserves_section_structure(tmp_path):
    """Comments stay attached to the variables they annotate across re-runs."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# === LLM ===\n"
        "LLM_PROVIDER=google\n"
        "LLM_MODEL=gemini\n"
        "\n"
        "# === OCR ===\n"
        "OCR_BACKEND=glm-llama\n",
        encoding="utf-8",
    )
    merge_env(env_path, {"LLM_MODEL": "gemini-3"})
    lines = env_path.read_text().splitlines()
    # The LLM section header still immediately precedes its keys, and OCR
    # remains a distinct block — not collapsed into a leading comment chunk.
    llm_header = lines.index("# === LLM ===")
    ocr_header = lines.index("# === OCR ===")
    assert lines[llm_header + 1].startswith("LLM_PROVIDER=")
    assert lines[llm_header + 2] == "LLM_MODEL=gemini-3"
    assert lines[ocr_header + 1].startswith("OCR_BACKEND=")


def test_merge_env_quotes_special_characters(tmp_path):
    """Values with ``#`` or whitespace are quoted so dotenv readers don't
    truncate them at the inline-comment marker."""
    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING=plain\n", encoding="utf-8")
    merge_env(env_path, {"REDIS_PASSWORD": "pa#ss word", "CLEAN": "abc"})
    content = env_path.read_text()
    assert 'REDIS_PASSWORD="pa#ss word"' in content
    assert "CLEAN=abc" in content  # no needless quoting


def test_merge_env_round_trip_through_parse(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("X=1\n", encoding="utf-8")
    pw = 'p#a"s\\s'
    merge_env(env_path, {"REDIS_PASSWORD": pw})
    assert parse_env(env_path)["REDIS_PASSWORD"] == pw


# --- file mode: a .env holds API keys (x-security-8) -------------------------

_POSIX_MODES = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")


@_POSIX_MODES
def test_wizard_writes_a_new_env_owner_only_whatever_the_umask(tmp_path):
    env_path = tmp_path / ".env"
    previous = os.umask(0o022)
    try:
        _write_env_fresh(env_path, {"GEMINI_API_KEY": "gemini-key-placeholder"})
    finally:
        os.umask(previous)
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    assert "GEMINI_API_KEY=gemini-key-placeholder" in env_path.read_text(encoding="utf-8")


@_POSIX_MODES
def test_rewrites_keep_the_existing_mode_and_leave_no_temp_file(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("A=1\n", encoding="utf-8")
    os.chmod(env_path, 0o640)
    merge_env(env_path, {"B": "2"})
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o640
    _write_env_fresh(env_path, {"C": "3"})
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


@_POSIX_MODES
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
def test_an_existing_env_is_rewritten_in_place(tmp_path):
    """A writable .env in a directory the user cannot write (or a single-file
    bind mount) must stay writable, and a hard link must see the update."""
    project = tmp_path / "project"
    project.mkdir()
    env_path = project / ".env"
    env_path.write_text("A=1\n", encoding="utf-8")
    other_name = tmp_path / "linked.env"
    os.link(env_path, other_name)
    os.chmod(project, 0o500)
    try:
        merge_env(env_path, {"B": "2"})
    finally:
        os.chmod(project, 0o700)
    assert parse_env(other_name) == {"A": "1", "B": "2"}


@_POSIX_MODES
def test_a_symlinked_env_is_rewritten_through_the_link(tmp_path):
    from bibr.env_utils import write_env_text

    target = tmp_path / "shared.env"
    target.write_text("A=1\n", encoding="utf-8")
    link = tmp_path / ".env"
    link.symlink_to(target)
    write_env_text(link, "A=2\n")
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "A=2\n"
