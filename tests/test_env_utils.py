from bibr.env_utils import merge_env, parse_env


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
