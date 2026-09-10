"""Introspection over the pydantic-settings model (docs generation + `bibr config`)."""

from bibr.config_introspect import SettingDoc, iter_setting_docs


def _by_env(docs: list[SettingDoc]) -> dict[str, SettingDoc]:
    return {d.env_name: d for d in docs}


def test_iter_covers_sections_and_top_level():
    docs = _by_env(iter_setting_docs())
    assert "LLM_PROVIDER" in docs
    assert "OCR_BACKEND" in docs
    assert "CROSSREF_API_EMAIL" in docs
    # Top-level (unprefixed) GlobalSettings fields
    assert "ENVIRONMENT" in docs
    assert "REF_SEG_STRATEGY" in docs
    # Section attachments themselves (llm, ocr, ...) must NOT appear as settings
    assert "LLM" not in docs and "OCR" not in docs


def test_env_name_uses_section_prefix():
    docs = _by_env(iter_setting_docs())
    assert docs["LLM_PROVIDER"].section == "LLM_"
    assert docs["LLM_PROVIDER"].field_name == "provider"


def test_validation_alias_wins_over_prefix():
    docs = _by_env(iter_setting_docs())
    # GOOGLE_API_KEY is declared with AliasChoices(GOOGLE_API_KEY, GEMINI_API_KEY, ...)
    d = docs["GOOGLE_API_KEY"]
    assert "GEMINI_API_KEY" in d.aliases


def test_defaults_rendered():
    docs = _by_env(iter_setting_docs())
    assert docs["LLM_PROVIDER"].default_repr == '"google"'
    assert docs["LLM_RATE_LIMIT_RPM"].default_repr == "60"
    # default_factory fields render as computed, not a live value
    assert docs["CACHE_VERSION"].default_repr == "(computed)"


def test_secret_detection():
    docs = _by_env(iter_setting_docs())
    assert docs["GOOGLE_API_KEY"].is_secret
    assert docs["OCR_API_KEY"].is_secret
    assert not docs["LLM_PROVIDER"].is_secret


def test_max_tokens_is_not_a_secret():
    """_SECRET_RE used to match the bare substring "TOKEN", so every

    *_MAX_TOKENS tuning knob (LLM_MAX_TOKENS, OCR_VISION_MAX_TOKENS,
    REF_PARSE_MAX_TOKENS) was wrongly classified as a credential.
    """
    docs = _by_env(iter_setting_docs())
    assert not docs["LLM_MAX_TOKENS"].is_secret
    assert not docs["OCR_VISION_MAX_TOKENS"].is_secret
    assert not docs["REF_PARSE_MAX_TOKENS"].is_secret
    for name in (
        "LLM_TITLE_MAX_TOKENS",
        "LLM_AUTHORS_MAX_TOKENS",
        "LLM_PAPER_CLASSIFICATION_MAX_TOKENS",
        "LLM_PAPER_TYPE_MAX_TOKENS",
        "LLM_SECTION_MAX_TOKENS",
        "LLM_INTEGRITY_MAX_TOKENS",
        "LLM_EQUATION_MAX_TOKENS",
        "LLM_CITATION_MAX_TOKENS",
    ):
        assert not docs[name].is_secret


def test_no_secret_leaks_a_live_default_into_docs():
    """Gate: any setting flagged ``is_secret`` must never render an actual

    default value in generated docs — only a safe placeholder. Every
    credential field in the model today defaults to ``None`` (rendered as
    the bare word "None"); if a future secret field ever shipped a
    non-placeholder default, this test must fail loudly rather than let it
    leak into the published settings reference.
    """
    docs = iter_setting_docs()
    secret_docs = [d for d in docs if d.is_secret]
    assert secret_docs, "expected at least one secret-flagged setting"
    safe_default_reprs = {"None", "''", '""'}
    for d in secret_docs:
        assert d.default_repr in safe_default_reprs, (
            f"{d.env_name} is_secret=True but default_repr={d.default_repr!r} "
            "looks like it could leak a real value into generated docs"
        )
