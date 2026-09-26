"""Tests for ``bibr.local.cli.resolve_run_config`` and ``chew`` parser round-trips."""

from pathlib import Path

import pytest

from bibr.local.cli import ResolvedRunConfig, _build_parser, resolve_run_config


def _chew_args(*argv: str):
    return _build_parser().parse_args(["chew", *argv])


def _resolve(*argv: str):
    return resolve_run_config(_chew_args(*argv))


class _StubOcrSettings:
    def __init__(self, backend=None):
        self.model_fields_set = {"backend"} if backend else set()
        self.backend = backend


class _StubLlmSettings:
    def __init__(self, backend="cloud"):
        self.backend = backend


class _StubSettings:
    def __init__(self, ocr_backend=None, llm_backend="cloud"):
        self.ocr = _StubOcrSettings(ocr_backend)
        self.llm = _StubLlmSettings(llm_backend)


# ---- parser round-trips ------------------------------------------------------


def test_chew_parser_defaults():
    args = _chew_args("paper.pdf")
    assert args.input == ["paper.pdf"]
    assert args.output is None
    assert args.ocr is None
    assert args.ocr_url is None
    assert args.ocr_model is None
    assert args.ocr_profile is None
    assert args.llm is None  # unset → resolved from LLM_BACKEND (default "cloud")
    assert args.memory is None
    assert args.batch_size == 0
    assert args.pages is None
    assert args.device is None
    assert args.no_crossref is False
    assert args.no_equations is False
    assert args.no_llm is False
    assert args.figure_images is False
    assert args.regions is False
    assert args.compact is False
    assert args.refs is None


def test_chew_parser_accepts_ner_refs_strategy():
    assert _chew_args("paper.pdf", "--refs", "ner").refs == "ner"


def test_help_examples_mention_refs_strategy():
    """--refs must be discoverable from the `bibr`/`bibr chew --help` examples."""
    from bibr.local.cli import _EXAMPLES

    assert "--refs" in _EXAMPLES


def test_chew_parser_rejects_unknown_ocr_backend():
    with pytest.raises(SystemExit):
        _chew_args("paper.pdf", "--ocr", "tesseract")


def test_chew_parser_rejects_removed_legacy_backend():
    with pytest.raises(SystemExit):
        _chew_args("paper.pdf", "--ocr", "fal" + "con")


def test_chew_parser_rejects_unknown_memory_mode():
    with pytest.raises(SystemExit):
        _chew_args("paper.pdf", "--memory", "yolo")


# ---- OCR backend resolution --------------------------------------------------


def test_resolve_explicit_backends_and_memory():
    config = _resolve(
        "paper.pdf", "--ocr", "glm-llama", "--llm", "vllm-mlx", "--memory", "keep_all"
    )
    assert config.ocr_backend == "glm-llama"
    assert config.llm_backend == "vllm-mlx"
    assert config.memory_mode == "keep_all"


def test_resolve_explicit_paddle_preserves_automatic_selector():
    config = _resolve("paper.pdf", "--ocr", "paddle", "--memory", "balanced")

    assert config.ocr_backend == "paddle"


def test_resolve_glm_alias_uses_platform_default(monkeypatch):
    monkeypatch.setattr("bibr.ocr.registry.sys.platform", "linux")
    monkeypatch.setattr("bibr.ocr.registry.platform.machine", lambda: "x86_64")
    config = _resolve("paper.pdf", "--ocr", "glm", "--memory", "balanced")
    assert config.ocr_backend == "glm-llama"


def test_resolve_ocr_url_implies_glm_http():
    config = _resolve("paper.pdf", "--ocr-url", "http://host:8080", "--memory", "balanced")
    assert config.ocr_backend == "paddle-http"
    assert config.ocr_url == "http://host:8080"


def test_resolve_explicit_glm_http_with_ocr_url_remains_glm():
    config = _resolve(
        "paper.pdf", "--ocr", "glm-http", "--ocr-url", "http://host:8080", "--memory", "balanced"
    )
    assert config.ocr_backend == "glm-http"


def test_resolve_ocr_profile_is_carried_to_run_config():
    config = _resolve(
        "paper.pdf",
        "--ocr",
        "glm-http",
        "--ocr-profile",
        "glm",
        "--memory",
        "balanced",
    )

    assert config.ocr_profile == "glm"


def test_custom_ocr_model_requires_explicit_profile():
    with pytest.raises(ValueError, match="--ocr-profile"):
        _resolve("paper.pdf", "--ocr-model", "private-alias", "--memory", "balanced")


def test_custom_ocr_model_accepts_explicit_profile():
    config = _resolve(
        "paper.pdf",
        "--ocr-model",
        "private-alias",
        "--ocr-profile",
        "paddle",
        "--memory",
        "balanced",
    )

    assert config.ocr_profile == "paddle"


def test_resolve_default_ocr_from_settings(monkeypatch):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.ocr.backend = "glm-llama"
    settings.ocr.model_fields_set.add("backend")
    monkeypatch.setattr("bibr.config.Settings", settings)
    config = _resolve("paper.pdf", "--memory", "balanced")
    assert config.ocr_backend == "glm-llama"


def test_resolve_default_paddle_from_settings_preserves_automatic_selector(monkeypatch):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.ocr.backend = "paddle"
    settings.ocr.model_fields_set.add("backend")
    monkeypatch.setattr("bibr.config.Settings", settings)

    config = _resolve("paper.pdf", "--memory", "balanced")

    assert config.ocr_backend == "paddle"


def test_resolve_unset_default_paddle_preserves_automatic_selector(monkeypatch):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.ocr.model_fields_set.discard("backend")
    monkeypatch.setattr("bibr.config.Settings", settings)
    config = _resolve("paper.pdf", "--memory", "balanced")
    assert config.ocr_backend == "paddle"


# ---- memory mode + chunk size --------------------------------------------------


def test_resolve_default_memory_mode_autodetected(monkeypatch):
    monkeypatch.setattr("bibr.local.pipeline._default_memory_mode", lambda: "aggressive")
    config = _resolve("paper.pdf", "--ocr", "glm-llama")
    assert config.memory_mode == "aggressive"
    assert config.chunk_size == 3


@pytest.mark.parametrize(
    ("memory", "expected"),
    [("aggressive", 3), ("balanced", 8), ("keep_all", 20)],
)
def test_resolve_chunk_size_auto_from_memory_mode(memory, expected):
    config = _resolve("paper.pdf", "--ocr", "glm-llama", "--memory", memory)
    assert config.chunk_size == expected


def test_resolve_chunk_size_explicit_batch_size_wins():
    config = _resolve(
        "paper.pdf", "--ocr", "glm-llama", "--memory", "keep_all", "--batch-size", "5"
    )
    assert config.chunk_size == 5


# ---- page ranges ---------------------------------------------------------------


def test_resolve_pages_range():
    config = _resolve("paper.pdf", "--ocr", "glm-llama", "--memory", "balanced", "--pages", "1-5")
    assert (config.start_page, config.end_page) == (0, 4)


def test_resolve_pages_single():
    config = _resolve("paper.pdf", "--ocr", "glm-llama", "--memory", "balanced", "--pages", "3")
    assert (config.start_page, config.end_page) == (2, 2)


def test_resolve_pages_default_is_full_document():
    config = _resolve("paper.pdf", "--ocr", "glm-llama", "--memory", "balanced")
    assert (config.start_page, config.end_page) == (None, None)


@pytest.mark.parametrize("pages", ["0", "5-2", "7-", "-3", "abc"])
def test_resolve_pages_invalid_raises(pages):
    args = _chew_args("paper.pdf", "--ocr", "glm-llama", "--memory", "balanced", f"--pages={pages}")
    with pytest.raises(ValueError):
        resolve_run_config(args)


# ---- flag derivations ----------------------------------------------------------


def test_resolve_flag_defaults():
    config = _resolve("paper.pdf", "--ocr", "glm-llama", "--memory", "balanced")
    # Tri-state: neither --crossref nor --no-crossref → None (CROSSREF_ENRICH decides).
    assert config.crossref is None
    assert config.equations is True
    assert config.no_llm is False
    assert config.figure_images is None
    assert config.include_regions is False
    assert config.device is None
    assert config.ocr_model is None
    assert config.ocr_url is None


def test_resolve_flag_inversions():
    config = _resolve(
        "paper.pdf",
        "--ocr",
        "glm-llama",
        "--memory",
        "balanced",
        "--no-crossref",
        "--no-equations",
        "--no-llm",
    )
    assert config.crossref is False
    assert config.equations is False
    assert config.no_llm is True


def test_resolve_crossref_flag_forces_on():
    config = _resolve("paper.pdf", "--ocr", "glm-llama", "--memory", "balanced", "--crossref")
    assert config.crossref is True


def test_crossref_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _chew_args("paper.pdf", "--crossref", "--no-crossref")


def test_mcp_parser_accepts_crossref_flag():
    args = _build_parser().parse_args(["mcp", "--crossref"])
    assert args.crossref is True
    assert args.no_crossref is False
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["mcp", "--crossref", "--no-crossref"])


def test_resolve_figure_images_flag():
    config = _resolve("paper.pdf", "--ocr", "glm-llama", "--memory", "balanced", "--figure-images")
    assert config.figure_images is True


def test_resolve_regions_device_and_model_passthrough():
    config = _resolve(
        "paper.pdf",
        "--ocr",
        "glm-llama",
        "--memory",
        "balanced",
        "--regions",
        "--device",
        "cpu",
        "--ocr-model",
        "THUDM/GLM-OCR",
    )
    assert config.include_regions is True
    assert config.device == "cpu"
    assert config.ocr_model == "THUDM/GLM-OCR"


# ---- active stages -------------------------------------------------------------


def _config(**overrides):
    base = {"ocr_backend": "glm-llama", "memory_mode": "balanced", "llm_backend": "cloud"}
    base.update(overrides)
    return ResolvedRunConfig(**base)


def test_active_stages_pdf_default(monkeypatch):
    # The default (crossref=None) follows CROSSREF_ENRICH, which is off unless
    # the deployment opted in — so a plain run has no enrich stage.
    from bibr.config import Settings

    monkeypatch.setattr(Settings.crossref, "enrich", False)
    assert _config().active_stages([Path("paper.pdf")]) == [
        "validate",
        "layout",
        "ocr",
        "parse",
        "extract",
        "export",
    ]
    monkeypatch.setattr(Settings.crossref, "enrich", True)
    assert _config().active_stages([Path("paper.pdf")]) == [
        "validate",
        "layout",
        "ocr",
        "parse",
        "extract",
        "enrich",
        "export",
    ]


def test_active_stages_docx_reads_native_format():
    stages = _config(crossref=True).active_stages([Path("paper.docx")])
    assert stages == ["validate", "docx", "parse", "extract", "enrich", "export"]
    assert "enrich" in stages


def test_active_stages_crossref_flag_overrides_setting(monkeypatch):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.crossref, "enrich", False)
    assert "enrich" in _config(crossref=True).active_stages([Path("paper.pdf")])
    monkeypatch.setattr(Settings.crossref, "enrich", True)
    assert "enrich" not in _config(crossref=False).active_stages([Path("paper.pdf")])


def test_active_stages_no_crossref_drops_enrich():
    assert "enrich" not in _config(crossref=False).active_stages([Path("paper.pdf")])


def test_active_stages_no_llm_drops_enrich():
    assert "enrich" not in _config(no_llm=True, crossref=True).active_stages([Path("paper.pdf")])


def test_active_stages_via_resolve_no_crossref():
    config = _resolve("paper.pdf", "--ocr", "glm-llama", "--memory", "balanced", "--no-crossref")
    assert "enrich" not in config.active_stages([Path("paper.pdf")])


# ---- consolidate flag ----------------------------------------------------------


def test_chew_consolidate_default_none():
    args = _chew_args("paper.pdf")
    assert args.consolidate is None
    assert _resolve("paper.pdf").consolidate is None


def test_chew_consolidate_bare_means_fill():
    assert _resolve("paper.pdf", "--consolidate").consolidate == "fill"


def test_chew_consolidate_replace():
    assert _resolve("paper.pdf", "--consolidate", "replace").consolidate == "replace"


def test_chew_consolidate_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        _chew_args("paper.pdf", "--consolidate", "merge")


class TestPrepareOutputPath:
    """Batch mode always treats -o as a directory, so it must always be
    created — including names with dots (Qwen3.5-...), which the old
    ``not output_path.suffix`` gate misread as file outputs and skipped,
    crashing the result write with FileNotFoundError."""

    def test_batch_dir_with_dotted_name_is_created(self, tmp_path):
        from bibr.local.cli import _prepare_output_path

        target = tmp_path / "Qwen3.5-4B-MLX-4bit"
        out = _prepare_output_path(str(target), is_batch=True)
        assert out == target
        assert target.is_dir()

    def test_batch_plain_dir_is_created(self, tmp_path):
        from bibr.local.cli import _prepare_output_path

        target = tmp_path / "results" / "nested"
        out = _prepare_output_path(str(target), is_batch=True)
        assert out == target
        assert target.is_dir()

    def test_single_mode_does_not_create(self, tmp_path):
        from bibr.local.cli import _prepare_output_path

        target = tmp_path / "out.json"
        out = _prepare_output_path(str(target), is_batch=False)
        assert out == target
        assert not target.exists()

    def test_none_passthrough(self):
        from bibr.local.cli import _prepare_output_path

        assert _prepare_output_path(None, is_batch=True) is None


# ---- refs off (no-references mode) ---------------------------------------------


def test_chew_parser_accepts_refs_off():
    assert _chew_args("paper.pdf", "--refs", "off").refs == "off"


def test_resolve_carries_refs_and_ref_seg():
    config = _resolve("paper.pdf", "--refs", "off", "--ref-seg", "region")
    assert config.refs == "off"
    assert config.ref_seg == "region"


def test_resolve_refs_default_none():
    config = _resolve("paper.pdf")
    assert config.refs is None
    assert config.ref_seg is None


def test_active_stages_refs_off_drops_enrich():
    assert "enrich" not in _config(refs="off", crossref=True).active_stages([Path("paper.pdf")])


def test_help_examples_mention_refs_off():
    from bibr.local.cli import _EXAMPLES

    assert "--refs off" in _EXAMPLES
