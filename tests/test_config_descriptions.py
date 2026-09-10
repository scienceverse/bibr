"""CI gates for generated settings and current OCR documentation."""

from pathlib import Path

from bibr.config_introspect import iter_setting_docs


def test_every_setting_has_a_description():
    missing = [d.env_name for d in iter_setting_docs() if not d.description.strip()]
    assert not missing, (
        f"{len(missing)} settings lack Field(description=...): {sorted(missing)}\n"
        "Add a one-line description in bibr/config.py — the settings reference "
        "on bibr.org is generated from these."
    )


def test_current_docs_describe_the_paddle_first_ocr_contract():
    """Keep the user-facing OCR examples aligned with the runtime selector."""
    root = Path(__file__).resolve().parents[1]
    docs = (
        "README.md",
        "docs/getting-started/install.md",
        "docs/getting-started/quickstart.md",
        "docs/guides/architecture.md",
        "docs/guides/configuration.md",
        "docs/guides/deployment.md",
        "docs/guides/library.md",
    )
    content = "\n".join((root / path).read_text(encoding="utf-8") for path in docs)

    for required in (
        "PaddleOCR-VL-1.6",
        "OCR_PROFILE",
        "paddle-ocr-vl-1.6",
        "olragon/PaddleOCR-VL-1.6-8bit",
        "startup-only",
        "GLM fallback",
        "OTSL",
        "_raw_ocr_content",
    ):
        assert required in content


def test_automatic_paddle_examples_do_not_pin_fallback_overrides():
    """Keep selector examples distinct from concrete Paddle endpoint settings."""
    root = Path(__file__).resolve().parents[1]
    automatic_docs = (
        "docs/getting-started/install.md",
        "docs/getting-started/quickstart.md",
        "docs/guides/configuration.md",
    )
    for path in automatic_docs:
        content = (root / path).read_text(encoding="utf-8")
        assert "OCR_BACKEND=paddle\nOCR_PROFILE=" not in content
        assert "OCR_BACKEND=paddle\nOCR_MODEL=" not in content

    configuration = (root / "docs/guides/configuration.md").read_text(encoding="utf-8")
    assert (
        "OCR_BACKEND=paddle-http\nOCR_MODEL=paddle-ocr-vl-1.6\nOCR_PROFILE=paddle" in configuration
    )


def test_architecture_input_stage_describes_the_selected_paddle_first_runtime():
    """Prevent the obsolete GLM-only input-stage description from returning."""
    architecture = (Path(__file__).resolve().parents[1] / "docs/guides/architecture.md").read_text(
        encoding="utf-8"
    )

    for obsolete in (
        "PDF files are OCR'd via the glmocr SDK",
        "OCR backend selection (`glm-llama`, `glm-mlx`, `glm-http`)",
        "Tables, formulas, and figures always go through GLM-OCR regardless.",
    ):
        assert obsolete not in architecture

    assert "PP-DocLayoutV3 detects PDF regions" in architecture
    assert "selected OCR runtime" in architecture
