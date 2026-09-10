"""Every Hub-loaded model artifact is pinned to a commit, and the serve image bakes the same."""

import re
from pathlib import Path

from bibr.config import GlobalSettings
from bibr.segmenter_base import resolve_wtpsplit_model

_SHA = re.compile(r"^[0-9a-f]{40}$")


def _defaults(monkeypatch) -> GlobalSettings:
    for var in (
        "ML_SECTION_CLASSIFIER_REVISION",
        "ML_PAPER_CLASSIFIER_REVISION",
        "LAYOUT_MODEL_REVISION",
        "LAYOUT_ONNX_REVISION",
        "NER_PARSER_REVISION",
        "ML_FRONT_ROLE_REVISION",
        "WTPSPLIT_MODEL_REVISION",
    ):
        monkeypatch.delenv(var, raising=False)
    return GlobalSettings()


def test_config_defaults_pin_every_hub_model(monkeypatch):
    """`main` is not reproducible: a hub push would silently change extraction output."""
    s = _defaults(monkeypatch)
    for value in (
        s.ml.section_classifier_revision,
        s.ml.paper_classifier_revision,
        s.layout.model_revision,
        s.layout.onnx_revision,
        s.ml.front_role_revision,
        s.NER_PARSER_REVISION,
        s.NER_SEG_REVISION,
    ):
        assert _SHA.match(value), value


def test_default_segmenter_is_pinned_and_other_hub_models_are_not():
    assert (
        resolve_wtpsplit_model("sat-6l-sm").revision == "d85d2b6ddfb19036c4c8e8b3b7ca45da684b0905"
    )
    assert resolve_wtpsplit_model("sat-12l-sm").revision is None
    assert resolve_wtpsplit_model("sat-6l-sm", revision="abc123").revision == "abc123"


def test_serve_image_bakes_the_same_revisions(monkeypatch):
    s = _defaults(monkeypatch)
    dockerfile = Path("Dockerfile.serve").read_text()
    assert f"ARG PAPER_CLASSIFIER_REVISION={s.ml.paper_classifier_revision}" in dockerfile
    assert f"ARG SECTION_CLASSIFIER_REVISION={s.ml.section_classifier_revision}" in dockerfile
    assert f"ARG LAYOUT_MODEL_REVISION={s.layout.model_revision}" in dockerfile


def test_serve_image_pins_the_torch_runtime():
    """The classifier revisions now carry an onnx/ bundle too, and the serve image is
    the torch one (it exists for torch.compile on layout). Leaving ML_RUNTIME unset
    would let "auto" silently move serve onto ONNX Runtime the next time it is built."""
    dockerfile = Path("Dockerfile.serve").read_text()
    assert "ENV ML_RUNTIME=torch" in dockerfile
    assert dockerfile.count("ignore_patterns=['onnx/*']") == 2
