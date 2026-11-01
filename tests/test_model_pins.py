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
        "LAYOUT_MODEL_ID",
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
    assert f"ARG LAYOUT_MODEL_ID={s.layout.model_id}" in dockerfile
    assert f"ARG LAYOUT_MODEL_REVISION={s.layout.model_revision}" in dockerfile


def test_serve_image_pins_the_torch_runtime():
    """The classifier revisions now carry an onnx/ bundle too, and the serve image is
    the torch one (it exists for torch.compile on layout). Leaving ML_RUNTIME unset
    would let "auto" silently move serve onto ONNX Runtime the next time it is built."""
    dockerfile = Path("Dockerfile.serve").read_text()
    # The default stack is the torch one; the runtime the image pins is the stack's.
    assert "ARG ML_STACK=torch" in dockerfile
    assert "ENV ML_RUNTIME=${ML_STACK}" in dockerfile
    assert dockerfile.count("ignore_patterns=['onnx/*']") == 2


def _dockerfile_code() -> str:
    """Dockerfile.serve without its comment lines."""
    lines = Path("Dockerfile.serve").read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def test_serve_image_torch_stack_keeps_the_ml_extra_in_every_sync():
    """The torch stack is the unchanged image: every uv sync and the lock export
    carry --extra ml, through $ml_extra, which the ONNX stack leaves empty."""
    code = _dockerfile_code()
    assert 'torch) ml_extra="--extra ml"' in code
    assert 'onnx) ml_extra=""' in code
    assert "ML_STACK must be torch or onnx" in code
    # Two syncs in the first RUN, two in the second, and the export; none spells the extra out.
    assert code.count("$ml_extra") == 5
    assert code.count("--extra ml") == 2  # the two places $ml_extra is set


def test_serve_image_bakes_the_front_tagger_onnx_bundle_on_both_stacks():
    """Fast mode reads the front matter with the tagger, which has no torch form: only its
    onnx/ files are baked, from bibr's own default id and revision so a re-pin there
    needs no edit in the image."""
    code = _dockerfile_code()
    assert code.count("allow_patterns=['onnx/*']") == 1
    assert "ml.front_tagger_model_id, revision=ml.front_tagger_revision" in code
    assert "bibr-front-tagger" not in code  # no id or revision of its own to keep in step
    # The bake sits outside the ML_STACK branches that split the classifier downloads.
    (bake,) = [run for run in code.split("\nRUN ") if "allow_patterns=['onnx/*']" in run]
    assert "ML_STACK" not in bake


def test_docs_and_dry_run_mirror_the_parser_pin():
    """The runtime table in the configuration guide and the dry-run download sizes
    name the default reference parser; a re-pin must update them too."""
    from bibr.config import Settings
    from bibr.local.cli.dry_run import _APPROX_MODEL_SIZES

    docs = Path("docs/guides/configuration.md").read_text()
    row = f"| Reference parser | `{Settings.NER_PARSER_CKPT}` | `{Settings.NER_PARSER_REVISION[:8]}` |"
    assert row in docs
    assert Settings.NER_PARSER_CKPT in _APPROX_MODEL_SIZES
