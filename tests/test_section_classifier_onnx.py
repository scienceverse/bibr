"""ONNX section classifier parity against the torch class on a tiny random model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from bibr.structure.section_classifier_common import HeaderContext  # noqa: E402
from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel  # noqa: E402
from tests import onnx_fixtures as fx  # noqa: E402

CONTEXTS = [
    HeaderContext("Methods", "Participants were recruited. " * 5, 0.3, "Introduction", "Results"),
    HeaderContext("Results", "The results of the sleep study.", 0.6, "Methods", "Discussion"),
    HeaderContext("Ethics", "IRB approved.", 0.95, "Discussion", ""),
    HeaderContext("Something unknown", "", 0.5, "", ""),
    ("Introduction", "memory and consciousness"),  # tuple form
]


@pytest.fixture
def pair(monkeypatch, tmp_path):
    model = fx.tiny_section_model(monkeypatch)
    tok = fx.tiny_tokenizer()
    bundle = fx.export_section_bundle(model, tok, tmp_path)
    return (
        fx.torch_section_classifier(model, tok),
        OnnxSectionClassifierModel(bundle, "cpu"),
        tmp_path,
    )


def test_predictions_match_torch(pair):
    torch_clf, onnx_clf, _ = pair
    a = torch_clf.classify_batch(CONTEXTS, max_length=32)
    b = onnx_clf.classify_batch(CONTEXTS, max_length=32)
    assert len(a) == len(b) == len(CONTEXTS)
    for x, y in zip(a, b, strict=True):
        assert x.canonical_type == y.canonical_type
        assert x.is_top_level == y.is_top_level
        assert abs(x.score - y.score) < 1e-5


def test_single_item_and_lone_surrogates(pair):
    _torch_clf, onnx_clf, _ = pair
    (pred,) = onnx_clf.classify_batch([HeaderContext("Meth\udce9ods", "Body \ud835text")])
    assert pred.canonical_type.value in fx.SECTION_LABELS
    assert 0.0 <= pred.score <= 1.0


def test_from_pretrained_uses_local_bundle(pair):
    _torch_clf, _onnx_clf, root = pair
    clf = OnnxSectionClassifierModel.from_pretrained(str(root), device="cpu")
    assert clf.label_classes == fx.SECTION_LABELS
    assert clf.template_version == 3
    assert clf.runtime == "onnx"


def test_loader_selects_onnx_for_local_bundle(pair, monkeypatch):
    _torch_clf, _onnx_clf, root = pair
    from bibr.config import GlobalSettings
    from bibr.structure.section_classifier_common import load_section_classifier

    monkeypatch.setenv("ML_RUNTIME", "auto")
    clf = load_section_classifier(
        str(root), revision="main", device="cpu", settings=GlobalSettings()
    )
    assert isinstance(clf, OnnxSectionClassifierModel)


def test_bundle_whose_tokenizer_forgets_the_separator_is_rejected(pair):
    """The template puts a literal "[SEP]" in the input text.

    A tokenizer.json carrying the specials in its vocabulary but not in
    ``added_tokens`` lowercases and splits that marker, which shifts every
    prediction with nothing raised — so loading such a bundle must fail.
    """
    import json

    from tokenizers import Tokenizer

    from bibr.exceptions import ConfigurationError

    _torch_clf, onnx_clf, _root = pair
    path = onnx_clf.bundle_dir / "tokenizer.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["added_tokens"] = []
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert len(Tokenizer.from_file(str(path)).encode("[SEP]", add_special_tokens=False).ids) > 1

    with pytest.raises(ConfigurationError, match="added special token"):
        OnnxSectionClassifierModel(onnx_clf.bundle_dir, "cpu")
