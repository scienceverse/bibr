"""ONNX paper classifier parity against the torch class on a tiny random model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from bibr.structure.paper_classifier_onnx import OnnxPaperClassifierModel  # noqa: E402
from tests import onnx_fixtures as fx  # noqa: E402

ITEMS = [
    ("Memory and consciousness", "Participants were recruited. Results and discussion."),
    ("Sleep science", ""),
    ("", "The journal of science, 2020."),
    ("A very long title " * 8, "and a long abstract " * 20),
]


@pytest.fixture
def pair(monkeypatch, tmp_path):
    model = fx.tiny_paper_model()
    tok = fx.tiny_tokenizer()
    bundle = fx.export_paper_bundle(model, tok, tmp_path)
    return fx.torch_paper_classifier(model, tok), OnnxPaperClassifierModel(bundle, "cpu"), tmp_path


def test_predictions_match_torch(pair):
    torch_clf, onnx_clf, _ = pair
    a = torch_clf.classify_batch(ITEMS)
    b = onnx_clf.classify_batch(ITEMS)
    for x, y in zip(a, b, strict=True):
        assert (x.oecd_l1, x.oecd_l2, x.paper_type) == (y.oecd_l1, y.oecd_l2, y.paper_type)
        assert abs(x.oecd_l1_score - y.oecd_l1_score) < 1e-5
        assert abs(x.oecd_l2_score - y.oecd_l2_score) < 1e-5
        # Temperature (1.5) is applied on both sides.
        assert abs(x.paper_type_score - y.paper_type_score) < 1e-5


def test_temperature_comes_from_manifest(pair):
    _torch_clf, onnx_clf, _ = pair
    assert onnx_clf.paper_type_temperature == 1.5
    assert onnx_clf.l2_classes == fx.L2


def test_loader_selects_onnx_for_local_bundle(pair, monkeypatch):
    _torch_clf, _onnx_clf, root = pair
    from bibr.config import GlobalSettings
    from bibr.structure.paper_classifier_common import load_paper_classifier

    monkeypatch.setenv("ML_RUNTIME", "auto")
    clf = load_paper_classifier(str(root), revision="main", device="cpu", settings=GlobalSettings())
    assert isinstance(clf, OnnxPaperClassifierModel)
