"""ONNX reference parser parity against ``RefParser`` on a tiny random model."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchcrf")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from bibr.ner.parser_onnx import OnnxRefParser  # noqa: E402
from tests import onnx_fixtures as fx  # noqa: E402

REFS = [
    "Smith J (2020). Memory and consciousness. Journal of science, 26(1), 1-12.",
    "Doe A. 2021. Sleep. Journal 12: 1.",
    "",
    "smith doe smith doe smith doe smith doe smith doe smith doe smith doe smith doe smith",  # > max_seq_len
    "   ",
    "Doe A, Smith J (2020) methods results discussion",
]


@pytest.fixture
def pair(monkeypatch, tmp_path):
    model = fx.tiny_ner_model(monkeypatch)
    tok = fx.tiny_tokenizer()
    bundle = fx.export_ner_bundle(model, tok, tmp_path)
    return fx.torch_ref_parser(model, tok), OnnxRefParser(bundle, device="cpu"), tmp_path


def test_emissions_match_torch_on_padded_batch(pair):
    torch_parser, onnx_parser, _ = pair
    texts = [r for r in REFS if r.strip()]
    enc = torch_parser.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
        add_special_tokens=False,
    )
    model = torch_parser.model
    with torch.no_grad():
        hidden = model.encoder(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        ).last_hidden_state
        zeros = torch.zeros((hidden.shape[0], hidden.shape[1], model.feature_dim))
        hidden = hidden + model.feature_proj(zeros * torch.sigmoid(model.feature_gate))
        expected = model.linear(hidden).numpy()
    got = onnx_parser._emissions(enc["input_ids"].numpy(), enc["attention_mask"].numpy())
    mask = enc["attention_mask"].numpy().astype(bool)
    assert np.abs(expected - got)[mask].max() < 1e-4


def test_parse_and_parse_batch_match_torch(pair):
    torch_parser, onnx_parser, _ = pair
    assert onnx_parser.parse_batch(REFS, batch_size=2) == torch_parser.parse_batch(
        REFS, batch_size=2
    )
    assert [onnx_parser.parse(r) for r in REFS] == [torch_parser.parse(r) for r in REFS]
    assert onnx_parser.parse("") == {} and onnx_parser.parse("   ") == {}


def test_parse_batch_preserves_order_and_empties(pair):
    _torch_parser, onnx_parser, _ = pair
    out = onnx_parser.parse_batch(["", REFS[0], "   ", REFS[1]], batch_size=1)
    assert out[0] == {} and out[2] == {}
    assert out[1] == onnx_parser.parse(REFS[0])
    assert out[3] == onnx_parser.parse(REFS[1])


def test_loader_selects_onnx_for_local_bundle(pair, monkeypatch):
    _torch_parser, _onnx_parser, root = pair
    from bibr.config import GlobalSettings
    from bibr.ner.runtime import load_ref_parser

    monkeypatch.setenv("ML_RUNTIME", "auto")
    parser = load_ref_parser(str(root), device="cpu", revision=None, settings=GlobalSettings())
    assert isinstance(parser, OnnxRefParser)
    # A checkpoint *file* next to the bundle resolves too (NER_PARSER_CKPT style).
    ckpt = root / "best.pt"
    ckpt.write_bytes(b"")
    parser = load_ref_parser(str(ckpt), device="cpu", revision=None, settings=GlobalSettings())
    assert isinstance(parser, OnnxRefParser)
