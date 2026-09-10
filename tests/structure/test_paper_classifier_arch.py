"""Unit tests for the vendored multitask paper-classifier arch.

Uses a tiny injected encoder so the tests stay offline (no HF download).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _tiny_encoder(hidden_size: int = 32):
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return BertModel(cfg)


def test_three_heads_output_shapes():
    from bibr.structure._paper_classifier_arch import PaperClassifierMultitaskModel

    model = PaperClassifierMultitaskModel(
        num_l1=6,
        num_l2=32,
        num_paper_type=8,
        encoder=_tiny_encoder(),
    )
    input_ids = torch.randint(0, 64, (3, 10))
    attention_mask = torch.ones(3, 10, dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert set(out) >= {"l1_logits", "l2_logits", "paper_type_logits"}
    assert out["l1_logits"].shape == (3, 6)
    assert out["l2_logits"].shape == (3, 32)
    assert out["paper_type_logits"].shape == (3, 8)


def test_head_dims_track_label_maps():
    from bibr.structure._paper_classifier_arch import PaperClassifierMultitaskModel

    model = PaperClassifierMultitaskModel(
        num_l1=2,
        num_l2=5,
        num_paper_type=3,
        encoder=_tiny_encoder(),
    )
    assert model.l1_head.out_features == 2
    assert model.l2_head.out_features == 5
    assert model.paper_type_head.out_features == 3


def test_builds_encoder_from_config_not_pretrained(monkeypatch):
    """When no encoder is injected, the arch must build via
    AutoModel.from_config (NOT from_pretrained) so fine-tuned weights loaded
    afterwards are not clobbered by base pretrained weights."""
    import bibr.structure._paper_classifier_arch as arch

    calls = {"from_config": 0, "from_pretrained": 0}

    real_config = _tiny_encoder().config

    def fake_from_pretrained(name):  # noqa: ARG001
        calls["from_pretrained"] += 1
        return real_config

    def fake_from_config(cfg):  # noqa: ARG001
        calls["from_config"] += 1
        return _tiny_encoder()

    monkeypatch.setattr(arch.AutoConfig, "from_pretrained", staticmethod(fake_from_pretrained))
    monkeypatch.setattr(arch.AutoModel, "from_config", staticmethod(fake_from_config))

    arch.PaperClassifierMultitaskModel(num_l1=6, num_l2=32, num_paper_type=8)

    assert calls["from_config"] == 1
    assert calls["from_pretrained"] == 1
