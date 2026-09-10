"""Exercise real local model loading with the supported torch dependency set."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_layout_checkpoint_roundtrip_preserves_predictions(tmp_path):
    """The single-device loader needs neither Accelerate dispatch nor Hub access."""
    from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor, ResNetConfig

    from bibr.layout_base import BaseLayoutDetector

    config = DetrConfig(
        backbone_config=ResNetConfig(
            depths=[1, 1, 1, 1],
            hidden_sizes=[16, 32, 64, 128],
            embedding_size=8,
            out_features=["stage4"],
        ),
        use_timm_backbone=False,
        use_pretrained_backbone=False,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_queries=3,
        num_labels=2,
    )
    original = DetrForObjectDetection(config).eval()
    original.save_pretrained(tmp_path)
    DetrImageProcessor(size={"height": 64, "width": 64}).save_pretrained(tmp_path)

    class Detector(BaseLayoutDetector):
        def _install_model(self, model):
            self._model = model

    detector = object.__new__(Detector)
    detector._settings = SimpleNamespace(layout=SimpleNamespace(model_revision=None))
    detector._init_torch(str(tmp_path), "cpu")
    pixels = torch.zeros(1, 3, 64, 64)
    with torch.inference_mode():
        expected = original(pixel_values=pixels)
        actual = detector._model(pixel_values=pixels)
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.pred_boxes, expected.pred_boxes)


def test_section_classifier_loads_local_encoder_config(tmp_path):
    from transformers import BertConfig

    from bibr.structure._section_minilm_arch import SectionMiniLMModel

    BertConfig(
        vocab_size=20,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    ).save_pretrained(tmp_path)
    model = SectionMiniLMModel(encoder_name=str(tmp_path)).eval()
    tokens = torch.ones(1, 4, dtype=torch.long)
    with torch.inference_mode():
        result = model(input_ids=tokens, attention_mask=torch.ones_like(tokens))
    assert result["type_logits"].shape == (1, 16)
    assert result["top_level_logits"].shape == (1,)
    assert all(torch.isfinite(value).all() for value in result.values())
