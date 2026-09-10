"""Unit tests for FeatureGatedEncoderCRF.

The encoder is stubbed (monkeypatched ``AutoModel.from_config`` /
``AutoConfig.from_pretrained``) so these run without a network download — both
model classes build the encoder via ``AutoModel.from_config``, so stubbing it is
the only way to exercise the real predict/decode logic cheaply.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

BIO = ["O", "B-X", "I-X"]
_VOCAB = 32
_HIDDEN = 16


class _TinyEncoder(nn.Module):
    """Deterministic stand-in for ModernBERT: embeds token ids."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=_HIDDEN)
        self.embedding = nn.Embedding(_VOCAB, _HIDDEN)

    def forward(self, input_ids, attention_mask=None):  # noqa: ARG002
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.fixture(autouse=True)
def _stub_encoder(monkeypatch):
    import bibr.ner.model as model_mod

    monkeypatch.setattr(model_mod.AutoConfig, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(model_mod.AutoModel, "from_config", lambda *a, **k: _TinyEncoder())


def test_state_dict_adds_only_feature_keys_over_base():
    from bibr.ner.model import EncoderCRFModel, FeatureGatedEncoderCRF

    fg = FeatureGatedEncoderCRF("stub", num_tags=len(BIO), bio_tags=BIO)
    base = EncoderCRFModel("stub", num_tags=len(BIO), bio_tags=BIO)

    extra = set(fg.state_dict()) - set(base.state_dict())
    assert extra == {"feature_proj.weight", "feature_proj.bias", "feature_gate"}
    # The base checkpoint keys must be unchanged so best.pt's shared params map 1:1.
    assert set(base.state_dict()) - set(fg.state_dict()) == set()


def test_feature_proj_and_gate_shapes():
    from bibr.ner.model import FeatureGatedEncoderCRF

    fg = FeatureGatedEncoderCRF("stub", num_tags=len(BIO), bio_tags=BIO)
    assert fg.feature_gate.shape == (8,)
    assert fg.feature_proj.weight.shape == (_HIDDEN, 8)


def test_predict_with_none_features_matches_base():
    from bibr.ner.model import EncoderCRFModel, FeatureGatedEncoderCRF

    fg = FeatureGatedEncoderCRF("stub", num_tags=len(BIO), bio_tags=BIO)
    base = EncoderCRFModel("stub", num_tags=len(BIO), bio_tags=BIO)
    # Copy shared encoder/linear/crf weights into base; ignore the feature keys.
    base.load_state_dict(fg.state_dict(), strict=False)
    fg.eval()
    base.eval()

    ids = torch.tensor([[1, 5, 9, 13, 2]])
    mask = torch.ones_like(ids)

    assert fg.predict(ids, mask, token_features=None) == base.predict(ids, mask)
