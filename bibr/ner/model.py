"""Encoder + CRF model for BIO sequence labeling.

Mirrors bibr_training.models.encoder_crf so trained checkpoints load
without modification. Inference-only: the training-time forward path that
returns loss given labels is preserved for compatibility but unused here.
"""

from __future__ import annotations

from bibr.utils.ml_extra import ml_import_error

try:
    import torch
    import torch.nn as nn
    from torchcrf import CRF
    from transformers import AutoConfig, AutoModel
except ImportError as e:  # pragma: no cover
    raise ml_import_error("NER encoder-CRF models (bibr.ner)") from e

_NEG_INF = -10000.0


def _apply_bio_constraints(crf: CRF, bio_tags: list[str]) -> None:
    tag_to_idx = {tag: idx for idx, tag in enumerate(bio_tags)}
    with torch.no_grad():
        for tag, idx in tag_to_idx.items():
            if tag.startswith("I-"):
                crf.start_transitions.data[idx] = _NEG_INF
        for from_tag, from_idx in tag_to_idx.items():
            for to_tag, to_idx in tag_to_idx.items():
                if not to_tag.startswith("I-"):
                    continue
                to_field = to_tag[2:]
                if from_tag == f"B-{to_field}" or from_tag == f"I-{to_field}":
                    continue
                crf.transitions.data[from_idx, to_idx] = _NEG_INF


class EncoderCRFModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_tags: int,
        dropout: float = 0.0,
        bio_tags: list[str] | None = None,
    ) -> None:
        super().__init__()
        # Build the encoder from config only (no weight download). The base
        # pretrained weights would be immediately overwritten by the trained
        # checkpoint in load_state_dict, so loading them is wasted I/O and emits
        # a spurious transformers "UNEXPECTED keys" load report for the base
        # MLM head we never use.
        from bibr.utils.hf_cache import disable_hf_cache_symlinks_on_windows

        disable_hf_cache_symlinks_on_windows()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(encoder_name))
        hidden_size: int = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

        if bio_tags is not None:
            _apply_bio_constraints(self.crf, bio_tags)
            self.register_buffer("_transition_mask", self.crf.transitions.data == _NEG_INF)
            self.register_buffer("_start_mask", self.crf.start_transitions.data == _NEG_INF)
        else:
            self.register_buffer("_transition_mask", None)
            self.register_buffer("_start_mask", None)

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[list[int]]:
        if self._transition_mask is not None:
            self.crf.transitions.data[self._transition_mask] = _NEG_INF
            self.crf.start_transitions.data[self._start_mask] = _NEG_INF
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.linear(outputs.last_hidden_state)
        return self.crf.decode(emissions, mask=attention_mask.bool())


# Per-token feature dim of the v4 parser (mirrors bibr_training.data.features).
_FEATURE_DIM = 8


class FeatureGatedEncoderCRF(EncoderCRFModel):
    """Encoder + CRF with an optional gated per-token feature residual.

    Faithful mirror of ``bibr_training.models.feature_encoder_crf`` so the v4
    checkpoint (``best.pt``) loads with ``strict=True``: it adds exactly
    ``feature_proj.{weight,bias}`` and ``feature_gate`` over the base.

    At parse time the plain reference string has no layout features. RefParser therefore passes an all-zero token_features tensor, preserving the learned feature_proj.bias residual from training. Passing None would skip that residual and change the model forward path.
    """

    def __init__(
        self,
        encoder_name: str,
        num_tags: int,
        dropout: float = 0.0,
        bio_tags: list[str] | None = None,
        feature_dim: int = _FEATURE_DIM,
    ) -> None:
        super().__init__(
            encoder_name=encoder_name,
            num_tags=num_tags,
            dropout=dropout,
            bio_tags=bio_tags,
        )
        self.feature_dim = feature_dim
        self.feature_proj = nn.Linear(feature_dim, self.encoder.config.hidden_size)
        self.feature_gate = nn.Parameter(torch.zeros(feature_dim))

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_features: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if self._transition_mask is not None:
            self.crf.transitions.data[self._transition_mask] = _NEG_INF
            self.crf.start_transitions.data[self._start_mask] = _NEG_INF
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if token_features is not None:
            gate = torch.sigmoid(self.feature_gate)
            hidden = hidden + self.feature_proj(token_features * gate)
        emissions = self.linear(hidden)
        return self.crf.decode(emissions, mask=attention_mask.bool())
