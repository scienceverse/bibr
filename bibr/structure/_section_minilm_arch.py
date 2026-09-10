"""MiniLM-L6 section classifier with two heads:
- canonical_type (16-way softmax + cross-entropy)
- is_top_level (sigmoid + binary cross-entropy)

Vendored verbatim from bibr-training so that bibr can load the trained
checkpoint without depending on the training package.
"""

from __future__ import annotations

from bibr.utils.ml_extra import ml_import_error

try:
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModel
except ImportError as e:  # pragma: no cover
    raise ml_import_error("ML section classification") from e

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class SectionMiniLMModel(nn.Module):
    def __init__(
        self,
        num_types: int = 16,
        encoder_name: str = ENCODER,
        dropout: float = 0.1,
        type_loss_weight: float = 1.0,
        top_level_loss_weight: float = 0.5,
    ):
        super().__init__()
        # Build the encoder from config only (no weight download). The base
        # pretrained weights are immediately overwritten by the trained
        # checkpoint in load_state_dict, so loading them is wasted I/O and emits
        # a spurious transformers "UNEXPECTED keys" load report.
        from bibr.utils.hf_cache import disable_hf_cache_symlinks_on_windows

        disable_hf_cache_symlinks_on_windows()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(encoder_name))
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_type = nn.Linear(hidden, num_types)
        self.head_top_level = nn.Linear(hidden, 1)
        self.type_loss_weight = type_loss_weight
        self.top_level_loss_weight = top_level_loss_weight

    def _pooled(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = outputs.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = (last * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom  # mean pool, (B, H)

    def forward(
        self,
        input_ids,
        attention_mask,
        type_labels=None,
        is_top_labels=None,
    ):
        pooled = self.dropout(self._pooled(input_ids, attention_mask))
        type_logits = self.head_type(pooled)  # (B, num_types)
        top_logits = self.head_top_level(pooled).squeeze(-1)  # (B,)
        out = {"type_logits": type_logits, "top_level_logits": top_logits}
        if type_labels is not None and is_top_labels is not None:
            type_loss = F.cross_entropy(type_logits, type_labels)
            top_loss = F.binary_cross_entropy_with_logits(top_logits, is_top_labels)
            out["loss"] = self.type_loss_weight * type_loss + self.top_level_loss_weight * top_loss
            out["type_loss"] = type_loss
            out["top_loss"] = top_loss
        return out
