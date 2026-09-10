"""Multitask paper classifier: shared encoder + three linear heads.

Predicts OECD L1, OECD L2, and paper_type from a mean-pooled title+abstract
encoding. Vendored into bibr (mirroring ``_section_minilm_arch.py``) so the
inference/product repo can load the trained checkpoint without depending on
the bibr-training package.

Like the section arch, the encoder is built from *config only*
(``AutoModel.from_config``, NOT ``from_pretrained``): the base pretrained
weights are immediately overwritten by the trained checkpoint in
``load_state_dict``, so downloading them is wasted I/O and emits a spurious
transformers "unexpected keys" load report. The encoder name is passed in by
the loader (read from the bundle's ``inference_config.json``) so the encoder
choice — MiniLM, ModernBERT, SPECTER2, BGE — is not hardcoded on the bibr side.
"""

from __future__ import annotations

from bibr.utils.ml_extra import ml_import_error

try:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel
except ImportError as e:  # pragma: no cover
    raise ml_import_error("ML paper classification") from e

# Same general-domain default the training multitask baseline used; the loader
# overrides it with the bundle's recorded encoder name.
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class PaperClassifierMultitaskModel(nn.Module):
    """Shared transformer encoder with OECD L1, OECD L2, and paper_type heads.

    Head output dims (``num_l1`` / ``num_l2`` / ``num_paper_type``) come from
    the bundle's ``label_maps.json`` so the class order is exactly what the
    checkpoint was trained against.
    """

    def __init__(
        self,
        num_l1: int,
        num_l2: int,
        num_paper_type: int,
        encoder_name: str = ENCODER,
        dropout: float = 0.1,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_l1 < 1 or num_l2 < 1 or num_paper_type < 1:
            raise ValueError("head dimensions must each be at least 1")

        if encoder is not None:
            self.encoder = encoder
        else:
            # Build the encoder from config only (no weight download) — the
            # trained checkpoint overwrites these weights in load_state_dict.
            from bibr.utils.hf_cache import disable_hf_cache_symlinks_on_windows

            disable_hf_cache_symlinks_on_windows()
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(encoder_name))

        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.l1_head = nn.Linear(hidden, num_l1)
        self.l2_head = nn.Linear(hidden, num_l2)
        self.paper_type_head = nn.Linear(hidden, num_paper_type)

    def _pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = outputs.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = (last * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom  # mean pool, (B, H)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pooled = self.dropout(self._pooled(input_ids, attention_mask))
        return {
            "l1_logits": self.l1_head(pooled),
            "l2_logits": self.l2_head(pooled),
            "paper_type_logits": self.paper_type_head(pooled),
        }
