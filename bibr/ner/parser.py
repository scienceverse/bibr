"""Trained ref parser (ModernBERT + feature-gated CRF), v4 checkpoint.

Takes a single reference string and emits BIO field tags, then maps the 39-tag
parser output to a flat dict suitable for ``PaperReference.model_validate``.

The v4 checkpoint (``best.pt``) is a
:class:`~bibr.ner.model.FeatureGatedEncoderCRF` trained with
``add_special_tokens=False`` and no tag shift. Its ``feature_proj.weight`` is
~0 (the per-token feature values were never learned), but ``feature_proj.bias``
is a learned constant that was added to every token during training. Inference
therefore passes an all-zeros ``token_features`` tensor rather than ``None``: the
zero values are inert (weight ~0) while the residual re-applies the bias, so the
forward path matches training. Passing ``None`` would drop the bias and suppress
the lowest-margin tags (e.g. URL DOIs).
"""

from __future__ import annotations

from pathlib import Path

from bibr.utils.ml_extra import ml_import_error

try:
    import torch
    from transformers import AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise ml_import_error("NER reference parsing (bibr.ner)") from e

from bibr.utils.text import strip_lone_surrogates

from .checkpoint import resolve_checkpoint
from .decode import decode_bio_spans, map_fields_to_paper_ref
from .model import FeatureGatedEncoderCRF
from .tags import BIO_TAGS

DEFAULT_ENCODER = "answerdotai/ModernBERT-base"


class RefParser:
    def __init__(
        self,
        ckpt_path: str | Path,
        encoder_name: str = DEFAULT_ENCODER,
        device: str | None = None,
        max_seq_len: int = 256,
        revision: str | None = None,
    ) -> None:
        if device is None:
            from bibr.utils.device import detect_torch_device

            device = detect_torch_device()
        self.device = device
        self.max_seq_len = max_seq_len
        from bibr.utils.hf_cache import disable_hf_cache_symlinks_on_windows

        disable_hf_cache_symlinks_on_windows()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.model = FeatureGatedEncoderCRF(
            encoder_name=encoder_name,
            num_tags=len(BIO_TAGS),
            dropout=0.0,
            bio_tags=BIO_TAGS,
        ).to(device)
        local_path = resolve_checkpoint(ckpt_path, revision=revision)
        state = torch.load(local_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def parse(self, ref_text: str) -> dict[str, str | int]:
        """Parse one reference. Returns a dict with PaperReference field names.

        Year is converted to int when possible. Multi-token spans for the same
        field are joined with a single space.
        """
        if not ref_text or not ref_text.strip():
            return {}
        ref_text = strip_lone_surrogates(ref_text)
        enc = self.tokenizer(
            ref_text,
            truncation=True,
            max_length=self.max_seq_len,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()
        # Zeros (not None): the checkpoint's feature_proj.weight is ~0, so the
        # values are inert, but the residual re-applies the learned
        # feature_proj.bias to match the training forward path. None would drop
        # the bias and suppress low-margin tags.
        token_features = torch.zeros(
            (1, input_ids.shape[1], self.model.feature_dim), device=self.device
        )
        preds = self.model.predict(input_ids, attn, token_features=token_features)[0]
        return self._decode(preds, offsets, ref_text)

    def parse_batch(self, ref_texts: list[str], batch_size: int = 32) -> list[dict[str, str | int]]:
        """Parse many references in padded forward passes of ``batch_size``.

        Equivalent to ``[self.parse(t) for t in ref_texts]`` but batched.
        Empty/whitespace entries map to ``{}`` without consuming a model slot.
        Chunking bounds peak activation memory on long bibliographies.
        """
        results: list[dict[str, str | int]] = [{} for _ in ref_texts]
        slots = [i for i, t in enumerate(ref_texts) if t and t.strip()]
        # Group similar-length refs so each chunk pads to a shorter common
        # length (less wasted compute). Results map back to original positions,
        # so output order is unchanged.
        slots.sort(key=lambda i: len(ref_texts[i]))
        for start in range(0, len(slots), max(1, batch_size)):
            chunk_slots = slots[start : start + max(1, batch_size)]
            texts = [strip_lone_surrogates(ref_texts[i]) for i in chunk_slots]
            enc = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_len,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            offsets = enc["offset_mapping"].tolist()  # (B, L, 2), right-padded
            token_features = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], self.model.feature_dim),
                device=self.device,
            )
            preds = self.model.predict(input_ids, attn, token_features=token_features)
            for slot, row_preds, row_offsets, text in zip(
                chunk_slots, preds, offsets, texts, strict=True
            ):
                # CRF decode trims each row to its mask length; align offsets.
                results[slot] = self._decode(row_preds, row_offsets[: len(row_preds)], text)
        return results

    def _decode(self, preds: list[int], offsets: list, text: str) -> dict[str, str | int]:
        tags = [BIO_TAGS[p] for p in preds]
        return map_fields_to_paper_ref(decode_bio_spans(tags, offsets, text))
