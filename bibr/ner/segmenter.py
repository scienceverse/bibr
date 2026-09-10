"""Trained ref segmenter (ModernBERT + CRF).

Segments a references-section text block into individual reference strings
via per-character BIO labels. Sliding-window inference with overlap merge
handles refs longer than max_seq_len.
"""

from __future__ import annotations

from pathlib import Path

from bibr.utils.ml_extra import ml_import_error

try:
    import torch
    from transformers import AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise ml_import_error("NER reference segmentation (bibr.ner)") from e

from bibr.utils.text import strip_lone_surrogates

from .checkpoint import resolve_checkpoint
from .model import EncoderCRFModel
from .tags import SEG_TAGS

DEFAULT_ENCODER = "answerdotai/ModernBERT-base"


class RefSegmenter:
    def __init__(
        self,
        ckpt_path: str | Path,
        encoder_name: str = DEFAULT_ENCODER,
        device: str | None = None,
        window: int = 2048,
        stride: int = 1536,
        revision: str | None = None,
    ) -> None:
        if device is None:
            from bibr.utils.device import detect_torch_device

            device = detect_torch_device()
        self.device = device
        self.window = window
        self.stride = stride
        from bibr.utils.hf_cache import disable_hf_cache_symlinks_on_windows

        disable_hf_cache_symlinks_on_windows()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.model = EncoderCRFModel(
            encoder_name=encoder_name,
            num_tags=len(SEG_TAGS),
            dropout=0.0,
            bio_tags=SEG_TAGS,
        ).to(device)
        local_path = resolve_checkpoint(ckpt_path, revision=revision)
        state = torch.load(local_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def segment(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        text = strip_lone_surrogates(text)
        # Tokenize WITH special tokens to match training input distribution
        # (training script re-tokenized with default add_special_tokens=True).
        # Then offset+pred extraction strips them and applies the same
        # +1 shift as the parser does — see RefParser.parse().
        enc = self.tokenizer(
            text,
            truncation=False,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        all_ids = enc["input_ids"]
        all_offsets = enc["offset_mapping"]
        n = len(all_ids)

        global_tags: list[str | None] = [None] * n
        global_trust: list[int] = [-1] * n

        start = 0
        while start < n:
            end = min(start + self.window, n)
            ids = all_ids[start:end]
            ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)
            attn_t = torch.ones_like(ids_t)
            preds = self.model.predict(ids_t, attn_t)[0]
            shifted = [0] + list(preds[:-1])
            tags = [SEG_TAGS[p] for p in shifted]
            for i, tag in enumerate(tags):
                gi = start + i
                trust = min(i, len(tags) - 1 - i)
                if trust > global_trust[gi]:
                    global_tags[gi] = tag
                    global_trust[gi] = trust
            if end >= n:
                break
            start += self.stride

        refs: list[str] = []
        current_start: int | None = None
        for i, tag in enumerate(global_tags):
            ts, te = all_offsets[i]
            if ts == 0 and te == 0:
                continue
            if tag == "B-REF":
                if current_start is not None:
                    refs.append(text[current_start:ts].strip())
                current_start = ts
            elif tag == "O":
                if current_start is not None:
                    refs.append(text[current_start:ts].strip())
                    current_start = None
        if current_start is not None:
            refs.append(text[current_start:].strip())
        return [r for r in refs if r]
