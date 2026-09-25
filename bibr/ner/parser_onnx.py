"""ONNX Runtime reference parser (ModernBERT emissions + numpy CRF decode).

Behaviourally identical to :class:`bibr.ner.parser.RefParser`: the same
tokenizer settings (``add_special_tokens=False``, truncation to
``max_seq_len``), the same all-zeros ``token_features`` residual (folded into
the exported graph, so the learned ``feature_proj.bias`` is applied to every
token as in training), and Viterbi decoding with the checkpoint's CRF
parameters taken from the bundle manifest.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from bibr.utils.ml_runtime import ONNX_MODEL, read_onnx_manifest
from bibr.utils.onnx_tokenizer import encode_batch, load_tokenizer
from bibr.utils.text import strip_lone_surrogates

from .crf_numpy import viterbi_decode
from .decode import decode_bio_spans, map_fields_to_paper_ref
from .tags import BIO_TAGS

logger = logging.getLogger(__name__)


class OnnxRefParser:
    runtime = "onnx"

    def __init__(
        self,
        bundle_dir: str | Path,
        device: str | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        from bibr.utils.onnx_providers import create_session

        self.bundle_dir = Path(bundle_dir)
        self.manifest = read_onnx_manifest(self.bundle_dir)
        self.tags: list[str] = [str(t) for t in self.manifest.get("bio_tags", BIO_TAGS)]
        self.max_seq_len = int(max_seq_len or self.manifest.get("max_length", 256))
        crf = self.manifest["crf"]
        self.start_transitions = np.asarray(crf["start_transitions"], dtype=np.float32)
        self.end_transitions = np.asarray(crf["end_transitions"], dtype=np.float32)
        self.transitions = np.asarray(crf["transitions"], dtype=np.float32)
        if self.transitions.shape != (len(self.tags), len(self.tags)):
            raise ValueError(
                f"ONNX parser bundle {self.bundle_dir}: transitions shape "
                f"{self.transitions.shape} does not match {len(self.tags)} tags"
            )
        self.tokenizer = load_tokenizer(self.bundle_dir, self.manifest)
        self.session, self.device = create_session(
            self.bundle_dir / self.manifest.get("model_file", ONNX_MODEL),
            device=device,
            model_name="NER reference parser (ONNX)",
        )
        self._input_names = [i.name for i in self.session.get_inputs()]

    def close(self) -> None:
        """Release the ORT session (see ``unload_ner_parser``). Idempotent."""
        self.session = None

    def parse(self, ref_text: str) -> dict[str, str | int]:
        """Parse one reference. Returns a dict with PaperReference field names."""
        if not ref_text or not ref_text.strip():
            return {}
        return self.parse_batch([ref_text], batch_size=1)[0]

    def parse_batch(self, ref_texts: list[str], batch_size: int = 32) -> list[dict[str, str | int]]:
        """Parse many references in padded forward passes of ``batch_size``.

        Empty/whitespace entries map to ``{}`` without consuming a model slot;
        similar-length refs are grouped so each chunk pads to a shorter common
        length. Output order matches the input.
        """
        results: list[dict[str, str | int]] = [{} for _ in ref_texts]
        slots = [i for i, t in enumerate(ref_texts) if t and t.strip()]
        slots.sort(key=lambda i: len(ref_texts[i]))
        step = max(1, batch_size)
        for start in range(0, len(slots), step):
            chunk_slots = slots[start : start + step]
            texts = [strip_lone_surrogates(ref_texts[i]) for i in chunk_slots]
            batch = encode_batch(
                self.tokenizer, texts, max_length=self.max_seq_len, add_special_tokens=False
            )
            if batch.input_ids.shape[1] == 0:
                continue
            emissions = self._emissions(batch.input_ids, batch.attention_mask)
            paths = viterbi_decode(
                emissions,
                batch.attention_mask,
                start_transitions=self.start_transitions,
                end_transitions=self.end_transitions,
                transitions=self.transitions,
            )
            for slot, path, offsets, text in zip(
                chunk_slots, paths, batch.offsets, texts, strict=True
            ):
                results[slot] = self._decode(path, offsets[: len(path)], text)
        return results

    def _emissions(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        (emissions,) = self.session.run(["emissions"], feeds)
        return np.asarray(emissions, dtype=np.float32)

    def _decode(self, preds: list[int], offsets: list, text: str) -> dict[str, str | int]:
        tags = [self.tags[p] for p in preds]
        return map_fields_to_paper_ref(decode_bio_spans(tags, offsets, text))
