"""ONNX Runtime inference for the trained MiniLM-L6 section classifier.

Same ``classify_batch`` surface as
:class:`bibr.structure.section_classifier_model.SectionClassifierModel`, but
fed by a ``tokenizers`` tokenizer and an ``onnxruntime`` session, so the core
install (no torch/transformers) classifies section headers locally.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from bibr.paper_contents import CanonicalSection
from bibr.structure.section_classifier_common import (
    _MAX_INFERENCE_BATCH,
    TEMPLATE_SEPARATOR,
    HeaderContext,
    SectionPrediction,
    _build_input_text,
    _coerce_header_context,
)
from bibr.utils.ml_runtime import ONNX_MODEL, read_onnx_manifest
from bibr.utils.onnx_tokenizer import encode_batch, load_tokenizer, require_added_token

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


class OnnxSectionClassifierModel:
    """Wraps the exported two-head section classifier for ONNX Runtime inference."""

    runtime = "onnx"

    def __init__(self, bundle_dir: str | Path, device: str | None = None) -> None:
        from bibr.utils.onnx_providers import create_session

        self.bundle_dir = Path(bundle_dir)
        self.manifest = read_onnx_manifest(self.bundle_dir)
        self.label_classes: list[str] = [str(x) for x in self.manifest["label_classes"]]
        self.template_version = int(self.manifest.get("template_version", 2))
        self.max_length = int(self.manifest.get("max_length", 256))
        self.tokenizer = load_tokenizer(self.bundle_dir, self.manifest)
        require_added_token(self.tokenizer, TEMPLATE_SEPARATOR, label="section classifier")
        self.session, self.device = create_session(
            self.bundle_dir / self.manifest.get("model_file", ONNX_MODEL),
            device=device,
            model_name="section classifier (ONNX)",
        )
        self._input_names = [i.name for i in self.session.get_inputs()]

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str = "main", device: str | None = None
    ) -> OnnxSectionClassifierModel:
        from bibr.exceptions import ConfigurationError
        from bibr.utils.ml_runtime import find_onnx_bundle

        bundle = find_onnx_bundle(repo_id, revision, label="section classifier")
        if bundle is None:
            raise ConfigurationError(f"No ONNX bundle for section classifier {repo_id}@{revision}")
        return cls(bundle, device=device)

    def classify_batch(
        self, items: list[HeaderContext] | list[tuple[str, str]], max_length: int = 256
    ) -> list[SectionPrediction]:
        contexts = [_coerce_header_context(item) for item in items]
        results: list[SectionPrediction] = []
        for start in range(0, len(contexts), _MAX_INFERENCE_BATCH):
            chunk = contexts[start : start + _MAX_INFERENCE_BATCH]
            results.extend(self._forward_chunk(chunk, max_length))
        return results

    def _forward_chunk(
        self, contexts: list[HeaderContext], max_length: int
    ) -> list[SectionPrediction]:
        if not contexts:
            return []
        texts = [_build_input_text(ctx, self.template_version) for ctx in contexts]
        batch = encode_batch(self.tokenizer, texts, max_length=max_length)
        feeds = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        type_logits, top_logits = self.session.run(["type_logits", "top_level_logits"], feeds)
        type_probs = _softmax(np.asarray(type_logits, dtype=np.float32))
        top_probs = 1.0 / (1.0 + np.exp(-np.asarray(top_logits, dtype=np.float32).reshape(-1)))
        results: list[SectionPrediction] = []
        for i in range(len(contexts)):
            idx = int(type_probs[i].argmax())
            results.append(
                SectionPrediction(
                    canonical_type=CanonicalSection(self.label_classes[idx]),
                    is_top_level=bool(top_probs[i] > 0.5),
                    score=float(type_probs[i, idx]),
                )
            )
        return results
