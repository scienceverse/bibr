"""HF Hub loader + inference for the trained MiniLM-L6 section classifier (torch).

The torch-free pieces (dataclasses, the input template, bundle resolution and
the runtime loader) live in :mod:`bibr.structure.section_classifier_common`
and are re-exported here for compatibility; the ONNX Runtime twin is
:mod:`bibr.structure.section_classifier_onnx`.
"""

from __future__ import annotations

from pathlib import Path

from bibr.utils.ml_extra import ml_import_error

try:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise ml_import_error("ML section classification") from e

from bibr.paper_contents import CanonicalSection
from bibr.structure._section_minilm_arch import SectionMiniLMModel
from bibr.structure.section_classifier_common import (
    _MAX_INFERENCE_BATCH,
    _POSITION_BUCKETS,
    HeaderContext,
    SectionPrediction,
    _build_input_text,
    _coerce_header_context,
    _download_snapshot,
    _position_bucket,
    load_bundle_labels,
)
from bibr.utils.hf_cache import (
    disable_hf_cache_symlinks_on_windows as _disable_hf_cache_symlinks_on_windows,
)
from bibr.utils.hf_cache import (
    is_windows_symlink_privilege_error as _is_windows_symlink_privilege_error,
)
from bibr.utils.tokenizer_safety import safe_tokenize

__all__ = [
    "HeaderContext",
    "SectionClassifierModel",
    "SectionPrediction",
    "_MAX_INFERENCE_BATCH",
    "_POSITION_BUCKETS",
    "_build_input_text",
    "_coerce_header_context",
    "_disable_hf_cache_symlinks_on_windows",
    "_download_snapshot",
    "_is_windows_symlink_privilege_error",
    "_pick_device",
    "_position_bucket",
]


def _pick_device() -> str:
    """CUDA → CPU → MPS.

    MPS is intentionally *not* a default for this 22M-param model: benchmarked
    2026-05-13 (1217-row test split), MPS+chunking takes ~47 s vs CPU ~23 s.
    Kernel-launch overhead dominates the tiny per-batch compute. Set
    ``ML_SECTION_CLASSIFIER_DEVICE=mps`` if you want it anyway.

    CUDA is skipped when the card's architecture is unsupported by the
    installed torch wheel (kernels would crash at launch) — same gate as the
    shared device ladder.
    """
    from bibr.utils.device import cuda_incompatibility

    if torch.cuda.is_available() and cuda_incompatibility() is None:
        return "cuda"
    return "cpu"


class SectionClassifierModel:
    """Wraps the trained two-head MiniLM section classifier for inference.

    Defaults to CUDA → CPU. ``classify_batch`` chunks input to
    ``_MAX_INFERENCE_BATCH`` per forward pass — required for MPS correctness
    on large batches (see ``section_classifier_common``), no-op for CUDA/CPU.
    """

    runtime = "torch"

    def __init__(self, model_dir: Path, device: str | None = None) -> None:
        self.device = device or _pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.label_classes, self.template_version = load_bundle_labels(model_dir)
        self.model = SectionMiniLMModel(num_types=len(self.label_classes))
        state = load_file(str(model_dir / "model.safetensors"))
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str = "main", device: str | None = None
    ) -> SectionClassifierModel:
        path = _download_snapshot(repo_id, revision=revision)
        return cls(path, device=device)

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
        texts = [_build_input_text(ctx, self.template_version) for ctx in contexts]
        enc = safe_tokenize(
            self.tokenizer,
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        # SectionMiniLMModel.forward only accepts input_ids + attention_mask;
        # BERT-style tokenizers also return token_type_ids which we drop here.
        model_inputs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        with torch.no_grad():
            out = self.model(**model_inputs)
        type_probs = torch.softmax(out["type_logits"], dim=-1)
        top_probs = torch.sigmoid(out["top_level_logits"]).squeeze(-1)
        # squeeze(-1) on a 1-D tensor is a no-op, but if batch=1 the head
        # already returned a 1-D vector — handle both shapes by ensuring 1-D.
        if top_probs.dim() == 0:
            top_probs = top_probs.unsqueeze(0)
        results: list[SectionPrediction] = []
        for i in range(len(contexts)):
            idx = int(type_probs[i].argmax())
            label = self.label_classes[idx]
            results.append(
                SectionPrediction(
                    canonical_type=CanonicalSection(label),
                    is_top_level=bool(top_probs[i] > 0.5),
                    score=float(type_probs[i, idx]),
                )
            )
        return results
