"""HF Hub loader + inference for the trained multitask paper classifier.

Mirrors ``section_classifier_model.py``: snapshot_download → safetensors +
tokenizer + label_maps.json + inference_config.json, then a synchronous
``classify_batch`` that predicts OECD L1, OECD L2, and paper_type (each with a
per-head softmax confidence) from title+abstract pairs.

The input text is built byte-identically to bibr-training's ``title_abstract_v1``
template (``build_input_text`` in
``bibr-training/src/bibr_training/paper_classifier/dataset.py``): the cleaned,
whitespace-collapsed title and abstract joined by ``" [SEP] "``, dropping either
part when empty.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from bibr.utils.ml_extra import ml_import_error

try:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise ml_import_error("ML paper classification") from e

from huggingface_hub import snapshot_download

from bibr.structure._paper_classifier_arch import ENCODER, PaperClassifierMultitaskModel
from bibr.utils.hf_cache import (
    disable_hf_cache_symlinks_on_windows,
    is_windows_symlink_privilege_error,
)
from bibr.utils.tokenizer_safety import safe_tokenize

# Byte-identical to bibr-training's dataset.TEXT_SEPARATOR (line 18).
_TEXT_SEPARATOR = " [SEP] "
# Byte-identical to bibr-training's dataset._clean_str whitespace normalization.
_WS_RE = re.compile(r"\s+")

# Cap per forward pass — same MPS-correctness rationale as the section model.
_MAX_INFERENCE_BATCH = 64


@dataclass
class PaperClassificationPrediction:
    """One paper's multitask prediction with per-head softmax confidences."""

    oecd_l1: str
    oecd_l1_score: float
    oecd_l2: str
    oecd_l2_score: float
    paper_type: str
    paper_type_score: float


def _clean_str(value: str | None) -> str:
    """Whitespace-collapse + strip, matching bibr-training's _clean_str."""
    if value is None:
        return ""
    return _WS_RE.sub(" ", str(value)).strip()


def _build_input_text(title: str | None, abstract: str | None) -> str:
    """Build the title_abstract_v1 input text, byte-identical to training.

    ``" [SEP] ".join(non-empty cleaned parts)`` — dropping the separator when
    either title or abstract is empty (see build_input_text in
    bibr-training/src/bibr_training/paper_classifier/dataset.py).
    """
    parts = [p for p in (_clean_str(title), _clean_str(abstract)) if p]
    return _TEXT_SEPARATOR.join(parts)


def _pick_device() -> str:
    """CUDA → CPU (skipping CUDA the installed torch wheel can't drive)."""
    from bibr.utils.device import cuda_incompatibility

    if torch.cuda.is_available() and cuda_incompatibility() is None:
        return "cuda"
    return "cpu"


def _download_snapshot(repo_id: str, revision: str) -> Path:
    disable_hf_cache_symlinks_on_windows()
    try:
        return Path(snapshot_download(repo_id, revision=revision))
    except OSError as e:
        if not is_windows_symlink_privilege_error(e):
            raise
        disable_hf_cache_symlinks_on_windows()
        return Path(snapshot_download(repo_id, revision=revision, max_workers=1))


class PaperClassifierModel:
    """Wraps the trained multitask (L1/L2/paper_type) classifier for inference."""

    def __init__(self, model_dir: Path, device: str | None = None) -> None:
        self.device = device or _pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        with (model_dir / "label_maps.json").open() as fh:
            label_maps = json.load(fh)
        self.l1_classes: list[str] = [str(x) for x in label_maps["l1_classes"]]
        self.l2_classes: list[str] = [str(x) for x in label_maps.get("l2_classes", [])]
        self.paper_type_classes: list[str] = [
            str(x) for x in label_maps.get("paper_type_classes", [])
        ]

        encoder_name = ENCODER
        self.max_length = 256
        # Temperature scaling for the paper_type head (Guo et al. 2017). Fitted on a
        # held-out split during training and written to inference_config.json; 1.0 is a
        # no-op, so an uncalibrated bundle behaves exactly as before. Recalibrating the
        # paper_type probability is what makes the confidence-gated LLM fallback
        # (paper_classifier_min_confidence) fire on the right papers.
        self.paper_type_temperature = 1.0
        inference_config_path = model_dir / "inference_config.json"
        if inference_config_path.exists():
            with inference_config_path.open() as fh:
                inference_config = json.load(fh)
            encoder_name = inference_config.get("encoder_name", ENCODER)
            self.max_length = int(inference_config.get("max_length", 256))
            self.paper_type_temperature = float(inference_config.get("paper_type_temperature", 1.0))

        self.model = PaperClassifierMultitaskModel(
            num_l1=len(self.l1_classes),
            num_l2=max(1, len(self.l2_classes)),
            num_paper_type=max(1, len(self.paper_type_classes)),
            encoder_name=encoder_name,
        )
        state = load_file(str(model_dir / "model.safetensors"))
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str = "main", device: str | None = None
    ) -> PaperClassifierModel:
        path = _download_snapshot(repo_id, revision=revision)
        return cls(path, device=device)

    def classify_batch(self, items: list[tuple[str, str]]) -> list[PaperClassificationPrediction]:
        results: list[PaperClassificationPrediction] = []
        for start in range(0, len(items), _MAX_INFERENCE_BATCH):
            chunk = items[start : start + _MAX_INFERENCE_BATCH]
            results.extend(self._forward_chunk(chunk))
        return results

    def _forward_chunk(self, items: list[tuple[str, str]]) -> list[PaperClassificationPrediction]:
        texts = [_build_input_text(title, abstract) for title, abstract in items]
        enc = safe_tokenize(
            self.tokenizer,
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        model_inputs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        with torch.no_grad():
            out = self.model(**model_inputs)

        l1_probs = torch.softmax(out["l1_logits"], dim=-1)
        l2_probs = torch.softmax(out["l2_logits"], dim=-1)
        pt_probs = torch.softmax(out["paper_type_logits"] / self.paper_type_temperature, dim=-1)

        results: list[PaperClassificationPrediction] = []
        for i in range(len(items)):
            results.append(
                PaperClassificationPrediction(
                    oecd_l1=self._label(self.l1_classes, l1_probs[i]),
                    oecd_l1_score=float(l1_probs[i].max()),
                    oecd_l2=self._label(self.l2_classes, l2_probs[i]),
                    oecd_l2_score=float(l2_probs[i].max()),
                    paper_type=self._label(self.paper_type_classes, pt_probs[i]),
                    paper_type_score=float(pt_probs[i].max()),
                )
            )
        return results

    @staticmethod
    def _label(classes: list[str], probs) -> str:
        """Argmax label, or "" when the head has no class space."""
        if not classes:
            return ""
        return classes[int(probs.argmax())]
