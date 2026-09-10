"""HF Hub loader + inference for the trained MiniLM-L6 section classifier."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from bibr.utils.ml_extra import ml_import_error

try:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise ml_import_error("ML section classification") from e

from huggingface_hub import constants as hf_constants
from huggingface_hub import snapshot_download

from bibr.paper_contents import CanonicalSection
from bibr.structure._section_minilm_arch import SectionMiniLMModel
from bibr.utils.tokenizer_safety import safe_tokenize


@dataclass
class SectionPrediction:
    canonical_type: CanonicalSection
    is_top_level: bool
    score: float


@dataclass
class HeaderContext:
    """Section-header context fed into the classifier's input template.

    ``relative_position``/``prev_heading``/``next_heading`` are only used by
    the v3 template; v2 ignores them entirely.
    """

    heading: str
    body: str
    relative_position: float = 0.5
    prev_heading: str = ""
    next_heading: str = ""


# Position buckets — byte-identical to bibr_training.data.section_template.
_POSITION_BUCKETS = [(0.10, "start"), (0.35, "early"), (0.65, "middle"), (0.85, "late")]


def _position_bucket(rel_pos: float) -> str:
    for threshold, name in _POSITION_BUCKETS:
        if rel_pos < threshold:
            return name
    return "end"


def _build_input_text(ctx: HeaderContext, template_version: int) -> str:
    """Build the model input text — byte-identical to the training template.

    v2: ``f"{heading} [SEP] {body[:1500]}"`` (unchanged legacy behavior).
    v3: ``f"{heading} [SEP] pos={bucket} | prev={prev} | next={next} [SEP] {body[:1500]}"``.
    """
    heading = ctx.heading
    body = ctx.body[:1500]
    if template_version == 2:
        return f"{heading} [SEP] {body}"
    prev = ctx.prev_heading or "-"
    nxt = ctx.next_heading or "-"
    bucket = _position_bucket(ctx.relative_position)
    return f"{heading} [SEP] pos={bucket} | prev={prev} | next={nxt} [SEP] {body}"


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


# Cap forward-pass batch size to avoid unstable attention/softmax results with very large MPS
# batches. Apply the same bound on every device for consistent chunking.
_MAX_INFERENCE_BATCH = 64
_WINDOWS_SYMLINK_PRIVILEGE_ERROR = 1314


def _disable_hf_cache_symlinks_on_windows() -> None:
    if sys.platform != "win32":
        return
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    # huggingface_hub reads env vars at import time; patch the loaded constant
    # too because Gradio/Transformers may have imported it before bibr.
    hf_constants.HF_HUB_DISABLE_SYMLINKS = True


def _is_windows_symlink_privilege_error(error: OSError) -> bool:
    return sys.platform == "win32" and (
        getattr(error, "winerror", None) == _WINDOWS_SYMLINK_PRIVILEGE_ERROR
        or f"[WinError {_WINDOWS_SYMLINK_PRIVILEGE_ERROR}]" in str(error)
    )


def _download_snapshot(repo_id: str, revision: str) -> Path:
    _disable_hf_cache_symlinks_on_windows()
    try:
        return Path(snapshot_download(repo_id, revision=revision))
    except OSError as e:
        if not _is_windows_symlink_privilege_error(e):
            raise
        _disable_hf_cache_symlinks_on_windows()
        return Path(snapshot_download(repo_id, revision=revision, max_workers=1))


class SectionClassifierModel:
    """Wraps the trained two-head MiniLM section classifier for inference.

    Defaults to CUDA → CPU. ``classify_batch`` chunks input to
    ``_MAX_INFERENCE_BATCH`` per forward pass — required for MPS correctness
    on large batches (see module-level comment), no-op for CUDA/CPU.
    """

    def __init__(self, model_dir: Path, device: str | None = None) -> None:
        self.device = device or _pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        with (model_dir / "label_classes.json").open() as fh:
            self.label_classes: list[str] = json.load(fh)
        self.template_version = 2
        inference_config_path = model_dir / "inference_config.json"
        if inference_config_path.exists():
            with inference_config_path.open() as fh:
                inference_config = json.load(fh)
            self.template_version = inference_config.get("template_version", 2)
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


def _coerce_header_context(item: HeaderContext | tuple[str, str]) -> HeaderContext:
    if isinstance(item, HeaderContext):
        return item
    heading = getattr(item, "heading", None)
    body = getattr(item, "body", None)
    if heading is not None and body is not None:
        return HeaderContext(
            heading=heading,
            body=body,
            relative_position=getattr(item, "relative_position", 0.5),
            prev_heading=getattr(item, "prev_heading", "") or "",
            next_heading=getattr(item, "next_heading", "") or "",
        )
    return HeaderContext(heading=item[0], body=item[1])
