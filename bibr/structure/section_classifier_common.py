"""Torch-free pieces shared by the section classifier's runtimes.

The input template, position buckets, prediction/context dataclasses and
bundle resolution are identical for the torch class
(:mod:`bibr.structure.section_classifier_model`) and the ONNX class
(:mod:`bibr.structure.section_classifier_onnx`); keeping them here means the
core install can build classifier inputs and select a runtime without
importing torch or transformers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from bibr.paper_contents import CanonicalSection
from bibr.utils.hf_cache import (
    disable_hf_cache_symlinks_on_windows,
    is_windows_symlink_privilege_error,
)

logger = logging.getLogger(__name__)


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


# The templates below embed this marker in the input *text*; the tokenizer has
# to give it back as a single id (see utils.onnx_tokenizer.require_added_token).
TEMPLATE_SEPARATOR = "[SEP]"


def _build_input_text(ctx: HeaderContext, template_version: int) -> str:
    """Build the model input text — byte-identical to the training template.

    v2: ``f"{heading} [SEP] {body[:1500]}"`` (unchanged legacy behavior).
    v3: ``f"{heading} [SEP] pos={bucket} | prev={prev} | next={next} [SEP] {body[:1500]}"``.
    """
    heading = ctx.heading
    body = ctx.body[:1500]
    sep = TEMPLATE_SEPARATOR
    if template_version == 2:
        return f"{heading} {sep} {body}"
    prev = ctx.prev_heading or "-"
    nxt = ctx.next_heading or "-"
    bucket = _position_bucket(ctx.relative_position)
    return f"{heading} {sep} pos={bucket} | prev={prev} | next={nxt} {sep} {body}"


# Cap forward-pass batch size. Required for MPS correctness — diagnosed
# 2026-05-13 in ``scripts/mps_section_diagnostic.py``: on the 1217-row held-out
# test split, MPS at batch_size=1217 collapses (macro F1 0.83 → 0.25,
# non-deterministic at 53 % self-agreement, logit L1 max ≈ 1.4). At
# batch_size=64 MPS matches CPU to float32 noise. The failure is in MPS
# attention/softmax with very large batches, not the model. CUDA/CPU are
# unaffected and the per-chunk overhead is negligible, so we cap unconditionally.
_MAX_INFERENCE_BATCH = 64


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


def _download_snapshot(repo_id: str, revision: str) -> Path:
    """Materialise a Hub repo snapshot (or pass an existing local directory through)."""
    local = Path(repo_id).expanduser()
    if local.is_dir():
        return local
    disable_hf_cache_symlinks_on_windows()
    try:
        return Path(snapshot_download(repo_id, revision=revision))
    except OSError as e:
        if not is_windows_symlink_privilege_error(e):
            raise
        disable_hf_cache_symlinks_on_windows()
        return Path(snapshot_download(repo_id, revision=revision, max_workers=1))


def load_bundle_labels(model_dir: Path) -> tuple[list[str], int]:
    """``(label_classes, template_version)`` from a torch bundle directory."""
    with (model_dir / "label_classes.json").open() as fh:
        label_classes: list[str] = json.load(fh)
    template_version = 2
    inference_config_path = model_dir / "inference_config.json"
    if inference_config_path.exists():
        with inference_config_path.open() as fh:
            template_version = json.load(fh).get("template_version", 2)
    return label_classes, template_version


def load_section_classifier(
    model_id: str, revision: str = "main", device: str | None = None, settings=None
):
    """Load the section classifier on the runtime ``ML_RUNTIME`` selects.

    Returns an object exposing ``classify_batch(items, max_length=256)`` —
    :class:`~bibr.structure.section_classifier_onnx.OnnxSectionClassifierModel`
    or :class:`~bibr.structure.section_classifier_model.SectionClassifierModel`.
    Raises :class:`~bibr.exceptions.ConfigurationError` when neither runtime
    is available.
    """
    from bibr.utils.ml_runtime import find_onnx_bundle, hub_bundle_hint, resolve_runtime

    if settings is None:
        from bibr.config import snapshot_settings

        settings = snapshot_settings()
    runtime, bundle = resolve_runtime(
        "section classifier",
        settings=settings,
        bundle=lambda: find_onnx_bundle(model_id, revision, label="section classifier"),
        bundle_hint=hub_bundle_hint("ML_SECTION_CLASSIFIER_MODEL_ID", model_id, revision),
    )
    if runtime == "onnx":
        from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel

        logger.info("Section classifier runtime: onnx (%s)", bundle)
        return OnnxSectionClassifierModel(bundle, device=device)

    from bibr.structure.section_classifier_model import SectionClassifierModel

    logger.info("Section classifier runtime: torch (%s@%s)", model_id, revision)
    return SectionClassifierModel.from_pretrained(model_id, revision=revision, device=device)
