"""Torch-free pieces shared by the paper classifier's runtimes.

The ``title_abstract_v1`` input template, the prediction dataclass and the
runtime loader are identical for the torch class
(:mod:`bibr.structure.paper_classifier_model`) and the ONNX class
(:mod:`bibr.structure.paper_classifier_onnx`).

The input text is built byte-identically to bibr-training's
``title_abstract_v1`` template (``build_input_text`` in
``bibr-training/src/bibr_training/paper_classifier/dataset.py``): the cleaned,
whitespace-collapsed title and abstract joined by ``" [SEP] "``, dropping
either part when empty.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Byte-identical to bibr-training's dataset.TEXT_SEPARATOR (line 18).
_TEXT_SEPARATOR = " [SEP] "
# The marker inside _TEXT_SEPARATOR: the tokenizer has to give it back as a
# single id (see utils.onnx_tokenizer.require_added_token).
TEMPLATE_SEPARATOR = _TEXT_SEPARATOR.strip()
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


def load_paper_classifier(
    model_id: str, revision: str = "main", device: str | None = None, settings=None
):
    """Load the paper classifier on the runtime ``ML_RUNTIME`` selects.

    Returns an object exposing ``classify_batch(items)`` —
    :class:`~bibr.structure.paper_classifier_onnx.OnnxPaperClassifierModel` or
    :class:`~bibr.structure.paper_classifier_model.PaperClassifierModel`.
    Raises :class:`~bibr.exceptions.ConfigurationError` when neither runtime
    is available.
    """
    from bibr.utils.ml_runtime import find_onnx_bundle, hub_bundle_hint, resolve_runtime

    if settings is None:
        from bibr.config import snapshot_settings

        settings = snapshot_settings()
    runtime, bundle = resolve_runtime(
        "paper classifier",
        settings=settings,
        bundle=lambda: find_onnx_bundle(model_id, revision, label="paper classifier"),
        bundle_hint=hub_bundle_hint("ML_PAPER_CLASSIFIER_MODEL_ID", model_id, revision),
    )
    if runtime == "onnx":
        from bibr.structure.paper_classifier_onnx import OnnxPaperClassifierModel

        logger.info("Paper classifier runtime: onnx (%s)", bundle)
        return OnnxPaperClassifierModel(bundle, device=device)

    from bibr.structure.paper_classifier_model import PaperClassifierModel

    logger.info("Paper classifier runtime: torch (%s@%s)", model_id, revision)
    return PaperClassifierModel.from_pretrained(model_id, revision=revision, device=device)
