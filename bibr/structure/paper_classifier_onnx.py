"""ONNX Runtime inference for the trained multitask paper classifier.

Same ``classify_batch`` surface as
:class:`bibr.structure.paper_classifier_model.PaperClassifierModel`; the label
maps, input length and paper_type temperature come from the bundle manifest.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from bibr.structure.paper_classifier_common import (
    _MAX_INFERENCE_BATCH,
    TEMPLATE_SEPARATOR,
    PaperClassificationPrediction,
    _build_input_text,
)
from bibr.utils.ml_runtime import ONNX_MODEL, read_onnx_manifest
from bibr.utils.onnx_tokenizer import encode_batch, load_tokenizer, require_added_token

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


class OnnxPaperClassifierModel:
    """Wraps the exported OECD L1/L2 + paper_type classifier for ONNX Runtime."""

    runtime = "onnx"

    def __init__(self, bundle_dir: str | Path, device: str | None = None) -> None:
        from bibr.utils.onnx_providers import create_session

        self.bundle_dir = Path(bundle_dir)
        self.manifest = read_onnx_manifest(self.bundle_dir)
        self.l1_classes: list[str] = [str(x) for x in self.manifest["l1_classes"]]
        self.l2_classes: list[str] = [str(x) for x in self.manifest.get("l2_classes", [])]
        self.paper_type_classes: list[str] = [
            str(x) for x in self.manifest.get("paper_type_classes", [])
        ]
        self.max_length = int(self.manifest.get("max_length", 256))
        # Temperature scaling for the paper_type head (Guo et al. 2017); 1.0 is a no-op.
        self.paper_type_temperature = float(self.manifest.get("paper_type_temperature", 1.0))
        self.tokenizer = load_tokenizer(self.bundle_dir, self.manifest)
        require_added_token(self.tokenizer, TEMPLATE_SEPARATOR, label="paper classifier")
        self.session, self.device = create_session(
            self.bundle_dir / self.manifest.get("model_file", ONNX_MODEL),
            device=device,
            model_name="paper classifier (ONNX)",
        )
        self._input_names = [i.name for i in self.session.get_inputs()]

    def check_graph(self) -> None:
        """Run the graph on titles and abstracts, in two shapes ``classify_batch`` sends.

        The load probe of an overridden classifier (:mod:`bibr.utils.onnx_probe`):
        titles and abstracts through the bundle's template and tokenizer at the
        manifest's ``max_length``, as a batch of two padded to the longest, which
        fills the length and holds the tokenizer's last id, and as one paper
        alone, cut short (:func:`~bibr.utils.onnx_probe.cut_short`): a run
        classifies its papers in batches of any size, padded to their longest.
        On both the graph must give finite logits for each head, one per class
        the manifest lists for it, and the paper-type temperature must be a
        positive number. Raises :class:`~bibr.exceptions.ConfigurationError`.
        """
        import math

        from bibr.exceptions import ConfigurationError
        from bibr.utils.onnx_probe import (
            PROBE_BODY,
            check_output,
            cut_short,
            run_probe,
            with_last_token,
        )

        if not (math.isfinite(self.paper_type_temperature) and self.paper_type_temperature > 0):
            raise ConfigurationError(
                f"{self.bundle_dir}: the manifest's paper_type_temperature "
                f"{self.paper_type_temperature!r} is not a positive number"
            )
        full = [
            ("Memory consolidation during sleep: a preregistered replication", PROBE_BODY * 8),
            ("A short note", "We comment on a recent meta-analysis."),
        ]
        short = [("Erratum", "")]
        model_path = self.bundle_dir / self.manifest.get("model_file", ONNX_MODEL)
        for items in (full, short):
            texts = [_build_input_text(title, abstract) for title, abstract in items]
            batch = encode_batch(self.tokenizer, texts, max_length=self.max_length)
            ids = with_last_token(batch.input_ids, batch.attention_mask, self.tokenizer)
            mask = batch.attention_mask
            if items is short:
                ids, mask = cut_short(ids, mask, self.max_length)
            feeds = {"input_ids": ids, "attention_mask": mask}
            feeds = {k: v for k, v in feeds.items() if k in self._input_names}
            shape = f"a {ids.shape} batch of titles and abstracts"
            outputs = run_probe(
                model_path,
                shape,
                lambda feeds=feeds: self.session.run(
                    ["l1_logits", "l2_logits", "paper_type_logits"], feeds
                ),
            )
            for name, value, classes, key in zip(
                ("l1_logits", "l2_logits", "paper_type_logits"),
                outputs,
                (self.l1_classes, self.l2_classes, self.paper_type_classes),
                ("l1_classes", "l2_classes", "paper_type_classes"),
                strict=True,
            ):
                # A head without a class space is read as no answer, whatever its width.
                check_output(
                    model_path,
                    name,
                    value,
                    (len(items), len(classes) if classes else None),
                    shape,
                    f"one per class of the manifest's {len(classes)} {key}",
                )

    def classify_batch(self, items: list[tuple[str, str]]) -> list[PaperClassificationPrediction]:
        results: list[PaperClassificationPrediction] = []
        for start in range(0, len(items), _MAX_INFERENCE_BATCH):
            chunk = items[start : start + _MAX_INFERENCE_BATCH]
            results.extend(self._forward_chunk(chunk))
        return results

    def _forward_chunk(self, items: list[tuple[str, str]]) -> list[PaperClassificationPrediction]:
        if not items:
            return []
        texts = [_build_input_text(title, abstract) for title, abstract in items]
        batch = encode_batch(self.tokenizer, texts, max_length=self.max_length)
        feeds = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        l1_logits, l2_logits, pt_logits = self.session.run(
            ["l1_logits", "l2_logits", "paper_type_logits"], feeds
        )
        l1_probs = _softmax(np.asarray(l1_logits, dtype=np.float32))
        l2_probs = _softmax(np.asarray(l2_logits, dtype=np.float32))
        pt_probs = _softmax(np.asarray(pt_logits, dtype=np.float32) / self.paper_type_temperature)

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
    def _label(classes: list[str], probs: np.ndarray) -> str:
        """Argmax label, or "" when the head has no class space."""
        if not classes:
            return ""
        return classes[int(probs.argmax())]
