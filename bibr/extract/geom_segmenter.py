"""Local geometry reference segmenter (the ``geom`` strategy).

Loads a frozen GBM bundle (public HF repo, via the offline-resilient
``resolve_checkpoint``, in the pickle-free ``bibr-gbm/1`` form of
:mod:`bibr.utils.gbm_bundle`), labels each
reference-section line B-REF/I-REF from
its geometry+text features, and aligns the ordered line sequence onto the
reference text (``align_line_starts`` — a monotonic cursor walk, so an
opening embedded in an earlier reference can never steal a boundary).
``segment_spans`` returns per-reference character spans over ``ref_text``, a
per-paper confidence (mean decision margin), the number of lines the GBM
labeled B-REF, and how many of those boundary lines survived alignment onto
``ref_text``. The caller compares confidence against
``REF_GEOM_SEG_CASCADE_THRESHOLD`` and the aligned/labeled ratio against
``REF_GEOM_MIN_ALIGN_YIELD`` to decide whether to cascade — see
``REF_GEOM_MIN_ALIGN_YIELD`` in ``bibr/config.py`` for the OOD rationale;
``segment`` derives the substrings from those spans.
"""

from __future__ import annotations

import logging

from bibr.extract.anchor_snap import align_line_starts, starts_to_spans
from bibr.extract.geom_adjacent_features import (
    ADJACENT_FEATURE_KEYS,
    augment_adjacent_features,
)
from bibr.extract.geom_features import PROD_FEATURE_KEYS, line_features
from bibr.ocr.ref_geometry import LineRecord, records_from_dicts
from bibr.ocr.ref_patterns import _OPEN_AUTHOR_LIST

logger = logging.getLogger(__name__)

# The bundle's file stem: geom_segmenter.json + .npz (bibr-gbm/1,
# bibr.utils.gbm_bundle.load_gbm_bundle).
_BUNDLE_STEM = "geom_segmenter"
MODEL_KIND_LINE_START_V1 = "line_start_v1"
MODEL_KIND_ADJACENT_BOUNDARY_V2 = "adjacent_boundary_v2"


def _feature_rows_for_kind(lines: list[LineRecord], model_kind: str) -> list[dict]:
    rows = line_features(lines)
    if model_kind == MODEL_KIND_LINE_START_V1:
        return rows
    if model_kind == MODEL_KIND_ADJACENT_BOUNDARY_V2:
        return augment_adjacent_features(rows, lines)
    raise ValueError(f"unknown geom model_kind: {model_kind}")


def _expected_feature_keys(model_kind: str) -> set[str]:
    if model_kind == MODEL_KIND_LINE_START_V1:
        return set(PROD_FEATURE_KEYS)
    if model_kind == MODEL_KIND_ADJACENT_BOUNDARY_V2:
        return {*PROD_FEATURE_KEYS, *ADJACENT_FEATURE_KEYS}
    raise ValueError(f"unknown geom model_kind: {model_kind}")


class GeomSegmenter:
    def __init__(self, model_spec: str, revision: str | None = None) -> None:
        from bibr.utils.gbm_bundle import load_gbm_bundle

        # A pickle-free bibr-gbm/1 bundle, read with numpy: no code in the file
        # runs on load (audit M6).
        path, bundle = load_gbm_bundle(model_spec, revision, stem=_BUNDLE_STEM)
        self.path = path
        feature_keys = list(bundle["feature_keys"])
        self.model_kind = str(bundle.get("model_kind", MODEL_KIND_LINE_START_V1))
        expected_keys = _expected_feature_keys(self.model_kind)
        if set(feature_keys) != expected_keys:
            raise ValueError(
                "geom bundle feature_keys do not match production feature keys "
                f"for model_kind={self.model_kind!r} "
                f"(bundle={sorted(feature_keys)}, prod={sorted(expected_keys)})"
            )
        self.model = bundle["model"]
        self.vectorizer = bundle["vectorizer"]
        self.feature_keys = feature_keys
        self.threshold = float(bundle.get("cascade_threshold", 0.0))
        self.boundary_f1_val = float(bundle.get("boundary_f1_val", 0.0))

    def _predict(self, lines: list[LineRecord]) -> tuple[list[str], list[float]]:
        rows = _feature_rows_for_kind(lines, self.model_kind)
        x = self.vectorizer.transform([{k: r[k] for k in self.feature_keys} for r in rows])
        probs = [float(p) for p in self.model.predict_proba(x)[:, 1]]
        labels = ["B-REF" if p >= 0.5 else "I-REF" for p in probs]
        for i in range(1, len(labels)):
            # The last author of a list wrapped at its "&" reads like a one-author
            # entry ("Ibarra, J. P. A. (2017). ..."), and would take the date,
            # title and DOI from the authors before it.
            if _OPEN_AUTHOR_LIST.search(lines[i - 1].text):
                labels[i] = "I-REF"
        if labels:
            # The first region line always opens the first reference — it cannot
            # be a *continuation* of a prior entry (there is none). The model can
            # mislabel a dense/firstname-first/numbered lead line I-REF while a
            # later line wins B-REF; without this the genuine leading reference
            # is dropped by anchor snapping (D2). probs (confidence) is left as
            # the model emitted it.
            labels[0] = "B-REF"
        return labels, probs

    @staticmethod
    def _confidence(probs: list[float]) -> float:
        """Per-paper certainty = mean decision margin, scaled to [0, 1]."""
        if not probs:
            return 0.0
        return sum(2.0 * abs(p - 0.5) for p in probs) / len(probs)

    def segment_spans(
        self, ref_text: str, line_dicts: list[dict]
    ) -> tuple[list[tuple[int, int]], float, int, int]:
        """Per-reference (start, end) character spans over *ref_text*, confidence,
        the number of lines labeled B-REF, and how many of those boundary lines
        survived alignment (``len(starts)`` -- see ``REF_GEOM_MIN_ALIGN_YIELD``)."""
        lines = records_from_dicts(line_dicts)
        if not lines:
            return [], 0.0, 0, 0
        labels, probs = self._predict(lines)
        labeled = labels.count("B-REF")
        starts = align_line_starts(
            ref_text, [ln.text for ln in lines], [lab == "B-REF" for lab in labels]
        )
        return starts_to_spans(ref_text, starts), self._confidence(probs), labeled, len(starts)

    def line_start_probabilities(self, line_dicts: list[dict]) -> list[float]:
        """Per-line probability that each line opens a reference, unaligned.

        The reference line stream reads the model's line evidence directly,
        with no alignment onto row text and no per-paper gate.
        """
        lines = records_from_dicts(line_dicts)
        if not lines:
            return []
        return self._predict(lines)[1]

    def segment(self, ref_text: str, line_dicts: list[dict]) -> tuple[list[str], float]:
        spans, confidence, _labeled, _aligned = self.segment_spans(ref_text, line_dicts)
        return [ref_text[s:e] for s, e in spans], confidence
