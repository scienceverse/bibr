"""Geometry line tagger: a sequence model over the geometry segmenter's line features.

An opt-in tier of the ``geom`` reference segmentation strategy, off by default
(``REF_GEOM_TAGGER_MODEL_ID`` unset). When it is set, the geom tier asks the
tagger first. The tagger reads the whole reference section as one sequence of
printed lines, the same 29 features per line the shipped GBM reads
(:func:`~bibr.extract.geom_features.line_features` plus
:func:`~bibr.extract.geom_adjacent_features.augment_adjacent_features`, in sorted
key order), and gives each line a probability of opening a reference. Its gate
decides whether that answer is taken; when the gate declines, the geom tier
goes on exactly as without the tagger (the GBM, then the fallback chain).

The gate comes with the model and is fitted on its development papers:

1. the line logits are divided by a temperature;
2. the paper score is the expected entry precision,
   ``1 - 2 * sum(min(p, 1 - p)) / predicted entries`` (the first line counts
   as an opening and as certain);
3. an isotonic map (its breakpoints, read with linear interpolation) turns the
   score into a calibrated confidence;
4. the answer is taken when that confidence reaches the gate's threshold and
   the geom tier's own checks (alignment yield, segment count against the
   layout's reference onsets) pass.

A bundle is a directory holding ``gate.json`` and the exported graph
``tagger.onnx`` with its manifest ``tagger.json``, either beside ``gate.json``
or in an ``onnx/`` subdirectory. The graph takes ``features`` (lines x 29,
float32, raw values: the standardisation is inside it) and returns ``logits``
(lines x 2). Only onnxruntime is needed.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bibr.exceptions import ConfigurationError
from bibr.extract.anchor_snap import align_line_starts, starts_to_spans
from bibr.extract.geom_adjacent_features import ADJACENT_FEATURE_KEYS, augment_adjacent_features
from bibr.extract.geom_features import PROD_FEATURE_KEYS, line_features
from bibr.ocr.ref_geometry import LineRecord, records_from_dicts

logger = logging.getLogger(__name__)

TAGGER_ONNX = "tagger.onnx"
TAGGER_MANIFEST = "tagger.json"
GATE_FILE = "gate.json"
GATE_SCORE = "expected_entry_precision"
INPUT_NAME = "features"
OUTPUT_NAME = "logits"
# The order the tagger was trained with: sorted names, as the GBM's DictVectorizer orders them.
FEATURE_KEYS: tuple[str, ...] = tuple(sorted({*PROD_FEATURE_KEYS, *ADJACENT_FEATURE_KEYS}))


def tagger_bundle_files(path: str | Path) -> tuple[Path, Path, Path]:
    """``(tagger.onnx, tagger.json, gate.json)`` of the bundle at ``path``.

    Raises :class:`ConfigurationError` naming what is missing.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        raise ConfigurationError(
            f"geometry line tagger {str(path)!r} is not a directory; it must hold {GATE_FILE} "
            f"and {TAGGER_ONNX} with {TAGGER_MANIFEST} (beside it or in onnx/)"
        )
    gate = root / GATE_FILE
    graph_dir = next(
        (d for d in (root, root / "onnx") if (d / TAGGER_ONNX).is_file()),
        None,
    )
    missing = [GATE_FILE] if not gate.is_file() else []
    if graph_dir is None:
        missing.append(TAGGER_ONNX)
    elif not (graph_dir / TAGGER_MANIFEST).is_file():
        missing.append(TAGGER_MANIFEST)
    if missing or graph_dir is None:
        raise ConfigurationError(
            f"geometry line tagger {str(path)!r} lacks {', '.join(missing)} "
            f"({TAGGER_ONNX} and {TAGGER_MANIFEST} may sit in an onnx/ subdirectory)"
        )
    return graph_dir / TAGGER_ONNX, graph_dir / TAGGER_MANIFEST, gate


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"geometry line tagger file {path} is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"geometry line tagger file {path} must hold a JSON object")
    return data


def _check_manifest(manifest: dict[str, Any], path: Path) -> None:
    keys = manifest.get("feature_keys")
    if manifest.get("input") != INPUT_NAME or not isinstance(keys, list):
        raise ConfigurationError(
            f"{path} does not describe a geometry line tagger (want input {INPUT_NAME!r} "
            "and a feature_keys list)"
        )
    if tuple(keys) != FEATURE_KEYS:
        raise ConfigurationError(
            f"{path}: the tagger's features differ from the ones this bibr computes "
            f"(tagger {keys}, bibr {list(FEATURE_KEYS)})"
        )
    outputs = manifest.get("outputs") or {}
    if OUTPUT_NAME not in outputs:
        raise ConfigurationError(f"{path}: the tagger has no {OUTPUT_NAME!r} output")


def _check_gate(gate: dict[str, Any], path: Path) -> tuple[float, np.ndarray, np.ndarray, float]:
    try:
        temperature = float(gate["temperature"])
        threshold = float(gate["threshold"])
        xs = np.asarray(gate["isotonic_x"], dtype=np.float64)
        ys = np.asarray(gate["isotonic_y"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{path}: gate needs temperature, threshold, isotonic_x and isotonic_y ({exc!r})"
        ) from exc
    score = gate.get("score", GATE_SCORE)
    problems = []
    if score != GATE_SCORE:
        problems.append(f"score {score!r} (this bibr computes {GATE_SCORE!r})")
    if not temperature > 0:
        problems.append(f"temperature {temperature}")
    if not 0.0 <= threshold <= 1.0:
        problems.append(f"threshold {threshold}")
    if xs.ndim != 1 or xs.shape != ys.shape or len(xs) < 2 or np.any(np.diff(xs) < 0):
        problems.append("isotonic_x/isotonic_y must be equal-length increasing breakpoints")
    if problems:
        raise ConfigurationError(f"{path}: unusable gate: {'; '.join(problems)}")
    return temperature, xs, ys, threshold


def expected_entry_precision(probs: np.ndarray) -> float:
    """The gate's paper score; the first line opens the first reference and counts as certain."""
    p = np.asarray(probs, dtype=np.float64)
    if not len(p):
        return 0.0
    starts = p >= 0.5
    starts[0] = True
    wrong = np.minimum(p, 1.0 - p)
    wrong[0] = 0.0
    return float(max(0.0, 1.0 - 2.0 * wrong.sum() / max(1, int(starts.sum()))))


def feature_matrix(lines: list[LineRecord]) -> np.ndarray:
    """The tagger's input: one row of :data:`FEATURE_KEYS` per line, float32."""
    rows = augment_adjacent_features(line_features(lines), lines)
    return np.asarray([[float(r[k]) for k in FEATURE_KEYS] for r in rows], dtype=np.float32)


class GeomLineTagger:
    """Loads a tagger bundle and segments reference lines; safe to share across threads."""

    def __init__(self, path: str | Path) -> None:
        self.onnx_path, manifest_path, gate_path = tagger_bundle_files(path)
        _check_manifest(_read_json(manifest_path), manifest_path)
        self.temperature, self._iso_x, self._iso_y, self.threshold = _check_gate(
            _read_json(gate_path), gate_path
        )
        self.files: tuple[Path, ...] = (self.onnx_path, manifest_path, gate_path)
        from bibr.utils.onnx_providers import create_session

        self._session, _device = create_session(
            self.onnx_path, device="cpu", model_name="geometry line tagger"
        )
        inputs = [i.name for i in self._session.get_inputs()]
        outputs = [o.name for o in self._session.get_outputs()]
        if INPUT_NAME not in inputs or OUTPUT_NAME not in outputs:
            raise ConfigurationError(
                f"{self.onnx_path}: graph inputs {inputs} / outputs {outputs}; want "
                f"{INPUT_NAME!r} in and {OUTPUT_NAME!r} out"
            )
        self._lock = threading.Lock()

    def line_probabilities(self, lines: list[LineRecord]) -> np.ndarray:
        """Tempered probability that each line opens a reference."""
        if not lines:
            return np.zeros(0, dtype=np.float64)
        x = feature_matrix(lines)
        with self._lock:
            logits = self._session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
        z = np.asarray(logits, dtype=np.float64) / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e[:, 1] / e.sum(axis=1)

    def confidence(self, probs: np.ndarray) -> float:
        """Calibrated paper confidence, compared with :attr:`threshold`."""
        return float(np.interp(expected_entry_precision(probs), self._iso_x, self._iso_y))

    def segment_spans(
        self, ref_text: str, line_dicts: list[dict]
    ) -> tuple[list[tuple[int, int]], float, int, int]:
        """As :meth:`GeomSegmenter.segment_spans`: spans over *ref_text*, the
        calibrated confidence, the lines labelled as openings, and how many of
        them aligned onto *ref_text*."""
        lines = records_from_dicts(line_dicts)
        if not lines:
            return [], 0.0, 0, 0
        probs = self.line_probabilities(lines)
        is_start = [bool(p >= 0.5) for p in probs]
        # The first line opens the first reference, as in the GBM tier.
        is_start[0] = True
        starts = align_line_starts(ref_text, [ln.text for ln in lines], is_start)
        return starts_to_spans(ref_text, starts), self.confidence(probs), sum(is_start), len(starts)


__all__ = [
    "FEATURE_KEYS",
    "GeomLineTagger",
    "expected_entry_precision",
    "feature_matrix",
    "tagger_bundle_files",
]
