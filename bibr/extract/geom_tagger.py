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
or in an ``onnx/`` subdirectory, or a Hub repo laid out the same way (graph and
manifest in ``onnx/``), fetched at ``REF_GEOM_TAGGER_REVISION``. The graph
takes ``features`` (lines x 29, float32, raw values: the standardisation is
inside it) and returns ``logits`` (lines x 2); loading runs it once on the
features of a short reference section to check that. Only onnxruntime is needed.

A declined answer has no effect on the references: the paper goes on exactly
as without the tagger. It is not invisible: the receipt lists the attempt, and
the export carries it in ``reference_yield.attempts``. An error on a paper is
also a ``MODEL_OVERRIDE_INFERENCE_FAILED`` warning. An accepted answer is
final: the line stream does not run over it.
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


def _probe_line(text: str, row: int, indent: float) -> dict[str, Any]:
    top = 700.0 - 12.0 * row
    return {
        "text": text,
        "page": 9,
        "x0": 72.0 + indent,
        "y_top": top,
        "x1": 523.0,
        "y_bottom": top - 9.5,
        "font_size": 9.0,
    }


# What the load probe reads: three hanging-indent entries of a reference section.
_PROBE_LINES: list[dict[str, Any]] = [
    _probe_line(text, row, indent)
    for row, (text, indent) in enumerate(
        [
            ("Smith, J. A., & Müller, K. (2019). A title of the first work that", 0.0),
            ("runs on to a second line. Journal of Examples, 12(3), 45–67.", 12.0),
            ("Lee, S., & García-López, M. (2020). Second title. Proceedings of the", 0.0),
            ("National Academy of Sciences, 117, 1–9. https://doi.org/10.1000/xyz", 12.0),
            ("Ōtsuka, H. (2021). Third title. Example Press.", 0.0),
        ]
    )
]


def tagger_bundle_files(path: str | Path, revision: str | None = None) -> tuple[Path, Path, Path]:
    """``(tagger.onnx, tagger.json, gate.json)`` of the bundle ``path`` names.

    ``path`` is a local directory, or a Hub repo id (``org/name``) fetched at
    ``revision`` the way the other ONNX bundles are: ``gate.json``, then
    ``onnx/tagger.json`` and ``onnx/tagger.onnx`` (the cache first at a pinned
    commit). A Hub-shaped id is never read from the working directory; write a
    local copy as a path (``./org/name``). Raises :class:`ConfigurationError`
    naming what is missing; a Hub error (a missing repo, revision or file, or
    the network) is raised as the Hub client raised it.
    """
    from bibr.utils.ml_runtime import is_hub_repo_id

    text = str(path)
    if is_hub_repo_id(text):
        return _hub_bundle_files(text, revision)
    root = Path(text).expanduser()
    if not root.is_dir():
        raise ConfigurationError(
            f"geometry line tagger {text!r} is not a directory; it must hold {GATE_FILE} "
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
            f"geometry line tagger {text!r} lacks {', '.join(missing)} "
            f"({TAGGER_ONNX} and {TAGGER_MANIFEST} may sit in an onnx/ subdirectory)"
        )
    return graph_dir / TAGGER_ONNX, graph_dir / TAGGER_MANIFEST, gate


def _hub_bundle_files(repo_id: str, revision: str | None) -> tuple[Path, Path, Path]:
    from bibr.utils.hf_cache import hf_download_or_cached

    gate = Path(hf_download_or_cached(repo_id, GATE_FILE, revision))
    manifest = Path(hf_download_or_cached(repo_id, f"onnx/{TAGGER_MANIFEST}", revision))
    graph = Path(hf_download_or_cached(repo_id, f"onnx/{TAGGER_ONNX}", revision))
    return graph, manifest, gate


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


def _finite(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)))


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
    if not (np.isfinite(temperature) and temperature > 0):
        problems.append(f"temperature {temperature} (want a finite number above 0)")
    if not (np.isfinite(threshold) and 0.0 < threshold <= 1.0):
        problems.append(f"threshold {threshold} (want a finite number in (0, 1])")
    if xs.ndim != 1 or ys.ndim != 1 or xs.shape != ys.shape or len(xs) < 2:
        problems.append("isotonic_x and isotonic_y must be equal-length lists of 2 or more")
    elif not (_finite(xs) and _finite(ys)):
        problems.append("isotonic breakpoints must be finite numbers")
    else:
        if np.any(np.diff(xs) < 0):
            problems.append("isotonic_x must be non-decreasing")
        if np.any(np.diff(ys) < 0):
            problems.append("isotonic_y must be non-decreasing")
        if np.any(ys < 0.0) or np.any(ys > 1.0):
            problems.append("isotonic_y must lie within [0, 1]")
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

    def __init__(self, path: str | Path, revision: str | None = None) -> None:
        self.onnx_path, manifest_path, gate_path = tagger_bundle_files(path, revision)
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
        self._check_graph()

    def _check_graph(self) -> None:
        """Run the graph on two reference sections of different lengths, as papers send them.

        The features of :data:`_PROBE_LINES` (hanging-indent entries on one page)
        and of its first two lines go through the graph: a paper's reference
        section is one call of as many rows as it has lines. On both the graph
        must give one finite logit pair per line.
        """
        for lines in (_PROBE_LINES, _PROBE_LINES[:2]):
            probe = feature_matrix(records_from_dicts(lines))
            shape = f"({len(probe)}, {len(FEATURE_KEYS)}) feature matrix of a reference section"
            try:
                out = np.asarray(self._session.run([OUTPUT_NAME], {INPUT_NAME: probe})[0])
            except Exception as exc:
                raise ConfigurationError(
                    f"{self.onnx_path}: the graph does not run on a {shape} "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
            if out.shape != (len(probe), 2):
                raise ConfigurationError(
                    f"{self.onnx_path}: the graph returns {OUTPUT_NAME} of shape {out.shape} "
                    f"for a {shape}; want ({len(probe)}, 2), one logit per class (inside, opens)"
                )
            if not np.isfinite(out).all():
                raise ConfigurationError(
                    f"{self.onnx_path}: the graph returns non-finite {OUTPUT_NAME} for a {shape}"
                )

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
        """Calibrated paper confidence, compared with :attr:`threshold` (NaN never passes)."""
        return float(np.interp(expected_entry_precision(probs), self._iso_x, self._iso_y))

    def accepts(self, confidence: float) -> bool:
        """Whether the gate takes an answer of this confidence; NaN is declined."""
        return bool(confidence >= self.threshold)

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
