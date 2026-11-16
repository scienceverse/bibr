"""Pickle-free gradient-boosted tree bundles (``bibr-gbm/1``), read without scikit-learn.

The geometry reference segmenter and the front-role classifier are scikit-learn
``HistGradientBoostingClassifier`` models over ``DictVectorizer`` features. They
used to ship as joblib pickles, which only a gadget-restricted unpickler
(:mod:`bibr.utils.safe_pickle`) and scikit-learn itself could read. A
``bibr-gbm/1`` bundle holds the same model as data:

- ``<stem>.json``, the manifest: the format name, the vectorizer's feature
  names and separator, the class labels, the loss, the bundle's own metadata
  (``feature_keys``, ``model_kind``, ``roles``, ...) and the name and SHA-256 of
  the arrays file;
- ``<stem>.npz``, the trees: every tree's nodes concatenated (feature index,
  float64 threshold, missing-value direction, children, leaf flag, leaf value),
  each tree's first node and class, and the baseline. It is read with
  ``allow_pickle=False``.

:class:`DictFeatures` and :class:`TreeEnsemble` stand in for the vectorizer and
the model with the methods the callers use (``transform``, ``predict_proba``,
``classes_``) and reproduce scikit-learn's arithmetic step for step, so the
probabilities are the same floats: a row is vectorized as ``DictVectorizer``
does it (a string value one-hot, ``None`` missing, an absent key zero), a tree
is walked as ``_predictor.pyx`` walks it (missing values follow the node's
missing direction, otherwise ``x <= threshold`` goes left), the leaf values are
added to the baseline tree by tree in training order, and the logit or softmax
link is applied as ``sklearn._loss`` applies it. Categorical splits are not
supported; ``scripts/export_gbm_bundle.py`` refuses a model that has them.

:func:`load_gbm_bundle` resolves a model setting to a bundle: the
``bibr-gbm/1`` manifest when the repo (at the pinned revision) or the local
path has one, else the joblib pickle through the restricted loader, which still
needs scikit-learn and joblib.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from numbers import Number
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

FORMAT = "bibr-gbm/1"
_ARRAYS_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.npz")
_NODE_ARRAYS = (
    "feature_idx",
    "threshold",
    "missing_go_to_left",
    "left",
    "right",
    "is_leaf",
    "value",
)


class GbmBundleError(ValueError):
    """A ``bibr-gbm/1`` bundle that cannot be read or is inconsistent."""


class DictFeatures:
    """``DictVectorizer.transform`` for a fitted, dense float64 vectorizer."""

    def __init__(self, feature_names: Sequence[str], separator: str = "=") -> None:
        self.feature_names_ = [str(n) for n in feature_names]
        self.vocabulary_ = {name: i for i, name in enumerate(self.feature_names_)}
        if len(self.vocabulary_) != len(self.feature_names_):
            raise GbmBundleError("vectorizer feature names repeat")
        self.separator = str(separator)

    def transform(self, rows: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> np.ndarray:
        rows = [rows] if isinstance(rows, Mapping) else list(rows)
        if not rows:
            raise ValueError("Sample sequence X is empty.")
        vocab = self.vocabulary_
        sep = self.separator
        out = np.zeros((len(rows), len(self.feature_names_)), dtype=np.float64)
        for i, row in enumerate(rows):
            for f, v in row.items():
                # The value rules of DictVectorizer._transform; a duplicate entry
                # adds up, as the sparse matrix it builds does on densifying.
                if isinstance(v, str):
                    j = vocab.get(f"{f}{sep}{v}")
                    if j is not None:
                        out[i, j] += np.float64(1)
                elif isinstance(v, Number) or v is None:
                    j = vocab.get(f)
                    if j is not None:
                        out[i, j] += np.float64(cast(Any, v))  # None is NaN, a missing value
                elif not isinstance(v, Mapping) and isinstance(v, Iterable):
                    for vv in v:
                        if not isinstance(vv, str):
                            raise TypeError(
                                f"Unsupported type {type(vv)} in iterable value. "
                                "Only iterables of string are supported."
                            )
                        j = vocab.get(f"{f}{sep}{vv}")
                        if j is not None:
                            out[i, j] += np.float64(1)
                else:
                    raise TypeError(
                        f"Unsupported value Type {type(v)} for {f}: {v}.\n"
                        f"{type(v)} objects are not supported."
                    )
        return out


class TreeEnsemble:
    """``HistGradientBoostingClassifier.predict_proba`` over numeric splits."""

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        classes: Sequence[int],
        loss: str,
        n_features: int,
    ) -> None:
        self.classes_ = np.asarray(classes, dtype=np.int64)
        self.n_features_in_ = int(n_features)
        self.loss = str(loss)
        baseline = np.asarray(arrays["baseline"], dtype=np.float64).ravel()
        k = baseline.shape[0]
        if self.loss == "binomial":
            if k != 1 or len(self.classes_) != 2:
                raise GbmBundleError("a binomial bundle has one baseline and two classes")
        elif self.loss == "multinomial":
            if k != len(self.classes_) or k < 3:
                raise GbmBundleError("a multinomial bundle has one baseline per class")
        else:
            raise GbmBundleError(f"unknown loss {self.loss!r}")
        self._baseline = baseline
        self._k = k

        feature = np.asarray(arrays["feature_idx"], dtype=np.int64)
        threshold = np.asarray(arrays["threshold"], dtype=np.float64)
        missing_left = np.asarray(arrays["missing_go_to_left"]).astype(bool)
        left = np.asarray(arrays["left"], dtype=np.int64)
        right = np.asarray(arrays["right"], dtype=np.int64)
        is_leaf = np.asarray(arrays["is_leaf"]).astype(bool)
        value = np.asarray(arrays["value"], dtype=np.float64)
        offsets = np.asarray(arrays["tree_offsets"], dtype=np.int64)
        tree_class = np.asarray(arrays["tree_class"], dtype=np.int64)
        n_nodes = feature.shape[0]
        if any(
            a.shape != (n_nodes,) for a in (threshold, missing_left, left, right, is_leaf, value)
        ):
            raise GbmBundleError("node arrays differ in length")
        n_trees = tree_class.shape[0]
        if offsets.shape != (n_trees + 1,) or offsets[0] != 0 or offsets[-1] != n_nodes:
            raise GbmBundleError("tree offsets do not cover the node arrays")
        sizes = np.diff(offsets)
        if n_trees == 0 or np.any(sizes < 1):
            raise GbmBundleError("a tree has no nodes")
        if n_trees % k or np.any(tree_class != np.arange(n_trees) % k):
            raise GbmBundleError("trees are not in iteration order, one per class")
        # Children are local to their tree; make them global, and require every
        # child to come after its parent so a walk always ends.
        owner = np.repeat(np.arange(n_trees), sizes)
        local = np.arange(n_nodes) - offsets[owner]
        split = ~is_leaf
        size_of = sizes[owner]
        for child in (left, right):
            bad = split & ((child <= local) | (child >= size_of))
            if np.any(bad):
                raise GbmBundleError("a split node's child is outside its tree or before it")
        if np.any(split & ((feature < 0) | (feature >= self.n_features_in_))):
            raise GbmBundleError("a split names a feature the vectorizer does not have")
        base = offsets[owner]
        self._feature = np.where(split, feature, 0)
        self._threshold = threshold
        self._missing_left = missing_left
        # children[2 * node + went_left]: the right child, then the left one.
        self._children = np.stack(
            [np.where(split, right + base, 0), np.where(split, left + base, 0)], axis=1
        ).ravel()
        self._is_leaf = is_leaf
        self._value = value
        self._roots = offsets[:-1]
        self._tree_class = tree_class
        self._max_steps = int(sizes.max())

    @property
    def n_trees(self) -> int:
        return int(self._roots.shape[0])

    def _leaf_values(self, x: np.ndarray) -> np.ndarray:
        """``(n_samples, n_trees)`` leaf value each sample reaches in each tree."""
        n, n_features = x.shape
        flat_x = np.ascontiguousarray(x).ravel()
        node = np.broadcast_to(self._roots, (n, self.n_trees)).copy().ravel()
        # Offset of each walk's sample row in flat_x.
        row_start = np.repeat(np.arange(n, dtype=np.int64) * n_features, self.n_trees)
        active = np.flatnonzero(~self._is_leaf[node])
        cur = node[active]
        row_start = row_start[active]
        feature = self._feature
        children = self._children
        for _ in range(self._max_steps):
            if active.size == 0:
                break
            data = flat_x[row_start + feature[cur]]
            go_left = data <= self._threshold[cur]
            # NaN <= threshold is False, so the comparison alone would send every
            # missing value right; the node's missing direction decides instead.
            missing = np.isnan(data)
            if missing.any():
                go_left[missing] = self._missing_left[cur[missing]]
            cur = children[2 * cur + go_left]
            going = ~self._is_leaf[cur]
            done = ~going
            node[active[done]] = cur[done]
            active, cur, row_start = active[going], cur[going], row_start[going]
        if active.size:
            raise GbmBundleError("a tree walk did not reach a leaf")
        return self._value[node].reshape(n, self.n_trees)

    def raw_predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[-1] if x.ndim else 0} features, but the model "
                f"expects {self.n_features_in_}"
            )
        # The layout and order of HistGradientBoosting._raw_predict: a
        # Fortran-ordered array holding the baseline, each tree's leaf values
        # added in training order.
        raw = np.zeros((x.shape[0], self._k), dtype=np.float64, order="F")
        raw += self._baseline
        if x.shape[0] == 0:
            return raw
        leaves = self._leaf_values(x)
        for t in range(self.n_trees):
            raw[:, self._tree_class[t]] += leaves[:, t]
        return raw

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = self.raw_predict(x)
        if self.loss == "binomial":
            # HalfBinomialLoss.predict_proba: expit(raw), then 1 - p for class 0.
            # scipy's expit is 1 / (1 + exp(-x)) with the C library's exp, which
            # math.exp calls too; numpy's vectorised exp can differ from it by an
            # ulp (on AVX-512 machines), so it is not used here.
            r = raw[:, 0]
            proba = np.empty((r.shape[0], 2), dtype=np.float64)
            proba[:, 1] = [_expit(float(v)) for v in r]
            proba[:, 0] = 1 - proba[:, 1]
            return proba
        # HalfMultinomialLoss.predict_proba: sklearn.utils.extmath.softmax.
        out = np.array(raw, copy=True)
        max_prob = np.reshape(np.max(out, axis=1), (-1, 1))
        out -= max_prob
        np.exp(out, out=out)
        sum_prob = np.reshape(np.sum(out, axis=1), (-1, 1))
        out /= sum_prob
        return out


def _expit(v: float) -> float:
    """scipy.special.expit for one double, through the C library's exp (math.exp)."""
    try:
        return 1.0 / (1.0 + math.exp(-v))
    except OverflowError:  # exp(-v) is inf in C: expit is 0
        return 0.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GbmBundleError(f"GBM bundle manifest {path} is unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise GbmBundleError(f"{path} is not a {FORMAT} manifest")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or not _ARRAYS_NAME.fullmatch(str(arrays.get("file", ""))):
        raise GbmBundleError(f"{path}: 'arrays.file' must name an .npz file beside it")
    return manifest


def bundle_from_files(manifest_path: str | Path, arrays_path: str | Path) -> dict[str, Any]:
    """The bundle a manifest and its arrays file hold, shaped like the joblib bundle.

    The keys are the manifest's ``bundle`` metadata plus ``model`` (a
    :class:`TreeEnsemble`) and ``vectorizer`` (a :class:`DictFeatures`).
    """
    manifest = read_manifest(manifest_path)
    expected = str(manifest["arrays"].get("sha256", ""))
    actual = sha256_file(arrays_path)
    if actual != expected:
        raise GbmBundleError(
            f"{arrays_path} does not match its manifest (sha256 {actual[:12]}, "
            f"manifest {expected[:12] or 'none'})"
        )
    try:
        with np.load(arrays_path, allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in npz.files}
    except (OSError, ValueError) as exc:
        raise GbmBundleError(f"{arrays_path} is unreadable: {exc}") from exc
    missing = [
        n for n in (*_NODE_ARRAYS, "tree_offsets", "tree_class", "baseline") if n not in arrays
    ]
    if missing:
        raise GbmBundleError(f"{arrays_path} lacks {', '.join(missing)}")
    vec = manifest.get("vectorizer") or {}
    vectorizer = DictFeatures(vec.get("feature_names") or [], vec.get("separator", "="))
    model = TreeEnsemble(
        arrays,
        classes=manifest.get("classes") or [],
        loss=str(manifest.get("loss", "")),
        n_features=len(vectorizer.feature_names_),
    )
    meta = manifest.get("bundle")
    if not isinstance(meta, dict):
        raise GbmBundleError(f"{manifest_path}: 'bundle' metadata missing")
    return {**meta, "model": model, "vectorizer": vectorizer}


def _from_hub_manifest(
    repo_id: str, manifest_file: str, revision: str | None
) -> tuple[str, dict[str, Any]]:
    from bibr.ner.checkpoint import resolve_checkpoint

    manifest_path = resolve_checkpoint(f"{repo_id}:{manifest_file}", revision=revision)
    arrays_file = read_manifest(manifest_path)["arrays"]["file"]
    arrays_path = resolve_checkpoint(f"{repo_id}:{arrays_file}", revision=revision)
    return manifest_path, bundle_from_files(manifest_path, arrays_path)


def _from_local_manifest(manifest_path: Path) -> tuple[str, dict[str, Any]]:
    arrays_path = manifest_path.with_name(read_manifest(manifest_path)["arrays"]["file"])
    return str(manifest_path), bundle_from_files(manifest_path, arrays_path)


def _from_joblib(spec: str, revision: str | None) -> tuple[str, dict[str, Any]]:
    from bibr.ner.checkpoint import resolve_checkpoint
    from bibr.utils.safe_pickle import safe_joblib_load

    path = resolve_checkpoint(spec, revision=revision)
    # joblib.load executes code in the file, so only the gadget-restricted
    # loader reads it; the *_MODEL_ID settings stay a trust boundary.
    return path, safe_joblib_load(path)


def load_gbm_bundle(
    model_spec: str, revision: str | None, *, stem: str
) -> tuple[str, dict[str, Any]]:
    """``(path, bundle)`` for a GBM model setting; ``stem`` is the bundle's file stem.

    - A Hub repo id (``org/name``) reads ``<stem>.json`` and its arrays at
      ``revision``, and the joblib pickle ``<stem>.joblib`` when the revision has
      no manifest (or it cannot be fetched).
    - ``org/name:file`` reads that file: a ``.json`` manifest, else a pickle.
    - A local directory reads ``<stem>.json`` in it, else ``<stem>.joblib``.
    - A local file is a manifest when it ends in ``.json``, else a pickle.

    The path returned is the file read: the manifest, or the pickle.
    """
    from bibr.utils.ml_runtime import is_hub_repo_id

    spec = str(model_spec)
    if ":" in spec:
        repo_id, _, filename = spec.partition(":")
        if is_hub_repo_id(repo_id, filename=filename, warn=False):
            if filename.endswith(".json"):
                return _from_hub_manifest(repo_id, filename, revision)
            return _from_joblib(spec, revision)
    elif is_hub_repo_id(spec, warn=False):
        try:
            return _from_hub_manifest(spec, f"{stem}.json", revision)
        except GbmBundleError:
            raise
        except Exception as exc:  # noqa: BLE001 — a revision without the manifest reads the pickle
            logger.debug(
                "no %s.json in %s@%s (%s); reading the joblib bundle", stem, spec, revision, exc
            )
        return _from_joblib(f"{spec}:{stem}.joblib", revision)
    local = Path(spec).expanduser()
    if local.is_dir():
        manifest = local / f"{stem}.json"
        if manifest.is_file():
            return _from_local_manifest(manifest)
        return _from_joblib(str(local / f"{stem}.joblib"), revision)
    if local.suffix == ".json":
        if not local.is_file():
            raise ValueError(f"GBM bundle manifest {spec!r} is not a file")
        return _from_local_manifest(local)
    return _from_joblib(spec, revision)


__all__ = [
    "FORMAT",
    "DictFeatures",
    "GbmBundleError",
    "TreeEnsemble",
    "bundle_from_files",
    "load_gbm_bundle",
    "read_manifest",
    "sha256_file",
]
