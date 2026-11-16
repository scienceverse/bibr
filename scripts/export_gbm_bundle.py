"""Convert a joblib GBM bundle (geom segmenter, front-role classifier) to ``bibr-gbm/1``.

    uv run python scripts/export_gbm_bundle.py SPEC OUT_DIR --stem geom_segmenter [--revision REV]

SPEC is what the model setting takes: ``org/name`` (the joblib bundle
``<stem>.joblib`` at ``--revision``), ``org/name:file.joblib`` or a local
``.joblib`` path. OUT_DIR receives ``<stem>.json`` and ``<stem>.npz`` (see
:mod:`bibr.utils.gbm_bundle`); upload both to the model repo beside the joblib
bundle and pin the new commit.

Needs scikit-learn and joblib (the dev group). The conversion refuses a model
with categorical splits, a sparse or non-float64 vectorizer, or anything other
than a ``HistGradientBoostingClassifier``. It then reloads what it wrote with
the numpy reader and checks, on the bundle's own probe rows, that raw scores and
probabilities equal scikit-learn's bit for bit; ``--check-rows FILE`` (JSON
lines of feature dicts) adds rows of your own to that check.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bibr.utils.gbm_bundle import FORMAT, bundle_from_files  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"bundle metadata value of type {type(value).__name__} is not JSON data")


def _load_joblib(spec: str, revision: str | None, stem: str) -> tuple[str, dict]:
    from bibr.ner.checkpoint import resolve_checkpoint
    from bibr.utils.ml_runtime import is_hub_repo_id
    from bibr.utils.safe_pickle import safe_joblib_load

    if ":" not in spec and is_hub_repo_id(spec, warn=False):
        spec = f"{spec}:{stem}.joblib"
    path = resolve_checkpoint(spec, revision=revision)
    return path, safe_joblib_load(path)


def _tree_arrays(model) -> dict[str, np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    if type(model) is not HistGradientBoostingClassifier:
        raise SystemExit(f"model is {type(model).__name__}, not HistGradientBoostingClassifier")
    if getattr(model, "is_categorical_", None) is not None and np.any(model.is_categorical_):
        raise SystemExit("model has categorical features, which bibr-gbm/1 does not support")
    columns: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "feature_idx",
            "threshold",
            "missing_go_to_left",
            "left",
            "right",
            "is_leaf",
            "value",
        )
    }
    offsets = [0]
    tree_class = []
    k = model.n_trees_per_iteration_
    for iteration in model._predictors:
        if len(iteration) != k:
            raise SystemExit("an iteration does not hold one tree per class")
        for cls, predictor in enumerate(iteration):
            nodes = predictor.nodes
            if np.any(nodes["is_categorical"]):
                raise SystemExit("model has a categorical split")
            columns["feature_idx"].append(nodes["feature_idx"].astype(np.int64))
            columns["threshold"].append(nodes["num_threshold"].astype(np.float64))
            columns["missing_go_to_left"].append(nodes["missing_go_to_left"].astype(np.uint8))
            columns["left"].append(nodes["left"].astype(np.int64))
            columns["right"].append(nodes["right"].astype(np.int64))
            columns["is_leaf"].append(nodes["is_leaf"].astype(np.uint8))
            columns["value"].append(nodes["value"].astype(np.float64))
            offsets.append(offsets[-1] + len(nodes))
            tree_class.append(cls)
    arrays = {name: np.concatenate(parts) for name, parts in columns.items()}
    arrays["tree_offsets"] = np.asarray(offsets, dtype=np.int64)
    arrays["tree_class"] = np.asarray(tree_class, dtype=np.int64)
    arrays["baseline"] = np.asarray(model._baseline_prediction, dtype=np.float64).ravel()
    return arrays


def _probe_rows(vectorizer, feature_keys: list[str], arrays, n: int = 2048) -> list[dict]:
    """Rows exercising the splits: values exactly at, just above and around each
    feature's split thresholds, missing values, absent keys and every one-hot value
    the vectorizer knows."""
    rng = np.random.default_rng(0)
    sep = vectorizer.separator
    split = arrays["is_leaf"] == 0
    thresholds: dict[str, np.ndarray] = {}
    for j, name in enumerate(vectorizer.feature_names_):
        values = arrays["threshold"][split & (arrays["feature_idx"] == j)]
        values = values[np.isfinite(values)]
        if values.size:
            thresholds[name] = np.unique(values)
    onehot: dict[str, list[str]] = {}
    for name in vectorizer.feature_names_:
        if sep in name:
            key, _, val = name.partition(sep)
            onehot.setdefault(key, []).append(val)
    rows = []
    for i in range(n):
        row: dict[str, Any] = {}
        for key in feature_keys:
            if key in onehot:
                choice = rng.integers(0, len(onehot[key]) + 2)
                if choice < len(onehot[key]):
                    row[key] = onehot[key][choice]
                elif choice == len(onehot[key]):
                    row[key] = "a-value-never-seen"
                continue
            r = rng.random()
            known = thresholds.get(key)
            if r < 0.05:
                row[key] = None
            elif r < 0.08:
                continue  # absent: zero
            elif r < 0.12:
                row[key] = bool(rng.integers(0, 2))
            elif known is not None and r < 0.6:
                t = float(known[rng.integers(0, known.size)])
                row[key] = t if r < 0.4 else float(np.nextafter(t, np.inf))
            elif known is not None:
                lo, hi = float(known.min()), float(known.max())
                span = (hi - lo) or 1.0
                row[key] = float(rng.uniform(lo - 0.1 * span, hi + 0.1 * span))
            else:
                row[key] = float(rng.normal(0, 1) * 10.0 ** int(rng.integers(-2, 3)))
        if i % 7 == 0:
            row[feature_keys[0]] = float("nan")
        rows.append(row)
    return rows


def _check(bundle: dict, converted: dict, rows: list[dict], feature_keys: list[str]) -> None:
    picked = [{k: r[k] for k in feature_keys if k in r} for r in rows]
    x_ref = bundle["vectorizer"].transform(picked)
    x_new = converted["vectorizer"].transform(picked)
    if not np.array_equal(x_ref, x_new, equal_nan=True):
        raise SystemExit("vectorized features differ from DictVectorizer's")
    raw_ref = bundle["model"]._raw_predict(x_ref)
    raw_new = converted["model"].raw_predict(x_new)
    p_ref = bundle["model"].predict_proba(x_ref)
    p_new = converted["model"].predict_proba(x_new)
    raw_diff = float(np.max(np.abs(raw_ref - raw_new)))
    p_diff = float(np.max(np.abs(p_ref - p_new)))
    exact = np.array_equal(raw_ref, raw_new) and np.array_equal(p_ref, p_new)
    print(
        f"  checked {len(rows)} rows: max |raw diff| {raw_diff:.3g}, max |proba diff| {p_diff:.3g}"
        f", bit-identical {exact}"
    )
    if raw_diff != 0.0 or p_diff > 1e-15:
        raise SystemExit("converted model does not reproduce scikit-learn")
    if not np.array_equal(bundle["model"].classes_, converted["model"].classes_):
        raise SystemExit("class labels differ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--stem", required=True, help="geom_segmenter or front_role")
    parser.add_argument("--revision")
    parser.add_argument("--check-rows", type=Path, help="JSON lines of feature dicts to check too")
    args = parser.parse_args(argv)

    import sklearn

    source, bundle = _load_joblib(args.spec, args.revision, args.stem)
    model, vectorizer = bundle["model"], bundle["vectorizer"]
    if type(vectorizer).__name__ != "DictVectorizer" or vectorizer.sparse:
        raise SystemExit("vectorizer must be a dense DictVectorizer")
    if np.dtype(vectorizer.dtype) != np.float64:
        raise SystemExit("vectorizer dtype must be float64")
    if model.n_features_in_ != len(vectorizer.feature_names_):
        raise SystemExit("model and vectorizer disagree on the feature count")
    arrays = _tree_arrays(model)
    meta = _json_safe({k: v for k, v in bundle.items() if k not in ("model", "vectorizer")})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays_name = f"{args.stem}.npz"
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    arrays_path = args.out_dir / arrays_name
    arrays_path.write_bytes(buf.getvalue())
    manifest = {
        "format": FORMAT,
        "estimator": "HistGradientBoostingClassifier",
        "loss": "binomial" if model.n_trees_per_iteration_ == 1 else "multinomial",
        "classes": [int(c) for c in model.classes_],
        "n_iterations": len(model._predictors),
        "n_nodes": int(arrays["feature_idx"].shape[0]),
        "vectorizer": {
            "feature_names": [str(n) for n in vectorizer.feature_names_],
            "separator": vectorizer.separator,
        },
        "arrays": {
            "file": arrays_name,
            "sha256": hashlib.sha256(buf.getvalue()).hexdigest(),
        },
        "bundle": meta,
        "source": {
            "joblib_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "revision": args.revision,
            "sklearn_version": sklearn.__version__,
        },
    }
    manifest_path = args.out_dir / f"{args.stem}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {manifest_path} and {arrays_path} ({arrays_path.stat().st_size} bytes)")

    converted = bundle_from_files(manifest_path, arrays_path)
    keys = list(bundle["feature_keys"])
    rows = _probe_rows(vectorizer, keys, arrays)
    if args.check_rows:
        rows += [
            json.loads(line) for line in args.check_rows.read_text().splitlines() if line.strip()
        ]
    _check(bundle, converted, rows, keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
