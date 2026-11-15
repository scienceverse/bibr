"""The pickle-free GBM bundle reader (bibr.utils.gbm_bundle) against scikit-learn."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer

from bibr.utils import gbm_bundle
from bibr.utils.gbm_bundle import (
    DictFeatures,
    GbmBundleError,
    bundle_from_files,
    load_gbm_bundle,
)
from scripts.export_gbm_bundle import main as export_main

KEYS = ["a", "b", "c", "label"]


def _rows(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {
            "a": float(rng.normal()),
            "b": None if i % 9 == 0 else float(rng.integers(0, 5)),
            "c": bool(i % 2),
            "label": ["x", "y", "z"][i % 3],
        }
        if i % 11 == 0:
            del row["a"]  # absent: zero
        if i % 13 == 0:
            row["a"] = float("nan")
        rows.append(row)
    return rows


def _fit_bundle(tmp_path: Path, n_classes: int, stem: str = "model") -> Path:
    rows = _rows(400, 0)
    dv = DictVectorizer(sparse=False)
    x = dv.fit_transform(rows)
    rng = np.random.default_rng(1)
    y = (x[:, 0] > 0).astype(int) if n_classes == 2 else rng.integers(0, n_classes, len(rows))
    y[:n_classes] = np.arange(n_classes)
    model = HistGradientBoostingClassifier(max_iter=25, random_state=0, min_samples_leaf=5)
    model.fit(x, y)
    path = tmp_path / f"{stem}.joblib"
    joblib.dump(
        {"model": model, "vectorizer": dv, "feature_keys": KEYS, "version": "test", "roles": ["r"]},
        path,
    )
    return path


def _export(src: Path, out: Path, stem: str = "model") -> tuple[Path, Path]:
    assert export_main([str(src), str(out), "--stem", stem]) == 0
    return out / f"{stem}.json", out / f"{stem}.npz"


@pytest.mark.parametrize("n_classes", [2, 4])
def test_converted_bundle_reproduces_sklearn_bit_for_bit(tmp_path, n_classes):
    src = _fit_bundle(tmp_path, n_classes)
    manifest, arrays = _export(src, tmp_path / "out")
    ref = joblib.load(src)
    new = bundle_from_files(manifest, arrays)

    rows = _rows(300, 7) + [{"label": "never-seen"}, {"a": None, "b": None}]
    x_ref = ref["vectorizer"].transform(rows)
    x_new = new["vectorizer"].transform(rows)
    assert np.array_equal(x_ref, x_new, equal_nan=True)
    assert np.array_equal(ref["model"]._raw_predict(x_ref), new["model"].raw_predict(x_new))
    assert np.array_equal(ref["model"].predict_proba(x_ref), new["model"].predict_proba(x_new))
    assert list(new["model"].classes_) == list(ref["model"].classes_)
    assert new["feature_keys"] == KEYS and new["version"] == "test"


def test_dict_features_follows_dictvectorizer_value_rules():
    dv = DictVectorizer(sparse=False).fit([{"n": 1.0, "s": "a", "t": ["p", "q"]}])
    ours = DictFeatures(dv.feature_names_, dv.separator)
    rows = [{"n": None, "s": "a", "t": ["q", "q"]}, {"n": True, "s": "zzz"}, {"other": 3.0}]
    assert np.array_equal(dv.transform(rows), ours.transform(rows), equal_nan=True)
    with pytest.raises(TypeError):
        ours.transform([{"n": object()}])
    with pytest.raises(TypeError):
        ours.transform([{"t": [1]}])
    with pytest.raises(ValueError):
        ours.transform([])


def _rewrite(manifest: Path, arrays: Path, **changes) -> None:
    with np.load(arrays) as npz:
        data = {k: npz[k] for k in npz.files}
    data.update(changes)
    np.savez(arrays, **data)
    meta = json.loads(manifest.read_text())
    meta["arrays"]["sha256"] = gbm_bundle.sha256_file(arrays)
    manifest.write_text(json.dumps(meta))


def test_bundle_with_tampered_arrays_is_refused(tmp_path):
    manifest, arrays = _export(_fit_bundle(tmp_path, 2), tmp_path / "out")
    arrays.write_bytes(arrays.read_bytes() + b"\0")
    with pytest.raises(GbmBundleError, match="does not match its manifest"):
        bundle_from_files(manifest, arrays)


def test_bundle_whose_tree_loops_is_refused(tmp_path):
    manifest, arrays = _export(_fit_bundle(tmp_path, 2), tmp_path / "out")
    with np.load(arrays) as npz:
        left = npz["left"].copy()
        is_leaf = npz["is_leaf"]
    root_split = int(np.flatnonzero(is_leaf == 0)[0])
    left[root_split] = root_split  # a child that points back at its parent
    _rewrite(manifest, arrays, left=left)
    with pytest.raises(GbmBundleError, match="child"):
        bundle_from_files(manifest, arrays)


def test_bundle_naming_an_unknown_feature_is_refused(tmp_path):
    manifest, arrays = _export(_fit_bundle(tmp_path, 2), tmp_path / "out")
    with np.load(arrays) as npz:
        feature = npz["feature_idx"].copy()
        is_leaf = npz["is_leaf"]
    feature[int(np.flatnonzero(is_leaf == 0)[0])] = 999
    _rewrite(manifest, arrays, feature_idx=feature)
    with pytest.raises(GbmBundleError, match="feature"):
        bundle_from_files(manifest, arrays)


def test_manifest_must_name_a_sibling_npz(tmp_path):
    manifest, arrays = _export(_fit_bundle(tmp_path, 2), tmp_path / "out")
    meta = json.loads(manifest.read_text())
    meta["arrays"]["file"] = "../elsewhere.npz"
    manifest.write_text(json.dumps(meta))
    with pytest.raises(GbmBundleError, match="beside it"):
        load_gbm_bundle(str(manifest), None, stem="model")


def test_local_directory_prefers_the_manifest_and_falls_back_to_joblib(tmp_path):
    (tmp_path / "dir").mkdir()
    src = _fit_bundle(tmp_path / "dir", 2)
    path, bundle = load_gbm_bundle(str(tmp_path / "dir"), None, stem="model")
    assert path.endswith("model.joblib")
    assert type(bundle["model"]).__name__ == "HistGradientBoostingClassifier"

    _export(src, tmp_path / "dir")
    path, bundle = load_gbm_bundle(str(tmp_path / "dir"), None, stem="model")
    assert path.endswith("model.json")
    assert isinstance(bundle["model"], gbm_bundle.TreeEnsemble)
    # A file spec reads what it names.
    path, bundle = load_gbm_bundle(str(src), None, stem="model")
    assert path.endswith("model.joblib")


def test_hub_id_reads_the_manifest_at_the_revision_when_it_has_one(tmp_path, monkeypatch):
    src = _fit_bundle(tmp_path, 2)
    manifest, arrays = _export(src, tmp_path / "out")
    asked = []

    def fake_resolve(spec, revision=None):
        asked.append((spec, revision))
        name = spec.split(":", 1)[1]
        local = tmp_path / "out" / name
        if name.endswith(".joblib"):
            return str(src)
        if not local.is_file():
            raise FileNotFoundError(name)
        return str(local)

    import bibr.ner.checkpoint as ckpt

    monkeypatch.setattr(ckpt, "resolve_checkpoint", fake_resolve)
    path, bundle = load_gbm_bundle("org/repo", "abc123", stem="model")
    assert path == str(manifest)
    assert isinstance(bundle["model"], gbm_bundle.TreeEnsemble)
    assert asked == [("org/repo:model.json", "abc123"), ("org/repo:model.npz", "abc123")]

    # A revision without the manifest reads the joblib bundle.
    asked.clear()
    path, bundle = load_gbm_bundle("org/repo", "abc123", stem="absent")
    assert path == str(src)
    assert asked == [("org/repo:absent.json", "abc123"), ("org/repo:absent.joblib", "abc123")]


def test_a_broken_manifest_on_the_hub_is_not_hidden_by_the_joblib_fallback(tmp_path, monkeypatch):
    bad = tmp_path / "model.json"
    bad.write_text(json.dumps({"format": "something-else"}))
    import bibr.ner.checkpoint as ckpt

    monkeypatch.setattr(ckpt, "resolve_checkpoint", lambda spec, revision=None: str(bad))
    with pytest.raises(GbmBundleError):
        load_gbm_bundle("org/repo", None, stem="model")


def test_reading_a_converted_bundle_never_imports_sklearn(tmp_path):
    manifest, _arrays = _export(_fit_bundle(tmp_path, 3), tmp_path / "out")
    code = (
        "import sys, json\n"
        "from bibr.utils.gbm_bundle import load_gbm_bundle\n"
        f"path, b = load_gbm_bundle({str(manifest)!r}, None, stem='model')\n"
        "p = b['model'].predict_proba(b['vectorizer'].transform([{'a': 0.1, 'label': 'x'}]))\n"
        "assert abs(float(p.sum()) - 1.0) < 1e-12\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('sklearn', 'joblib', 'scipy'))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_probabilities_are_finite_and_sum_to_one(tmp_path):
    manifest, arrays = _export(_fit_bundle(tmp_path, 4), tmp_path / "out")
    b = bundle_from_files(manifest, arrays)
    p = b["model"].predict_proba(b["vectorizer"].transform(_rows(50, 3)))
    assert np.all(np.isfinite(p))
    assert all(math.isclose(s, 1.0, abs_tol=1e-12) for s in p.sum(axis=1))
