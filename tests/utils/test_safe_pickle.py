"""Tests for the restricted joblib loader (audit M6)."""

import pickle

import pytest

joblib = pytest.importorskip("joblib")

from bibr.utils.safe_pickle import safe_joblib_load


def test_loads_a_legitimate_bundle(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "bundle.joblib"
    bundle = {"model": np.array([1, 2, 3]), "feature_keys": ["a", "b"], "threshold": 0.5}
    joblib.dump(bundle, path)

    loaded = safe_joblib_load(path)
    assert list(loaded["feature_keys"]) == ["a", "b"]
    assert loaded["threshold"] == 0.5
    assert list(loaded["model"]) == [1, 2, 3]


def test_blocks_os_system_gadget(tmp_path):
    """A malicious bundle whose __reduce__ invokes os.system must be refused
    before the gadget executes (RCE via operator-pointable REF_*_MODEL_ID)."""

    class _Evil:
        def __reduce__(self):
            import os

            return (os.system, ("echo pwned > /dev/null",))

    path = tmp_path / "evil.joblib"
    joblib.dump(_Evil(), path)

    with pytest.raises((pickle.UnpicklingError, Exception)) as exc:
        safe_joblib_load(path)
    assert "os" in str(exc.value) or "refus" in str(exc.value).lower()


def test_blocks_builtins_eval_gadget(tmp_path):
    class _Evil:
        def __reduce__(self):
            return (eval, ("__import__('os').getcwd()",))

    path = tmp_path / "evil2.joblib"
    joblib.dump(_Evil(), path)

    with pytest.raises(Exception):  # noqa: B017
        safe_joblib_load(path)


# --- allowlist, not denylist (ml-runtime-utils-1) -----------------------------
# A denylist of module roots cannot hold: any allowed module can hand a denied
# one back as an attribute. These files only *name* globals or hold inert
# objects; nothing in them is ever called.


def test_refuses_a_builtin_a_bundle_never_needs(tmp_path):
    path = tmp_path / "getattr.joblib"
    path.write_bytes(pickle.dumps(getattr, protocol=4))  # nosemgrep

    with pytest.raises(pickle.UnpicklingError) as exc:
        safe_joblib_load(path)
    assert str(exc.value) == (
        "refusing to unpickle builtins.getattr: not a class bibr model bundles use"
    )


def test_refuses_a_class_outside_the_bundle_allowlist(tmp_path):
    import collections

    path = tmp_path / "ordered.joblib"
    joblib.dump({"x": collections.OrderedDict(a=1)}, path)

    with pytest.raises(pickle.UnpicklingError) as exc:
        safe_joblib_load(path)
    assert str(exc.value) == (
        "refusing to unpickle collections.OrderedDict: not a class bibr model bundles use"
    )


def test_object_array_payload_is_read_under_the_allowlist(tmp_path):
    """joblib unpickles an object-dtype array's payload with a bare pickle.load."""
    import collections

    np = pytest.importorskip("numpy")
    blocked = np.empty(1, dtype=object)
    blocked[0] = collections.OrderedDict(a=1)
    joblib.dump({"arr": blocked}, tmp_path / "blocked.joblib")
    with pytest.raises(pickle.UnpicklingError, match="collections.OrderedDict"):
        safe_joblib_load(tmp_path / "blocked.joblib")

    labels = np.array(["B-REF", "I-REF"], dtype=object)
    joblib.dump({"classes": labels}, tmp_path / "labels.joblib")
    assert list(safe_joblib_load(tmp_path / "labels.joblib")["classes"]) == ["B-REF", "I-REF"]


def test_object_array_payload_may_not_nest_a_joblib_array_wrapper(tmp_path):
    """Only joblib's own unpickler reads a wrapper's array bytes; inside an
    object-array payload one is refused rather than left half-built."""
    np = pytest.importorskip("numpy")
    import joblib.numpy_pickle as jnp

    nested = np.empty(1, dtype=object)
    nested[0] = jnp.NumpyArrayWrapper(np.ndarray, (1,), "C", np.dtype("float64"))
    joblib.dump({"arr": nested}, tmp_path / "nested.joblib")

    with pytest.raises(pickle.UnpicklingError) as exc:
        safe_joblib_load(tmp_path / "nested.joblib")
    assert str(exc.value) == "refusing a nested joblib array wrapper"


def test_refuses_the_pre_0_10_joblib_format(tmp_path):
    """That format is read by joblib's compatibility unpickler, not the restricted one."""
    path = tmp_path / "legacy.joblib"
    path.write_bytes(b"ZF0x00000000000000000")

    with pytest.raises(pickle.UnpicklingError) as exc:
        safe_joblib_load(path)
    assert str(exc.value) == "refusing a pre-0.10 joblib file: re-export the bundle"


def test_a_trained_bundle_round_trips_and_predicts_identically(tmp_path):
    np = pytest.importorskip("numpy")
    ensemble = pytest.importorskip("sklearn.ensemble")
    feature_extraction = pytest.importorskip("sklearn.feature_extraction")

    rng = np.random.default_rng(0)
    rows = [{"a": float(rng.random()), "b": float(rng.random())} for _ in range(60)]
    for classes in (2, 3):
        dv = feature_extraction.DictVectorizer(sparse=False)
        x = dv.fit_transform(rows)
        model = ensemble.HistGradientBoostingClassifier(max_iter=10, random_state=0)
        model.fit(x, rng.integers(0, classes, len(rows)))
        path = tmp_path / f"bundle{classes}.joblib"
        joblib.dump({"model": model, "vectorizer": dv, "feature_keys": ["a", "b"]}, path)

        loaded = safe_joblib_load(path)
        again = loaded["vectorizer"].transform(rows)
        assert (loaded["model"].predict_proba(again) == model.predict_proba(x)).all()


@pytest.mark.parametrize(
    ("repo", "revision", "filename"),
    [
        (
            "scienceverse/bibr-front-role-v1",
            "7f01b57e1999d93cb5f17895ed10fdfa27cf6e0b",
            "front_role.joblib",
        ),
        (
            "scienceverse/bibr-geom-segmenter-v1",
            "4d1702e2c766d96c8887bd4b30ef56637aa9b32c",
            "geom_segmenter.joblib",
        ),
    ],
)
def test_the_shipped_bundles_still_load(repo, revision, filename):
    """The pinned default bundles, when this machine has them cached (no download)."""
    hub = pytest.importorskip("huggingface_hub")
    path = hub.try_to_load_from_cache(repo, filename, revision=revision)
    if not isinstance(path, str):
        pytest.skip(f"{repo}@{revision[:7]} is not in the local Hugging Face cache")

    bundle = safe_joblib_load(path)
    assert type(bundle["model"]).__name__ == "HistGradientBoostingClassifier"
    assert type(bundle["vectorizer"]).__name__ == "DictVectorizer"
