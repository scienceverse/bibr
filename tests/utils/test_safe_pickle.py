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
