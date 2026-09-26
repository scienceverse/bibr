"""Restricted joblib/pickle loading for model checkpoints.

``joblib.load`` (like ``pickle.load``) executes arbitrary code embedded in the
file, so a ``*_MODEL_ID`` setting pointed at an attacker-controlled ``.joblib``
is remote-code-execution (audit M6). You cannot validate a pickle's *shape*
before unpickling — the payload runs during load — so this module instead
restricts *what* the unpickler may resolve: an allowlist of the exact classes a
bibr model bundle is made of. A denylist of dangerous modules cannot work,
because any allowed module can hand a denied one back as an attribute.

This is a defense-in-depth backstop; ``*_MODEL_ID`` settings remain a trust
boundary and should only point at artifacts you control.
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Any

# Every global a bibr model bundle references, and nothing else. A bundle is a
# dict holding a DictVectorizer and a HistGradientBoostingClassifier (the
# front-role and geometry-segmenter trainers), dumped by joblib. Recorded from
# the shipped bundles and from bundles the locked scikit-learn/numpy dump. A
# different estimator, or an upgrade that pickles another class, is refused by
# name: review the class, then add it here.
_ALLOWED_GLOBALS = frozenset(
    {
        ("joblib.numpy_pickle", "NumpyArrayWrapper"),
        # Arrays, dtypes and scalars; numpy 2 pickles numpy._core, numpy 1 numpy.core.
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        *(
            ("numpy", scalar)
            for scalar in (
                "bool_",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "float16",
                "float32",
                "float64",
            )
        ),
        *(
            (module, name)
            for module in ("numpy._core.multiarray", "numpy.core.multiarray")
            for name in ("_reconstruct", "scalar")
        ),
        # Random state kept by the estimators.
        ("numpy.random._pickle", "__bit_generator_ctor"),
        ("numpy.random._pickle", "__generator_ctor"),
        ("numpy.random._pickle", "__randomstate_ctor"),
        ("numpy.random._pcg64", "PCG64"),
        ("numpy.random._mt19937", "MT19937"),
        ("numpy.random.bit_generator", "SeedSequence"),
        ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"),
        # scikit-learn: the vectorizer, the classifier and what it is built from.
        ("sklearn.feature_extraction._dict_vectorizer", "DictVectorizer"),
        ("sklearn.preprocessing._label", "LabelEncoder"),
        (
            "sklearn.ensemble._hist_gradient_boosting.gradient_boosting",
            "HistGradientBoostingClassifier",
        ),
        ("sklearn.ensemble._hist_gradient_boosting.binning", "_BinMapper"),
        ("sklearn.ensemble._hist_gradient_boosting.predictor", "TreePredictor"),
        ("sklearn._loss.loss", "HalfBinomialLoss"),
        ("sklearn._loss.loss", "HalfMultinomialLoss"),
        ("sklearn._loss.link", "Interval"),
        ("sklearn._loss.link", "LogitLink"),
        ("sklearn._loss.link", "MultinomialLogit"),
        ("sklearn._loss._loss", "CyHalfBinomialLoss"),
        ("sklearn._loss._loss", "__pyx_unpickle_CyHalfBinomialLoss"),
        ("sklearn._loss._loss", "CyHalfMultinomialLoss"),
        ("sklearn._loss._loss", "__pyx_unpickle_CyHalfMultinomialLoss"),
    }
)

_ARRAY_WRAPPER = ("joblib.numpy_pickle", "NumpyArrayWrapper")

# safe_joblib_load swaps module globals of joblib for the duration of a load.
_LOAD_LOCK = threading.Lock()


class RestrictedUnpicklingError(pickle.UnpicklingError):
    """Raised when a checkpoint tries to unpickle a disallowed global."""


def _check_global(module: str, name: str) -> None:
    if (module, name) not in _ALLOWED_GLOBALS:
        raise RestrictedUnpicklingError(
            f"refusing to unpickle {module}.{name}: not a class bibr model bundles use"
        )


class _RestrictedUnpickler(pickle.Unpickler):
    """Plain unpickler under the same allowlist, for object-dtype array payloads."""

    def find_class(self, module: str, name: str):
        _check_global(module, name)
        if (module, name) == _ARRAY_WRAPPER:
            raise RestrictedUnpicklingError("refusing a nested joblib array wrapper")
        return super().find_class(module, name)


def _restricted_array_wrapper():
    import joblib.numpy_pickle as jnp

    class _RestrictedNumpyArrayWrapper(jnp.NumpyArrayWrapper):
        """joblib hands an object-dtype array's payload to a bare ``pickle.load``,
        which resolves any global at all; read it under the allowlist instead."""

        def read_array(self, unpickler, ensure_native_byte_order):
            if not self.dtype.hasobject:
                return super().read_array(unpickler, ensure_native_byte_order)
            return _RestrictedUnpickler(unpickler.file_handle).load()

    return _RestrictedNumpyArrayWrapper


class _RestrictedFindClass:
    """Mixin overriding ``find_class`` to resolve only allowlisted globals.

    Cooperative: ``super().find_class`` dispatches to the concrete unpickler next
    in the MRO (e.g. joblib's ``NumpyUnpickler``).
    """

    _array_wrapper: type | None = None

    def find_class(self, module: str, name: str):  # type: ignore[override]
        _check_global(module, name)
        if (module, name) == _ARRAY_WRAPPER and self._array_wrapper is not None:
            return self._array_wrapper
        return super().find_class(module, name)  # type: ignore[misc]


def _refuse_legacy_format(*_args: object, **_kwargs: object) -> Any:
    # Files from joblib < 0.10 are read by a separate, unrestricted unpickler.
    raise RestrictedUnpicklingError("refusing a pre-0.10 joblib file: re-export the bundle")


def safe_joblib_load(path: str | Path) -> Any:
    """Load a joblib artifact with the gadget-restricted unpickler.

    Uses joblib's own (compression-aware) file handling by swapping only the
    unpickler class it instantiates, so it stays correct across joblib's
    file-format internals. Intended for trusted-startup loading of model
    checkpoints. It temporarily substitutes ``joblib.numpy_pickle.NumpyUnpickler``
    (and refuses joblib's pre-0.10 format, which bypasses it), so safe loads are
    serialized; an unrelated ``joblib.load`` running at the same moment in
    another thread is restricted too.
    """
    import joblib
    import joblib.numpy_pickle as jnp

    class _RestrictedNumpyUnpickler(_RestrictedFindClass, jnp.NumpyUnpickler):
        _array_wrapper = _restricted_array_wrapper()

    with _LOAD_LOCK:
        original = jnp.NumpyUnpickler
        original_legacy = jnp.load_compatibility
        jnp.NumpyUnpickler = _RestrictedNumpyUnpickler
        jnp.load_compatibility = _refuse_legacy_format
        try:
            return joblib.load(path)
        finally:
            jnp.NumpyUnpickler = original
            jnp.load_compatibility = original_legacy
