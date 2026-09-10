"""Restricted joblib/pickle loading for model checkpoints.

``joblib.load`` (like ``pickle.load``) executes arbitrary code embedded in the
file, so a ``*_MODEL_ID`` setting pointed at an attacker-controlled ``.joblib``
is remote-code-execution (audit M6). You cannot validate a pickle's *shape*
before unpickling — the payload runs during load — so this module instead
restricts *what* the unpickler is allowed to resolve, blocking the standard
gadget modules/callables (``os``/``subprocess``/``eval``/…) while leaving the
numpy/scipy/scikit-learn classes a real model bundle needs.

This is a defense-in-depth backstop; ``*_MODEL_ID`` settings remain a trust
boundary and should only point at artifacts you control.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

# Modules that a legitimate model bundle never references but pickle RCE gadgets
# do. Matched on the top-level package so ``os.path`` etc. are covered too.
_DANGEROUS_MODULES = frozenset(
    {
        "os",
        "posix",
        "nt",
        "subprocess",
        "sys",
        "socket",
        "shutil",
        "pty",
        "tty",
        "ctypes",
        "cffi",
        "importlib",
        "runpy",
        "code",
        "codeop",
        "pdb",
        "bdb",
        "multiprocessing",
        "threading",
        "signal",
        "asyncio",
        "webbrowser",
        "platform",
        "commands",
        "popen2",
        "pickle",
        "pickletools",
        "timeit",
    }
)

# Direct code-execution primitives in ``builtins`` (getattr/setattr are left
# alone — they are used by legitimate reconstructors and are inert without a
# dangerous module, which is already blocked above).
_DANGEROUS_BUILTINS = frozenset(
    {"eval", "exec", "compile", "__import__", "open", "breakpoint", "input", "exit", "quit"}
)


class RestrictedUnpicklingError(pickle.UnpicklingError):
    """Raised when a checkpoint tries to unpickle a disallowed global."""


class _RestrictedFindClass:
    """Mixin overriding ``find_class`` to reject gadget globals.

    Cooperative: ``super().find_class`` dispatches to the concrete unpickler next
    in the MRO (e.g. joblib's ``NumpyUnpickler``).
    """

    def find_class(self, module: str, name: str):  # type: ignore[override]
        root = module.split(".", 1)[0]
        if root in _DANGEROUS_MODULES:
            raise RestrictedUnpicklingError(f"refusing to unpickle {module}.{name}")
        if module == "builtins" and name in _DANGEROUS_BUILTINS:
            raise RestrictedUnpicklingError(f"refusing to unpickle builtins.{name}")
        return super().find_class(module, name)  # type: ignore[misc]


def safe_joblib_load(path: str | Path) -> Any:
    """Load a joblib artifact with the gadget-restricted unpickler.

    Uses joblib's own (compression-aware) file handling by swapping only the
    unpickler class it instantiates, so it stays correct across joblib's
    file-format internals. Intended for trusted-startup loading of model
    checkpoints; not thread-safe against concurrent joblib loads (it temporarily
    substitutes ``joblib.numpy_pickle.NumpyUnpickler``).
    """
    import joblib
    import joblib.numpy_pickle as jnp

    class _RestrictedNumpyUnpickler(_RestrictedFindClass, jnp.NumpyUnpickler):
        pass

    original = jnp.NumpyUnpickler
    jnp.NumpyUnpickler = _RestrictedNumpyUnpickler
    try:
        return joblib.load(path)
    finally:
        jnp.NumpyUnpickler = original
