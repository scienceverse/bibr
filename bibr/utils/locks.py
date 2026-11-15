"""Process-wide locks shared across local native-model wrappers.

Stdlib-only on purpose: imported by modules that must stay importable in
torch-free core installs (the ``ml`` extra is optional).
"""

import contextlib
import threading

# Serialize native model loads and inference across local post-processing threads. The reentrant
# lock also permits wrapped helpers to call each other safely.
LOCAL_INFERENCE_LOCK = threading.RLock()

# Concurrent native model work from sibling post-parse threads (torch-MPS loads
# and forwards, OpenMP tree-model predicts) segfaulted the process; the lock
# predates ONNX Runtime in bibr. ``InferenceSession.run`` on the CPU provider is
# documented thread-safe and touches neither torch nor OpenMP, so such a
# session runs without it. Every other provider (CUDA, CoreML, DirectML, ...)
# and every torch model keeps the lock.
_CONCURRENT_SAFE_ORT_PROVIDERS = frozenset({"CPUExecutionProvider"})


def inference_lock_for(session) -> contextlib.AbstractContextManager:
    """The lock a run of *session* must hold: none for an ONNX Runtime session
    on the CPU provider alone, :data:`LOCAL_INFERENCE_LOCK` for anything else
    (including an object that is not an ONNX Runtime session)."""
    try:
        providers = set(session.get_providers())
    except Exception:  # noqa: BLE001 — not an ORT session: keep the lock
        return LOCAL_INFERENCE_LOCK
    if providers and providers <= _CONCURRENT_SAFE_ORT_PROVIDERS:
        return contextlib.nullcontext()
    return LOCAL_INFERENCE_LOCK
