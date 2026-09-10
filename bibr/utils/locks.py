"""Process-wide locks shared across local native-model wrappers.

Stdlib-only on purpose: imported by modules that must stay importable in
torch-free core installs (the ``ml`` extra is optional).
"""

import threading

# Serialize native model loads and inference across local post-processing threads. The reentrant
# lock also permits wrapped helpers to call each other safely.
LOCAL_INFERENCE_LOCK = threading.RLock()
