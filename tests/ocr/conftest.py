"""Snapshot/restore the OCR backend registry per test.

Tests that call ``registry.register(_FakeBackend)`` mutate the module-level
``_BACKENDS`` dict. Without this fixture the mutation would leak to other
tests in the same session.

The baseline snapshot is captured after importing ``bibr.local.ocr`` so that
the real ``@register`` side-effects land before the first test runs; module
import is memoised, so restoring an empty snapshot would otherwise permanently
wipe the production entries.
"""

import pytest

from bibr.local import ocr as _ocr  # noqa: F401 — triggers @register
from bibr.local import ocr_cloud as _ocr_cloud  # noqa: F401 — triggers cloud @register
from bibr.ocr import registry
from bibr.serve import (
    ocr_backend as _serve_ocr_backend,  # noqa: F401 (side-effect: register serve-http)
)

_BASELINE = dict(registry._BACKENDS)


@pytest.fixture(autouse=True)
def _ocr_registry_snapshot():
    registry._BACKENDS.clear()
    registry._BACKENDS.update(_BASELINE)
    yield
    registry._BACKENDS.clear()
    registry._BACKENDS.update(_BASELINE)
