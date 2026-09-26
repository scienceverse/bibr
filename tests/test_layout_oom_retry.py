"""Layout OOM retry restores torch.compile padding (ml-runtime-utils-6).

Torch-free on purpose: the retry logic only needs ``_compiled`` and a
stubbed ``_detect_pytorch``, so this runs where tests/test_layout_base.py
skips (no torch installed).
"""

from __future__ import annotations

from PIL import Image

from bibr.layout_base import BaseLayoutDetector


def _stub_detector(*, compiled: bool):
    """A detector with stubbed inference that OOMs once on batches over 2."""
    det = object.__new__(BaseLayoutDetector)
    det._compiled = compiled
    calls: list[int] = []
    seen_compiled: list[bool] = []

    def detect(images, orig_sizes):
        calls.append(len(images))
        seen_compiled.append(det._compiled)
        if len(images) > 2:
            raise RuntimeError("CUDA out of memory")
        return [[{"width": image.width}] for image in images]

    det._detect_pytorch = detect  # type: ignore[method-assign]
    return det, calls, seen_compiled


def test_oom_retry_restores_compiled_padding():
    """The padding disable is scoped to the halved retry (ml-6)."""
    det, calls, seen_compiled = _stub_detector(compiled=True)
    images = [Image.new("RGB", (width, 100)) for width in range(10, 15)]

    out = det._detect_images(images)

    assert [rows[0]["width"] for rows in out] == [10, 11, 12, 13, 14]
    assert calls == [5, 2, 3, 1, 2]
    assert det._compiled is True
    # The full batch ran compiled; the halved retries ran unpadded.
    assert seen_compiled[0] is True
    assert all(v is False for v in seen_compiled[1:])


def test_oom_retry_leaves_eager_mode_alone():
    """An eager detector stays eager through the retry (ml-6 guard)."""
    det, calls, _ = _stub_detector(compiled=False)
    images = [Image.new("RGB", (width, 100)) for width in range(10, 13)]

    out = det._detect_images(images)

    assert [rows[0]["width"] for rows in out] == [10, 11, 12]
    assert det._compiled is False
