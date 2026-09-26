"""Resize-kernel threading equivalence (ml-runtime-utils-5).

Torch-free on purpose: the resize kernels are pure numpy, so these run in
minimal environments where tests/test_layout_onnx.py skips (torch gate).
"""

from __future__ import annotations

import numpy as np

from bibr import layout_onnx as mod


def test_x_pass_threaded_matches_single_worker(monkeypatch):
    """Threaded column scaling is bit-identical to the serial loop (ml-5).

    Each row block runs the same per-tap accumulation into disjoint slices,
    so threads only overlap the strided gathers. A tall random array forces
    the multi-worker path; forcing one worker replays the old serial order.
    Both are also checked against an independently computed reference loop,
    so a worker that silently skips its block cannot hide behind a reused
    output buffer.
    """
    rng = np.random.default_rng(11)
    src = rng.integers(0, 256, size=(3, 600, 500)).astype(np.float32)
    ix, wx = mod._resize_plan(500, 200)
    c, h, _w = src.shape
    reference = np.zeros((c, h, 200), dtype=np.float32)
    for k in range(4):
        reference += src[:, :, ix[:, k]] * wx[None, None, :, k]
    monkeypatch.setattr(mod, "_RESIZE_THREADS", 1)
    serial = mod._x_pass(src, 200, ix, wx)
    assert np.array_equal(serial, reference)
    monkeypatch.setattr(mod, "_RESIZE_THREADS", 8)
    threaded = mod._x_pass(src, 200, ix, wx)
    assert np.array_equal(serial, threaded)
    assert np.array_equal(threaded, reference)


def test_resize_kernels_agree_with_threaded_x_pass(monkeypatch):
    """End-to-end kernels stay stable while threading is on (ml-5 guard)."""
    rng = np.random.default_rng(12)
    img = rng.integers(0, 256, size=(3, 300, 260), dtype=np.uint8)
    monkeypatch.setattr(mod, "_RESIZE_THREADS", 1)
    serial_u8 = mod.resize_bicubic_no_antialias(img, 96, 80)
    serial_f = mod.resize_bicubic_float(img.astype(np.float32), 96, 80)
    monkeypatch.setattr(mod, "_RESIZE_THREADS", 8)
    assert np.array_equal(mod.resize_bicubic_no_antialias(img, 96, 80), serial_u8)
    assert np.array_equal(mod.resize_bicubic_float(img.astype(np.float32), 96, 80), serial_f)
