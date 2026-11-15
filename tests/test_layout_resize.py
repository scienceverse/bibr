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


def _u8_reference(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """The float32 passes with rounding between them: what the uint8 resize replaced."""
    if img.shape[1:] == (out_h, out_w):
        return img.copy()
    passes = mod._bicubic_passes(img.astype(np.float32), out_h, out_w, round_between=True)
    return mod._to_uint8(passes).astype(np.uint8)


def test_u8_planar_resize_is_bit_identical_to_the_float_passes():
    """The block-wise uint8 resize returns exactly the float32 passes' bytes.

    Random noise is the hard case (bicubic overshoot clips at 0 and 255 on almost
    every pixel). The shapes cover a page-size downscale, ratios that do not divide,
    an upscale, a block-size boundary, single rows and columns, and no resize at all.
    """
    rng = np.random.default_rng(13)
    shapes = [
        (2339, 1654, 800, 800),
        (97, 61, 40, 24),
        (30, 50, 64, 96),
        (33, 31, 32, 33),
        (65, 64, 64, 65),
        (5, 7, 3, 4),
        (1, 9, 2, 3),
        (33, 1, 8, 8),
        (64, 64, 64, 64),
    ]
    for in_h, in_w, out_h, out_w in shapes:
        img = rng.integers(0, 256, size=(3, in_h, in_w), dtype=np.uint8)
        ours = mod.resize_bicubic_no_antialias(img, out_h, out_w)
        assert ours.dtype == np.uint8
        assert np.array_equal(ours, _u8_reference(img, out_h, out_w)), (in_h, in_w, out_h, out_w)


def test_u8_planar_resize_accepts_non_contiguous_views():
    """``preprocess_images`` hands the resize a transposed view of the PIL buffer."""
    rng = np.random.default_rng(14)
    hwc = rng.integers(0, 256, size=(150, 110, 3), dtype=np.uint8)
    chw = hwc.transpose(2, 0, 1)
    assert not chw.flags["C_CONTIGUOUS"]
    ours = mod.resize_bicubic_no_antialias(chw, 48, 40)
    assert np.array_equal(ours, _u8_reference(np.ascontiguousarray(chw), 48, 40))
