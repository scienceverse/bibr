"""ONNX layout backend parity: numpy pre/post-processing and the detector wiring.

A tiny random conv stand-in with PP-DocLayoutV3's output signature is exported
in-test; no download.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from bibr.layout_onnx import (  # noqa: E402
    OnnxLayoutBackend,
    decode_detections,
    order_sequences,
    preprocess_images,
    resize_bicubic_no_antialias,
)
from tests import onnx_fixtures as fx  # noqa: E402


@pytest.fixture(scope="module")
def layout_bundle(tmp_path_factory):
    module = fx.tiny_layout_module()
    out = tmp_path_factory.mktemp("layout")
    bundle = fx.export_layout_bundle(module, out)
    return module, out, bundle


# -- preprocessing -----------------------------------------------------------


def test_bicubic_resize_matches_torchvision_uint8_kernel():
    """The reference is torchvision's *uint8* bicubic, not float bicubic.

    ``transformers``' fast processor resizes the uint8 tensor, and for bicubic
    on CPU torchvision passes uint8 straight to ``interpolate``, which saturates
    between the separable passes. Comparing against float ``F.interpolate``
    instead hides a 20-grey-level divergence on sharp edges — page scans are
    made of sharp edges — so pin the kernel that actually runs.
    """
    pytest.importorskip("torchvision")
    import torchvision.transforms.v2.functional as tvF
    from torchvision.transforms import InterpolationMode

    rng = np.random.default_rng(0)
    for h, w in ((100, 70), (33, 129), (64, 64), (17, 5)):
        img = rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)
        ours = resize_bicubic_no_antialias(img, 40, 24).astype(np.int32)
        ref = (
            tvF.resize(
                torch.from_numpy(img.copy()),
                [40, 24],
                interpolation=InterpolationMode.BICUBIC,
                antialias=False,
            )
            .numpy()
            .astype(np.int32)
        )
        # ±1-2 grey levels: ATen quantises the tap weights, we keep them float.
        assert np.abs(ours - ref).max() <= 2


def test_resize_identity_when_size_matches():
    img = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)
    assert np.array_equal(resize_bicubic_no_antialias(img, 8, 8), img)


def test_preprocess_matches_hf_image_processor_within_rounding():
    """End-to-end pixel_values vs transformers' PPDocLayoutV3 processor.

    Within one grey level everywhere. The bound is tight on purpose: at the
    20 levels a naive float bicubic gives, the detector returns different
    regions (measured by ``scripts/export_onnx_layout.py``'s parity run), so a
    loose bound here would let a real behaviour change through.
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    proc_cls = getattr(transformers, "PPDocLayoutV3ImageProcessor", None)
    if proc_cls is None:
        pytest.skip("transformers without PPDocLayoutV3")
    proc = proc_cls(size={"height": 64, "width": 64})
    pages = fx.sample_pages()
    hf = proc(images=pages, return_tensors="pt")["pixel_values"].numpy()
    ours = preprocess_images(
        pages, size=(64, 64), rescale_factor=1 / 255, image_mean=[0, 0, 0], image_std=[1, 1, 1]
    )
    assert hf.shape == ours.shape == (len(pages), 3, 64, 64)
    diff = np.abs(hf - ours) * 255
    assert diff.max() <= 1  # ATen quantises the tap weights, we keep them float
    assert (diff > 0.5).mean() < 0.02


# -- postprocessing -----------------------------------------------------------


def test_decode_matches_hf_post_process_object_detection():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("cv2")
    proc_cls = getattr(transformers, "PPDocLayoutV3ImageProcessor", None)
    if proc_cls is None:
        pytest.skip("transformers without PPDocLayoutV3")
    proc = proc_cls(size={"height": 64, "width": 64})
    rng = np.random.default_rng(1)
    b, q, c = 2, 8, 5
    logits = rng.normal(0, 2, size=(b, q, c)).astype(np.float32)
    boxes = np.clip(rng.uniform(0.1, 0.9, size=(b, q, 4)), 0.05, 0.95).astype(np.float32)
    boxes[..., 2:] *= 0.3
    order = rng.normal(0, 3, size=(b, q, q)).astype(np.float32)
    sizes = [(120, 80), (200, 150)]
    hf = proc.post_process_object_detection(
        fx.fake_hf_outputs(logits, boxes, order),
        threshold=0.3,
        target_sizes=torch.tensor(sizes, dtype=torch.float32),
    )
    ours = decode_detections(logits, boxes, order, sizes, threshold=0.3)
    for h, o in zip(hf, ours, strict=True):
        assert len(o["scores"]) == len(h["scores"]) > 0
        np.testing.assert_allclose(o["scores"], h["scores"].numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(o["labels"], h["labels"].numpy())
        np.testing.assert_allclose(o["boxes"], h["boxes"].numpy(), rtol=1e-5, atol=1e-4)
        np.testing.assert_array_equal(o["order_seq"], h["order_seq"].numpy())


def test_order_sequences_is_a_permutation():
    rng = np.random.default_rng(2)
    seq = order_sequences(rng.normal(size=(3, 7, 7)).astype(np.float32))
    for row in seq:
        assert sorted(row.tolist()) == list(range(7))


# -- the backend + detector --------------------------------------------------


def test_backend_matches_torch_module_end_to_end(layout_bundle):
    module, _out, bundle = layout_bundle
    backend = OnnxLayoutBackend(bundle, device="cpu", threshold=0.3)
    pages = fx.sample_pages()
    pixel_values = preprocess_images(
        pages, size=(64, 64), rescale_factor=1 / 255, image_mean=[0, 0, 0], image_std=[1, 1, 1]
    )
    with torch.no_grad():
        t_logits, t_boxes, t_order = (x.numpy() for x in module(torch.from_numpy(pixel_values)))
    o_logits, o_boxes, o_order = backend.forward(pixel_values)
    assert np.abs(t_logits - o_logits).max() < 1e-4
    assert np.abs(t_boxes - o_boxes).max() < 1e-5
    assert np.abs(t_order - o_order).max() < 1e-4

    expected = decode_detections(
        t_logits, t_boxes, t_order, [(p.height, p.width) for p in pages], 0.3
    )
    got = backend.run(pages)
    for e, g in zip(expected, got, strict=True):
        np.testing.assert_array_equal(e["labels"], g["labels"])
        np.testing.assert_allclose(e["scores"], g["scores"], atol=1e-5)
        np.testing.assert_array_equal(e["order_seq"], g["order_seq"])


def test_local_detector_selects_onnx_runtime_and_unloads(layout_bundle, monkeypatch):
    _module, out, _bundle = layout_bundle
    from bibr.config import GlobalSettings
    from bibr.local.layout import LayoutDetector

    monkeypatch.setenv("ML_RUNTIME", "onnx")
    monkeypatch.setenv("LAYOUT_ONNX_MODEL_ID", str(out))
    settings = GlobalSettings()
    det = LayoutDetector(device="cpu", settings=settings)
    assert det._runtime == "onnx"
    assert det._device.type == "cpu"
    assert det.loaded

    import asyncio

    regions = asyncio.run(det.detect_batch(fx.sample_pages()))
    assert len(regions) == 2
    for page in regions:
        for r in page:
            assert set(r) >= {"label", "bbox_2d", "score", "read_order", "task_type"}
            assert 0 <= r["bbox_2d"][0] <= r["bbox_2d"][2] <= 1000
    det.unload()
    assert not det.loaded


def test_onnx_mode_without_bundle_is_a_configuration_error(monkeypatch, tmp_path):
    from bibr.config import GlobalSettings
    from bibr.exceptions import ConfigurationError
    from bibr.local.layout import LayoutDetector

    monkeypatch.setenv("ML_RUNTIME", "onnx")
    monkeypatch.setenv("LAYOUT_ONNX_MODEL_ID", str(tmp_path))  # exists, but no onnx/
    with pytest.raises(ConfigurationError, match="LAYOUT_ONNX_MODEL_ID"):
        LayoutDetector(device="cpu", settings=GlobalSettings())


def test_serve_detector_runs_onnx_through_the_batcher(layout_bundle, monkeypatch):
    _module, out, _bundle = layout_bundle
    from bibr.config import GlobalSettings
    from bibr.serve.deployments.layout import LayoutDetector

    monkeypatch.setenv("ML_RUNTIME", "onnx")
    monkeypatch.setenv("LAYOUT_ONNX_MODEL_ID", str(out))
    det = LayoutDetector(device="cpu", settings=GlobalSettings())
    assert det._runtime == "onnx"

    import asyncio

    async def run():
        try:
            return await det.detect_batch(fx.sample_pages())
        finally:
            await det.aclose()

    regions = asyncio.run(run())
    assert len(regions) == 2
