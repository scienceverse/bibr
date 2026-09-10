"""BaseLayoutDetector core: PIL interface, compile-gated padding, throttled cache."""

import builtins
import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from bibr import layout_base  # noqa: E402
from bibr.config import GlobalSettings  # noqa: E402
from bibr.layout_base import BaseLayoutDetector  # noqa: E402
from bibr.layout_utils import _CORRECT_ID2LABEL, _MAX_BATCH_SIZE  # noqa: E402


def test_missing_ml_extra_explains_cloud_ocr_pdf_dependency(monkeypatch):
    real_import = builtins.__import__

    def reject_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch intentionally unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_torch)

    with pytest.raises(
        ImportError,
        match=r"PDF processing.*cloud OCR.*uv sync --extra torch",
    ):
        BaseLayoutDetector()


def _bare_detector(compiled: bool):
    """A detector instance wired for CPU inference without loading a model."""
    det = object.__new__(BaseLayoutDetector)
    det.threshold = 0.5
    det._id2label = _CORRECT_ID2LABEL
    det._device = torch.device("cpu")
    det._compiled = compiled
    det._settings = GlobalSettings()
    det._pad_image = Image.new("RGB", (640, 480))

    captured: dict = {}

    class _FakeProcessor:
        def __call__(self, images, return_tensors):
            captured["n_images"] = len(images)
            return {"pixel_values": torch.zeros(len(images), 1)}

        def post_process_object_detection(self, outputs, threshold, target_sizes):
            n = len(target_sizes)
            return [
                {
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                    "boxes": torch.zeros(0, 4),
                }
                for _ in range(n)
            ]

    det._image_processor = _FakeProcessor()
    det._model = lambda **inputs: {"pred": inputs["pixel_values"]}
    return det, captured


def test_eager_path_does_not_pad_the_batch():
    det, captured = _bare_detector(compiled=False)
    imgs = [Image.new("RGB", (800, 1000)), Image.new("RGB", (800, 1000))]
    out = det._detect_pytorch(imgs, [(1000, 800), (1000, 800)])
    assert captured["n_images"] == 2  # exactly the real images, no dummy padding
    assert len(out) == 2


def test_compiled_path_pads_to_max_batch_size():
    det, captured = _bare_detector(compiled=True)
    imgs = [Image.new("RGB", (800, 1000)), Image.new("RGB", (800, 1000))]
    out = det._detect_pytorch(imgs, [(1000, 800), (1000, 800)])
    assert captured["n_images"] == _MAX_BATCH_SIZE  # padded so torch.compile won't recompile
    assert len(out) == 2  # results sliced back to real images


def test_detect_images_derives_orig_sizes_from_pil_dims():
    det = object.__new__(BaseLayoutDetector)
    captured: dict = {}
    det._detect_pytorch = lambda images, orig_sizes: (
        captured.setdefault(  # type: ignore[method-assign]
            "orig_sizes", orig_sizes
        )
        or [[] for _ in images]
    )
    imgs = [Image.new("RGB", (800, 1000)), Image.new("RGB", (640, 480))]
    out = det._detect_images(imgs)
    assert captured["orig_sizes"] == [(1000, 800), (480, 640)]  # (height, width)
    assert len(out) == 2


def test_detect_images_halves_batch_after_oom_preserving_order():
    det = object.__new__(BaseLayoutDetector)
    det._compiled = True
    calls = []

    def detect(images, orig_sizes):
        calls.append(len(images))
        if len(images) > 2:
            raise RuntimeError("CUDA out of memory")
        return [[{"width": image.width}] for image in images]

    det._detect_pytorch = detect  # type: ignore[method-assign]
    images = [Image.new("RGB", (width, 100)) for width in range(10, 15)]

    out = det._detect_images(images)

    assert [rows[0]["width"] for rows in out] == [10, 11, 12, 13, 14]
    assert calls == [5, 2, 3, 1, 2]
    assert det._compiled is False


def test_maybe_empty_cache_throttles_to_one_call_per_interval(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    fake_now = [1000.0]
    monkeypatch.setattr(layout_base.time, "monotonic", lambda: fake_now[0])
    monkeypatch.setattr(layout_base, "_last_empty_cache_time", 0.0)

    layout_base._maybe_empty_cache()  # first call allowed
    assert len(calls) == 1

    fake_now[0] += 5.0
    layout_base._maybe_empty_cache()  # within 30s window → throttled
    assert len(calls) == 1

    fake_now[0] += 30.0
    layout_base._maybe_empty_cache()  # window elapsed → allowed again
    assert len(calls) == 2


def test_init_disables_hf_cache_symlinks_before_transformers_load(monkeypatch):
    pytest.importorskip("transformers")
    import huggingface_hub.constants as hf_constants
    import transformers

    image_processor_cls = transformers.AutoImageProcessor
    model_cls = transformers.AutoModelForObjectDetection
    calls = []

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_SYMLINKS", False)
    monkeypatch.setattr("bibr.utils.device.detect_torch_device", lambda: "cpu")
    monkeypatch.setattr("bibr.utils.device.report_device", lambda *a, **k: None)

    class _FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_from_pretrained(model_id, **kwargs):  # noqa: ARG001
        calls.append((model_id, os.environ.get("HF_HUB_DISABLE_SYMLINKS")))
        return object()

    monkeypatch.setattr(image_processor_cls, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        model_cls,
        "from_pretrained",
        lambda model_id, **kwargs: _FakeModel(),  # noqa: ARG005
    )

    class _Detector(BaseLayoutDetector):
        def _install_model(self, model):
            self._model = model

    _Detector()

    assert calls == [("PaddlePaddle/PP-DocLayoutV3_safetensors", "1")]
    assert hf_constants.HF_HUB_DISABLE_SYMLINKS is True
