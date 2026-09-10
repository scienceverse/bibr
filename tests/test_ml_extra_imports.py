"""Missing 'ml'-extra dependencies must raise a clear ImportError, not a bare one."""

import builtins
import importlib
import sys

import pytest

_MATCH = r"requires the 'ml' extra"


def _block_imports(monkeypatch, blocked):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in blocked or any(name.startswith(b + ".") for b in blocked):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_ml_import_error_message():
    from bibr.utils.ml_extra import ml_import_error

    err = ml_import_error("layout detection")
    assert isinstance(err, ImportError)
    assert "pip install 'bibr[ml]'" in str(err)
    assert "uv sync --extra ml" in str(err)


@pytest.mark.parametrize(
    ("module", "blocked"),
    [
        ("bibr.ner.parser", ("torch", "transformers", "torchcrf")),
        ("bibr.ner.segmenter", ("torch", "transformers", "torchcrf")),
        ("bibr.ner.model", ("torch", "transformers", "torchcrf")),
        ("bibr.structure.section_classifier_model", ("torch", "transformers", "safetensors")),
        ("bibr.structure._section_minilm_arch", ("torch", "transformers")),
        ("bibr.utils.device", ("torch",)),
        ("bibr.ocr.image_processing", ("cv2",)),
    ],
)
def test_module_import_raises_ml_extra_error(monkeypatch, module, blocked):
    _block_imports(monkeypatch, blocked)
    monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.raises(ImportError, match=_MATCH):
        importlib.import_module(module)


def test_get_ort_providers_raises_ml_extra_error(monkeypatch):
    from bibr.utils import onnx_providers

    _block_imports(monkeypatch, ("onnxruntime",))
    with pytest.raises(ImportError, match=_MATCH):
        onnx_providers.get_ort_providers()
