"""Missing 'torch'-extra dependencies must raise a clear ImportError, not a bare one."""

import builtins
import importlib
import sys

import pytest

_MATCH = r"requires the 'torch' extra"


def _forget_module(monkeypatch, module):
    """Drop ``module`` (and its submodules) so it can be re-imported, reversibly.

    ``monkeypatch.delitem(sys.modules, ...)`` alone is not enough: re-importing
    rebinds the *parent package's* attribute to the new module object, and that
    rebinding outlives the test. Anything that imported the old module keeps
    functions whose ``__globals__`` are the old dict, while ``from pkg import
    mod`` in a later test resolves the new one — so a monkeypatch applied to the
    module no longer reaches the code under test, and it silently calls the real
    thing. Record the parent attribute too, so teardown puts both back.
    """
    for name in list(sys.modules):
        if name != module and not name.startswith(module + "."):
            continue
        parent, _, leaf = name.rpartition(".")
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], leaf, sys.modules[name], raising=False)
        monkeypatch.delitem(sys.modules, name, raising=False)


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
    assert "pip install 'bibr[torch]'" in str(err)
    assert "uv sync --extra torch" in str(err)


@pytest.mark.parametrize(
    ("module", "blocked"),
    [
        ("bibr.ner.parser", ("torch", "transformers", "torchcrf")),
        ("bibr.ner.segmenter", ("torch", "transformers", "torchcrf")),
        ("bibr.ner.model", ("torch", "transformers", "torchcrf")),
        ("bibr.structure.section_classifier_model", ("torch", "transformers", "safetensors")),
        ("bibr.structure._section_minilm_arch", ("torch", "transformers")),
        ("bibr.structure.paper_classifier_model", ("torch", "transformers", "safetensors")),
        ("bibr.structure._paper_classifier_arch", ("torch", "transformers")),
    ],
)
def test_module_import_raises_ml_extra_error(monkeypatch, module, blocked):
    _block_imports(monkeypatch, blocked)
    _forget_module(monkeypatch, module)
    with pytest.raises(ImportError, match=_MATCH):
        importlib.import_module(module)


@pytest.mark.parametrize(
    "module",
    [
        # Core-install modules: importable without torch, transformers or cv2.
        "bibr.utils.device",
        "bibr.ocr.image_processing",
        "bibr.structure.section_classifier_common",
        "bibr.structure.section_classifier_onnx",
        "bibr.structure.paper_classifier_common",
        "bibr.structure.paper_classifier_onnx",
        "bibr.ner.crf_numpy",
        "bibr.ner.parser_onnx",
        "bibr.ner.runtime",
        "bibr.layout_onnx",
        "bibr.layout_base",
        "bibr.utils.ml_runtime",
    ],
)
def test_core_modules_import_without_torch(monkeypatch, module):
    _block_imports(monkeypatch, ("torch", "transformers", "torchcrf", "safetensors", "cv2"))
    _forget_module(monkeypatch, module)
    importlib.import_module(module)


def test_device_functions_needing_torch_raise_the_extra_error(monkeypatch):
    import bibr.utils.device as device

    _block_imports(monkeypatch, ("torch",))
    monkeypatch.setattr(device, "torch", None)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with pytest.raises(ImportError, match=_MATCH):
        device.detect_torch_device()
    # Reporting a device never needs torch (the ONNX path logs through it).
    assert device.cuda_incompatibility() is None
    device.report_device("test component", "cpu")


def test_get_ort_providers_names_onnxruntime_when_missing(monkeypatch):
    from bibr.utils import onnx_providers

    _block_imports(monkeypatch, ("onnxruntime",))
    with pytest.raises(ImportError, match=r"requires onnxruntime"):
        onnx_providers.get_ort_providers()
