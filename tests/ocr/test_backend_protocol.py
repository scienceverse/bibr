"""`OcrBackend` Protocol conformance for existing OCR clients."""

from typing import get_type_hints

import pytest


def test_ocr_backend_protocol_exists():
    from bibr.ocr.backend import OcrBackend

    hints = get_type_hints(OcrBackend)
    assert "name" in hints


def test_ocr_backend_runtime_checkable():
    """The Protocol is runtime_checkable so `isinstance(x, OcrBackend)` works."""
    from bibr.ocr.backend import OcrBackend

    # `runtime_checkable` sets _is_runtime_protocol = True on the Protocol.
    assert getattr(OcrBackend, "_is_runtime_protocol", False) is True


def test_ocr_backend_declares_async_methods():
    from bibr.ocr.backend import OcrBackend

    for method_name in ("recognize", "wait_for_server", "shutdown"):
        assert hasattr(OcrBackend, method_name), f"missing {method_name}"


@pytest.mark.parametrize(
    "method_name",
    ["recognize", "wait_for_server", "shutdown"],
)
def test_ocr_backend_method_is_async(method_name):
    import inspect

    from bibr.ocr.backend import OcrBackend

    member = getattr(OcrBackend, method_name)
    assert inspect.iscoroutinefunction(member), f"{method_name} must be async"


class _FakeBackend:
    name = "fake-backend"

    def __init__(self, **_kw: object) -> None:
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def recognize(self, image, prompt: str) -> str:
        return "ok"

    async def wait_for_server(self) -> None:
        return None

    async def shutdown(self) -> None:
        self._loaded = False


def test_registry_register_and_create():
    from bibr.ocr import registry
    from bibr.ocr.backend import OcrBackend

    registered = registry.register(_FakeBackend)
    assert registered is _FakeBackend

    instance = registry.create("fake-backend", ocr_url="ignored")
    assert isinstance(instance, OcrBackend)
    assert instance.name == "fake-backend"


def test_registry_unknown_name_raises():
    from bibr.ocr import registry

    with pytest.raises(ValueError) as exc:
        registry.create("nope-not-real")
    assert "nope-not-real" in str(exc.value)


def test_registry_duplicate_name_raises():
    from bibr.ocr import registry

    class _Dup:
        name = "dup-backend"

    class _DupConflict:
        name = "dup-backend"

    registry.register(_Dup)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_DupConflict)


def test_registry_missing_name_raises():
    from bibr.ocr import registry

    class _NoName:
        pass

    with pytest.raises(ValueError, match="class-level `name` attribute"):
        registry.register(_NoName)


@pytest.mark.parametrize(
    "class_path,expected_name",
    [
        ("bibr.local.ocr.LlamaCppOcrClient", "glm-llama"),
        ("bibr.local.ocr.HttpOcrClient", "glm-http"),
        ("bibr.local.ocr.VllmMlxOcrClient", "glm-mlx"),
    ],
)
def test_real_clients_declare_name(class_path, expected_name):
    import importlib

    module_path, attr = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), attr)
    assert cls.name == expected_name


@pytest.mark.parametrize(
    "class_path",
    [
        "bibr.local.ocr.LlamaCppOcrClient",
        "bibr.local.ocr.HttpOcrClient",
    ],
)
def test_real_clients_shutdown_is_async(class_path):
    import importlib
    import inspect

    module_path, attr = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), attr)
    assert inspect.iscoroutinefunction(cls.shutdown), f"{class_path}.shutdown must be async"


@pytest.mark.parametrize(
    "class_path",
    [
        "bibr.local.ocr.LlamaCppOcrClient",
        "bibr.local.ocr.HttpOcrClient",
    ],
)
def test_real_clients_have_wait_for_server(class_path):
    import importlib
    import inspect

    module_path, attr = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), attr)
    assert hasattr(cls, "wait_for_server")
    assert inspect.iscoroutinefunction(cls.wait_for_server)


def test_supported_clients_are_registered():
    from bibr.local import ocr as _ocr  # noqa: F401 — importing triggers @register
    from bibr.ocr import registry

    assert {"glm-llama", "glm-http", "glm-mlx"}.issubset(registry.known_backends())
    removed_name = "fal" + "con"
    assert not any(removed_name in name for name in registry.known_backends())


def test_resource_manager_ocr_kwargs_shape():
    """`ResourceManager._ocr_kwargs()` returns a dict of all backend-relevant fields."""
    from bibr.pipeline.resources import ResourceManager

    rm = ResourceManager(
        ocr_backend="glm-llama",
        ocr_url="http://example.invalid",
        ocr_model="custom/model",
        device="cpu",
    )
    kwargs = rm._ocr_kwargs()
    assert kwargs["base_url"] == "http://example.invalid"
    assert kwargs["model_path"] == "custom/model"
    assert kwargs["device"] == "cpu"


def test_resource_manager_default_keeps_the_paddle_startup_selector():
    """Direct ResourceManager users receive the same OCR default as LocalPipeline."""
    from bibr.pipeline.resources import ResourceManager

    assert ResourceManager().ocr_backend == "paddle"


def test_resource_manager_create_ocr_dispatches_via_registry(monkeypatch):
    """`_create_ocr_client` delegates to `bibr.ocr.registry.create`."""
    from bibr.ocr import registry
    from bibr.pipeline.resources import ResourceManager

    captured: dict = {}

    def fake_create(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs

        class _Stub:
            loaded = True

        return _Stub()

    monkeypatch.setattr(registry, "create", fake_create)

    rm = ResourceManager(ocr_backend="glm-http", ocr_url="http://foo")
    rm._create_ocr_client()
    assert captured["name"] == "glm-http"
    assert captured["kwargs"]["base_url"] == "http://foo"


def test_resource_manager_ocr_url_forces_glm_http():
    from bibr.pipeline.resources import ResourceManager

    rm = ResourceManager(ocr_backend="glm-llama", ocr_url="http://example.invalid")
    assert rm._resolve_ocr_backend_name() == "glm-http"
