"""Stateless layout and segmenter factory registries."""

import pytest

from bibr.config import GlobalSettings
from bibr.layout import registry as layout_registry
from bibr.segmenter import registry as segmenter_registry

SETTINGS = GlobalSettings()


def test_layout_registry_creates_a_fresh_instance():
    @layout_registry.register
    class FakeLayout:
        name = "test-layout-fresh"

        def __init__(self, *, device=None, settings=None):
            self.device = device
            self.settings = settings

    first = layout_registry.create("test-layout-fresh", device="cpu", settings=SETTINGS)
    second = layout_registry.create("test-layout-fresh", device="cpu", settings=SETTINGS)

    assert first is not second
    assert first.device == "cpu"
    assert first.settings is SETTINGS


def test_segmenter_registry_creates_a_fresh_instance():
    @segmenter_registry.register
    class FakeSegmenter:
        name = "test-segmenter-fresh"

        def __init__(self, *, settings=None):
            self.settings = settings

    first = segmenter_registry.create("test-segmenter-fresh", settings=SETTINGS)
    second = segmenter_registry.create("test-segmenter-fresh", settings=SETTINGS)

    assert first is not second
    assert first.settings is SETTINGS


@pytest.mark.parametrize("registry", [layout_registry, segmenter_registry])
def test_registry_rejects_duplicate_names(registry):
    class First:
        name = f"duplicate-{registry.__name__}"

    class Second:
        name = First.name

    registry.register(First)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Second)


@pytest.mark.parametrize("registry", [layout_registry, segmenter_registry])
def test_registry_unknown_name_lists_known_backends(registry):
    with pytest.raises(ValueError, match="known") as exc_info:
        registry.create("does-not-exist")

    assert registry.known_backends()
    assert registry.known_backends()[0] in str(exc_info.value)
