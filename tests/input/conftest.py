"""Fixtures that control the optional libmagic hint used by input sniffing."""

import sys
import types

import pytest

from bibr.input import sniff


@pytest.fixture
def no_libmagic(monkeypatch):
    """Make ``import magic`` fail, as on a host without python-magic or libmagic."""
    # A None entry in sys.modules makes the import statement raise ImportError.
    monkeypatch.setitem(sys.modules, "magic", None)
    sniff._load_libmagic.cache_clear()
    yield
    sniff._load_libmagic.cache_clear()


@pytest.fixture
def fake_libmagic(monkeypatch):
    """Install a stand-in ``magic`` module whose ``from_buffer`` is *from_buffer*."""

    def install(from_buffer):
        module = types.ModuleType("magic")
        module.from_buffer = from_buffer
        monkeypatch.setitem(sys.modules, "magic", module)
        sniff._load_libmagic.cache_clear()
        return module

    yield install
    sniff._load_libmagic.cache_clear()
