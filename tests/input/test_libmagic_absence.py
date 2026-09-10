"""libmagic is a system library, not something pip can supply.

``python-magic`` raises at import time when the shared library is missing, so a
fresh macOS/Linux install used to take down ``import bibr`` and every extraction
with a bare "failed to find libmagic" (issue #64). The binding is imported
defensively instead: the failure is deferred to first use and reported as a
named, fixable ``bibr doctor`` check.
"""

import pytest

from bibr.input import validate


@pytest.fixture
def missing_libmagic(monkeypatch):
    """Simulate a host whose system libmagic could not be loaded."""
    monkeypatch.setattr(validate, "magic", None)
    monkeypatch.setattr(validate, "_MAGIC_IMPORT_ERROR", ImportError("failed to find libmagic"))


def test_reason_is_none_when_libmagic_loads():
    assert validate.libmagic_unavailable_reason() is None


def test_reason_names_the_macos_install_command(monkeypatch, missing_libmagic):
    monkeypatch.setattr(validate.sys, "platform", "darwin")
    assert "brew install libmagic" in validate.libmagic_unavailable_reason()


def test_reason_names_the_linux_install_commands(monkeypatch, missing_libmagic):
    monkeypatch.setattr(validate.sys, "platform", "linux")
    reason = validate.libmagic_unavailable_reason()
    assert "libmagic1" in reason
    assert "file-libs" in reason


def test_reason_falls_back_on_an_unknown_platform(monkeypatch, missing_libmagic):
    monkeypatch.setattr(validate.sys, "platform", "sunos5")
    reason = validate.libmagic_unavailable_reason()
    assert "libmagic is not installed" in reason


def test_detect_mime_type_raises_the_actionable_message(monkeypatch, missing_libmagic):
    monkeypatch.setattr(validate.sys, "platform", "darwin")
    with pytest.raises(ImportError, match="brew install libmagic"):
        validate.detect_mime_type(b"%PDF-1.7")


def test_detect_mime_type_works_when_libmagic_is_present():
    assert validate.detect_mime_type(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3") == "application/pdf"
