"""libmagic is an optional hint, never a requirement.

``python-magic`` is only a binding: it raises at import time when the system
libmagic library is missing, which a plain ``pip install`` cannot supply. Input
types are decided by built-in rules, so the binding is imported lazily, only for
content those rules do not recognize, and any failure to import or use it just
means no hint.
"""

import subprocess
import sys
import textwrap

import pytest

from bibr.input import sniff
from bibr.input.validate import validate_input_file

_PDF = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
_GIF = b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"


def test_missing_python_magic_is_not_an_error(no_libmagic):
    assert sniff.libmagic_available() is False
    assert sniff.detect_mime_type(_PDF) == "application/pdf"

    result = validate_input_file("paper.pdf", _PDF)

    assert result.is_valid is True


def test_libmagic_that_fails_on_first_use_is_treated_as_missing(fake_libmagic):
    def broken(_content, mime=False):
        raise OSError("could not find any valid magic files")

    fake_libmagic(broken)

    assert sniff.libmagic_available() is False
    assert sniff.detect_mime_type(_GIF) == "application/octet-stream"
    assert validate_input_file("paper.pdf", _PDF).is_valid is True


def test_hint_names_content_the_built_in_rules_do_not_recognize(fake_libmagic):
    fake_libmagic(lambda _content, mime=False: "image/gif")

    assert sniff.libmagic_available() is True
    assert sniff.detect_mime_type(_GIF) == "image/gif"


def test_hint_is_not_consulted_for_recognized_formats(fake_libmagic):
    calls = []

    def spy(content, mime=False):
        calls.append(content)
        return "image/gif"

    fake_libmagic(spy)
    sniff.libmagic_available()
    calls.clear()

    assert sniff.detect_mime_type(_PDF) == "application/pdf"
    assert calls == []


def test_hint_cannot_claim_a_type_the_built_in_rules_decide(fake_libmagic):
    """A libmagic guess of PDF for plain text must not reject the file."""
    fake_libmagic(lambda _content, mime=False: "application/pdf")
    text = b"Plain notes about a paper.\n"

    assert sniff.detect_mime_type(text) == "text/plain"
    result = validate_input_file("notes.docx", text)
    assert result.is_supported is True


def test_hint_failure_at_call_time_falls_back(fake_libmagic):
    def flaky(content, mime=False):
        if content == _GIF:
            raise RuntimeError("magic_buffer failed")
        return "application/octet-stream"

    fake_libmagic(flaky)

    assert sniff.libmagic_available() is True
    assert sniff.detect_mime_type(_GIF) == "application/octet-stream"


def test_real_libmagic_names_unrecognized_content_when_installed():
    sniff._load_libmagic.cache_clear()
    if not sniff.libmagic_available():
        pytest.skip("libmagic is not installed on this host")
    assert sniff.detect_mime_type(_GIF) == "image/gif"


def test_validating_supported_formats_does_not_import_python_magic():
    script = textwrap.dedent(
        f"""
        import sys
        from bibr.input.validate import validate_input_file

        result = validate_input_file("paper.pdf", {_PDF!r})
        assert result.is_valid, result
        print("magic" in sys.modules)
        """
    )
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and inline script
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False"
