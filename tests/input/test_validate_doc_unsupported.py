"""`.doc` (legacy Word) is rejected at validate."""

import pytest

from bibr.exceptions import InputValidationError
from bibr.input.validate import validate_input_file


def test_doc_extension_is_unsupported():
    with pytest.raises(InputValidationError) as exc:
        validate_input_file("legacy.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    assert ".doc" in str(exc.value)
    assert "unsupported" in str(exc.value).lower()


def test_docx_extension_still_supported():
    result = validate_input_file(
        "modern.docx",
        b"PK\x03\x04some-zip-bytes",
    )
    assert result.is_supported is True
