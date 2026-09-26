"""DOCX footnote loader test."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def docx_with_footnotes():
    """Committed synthetic fixture with two real footnotes (plus separators)."""
    return Path(__file__).parent.parent / "fixtures" / "footnotes_sample.docx"


def test_load_footnotes_returns_dict(docx_with_footnotes):
    pytest.importorskip("docx")
    from docx import Document

    from bibr.input.docx_footnotes import load_footnotes

    doc = Document(str(docx_with_footnotes))
    fns = load_footnotes(doc)
    assert fns == {
        "2": "First synthetic footnote text.",
        "3": "Second synthetic footnote with split runs.",
    }


def test_load_footnotes_empty_when_no_footnotes_part():
    """A doc with no footnotes part yields an empty dict."""
    from bibr.input.docx_footnotes import load_footnotes

    doc = MagicMock()
    doc.part.footnotes_part = None
    doc.part.rels = {}
    assert load_footnotes(doc) == {}


class _BlobPart:
    """Footnotes part without a pre-parsed ``element`` — forces the blob path."""

    def __init__(self, blob: bytes):
        self.blob = blob


def _doc_with_blob(blob: bytes):
    doc = MagicMock()
    part = MagicMock(spec=["footnotes_part"])
    part.footnotes_part = _BlobPart(blob)
    doc.part = part
    return doc


def test_blob_fallback_parses_normal_footnotes():
    """Sanity: the hardened parser still reads ordinary footnotes."""
    from bibr.input.docx_footnotes import load_footnotes

    blob = (
        b'<?xml version="1.0"?>'
        b'<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b'<w:footnote w:id="1"><w:p><w:r><w:t>A normal footnote.</w:t></w:r></w:p></w:footnote>'
        b"</w:footnotes>"
    )
    assert load_footnotes(_doc_with_blob(blob)) == {"1": "A normal footnote."}


def test_blob_fallback_does_not_resolve_external_entities(tmp_path):
    """XXE: an external entity must never leak local file contents."""
    from bibr.input.docx_footnotes import load_footnotes

    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    blob = (
        b'<?xml version="1.0"?>'
        b'<!DOCTYPE w:footnotes [<!ENTITY xxe SYSTEM "file://' + str(secret).encode() + b'">]>'
        b'<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b'<w:footnote w:id="1"><w:p><w:r><w:t>&xxe;</w:t></w:r></w:p></w:footnote>'
        b"</w:footnotes>"
    )
    result = load_footnotes(_doc_with_blob(blob))
    assert "TOP-SECRET-CONTENT" not in "".join(result.values())


def test_blob_fallback_survives_entity_expansion_bomb():
    """Billion-laughs: nested entity expansion must not be expanded."""
    from bibr.input.docx_footnotes import load_footnotes

    blob = (
        b'<?xml version="1.0"?>'
        b"<!DOCTYPE w:footnotes ["
        b'<!ENTITY a "aaaaaaaaaa">'
        b'<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
        b'<!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">'
        b'<!ENTITY d "&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;">'
        b"]>"
        b'<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b'<w:footnote w:id="1"><w:p><w:r><w:t>&d;</w:t></w:r></w:p></w:footnote>'
        b"</w:footnotes>"
    )
    result = load_footnotes(_doc_with_blob(blob))
    # The expansion (10^4 chars here, unbounded in the wild) must not appear.
    assert all(len(v) < 1000 for v in result.values())
