"""References cut at the NER parser's token window are counted and reported.

The parser tags only the first ``max_seq_len`` tokens; a reference with a long
author list lost its trailing pages, DOI or URL with no trace in the export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from bibr.config import GlobalSettings
from bibr.extract import ref_extractor as ex
from bibr.ner.parser_onnx import OnnxRefParser, count_truncated
from bibr.processing_warnings import WarningCode

# Torch-free tiny bundle (max_length 16); see scripts/generate_onnx_test_bundles.py.
NER_BUNDLE = Path(__file__).parent.parent / "fixtures" / "onnx" / "ner" / "onnx"

SHORT = "Doe A. 2021. Sleep. Journal 12: 1."
LONG = "Smith J, Doe A, Roe B, Poe C, Moe D, Loe E (2020). Memory. Journal, 26(1), 1-12. doi:10.1/x"


def test_onnx_parser_counts_references_cut_at_its_window(caplog):
    parser = OnnxRefParser(NER_BUNDLE, device="cpu")
    assert parser.max_seq_len == 16

    with caplog.at_level(logging.DEBUG, logger="bibr.ner.parser_onnx"):
        parser.parse_batch([SHORT, "", LONG, LONG], batch_size=2)
    assert parser.last_truncated_count == 2
    assert "cut at 16 tokens" in caplog.text

    parser.parse_batch([SHORT])
    assert parser.last_truncated_count == 0
    parser.parse(LONG)
    assert parser.last_truncated_count == 1


def test_count_truncated_ignores_trailing_whitespace_and_short_rows():
    text = "a b c   "
    offsets = [(0, 1), (2, 3), (4, 5)]
    assert count_truncated([text], [offsets], 3) == 0  # only whitespace after the window
    assert count_truncated([text + "d"], [offsets], 3) == 1
    assert count_truncated([text + "d"], [offsets], 4) == 0  # not at the limit


class _Parser:
    def __init__(self, truncated):
        self.last_truncated_count = truncated

    def parse_batch(self, texts, batch_size=32):
        return [{"title": "A title", "authors": "Doe, A."} for _ in texts]


def _extract(monkeypatch, parser):
    monkeypatch.setattr(ex, "_get_ner_parser", lambda settings, memory_mode=None: parser)
    contents = SimpleNamespace(processing_warnings=[])
    extractor = ex.ReferenceExtractor(contents=contents, file_hash="x", settings=GlobalSettings())
    refs = extractor._parse_references_ner_aligned(["Doe, A. A title.", "Doe, A. A title."])
    assert all(ref is not None for ref in refs)
    return contents.processing_warnings


def test_truncated_references_surface_as_a_processing_warning(monkeypatch):
    warnings = _extract(monkeypatch, _Parser(2))
    assert [w.code for w in warnings] == [WarningCode.REF_PARSE_TRUNCATED]
    assert warnings[0].message.startswith("2 reference(s)")


def test_no_warning_without_truncation(monkeypatch):
    assert _extract(monkeypatch, _Parser(0)) == []
    assert _extract(monkeypatch, SimpleNamespace(parse_batch=_Parser(0).parse_batch)) == []
