"""Regression tests for Fix 1D (copy-on-write JSON sanitizer and compiled surrogate stripper)."""

from __future__ import annotations

from bibr.export.json_export import _sanitize_json_strings
from bibr.utils.text import strip_lone_surrogates


def test_sanitize_json_strings_zero_copy_when_clean_and_pure_when_dirty() -> None:
    clean = {"a": "hello", "b": ["world", "naïve", 42, None], "c": {"k": "v"}}
    assert _sanitize_json_strings(clean) is clean

    dirty_val = {"a": "ok", "nested": ["clean", "bad\ud800char"]}
    out = _sanitize_json_strings(dirty_val)
    assert out == {"a": "ok", "nested": ["clean", "bad\ufffdchar"]}
    assert dirty_val["nested"][1] == "bad\ud800char"  # caller input not mutated

    dirty_key = {"ok": 1, "bad\udfffkey": "v\ud800", "tail": 2}
    out_k = _sanitize_json_strings(dirty_key)
    assert list(out_k.keys()) == ["ok", "bad\ufffdkey", "tail"]
    assert out_k["bad\ufffdkey"] == "v\ufffd"


def test_strip_lone_surrogates_preserves_length_and_identity() -> None:
    ascii_text = "plain ASCII text 123"
    assert strip_lone_surrogates(ascii_text) is ascii_text
    uni_clean = "naïve café α + β"
    assert strip_lone_surrogates(uni_clean) is uni_clean
    dirty = "a\ud800b\udfffc"
    cleaned = strip_lone_surrogates(dirty)
    assert cleaned == "a\ufffdb\ufffdc"
    assert len(cleaned) == len(dirty)
