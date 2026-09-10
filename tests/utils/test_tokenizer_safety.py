"""Tests for Hugging Face tokenizer handling of invalid OCR Unicode."""

from __future__ import annotations

import pytest

from bibr.utils.text import strip_lone_surrogates
from bibr.utils.tokenizer_safety import safe_tokenize

LONE_SURROGATE = "\udce9"
_TEXTENCODE_MSG = (
    "TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]"
)


def _has_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)


def test_strip_lone_surrogates_replaces_and_preserves_length():
    text = f"Meth{LONE_SURROGATE}ods"

    scrubbed = strip_lone_surrogates(text)

    assert scrubbed == "Meth�ods"
    assert len(scrubbed) == len(text)
    scrubbed.encode("utf-8")


def test_strip_lone_surrogates_returns_same_object_when_clean():
    text = "Introduction — Méthodes 実験 [SEP] body"

    assert strip_lone_surrogates(text) is text


class _FakeHFTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.last_kwargs = {}

    def __call__(self, texts, **kwargs):
        batch = [texts] if isinstance(texts, str) else list(texts)
        self.calls.append(batch)
        self.last_kwargs = kwargs
        if any(_has_surrogate(text) for text in batch):
            raise TypeError(_TEXTENCODE_MSG)
        return {"ok": batch, "kwargs": kwargs}


def test_safe_tokenize_retries_with_scrubbed_batch_on_textencodeinput():
    tokenizer = _FakeHFTokenizer()

    result = safe_tokenize(
        tokenizer,
        [f"Meth{LONE_SURROGATE}ods", "Results"],
        padding=True,
        max_length=64,
    )

    assert result["ok"] == ["Meth�ods", "Results"]
    assert len(tokenizer.calls) == 2
    assert tokenizer.last_kwargs == {"padding": True, "max_length": 64}


def test_safe_tokenize_calls_once_for_clean_input():
    tokenizer = _FakeHFTokenizer()

    result = safe_tokenize(tokenizer, ["clean header", "clean body"])

    assert result["ok"] == ["clean header", "clean body"]
    assert len(tokenizer.calls) == 1


def test_safe_tokenize_reraises_unrelated_type_error():
    def tokenizer(texts, **kwargs):  # noqa: ARG001
        raise TypeError("some other tokenizer problem")

    with pytest.raises(TypeError, match="some other tokenizer problem"):
        safe_tokenize(tokenizer, ["x"])
