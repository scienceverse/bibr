"""Graceful handling of Hugging Face tokenizer failures on OCR garbage."""

from __future__ import annotations

import logging
from typing import Any

from bibr.utils.text import strip_lone_surrogates

logger = logging.getLogger(__name__)


def _is_textencode_error(exc: TypeError) -> bool:
    return "TextEncodeInput" in str(exc)


def safe_tokenize(tokenizer: Any, texts: str | list[str], **tokenizer_kwargs: Any) -> Any:
    """Tokenize, scrubbing invalid OCR surrogates and retrying once when needed."""
    try:
        return tokenizer(texts, **tokenizer_kwargs)
    except TypeError as exc:
        if not _is_textencode_error(exc):
            raise
        if isinstance(texts, str):
            scrubbed: str | list[str] = strip_lone_surrogates(texts)
            count = 1
        else:
            scrubbed = [
                strip_lone_surrogates(text) if isinstance(text, str) else text for text in texts
            ]
            count = len(texts)
        logger.warning(
            "HF tokenizer rejected %d OCR text(s); replacing invalid Unicode and retrying",
            count,
        )
        return tokenizer(scrubbed, **tokenizer_kwargs)
