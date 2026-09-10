"""Small metadata normalization predicates shared across pipeline stages."""

from __future__ import annotations

import unicodedata

EXACT_GENERIC_ARTICLE_LABELS = frozenset(
    {"research article", "original article", "original research"}
)


def _normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = " ".join(normalized.split())

    def _is_edge_noise(character: str) -> bool:
        category = unicodedata.category(character)
        return character.isspace() or category.startswith(("P", "S")) or category == "Cf"

    start = 0
    end = len(normalized)
    while start < end and _is_edge_noise(normalized[start]):
        start += 1
    while end > start and _is_edge_noise(normalized[end - 1]):
        end -= 1
    return normalized[start:end].strip()


def is_exact_generic_article_label(value: str | None) -> bool:
    """Return whether *value* is exactly one of the generic article labels.

    Surrounding whitespace and punctuation are presentation noise. Internal
    punctuation and suffix text remain significant, so ``Research Article:
    Effects of X`` is a real title rather than a generic-label match.
    """

    return bool(value and _normalize_label(value) in EXACT_GENERIC_ARTICLE_LABELS)


__all__ = ["EXACT_GENERIC_ARTICLE_LABELS", "is_exact_generic_article_label"]
