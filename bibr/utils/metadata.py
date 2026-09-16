"""Small metadata normalization predicates shared across pipeline stages."""

from __future__ import annotations

import unicodedata

EXACT_GENERIC_ARTICLE_LABELS = frozenset(
    {
        "research article",
        "original article",
        "original research",
        "review article",
        "review (narrative)",
        "review (systematic)",
    }
)

# Exact printed headings only: significance statements and research highlights
# may be classified as ABSTRACT, but do not identify the paper's abstract.
PRINTED_ABSTRACT_LABELS = frozenset(
    {
        "abstract",
        "summary",
        "precis",
        "executive summary",
        "resumen",
        "resumo",
        "résumé",
        "resume",
        "zusammenfassung",
        "samenvatting",
        "аннотация",
        "анотація",
        "abstrak",
        "özet",
        "streszczenie",
        "摘要",
        "要旨",
        "초록",
        "ملخص",
    }
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

    return bool(
        value
        and _normalize_label(value)
        in {_normalize_label(label) for label in EXACT_GENERIC_ARTICLE_LABELS}
    )


def strip_leading_article_label(value: str) -> str:
    """Remove a standalone article-type line preceding a wrapped title.

    Require an actual line boundary in the source. A title that mentions an
    article type inline, or starts with an unqualified word like "Review",
    remains verbatim.
    """
    lines = value.strip().splitlines()
    if len(lines) > 1 and is_exact_generic_article_label(lines[0]):
        remainder = "\n".join(lines[1:]).strip()
        if remainder:
            return remainder
    return value


def is_printed_abstract_heading(value: str) -> bool:
    """Recognize an exact printed abstract label without semantic inference."""
    return _normalize_label(value) in PRINTED_ABSTRACT_LABELS


__all__ = [
    "EXACT_GENERIC_ARTICLE_LABELS",
    "PRINTED_ABSTRACT_LABELS",
    "is_exact_generic_article_label",
    "is_printed_abstract_heading",
    "strip_leading_article_label",
]
