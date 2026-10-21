"""Tag set, word split and per-word features of the front-matter tagger.

The single source of truth for the tagger's input contract: bibr-training's
corpus builder and trainer import this module, so the model sees the same
words, the same input string and the same feature vectors at fit time and at
inference (the arrangement ``front_role_features`` has with the region-role
GBM).

The input is every text region of the paper's first ``MAX_PAGES`` pages in
reading order. Inside a region, words keep their printed spacing; regions are
joined with ``"\\n"``, so a newline in the input always means a region
boundary. Each word is labelled with one BIO tag; :mod:`.assemble` reads the
labelled spans back as printed text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from bibr.extract.front_role_features import NON_TEXT_LABELS, FrontRegion, region_features

CONTRACT_VERSION = 1

#: Pages of front matter the tagger reads (counted from the paper's first page).
MAX_PAGES = 2

FIELD_TYPES: list[str] = [
    "TITLE",
    "SUBTITLE",
    "GIVEN",
    "FAMILY",
    "AUTHOR_MARK",
    "AFFILIATION",
    "AFF_MARK",
    "EMAIL",
    "ORCID",
    "ABSTRACT",
    "KEYWORD",
    "JOURNAL",
    "VOLUME",
    "ISSUE",
    "PAGES",
    "ISSN",
    "PUBLISHER",
    "DATE",
    "LICENSE",
    "DOI",
    # Running heads, journal banners, section rubrics, page numbers and other
    # page furniture. Labelled explicitly so the model learns to cut it out of
    # a title region the parser fused it into.
    "FURNITURE",
]

BIO_TAGS: list[str] = ["O"] + [
    f"{prefix}-{field}" for field in FIELD_TYPES for prefix in ("B", "I")
]

# Layout labels one-hot encoded per word; anything else maps to "other".
LAYOUT_LABELS: tuple[str, ...] = (
    "doc_title",
    "paragraph_title",
    "text",
    "abstract",
    "header",
    "footer",
    "number",
    "footnote",
    "vision_footnote",
    "aside_text",
    "reference",
    "reference_content",
    "other",
)

_REGION_KEYS: tuple[str, ...] = (
    "x0_rel",
    "y0_rel",
    "width_rel",
    "height_rel",
    "center_offset",
    "font_rel_max",
    "font_bold",
    "is_first_on_page",
    "index_rel",
)
_WORD_KEYS: tuple[str, ...] = (
    "page_rel",
    "is_region_start",
    "word_pos_rel",
    "is_upper",
    "is_capitalized",
    "is_digit",
    "is_punct",
    "is_mark",
    "is_cjk",
    "len_scaled",
)

#: Feature vector layout, one float per key, in this order.
FEATURE_KEYS: tuple[str, ...] = (
    _REGION_KEYS + _WORD_KEYS + tuple(f"label_{label}" for label in LAYOUT_LABELS)
)
FEATURE_DIM = len(FEATURE_KEYS)

# Han, kana and Hangul characters are words of their own: unspaced CJK text
# would otherwise come out as one word carrying one tag.
_CJK = "[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]"
_LETTERS = rf"(?:(?!{_CJK})[^\W\d_])+"
# A letter run may carry inner apostrophes and hyphens (O'Neil, Jean-Pierre);
# digits and every other character stand alone, so a superscript key glued to
# a name ("Smith1,2") splits into "Smith", "1", ",", "2".
WORD_RE = re.compile(rf"{_CJK}|{_LETTERS}(?:['’\-]{_LETTERS})*|\d+|\S")
_CJK_RE = re.compile(_CJK)
_MARKS = frozenset("*†‡§¶#‖⁎∗")


@dataclass(frozen=True)
class TaggerWord:
    """One word of the tagger input; ``start``/``end`` index ``TaggerInput.text``."""

    text: str
    start: int
    end: int
    region: int  # position in ``TaggerInput.regions``


@dataclass(frozen=True)
class TaggerInput:
    text: str
    words: tuple[TaggerWord, ...]
    features: tuple[tuple[float, ...], ...]  # one FEATURE_KEYS vector per word
    regions: tuple[FrontRegion, ...]


def _region_text(region: FrontRegion) -> str:
    # A newline inside a region becomes a space: newlines mark region bounds.
    return re.sub(r"\s*\n\s*", " ", region.text or "").strip()


def _word_features(word: str, pos: int, n_words: int) -> dict[str, float]:
    alpha = [c for c in word if c.isalpha()]
    return {
        "is_region_start": float(pos == 0),
        "word_pos_rel": pos / n_words if n_words else 0.0,
        "is_upper": float(bool(alpha) and len(alpha) > 1 and all(c.isupper() for c in alpha)),
        "is_capitalized": float(word[:1].isupper()),
        "is_digit": float(word.isdigit()),
        "is_punct": float(not any(c.isalnum() for c in word)),
        "is_mark": float(word in _MARKS),
        "is_cjk": float(bool(_CJK_RE.match(word))),
        "len_scaled": min(len(word), 20) / 20.0,
    }


def build_tagger_input(
    regions: Sequence[FrontRegion],
    *,
    first_page_index: int = 0,
    max_pages: int = MAX_PAGES,
) -> TaggerInput:
    """The tagger's words, input string and feature rows for one paper.

    ``regions`` is the reading-order output of
    :func:`bibr.extract.front_role.build_front_regions` (every region, non-text
    ones included, so the region features match the region-role model's).
    """
    all_regions = list(regions)
    region_rows = region_features(all_regions)
    kept: list[tuple[FrontRegion, dict, str]] = []
    for region, row in zip(all_regions, region_rows, strict=True):
        if (region.label or "") in NON_TEXT_LABELS:
            continue
        if not first_page_index <= region.page < first_page_index + max_pages:
            continue
        text = _region_text(region)
        if text:
            kept.append((region, row, text))

    parts: list[str] = []
    words: list[TaggerWord] = []
    features: list[tuple[float, ...]] = []
    offset = 0
    for region_pos, (region, row, text) in enumerate(kept):
        if parts:
            parts.append("\n")
            offset += 1
        matches = list(WORD_RE.finditer(text))
        label = region.label if region.label in LAYOUT_LABELS else "other"
        base = {key: float(row[key]) for key in _REGION_KEYS}
        base["page_rel"] = float(region.page - first_page_index)
        for pos, match in enumerate(matches):
            values = {**base, **_word_features(match.group(), pos, len(matches))}
            for layout_label in LAYOUT_LABELS:
                values[f"label_{layout_label}"] = float(layout_label == label)
            words.append(
                TaggerWord(
                    text=match.group(),
                    start=offset + match.start(),
                    end=offset + match.end(),
                    region=region_pos,
                )
            )
            features.append(tuple(values[key] for key in FEATURE_KEYS))
        parts.append(text)
        offset += len(text)

    return TaggerInput(
        text="".join(parts),
        words=tuple(words),
        features=tuple(features),
        regions=tuple(region for region, _, _ in kept),
    )


__all__ = [
    "BIO_TAGS",
    "CONTRACT_VERSION",
    "FEATURE_DIM",
    "FEATURE_KEYS",
    "FIELD_TYPES",
    "LAYOUT_LABELS",
    "MAX_PAGES",
    "WORD_RE",
    "TaggerInput",
    "TaggerWord",
    "build_tagger_input",
]
