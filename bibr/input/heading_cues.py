"""Headings that no markup marks, shared by the native DOCX and HTML parsers.

A manuscript's author often makes headings by hand: a Normal paragraph in bold
reading "Method", rather than Word's Heading 1 style or an ``<h2>``. Without
them the paper has no sections, and without a "References" section its
reference list is not found (#215). These are the text cues both parsers use
to read such a paragraph as a heading.
"""

from __future__ import annotations

import re

from bibr.ocr.ref_patterns import _REF_HEADER_RE
from bibr.structure.float_labels import (
    FIGURE_WORD,
    SUPPLEMENT_WORD,
    TABLE_WORD,
    label_element_label,
)

# The header of the section a parser opens for a reference list that no
# references heading precedes.
REFERENCES_HEADER = "References"

# A reference-list word anywhere in a heading ("Selected References"): the
# list under it needs no heading of its own.
_REF_HEADING_WORD_RE = re.compile(
    r"\b(?:references|bibliography|works cited|literature cited|reference list)\b",
    re.IGNORECASE,
)

# A paragraph opening with a figure or table word, labelled or not.
_CAPTION_START_RE = re.compile(
    rf"{SUPPLEMENT_WORD}?(?:{FIGURE_WORD}|{TABLE_WORD})(?![A-Za-z])", re.IGNORECASE
)

# A reference entry, which an author may bold whole: a year in parentheses
# followed by the entry's next field ("Smith, J. (2020). Title"), a Vancouver
# year and volume ("2020;12:"), or a DOI or URL. A heading can hold a year
# ("Lockdown Effects (2020)"), but not one followed by more fields.
_REFERENCE_ENTRY_RE = re.compile(
    r"\((?:1[6-9]|20)\d{2}[a-z]?(?:,[^)]{0,30})?\)\s*[.,]"
    r"|\b(?:1[6-9]|20)\d{2}\s*;\s*\d"
    r"|\b10\.\d{4,9}/|https?://|\bwww\.",
    re.IGNORECASE,
)

# A paragraph bold throughout is a heading when it is at most this long.
_MAX_BOLD_HEADING_CHARS = 200
_MAX_BOLD_HEADING_WORDS = 20
# A sentence-final full stop ends a heading only when it is this short
# ("Introduction.", "2. Methods.").
_MAX_DOTTED_HEADING_WORDS = 6
# A references heading paragraph is at most this long ("6. Literaturverzeichnis").
_MAX_REFERENCE_HEADING_CHARS = 60
# Body text runs on after a word that ends in these: a heading does not.
_RUN_ON_PUNCTUATION = frozenset(",;:")


def reads_as_reference_heading(text: str) -> bool:
    """Whether a paragraph's whole text is a references heading, in any
    language the shared heading grammar lists ("References", "6. Literatur")."""
    text = text.strip()
    return (
        bool(text)
        and len(text) <= _MAX_REFERENCE_HEADING_CHARS
        and "\n" not in text
        and _REF_HEADER_RE.match(text) is not None
    )


def names_reference_list(header: str) -> bool:
    """Whether a heading heads a reference list: it is a references heading,
    or holds a reference-list word ("Selected References")."""
    return bool(_REF_HEADING_WORD_RE.search(header)) or reads_as_reference_heading(header)


def reads_as_bold_heading(text: str) -> bool:
    """Whether a paragraph bold throughout is short enough, and shaped enough
    like a heading, to be one: not a caption, not a sentence, not a lead-in
    ("Keywords:"), not a reference entry."""
    text = text.strip()
    words = text.split()
    if not words or len(text) > _MAX_BOLD_HEADING_CHARS or len(words) > _MAX_BOLD_HEADING_WORDS:
        return False
    if not any(char.isalpha() for char in text) or text[-1] in _RUN_ON_PUNCTUATION:
        return False
    if text[-1] == "." and len(words) > _MAX_DOTTED_HEADING_WORDS:
        return False
    return _CAPTION_START_RE.match(text) is None and _REFERENCE_ENTRY_RE.search(text) is None


# Headings that open a manuscript's body. Bold paragraphs above the first of
# them are its title page: the title (often over several lines), the authors
# and their affiliations.
_BODY_OPENERS = frozenset(
    {
        "abstract",
        "summary",
        "introduction",
        "background",
        "keywords",
        "key words",
        "method",
        "methods",
        "significance statement",
        "public significance statement",
    }
)


def opens_the_body(text: str) -> bool:
    """Whether a heading is one a manuscript's body starts with ("Abstract",
    "1. Introduction")."""
    norm = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", text.lower()).strip().rstrip(".:")
    return norm in _BODY_OPENERS


def reads_as_outline_heading(text: str) -> bool:
    """Whether a paragraph that an outline level alone marks as a heading
    reads as one: not a reference entry or caption a template happened to give
    an outline level, nor a whole paragraph of text."""
    text = text.strip()
    return (
        len(text) <= _MAX_BOLD_HEADING_CHARS
        and len(text.split()) <= _MAX_BOLD_HEADING_WORDS
        and _CAPTION_START_RE.match(text) is None
        and _REFERENCE_ENTRY_RE.search(text) is None
    )


def starts_as_caption(text: str) -> bool:
    """Whether a paragraph opens with a figure or table word ("Table S1: ...")."""
    return _CAPTION_START_RE.match(text.strip()) is not None


def reads_as_float_label(text: str) -> bool:
    """Whether a paragraph is only a figure or table label ("Table S1",
    "Figure 2"), as APA sets one above the float's title."""
    return any(label_element_label(text, kind) is not None for kind in ("figure", "table"))
