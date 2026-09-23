"""Printed figure and table labels.

A float's label is what its caption prints after the word, as printed with
whitespace removed: "3" for "Table 3.", and likewise "3.1", "S2", "A1", "IV"
or "C". A caption opening "Supplementary Table 4" or "Suppl. Fig. 4" is
labelled "S4", so it is found by a mention of "Table S4" and of "Supplementary
Table 4" alike. In-text mentions resolve against these labels
(:func:`bibr.structure.xref_utils.detect_xrefs`), compared with
:func:`normalize_label`.

The patterns here are shared by caption detection in the PDF parser and by
mention detection, so a label printed one way is read the same way in both.
"""

from __future__ import annotations

import re
from typing import Literal

FloatKind = Literal["figure", "table"]

# The words a figure or table caption, and a mention of one, open with.
FIGURE_WORD = r"(?:Figures?|Figs?\.?)"
TABLE_WORD = r"(?:Tables?|Tab\.?|Tbl\.?)"
_WORDS: dict[str, str] = {"figure": FIGURE_WORD, "table": TABLE_WORD}

# "Supplementary Table 4", "Supplemental Figure 2", "Suppl. Fig. 1".
SUPPLEMENT_WORD = r"(?:(?:Online\s+)?(?:Supplementa(?:ry|l)\s+|Suppl?\.\s*))"

# One printed label. A numbered label is a number, possibly dotted ("3.1"),
# with an optional letter prefix ("S2", "S 2", "A1", "A.1", "A-1") and an
# optional letter after it ("2a" — a panel, or a table printed as "2a"). It
# may be glued to the next word, as extracted text sometimes is ("Figure
# 3shows"). A lettered label is a roman numeral or a single capital ("IV",
# "C"); it is case-sensitive even inside an IGNORECASE pattern, so prose
# ("table civil") is no label. Numbers take at most three digits, so a year
# ("Tables 2019") is not read as one.
NUMBERED_LABEL = r"(?:[Ss]\s?|[A-Za-z][.-]?)?\d{1,3}(?:\.\d{1,3})*(?:[A-Za-z](?![A-Za-z]))?(?!\d)"
LETTERED_LABEL = r"(?-i:[IVXLCDM]+|[A-Z])(?![A-Za-z0-9])"
LABEL = rf"(?:{NUMBERED_LABEL}|{LETTERED_LABEL})"

# eLife-style supplements print their parent's label and then name themselves
# ("Figure 1—figure supplement 1", "Table 2—source data 1"). They are not the
# parent, so they get no label rather than a duplicate of the parent's.
_NOT_A_SUPPLEMENT_OF = (
    r"(?!\s*[—–-]\s*(?-i:(?:figure|table)\s+supplement|source\s+(?:data|code)|video|animation)\b)"
)

_CAPTION_LABEL_RES = {
    kind: re.compile(
        rf"(?P<supplement>{SUPPLEMENT_WORD})?{word}\s*(?P<label>{LABEL}){_NOT_A_SUPPLEMENT_OF}",
        re.IGNORECASE,
    )
    for kind, word in _WORDS.items()
}

# A label element on its own (JATS ``<label>``): "Table 2", "Fig. 3.", a bare
# "S1", or PLOS's "S1 Table".
_LABEL_ELEMENT_RES = {
    kind: re.compile(
        rf"(?:(?P<supplement>{SUPPLEMENT_WORD})?{word}\s*)?(?P<label>{LABEL})"
        rf"(?:\s*{word})?\s*[.:]?",
        re.IGNORECASE,
    )
    for kind, word in _WORDS.items()
}

_SUPPLEMENTARY_LABEL_RE = re.compile(r"s[.-]?\d")


def printed_label(label: str, *, supplement: bool = False) -> str:
    """*label* as printed with whitespace removed; "S" prefixed when the word
    before it said "Supplementary" and the label does not start with one."""
    label = "".join(label.split())
    if supplement and label[:1].isdigit():
        return f"S{label}"
    return label


def caption_label(caption: str | None, kind: FloatKind) -> str | None:
    """The label *caption* prints after its figure or table word, or ``None``.

    "Table 3.1. Descriptives" → "3.1"; "TABLE IV" → "IV"; "Supplementary Table
    4" → "S4"; "Figure A1:" → "A1". A caption that opens with the other kind's
    word ("Table 2" for a figure) has no label of this kind.
    """
    if not caption:
        return None
    match = _CAPTION_LABEL_RES[kind].match(caption.strip())
    if match is None:
        return None
    return printed_label(match.group("label"), supplement=bool(match.group("supplement")))


def label_element_label(text: str | None, kind: FloatKind) -> str | None:
    """The label in a stand-alone label element such as JATS ``<label>``.

    The word is stripped ("Table 2" → "2", "Fig. 3" → "3"); a bare "S1" stays
    "S1". Anything else in the element ("Scheme 1", "Figure 1—figure supplement
    1") means it is no label of this kind.
    """
    if not text:
        return None
    match = _LABEL_ELEMENT_RES[kind].fullmatch(text.strip())
    if match is None:
        return None
    return printed_label(match.group("label"), supplement=bool(match.group("supplement")))


def normalize_label(label: str) -> str:
    """The form labels are compared in: case-insensitive, whitespace removed."""
    return "".join(label.split()).casefold()


def is_supplementary_label(label: str) -> bool:
    """True for an "S"-prefixed label: "S2", "S1.3", "S.2"."""
    return bool(_SUPPLEMENTARY_LABEL_RE.match(normalize_label(label)))
