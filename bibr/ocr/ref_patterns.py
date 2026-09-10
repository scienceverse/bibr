"""Shared reference-line regexes.

Compiled patterns used by the geometry-based reference path
(``bibr.ocr.ref_geometry``) and reference feature extraction
(``bibr.extract.geom_features``). Kept in a neutral module with no heavy
dependencies so importers pay nothing to load them.
"""

from __future__ import annotations

import re

# A missing reference heading bypasses geometry segmentation. Accept translated, singular,
# numbered, and decorated heading forms through an explicit multilingual grammar.
_REF_HEADER_WORDS = (
    # English
    r"references?(?:\s+and\s+notes)?",
    r"notes\s+and\s+references",
    r"reference\s+list",
    r"list\s+of\s+references",
    r"literature\s+cited",
    r"cited\s+literature",
    r"works\s+cited",
    # Romance
    r"referencias(?:\s+bibliogr[áa]ficas)?",
    r"refer[êe]ncias(?:\s+bibliogr[áa]ficas)?",
    r"r[ée]f[ée]rences(?:\s+bibliographiques)?",
    r"riferimenti\s+bibliografici",
    r"obras\s+citadas",
    r"literatura\s+citada",
    # bibliografi / bibliografia / bibliografía / bibliografie / bibliography / bibliographie
    r"bibliogra(?:f|ph)(?:ia|ía|ie|y|i)?",
    # Germanic / Nordic
    r"literatur(?:verzeichnis)?",
    r"quellenverzeichnis",
    r"literatuur(?:lijst)?",
    r"referenties",
    r"litteratur",
    r"referanser|referencer|referenser",
    r"k[äa]llor",
    r"l[äa]hteet",
    r"kirjallisuus",
    # Slavic
    r"список\s+литературы",
    r"список\s+использованной\s+литературы",
    r"использованная\s+литература",
    r"список\s+источников",
    r"литература",
    r"библиография",
    r"список\s+використаних\s+джерел",
    r"використані\s+джерела",
    r"перелік\s+посилань",
    r"література",
    r"pi[śs]miennictwo",
    r"literat[uú]ra",
    r"(?:zoznam|seznam)\s+literat[uú]ry",
    r"popis\s+literature",
    # Turkic / Uralic
    r"kaynak(?:lar|ça|ca)",
    r"irodalomjegyz[ée]k",
    r"hivatkoz[áa]sok",
    # Malay / Indonesian
    r"referensi",
    r"daftar\s+(?:pustaka|rujukan|referensi)",
    r"kepustakaan",
    r"rujukan",
    # CJK / other scripts
    r"引用文献",
    r"参考文献",
    r"參考文獻",
    r"참고문헌",
    r"tài\s+liệu\s+tham\s+khảo",
    r"βιβλιογραφία",
    r"αναφορές",
    r"المراجع",
    r"قائمة\s+المراجع",
    r"منابع",
)

# Optional section numbering ("6 REFERENCES", "IV. References", "E. Referensi")
# and typographic decoration ("■ REFERENCES"). Letters must carry punctuation so
# that a stray word cannot be read as an enumerator.
_REF_HEADER_PREFIX = r"(?:[■▪●◆•*#§]\s*)?(?:(?:\d{1,2}[.)]?|[IVX]{1,5}[.)]|[A-Z][.)])\s+)?"
# Trailing colon ("Список використаних джерел:") or decoration ("References»»»").
_REF_HEADER_SUFFIX = r"\s*[:：.．。]?\s*[»«\-–—]*"

_REF_HEADER_RE = re.compile(
    rf"(?im)^\s*{_REF_HEADER_PREFIX}(?:{'|'.join(_REF_HEADER_WORDS)}){_REF_HEADER_SUFFIX}\s*$"
)

# A line that starts like "Surname," — the classic author-date reference opener.
_AUTHOR_DATE_START = re.compile(r"^[\"'(]?[A-Z][A-Za-z'’.\-]+,")

# Unicode/full-given-name fallback for already reference-shaped regions. Keep
# the trained geometry feature above byte-for-byte stable; callers that need a
# higher-recall structural onset check use this helper instead. Both sides of
# the comma stay name-shaped so prose such as "The report, published in 2020"
# cannot become a reference anchor merely because it contains a year.
_NAME_COMPONENT_RE = r"[A-ZÀ-ÖØ-Þ][^\W\d_]*(?:[-'’][^\W\d_]+)*\.?"
_FAMILY_PARTICLE = r"(?i:da|de|del|della|der|di|du|la|le|van|von)"
_AUTHOR_DATE_COMMA_LEAD = re.compile(
    rf"^[\"'(]?(?P<family>{_NAME_COMPONENT_RE}"
    rf"(?:\s+(?:{_FAMILY_PARTICLE}|{_NAME_COMPONENT_RE})){{0,3}}),"
    rf"\s*(?P<given>{_NAME_COMPONENT_RE}"
    rf"(?:\s+{_NAME_COMPONENT_RE}){{0,3}})(?P<tail>.{{0,140}})",
)

# A 4-digit year (optionally with a disambiguating letter) or an "(in press)" marker.
_YEAR = re.compile(r"\(in press[a-z]?\)|\b(?:19|20)\d\d[a-z]?\b", re.IGNORECASE)


def _looks_like_author_date_start(text: str) -> bool:
    """Return whether *text* has a credible ``Family, Given ... Year`` onset.

    ``_AUTHOR_DATE_START`` is an intentionally stable ASCII feature consumed
    by the trained geometry model.  This structural fallback adds Unicode
    surnames and full given names without changing that model feature.
    """
    if _AUTHOR_DATE_START.match(text):
        return True
    match = _AUTHOR_DATE_COMMA_LEAD.match(text)
    if match is None:
        return False
    return bool(_YEAR.search(match.group("tail")))
