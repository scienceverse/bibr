"""Shared reference-line regexes.

Compiled patterns used by the geometry-based reference path
(``bibr.ocr.ref_geometry``) and reference feature extraction
(``bibr.extract.geom_features``). Kept in a neutral module with no heavy
dependencies so importers pay nothing to load them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapidfuzz.distance import MatchingBlock

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
# Whitespace within a line: "\n" is the only line boundary ^ and $ see, so a
# "\r" before it (pdfium's line breaks) still trails the header.
_LINE_SPACE = r"[^\S\n]"
# Trailing colon ("Список використаних джерел:") or decoration ("References»»»").
# One whitespace run between tokens: adjacent runs that could share the same
# spaces backtracked cubically over a long one.
_REF_HEADER_SUFFIX = rf"{_LINE_SPACE}*(?:[:：.．。]{_LINE_SPACE}*)?(?:[»«\-–—]+{_LINE_SPACE}*)?"

# Leading whitespace crosses blank lines only from the start of the string, as
# ``match`` callers need; a search that let every line start run over the blank
# lines below it took quadratic time.
_REF_HEADER_RE = re.compile(
    rf"(?im)(?:\A\s*|^{_LINE_SPACE}*){_REF_HEADER_PREFIX}"
    rf"(?:{'|'.join(_REF_HEADER_WORDS)}){_REF_HEADER_SUFFIX}$"
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


# A line holding only entry labels: "[12]", "(3)", "7.", or "[3 [41" where the
# OCR or the PDF text layer lost a bracket or read "]" as "1". The text layer of
# a numbered list can hold its labels as one column before the entries (every
# label, then every entry), while each entry box starts with its own label, so
# the aggregate box's labels are one stretch its entry boxes seem to lack.
_LABEL = r"[\[(]?\d{1,3}[\]).]?"
_LABEL_ONLY_LINE = re.compile(rf"\s*{_LABEL}(?:\s+{_LABEL})*\s*")


def alnum_key(text: str) -> str:
    """Lowercased alphanumeric-only projection of *text* for coverage checks.

    Lines holding only entry labels are left out on both sides of a comparison.
    """
    kept = "".join(line for line in text.splitlines() if not _LABEL_ONLY_LINE.fullmatch(line))
    return "".join(c for c in kept.lower() if c.isalnum())


# Fuzzy score at which one region's normalized text counts as already present in
# other regions' text, and the needle length below which only exact containment
# is trusted (short needles fuzzy-match too easily).
_COVERED_MIN_SCORE = 95
_COVERED_MIN_CHARS = 30
# A needle with 16 or more characters without a copy in the haystack within any
# 24 consecutive characters is not covered. OCR noise between two reads of the
# same text is scattered ("rn" for "m", "l" for "1", at most a dropped word);
# text the haystack lacks, such as an entry without a box of its own, a line or
# a DOI, is one stretch. In the entry boxes of the lfm25 OCR caches, 99% of
# lines other than an entry's last hold 22 or more alphanumeric characters.
_COVERED_STRETCH = 24
_COVERED_MAX_MISSING = 16
# Matching blocks shorter than this are not a copy of needle text: an alignment
# with unrelated text pairs single letters and pairs of letters by chance.
_COVERED_MIN_BLOCK = 3
# Matching blocks at least this long locate the needle's copy in the haystack,
# and the room left around that copy for noise at its edges.
_COVERED_ANCHOR_BLOCK = 8
_COVERED_MARGIN = 8
# Longest needle searched as a substring of the haystack. partial_ratio grows
# with the square of the needle length on noisy text (seconds at 16k chars,
# tens of seconds at 50k, as an OCR repetition loop can produce); a longer
# needle is compared with the whole haystack instead.
_COVERED_MAX_PARTIAL_CHARS = 10_000
# Length of the needle runs looked up anywhere in the haystack when the two
# reads hold the same text in a different order (``_covered_in_any_order``).
_COVERED_MOVED_CHARS = 40


def _matching_blocks(needle: str, haystack: str, min_size: int) -> list[MatchingBlock]:
    from rapidfuzz.distance import Levenshtein

    blocks = Levenshtein.opcodes(needle, haystack).as_matching_blocks()
    return [block for block in blocks if block.size >= min_size]


def _missing_chars(needle: str, haystack: str, start: int = 0, end: int | None = None) -> int:
    """Most characters without a copy in ``haystack[start:end]`` in any stretch of *needle*.

    A stretch is 24 consecutive characters; a character has a copy when it is
    in a matching block of three or more characters. The needle is aligned
    twice. The first alignment locates its copy by the long matching blocks,
    and the second aligns it with only that copy and a margin for noise: spare
    haystack text around the copy lets the aligner pair a noisy edge of the
    needle letter by letter with unrelated text, at no more cost than matching
    it to its copy. A word the unrelated text happens to share does not hide a
    missing stretch either, since the stretch keeps its other characters.
    """
    end = len(haystack) if end is None else end
    anchors = _matching_blocks(needle, haystack[start:end], _COVERED_ANCHOR_BLOCK)
    stretch = min(_COVERED_STRETCH, len(needle))
    if not anchors:
        return stretch
    head = anchors[0].a
    tail = len(needle) - anchors[-1].a - anchors[-1].size
    lo = max(0, start + anchors[0].b - head - head // 10 - _COVERED_MARGIN)
    hi = start + anchors[-1].b + anchors[-1].size + tail + tail // 10 + _COVERED_MARGIN
    copied = bytearray(len(needle))
    for block in _matching_blocks(needle, haystack[lo:hi], _COVERED_MIN_BLOCK):
        copied[block.a : block.a + block.size] = b"\x01" * block.size
    missing = most = stretch - sum(copied[:stretch])
    for i in range(stretch, len(needle)):
        missing += copied[i - stretch] - copied[i]
        most = max(most, missing)
    return most


def _covered_in_any_order(needle: str, haystack: str) -> bool:
    """Whether *needle*'s text is in *haystack*, in whatever order.

    Every run of 40 needle characters is looked up anywhere in the haystack,
    exactly or with a fuzzy score of at least 95, and the characters in its
    matching blocks there count as copied. The needle is covered when 95% of
    its characters are and no 24 consecutive ones hold 16 without a copy, the
    same bounds as for an ordered match. A run this long is usually an
    entry's own text rather than a volume line many entries share, and the
    characters where a run leaves its entry do not match the next entry's
    text, so an entry the haystack lacks keeps its missing characters. A long
    string printed twice, such as a journal name in a running header and in an
    entry, can still be credited to the other copy.
    """
    from rapidfuzz import fuzz

    size = min(_COVERED_MOVED_CHARS, len(needle))
    stretch = min(_COVERED_STRETCH, len(needle))
    budget = len(needle) - len(needle) * _COVERED_MIN_SCORE // 100
    copied = bytearray(len(needle))
    missing = window = 0
    for i in range(len(needle)):
        if i + size <= len(needle) and not all(copied[i : i + size]):
            piece = needle[i : i + size]
            if piece in haystack:
                copied[i : i + size] = b"\x01" * size
            else:
                found = fuzz.partial_ratio_alignment(
                    piece, haystack, score_cutoff=_COVERED_MIN_SCORE
                )
                if found is not None:
                    copy = haystack[found.dest_start : found.dest_end]
                    for block in _matching_blocks(piece, copy, _COVERED_MIN_BLOCK):
                        copied[i + block.a : i + block.a + block.size] = b"\x01" * block.size
        # No later run holds character i: its verdict is final.
        missing += not copied[i]
        window += not copied[i]
        if i >= stretch:
            window -= not copied[i - stretch]
        if missing > budget or window >= _COVERED_MAX_MISSING:
            return False
    return True


def alnum_text_covered(needle: str, haystack: str) -> bool:
    """Whether *needle* is already contained in *haystack* (both ``alnum_key`` output).

    Exact containment, or, for needles of 30 or more characters, a match that
    absorbs OCR noise between two reads of the same text: a fuzzy score of at
    least 95, and no 24 consecutive needle characters of which 16 or more have
    no copy in the haystack. The score bounds the total noise. The second test
    keeps a needle that holds text the haystack lacks, however small a share of
    a long needle that text is: an entry without a box of its own, a line, a
    DOI.

    Two reads of one aggregate box and of its entry boxes differ by "rn" for
    "m", "l" for "1" and the like, and either read can be the longer one, so
    the two are first compared whole (``ratio``, symmetric, and fast even on
    equal lengths, where ``partial_ratio`` is not). A needle shorter than the
    haystack is then also searched as a substring (``partial_ratio``), and the
    second test is applied where it matched.
    """
    if not needle:
        return True
    if needle in haystack:
        return True
    if len(needle) < _COVERED_MIN_CHARS:
        return False
    from rapidfuzz import fuzz

    if (
        fuzz.ratio(needle, haystack) >= _COVERED_MIN_SCORE
        and _missing_chars(needle, haystack) < _COVERED_MAX_MISSING
    ):
        return True
    if len(needle) > _COVERED_MAX_PARTIAL_CHARS:
        return False
    if len(needle) < len(haystack):
        alignment = fuzz.partial_ratio_alignment(needle, haystack, score_cutoff=_COVERED_MIN_SCORE)
        if alignment is not None:
            start = max(0, alignment.dest_start - _COVERED_MARGIN)
            end = alignment.dest_end + _COVERED_MARGIN
            if _missing_chars(needle, haystack, start, end) < _COVERED_MAX_MISSING:
                return True
    # Both matches keep the needle's order. The two reads can hold the same
    # text in different orders, though: a PDF text layer can list an entry
    # after the one the entry boxes sort it after, or hold a numbered list's
    # labels as one column before the entries.
    return _covered_in_any_order(needle, haystack)


# Most words on a line that ``drop_edge_fragments`` treats as a stray fragment.
_EDGE_FRAGMENT_MAX_WORDS = 3


def drop_edge_fragments(text: str, haystack: str) -> str:
    """*text* without its leading and trailing short lines that *haystack* holds.

    A diagonal watermark crosses an aggregate reference box and the entry boxes
    inside it. The aggregate box reads the whole stamp, and each entry box a
    few of its words on lines of their own around the entry ("ts contents are",
    "view purposes", "osed."). The aggregate holds those words, but not next to
    the entry, so ``alnum_text_covered`` finds the entry box holding text the
    aggregate lacks. A line of at most three words at either end of *text*
    counts as such a fragment only when *haystack* (``alnum_key`` output)
    contains it, so no text the haystack lacks is dropped.
    """
    lines = text.splitlines()

    def fragment(line: str) -> bool:
        return len(line.split()) <= _EDGE_FRAGMENT_MAX_WORDS and alnum_key(line) in haystack

    start, end = 0, len(lines)
    while start < end and fragment(lines[start]):
        start += 1
    while end > start and fragment(lines[end - 1]):
        end -= 1
    return "\n".join(lines[start:end])


def covered_without_edge_fragments(text: str, haystack: str) -> bool:
    """Whether *haystack* holds *text* once its edge fragments are set aside.

    ``alnum_text_covered`` on ``drop_edge_fragments(text, haystack)``, but only
    when at least 30 letters and digits remain: short keys ("nature2019") turn
    up in a long haystack by chance, and a box of short lines only ("Smith J,",
    "Ibid.", "Nature 2019.") would otherwise lose every line and count as held.
    """
    rest = alnum_key(drop_edge_fragments(text, haystack))
    return len(rest) >= _COVERED_MIN_CHARS and alnum_text_covered(rest, haystack)
