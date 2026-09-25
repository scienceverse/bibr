"""Detect and split under-segmented reference strings.

A merged reference segment — two adjacent references glued into one string by an
under-segmenting segmenter — carries a structural tell that is independent of the
segmenter's own confidence: a *second* author-date onset mid-string. This module
finds those interior onsets (``find_interior_onsets``), reports them flag-only for
measurement (``detect_merges``), and splits at them (``split_merged_refs``, added
in a later task). Pure stdlib; no network, no LLM.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# Parenthesized date anchor — mirrors geom_features.early_year_paren's date body,
# extended with "(in press)". The date must sit immediately after "(", so a year
# embedded in a longer parenthetical — "(Original work published 1900)" — does NOT
# match. That immediacy is the main precision safeguard.
_DATE_ANCHOR = re.compile(
    r"\((?:(?:19|20)\d\d[a-z]?|n\.?\s?d\.?|in press[a-z]?)\)",
    re.IGNORECASE,
)

# Words that mark a parenthetical date as part of one reference, not a new onset.
_META_BEFORE = re.compile(
    r"published|reprinted|retrieved|cited|edition|originally"
    r"|\bed\.|\bvol\.|\bno\.|\bpp\.|\btrans\.",
    re.IGNORECASE,
)

# A parenthetical date that is an in-text citation inside a reference's own title —
# "...A correction to Cousineau (2005)", "Comment on Bem (2011)", "Reply to Wilson
# and Gilbert (2013)" — belongs to the SAME reference, not a new one. The cue verb
# (reply/comment/correction/...) plus to/on/of must abut the date with no sentence
# or paren break in between, so a ref1 title that merely *ends* "...A comment." or
# contains "Corrections for"/"...responses." far upstream is NOT suppressed.
_INTITLE_CITE = re.compile(
    r"\b(?:repl(?:y|ies|ied)|comment\w*|response|respond\w*|rejoinder|correction"
    r"|corrigend\w*|erratum|errata|critique|reanalys\w*|re-?analys\w*|replicat\w*"
    r"|reconsider\w*|revisit\w*|reexamin\w*|re-?examin\w*)\b[^.(]{0,35}"
    r"\b(?:to|on|of)\b[^.(]{0,40}$",
    re.IGNORECASE,
)

_CAP_TOKEN = re.compile(r"[A-Z][\w.''\-]*$")
_CONNECTOR = {"and", "of", "for", "the", "&", "de", "van", "von", "der"}
_MAX_LEAD = 60

# Bare-year reference onset (arXiv / ACL / Springer "FirstName LastName, ...
# YEAR. Title." style). Unlike _DATE_ANCHOR the year is unparenthesized, so the
# "<byline>. YEAR. <title>" shape is the whole signal: a year that opens its own
# mini-sentence — a period boundary before it, a period after it — with an author
# byline immediately preceding. group(1) is the bare 4-digit year (an optional
# disambiguating suffix letter follows) for the 1900-2035 sanity range.
_BARE_YEAR_ONSET = re.compile(r"(?<=[.)])\s+((?:19|20)\d\d)[a-z]?\.\s")
# An interior onset must start this far into the string — the first reference's
# own byline opens at offset 0 and is never an interior split.
_MIN_INTERIOR_OFFSET = 40
# Bylines run longer than _MAX_LEAD for multi-author arXiv lists ("Luisa
# Bentivogli, Bernardo Magnini, Ido Dagan, ..."); cap the byline heuristic here.
_MAX_BYLINE_LEAD = 120

# Numbered-list reference onset (Vancouver/Nature/IEEE): a reference-ending
# boundary (')' or '.') then "N. " then a letter — the next author, lowercase
# particle names ("ten Cate", "van der …") included. The lookbehind keeps the
# boundary out of the match so the onset lands on the digit (the new ref keeps
# its number). The boundary requirement excludes "Edition 2." (a word, not "."/")"
# precedes it); the increment check (below) excludes volume/page/year integers.
_NUM_ONSET = re.compile(r"(?<=[.)])\s+(\d{1,3})\.\s+(?=[^\W\d_])")
_NUMBERED_ENTRY_START = re.compile(r"\s*(\d{1,3})\.\s+(?=[^\W\d_])")

# Abbreviations that legitimately precede an interior "N." inside ONE reference
# (volume/edition/page/section markers) — a number after these is not a new ref.
_NUM_ABBREV_BEFORE = re.compile(
    r"\b(?:vol|vols|no|nos|pp|p|ed|eds|edn|chap|chaps|pt|pts|sec|secs"
    r"|fig|figs|suppl|ser|bd)\.\s*$",
    re.IGNORECASE,
)

# Edition/volume words printed AFTER the number ("2. Aufl.", "4. Auflage",
# "3. udg.", "2. uppl.", "2. ed.", "5. baskı") in German, Nordic, Slavic,
# Hungarian, Turkish and Romance bibliographies. An interior "M. " followed by
# one of these continues the same reference even when M is the next entry
# number.
_NUM_EDITION_AFTER = re.compile(
    r"(?:aufl(?:age)?|ausg(?:abe)?|udg(?:ave)?|utg(?:ave)?|uppl(?:aga)?|oppl(?:ag)?"
    r"|painos|ed|edn|izd|изд|vyd|wyd|kiad(?:ás)?|bask[ıi]|bd|jg|jahrg(?:ang)?|hrsg)\b",
    re.IGNORECASE,
)

# Any letter: the title text that separates two references' date anchors.
_LETTER = re.compile(r"[^\W\d_]")


@dataclass(frozen=True)
class MergeCandidate:
    """A ref string flagged as containing >1 reference (flag-only, non-mutating)."""

    index: int
    offsets: list[int]
    contexts: list[str]


def _last_cap_run_start(head: str) -> int | None:
    """Start offset of the trailing run of capitalized author/org tokens in
    ``head`` (lowercase connectors allowed mid-run), or None if the last token is
    not capitalized. Numbers break the run, so an earlier journal/volume cannot be
    swept in."""
    toks = [(m.start(), m.group()) for m in re.finditer(r"\S+", head)]
    start: int | None = None
    for off, tok in reversed(toks):
        clean = tok.strip(".,;&")
        if _CAP_TOKEN.match(clean) or (start is not None and clean.lower() in _CONNECTOR):
            start = off
        else:
            break
    return start


def _is_author_like(lead: str) -> bool:
    """True when ``lead`` looks like a short author/org name, not prose or a title."""
    if not lead or len(lead) > _MAX_LEAD or not lead[:1].isupper():
        return False
    words = re.findall(r"[^\W\d_]+", lead)
    if not words:
        return False
    capped = sum(1 for w in words if w[:1].isupper())
    return capped / len(words) >= 0.6


def _lead_start(s: str, anchor_pos: int) -> int | None:
    """Offset where the reference owning the date anchor at ``anchor_pos`` begins,
    or None when the text before the anchor is not an author/org lead.

    Considers two candidate starts — the last ". " sentence boundary, and the
    trailing capitalized-token run (the softening for URL-ended prior refs like
    "...fungus-strain United Nations. (n.d.)"). Among the candidates whose lead is
    author-like, returns the smallest offset so a leading "U.S." is not dropped.
    """
    head = s[:anchor_pos].rstrip()
    candidates: set[int] = set()
    term = head.rfind(". ")
    if term != -1:
        candidates.add(term + 2)
    cap = _last_cap_run_start(head)
    if cap is not None:
        candidates.add(cap)
    valid = [c for c in candidates if c > 0 and _is_author_like(s[c:anchor_pos].strip(" ."))]
    return min(valid) if valid else None


def _numbered_interior_onsets(ref_string: str) -> list[int]:
    """Char offsets (>0) where a new NUMBERED reference begins inside ``ref_string``.

    For Vancouver/Nature/IEEE bibliographies the year sits at the END of each
    reference, so the author-date heuristic finds no author lead before the second
    date and under-splits. The unambiguous delimiter is instead the sequential
    "N." marker. Anchored on a leading "N.", an interior "M." is an onset only when
    it sits at a ref-ending boundary (``_NUM_ONSET``), is not an abbreviation
    (``_NUM_ABBREV_BEFORE``, e.g. "Vol. 2."), is not an edition or volume
    number (``_NUM_EDITION_AFTER``, e.g. "2. Aufl."), and continues the
    sequence (``M == previous + 1``). The strict increment is the precision
    safeguard: volume/page/year integers don't form a run, so they can't be
    onsets. Empty unless the string starts with a number followed by a letter."""
    m0 = _NUMBERED_ENTRY_START.match(ref_string)
    if m0 is None:
        return []
    expected = int(m0.group(1)) + 1
    offsets: list[int] = []
    for m in _NUM_ONSET.finditer(ref_string):
        if int(m.group(1)) != expected:
            continue
        if _NUM_ABBREV_BEFORE.search(ref_string[: m.start()]):
            continue
        if _NUM_EDITION_AFTER.match(ref_string, m.end()):
            continue
        offsets.append(m.start(1))
        expected += 1
    return offsets


def _parendate_interior_onsets(ref_string: str) -> list[int]:
    """Char offsets (>0) where a new PARENTHESIZED-date reference begins inside
    ``ref_string`` — the author-date onset shape ("Author, A. (YEAR). ..."). An
    onset is a date anchor (``_DATE_ANCHOR``) whose preceding text is an
    author/org lead (``_lead_start``), skipping second dates that are reference
    metadata (``_META_BEFORE``) or an in-title citation (``_INTITLE_CITE``).
    Empty unless the string carries at least two parenthesized dates.

    A merge puts reference 1's title (and container) between its date and
    reference 2's author lead. A lead that opens right after the date of the
    reference it would end, as in "Brown, T. (2018). Beyond Kahneman and
    Tversky (1979): ...", is the start of a title that cites another work, so
    it is not an onset. That date is the one that opened the current
    reference (the first anchor, then each accepted onset's), not merely the
    previous anchor: a reference whose title ends in its own "(2000)" still
    ends there when the next reference follows."""
    anchors = list(_DATE_ANCHOR.finditer(ref_string))
    if len(anchors) < 2:
        return []
    offsets: list[int] = []
    head = anchors[0]
    for i in range(1, len(anchors)):
        between = ref_string[anchors[i - 1].end() : anchors[i].start()]
        if _META_BEFORE.search(between):
            continue
        if _INTITLE_CITE.search(ref_string[: anchors[i].start()]):
            continue
        start = _lead_start(ref_string, anchors[i].start())
        if start is None or start <= 0 or start in offsets:
            continue
        if not _LETTER.search(ref_string[head.end() : start]):
            continue
        offsets.append(start)
        head = anchors[i]
    return sorted(offsets)


def _looks_like_byline(lead: str) -> bool:
    """True when ``lead`` is a short author byline — mostly capitalized name
    tokens, initials, and connectors ("and"/"&"/"de"/…) rather than a title or
    prose. Starts with a capital and stays under ``_MAX_BYLINE_LEAD`` chars, so a
    title's lowercase-heavy word run is rejected."""
    lead = lead.strip()
    if not lead or len(lead) > _MAX_BYLINE_LEAD or not lead[:1].isupper():
        return False
    words = re.findall(r"[^\W\d_]+", lead)
    if not words:
        return False
    namey = sum(1 for w in words if w[:1].isupper() or w.lower() in _CONNECTOR)
    return namey / len(words) >= 0.8


def _bareyear_interior_onsets(ref_string: str) -> list[int]:
    """Char offsets (>0) where a new BARE-YEAR reference begins inside
    ``ref_string``.

    The arXiv/ACL/Springer style ("FirstName LastName, ... YEAR. Title.")
    carries neither a numbered marker nor a parenthesized date, so it slips past
    both other detectors and the geom segmenter merges the whole run into one
    record. The tell is the "<byline>. YEAR. <title>" shape: a standalone year
    (``_BARE_YEAR_ONSET``, 1900-2035) immediately preceded by a
    period-terminated author byline (``_looks_like_byline``) that itself starts
    at least ``_MIN_INTERIOR_OFFSET`` chars in. Precision guards: the byline must
    open a new reference — the preceding text must end at a full ". " boundary
    (not the whole string's start, and not on a lone author initial like
    "K. Toutanova. 2019."), and the byline span must read as names, not a title
    (which keeps a title's own embedded year from firing)."""
    offsets: list[int] = []
    for m in _BARE_YEAR_ONSET.finditer(ref_string):
        year = int(m.group(1))
        if not (1900 <= year <= 2035):
            continue
        # m.start() sits on the whitespace after the byline-terminating period.
        byline_end = m.start()
        term = ref_string.rfind(". ", 0, byline_end)
        if term == -1:
            continue  # first reference — its byline opens the whole string
        byline_start = term + 2
        if byline_start < _MIN_INTERIOR_OFFSET:
            continue
        # A lone-initial boundary ("... and K. Toutanova. 2019.") pauses inside a
        # single byline, not between references — the year is that ref's own.
        prev_toks = ref_string[:term].rsplit(None, 1)
        if prev_toks and len(prev_toks[-1].strip(".")) <= 1:
            continue
        byline = ref_string[byline_start:byline_end].strip(" .")
        if not _looks_like_byline(byline):
            continue
        if byline_start not in offsets:
            offsets.append(byline_start)
    return sorted(offsets)


def find_interior_onsets(ref_string: str) -> list[int]:
    """Sorted unique char offsets (>0) where a new reference begins inside
    ``ref_string``. Empty when the string looks like a single reference."""
    # Numbered (Vancouver/Nature/IEEE) lists: the "N." markers are authoritative,
    # so use them exclusively — the author-date pass mis-fires on year-at-end
    # references (splitting at an interior journal/volume run).
    numbered = _numbered_interior_onsets(ref_string)
    if numbered:
        return numbered
    # Parenthesized author-date ("Author, A. (YEAR). ...") next; only when it
    # finds nothing do we try the bare-year ("Author. YEAR. ...") shape, so the
    # two never fight over the same string.
    parendate = _parendate_interior_onsets(ref_string)
    if parendate:
        return parendate
    return _bareyear_interior_onsets(ref_string)


def _onset_finder_for_bibliography(
    ref_strings: list[str],
) -> Callable[[str], list[int]]:
    """Choose one onset strategy for the whole bibliography.

    Printed numbering is authoritative when it appears on a strict majority
    of non-empty segments. An unnumbered continuation fragment is tolerated,
    but numbered single references never fall through to author-date rules.
    """
    nonempty = [segment for segment in ref_strings if segment.strip()]
    marked = sum(1 for segment in nonempty if _NUMBERED_ENTRY_START.match(segment))
    if nonempty and marked * 2 > len(nonempty):
        return _numbered_interior_onsets
    return find_interior_onsets


def detect_merges(ref_strings: list[str]) -> list[MergeCandidate]:
    """Flag-only: one MergeCandidate per string that has interior onsets."""
    find_onsets = _onset_finder_for_bibliography(ref_strings)
    out: list[MergeCandidate] = []
    for i, s in enumerate(ref_strings):
        offsets = find_onsets(s)
        if offsets:
            contexts = [s[max(0, o - 20) : o + 20] for o in offsets]
            out.append(MergeCandidate(index=i, offsets=offsets, contexts=contexts))
    return out


def split_merged_refs(ref_strings: list[str]) -> tuple[list[str], int]:
    """Split each string at its interior onsets. Returns (corrected, n_new).

    One-sided-safe: ``len(corrected) >= len(ref_strings)``. Best-effort: any error
    on a single string leaves that string unchanged (never raises). A split is
    accepted only when every resulting piece is non-empty after stripping; an
    all-non-empty check keeps the count strictly non-decreasing.
    """
    find_onsets = _onset_finder_for_bibliography(ref_strings)
    out: list[str] = []
    for s in ref_strings:
        try:
            offsets = find_onsets(s)
        except Exception:  # noqa: BLE001 — never fail extraction over a split
            offsets = []
        if not offsets:
            out.append(s)
            continue
        bounds = [0, *offsets, len(s)]
        pieces = [s[a:b].strip() for a, b in zip(bounds, bounds[1:], strict=False)]
        if all(pieces):
            out.extend(pieces)
        else:
            out.append(s)
    return out, len(out) - len(ref_strings)
