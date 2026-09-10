"""Drop non-reference segments produced by reference segmentation.

The anchor-emit segmenter faithfully segments whatever sits in the references
block — including front/back-matter that boundary-orphan reclaim or loose
section boundaries pull in: Supplemental-Material pointer lines, Open
Practices box text, MDPI Disclaimer/Publisher's Notes, journal DOI footers,
and the paper's own byline. The retired rule splitter dropped these
incidentally; this filter does it deliberately, before parsing.

Patterns are mined from the exp #2 judge corrections
(``evaluation/results/exp2_verdict.md``) and kept deliberately conservative:
explicit start-anchored junk phrases, segments that are nothing but a
URL/DOI, and short fragments that carry no year-like token and no locator
(bylines, stray headings). Real references — including ``(n.d.)`` and
``(in press)`` entries — must never match.
"""

from __future__ import annotations

import logging
import re

from bibr.utils.text import DOI_BODY, YEARISH_RE

logger = logging.getLogger(__name__)

# Front/back-matter phrases observed as bogus bib entries (start-anchored so a
# real reference whose *title* contains these words is never dropped).
_JUNK_START_RE = re.compile(
    r"^(?:"
    r"additional supporting information\b"
    r"|supplemental material\b"
    r"|more information about the open practices\b"
    r"|open practices\b"
    r"|disclaimer\s*/\s*publisher[’']s note\b"
    r"|publisher[’']s note\s*:"
    r"|please go to\s*:"
    r")",
    re.IGNORECASE,
)

# A segment that is nothing but a URL / DOI (journal footer fragments).
_URL_ONLY_RE = re.compile(
    r"^(?:https?://\S+|doi:\s*\S+|(?:https?://)?(?:dx\.)?doi\.org/\S+|10\.\d{4,9}/\S+)$",
    re.IGNORECASE,
)

_LOCATOR_RE = re.compile(r"https?://|\b" + DOI_BODY, re.IGNORECASE)

# Below this length, a segment with no year-like token and no URL/DOI cannot
# plausibly be a complete reference (authors + title + container alone exceed
# it) — observed cases are bylines and stray headings.
_MIN_PLAUSIBLE_NO_YEAR_LEN = 70

# A 4-digit publication year with optional disambiguation suffix.
_YEAR_RE = re.compile(r"(?:1[6-9]|20)\d{2}[a-z]?")
# A parenthesized publication year — present in real references
# ("Keynes, J. M. (1930)…"), absent from bare in-text cites ("Gelman, 2006").
_PAREN_YEAR_RE = re.compile(r"\((?:1[6-9]|20)\d{2}[a-z]?\)")
# Trailing punctuation an in-text citation can carry (";", ")", ".", "]" …).
_CITATION_TAIL_RE = re.compile(r"[\s.,;:)\]]*\Z")
# Above this length a year-terminal segment may be a real reference whose
# title was lost upstream — leave it alone rather than risk dropping a real
# (if incomplete) entry. In-text-cite fragments observed are far shorter.
_MAX_INTEXT_CITATION_LEN = 80
# Structure a bare in-text citation never carries. A printed list number
# introduces a numbered bibliography entry ("12." / "[3]"), and a sentence
# break before the year means a title/container preceded it. Both mark
# Vancouver and IEEE entries, which terminate at the year the way an in-text
# cite does ("… Boston: Little, Brown; 1986.") and would otherwise be dropped.
_LIST_NUMBER_RE = re.compile(r"^\[?\d{1,3}[\].)]")
_SENTENCE_BREAK_RE = re.compile(r"\.\s")


def _is_intext_citation(s: str) -> bool:
    """True if *s* is a bare in-text parenthetical citation, not a reference.

    In-text cites that leak into ref_text (e.g. an acknowledgment sentence
    reclaimed as a boundary orphan: ``(Gelman & Stern, 2006; Nieuwenhuis
    et al., 2011)``) get split by the anchor segmenter into ``Author, YEAR``
    fragments. They carry a year, so the year guard keeps them — but they
    *terminate* at the year with no title following, unlike a real reference
    (``Author, A. (2006). Title. Journal…``). Detect: short, ends at a year
    (modulo trailing punctuation), and that year is not a page-range number.
    """
    core = s.strip().lstrip("(").rstrip()
    if len(core) > _MAX_INTEXT_CITATION_LEN:
        return False
    # A parenthesized year marks a real reference's publication year — an
    # in-text cite uses a bare year. Protects reprint-year-terminal refs
    # ("Keynes, J. M. (1930). … Classiques Garnier Edition, 2019.").
    if _PAREN_YEAR_RE.search(core):
        return False
    years = list(_YEAR_RE.finditer(core))
    if not years:
        return False
    last = years[-1]
    # Page-range numbers (e.g. "1943–1977") are not citation years.
    if last.start() > 0 and core[last.start() - 1] in "-–—/":
        return False
    # Numbered styles (Vancouver, IEEE) do end at the year, so the tail test
    # below cannot separate them from an in-text cite — reference structure in
    # the text *before* the year does.
    if _LIST_NUMBER_RE.match(core) or _SENTENCE_BREAK_RE.search(core[: last.start()]):
        return False
    # A real reference has its title *after* the year; an in-text cite has
    # nothing but punctuation there.
    return bool(_CITATION_TAIL_RE.match(core, last.end()))


def is_non_reference_segment(segment: str) -> bool:
    """True if *segment* is front/back-matter rather than a bibliography entry."""
    s = segment.strip()
    if not s:
        return True
    if _JUNK_START_RE.match(s):
        return True
    if _URL_ONLY_RE.match(s):
        return True
    if _is_intext_citation(s):
        return True
    return (
        len(s) < _MIN_PLAUSIBLE_NO_YEAR_LEN
        and not YEARISH_RE.search(s)
        and not _LOCATOR_RE.search(s)
    )


def drop_non_reference_segments(segments: list[str]) -> list[str]:
    """Filter *segments*, keeping order; logs what was dropped."""
    kept = [s for s in segments if not is_non_reference_segment(s)]
    dropped = len(segments) - len(kept)
    if dropped:
        logger.info(f"Dropped {dropped} non-reference segment(s) before parsing")
    return kept
