"""Snap reference anchors / layout lines onto the reference text.

Two entry points share the exact/fuzzy locating machinery:

* :func:`segment_by_anchors` — the LLM segmentation step emits one short
  VERBATIM anchor per reference (the opening ~30 chars). Each anchor is
  located independently (exact match, then rapidfuzz fallback for OCR
  noise), with line-start preference and full-anchor scoring to resolve
  embedded sub-list-authorship collisions.
* :func:`align_line_starts` — the geometry segmenter knows the full ORDERED
  sequence of reference-section lines and which of them open a reference.
  The ordered walk (monotonic cursor) locates every line, so an opening
  embedded inside an earlier reference can never steal a match: the cursor
  has already passed it.

Pure Python — no torch, no gold dependency. Ported from the boundary-eval
(`evaluation/llm_vs_crf_seg.py`), itself derived from
`bibr_training.data.gold_seg`, so production behavior matches what was measured.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from bibr.utils.constants import ANCHOR_LEN, ANCHOR_PROMPT_CHARS
from bibr.utils.text import collapse_ws

__all__ = [
    "ANCHOR_LEN",
    "ANCHOR_PROMPT_CHARS",
    "align_line_starts",
    "find_anchor_starts",
    "segment_by_anchors",
    "starts_to_spans",
]

_FUZZY_MIN_SCORE = 85

# A layout line whose collapsed text is shorter than this cannot be located
# reliably (page numbers, stray markers) — skip it rather than fuzzy-match it
# somewhere destructive.
_MIN_LINE_PROBE = 12


def _anchor(s: str | None) -> str:
    """First ``ANCHOR_LEN`` chars of a stripped anchor string."""
    if not s:
        return ""
    return s.strip()[:ANCHOR_LEN]


def _all_exact_positions(text: str, anchor: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        pos = text.find(anchor, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def _best_fuzzy_position(text: str, anchor: str) -> int:
    """Best-scoring start of *anchor* in *text*, or -1 below ``_FUZZY_MIN_SCORE``.

    Uses rapidfuzz's C-level alignment search (one call) instead of a Python
    loop calling ``fuzz.ratio`` at every character offset.
    """
    n = len(anchor)
    if n == 0 or len(text) < n:
        return -1
    alignment = fuzz.partial_ratio_alignment(anchor, text, score_cutoff=_FUZZY_MIN_SCORE)
    if alignment is None:
        return -1
    return alignment.dest_start


def _is_line_start(text: str, pos: int) -> bool:
    return pos == 0 or text[pos - 1] == "\n"


def _choose_exact_position(
    ref_text: str, full_anchor: str, candidates: list[int], claimed: set[int]
) -> int:
    """Pick the best unclaimed exact-match position for an anchor.

    True reference starts sit at line starts in row-joined ref text, while a
    sibling reference's truncated anchor can also exact-match *inside* a
    longer reference's author list (sub-list authorship — e.g. the
    "Hoshi, T., & Kashyap, A. K. (2" prefix of a 2010 reference embedded in
    "Caballero, R. J., Hoshi, T., & Kashyap, A. K. (2008)"). Prefer
    line-start candidates; among several, let the full emitted anchor (which
    usually carries the disambiguating year) score them, breaking ties by
    lowest position.
    """
    unclaimed = [p for p in candidates if p not in claimed]
    if not unclaimed:
        return -1
    line_starts = [p for p in unclaimed if _is_line_start(ref_text, p)]
    pool = line_starts or unclaimed
    if len(pool) == 1:
        return pool[0]
    n = len(full_anchor)
    return max(pool, key=lambda p: (fuzz.ratio(ref_text[p : p + n], full_anchor), -p))


def find_anchor_starts(ref_text: str, anchors: list[str]) -> list[int]:
    """Claim the best unclaimed start position for each anchor.

    Exact matches of the truncated anchor are preferred (line-start
    occurrences over embedded mid-reference ones, full-anchor score among
    several — see :func:`_choose_exact_position`), with a fuzzy fallback for
    OCR noise. Anchors that cannot be located are dropped. Returns sorted
    start offsets.
    """
    claimed: set[int] = set()
    starts: list[int] = []
    for a in anchors:
        anc = _anchor(a)
        if not anc:
            continue
        candidates = _all_exact_positions(ref_text, anc)
        chosen = _choose_exact_position(ref_text, a.strip(), candidates, claimed)
        if chosen == -1:
            fuzzy = _best_fuzzy_position(ref_text, anc)
            if fuzzy != -1 and fuzzy not in claimed:
                chosen = fuzzy
        if chosen == -1:
            continue
        claimed.add(chosen)
        starts.append(chosen)
    return sorted(starts)


def align_line_starts(ref_text: str, line_texts: list[str], is_boundary: list[bool]) -> list[int]:
    """Monotonically align ordered layout lines to *ref_text*; return the
    start offsets of the boundary (reference-opening) lines.

    Every line — boundary or continuation — is located from a cursor that
    prefers moving forward, so the alignment tracks the text line by line
    and a boundary opening embedded in an ALREADY-PASSED line cannot steal
    a match that also exists ahead. Within the forward candidates the
    line-start preference and full-line scoring of
    :func:`_choose_exact_position` still resolve duplicates ahead of the
    cursor. Monotonicity is SOFT: the layout line stream (pdfium content
    order) and the row text (layout-region order) can locally disagree
    about the order of same-first-author references, so a boundary with no
    forward exact match may claim an unclaimed exact match behind the
    cursor — without moving the cursor backward. Lines that cannot be
    located at all (page furniture, dropped rows, appendix tail) are
    skipped without advancing the cursor; a boundary line additionally gets
    a fuzzy attempt (OCR noise) before being dropped. Line text is
    whitespace-collapsed to match ``_build_ref_text``'s row collapsing.
    """
    cursor = 0
    starts: list[int] = []
    for line, boundary in zip(line_texts, is_boundary, strict=True):
        full = collapse_ws(line)
        probe = full[:ANCHOR_LEN]
        if len(probe) < _MIN_LINE_PROBE:
            continue
        positions = _all_exact_positions(ref_text, probe)
        forward = [p for p in positions if p >= cursor]
        chosen = _choose_exact_position(ref_text, full, forward, claimed=set(starts))
        advance = True
        if chosen == -1 and boundary:
            behind = [p for p in positions if p < cursor]
            chosen = _choose_exact_position(ref_text, full, behind, claimed=set(starts))
            advance = False
            if chosen == -1:
                fuzzy = _best_fuzzy_position(ref_text[cursor:], probe)
                if fuzzy != -1 and cursor + fuzzy not in starts:
                    chosen = cursor + fuzzy
                    advance = True
        if chosen == -1:
            continue
        if boundary:
            starts.append(chosen)
        if advance:
            cursor = chosen + len(probe)
    return starts


def starts_to_spans(ref_text: str, starts: list[int]) -> list[tuple[int, int]]:
    """Convert sorted start offsets into (start, end) spans.

    Each span runs to the next start (or end of text), with trailing
    whitespace trimmed off the end. Reference text preceding the first anchor is
    never silently discarded: when the first entry's anchor fails to snap (OCR
    noise / fuzzy miss), that leading block is emitted as its own span so the
    genuine first reference survives (D2). ``segment_filter`` drops it later if
    it is only header/byline noise.
    """
    spans: list[tuple[int, int]] = []
    starts = sorted(starts)
    if starts:
        lead_start = 0
        while lead_start < starts[0] and ref_text[lead_start] in " \t\n":
            lead_start += 1
        lead_end = starts[0]
        while lead_end > lead_start and ref_text[lead_end - 1] in " \t\n":
            lead_end -= 1
        if lead_end > lead_start:
            spans.append((lead_start, lead_end))
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(ref_text)
        ref_end = end
        while ref_end > start and ref_text[ref_end - 1] in " \t\n":
            ref_end -= 1
        if ref_end > start:
            spans.append((start, ref_end))
    return spans


def segment_by_anchors(ref_text: str, anchors: list[str]) -> list[str]:
    """Return the reference substrings located by *anchors*, in text order."""
    starts = find_anchor_starts(ref_text, anchors)
    spans = starts_to_spans(ref_text, starts)
    return [ref_text[s:e] for s, e in spans]
