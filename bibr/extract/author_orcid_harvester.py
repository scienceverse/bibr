"""Document-wide and PDF-link ORCID harvester for paper authors.

Bridges parsed ``PaperContents`` to extracted authors (shared by both the
default LLM route and ``--fast``):

1. **Printed-text pass (all non-reference pages, including scans):** scans
   every non-reference section for printed ORCID URLs and bare 16-digit IDs,
   rejoining IDs wrapped across a line break (e.g. Taylor & Francis back-matter
   ``http://orcid.org/0000-0003-\\n3786-872X``), validating every candidate with
   the ISO 7064 (MOD 11-2) check digit, and attributing each ID to the author
   named beside it (full name, surname with given-name disambiguation, or
   unique initials prefix like Royal Society Open Science ``MKS, 0000-...``).

2. **PDF URI-link pass (born-digital PDFs):** reads ``orcid.org`` link
   annotations already collected in ``contents.pdf_uri_links`` alongside the
   same-coordinate-frame ``contents.ref_page_lines`` to attach link-only ORCID
   icons (which ~60% of PMC OA papers print only as clickable badges next to
   byline names, with no ORCID string in the text layer).

Only fills empty ``author.orcid`` slots; never overwrites an existing valid
ORCID and never assigns the same ORCID to two different authors.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from bibr.extract.author_email_harvester import (
    _fold,
    _fold_letters,
    _given_tokens,
    _initials,
    _initials_compatible,
)
from bibr.models import canonicalize_orcid, is_valid_orcid_checksum

if TYPE_CHECKING:
    from bibr.models import PaperAuthor
    from bibr.paper_contents import PaperContents

logger = logging.getLogger(__name__)

# Matches an ORCID URL or bare 16-digit hyphenated ORCID. Allows a single space
# or hyphen between 4-digit groups when preceded by orcid.org/, and strict
# hyphens for bare IDs so spaced table numbers are never scanned.
_ORCID_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?orcid\.org/\s*([0-9]{4})[\s-]([0-9]{4})[\s-]([0-9]{4})[\s-]([0-9]{3}[0-9Xx])\b",
    re.IGNORECASE,
)
_ORCID_BARE_RE = re.compile(
    r"(?<![0-9A-Za-z-])([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9]{3}[0-9Xx])(?![0-9A-Za-z-])"
)
_ORCID_ANY_RE = re.compile(
    rf"(?:{_ORCID_URL_RE.pattern})|(?:{_ORCID_BARE_RE.pattern})",
    re.IGNORECASE,
)

# Rejoin an ORCID split across a line wrap right after a hyphen or slash:
# "http://orcid.org/0000-0003-\n3786-872X" or "http://orcid.org/\n0000-0003-..."
_WRAPPED_ORCID_HYPHEN_RE = re.compile(
    r"(orcid\.org/[0-9-]*-|[0-9]{4}(?:-[0-9]{4}){0,2}-)[ \t\r]*\n[ \t]*([0-9]{3,4}[0-9Xx]?)",
    re.IGNORECASE,
)
_WRAPPED_ORCID_SLASH_RE = re.compile(
    r"(orcid\.org/)[ \t\r]*\n[ \t]*([0-9]{4}-[0-9]{4})",
    re.IGNORECASE,
)

# Strip ORCID badges glued to surnames ("SmithID1") and trailing punctuation.
_ORCID_BADGE_RE = re.compile(r"(?<=[^\W\d_])ID(?=\d|\b)")

# Editorial/reviewer block cues — prevent attaching editor/reviewer ORCID icons
# (e.g. Frontiers "Edited by" / "Reviewed by" sidebars) to paper authors.
_EDITORIAL_CONTEXT_RE = re.compile(
    r"\b(?:edited\s+by|reviewed\s+by|(?:academic|handling|responsible|action|guest|associate|managing|section)\s+editor|editor\s*:)\b",
    re.IGNORECASE,
)

# Trailing labels right before an ORCID that belong to the label, not a name.
_TRAILING_ORCID_LABEL_RE = re.compile(
    r"(?:\b(?:author(?:'?s)?|orcid(?:\s+i?d)?|i?d|https?|url|link)\b|[:;,(\[\-–—/\s])+$",
    re.IGNORECASE,
)

# Trailing initials token directly before an ORCID (e.g. "MKS, 0000-..." or "O.L.: 0000-...").
_TRAILING_INITIALS_RE = re.compile(
    r"(?:^|[\s;,(\[])([A-Z](?:\.?[ \t]?[A-Z]){1,4}\.?)\s*[,;:\-–—]?\s*$"
)

# Word tokenizer for name matching inside context snippets.
_NAME_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")

# Horizontal gap (in 0..1000 layout units) across which a separate column's text
# (e.g. a left sidebar's "Accepted: ..." line) is not part of the link's line.
_MAX_COL_GAP_LAYOUT = 90.0

# Maximum vertical center delta (in 0..1000 layout units) for merging split
# link rects that belong to the same ORCID icon on one line.
_MERGE_LINK_Y_TOL = 20.0


def _rejoin_wrapped_orcids(text: str) -> str:
    """Rejoin ORCIDs split across a newline after a hyphen or ``orcid.org/``."""
    if not text or "\n" not in text:
        return text
    prev = None
    cur = _WRAPPED_ORCID_SLASH_RE.sub(r"\1\2", text)
    while cur != prev:
        prev = cur
        cur = _WRAPPED_ORCID_HYPHEN_RE.sub(r"\1\2", cur)
    return cur


def _extract_valid_orcid(match: re.Match[str]) -> str | None:
    """Canonicalize and ISO-7064-validate a regex match from ``_ORCID_ANY_RE``."""
    groups = match.groups()
    parts = groups[0:4] if groups[0] is not None else groups[4:8]
    if any(p is None for p in parts):
        return None
    raw = "-".join(parts).upper()
    canonical = canonicalize_orcid(raw)
    if canonical is None or not is_valid_orcid_checksum(canonical):
        return None
    return canonical


def _author_family_tokens(family: str | None) -> tuple[str, ...]:
    """Folded word tokens of a surname (handles multi-word and hyphenated surnames)."""
    folded = _fold_letters(family)
    if not folded:
        return ()
    return tuple(tok for tok in re.split(r"[^a-z0-9]+", folded) if tok)


def _author_given_tokens(given: str | None) -> tuple[str, ...]:
    folded = _fold_letters(given)
    if not folded:
        return ()
    return tuple(tok for tok in re.split(r"[^a-z0-9]+", folded) if tok)


def _author_initial_keys(author: PaperAuthor) -> set[str]:
    """All plausible full-name initialisms for *author* (e.g. Milla Karoliina Salonen -> MKS, MS)."""
    given_inits = _initials(_given_tokens(author.given))
    fam_tokens = _author_family_tokens(author.family)
    if not given_inits or not fam_tokens:
        return set()
    fam_inits = [t[0] for t in fam_tokens]
    keys = {
        "".join(given_inits + fam_inits),
        given_inits[0] + "".join(fam_inits),
        "".join(given_inits) + fam_inits[-1],
        given_inits[0] + fam_inits[-1],
    }
    return {k.upper() for k in keys if len(k) >= 2}


def _disambiguate_by_given(
    candidates: list[PaperAuthor],
    prefix_text: str,
) -> PaperAuthor | None:
    """Pick the best author among same-surname candidates using tokens before the surname."""
    if len(candidates) == 1:
        return candidates[0]
    if not prefix_text.strip():
        return None
    # Inspect the last few tokens directly before the surname.
    raw_words = _NAME_WORD_RE.findall(prefix_text)[-5:]
    if not raw_words:
        return None
    folded_words = [_fold(w) for w in raw_words if _fold(w)]
    prefix_inits = _initials(raw_words)

    scored: list[tuple[int, PaperAuthor]] = []
    for author in candidates:
        g_tokens = _author_given_tokens(author.given)
        g_inits = _initials(_given_tokens(author.given))
        score = 0
        # Exact full given-name token match (strongest signal, e.g. "Omid" vs "Fateme").
        for gt in g_tokens:
            if len(gt) >= 2 and gt in folded_words:
                score += 4
        # Initials compatibility (e.g. "O." / "F.").
        if prefix_inits and g_inits:
            if prefix_inits[-len(g_inits) :] == g_inits:
                score += 3
            elif _initials_compatible(prefix_inits[-1:], g_inits):
                score += 1
        scored.append((score, author))
    scored.sort(key=lambda item: -item[0])
    if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def _match_author_in_context(
    context: str,
    authors: list[PaperAuthor],
    *,
    prefer_rightmost: bool = True,
) -> PaperAuthor | None:
    """Find the author named nearest the ORCID edge of *context*.

    When ``prefer_rightmost=True`` (left-of-ORCID context), picks the surname
    ending furthest right; when ``False`` (right-of-ORCID context), picks the
    surname starting furthest left. Shared surnames are disambiguated by the
    given name immediately preceding the matched surname; unresolved ties return
    ``None``.
    """
    cleaned = _ORCID_BADGE_RE.sub("", context or "")
    if not cleaned.strip():
        return None

    # Build word spans with folded tokens for multi-word surname matching.
    word_matches = list(_NAME_WORD_RE.finditer(cleaned))
    if not word_matches:
        return None
    # Expand hyphenated words into sub-tokens while preserving character offsets.
    tokens: list[tuple[str, int, int]] = []
    for m in word_matches:
        w = m.group(0)
        sub_pos = m.start()
        for part in re.split(r"([-''’])", w):
            if not part:
                continue
            folded = _fold(part)
            if folded:
                tokens.append((folded, sub_pos, sub_pos + len(part)))
            sub_pos += len(part)
    if not tokens:
        return None

    # Group authors by their folded family token tuple.
    by_fam: dict[tuple[str, ...], list[PaperAuthor]] = {}
    for author in authors:
        ft = _author_family_tokens(author.family)
        if ft and sum(len(t) for t in ft) >= 2:
            by_fam.setdefault(ft, []).append(author)

    hits: list[tuple[int, int, int, tuple[str, ...], list[PaperAuthor]]] = []
    n_tok = len(tokens)
    for fam_toks, group in by_fam.items():
        k = len(fam_toks)
        for i in range(n_tok - k + 1):
            if all(tokens[i + j][0] == fam_toks[j] for j in range(k)):
                char_start = tokens[i][1]
                char_end = tokens[i + k - 1][2]
                hits.append((char_start, char_end, k, fam_toks, group))

    if not hits:
        return None

    if prefer_rightmost:
        # Furthest right end offset; break ties by longest multi-word surname.
        hits.sort(key=lambda h: (h[1], h[2]), reverse=True)
    else:
        # Furthest left start offset; break ties by longest multi-word surname.
        hits.sort(key=lambda h: (h[0], -h[2]))

    best_start, _best_end, _best_len, _fam, group = hits[0]
    prefix = cleaned[:best_start]
    # Also check suffix for family-first layouts if prefix has no given name.
    chosen = _disambiguate_by_given(group, prefix)
    if chosen is None and len(group) > 1:
        chosen = _disambiguate_by_given(group, cleaned[_best_end:])
    return chosen


def _match_author_by_initials(context: str, authors: list[PaperAuthor]) -> PaperAuthor | None:
    """Match a trailing initials token (e.g. ``MKS,`` or ``OL``) to a unique author."""
    m = _TRAILING_INITIALS_RE.search(context or "")
    if not m:
        return None
    code = re.sub(r"[^A-Z]", "", m.group(1))
    if len(code) < 2:
        return None
    matches = [a for a in authors if code in _author_initial_keys(a)]
    return matches[0] if len(matches) == 1 else None


def _left_and_right_text_from_lines(
    link_bbox: list[float],
    page_lines: list[dict[str, Any]],
) -> tuple[str, str, bool]:
    """Return ``(left_text, right_text, is_editorial)`` around *link_bbox* on its page.

    All coordinates are in the 0..1000 layout frame shared by
    ``contents.pdf_uri_links`` and ``contents.ref_page_lines``.
    """
    lx0, ly0, lx1, ly1 = link_bbox
    lym = (ly0 + ly1) / 2.0

    overlapping: list[dict[str, Any]] = []
    for ln in page_lines:
        bbox = ln.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        if bbox[1] - 4.0 <= lym <= bbox[3] + 4.0:
            overlapping.append(ln)
    if not overlapping:
        return "", "", False

    overlapping.sort(key=lambda ln: ln["bbox"][0])

    # Check if the row itself or an immediately adjacent line is an editorial block.
    row_full = " ".join(str(ln.get("text") or "") for ln in overlapping)
    if _EDITORIAL_CONTEXT_RE.search(row_full):
        return "", "", True
    # Also check lines directly above (within 30 layout units) in the same column.
    for ln in page_lines:
        bbox = ln.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        if (
            0.0 < ly0 - bbox[3] <= 30.0
            and bbox[0] <= lx1 + 40.0
            and bbox[2] >= lx0 - 200.0
            and _EDITORIAL_CONTEXT_RE.search(str(ln.get("text") or ""))
        ):
            return "", "", True

    left_parts: list[tuple[float, float, str]] = []
    right_parts: list[tuple[float, float, str]] = []

    for ln in overlapping:
        x0, _y0, x1, _y1 = ln["bbox"]
        text = str(ln.get("text") or "")
        if not text:
            continue
        if x1 <= lx0 + 12.0:
            left_parts.append((x0, x1, text))
        elif x0 >= lx1 - 6.0:
            right_parts.append((x0, x1, text))
        else:
            # Link sits inside this line's horizontal span: estimate char split.
            width = max(x1 - x0, 1.0)
            frac = max(0.0, min(1.0, (lx0 - x0) / width))
            cut = round(frac * len(text))
            # Finish the word directly under the icon without crossing a separator.
            while cut < len(text) and (text[cut].isalnum() or text[cut] in "-'’"):
                cut += 1
            left_slice = text[:cut].strip()
            right_slice = text[cut:].strip()
            if left_slice:
                left_parts.append((x0, lx0, left_slice))
            if right_slice:
                right_parts.append((lx1, x1, right_slice))

    # Drop unrelated left-column segments separated by a wide horizontal gap.
    kept_left: list[str] = []
    anchor_x = lx0
    for x0, x1, text in reversed(left_parts):
        if anchor_x - x1 > _MAX_COL_GAP_LAYOUT:
            break
        kept_left.append(text)
        anchor_x = x0
    kept_left.reverse()

    kept_right: list[str] = []
    anchor_rx = lx1
    for x0, x1, text in right_parts:
        if x0 - anchor_rx > _MAX_COL_GAP_LAYOUT:
            break
        kept_right.append(text)
        anchor_rx = x1

    return " ".join(kept_left).strip(), " ".join(kept_right).strip(), False


def _deduplicate_orcid_links(
    uri_links: Sequence[dict[str, Any]],
) -> list[tuple[int, str, list[float]]]:
    """Extract checksum-valid ``orcid.org`` links and merge split rects per icon."""
    raw: list[tuple[int, str, list[float]]] = []
    for link in uri_links:
        uri = link.get("uri") or ""
        if "orcid.org" not in uri.lower():
            continue
        canonical = canonicalize_orcid(uri)
        if canonical is None or not is_valid_orcid_checksum(canonical):
            continue
        page = link.get("page")
        bbox = link.get("bbox")
        if not isinstance(page, int) or not bbox or len(bbox) < 4:
            continue
        raw.append((page, canonical, [float(v) for v in bbox[:4]]))

    merged: list[tuple[int, str, list[float]]] = []
    for page, canonical, bbox in raw:
        ym = (bbox[1] + bbox[3]) / 2.0
        combined = False
        for idx, (mp, mc, mb) in enumerate(merged):
            if mp == page and mc == canonical:
                mym = (mb[1] + mb[3]) / 2.0
                if abs(ym - mym) <= _MERGE_LINK_Y_TOL:
                    merged[idx] = (
                        mp,
                        mc,
                        [
                            min(mb[0], bbox[0]),
                            min(mb[1], bbox[1]),
                            max(mb[2], bbox[2]),
                            max(mb[3], bbox[3]),
                        ],
                    )
                    combined = True
                    break
        if not combined:
            merged.append((page, canonical, bbox))

    # Pages 1-2 (byline) first, then back-matter pages, top-to-bottom within page.
    merged.sort(key=lambda item: (0 if item[0] <= 2 else 1, item[0], item[2][1], item[2][0]))
    return merged


def collect_printed_orcid_lines(
    contents: PaperContents,
    *,
    already: str = "",
    max_chars: int = 3000,
) -> list[str]:
    """Collect non-reference lines carrying valid printed ORCIDs not yet in *already*.

    Used to supply back-matter ORCID sections (Taylor & Francis ``ORCID``,
    eLife ``Author ORCIDs``, Ubiquity ``AUTHOR AFFILIATIONS``) to the authors
    LLM context even when ``FrontMatterResolution`` narrows the front matter.
    """
    from bibr.paper_contents import CanonicalSection

    raw_sections = getattr(contents, "sections", None)
    sections = raw_sections if isinstance(raw_sections, (list, tuple)) else ()
    raw_sentences = getattr(contents, "sentences", None)
    sentences = raw_sentences if isinstance(raw_sentences, (list, tuple)) else ()
    if not sentences:
        return []

    ref_section_ids = {
        s.section_id for s in sections if s.section_type == CanonicalSection.REFERENCES
    }
    by_section: dict[int, list[str]] = {}
    for sent in sentences:
        if sent.section_id in ref_section_ids:
            continue
        txt = (sent.text or "").strip()
        if txt:
            by_section.setdefault(sent.section_id, []).append(txt)

    seen_orcids: set[str] = set()
    for m in _ORCID_ANY_RE.finditer(_rejoin_wrapped_orcids(already or "")):
        c = _extract_valid_orcid(m)
        if c:
            seen_orcids.add(c)

    out: list[str] = []
    budget = max_chars
    for sec_texts in by_section.values():
        joined = _rejoin_wrapped_orcids("\n".join(sec_texts))
        if not _ORCID_ANY_RE.search(joined):
            continue
        lines = [ln.strip() for ln in joined.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            matches = [c for m in _ORCID_ANY_RE.finditer(line) if (c := _extract_valid_orcid(m))]
            new_ids = [c for c in matches if c not in seen_orcids]
            if not new_ids:
                continue
            # Include the previous line when the ORCID line carries no author name.
            snippet = line
            if i > 0 and not _NAME_WORD_RE.search(_ORCID_ANY_RE.sub("", line)):
                snippet = f"{lines[i - 1]} {line}"
            if len(snippet) > budget:
                break
            out.append(snippet)
            budget -= len(snippet) + 1
            seen_orcids.update(new_ids)
    return out


class AuthorOrcidHarvester:
    """Attach printed and PDF-linked ORCID iDs to extracted ``PaperAuthor`` objects."""

    def __init__(self, contents: PaperContents):
        self.contents = contents

    def harvest(self, authors: list[PaperAuthor]) -> int:
        """Backfill ``author.orcid`` from printed text and PDF URI link annotations.

        Mutates *authors* in place. Returns the number of authors updated.
        """
        if not authors:
            return 0

        claimed: set[str] = set()
        for author in authors:
            if author.orcid:
                canonical = canonicalize_orcid(author.orcid)
                if canonical:
                    claimed.add(canonical)

        attached = self._harvest_from_printed_text(authors, claimed)
        attached += self._harvest_from_uri_links(authors, claimed)
        if attached:
            logger.info("Harvested %d author ORCID iD(s)", attached)
        return attached

    def _harvest_from_printed_text(self, authors: list[PaperAuthor], claimed: set[str]) -> int:
        from bibr.paper_contents import CanonicalSection

        raw_sections = getattr(self.contents, "sections", None)
        sections = raw_sections if isinstance(raw_sections, (list, tuple)) else ()
        raw_sentences = getattr(self.contents, "sentences", None)
        sentences = raw_sentences if isinstance(raw_sentences, (list, tuple)) else ()
        if not sentences:
            return 0

        ref_section_ids = {
            s.section_id for s in sections if s.section_type == CanonicalSection.REFERENCES
        }
        by_section: dict[int, list[str]] = {}
        for sent in sentences:
            if sent.section_id in ref_section_ids:
                continue
            txt = (sent.text or "").strip()
            if txt:
                by_section.setdefault(sent.section_id, []).append(txt)

        attached = 0
        for sec_texts in by_section.values():
            joined = _rejoin_wrapped_orcids("\n".join(sec_texts))
            if not _ORCID_ANY_RE.search(joined):
                continue
            lines = [ln.strip() for ln in joined.splitlines() if ln.strip()]
            for line_idx, line in enumerate(lines):
                matches = list(_ORCID_ANY_RE.finditer(line))
                if not matches:
                    continue
                if _EDITORIAL_CONTEXT_RE.search(line):
                    continue
                prev_end = 0
                for idx, match in enumerate(matches):
                    canonical = _extract_valid_orcid(match)
                    if canonical is None or canonical in claimed:
                        prev_end = match.end()
                        continue
                    left_raw = line[prev_end : match.start()]
                    left_clean = _TRAILING_ORCID_LABEL_RE.sub("", left_raw).strip()
                    next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
                    right_raw = line[match.end() : next_start].strip()
                    prev_end = match.end()

                    has_left_name = bool(_NAME_WORD_RE.search(left_clean))
                    owner = _match_author_in_context(left_clean, authors, prefer_rightmost=True)
                    if owner is None:
                        owner = _match_author_by_initials(left_clean, authors)
                    if owner is None and idx == 0 and not has_left_name:
                        # Author name printed on the 1-2 lines directly above a bare ORCID line.
                        prev_ctx = " ".join(
                            lines[j]
                            for j in range(max(0, line_idx - 2), line_idx)
                            if not _ORCID_ANY_RE.search(lines[j])
                            and not _EDITORIAL_CONTEXT_RE.search(lines[j])
                        )
                        if prev_ctx:
                            owner = _match_author_in_context(
                                prev_ctx, authors, prefer_rightmost=True
                            )
                            if owner is None:
                                owner = _match_author_by_initials(prev_ctx, authors)
                    if owner is None and right_raw and not has_left_name:
                        owner = _match_author_in_context(right_raw, authors, prefer_rightmost=False)

                    if owner is not None and not owner.orcid:
                        owner.orcid = canonical
                        claimed.add(canonical)
                        attached += 1
        return attached

    def _harvest_from_uri_links(self, authors: list[PaperAuthor], claimed: set[str]) -> int:
        if all(author.orcid for author in authors):
            return 0
        raw_links = getattr(self.contents, "pdf_uri_links", None)
        uri_links = raw_links if isinstance(raw_links, (list, tuple)) else ()
        raw_lines = getattr(self.contents, "ref_page_lines", None)
        page_lines = raw_lines if isinstance(raw_lines, (list, tuple)) else ()
        if not uri_links or not page_lines:
            return 0

        deduped = _deduplicate_orcid_links(uri_links)
        if not deduped:
            return 0

        lines_by_page: dict[int, list[dict[str, Any]]] = {}
        for ln in page_lines:
            p = ln.get("page")
            if isinstance(p, int):
                lines_by_page.setdefault(p, []).append(ln)

        attached = 0
        for page, canonical, bbox in deduped:
            if canonical in claimed:
                continue
            p_lines = lines_by_page.get(page)
            if not p_lines:
                continue
            left_text, right_text, is_editorial = _left_and_right_text_from_lines(bbox, p_lines)
            if is_editorial:
                continue
            left_clean = _TRAILING_ORCID_LABEL_RE.sub("", left_text).strip()
            has_left_name = bool(_NAME_WORD_RE.search(left_clean))
            owner = _match_author_in_context(left_clean, authors, prefer_rightmost=True)
            if owner is None:
                owner = _match_author_by_initials(left_clean, authors)
            if owner is None and right_text and not has_left_name:
                owner = _match_author_in_context(right_text, authors, prefer_rightmost=False)
            if owner is not None and not owner.orcid:
                owner.orcid = canonical
                claimed.add(canonical)
                attached += 1
        return attached


__all__ = [
    "AuthorOrcidHarvester",
    "collect_printed_orcid_lines",
]
