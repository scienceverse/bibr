"""Layout-region anchor segmentation — zero-cost cascade fallback tier.

PP-DocLayoutV3 reference_content regions can mark bibliography entries, but the raw region stream also includes contribution statements, funding prose, and page-break continuations. This module keeps reference-onset anchors and snaps them onto flat ref_text with the shared anchor machinery (:mod:`bibr.extract.anchor_snap`).

Because the region signal comes from the layout detector, it exists even
for scanned PDFs with no text layer — exactly the papers the geometry
segmenter cannot handle. The tier claims a segmentation only when most of
its anchors align, so stale or noisy summaries decline to the next tier
instead of producing garbage.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

from bibr.extract.anchor_snap import find_anchor_starts, starts_to_spans
from bibr.extract.geom_features import _EARLY_YEAR_PAREN, _NUMBERED, _STARTS_YEAR
from bibr.ocr.ref_patterns import _looks_like_author_date_start
from bibr.utils.text import collapse_ws

if TYPE_CHECKING:
    from bibr.paper_contents import RegionSummary

# RegionSummary.content is truncated to 200 chars at capture; 80 keeps the
# anchor comfortably inside that while giving find_anchor_starts (which
# re-truncates to ANCHOR_LEN) full-anchor text for collision scoring.
_ANCHOR_LEN = 80
# Below this many usable anchors the "segmentation" would be trivial — the
# downstream tiers (CRF, marker split) handle short lists better.
_MIN_ANCHORS = 3
# Fraction of anchors that must locate in ref_text for the tier to claim the
# segmentation. In-document regions align near 1.0; wrong-document or noisy
# summaries fall to ~0, so anything in between just needs to separate the two.
_MIN_ALIGN_FRACTION = 0.6

_REGION_LABEL = "reference_content"

# Numbering styles include whitespace-separated, parenthesized, and dash-delimited prefixes.
# Missing these can starve the region tier and its independent segment-count sanity check.
_NUMBERED_LOOSE = re.compile(r"^\(?\d{1,3}\)?\s*[-–—]?\s+(?=[\"'“(\[]?[A-ZÀ-ÖØ-ÞЀ-ӿͰ-Ͽ])")

# Vancouver author onset without the author-date comma: "Alpert MA. Obesity …",
# "Bogers RP, …", "Абилова Г.А. Влияние …". Family name followed by 1-3 initial
# capitals and a delimiter — deliberately narrow so prose ("The WHO reported …")
# cannot match. Cyrillic is in the class because GOST-style bibliographies use
# exactly this shape.
_UPPER = r"A-ZÀ-ÖØ-ÞЀ-ЯЀ-ҁҊ-Ҿ"
_VANCOUVER_START = re.compile(
    rf"^[\"'(]?[{_UPPER}][^\W\d_]+(?:[-'’][^\W\d_]+)?(?:\s+[{_UPPER}][^\W\d_]+)?"
    rf"\s+[{_UPPER}]{{1,3}}\.?(?:\s?[{_UPPER}]\.?){{0,2}}[,.]"
)

# Name-first onset whose title is quoted rather than dated — common in
# Indonesian, Turkish and legal styles ("Dandan Irawan, “Peningkatan Daya …”").
# The author-date patterns all require a year, which these entries simply do not
# print.
_QUOTED_TITLE_START = re.compile(
    rf"^[\"'(]?[{_UPPER}][^\W\d_]+(?:\s+[{_UPPER}][^\W\d_.]*){{0,5}},\s*[“\"«‘„]"
)

# CJK reference onset: names are set without spaces and separated by "・" or "、",
# then the year. There is no comma or Latin capital for the other patterns to
# catch, so match on script plus an early year instead.
_CJK_START = re.compile(
    r"^[々぀-ヿ㐀-䶿一-鿿]"
    r".{0,60}?(?:19|20)\d\d"
)


def _looks_like_ref_onset(text: str) -> bool:
    """True when *text* starts the way references start.

    Filters the two known noise families in the ``reference_content`` stream:
    prose statements (CRediT, funding, conflicts — no author-date/numbered
    onset) and page-break continuation fragments (start mid-reference,
    typically lowercase or venue text).

    The patterns beyond the four trained-feature ones are local to this module
    on purpose: ``_AUTHOR_DATE_START``/``_NUMBERED``/``_STARTS_YEAR`` are inputs
    to the geometry GBM and must stay byte-for-byte stable, while anchor recall
    here is free to improve.
    """
    return bool(
        _looks_like_author_date_start(text)
        or _NUMBERED.match(text)
        or _STARTS_YEAR.match(text)
        # corporate authors ("World Health Organization. (2018).") carry no
        # comma for the author-date regex but show the year paren early
        or _EARLY_YEAR_PAREN.search(text)
        or _NUMBERED_LOOSE.match(text)
        or _VANCOUVER_START.match(text)
        or _QUOTED_TITLE_START.match(text)
        or _CJK_START.match(text)
    )


def region_anchor_texts(region_summaries: Iterable[RegionSummary]) -> list[str]:
    """Ordered reference-onset anchor prefixes from the region stream."""
    anchors: list[str] = []
    for rs in region_summaries:
        if (rs.label or "") != _REGION_LABEL:
            continue
        content = collapse_ws(rs.content or "").strip()
        if not content or not _looks_like_ref_onset(content):
            continue
        anchors.append(content[:_ANCHOR_LEN])
    return anchors


def _aligned_spans(
    ref_text: str, region_summaries: Iterable[RegionSummary]
) -> list[tuple[int, int]] | None:
    """Shared anchor→align→threshold decline sequence, or ``None`` to decline.

    Declines when fewer than ``_MIN_ANCHORS`` usable anchors exist or fewer
    than ``_MIN_ALIGN_FRACTION`` of them locate in *ref_text*. Used by
    :func:`segment_by_region_anchors`, :func:`region_chunks`, and the
    region-vs-geom eval script (``evaluation/seg_ab_region_vs_geom.py``) so
    all three share one decline definition.
    """
    anchors = region_anchor_texts(region_summaries)
    if len(anchors) < _MIN_ANCHORS:
        return None
    starts = find_anchor_starts(ref_text, anchors)
    if len(starts) < _MIN_ALIGN_FRACTION * len(anchors):
        return None
    spans = starts_to_spans(ref_text, starts)
    return spans or None


def segment_by_region_anchors(
    ref_text: str, region_summaries: Iterable[RegionSummary]
) -> list[str] | None:
    """Segment *ref_text* by layout-region onsets, or ``None`` to decline.

    Declines (returns ``None``) when fewer than ``_MIN_ANCHORS`` usable
    anchors exist or fewer than ``_MIN_ALIGN_FRACTION`` of them locate in
    *ref_text* — the caller cascades to the next tier.
    """
    spans = _aligned_spans(ref_text, region_summaries)
    if spans is None:
        return None
    segments = [ref_text[s:e] for s, e in spans]
    return segments or None


# Parse-sized chunk target: ~12-18 refs per chunk, comfortably inside one
# LLM parse call and far below LLM max_input_chars. Chunks always start at a
# region-anchored reference onset, so under-segmentation inside a chunk is
# recoverable by the chunk-tolerant parse prompt.
_CHUNK_TARGET_CHARS = 6000


def region_chunks(
    ref_text: str,
    region_summaries: Iterable[RegionSummary],
    *,
    target_chars: int = _CHUNK_TARGET_CHARS,
) -> list[str] | None:
    """Group region-anchored spans into parse-sized chunks, or ``None`` to decline.

    Same decline conditions as :func:`segment_by_region_anchors`; a single
    span longer than *target_chars* still becomes its own chunk.
    """
    spans = _aligned_spans(ref_text, region_summaries)
    if spans is None:
        return None
    chunks: list[str] = []
    cur_start, cur_end = spans[0]
    for s, e in spans[1:]:
        if e - cur_start <= target_chars:
            cur_end = e
        else:
            chunks.append(ref_text[cur_start:cur_end])
            cur_start, cur_end = s, e
    chunks.append(ref_text[cur_start:cur_end])
    return chunks or None
