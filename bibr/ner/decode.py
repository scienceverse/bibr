"""Pure BIO-span decoder for the ref parser.

Turns a parallel ``(tags, offsets)`` sequence over an original reference string
into ``{field: verbatim_text}``, where ``field`` is the BIO field name (the part
after the ``B-``/``I-`` prefix, e.g. ``AUTHOR``, ``TITLE``, ``YEAR``). Disjoint
spans for the same field are joined with a single space. No torch / model deps,
so it is cheap to unit-test.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# A whole page range the tagger dropped into the PAGE_RANGE_END slot without
# tagging its start ("20-26", "41–49", "S13-S20"). Both sides must carry a
# digit; a leading/trailing letter is allowed for supplement/article labels.
_PAGE_SPAN_RE = re.compile(
    r"^([A-Za-z]{0,2}\d{1,6}[A-Za-z]?)\s*[-–—]{1,2}\s*([A-Za-z]{0,2}\d{1,6}[A-Za-z]?)$"
)

# BIO field name -> PaperReference field name. Every field type in the tag
# scheme has a target: the five that used to be dropped here (ARXIV, PMID,
# SERIES, ACCESS_DATE, NOTE) were trained -- PMID reaches 0.947 F1 on the
# JATS-supervised corpus -- and then thrown away at decode. YEAR is the one
# exception, handled below because it is the only field converted to an int.
# ``test_every_tagged_field_reaches_paper_reference`` guards the invariant.
_FIELD_TO_PAPER_REF: dict[str, str] = {
    "TITLE": "title",
    "AUTHOR": "authors",
    "CONTAINER": "container",
    "VOLUME": "volume",
    "ISSUE": "issue",
    "DOI": "doi",
    "PAGES": "first_page",
    "PAGE_RANGE_START": "first_page",
    "PAGE_RANGE_END": "last_page",
    "PUBLISHER": "publisher",
    "URL": "url",
    "EDITION": "edition",
    "EDITOR": "editors",
    "ARXIV": "arxiv",
    "PMID": "pmid",
    "SERIES": "series",
    "ACCESS_DATE": "access_date",
    "NOTE": "note",
}


def decode_bio_spans(
    tags: list[str],
    offsets: list[tuple[int, int]],
    text: str,
) -> dict[str, str]:
    """Decode BIO ``tags`` (aligned to ``offsets``) into ``{field: text}``."""
    spans: dict[str, list[tuple[int, int]]] = {}
    cur_field: str | None = None
    cur_start: int | None = None

    for tag, (ts, _te) in zip(tags, offsets, strict=False):
        if tag.startswith("B-"):
            if cur_field is not None and cur_start is not None:
                spans.setdefault(cur_field, []).append((cur_start, ts))
            cur_field = tag[2:]
            cur_start = ts
        elif tag.startswith("I-"):
            field = tag[2:]
            if cur_field != field:
                if cur_field is not None and cur_start is not None:
                    spans.setdefault(cur_field, []).append((cur_start, ts))
                cur_field = field
                cur_start = ts
        else:  # "O" (or anything non-B/I)
            if cur_field is not None and cur_start is not None:
                spans.setdefault(cur_field, []).append((cur_start, ts))
            cur_field = None
            cur_start = None

    if cur_field is not None and cur_start is not None:
        last_end = offsets[-1][1] if offsets else len(text)
        spans.setdefault(cur_field, []).append((cur_start, last_end))

    out: dict[str, str] = {}
    for field, span_list in spans.items():
        parts = [text[s:e].strip() for s, e in span_list if e > s]
        parts = [p for p in parts if p]
        if parts:
            out[field] = " ".join(parts)
    return out


def _split_page_span(out: dict[str, str | int]) -> None:
    """Repair a dashed span left whole in the ``last_page`` slot, in place.

    The tagger routinely emits ``PAGE_RANGE_END`` for the entire "20-26" span
    and leaves the start token ``O``, which exports a ``last_page`` with a null
    ``first_page``. Splitting is only applied when it cannot destroy a value:
    ``first_page`` must be absent, or already equal to the span's start.
    """
    last = out.get("last_page")
    if not isinstance(last, str):
        return
    match = _PAGE_SPAN_RE.match(last.strip())
    if match is None:
        return
    start, end = match.group(1), match.group(2)
    first = out.get("first_page")
    if first is None:
        out["first_page"] = start
    elif not (isinstance(first, str) and first.strip() == start):
        # Independently tagged, disagreeing start — don't guess which is right.
        logger.debug("last_page span %r disagrees with first_page %r; left as-is", last, first)
        return
    out["last_page"] = end


def map_fields_to_paper_ref(raw: dict[str, str]) -> dict[str, str | int]:
    """Map BIO field names (``AUTHOR``, ``TITLE``, ``YEAR``, ...) to the flat
    dict of ``PaperReference`` field names expected by ``RefParser.parse``.

    ``YEAR`` is converted to an int from its first four digits; a non-numeric
    year (e.g. "in press", "n.d.") is dropped. Unmapped fields are skipped.
    Both ``PAGES`` and ``PAGE_RANGE_START`` map to ``first_page``; when both are
    present, the more specific ``PAGE_RANGE_START`` wins. A dashed span left
    whole in the last-page slot is split into ``first_page`` / ``last_page``.
    """
    out: dict[str, str | int] = {}
    for field, value in raw.items():
        if field == "PAGES" and "PAGE_RANGE_START" in raw:
            logger.debug(
                "Both PAGES and PAGE_RANGE_START tagged; keeping PAGE_RANGE_START for first_page"
            )
            continue
        if field == "YEAR":
            digits = "".join(c for c in value if c.isdigit())
            if digits:
                try:
                    out["year"] = int(digits[:4])
                except ValueError:
                    pass
            continue
        paper_field = _FIELD_TO_PAPER_REF.get(field)
        if paper_field is not None:
            out[paper_field] = value
    _split_page_span(out)
    return out
