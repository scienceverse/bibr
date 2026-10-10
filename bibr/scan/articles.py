"""Article boundaries on scanned pages.

A scanned journal issue is cut into papers at page boundaries, so the first
page of a paper often opens with the end of the previous article (its last
paragraphs, its references, a float) and the last page often closes with the
start of the next one (its title, byline and first paragraphs). Without a
split those regions are parsed as the paper's own, and the neighbours'
references leak into the reference list (issue #146).

The split reads only the layout labels and the OCR text, in reading order:

1. The paper's own title is the first ``doc_title`` on the first two scanned
   pages that have text.
2. A later ``doc_title`` starts the next article only when the paper's own
   references were already printed before it (a ``reference`` region or a
   references heading), it sits on a scanned page, and it is not itself a
   references, appendix or supplement heading. Everything from it on is the
   next article's.
3. What precedes the paper's title is the previous article's tail when it
   holds that article's references; then its body regions are dropped. On the
   title's own page, floats and captions printed above the title are dropped
   whether or not it does, as they belong to the previous article.

Page furniture (headers, footers, page numbers, margin text) is always kept:
the downstream furniture logic already handles it and the running head can
carry the journal name and DOI.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from bibr.ocr.types import OcrRegionResult
from bibr.scan.page_kind import PageKind

# Never dropped: page furniture.
_FURNITURE_LABELS = frozenset(
    {"header", "footer", "number", "aside_text", "header_image", "footer_image", "seal"}
)
# Floats and captions, dropped above the paper's title on its own page.
_FLOAT_LABELS = frozenset({"image", "chart", "table", "figure_title", "algorithm"})
_REFERENCE_LABELS = frozenset({"reference", "reference_content"})
_HEADING_LABELS = frozenset({"paragraph_title", "doc_title"})

# How many leading pages with text the paper's own title is looked for on: a
# repository cover sheet or the previous article's tail can fill the first.
_TITLE_SEARCH_PAGES = 2
# A doc_title shorter than this (letters and digits) is not trusted as the
# start of another article: OCR fragments and section numbers.
_MIN_TITLE_CHARS = 8
# The tail before the title is dropped only when it holds at least this many
# reference regions, or a references heading.
_MIN_TAIL_REFERENCE_REGIONS = 1
# A float counts as above the title when its bottom edge is no lower than the
# title's top edge plus this slack (0..1000 page units).
_ABOVE_SLACK = 5.0


def _fold(text: str) -> str:
    """Lowercase letters without accents, with runs of other characters as one space."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    letters = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[\W\d_]+", " ", letters).strip()


# Reference-list headings, folded (see _fold). Non-Latin entries keep their script.
_REFERENCE_HEADINGS = frozenset(
    {
        "references",
        "reference",
        "references and notes",
        "notes and references",
        "literature cited",
        "literature",
        "works cited",
        "cited literature",
        "bibliography",
        "bibliographie",
        "bibliografia",
        "referencias",
        "referencias bibliograficas",
        "references bibliographiques",
        "referenzen",
        "literatur",
        "literaturverzeichnis",
        "riferimenti bibliografici",
        "литература",
        "список литературы",
        "参考文献",
    }
)
# Headings that continue the paper after its references: not a new article.
_CONTINUATION_RE = re.compile(
    r"^(?:appendi(?:x|ces)|annex|supplementa(?:ry|l)|supporting information|"
    r"acknowledge?ments?|notes?|figure|fig|table|errat(?:um|a)|corrigend(?:um|a)|"
    r"author information|about the authors?)\b"
)


def is_reference_heading(text: str) -> bool:
    """Whether a heading reads as a reference-list heading in one of the known languages."""
    return _fold(text) in _REFERENCE_HEADINGS


def _label(region: OcrRegionResult) -> str:
    return region.native_label or region.label


def _is_reference_marker(region: OcrRegionResult) -> bool:
    label = _label(region)
    if label in _REFERENCE_LABELS:
        return bool(region.content.strip())
    return label in _HEADING_LABELS and is_reference_heading(region.content)


def _title_chars(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _can_start_article(region: OcrRegionResult) -> bool:
    if _label(region) != "doc_title":
        return False
    text = region.content.strip()
    if _title_chars(text) < _MIN_TITLE_CHARS or is_reference_heading(text):
        return False
    return not _CONTINUATION_RE.match(_fold(text))


def _above(region: OcrRegionResult, title: OcrRegionResult) -> bool:
    if not region.bbox_2d or not title.bbox_2d:
        return False
    return region.bbox_2d[3] <= title.bbox_2d[1] + _ABOVE_SLACK


@dataclass(frozen=True)
class DroppedRegion:
    page: int
    index: int
    label: str
    # previous_article | next_article | float_above_title
    reason: str


@dataclass
class ArticleSplit:
    """The pages with the neighbouring articles' regions removed, and what was removed."""

    pages: list[list[OcrRegionResult]]
    dropped: list[DroppedRegion] = field(default_factory=list)
    # Absolute page of the paper's own title, None when none was found.
    title_page: int | None = None
    # (page, index) of the doc_title that starts the next article, if any.
    next_article_at: tuple[int, int] | None = None

    @property
    def changed(self) -> bool:
        return bool(self.dropped)


def split_articles(
    pages: Sequence[Sequence[OcrRegionResult]],
    kinds: Mapping[int, PageKind],
) -> ArticleSplit:
    """Drop the regions of neighbouring articles from a paper's scanned pages.

    ``pages`` is indexed by absolute page (leading pages outside the
    processed range are empty) and ``kinds`` holds each page's class. Pages
    that are not scans are never split, so a born-digital paper comes back
    unchanged. Kept regions are renumbered so each region's ``index`` is its
    position on its page again.
    """
    unchanged = ArticleSplit(pages=[list(page) for page in pages])
    if not any(kind == PageKind.SCAN for kind in kinds.values()):
        return unchanged

    order = [
        (page_idx, pos, region)
        for page_idx, page in enumerate(pages)
        for pos, region in enumerate(page)
    ]

    # 1. The paper's own title.
    # Only scanned pages are searched: a born-digital cover sheet in front of
    # a scan repeats the title, and starting there would read the previous
    # article's tail on the first scanned page as part of the paper.
    text_pages = [
        i
        for i, page in enumerate(pages)
        if kinds.get(i) == PageKind.SCAN and any(r.content.strip() for r in page)
    ]
    search = set(text_pages[:_TITLE_SEARCH_PAGES])
    start = next(
        (
            k
            for k, (page_idx, _, region) in enumerate(order)
            if page_idx in search and _can_start_article(region)
        ),
        None,
    )
    if start is None:
        return unchanged
    title_page, _, title = order[start]

    drop: dict[tuple[int, int], str] = {}

    # 2. The next article.
    seen_references = False
    end = len(order)
    for k in range(start + 1, len(order)):
        page_idx, pos, region = order[k]
        if seen_references and kinds.get(page_idx) == PageKind.SCAN and _can_start_article(region):
            end = k
            break
        if _is_reference_marker(region):
            seen_references = True
    for page_idx, pos, region in order[end:]:
        if _label(region) not in _FURNITURE_LABELS:
            drop[(page_idx, pos)] = "next_article"

    # 3. The previous article's tail.
    prefix = order[:start]
    tail_references = sum(1 for *_, region in prefix if _is_reference_marker(region))
    tail_is_foreign = tail_references >= _MIN_TAIL_REFERENCE_REGIONS and any(
        kinds.get(page_idx) == PageKind.SCAN for page_idx, _, _ in prefix
    )
    for page_idx, pos, region in prefix:
        label = _label(region)
        if label in _FURNITURE_LABELS:
            continue
        if tail_is_foreign and label != "abstract" and kinds.get(page_idx) == PageKind.SCAN:
            drop[(page_idx, pos)] = "previous_article"
        elif (
            page_idx == title_page
            and kinds.get(page_idx) == PageKind.SCAN
            and label in _FLOAT_LABELS
            and _above(region, title)
        ):
            drop[(page_idx, pos)] = "float_above_title"

    if not drop:
        return ArticleSplit(pages=unchanged.pages, title_page=title_page)

    kept_pages: list[list[OcrRegionResult]] = []
    dropped: list[DroppedRegion] = []
    for page_idx, page in enumerate(pages):
        kept: list[OcrRegionResult] = []
        for pos, region in enumerate(page):
            reason = drop.get((page_idx, pos))
            if reason is not None:
                dropped.append(DroppedRegion(page_idx, region.index, _label(region), reason))
                continue
            kept.append(region if region.index == len(kept) else _renumber(region, len(kept)))
        kept_pages.append(kept)
    next_at = (order[end][0], order[end][2].index) if end < len(order) else None
    return ArticleSplit(
        pages=kept_pages, dropped=dropped, title_page=title_page, next_article_at=next_at
    )


def _renumber(region: OcrRegionResult, index: int) -> OcrRegionResult:
    return dataclasses.replace(region, index=index)


def describe(split: ArticleSplit) -> str:
    """One sentence naming what the split removed, by reason and page (1-based)."""
    by_reason: dict[str, set[int]] = {}
    for item in split.dropped:
        by_reason.setdefault(item.reason, set()).add(item.page + 1)
    names = {
        "previous_article": "the previous article's tail",
        "next_article": "the next article",
        "float_above_title": "floats above the title",
    }
    parts = [
        f"{names[reason]} (pages {', '.join(str(p) for p in sorted(pages))})"
        for reason, pages in sorted(by_reason.items())
    ]
    return (
        f"Dropped {len(split.dropped)} regions of neighbouring articles from scanned pages: "
        + "; ".join(parts)
    )
