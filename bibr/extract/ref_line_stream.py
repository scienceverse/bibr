"""Reference segmentation over one cleaned line stream with pooled boundary evidence.

The cascade in :mod:`bibr.extract.ref_extractor` segments the reference rows
(one row per layout region, printed line breaks flattened) by tiers that are
alternatives: the geometry model's line labels are snapped onto the rows with
exact text probes under one global gate, and a declined tier's evidence is
thrown away. This module works on the printed lines instead:

1. **One stream.** The located section's layout regions, in reading order, are
   read back as lines: text-layer lines (with their geometry) inside each
   region's box, or, for a region without a usable text layer (OCR, scans),
   the region text split at its line breaks with the region box as
   approximate geometry.
2. **Furniture removal** on the stream: lines inside layout header, footer and
   page-number boxes; page-edge lines whose digit-masked text repeats at the
   edge of two or more pages; standalone page numbers and roman numerals at a
   page edge; and the end of the list (acknowledgements, funding, appendix and
   the like).
3. **Pooled start votes.** Every line collects evidence that it opens an
   entry: the geometry model's per-line probability where the line has
   geometry (even when the model is unconfident overall), author/year and
   Vancouver onsets, dash-led "same author" openings, the first line of a
   layout region, hanging indent, a vertical gap, and the previous line ending
   in a DOI, a URL or a DOI link annotation. A printed numbering sequence
   ("[n]", "n.", "(n)", roman numerals) that increments by one overrides the
   votes.
4. **Per-entry repair.** A fragment that opens in lower case and carries no
   year or DOI rejoins the previous entry; an entry holding two DOIs (printed
   or linked) is split after the first.

Link annotations whose target is a DOI are mapped onto the lines beneath them,
so an entry whose text prints no DOI can take the one its "[CrossRef]" label
links to.

The caller (``ReferenceExtractor``) compares the result with the cascade's
through :func:`segmentation_quality` and keeps the cascade's output unless it
declined, fell back or under-yielded, or the stream is clearly better.
"""

from __future__ import annotations

import bisect
import itertools
import logging
import re
import statistics
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bibr.extract.merge_split import find_interior_onsets
from bibr.extract.region_seg import (
    _CJK_START,
    _QUOTED_TITLE_START,
    _VANCOUVER_START,
    _looks_like_ref_onset,
)
from bibr.input.consolidate_text import fix_ocr_artifacts
from bibr.ocr.pdf_links import doi_from_uri
from bibr.utils.text import DOI_BODY, YEARISH_RE, collapse_ws, normalize_doi

if TYPE_CHECKING:
    import pandas as pd

    from bibr.paper_contents import PaperContents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stream building
# ---------------------------------------------------------------------------

# Layout-space tolerance (0..1000) around a region box when assigning a
# text-layer line to it by its centre: region boxes are tight around the ink.
_REGION_BOX_TOLERANCE = 3.0

# Labels whose boxes hold page furniture, never reference text.
_FURNITURE_LABELS = frozenset(
    {"header", "footer", "number", "header_image", "footer_image", "page_number"}
)
# Page lines at the top or bottom edge that may be furniture: running title
# paired with a page number.
_EDGE_LINES = 2
# Layout-space margin bands for regions without text-layer lines, matching the
# running-header bands in ``bibr.structure.pdf_parser``.
_TOP_BAND = 100.0
_BOTTOM_BAND = 900.0

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
# A lone page number, arabic or roman, optionally as "page 12", "- 12 -" or
# "12 of 30".
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:(?:page|p\.|pp\.|seite|página|pagina|str\.)\s*)?[-–—]?\s*(\d{1,4}|[ivxlc]{1,7})\s*"
    r"[-–—]?\s*(?:(?:/|of|von|de)\s*\d{1,4})?\s*$",
    re.IGNORECASE,
)
# Running heads are mostly words. A line holding a DOI, URL, arXiv id or ISBN
# is never furniture, however alike two of them look once their digits are
# masked ("https://doi.org/#.#/#" on two pages of one journal).
_FURNITURE_MIN_LETTERS = 6
_LOCATOR_TEXT = re.compile(r"10\.\d{4,9}/|https?://|www\.|\bdoi\s*:|\barxiv\b|\bisbn\b", re.I)

# Headings that end a reference list. Whole-heading shaped: the cue opens a
# short line and ends it, or is followed by a colon ("Funding: ...").
_END_OF_LIST_RE = re.compile(
    r"^\s*(?:[A-Z]?\d{0,2}[.)]?\s*)?"
    r"(?:acknowledg(?:e)?ments?|funding(?:\s+(?:information|sources?|statement))?"
    r"|appendix(?:\s+[A-Z0-9]{1,3})?|appendices|data\s+(?:availability|accessibility)"
    r"(?:\s+statement)?|supplementary\s+(?:material|materials|information|data)"
    r"|supporting\s+information|author\s+contributions?|conflicts?\s+of\s+interests?"
    r"|competing\s+interests?|declaration\s+of\s+(?:competing\s+)?interests?"
    r"|ethics\s+statement|notes\s+on\s+contributors?|about\s+the\s+authors?"
    r"|author\s+biograph(?:y|ies)|biograph(?:y|ies)|abbreviations"
    r"|agradecimientos|agradecimentos|danksagung|remerciements|financiamento|financiación)"
    r"\s*(?::|\.?\s*$)",
    re.IGNORECASE,
)
_END_OF_LIST_MAX_CHARS = 60


@dataclass
class StreamLine:
    """One printed line of the reference section."""

    text: str
    page: int
    bbox: tuple[float, float, float, float] | None
    region: int
    region_label: str
    region_first: bool
    # PDF-point record for the geometry features (text-layer lines only).
    geometry: dict[str, Any] | None = None
    link_dois: list[str] = field(default_factory=list)
    edge: bool = False
    # Text column on the page, from the left edges of the section's boxes.
    column: int = 0

    @property
    def from_text_layer(self) -> bool:
        return self.geometry is not None


@dataclass
class LineStream:
    """The located reference section as ordered, cleaned lines."""

    lines: list[StreamLine]
    # Lines dropped as page furniture, and the text cut at the end of the list.
    furniture_removed: int = 0
    tail_text: str = ""
    text_layer_regions: int = 0
    region_text_regions: int = 0
    # Alphanumeric projection of the section's rows (furniture and any text
    # after the list included), the text both the cascade's and the stream's
    # entries are measured against.
    section_key: str = ""
    # The rows of a references section split in two at a page break were
    # added in front of the located ones.
    split_section: bool = False
    # A page of the section is rotated. Its lines' boxes are turned into the
    # layout frame, but their point geometry (indent, gaps, the geometry
    # model's features) is not, so the stream's votes there are unreliable.
    rotated: bool = False


@dataclass(frozen=True)
class StreamSegmentation:
    """Entries found on a line stream, with the text they were built from."""

    entries: tuple[str, ...]
    # Stream text the spans index: the entries joined by newlines, then the
    # text cut at the end of the list.
    text: str
    spans: tuple[tuple[int, int], ...]
    # Entries opened by a verified printed numbering sequence; the junk filter
    # never drops them.
    numbered: tuple[bool, ...]
    # DOI from a link annotation, per entry, when its text prints none.
    link_dois: tuple[str | None, ...]
    numbered_style: bool
    reason_flags: tuple[str, ...]
    # ``LineStream.section_key``: what the cascade's and the stream's entries
    # are both measured against.
    section_key: str = ""


def _alnum(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _furniture_key(text: str) -> str | None:
    """Digit-masked form under which a running head repeats, or None.

    None for a line that cannot be a running head: one holding a locator
    (``_LOCATOR_TEXT``) or fewer than six letters.
    """
    if _LOCATOR_TEXT.search(text):
        return None
    key = _DIGITS.sub("#", _WS.sub(" ", text.casefold()).strip())
    if sum(ch.isalpha() for ch in key) < _FURNITURE_MIN_LETTERS:
        return None
    return key


def _page_number_value(text: str) -> int | None:
    match = _PAGE_NUMBER_LINE.match(text)
    if match is None:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else _roman_value(token)


def _confirmed_page_numbers(candidates: list[tuple[int, int, int]]) -> set[int]:
    """Candidates ``(key, page, value)`` whose value runs with the page number.

    A lone number is a page number when another page carries one at the same
    offset from its page index; a year or a volume that happens to stand alone
    on an edge line does not.
    """
    pages_by_offset: dict[int, set[int]] = defaultdict(set)
    for _key, page, value in candidates:
        pages_by_offset[value - page].add(page)
    return {key for key, page, value in candidates if len(pages_by_offset[value - page]) >= 2}


def _box(values: Sequence[float]) -> tuple[float, float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _center(bbox: Sequence[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _inside(point: tuple[float, float], bbox: Sequence[float], tolerance: float = 0.0) -> bool:
    x, y = point
    return (
        bbox[0] - tolerance <= x <= bbox[2] + tolerance
        and bbox[1] - tolerance <= y <= bbox[3] + tolerance
    )


@dataclass
class _SectionRegion:
    page: int
    bbox: tuple[float, float, float, float] | None
    label: str
    rejected_native: bool = False
    # Rows whose first source region this is: the region text read when the
    # region has no usable text-layer lines.
    rows: list[str] = field(default_factory=list)


def _section_regions(
    contents: PaperContents, ref_df: pd.DataFrame
) -> tuple[list[_SectionRegion], list[str]]:
    """Layout regions behind the located reference rows, in reading order, and the row texts.

    A paragraph carried over several regions gives each of its sentences the
    whole region list as provenance, so a row's text is lent only to its first
    region; the text-layer lines of every region are checked against the
    whole section instead.
    """
    sentences = {sentence.text_id: sentence for sentence in contents.sentences}
    summaries: dict[tuple[int, tuple[float, ...]], Any] = {}
    for summary in getattr(contents, "region_summaries", None) or []:
        if summary.bbox is not None:
            summaries.setdefault((summary.page, tuple(summary.bbox)), summary)
    regions: list[_SectionRegion] = []
    by_key: dict[tuple[int, tuple[float, float, float, float] | None, int], _SectionRegion] = {}
    texts: list[str] = []
    for row_position, (text_id, text) in enumerate(
        zip(ref_df["text_id"].tolist(), ref_df["text"].astype(str).tolist(), strict=True)
    ):
        texts.append(text)
        sentence = sentences.get(text_id)
        keys: list[tuple[int, tuple[float, float, float, float] | None, int]] = [
            (p.page_no, p.bbox, -1)
            for p in (sentence.provenance if sentence else [])
            if p.bbox is not None
        ]
        if not keys:
            # No box: the row is its own pseudo-region, read as region text.
            keys = [(getattr(sentence, "page_number", None) or 0, None, row_position)]
        for index, key in enumerate(keys):
            region = by_key.get(key)
            if region is None:
                page, bbox, _ = key
                summary = summaries.get((page, tuple(bbox))) if bbox is not None else None
                label = (sentence.region_meta or {}).get("region_type") if sentence else None
                if summary is not None:
                    label = summary.label
                region = _SectionRegion(
                    page=page,
                    bbox=bbox,
                    label=str(label or ""),
                    rejected_native=bool(
                        summary is not None
                        and getattr(summary, "native_text_rejection_reason", None)
                    ),
                )
                by_key[key] = region
                regions.append(region)
            if index == 0:
                region.rows.append(text)
    return regions, texts


def _page_lines_by_page(page_lines: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in page_lines:
        if line.get("bbox") and str(line.get("text") or "").strip():
            by_page[int(line["page"])].append(line)
    return by_page


def _edge_line_ids(by_page: dict[int, list[dict[str, Any]]]) -> set[int]:
    """Lines among the top or bottom two of their page, by position."""
    edges: set[int] = set()
    for lines in by_page.values():
        ordered = sorted(lines, key=lambda line: line["bbox"][1])
        for line in ordered[:_EDGE_LINES] + ordered[-_EDGE_LINES:]:
            edges.add(id(line))
    return edges


def _edge_furniture_ids(by_page: dict[int, list[dict[str, Any]]], edge_ids: set[int]) -> set[int]:
    """Edge lines that are page furniture.

    A running head: its furniture key recurs at an edge of two or more pages.
    A page number: a lone number whose offset from the page index recurs.
    Only lines at a page edge ever qualify, as in the geometry segmenter's own
    capture (``bibr.ocr.ref_geometry``).
    """
    pages_by_key: dict[str, set[int]] = defaultdict(set)
    keyed: list[tuple[int, str]] = []
    numbered: list[tuple[int, int, int]] = []
    for page, lines in by_page.items():
        for line in lines:
            if id(line) not in edge_ids:
                continue
            text = str(line["text"])
            value = _page_number_value(text)
            if value is not None:
                numbered.append((id(line), page, value))
                continue
            key = _furniture_key(text)
            if key is not None:
                pages_by_key[key].add(page)
                keyed.append((id(line), key))
    furniture = {line_id for line_id, key in keyed if len(pages_by_key[key]) >= 2}
    return furniture | _confirmed_page_numbers(numbered)


def _in_band(bbox: Sequence[float] | None) -> bool:
    return bbox is not None and (bbox[3] <= _TOP_BAND or bbox[1] >= _BOTTOM_BAND)


def _band_furniture(contents: PaperContents) -> tuple[set[str], set[int]]:
    """Running-head keys and page-number offsets from the margin-band regions.

    The same two tests as ``_edge_furniture_ids``, for regions read without
    text-layer lines (scans): a furniture key recurring in the bands of two
    pages, and the offsets at which lone numbers there run with the page.
    """
    pages_by_key: dict[str, set[int]] = defaultdict(set)
    numbered: list[tuple[int, int, int]] = []
    for summary in getattr(contents, "region_summaries", None) or []:
        if not _in_band(summary.bbox):
            continue
        text = collapse_ws(summary.content or "")
        value = _page_number_value(text)
        if value is not None:
            numbered.append((len(numbered), summary.page, value))
            continue
        key = _furniture_key(text)
        if key is not None:
            pages_by_key[key].add(summary.page)
    keys = {key for key, pages in pages_by_key.items() if len(pages) >= 2}
    confirmed = _confirmed_page_numbers(numbered)
    offsets = {value - page for index, page, value in numbered if index in confirmed}
    return keys, offsets


def _furniture_boxes(contents: PaperContents) -> dict[int, list[tuple[float, float, float, float]]]:
    boxes: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for summary in getattr(contents, "region_summaries", None) or []:
        if (summary.label or "") in _FURNITURE_LABELS and summary.bbox is not None:
            boxes[summary.page].append(summary.bbox)
    return boxes


def _region_text_lines(region: _SectionRegion) -> list[str]:
    lines: list[str] = []
    for row in region.rows:
        lines.extend(
            part for part in (collapse_ws(p) for p in re.split(r"\r\n|\r|\n", row)) if part
        )
    return lines


def _match_key(text: str) -> str:
    """Alphanumeric projection under NFKC, so ligatures read as their letters."""
    return _alnum(unicodedata.normalize("NFKC", text))


# A text-layer line shorter than this (alphanumerics) is kept on region
# membership alone: "[12]", "pp." or a DOI fragment cannot be located by text.
_MIN_MATCH_CHARS = 6
# Share of a region's text-layer characters that must be found in the
# section's rows for the region to be read from the text layer rather than
# from its row text (an OCR'd region, a garbled text layer).
_TEXT_LAYER_MIN_SHARE = 0.6
# Region text this long that the stream already holds is a second read of the
# same printed text (an entry box inside an aggregate box), not a new entry.
_MIN_REPEAT_CHARS = 30


# How far (layout units) a line must run past its region box before a number
# at that end is taken for a margin line number rather than a wrapped page or
# volume a slightly loose box cut off.
_MARGIN_NUMBER_OVERHANG = 15.0
_LEADING_NUMBER = re.compile(r"^\s*\d{1,4}\s+(?=\S)")
_TRAILING_NUMBER = re.compile(r"(?<=\S)\s+\d{1,4}\s*$")


def _trim_margin_numbers(
    text: str, line_bbox: Sequence[float], region_bbox: Sequence[float] | None
) -> str:
    """Drop manuscript line numbers the text layer joined onto a line.

    A line-numbered manuscript prints "582" in the margin on the same baseline
    as the reference text, so the text layer reads "582 Evans, J. R., ...";
    the layout region, and so the row text, starts after it. Only a number on
    the side where the line runs out of the region box is dropped.
    """
    if region_bbox is None:
        return text
    if line_bbox[0] < region_bbox[0] - _MARGIN_NUMBER_OVERHANG:
        text = _LEADING_NUMBER.sub("", text, count=1)
    if line_bbox[2] > region_bbox[2] + _MARGIN_NUMBER_OVERHANG:
        text = _TRAILING_NUMBER.sub("", text, count=1)
    return text


def _with_split_section(contents: PaperContents, ref_df: pd.DataFrame) -> pd.DataFrame | None:
    """The located rows plus the first half of a reference list split in two sections.

    A list under a heading the parser does not map to "References"
    ("Referencias", "Literatur") whose entries on the next page come as
    reference boxes gets a second, synthetic "References" section for them,
    and the English heading lookup locates only that one. When the section
    just before the located rows is a references section (by type or by a
    whole multilingual references heading) and exactly one of the two is
    synthetic, its contiguous rows are prepended. None when nothing applies.
    """
    import pandas as pd

    from bibr.ocr.ref_patterns import _REF_HEADER_RE
    from bibr.paper_contents import CanonicalSection

    if "section_id" not in ref_df.columns:
        return None
    sentences = contents.sentences_df
    if sentences.empty or "section_id" not in sentences.columns:
        return None
    sections = {section.section_id: section for section in contents.sections}
    positions = {text_id: i for i, text_id in enumerate(sentences["text_id"].tolist())}
    first = positions.get(ref_df["text_id"].iloc[0])
    if not first:
        return None
    section_ids = sentences["section_id"].tolist()
    located = sections.get(ref_df["section_id"].iloc[0])
    previous = sections.get(section_ids[first - 1])
    if located is None or previous is None or previous.section_id == located.section_id:
        return None
    if bool(previous.header_is_synthetic) == bool(located.header_is_synthetic):
        return None
    header = (previous.header or "").strip()
    if not (previous.section_type == CanonicalSection.REFERENCES or _REF_HEADER_RE.match(header)):
        return None
    start = first - 1
    while start > 0 and section_ids[start - 1] == previous.section_id:
        start -= 1
    head = sentences.iloc[start:first]
    head_texts = head["text"].astype(str).tolist()
    entry_like = sum(bool(_line_onset(text) or YEARISH_RE.search(text)) for text in head_texts)
    if not head_texts or 2 * entry_like < len(head_texts):
        return None
    return pd.concat([head[ref_df.columns.intersection(head.columns)], ref_df])


def build_line_stream(contents: PaperContents, ref_df: pd.DataFrame) -> LineStream | None:
    """Read the located reference section back as one stream of printed lines.

    Returns None when the rows carry no text.
    """
    if ref_df is None or ref_df.empty or "text_id" not in ref_df.columns:
        return None
    extended = _with_split_section(contents, ref_df)
    if extended is not None:
        ref_df = extended
    regions, row_texts = _section_regions(contents, ref_df)
    if not regions:
        return None
    section_key = _match_key(" ".join(row_texts))
    by_page = _page_lines_by_page(getattr(contents, "ref_page_lines", None) or [])
    edge_ids = _edge_line_ids(by_page)
    edge_furniture = _edge_furniture_ids(by_page, edge_ids)
    band_keys, band_page_offsets = _band_furniture(contents)
    furniture_boxes = _furniture_boxes(contents)

    # The section's own text is what the cascade's and the stream's entries are
    # both measured against, furniture and text after the list included: text
    # the stream leaves out costs it coverage.
    stream = LineStream(lines=[], section_key=section_key, split_section=extended is not None)
    assigned: set[int] = set()
    emitted_key = ""
    for region_index, region in enumerate(regions):
        candidates: list[dict[str, Any]] = []
        if region.bbox is not None and not region.rejected_native:
            for line in by_page.get(region.page, ()):
                if _inside(_center(line["bbox"]), region.bbox, _REGION_BOX_TOLERANCE):
                    candidates.append(line)
        # Keep the lines whose text the section's rows hold: a region shared
        # with the previous section contributes only its reference lines.
        kept: list[tuple[dict[str, Any], str]] = []
        total = matched = repeated = 0
        for line in candidates:
            text = _trim_margin_numbers(str(line["text"]), line["bbox"], region.bbox)
            key = _match_key(text)
            if len(key) < _MIN_MATCH_CHARS:
                if id(line) not in assigned:
                    kept.append((line, text))
                continue
            total += len(key)
            if key in section_key:
                matched += len(key)
                if id(line) in assigned:
                    repeated += len(key)
                else:
                    kept.append((line, text))
        if total and repeated >= _TEXT_LAYER_MIN_SHARE * total:
            # An entry box inside an aggregate box already read: its lines
            # are in the stream once.
            continue
        if kept and total and matched >= _TEXT_LAYER_MIN_SHARE * total:
            stream.text_layer_regions += 1
            first = True
            for line, text in kept:
                assigned.add(id(line))
                edge = id(line) in edge_ids
                center = _center(line["bbox"])
                if id(line) in edge_furniture or any(
                    _inside(center, box) for box in furniture_boxes.get(region.page, ())
                ):
                    stream.furniture_removed += 1
                    continue
                stream.lines.append(
                    StreamLine(
                        text=text,
                        page=region.page,
                        bbox=_box(line["bbox"]),
                        region=region_index,
                        region_label=region.label,
                        region_first=first,
                        geometry=line,
                        edge=edge,
                    )
                )
                emitted_key += _match_key(text)
                first = False
            continue
        if not region.rows:
            continue
        rows_key = _match_key(" ".join(region.rows))
        if len(rows_key) >= _MIN_REPEAT_CHARS and rows_key in emitted_key:
            # The same text read again through an overlapping region box.
            continue
        stream.region_text_regions += 1
        in_band = _in_band(region.bbox)
        if in_band and _furniture_key(collapse_ws(" ".join(region.rows))) in band_keys:
            stream.furniture_removed += len(region.rows)
            continue
        first = True
        for text in _region_text_lines(region):
            value = _page_number_value(text) if in_band else None
            if value is not None and value - region.page in band_page_offsets:
                stream.furniture_removed += 1
                continue
            stream.lines.append(
                StreamLine(
                    text=text,
                    page=region.page,
                    bbox=region.bbox,
                    region=region_index,
                    region_label=region.label,
                    region_first=first,
                    edge=in_band,
                )
            )
            emitted_key += _match_key(text)
            first = False
    if not stream.lines:
        return None
    stream.rotated = any((line.geometry or {}).get("rotation") for line in stream.lines)
    _assign_columns(stream, regions)
    _cut_end_of_list(stream)
    _attach_link_dois(stream, getattr(contents, "pdf_uri_links", None) or [])
    return stream


# Section boxes on one page whose left edges lie within this many layout units
# share a text column.
_COLUMN_TOLERANCE = 60.0


def _assign_columns(stream: LineStream, regions: list[_SectionRegion]) -> None:
    """Number the text columns of each page by the left edges of the section's boxes."""
    edges: dict[int, list[float]] = defaultdict(list)
    for region in regions:
        if region.bbox is not None:
            edges[region.page].append(region.bbox[0])
    columns: dict[tuple[int, float], int] = {}
    for page, lefts in edges.items():
        column, previous = 0, None
        for left in sorted(set(lefts)):
            if previous is not None and left - previous > _COLUMN_TOLERANCE:
                column += 1
            columns[(page, left)] = column
            previous = left
    for line in stream.lines:
        bbox = regions[line.region].bbox
        if bbox is not None:
            line.column = columns[(line.page, bbox[0])]


# The line before an end-of-list heading closes an entry: it ends on a
# period, a bracket, a digit or a locator, not mid-sentence ("reported in").
_CLOSES_ENTRY = re.compile(r"(?:[.)\]\d]|https?://\S+|10\.\d{4,9}/\S+)\s*$")


def _cut_end_of_list(stream: LineStream) -> None:
    """Stop the stream at a heading that ends the reference list."""
    starts_seen = 0
    for index, line in enumerate(stream.lines):
        text = collapse_ws(line.text)
        if (
            index >= 2
            and starts_seen >= 2
            and len(text) <= _END_OF_LIST_MAX_CHARS
            and _END_OF_LIST_RE.match(text)
            and not YEARISH_RE.search(text)
            and _CLOSES_ENTRY.search(collapse_ws(stream.lines[index - 1].text))
        ):
            tail = stream.lines[index:]
            stream.tail_text = collapse_ws(" ".join(t.text for t in tail))
            stream.lines = stream.lines[:index]
            return
        if _line_onset(text) or _parse_marker(text) is not None:
            starts_seen += 1


def _attach_link_dois(stream: LineStream, links: Sequence[dict[str, Any]]) -> None:
    """Hang each DOI link annotation on the text-layer line beneath it."""
    if not links:
        return
    by_page: dict[int, list[StreamLine]] = defaultdict(list)
    for line in stream.lines:
        if line.from_text_layer and line.bbox is not None:
            by_page[line.page].append(line)
    for link in links:
        doi = doi_from_uri(link.get("uri"))
        bbox = link.get("bbox")
        if doi is None or not bbox:
            continue
        center = _center(bbox)
        for line in by_page.get(int(link.get("page") or 0), ()):
            if _inside(center, line.bbox, 1.0):  # type: ignore[arg-type]
                if doi not in line.link_dois:
                    line.link_dois.append(doi)
                break


# ---------------------------------------------------------------------------
# Line evidence
# ---------------------------------------------------------------------------

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bracket", re.compile(r"^\[\s*(\d{1,4})\s*\]")),
    ("paren", re.compile(r"^\(\s*(\d{1,4})\s*\)")),
    ("dot", re.compile(r"^(\d{1,4})\s*[.)](?=\s|[^\d\s])")),
    ("space", re.compile(r"^(\d{1,4})\s+(?=[\"'“(\[]?[A-ZÀ-ÖØ-ÞЀ-Я])")),
    ("roman", re.compile(r"^([IVXLC]{1,7}|[ivxlc]{1,7})[.)]\s+(?=\S)")),
)
_LABEL_MARKER = re.compile(r"^\[\s*[A-Za-z][A-Za-z.+\-]{0,10}\d{0,4}[a-z]?\s*\]")
# A bullet glyph opening each entry of a bulleted list.
_BULLET = re.compile(r"^[•∎■▪●◦‣⁃□►▶◆◇○➢➤]\s*")


def _roman_value(token: str) -> int | None:
    values = [_ROMAN_VALUES.get(ch) for ch in token.lower()]
    if not values or any(v is None for v in values):
        return None
    total = 0
    for i, value in enumerate(values):
        assert value is not None  # noqa: S101 — checked above
        following = values[i + 1] if i + 1 < len(values) else None
        total += -value if following is not None and following > value else value
    return total if 0 < total < 200 else None


def _parse_marker(text: str) -> tuple[str, int] | None:
    """The printed list marker opening *text*, as ``(kind, number)``."""
    for kind, pattern in _MARKER_PATTERNS:
        match = pattern.match(text)
        if match is None:
            continue
        if kind == "roman":
            value = _roman_value(match.group(1))
            if value is None:
                continue
            return kind, value
        return kind, int(match.group(1))
    return None


def _strip_marker(text: str) -> str:
    for _, pattern in _MARKER_PATTERNS:
        match = pattern.match(text)
        if match is not None:
            return text[match.end() :].lstrip()
    match = _LABEL_MARKER.match(text) or _BULLET.match(text)
    if match is not None:
        return text[match.end() :].lstrip()
    return text


def _list_mark(text: str) -> str | None:
    """The unnumbered list mark opening *text*: a bullet glyph or a bracketed label."""
    match = _BULLET.match(text)
    if match is not None:
        return "bullet:" + text[0]
    if _LABEL_MARKER.match(text) and _parse_marker(text) is None:
        return "label"
    return None


# "Same author as above": a dash run or underscores opening the line, then a
# year, a period/comma or a capital (coauthors, the title).
_DASH_START = re.compile(
    r"^(?:(?:[—―⸺⸻]+|_{2,})\s*[.,:]?\s*|[–-]{1,6}(?:\s*[.,:]\s*|\s+|(?=[(\[]?\d{4})))"
    r"(?=[(\[]?\d{4}|[A-ZÀ-Þ]|$|and\b|&)"
)
# All-caps family name opening an ABNT/ISO 690 entry, then an initial or a
# capitalised word: "SILVA, J." "BRASIL. Ministério". An all-caps journal
# continuing an entry ("PLOS ONE, 18(12)") is followed by digits instead.
_CAPS_SURNAME_START = re.compile(
    r"^[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’\-]{1,}(?:\s+[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’\-]+){0,3}[,.;]\s*[A-ZÀ-ÖØ-Þ]"
)
_NAME_WORD = r"[A-ZÀ-ÖØ-ÞĀ-Ž][^\W\d_]*(?:[-'’][^\W\d_]+)*"
_PARTICLE = r"(?:van|von|de|der|den|del|della|di|da|do|dos|das|du|la|le|ten|ter|op|zu|af|al|el)"
# "Family, I." / "Family, IJ," / "Family-Name, J.-L." / "van der Family, I.":
# a family name of up to three words, a comma, then initials. A continuation
# line opening on a journal or place and a comma is followed by digits, a
# colon or lower case instead ("Psychology, 25(3)", "Cambridge, MA:").
_AUTHOR_INITIALS_LINE = re.compile(
    rf"^[\"'“(]?(?:{_PARTICLE}\s+){{0,2}}{_NAME_WORD}(?:[\s-]+(?:{_PARTICLE}\s+){{0,2}}{_NAME_WORD}){{0,2}}"
    r",\s*(?:[A-ZÀ-ÖØ-Þ]\.|[A-ZÀ-ÖØ-Þ]{1,3}(?=[\s,.;&]|$)|[A-ZÀ-ÖØ-Þ]-[A-ZÀ-ÖØ-Þ]\.)"
)
# "Family, Given" needs a date on the line or an author list running on
# ("Smith, John, and"), since "Oxford, England: Blackwell" has the same shape.
_AUTHOR_GIVEN_LINE = re.compile(
    rf"^[\"'“(]?(?:{_PARTICLE}\s+){{0,2}}{_NAME_WORD}(?:[\s-]+{_NAME_WORD}){{0,2}},\s*[A-ZÀ-ÖØ-Þ][a-zß-ÿ]+"
)
# "Eisenberg A and Spinner-Havel J (eds)": family names with initials joined
# by a conjunction, no comma (OSCOLA, Vancouver variants).
_AUTHOR_AND_LINE = re.compile(
    rf"^{_NAME_WORD}\s+[A-ZÀ-ÖØ-Þ]{{1,3}}\.?(?:\s?[A-ZÀ-ÖØ-Þ]\.?)?\s+(?:and|&|y|e|und|et)\s+"
    rf"{_NAME_WORD}\s+[A-ZÀ-ÖØ-Þ]{{1,3}}\b"
)
# A quoted title opening a reference box: web pages and reports cited by title.
_QUOTED_LEAD = re.compile(r"^[‘“\"'][A-ZÀ-ÖØ-Þ]")
_AUTHOR_LIST_RUNS_ON = re.compile(r"(?:,|&|\band|\bund|\bet|\by|\be)\s*$")
# A corporate author or title-first entry dated in parentheses right after it:
# "World Health Organization. (2018)." "Statistics Canada (2020):".
_PAREN_DATE_LEAD = re.compile(
    r"^[\"'“]?[A-ZÀ-ÖØ-Þ][^\s()]*(?:\s+[^\s()]+){0,8}?\.?\s*\((?:1[6-9]|20)\d\d[a-z]?(?:,[^)]{0,20})?\)\s*[.:,]"
    r"|^[\"'“]?[A-ZÀ-ÖØ-Þ][^\s()]*(?:\s+[^\s()]+){0,8}?\.?\s*\(n\.\s?d\.\)",
)


# An APA date opening the line follows an author list on the line above.
_DATE_FIRST = re.compile(r"^\((?:1[6-9]|20)\d\d[a-z]?(?:,[^)]{0,20})?\)|^\(n\.\s?d\.\)", re.I)
_LOWER_START = re.compile(r"^[\"'“‘(\[]*[a-zß-öø-ÿ]")
_PUNCT_START = re.compile(r"^[,;:.)\]}]")
# The previous line runs on: a word broken at a hyphen, an author list or a
# locator cut mid-way.
_CONTINUES_NEXT = re.compile(r"(?:[A-Za-zß-ÿ]-|[,&]|\band|\bin|\bIn:?|\bet|\bpp\.?|\bvol\.?)\s*$")
_ENDS_WITH_LOCATOR = re.compile(
    r"(?:https?://\S+|\b" + DOI_BODY + r"\S+|doi:\s*\S+)\s*[.,;]?\s*$", re.I
)


def _line_onset(text: str) -> bool:
    """Whether a printed line opens like a reference entry (after its list marker).

    Stricter than the region-start patterns in ``bibr.extract.region_seg``,
    which judge the first line of a layout region: here every line of the
    section is judged, continuation lines included.
    """
    stripped = _strip_marker(text)
    if not stripped:
        return False
    if (
        _AUTHOR_INITIALS_LINE.match(stripped)
        or _CAPS_SURNAME_START.match(stripped)
        or _AUTHOR_AND_LINE.match(stripped)
    ):
        return True
    if _AUTHOR_GIVEN_LINE.match(stripped) and (
        YEARISH_RE.search(stripped) or _AUTHOR_LIST_RUNS_ON.search(stripped)
    ):
        return True
    return bool(
        _VANCOUVER_START.match(stripped)
        or _PAREN_DATE_LEAD.match(stripped)
        or _QUOTED_TITLE_START.match(stripped)
        or _CJK_START.match(stripped)
    )


def _region_onset(text: str) -> bool:
    """The looser region-start test, for the first line of a layout region."""
    stripped = _strip_marker(text)
    return bool(stripped) and bool(
        _line_onset(text) or _looks_like_ref_onset(stripped) or _QUOTED_LEAD.match(stripped)
    )


def _is_dash_start(text: str) -> bool:
    return bool(_DASH_START.match(text))


# ---------------------------------------------------------------------------
# Numbering sequence
# ---------------------------------------------------------------------------


# Words that follow a number inside an entry rather than an entry number:
# edition, supplement, part, volume and month words ("3. Aufl.", "10 Suppl",
# "5. Juni").
_NUMBER_CONTINUATION = re.compile(
    r"(?:aufl(?:age)?|ausg(?:abe)?|ed(?:n|ition)?|udg(?:ave)?|utg(?:ave)?|uppl(?:aga)?"
    r"|oppl(?:ag)?|painos|izd|изд|vyd|wyd|kiad(?:ás)?|bask[ıi]|suppl(?:ement)?|pt|part|teil"
    r"|bd|band|jg|jahrg(?:ang)?|hrsg|vol|no|nr"
    r"|jan(?:uary|uar)?|feb(?:ruary|ruar)?|mar(?:ch)?|märz|apr(?:il)?|may|mai|june?|juni"
    r"|july?|juli|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|okt(?:ober)?|nov(?:ember)?"
    r"|dec(?:ember)?|dez(?:ember)?)\b",
    re.IGNORECASE,
)
# How far (PDF points) a numbered line may sit off the list's marker column.
_MARKER_COLUMN_TOLERANCE = 4.0


def _marker_family(kind: str) -> str:
    return "roman" if kind == "roman" else "arabic"


def _chain_candidates(
    markers: list[tuple[int, str, int]], texts: list[str], offsets: list[float | None]
) -> list[tuple[int, str, int]]:
    """Marker lines that may claim a place in the numbering sequence.

    Not number 0, not a number followed by an edition, supplement, part or
    month word, and at the list's marker column where the line has geometry:
    within a few points of where most lines with that marker style start. A
    continuation line that happens to open on a number ("3. Aufl.", "X. Li,")
    sits at the continuation indent instead.
    """
    kept = [
        (line_index, kind, number)
        for line_index, kind, number in markers
        if number > 0 and not _NUMBER_CONTINUATION.match(_strip_marker(texts[line_index]))
    ]
    marker_lines = {line_index for line_index, _, _ in markers}
    others = [
        offset for i, offset in enumerate(offsets) if offset is not None and i not in marker_lines
    ]
    other_column = statistics.median(others) if others else None
    columns: dict[str, tuple[float, float]] = {}
    for kind in {kind for _, kind, _ in kept}:
        placed = [offset for i, k, _ in kept if k == kind and (offset := offsets[i]) is not None]
        if not placed:
            continue
        column = statistics.median(placed)
        # Right-aligned numbers ("9." over "10.", "VIII." over "IX.") start at
        # different points; half the distance to the continuation indent
        # still tells a numbered line from a continuation.
        spread = abs(other_column - column) / 2 if other_column is not None else 0.0
        columns[kind] = (column, max(_MARKER_COLUMN_TOLERANCE, spread))
    placed_ok: list[tuple[int, str, int]] = []
    for line_index, kind, number in kept:
        offset = offsets[line_index]
        if offset is not None and kind in columns:
            column, tolerance = columns[kind]
            if abs(offset - column) > tolerance:
                continue
        placed_ok.append((line_index, kind, number))
    return placed_ok


def _longest_run(items: list[tuple[int, int]]) -> list[int]:
    """Longest run of ``(line_index, number)`` counting up by one, one gap allowed.

    The run must open at 1-3: page numbers and years never do. A 1 after a run
    that reached 3 starts a second list in the same run (supplementary
    references numbered from 1 again).
    """
    chains: dict[int, list[int]] = {}
    best: list[int] = []
    best_last = 0
    for line_index, number in items:
        previous = chains.get(number - 1) or chains.get(number - 2)
        if previous is not None:
            candidate = [*previous, line_index]
        elif number == 1 and best_last >= 3:
            candidate = [*best, line_index]
        elif number <= 3:
            candidate = [line_index]
        else:
            continue
        if len(candidate) > len(chains.get(number, [])):
            chains[number] = candidate
            if len(candidate) > len(best):
                best, best_last = candidate, number
    return best


def _numbering_chain(candidates: list[tuple[int, str, int]]) -> list[int]:
    """Line indices of the printed numbering sequence(s), in stream order.

    ``candidates`` holds ``(line_index, kind, number)`` from
    ``_chain_candidates``. The sequence keeps one marker style ("[n]", "n.",
    "(n)", "n ", roman), so a roman initial never joins an arabic list or the
    reverse. A second list numbered from 1 again after the first
    (supplementary references) continues the sequence.
    """
    by_kind: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for line_index, kind, number in candidates:
        by_kind[kind].append((line_index, number))
    chain: list[int] = []
    for items in by_kind.values():
        run = _longest_run(items)
        if len(run) > len(chain):
            chain = run
    return chain


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

# Start-vote weights. A line opens an entry when its pooled score reaches
# ``_START_THRESHOLD``; the first line of the stream always does.
_W_ONSET = 0.45
_W_DASH = 0.7
_W_REGION_ENTRY = 0.45
_W_REGION_OTHER = 0.25
_W_GEOM = 1.0
# The model is decisive on styles it never saw (bulleted, OSCOLA, title-first
# web entries), so its evidence against a start is capped: the layout's own
# entry boxes and gaps can still outvote it.
_GEOM_MIN_VOTE = -0.35
_W_MARGIN = 0.3
_W_INDENTED = -0.4
_W_GAP = 0.35
_W_PREV_ENDS_ENTRY = 0.2
_W_PREV_CONTINUES = -0.4
_W_LOWER = -0.7
_W_LOWER_ENTRY_BOX = -0.4
_W_PUNCT = -0.6
_W_DATE_FIRST = -0.8
_START_THRESHOLD = 0.35
# Layout labels the detector gives one reference entry at a time.
_ENTRY_LABELS = frozenset({"reference_content", "reference"})
# Indent geometry, in PDF points.
_INDENT_MIN_PT = 4.0
_MARGIN_MAX_PT = 2.0
# Hanging indent needs at-margin lines opening entries more often than
# indented ones, by this onset-rate margin.
_STYLE_ONSET_MARGIN = 0.2


def _column_key(line: StreamLine) -> tuple[int, int]:
    return (line.page, line.column)


def _indent_offsets(lines: list[StreamLine]) -> list[float | None]:
    margins: dict[tuple[int, int], float] = {}
    for line in lines:
        if line.geometry is not None:
            key = _column_key(line)
            x0 = float(line.geometry["x0"])
            margins[key] = min(margins.get(key, x0), x0)
    offsets: list[float | None] = []
    for line in lines:
        if line.geometry is None:
            offsets.append(None)
        else:
            offsets.append(float(line.geometry["x0"]) - margins[_column_key(line)])
    return offsets


def _indent_style(offsets: list[float | None], onsets: list[bool]) -> str:
    """The list's indent style from the onset rate at each indent.

    "hanging" when lines at the margin open entries more often than indented
    ones, "first_line" for the reverse, else "flush".
    """
    margin_onsets = margin_total = indent_onsets = indent_total = 0
    for offset, onset in zip(offsets, onsets, strict=True):
        if offset is None:
            continue
        if offset <= _MARGIN_MAX_PT:
            margin_total += 1
            margin_onsets += onset
        elif offset >= _INDENT_MIN_PT:
            indent_total += 1
            indent_onsets += onset
    if margin_total < 2 or indent_total < 2:
        return "flush"
    margin_rate = margin_onsets / margin_total
    indent_rate = indent_onsets / indent_total
    if margin_rate >= indent_rate + _STYLE_ONSET_MARGIN:
        return "hanging"
    if indent_rate >= margin_rate + _STYLE_ONSET_MARGIN:
        return "first_line"
    return "flush"


def _gap_flags(lines: list[StreamLine]) -> list[bool]:
    """Whether a text-layer line sits below a wider gap than its column's spacing."""
    gaps: list[float | None] = [None] * len(lines)
    per_column: dict[tuple[int, int], list[float]] = defaultdict(list)
    for i in range(1, len(lines)):
        previous, line = lines[i - 1], lines[i]
        if previous.geometry is None or line.geometry is None:
            continue
        if previous.page != line.page or _column_key(previous) != _column_key(line):
            continue
        gap = float(previous.geometry["y_bottom"]) - float(line.geometry["y_top"])
        if gap < -2.0:
            continue
        gaps[i] = gap
        per_column[_column_key(line)].append(gap)
    typical = {key: statistics.median(values) for key, values in per_column.items() if values}
    flags: list[bool] = []
    for line, line_gap in zip(lines, gaps, strict=True):
        if line_gap is None:
            flags.append(False)
            continue
        base = typical.get(_column_key(line), line_gap)
        font = float(line.geometry["font_size"]) if line.geometry else 0.0
        flags.append(line_gap > max(1.6 * base, base + 0.4 * font, base + 2.0))
    return flags


def _start_scores(
    lines: list[StreamLine], probabilities: Sequence[float | None] | None
) -> tuple[list[float], list[bool]]:
    texts = [collapse_ws(line.text) for line in lines]
    onsets = [
        _region_onset(text) if line.region_first else _line_onset(text)
        for line, text in zip(lines, texts, strict=True)
    ]
    offsets = _indent_offsets(lines)
    style = _indent_style(offsets, onsets)
    gap_flags = _gap_flags(lines)
    scores: list[float] = []
    for i, (line, text) in enumerate(zip(lines, texts, strict=True)):
        score = 0.0
        stripped = _strip_marker(text)
        if onsets[i]:
            score += _W_ONSET
        if _is_dash_start(text):
            score += _W_DASH
        if line.region_first:
            score += _W_REGION_ENTRY if line.region_label in _ENTRY_LABELS else _W_REGION_OTHER
        probability = probabilities[i] if probabilities is not None else None
        if probability is not None:
            score += max(_GEOM_MIN_VOTE, _W_GEOM * (probability - 0.5))
        offset = offsets[i]
        if offset is not None and style != "flush":
            at_margin = offset <= _MARGIN_MAX_PT
            indented = offset >= _INDENT_MIN_PT
            if style == "hanging":
                score += _W_MARGIN if at_margin else (_W_INDENTED if indented else 0.0)
            else:
                score += _W_MARGIN if indented else (_W_INDENTED if at_margin else 0.0)
        if gap_flags[i]:
            score += _W_GAP
        if i > 0:
            previous = collapse_ws(lines[i - 1].text)
            if _ENDS_WITH_LOCATOR.search(previous) or lines[i - 1].link_dois:
                score += _W_PREV_ENDS_ENTRY
            elif _CONTINUES_NEXT.search(previous):
                score += _W_PREV_CONTINUES
        if _LOWER_START.match(stripped) and not _is_dash_start(text):
            # A reference box opening in lower case is still more often an
            # entry ("frendy rangkuti, ...", OCR's "yon Richter") than not.
            entry_box = line.region_first and line.region_label in _ENTRY_LABELS
            score += _W_LOWER_ENTRY_BOX if entry_box else _W_LOWER
        elif _PUNCT_START.match(text):
            score += _W_PUNCT
        elif _DATE_FIRST.match(text):
            score += _W_DATE_FIRST
        scores.append(score)
    return scores, onsets


def _entry_has_date_or_doi(text: str) -> bool:
    return bool(YEARISH_RE.search(text) or re.search(DOI_BODY, text))


def _printed_dois(text: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"10\.\d{4,9}/[^\s\"<>]+", text):
        doi = normalize_doi(match.group(0))
        if doi and doi.lower() not in (d.lower() for d in found):
            found.append(doi)
    return found


def _entry_text(lines: Sequence[StreamLine]) -> str:
    return collapse_ws(fix_ocr_artifacts("\r\n".join(line.text for line in lines)))


def segment_line_stream(
    stream: LineStream,
    probabilities: Sequence[float | None] | None = None,
) -> StreamSegmentation | None:
    """Pool the start votes on *stream* into entries, then repair them.

    ``probabilities`` holds the geometry model's per-line start probability,
    aligned with ``stream.lines`` (None for a line without geometry).
    """
    lines = stream.lines
    if not lines:
        return None
    tail_text = stream.tail_text
    texts = [collapse_ws(line.text) for line in lines]
    scores, onsets = _start_scores(lines, probabilities)
    voted = [0] + [i for i in range(1, len(lines)) if scores[i] >= _START_THRESHOLD]
    markers: list[tuple[int, str, int]] = []
    for i, text in enumerate(texts):
        parsed = _parse_marker(text)
        if parsed is not None:
            markers.append((i, parsed[0], parsed[1]))
    candidates = _chain_candidates(markers, texts, _indent_offsets(lines))
    chain = _numbering_chain(candidates)
    reason_flags: list[str] = []
    # A printed sequence decides the boundaries when it accounts for at least
    # half of the voted starts; three stray "1." "2." "3." lines in an
    # unnumbered list do not.
    numbered_style = len(chain) >= _MIN_CHAIN and 2 * len(chain) >= len(voted)
    marked: dict[str, list[int]] = defaultdict(list)
    for i, text in enumerate(texts):
        mark = _list_mark(text)
        if mark is not None:
            marked[mark].append(i)
    bulleted = max(marked.values(), key=len, default=[])
    if numbered_style:
        missing = _chain_missing_numbers(chain, candidates, texts)
        starts = sorted({0, *chain, *missing, *_chain_gap_starts(chain, texts, scores, onsets)})
        protected = {*chain, *missing}
        reason_flags.append("numbering_sequence")
        cut = _numbered_list_end(chain, lines)
        if cut is not None:
            after = collapse_ws(" ".join(line.text for line in lines[cut:]))
            tail_text = collapse_ws(f"{after} {tail_text}")
            lines = lines[:cut]
            texts = texts[:cut]
            starts = [start for start in starts if start < cut]
    elif len(bulleted) >= _MIN_CHAIN and 2 * len(bulleted) >= len(voted):
        # A bullet or a bracketed label ("[E1]", "[B]") opens every entry.
        starts = sorted({0, *bulleted})
        protected = set()
        reason_flags.append("list_marks")
    else:
        starts = voted
        protected = set()

    bounds = [*starts[1:], len(lines)]
    groups = [list(range(start, end)) for start, end in zip(starts, bounds, strict=True)]
    groups = _rejoin_fragments(groups, lines, texts, protected)
    groups = _split_multi_doi(groups, lines, texts, scores)

    entries: list[str] = []
    numbered: list[bool] = []
    link_dois: list[str | None] = []
    for group in groups:
        text = _entry_text([lines[i] for i in group])
        if not text:
            continue
        entries.append(text)
        numbered.append(group[0] in protected)
        linked: list[str] = []
        for i in group:
            for doi in lines[i].link_dois:
                if doi not in linked:
                    linked.append(doi)
        link_dois.append(linked[0] if len(linked) == 1 and not _printed_dois(text) else None)
    if not entries:
        return None
    spans: list[tuple[int, int]] = []
    cursor = 0
    for entry in entries:
        spans.append((cursor, cursor + len(entry)))
        cursor += len(entry) + 1
    text = "\n".join(entries)
    if tail_text:
        text += "\n" + tail_text
        reason_flags.append("end_of_list_cut")
    if stream.furniture_removed:
        reason_flags.append("furniture_removed")
    if stream.split_section:
        reason_flags.append("split_section_joined")
    if any(link_dois):
        reason_flags.append("doi_from_link_annotation")
    return StreamSegmentation(
        entries=tuple(entries),
        text=text,
        spans=tuple(spans),
        numbered=tuple(numbered),
        link_dois=tuple(link_dois),
        numbered_style=numbered_style,
        reason_flags=tuple(reason_flags),
        section_key=stream.section_key,
    )


_MIN_CHAIN = 3


def _chain_missing_numbers(
    chain: list[int], candidates: list[tuple[int, str, int]], texts: list[str]
) -> list[int]:
    """Candidate lines carrying a number the sequence skipped, wherever they stand.

    Layout reading order can put entry 5 before entry 4 across a column or box
    boundary; entry 4 still opens an entry. Only a line whose marker is of the
    sequence's family (arabic or roman) qualifies.
    """
    markers = [marker for marker in (_parse_marker(texts[i]) for i in chain) if marker]
    if not markers:
        return []
    family = _marker_family(markers[0][0])
    numbers = [marker[1] for marker in markers]
    skipped = set(range(min(numbers), max(numbers) + 1)) - set(numbers)
    in_chain = set(chain)
    found: list[int] = []
    for line_index, kind, number in candidates:
        if line_index in in_chain or number not in skipped or _marker_family(kind) != family:
            continue
        skipped.discard(number)
        found.append(line_index)
    return found


def _numbered_list_end(chain: list[int], lines: list[StreamLine]) -> int | None:
    """Where the text after a numbered list's last entry stops being part of it.

    The last entry runs to the end of its own reference box; boxes after it
    that the layout did not label as reference entries (the next article's
    title, a notice, body text) are not part of it.
    """
    last = chain[-1]
    last_line = lines[last]
    if last_line.region_label not in _ENTRY_LABELS:
        return None
    for index in range(last + 1, len(lines)):
        line = lines[index]
        if line.region != last_line.region:
            return None if line.region_label in _ENTRY_LABELS else index
    return None


def _chain_gap_starts(
    chain: list[int], texts: list[str], scores: list[float], onsets: list[bool]
) -> list[int]:
    """Where the sequence skips a number, the best voted start between the two."""
    added: list[int] = []
    numbers = [_parse_marker(texts[i]) for i in chain]
    for (a, marker_a), (b, marker_b) in zip(
        zip(chain, numbers, strict=True), zip(chain[1:], numbers[1:], strict=True), strict=False
    ):
        if marker_a is None or marker_b is None or marker_b[1] - marker_a[1] != 2:
            continue
        between = [i for i in range(a + 1, b) if scores[i] >= _START_THRESHOLD and onsets[i]]
        if between:
            added.append(max(between, key=lambda i: scores[i]))
    return added


def _rejoin_fragments(
    groups: list[list[int]], lines: list[StreamLine], texts: list[str], protected: set[int]
) -> list[list[int]]:
    """Rejoin an entry that opens in lower case (or on punctuation) with no date or DOI."""
    out: list[list[int]] = []
    for group in groups:
        first = group[0]
        text = " ".join(texts[i] for i in group)
        stripped = _strip_marker(texts[first])
        fragment = (
            out
            and first not in protected
            and not _is_dash_start(texts[first])
            and (_LOWER_START.match(stripped) or _PUNCT_START.match(texts[first]))
            and not _entry_has_date_or_doi(text)
            and not any(lines[i].link_dois for i in group)
        )
        if fragment:
            out[-1].extend(group)
        else:
            out.append(list(group))
    return out


def _split_multi_doi(
    groups: list[list[int]], lines: list[StreamLine], texts: list[str], scores: list[float]
) -> list[list[int]]:
    """Split an entry that carries two DOIs (printed or linked) after the first.

    The cut goes after the last line carrying the first DOI, or at the
    best-voted line between it and the line bringing the second DOI.
    """

    def line_dois(i: int) -> set[str]:
        return {d.lower() for d in _printed_dois(texts[i])} | {
            d.lower() for d in lines[i].link_dois
        }

    out: list[list[int]] = []
    for group in groups:
        pending = list(group)
        while len(pending) >= 2:
            seen: set[str] = set()
            last_seen: int | None = None
            second: int | None = None
            for position, i in enumerate(pending):
                dois = line_dois(i)
                if not dois:
                    continue
                if seen and dois - seen:
                    second = position
                    break
                seen |= dois
                last_seen = position
            if second is None or last_seen is None or second <= last_seen:
                break
            window = range(last_seen + 1, second + 1)
            voted = [p for p in window if scores[pending[p]] >= _START_THRESHOLD]
            cut = max(voted, key=lambda p: scores[pending[p]]) if voted else last_seen + 1
            out.append(pending[:cut])
            pending = pending[cut:]
        out.append(pending)
    return out


# A line with fewer alphanumerics than this cannot be placed in a segment.
_PLACE_MIN_CHARS = 6


def link_dois_for_segments(
    segments: Sequence[str], lines: Sequence[StreamLine]
) -> list[str | None]:
    """The DOI link over each segment's own lines, when there is exactly one.

    Walks the stream's lines through the segments' text in order; a line lies
    within a segment when its whole text is found there, and a line that
    straddles two segments places nothing. A segment takes a DOI only when
    the DOI links over its lines name exactly one and it prints none itself.
    """
    mapped: list[str | None] = [None] * len(segments)
    if not segments or not any(line.link_dois for line in lines):
        return mapped
    keys = [_match_key(segment) for segment in segments]
    ends = list(itertools.accumulate(len(key) for key in keys))
    joined = "".join(keys)
    claims: dict[int, dict[str, str]] = defaultdict(dict)
    cursor = 0
    for line in lines:
        key = _match_key(line.text)
        if len(key) < _PLACE_MIN_CHARS:
            continue
        position = joined.find(key, cursor)
        if position < 0:
            continue
        segment = bisect.bisect_right(ends, position)
        if segment >= len(ends) or position + len(key) > ends[segment]:
            continue
        cursor = position + len(key)
        for doi in line.link_dois:
            claims[segment][doi.lower()] = doi
    for segment, dois in claims.items():
        if len(dois) == 1 and not _printed_dois(segments[segment]):
            mapped[segment] = next(iter(dois.values()))
    return mapped


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

_MIN_ENTRY_CHARS = 20
# An entry this many times the typical entry length (and at least this long)
# holds several references.
_OUTLIER_LENGTH_FACTOR = 3.0
_OUTLIER_MIN_CHARS = 500
# An author-date opening "(2001)" or a Vancouver date "2001;12" opens one
# reference: two in one entry mark a merge.
_PAREN_YEAR = re.compile(r"\((?:1[6-9]|20)\d\d[a-z]?(?:,[^)]{0,20})?\)")
_VANCOUVER_DATE = re.compile(r"(?<!\d)(?:1[6-9]|20)\d\d[a-z]?\s*;\s*\d")


# A year standing on its own, not a page in a range ("pp. 1765-1770"), part of
# a DOI or an ISSN.
_STANDALONE_YEAR = re.compile(r"(?<![\d\-–—/.:])(?:1[6-9]|20)\d\d(?![\d\-–—/])")
# A second date one reference prints: when it was accessed, or first
# published.
_SECOND_DATE = re.compile(
    r"(?:acess?o em|accessed(?: on)?|retrieved(?: on)?|abgerufen am|zugegriffen am"
    r"|consultado(?: em| el)?|visited(?: on)?|date of access|original(?:ly)? (?:work )?published"
    r"|first published|reprinted?(?: in)?)[^;)\]]{0,30}?(?:1[6-9]|20)\d\d",
    re.IGNORECASE,
)


def _year_count(text: str) -> int:
    text = _SECOND_DATE.sub(" ", text)
    return len({match.group(0) for match in _STANDALONE_YEAR.finditer(text)})


# Coverage is measured on fixed 12-character pieces of the section's text, so
# a row the layout read twice (an entry box inside an aggregate box) counts
# once and a segmentation that reads it once is not short of text.
_SHINGLE_CHARS = 12


def _section_coverage(entries: Sequence[str], section_key: str) -> float:
    if len(section_key) < _SHINGLE_CHARS:
        return 1.0 if entries else 0.0
    pieces = {
        section_key[i : i + _SHINGLE_CHARS]
        for i in range(0, len(section_key) - _SHINGLE_CHARS + 1, _SHINGLE_CHARS)
    }
    # Entries are read in section order, so their joined text also holds the
    # pieces that straddle two entries.
    joined = "".join(_match_key(entry) for entry in entries)
    found = {joined[j : j + _SHINGLE_CHARS] for j in range(len(joined) - _SHINGLE_CHARS + 1)}
    return len(pieces & found) / len(pieces)


def numbered_entries(entries: Sequence[str]) -> list[bool]:
    """Entries whose printed number continues the previous entry's or leads into the next.

    Computed from the entries alone, so it treats every segmentation alike.
    """
    numbers: list[tuple[str, int] | None] = []
    for entry in entries:
        marker = _parse_marker(collapse_ws(entry))
        numbers.append((_marker_family(marker[0]), marker[1]) if marker and marker[1] > 0 else None)
    flags: list[bool] = []
    for index, current in enumerate(numbers):
        previous = numbers[index - 1] if index else None
        following = numbers[index + 1] if index + 1 < len(numbers) else None
        flags.append(
            current is not None
            and (
                (previous is not None and previous == (current[0], current[1] - 1))
                or (following is not None and following == (current[0], current[1] + 1))
            )
        )
    return flags


def typical_entry_length(*segmentations: Sequence[str]) -> float:
    """Median entry length over every segmentation being compared."""
    lengths = [len(entry.strip()) for entries in segmentations for entry in entries]
    return float(statistics.median(lengths)) if lengths else 0.0


def is_merged_entry(text: str, typical_length: float) -> bool:
    """Whether an entry holds more than one reference.

    Two distinct DOIs, two distinct years (pages and identifiers aside), two
    author-date openings or Vancouver dates, a second reference the
    merged-reference splitter can see, or a length far above the typical
    entry. A single reference that prints two years (a reprint, an access
    date) is flagged too; the flag weighs alike in every segmentation that
    keeps that reference whole.
    """
    return bool(
        len(_printed_dois(text)) >= 2
        or _year_count(text) >= 2
        or len(_PAREN_YEAR.findall(text)) >= 2
        or len(_VANCOUVER_DATE.findall(text)) >= 2
        or len(text) > max(_OUTLIER_MIN_CHARS, _OUTLIER_LENGTH_FACTOR * typical_length)
        or find_interior_onsets(text)
    )


def segmentation_quality(
    entries: Sequence[str], section_key: str, *, typical_length: float | None = None
) -> float:
    """Share of the section plausibly split into single references, in [0, 1].

    Coverage of the section's text (``LineStream.section_key``, which keeps
    furniture and anything after the list, so text a segmentation leaves out
    costs it) times the share of entries that look like one complete
    reference: not a fragment (opens in lower case, is very short, or carries
    no date or DOI unless its printed number continues the list's) and not a
    merge (``is_merged_entry``). ``typical_length`` should come from every
    segmentation being compared (``typical_entry_length``) so that all are
    judged alike; it defaults to these entries' median.
    """
    if not entries or not section_key:
        return 0.0
    coverage = _section_coverage(entries, section_key)
    numbered = numbered_entries(entries)
    typical = typical_length if typical_length else typical_entry_length(entries)
    good = 0
    for entry, is_numbered in zip(entries, numbered, strict=True):
        text = entry.strip()
        stripped = _strip_marker(text)
        fragment = (
            len(text) < _MIN_ENTRY_CHARS
            or bool(_LOWER_START.match(stripped) and not _is_dash_start(text))
            or (not is_numbered and not _entry_has_date_or_doi(text))
        )
        good += not (fragment or is_merged_entry(text, typical))
    return coverage * good / len(entries)


def geom_line_records(stream: LineStream) -> tuple[list[dict[str, Any]], list[int]]:
    """The text-layer lines' geometry records and their stream positions."""
    records: list[dict[str, Any]] = []
    positions: list[int] = []
    for position, line in enumerate(stream.lines):
        if line.geometry is not None:
            records.append(line.geometry)
            positions.append(position)
    return records, positions


def stream_probabilities(
    stream: LineStream, predict: Callable[[list[dict[str, Any]]], list[float]] | None
) -> list[float | None] | None:
    """Per-line start probabilities from *predict* (the geometry model), aligned to the stream."""
    if predict is None:
        return None
    records, positions = geom_line_records(stream)
    if not records:
        return None
    probabilities = predict(records)
    if len(probabilities) != len(records):
        return None
    aligned: list[float | None] = [None] * len(stream.lines)
    for position, probability in zip(positions, probabilities, strict=True):
        aligned[position] = float(probability)
    return aligned
