"""Link annotations: where each one goes and what it points at.

pdfium reads a link's rectangle, quadrilaterals, action and destination (the
annotation's own ``/Dest`` or a GoTo action's ``/D``); :func:`read_page_links`
takes them page by page under the ``pdfium_lock``. What a link points at is
decided afterwards, from the layer's own text, by :func:`build_links`:

- an action that leaves the document (URI, a jump into another file, launch) is
  ``external``;
- a destination reached by name is classed by the name, with the rules the
  pdf_info census used (bibliography ``cite.``/``bib``/``ref``..., float
  ``figure.``/``table.``/``fig``..., section ``section.``/``sec``...), plus
  footnote and equation names; a name that is only a page number
  (``page.12``, ``Item.5``) says nothing;
- otherwise, or when the name does not class it, the text under the
  destination does: the first 14 words of the band from 6 points above to 40
  below the destination's y, read in order from the top, are a float when they
  start like a caption ("Figure 3", "Table 1") and a bibliography entry when
  they look like a reference ("[12] Smith", "Doe, J.", a year);
- a destination that lands on no page of the document is ``unresolved``.

The text rule reads the target page in its unrotated frame, where the census
read it as displayed, so on a rotated target page it sees another band. A
target page without text gives it none: a page the layer lacks (a layer built
for a page range), one the harvest failed on and one that could not be sized.

A link is read from pdfium's page, not from its text, so a page whose text
could not be read keeps its links, without the spans they cover.
"""

from __future__ import annotations

import ctypes
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bibr.document import destinations, ids
from bibr.document.model import Box, Decided, Link, Page

if TYPE_CHECKING:
    from bibr.document.harvest import _Api

APIS = (
    "FPDFLink_Enumerate",
    "FPDFLink_GetAnnotRect",
    "FPDFLink_CountQuadPoints",
    "FPDFLink_GetQuadPoints",
    "FPDFLink_GetAction",
    "FPDFLink_GetDest",
    "FPDFAction_GetType",
    "FPDFAction_GetURIPath",
    "FPDFAction_GetFilePath",
)
# The annotation's own /Dest names the destination exactly; without these the
# name is found from the destination alone (see ``destinations``).
NAME_APIS = (
    "FPDFLink_GetAnnot",
    "FPDFPage_CloseAnnot",
    "FPDFAnnot_GetStringValue",
    "FPDFAnnot_GetValueType",
)

# The types of an annotation's /Dest value that give the link a destination:
# string, name, array, dictionary.
_DEST_VALUE_TYPES = {3, 4, 5, 6}
# pdfium's action types as the layer names them; anything else is "other".
_ACTIONS = {1: "goto", 2: "remote", 3: "uri", 4: "launch", 5: "remote"}

# A document with more link annotations than this is read to this many, and a link with more
# quadrilaterals than MAX_QUADS is read to that many, so a hostile PDF cannot hold the lock
# for long: 10,000 links take 0.2 s to read. The busiest gate192 paper has 920 links, the
# busiest manuscript 1,340, and the most quadrilaterals any link has is 2.
MAX_LINKS = 10_000
MAX_QUADS = 256

# The kind prefix of a link's id (``p3.lk4``).
KIND = "lk"
RULE_VERSION = "link_target/1"
_BY_ACTION = Decided("link_target.action", RULE_VERSION, calibrated=False)
_BY_NAME = Decided("link_target.name", RULE_VERSION, calibrated=False)
_BY_DEFAULT = Decided("link_target.default", RULE_VERSION, calibrated=False)

# The name rules, in the census's order; footnote and equation come after them,
# so a name the census classes keeps its class.
_NAME_RULES = [
    ("bib", re.compile(r"^(cite\.|bib|ref|cr\d|b\d|bb\d|r\d|en\d|c\d+$|ref-)", re.I)),
    (
        "float",
        re.compile(r"^(figure\.|table\.|fig|f\d|ff\d|tab|t\d|tbl|tf\d|scheme)", re.I),
    ),
    (
        "section",
        re.compile(
            r"^(section\.|subsection\.|subsubsection\.|chapter\.|appendix\.|sec|s\d|sect)", re.I
        ),
    ),
    ("footnote", re.compile(r"^(footnote|hfootnote|fn|ftn|fnref)", re.I)),
    ("equation", re.compile(r"^(equation|eq[\d:._-]|eqn)", re.I)),
]
# Names that only number a page, not what is on it.
_PAGE_NAME = re.compile(r"(page|pg|p)\.?\d+|item\.\d+|doc-start|toc\d*|page\d+", re.I)
_FLOAT_TEXT = re.compile(r"^\s*(fig(ure)?s?\.?|table|tab\.|scheme)\s*[a-z]?\d", re.I)
_BIB_TEXT = re.compile(
    r"^\s*\[?\d{1,3}[\].)]\s+\S|^\s*[A-Z][A-Za-z\'’-]+,\s+[A-Z]\.|\((19|20)\d{2}[a-z]?\)"
    r"|\b(19|20)\d{2}[a-z]?[.;,)]"
)
# The words read from the band under a destination (``destinations.BAND_ABOVE``/``BAND_BELOW``).
_BAND_WORDS = 14


@dataclass(slots=True)
class RawLink:
    """What one link annotation says, read under the lock."""

    page: int
    # The link's position among the page's link annotations, counting any that failed.
    number: int
    rect: Box
    quads: tuple[tuple[float, ...], ...]
    action: str
    uri: str | None
    dest_name: str | None
    name_source: str | None
    target_page: int | None
    target_xy: destinations.Position | None


@dataclass(slots=True)
class PageLinks:
    """What reading one page's link annotations gave."""

    found: list[RawLink]
    # The annotations gone through, those that failed included.
    enumerated: int
    # Whether every link annotation of the page was read and kept. It is not when one
    # was left out (it failed, or has no rectangle that is finite) or the document's
    # allowance of annotations ran out; a link cut short is kept, so it does not count.
    complete: bool
    # What left a link out, cut one short or stopped the read, each once: the reader of
    # the document says each once for all its pages, not once for each page.
    cuts: list[str]
    # Whether the document's allowance of annotations ran out on this page: another
    # annotation was there. The caller reads no later page then.
    stopped: bool
    # Why a link failed (the first on the page): it is left out, and the others are kept.
    failure: str | None
    # The number of the first link of the page whose destination was left unresolved because the
    # resolver's allowance of destinations that point at no page was spent.
    first_unresolved: int | None


def read_page_links(
    api: _Api,
    doc,
    page,
    page_index: int,
    names: destinations.NamedDests | None,
    resolver: destinations.Resolver,
    *,
    with_names: bool,
    limit: int = MAX_LINKS,
) -> PageLinks:
    """The page's link annotations: at most *limit*, what the document's allowance has left.

    A link that fails is left out; the others are kept. When the allowance is spent and
    another annotation is there, the read stops and says so (:attr:`PageLinks.stopped`). A
    document with exactly the allowance is read whole and says nothing.
    """
    found: list[RawLink] = []
    cuts: list[str] = []
    failure: str | None = None
    first_unresolved: int | None = None
    complete = True
    stopped = False
    position = ctypes.c_int(0)
    link = api.c.FPDF_LINK()
    number = -1
    while api.FPDFLink_Enumerate(page.raw, ctypes.byref(position), ctypes.byref(link)):
        if number + 1 >= limit:
            complete = False
            stopped = True
            _say(cuts, f"more than {MAX_LINKS} link annotations, the rest unread")
            break
        number += 1
        skipped = resolver.skipped
        try:
            raw, cut = _read_link(
                api, doc, page, link, (page_index, number), names, resolver, with_names
            )
        except Exception as exc:  # noqa: BLE001 - a layer component never fails the paper
            failure = failure or f"{type(exc).__name__}: {exc}"[:500]
            complete = False
            continue
        if first_unresolved is None and resolver.skipped != skipped:
            first_unresolved = number
        _say(cuts, cut)
        if raw is None:
            complete = False
            continue
        found.append(raw)
    return PageLinks(found, number + 1, complete, cuts, stopped, failure, first_unresolved)


def _say(cuts: list[str], text: str | None) -> None:
    """Add *text* to *cuts* unless it is None or already there."""
    if text is not None and text not in cuts:
        cuts.append(text)


def _read_link(
    api: _Api,
    doc,
    page,
    link,
    where: tuple[int, int],
    names: destinations.NamedDests | None,
    resolver: destinations.Resolver,
    with_names: bool,
) -> tuple[RawLink | None, str | None]:
    """The link and a note when it was cut short; None, with a note, for a link left out.

    A link is left out when it has no rectangle, or one with a coordinate that is not
    finite (pdfium reads a number too large for a float as infinity): it cannot be placed.
    A quadrilateral that is not finite is dropped.
    """
    pdfium_c = api.c
    rect = pdfium_c.FS_RECTF()
    if not api.FPDFLink_GetAnnotRect(link, ctypes.byref(rect)):
        return None, "a link annotation with no rectangle, left out"
    corners = (float(rect.left), float(rect.right), float(rect.bottom), float(rect.top))
    if not all(map(math.isfinite, corners)):
        return None, "a link annotation with a rectangle that is not finite, left out"
    left, right = sorted(corners[:2])
    bottom, top = sorted(corners[2:])
    quads = []
    n_quads = api.FPDFLink_CountQuadPoints(link)
    for index in range(min(n_quads, MAX_QUADS)):
        quad = pdfium_c.FS_QUADPOINTSF()
        if api.FPDFLink_GetQuadPoints(link, index, ctypes.byref(quad)):
            points = tuple(
                float(getattr(quad, f"{axis}{n}")) for n in (1, 2, 3, 4) for axis in "xy"
            )
            if all(map(math.isfinite, points)):
                quads.append(points)

    raw_name, has_dest_entry = _annotation_dest(api, page, link) if with_names else (None, False)
    action = api.FPDFLink_GetAction(link)
    kind = "none"
    uri = None
    if action:
        kind = _ACTIONS.get(api.FPDFAction_GetType(action), "other")
        if kind == "uri":
            uri = _action_uri(api, doc, action)
        elif kind in ("remote", "launch"):
            uri = _action_file(api, action)
    # A jump inside this document: its destination is the annotation's /Dest or
    # the GoTo action's /D. A remote jump's destination is in another file.
    dest = None
    if kind in ("none", "goto"):
        dest = api.FPDFLink_GetDest(doc.raw, link)
        if not dest and raw_name and names is not None:
            # pdfium finds a name by searching the name tree, which misses names a
            # tree not sorted as it expects holds; the table lists every one.
            dest = names.dest_of(raw_name)
        if kind == "none" and (dest or raw_name or has_dest_entry):
            kind = "dest"

    name, source = (raw_name, "annot") if raw_name else (None, None)
    target_page = target_xy = None
    if dest:
        if name is None and names is not None:
            name = names.name_of(dest)
            source = "table" if name is not None else None
        target_page = resolver.page(dest)
        if target_page is not None:
            target_xy = destinations.dest_position(api, dest)
    raw = RawLink(
        page=where[0],
        number=where[1],
        rect=(left, bottom, right, top),
        quads=tuple(quads),
        action=kind,
        uri=uri,
        dest_name=name,
        name_source=source,
        target_page=target_page,
        target_xy=target_xy,
    )
    cut = f"a link with more than {MAX_QUADS} quadrilaterals, the rest unread"
    return raw, (cut if n_quads > MAX_QUADS else None)


def _annotation_dest(api: _Api, page, link) -> tuple[str | None, bool]:
    """The annotation's own /Dest as a name, and whether it has a /Dest entry at all."""
    annot = api.FPDFLink_GetAnnot(page.raw, link)
    if not annot:
        return None, False
    try:
        name = destinations.utf16_text(api.FPDFAnnot_GetStringValue, annot, b"Dest")
        has_entry = api.FPDFAnnot_GetValueType(annot, b"Dest") in _DEST_VALUE_TYPES
    finally:
        api.FPDFPage_CloseAnnot(annot)
    return (name or None), has_entry


def _action_uri(api: _Api, doc, action) -> str | None:
    # The URI is 7-bit ASCII with a terminating NUL; the first call sizes it.
    size = api.FPDFAction_GetURIPath(doc.raw, action, None, 0)
    if size <= 1 or size > destinations.MAX_TEXT:
        return None
    buffer = ctypes.create_string_buffer(size)
    api.FPDFAction_GetURIPath(doc.raw, action, buffer, size)
    return buffer.raw[: size - 1].decode("utf-8", "replace").strip() or None


def _action_file(api: _Api, action) -> str | None:
    # The file path is UTF-8 with a terminating NUL.
    size = api.FPDFAction_GetFilePath(action, None, 0)
    if size <= 1 or size > destinations.MAX_TEXT:
        return None
    buffer = ctypes.create_string_buffer(size)
    api.FPDFAction_GetFilePath(action, buffer, size)
    return buffer.raw[: size - 1].decode("utf-8", "replace") or None


# --- What a link points at ----------------------------------------------------------


def class_by_name(name: str | None) -> str | None:
    """The class a destination name gives, or None when the name does not decide."""
    if not name or _PAGE_NAME.fullmatch(name):
        return None
    for target_class, rule in _NAME_RULES:
        if rule.search(name):
            return target_class
    return None


def class_by_text(text: str) -> str | None:
    """``float`` or ``bib`` when *text* (the words under a destination) reads like one."""
    if _FLOAT_TEXT.search(text):
        return "float"
    if _BIB_TEXT.search(text):
        return "bib"
    return None


class PageWords:
    """The words of a layer page by their top edge, for the text rule."""

    def __init__(self, page: Page) -> None:
        cols = page.cols
        if cols is None or page.crop_box is None:
            raise ValueError("a page without a text layer or a size has no words")
        self._cols = cols
        self._cy1 = page.crop_box[3]
        blank = (cols.rec_cp <= 32) | (cols.rec_cp == 0xA0) | cols.rec_newline
        inside = np.flatnonzero(~blank)
        self.starts = self.ends = np.empty(0, dtype=np.int64)
        self.top = self.left = np.empty(0)
        if not inside.size:
            return
        breaks = np.flatnonzero(np.diff(inside) > 1) + 1
        first = np.concatenate(([0], breaks))
        self.starts = inside[first]
        self.ends = inside[np.concatenate((breaks - 1, [inside.size - 1]))] + 1
        # A record that came from no glyph (src < 0) has no box: NaN, ignored by fmax/fmin.
        source = cols.rec_src[inside]
        has_box = source >= 0
        tops = np.full(inside.size, np.nan)
        lefts = np.full(inside.size, np.nan)
        tops[has_box] = cols.loose[source[has_box], 3]
        lefts[has_box] = cols.loose[source[has_box], 0]
        self.top = np.fmax.reduceat(tops, first)
        self.left = np.fmin.reduceat(lefts, first)

    def band(self, y: float) -> tuple[str, np.ndarray]:
        """The first words of the band under a destination at *y*, and their first records."""
        below, above = destinations.BAND_BELOW, destinations.BAND_ABOVE
        selected = np.flatnonzero((self.top >= y - below) & (self.top <= y + above))
        # Read from the top down, as the census read the displayed page: its
        # y grows downwards and is rounded before words compare.
        order = np.lexsort((self.left[selected], np.round(self._cy1 - self.top[selected])))
        words = selected[order][:_BAND_WORDS]
        text = " ".join(
            "".join(self._cols.record_char(int(i)) for i in range(self.starts[w], self.ends[w]))
            for w in words
        )
        return text, self.starts[words]


def _span_of(cols, records: np.ndarray) -> np.ndarray:
    """The span each of *records* is in (-1 for a record in none)."""
    begin, end = cols.span_rec[:, 0], cols.span_rec[:, 1]
    if not len(begin):
        return np.full(len(records), -1)
    spans = np.searchsorted(begin, records, side="right") - 1
    inside = (spans >= 0) & (records < end[np.clip(spans, 0, None)])
    return np.where(inside, spans, -1)


def _span_ids(page_index: int, cols, records: np.ndarray) -> tuple[str, ...]:
    spans = _span_of(cols, records)
    return tuple(ids.span(page_index, span) for span in sorted({int(s) for s in spans if s >= 0}))


def covered_spans(page: Page, link: RawLink) -> tuple[str, ...]:
    """The spans of the text a link covers: those with a character centred in a quad (or the rect)."""
    cols = page.cols
    if cols is None or not len(cols.rec_cp):
        return ()
    boxes = [
        (min(quad[0::2]), min(quad[1::2]), max(quad[0::2]), max(quad[1::2])) for quad in link.quads
    ] or [link.rect]
    drawn = ~((cols.rec_cp <= 32) | (cols.rec_cp == 0xA0) | cols.rec_newline)
    hit = np.zeros(len(cols.rec_cp), dtype=bool)
    for left, bottom, right, top in boxes:
        hit |= (
            drawn
            & (cols.rec_cx >= left)
            & (cols.rec_cx <= right)
            & (cols.rec_cy >= bottom)
            & (cols.rec_cy <= top)
        )
    return _span_ids(page.index, cols, np.flatnonzero(hit))


def build_links(raws: list[RawLink], pages: list[Page]) -> list[Link]:
    """The layer's links: *raws* in page order, each with its class and the spans it covers."""
    by_index = {page.index: page for page in pages}
    words: dict[int, PageWords] = {}
    links = []
    for raw in sorted(raws, key=lambda raw: (raw.page, raw.number)):
        source = by_index.get(raw.page)
        target_class, decided = _classify(raw, by_index, words)
        links.append(
            Link(
                link_id=ids.make(raw.page, KIND, raw.number),
                page=raw.page,
                rect=raw.rect,
                quads=raw.quads,
                action=raw.action,
                uri=raw.uri,
                dest_name=raw.dest_name,
                name_source=raw.name_source,
                target_page=raw.target_page,
                target_xy=raw.target_xy,
                target_class=target_class,
                target=decided,
                source_span_ids=covered_spans(source, raw) if source is not None else (),
            )
        )
    return links


def _classify(
    raw: RawLink, pages: dict[int, Page], words: dict[int, PageWords]
) -> tuple[str | None, Decided | None]:
    if raw.action in ("uri", "remote", "launch"):
        return "external", _BY_ACTION
    if raw.action not in ("dest", "goto"):
        return None, None
    by_name = class_by_name(raw.dest_name)
    if by_name is not None:
        return by_name, _BY_NAME
    if raw.target_page is None:
        return "unresolved", _BY_DEFAULT
    target = pages.get(raw.target_page)
    # A target page the layer lacks, one that failed and one without a text layer or a
    # size has no words: the text rule has nothing to read.
    if (
        target is not None
        and target.cols is not None
        and target.crop_box is not None
        and raw.target_xy is not None
    ):
        y = raw.target_xy[1]
        if y is not None:
            if raw.target_page not in words:
                words[raw.target_page] = PageWords(target)
            text, records = words[raw.target_page].band(y)
            by_text = class_by_text(text)
            if by_text is not None:
                evidence = _span_ids(target.index, target.cols, records)
                return by_text, Decided(
                    "link_target.text", RULE_VERSION, calibrated=False, evidence=evidence
                )
    return "other", _BY_DEFAULT
