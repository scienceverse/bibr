"""Page roles: the one writer of a document's page furniture.

Running heads and feet, page numbers, watermarks and line numbers are printed
by the page, not by the article. This module decides them once per document
and writes each decision as a :class:`~bibr.document.model.RoleTag` whose
target is a layer id (:mod:`bibr.document.ids`): a block, the post-OCR region
of that position on its page. Consumers that keep furniture out of the text
read the tags; none re-derives furniture with a margin band of its own.

A role is evidence, never a gate. A tagged block stays on the layer and in the
region summaries; a consumer that drops furniture from text drops it by tag,
so the drop can be traced to the tag's :class:`~bibr.document.model.Decided`
record (the rule, its score and the ids of the objects it rests on).

The writer runs with the document layer on or off, on the evidence the run
has. Its rules for a block, in the order they are tried:

1. The layout label: a ``header``, ``footer`` or page-number region is the
   layout model's own running head, foot or page number.
2. Recurrence: a heading or a short body row whose normalised text is printed
   on two or more pages, in a margin band. The first printed occurrence of a
   heading on the first page is the title, not its running head, unless it
   reads as furniture itself (a copyright line, a masthead).
3. A later title: a ``doc_title`` in a margin band on a page after the
   title's, which the layout model gives to running heads and banners.

The margin bands (``_TOP_BAND`` / ``_BOTTOM_BAND``, 0..1000 layout space) are
the position evidence; a block without a box has none, and counts as in a
band, the behaviour from before regions carried coordinates.

Text-layer lines (``PdfInspection.page_lines``, which the reference line
stream reads; ids ``p{page}.pl{n}``) get roles too:

4. A line whose centre lies in the box of a block the layout labels as
   furniture (rule 1) has that block's role.
5. Edge recurrence: among the top and bottom two lines of each page, a line
   whose digit-masked text (:func:`furniture_key`) recurs at the edge of two
   or more pages is a running head or foot.
6. Page numbers: a lone number at a page edge whose offset from the page
   number recurs on two or more pages.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bibr.document import ids
from bibr.document.model import Decided, RoleTag
from bibr.input.consolidate_text import fix_ocr_artifacts
from bibr.ocr.types import OcrRegionResult
from bibr.paper_contents import FRONT_MATTER_MASTHEAD_RE, is_exact_front_matter_furniture
from bibr.structure.parse_text import TextHandlersMixin
from bibr.structure.text_repair import bbox_to_tuple, strip_markdown_emphasis

RUNNING_HEADER = "running_header"
RUNNING_FOOTER = "running_footer"
PAGE_NUMBER = "page_number"
WATERMARK = "watermark"
LINE_NUMBER = "line_number"
PAGE_ROLES = frozenset({RUNNING_HEADER, RUNNING_FOOTER, PAGE_NUMBER, WATERMARK, LINE_NUMBER})

# The writer's rule version, on every tag it writes.
PAGE_ROLES_RULE = "page_roles/1"
_LABEL = "page_roles.label"
_REPEAT = "page_roles.repeat"
_LATER_TITLE = "page_roles.later_title"
_IN_LABEL_BOX = "page_roles.in_label_box"
_EDGE_REPEAT = "page_roles.edge_repeat"
_PAGE_NUMBER_RUN = "page_roles.page_number_run"

# Labels whose boxes hold page furniture, never text: the layout model's own
# page role for the block, and for the text-layer lines inside its box.
_LABEL_ROLES = {
    "header": RUNNING_HEADER,
    "header_image": RUNNING_HEADER,
    "footer": RUNNING_FOOTER,
    "footer_image": RUNNING_FOOTER,
    "number": PAGE_NUMBER,
    "page_number": PAGE_NUMBER,
}

_HEADING_LABELS = frozenset({"doc_title", "paragraph_title"})
# Body-text labels a running head can carry. A running header GLM-OCR tags
# ``text`` (a wide-letter-spaced preprint banner) would otherwise leak into
# whatever section is active at the page break, the References block included.
_BODY_TEXT_LABELS = frozenset({"text", "content", "vertical_text"})
# Page furniture is short; a real paragraph that happens to repeat across
# pages is longer and must be preserved. A longer body row counts only when it
# lies wholly inside a margin band: a preprint banner (the medRxiv rights,
# licence and DOI lines run to about 340 chars) that the layout model labels
# ``text`` on some pages. A manuscript that prints its body twice repeats long
# paragraphs that start or end in the band.
REPEAT_MAX_LEN = 200
# Running heads sit in the top or bottom margin band of the page. ``bbox_2d``
# is 0..1000 image space, matching the page-edge test in ``parse_media``.
_TOP_BAND = 100.0
_BOTTOM_BAND = 900.0

# Text-layer lines at the top or bottom edge of their page that may be
# furniture: a running title paired with a page number.
_EDGE_LINES = 2
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
# A lone page number, arabic or roman, optionally as "page 12", "- 12 -" or
# "12 of 30".
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:(?:page|p\.|pp\.|seite|página|pagina|str\.)\s*)?[-–—]?\s*(\d{1,4}|[ivxlc]{1,7})\s*"
    r"[-–—]?\s*(?:(?:/|of|von|de)\s*\d{1,4})?\s*$",
    re.IGNORECASE,
)
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
# Running heads are mostly words. A line holding a DOI, URL, arXiv id or ISBN
# is never furniture, however alike two of them look once their digits are
# masked ("https://doi.org/#.#/#" on two pages of one journal).
_FURNITURE_MIN_LETTERS = 6
_LOCATOR_TEXT = re.compile(r"10\.\d{4,9}/|https?://|www\.|\bdoi\s*:|\barxiv\b|\bisbn\b", re.I)

Occurrence = tuple[int, int]
Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class TextRepeat:
    """Regions printing the same normalised text on two or more pages.

    ``kind`` is ``heading`` or ``body``, the label family the key was built
    for; ``members`` are ``(page, region index)`` in document order.
    """

    kind: str
    key: str
    members: tuple[Occurrence, ...]
    pages: int


class PageRoles:
    """One document's page-role tags, looked up by the object they tag.

    Blocks are keyed as :func:`bibr.document.rebuild.attach_blocks` keys them:
    by their position in the page's post-OCR region list, on the absolute
    0-based page. ``repeats`` holds the recurrence the writer found, for the
    consumers that decide other things from it (a repeated legend row, a
    later title printed again).
    """

    __slots__ = ("_by_target", "repeats", "tags")

    def __init__(
        self, tags: Iterable[RoleTag] = (), repeats: Mapping[Occurrence, TextRepeat] | None = None
    ) -> None:
        self.tags: tuple[RoleTag, ...] = tuple(tags)
        self.repeats: Mapping[Occurrence, TextRepeat] = dict(repeats or {})
        self._by_target = {tag.target: tag for tag in self.tags}

    def tag(self, target: str) -> RoleTag | None:
        return self._by_target.get(target)

    def block(self, page: int, region_index: int) -> RoleTag | None:
        """The tag on region *region_index* of 0-based *page*, if any."""
        return self._by_target.get(ids.block(page, region_index))

    def repeat(self, page: int, region_index: int) -> TextRepeat | None:
        return self.repeats.get((page, region_index))


def heading_key(content: str) -> str:
    """A heading's text as it repeats: no markdown prefix or emphasis, lowercased.

    Identical text at different layout levels keys the same.
    """
    normalized = re.sub(r"^#{1,6}\s*", "", content)
    normalized = strip_markdown_emphasis(normalized).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def body_key(content: str) -> str:
    return re.sub(r"\s+", " ", content).lower().strip()


def _touches_band(box: tuple[float, float, float, float] | None) -> bool:
    """Whether a box reaches into the top or bottom margin band; True without a box."""
    if box is None:
        return True
    _, y1, _, y2 = box
    return y1 <= _TOP_BAND or y2 >= _BOTTOM_BAND


def _inside_band(box: tuple[float, float, float, float] | None) -> bool:
    """Whether a box lies wholly inside a margin band: a banner, not a paragraph near the top."""
    if box is None:
        return False
    _, y1, _, y2 = box
    return y2 <= _TOP_BAND or y1 >= _BOTTOM_BAND


def _edge_role(box: tuple[float, float, float, float] | None) -> str:
    if box is None:
        return RUNNING_HEADER
    return RUNNING_FOOTER if (box[1] + box[3]) / 2.0 >= 500.0 else RUNNING_HEADER


def furniture_key(text: str) -> str | None:
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


def page_number_value(text: str) -> int | None:
    """The number *text* prints when it is a lone page number, else None."""
    match = _PAGE_NUMBER_LINE.match(text)
    if match is None:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else _roman_value(token)


def page_line_targets(page_lines: Sequence[Mapping[str, Any]]) -> list[str | None]:
    """The id of each of *page_lines*, in order (None for a line without a page).

    A page line is ``p{page}.pl{n}``: *n* is its position among its page's
    lines, *page* the 0-based page of its 1-based ``page``.
    """
    seen: dict[int, int] = defaultdict(int)
    targets: list[str | None] = []
    for line in page_lines:
        page = int(line.get("page") or 0) - 1
        if page < 0:
            targets.append(None)
            continue
        targets.append(ids.page_line(page, seen[page]))
        seen[page] += 1
    return targets


def _center(box: Sequence[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _inside(point: tuple[float, float], box: Box) -> bool:
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _line_roles(
    page_lines: Sequence[Mapping[str, Any]],
    label_boxes: Mapping[int, Sequence[tuple[str, str, Box]]],
) -> list[RoleTag]:
    """Rules 4-6: the page roles of the text-layer lines.

    *label_boxes* holds, per 0-based page, ``(block id, role, box)`` for the
    blocks rule 1 tagged.
    """
    targets = page_line_targets(page_lines)
    by_page: dict[int, list[int]] = defaultdict(list)
    for position, line in enumerate(page_lines):
        if (
            targets[position] is not None
            and line.get("bbox")
            and str(line.get("text") or "").strip()
        ):
            by_page[int(line["page"])].append(position)
    tags: dict[int, RoleTag] = {}

    def tag(position: int, role: str, component: str, *, score=None, evidence=()) -> None:
        if position in tags:
            return
        target = targets[position]
        assert target is not None  # noqa: S101 - lines without a page are skipped
        tags[position] = RoleTag(
            target=target,
            role=role,
            decided=Decided(
                component,
                PAGE_ROLES_RULE,
                score=score,
                evidence=tuple(evidence_id for evidence_id in evidence if evidence_id != target),
            ),
        )

    # Rule 4: inside a furniture block's box, by the line's centre.
    for page, positions in by_page.items():
        boxes = label_boxes.get(page - 1, ())
        for position in positions if boxes else ():
            center = _center(page_lines[position]["bbox"])
            for block_id, role, box in boxes:
                if _inside(center, box):
                    tag(position, role, _IN_LABEL_BOX, evidence=(block_id,))
                    break

    # Rules 5 and 6 read the top and bottom two lines of each page only, as
    # the geometry segmenter's own capture does (``bibr.ocr.ref_geometry``).
    pages_by_key: dict[str, set[int]] = defaultdict(set)
    keyed: dict[str, list[int]] = defaultdict(list)
    numbered: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for page, positions in by_page.items():
        ordered = sorted(positions, key=lambda position: page_lines[position]["bbox"][1])
        edge = set(ordered[:_EDGE_LINES] + ordered[-_EDGE_LINES:])
        for position in positions:
            if position not in edge:
                continue
            text = str(page_lines[position]["text"])
            value = page_number_value(text)
            if value is not None:
                numbered[value - page].append((position, page))
                continue
            key = furniture_key(text)
            if key is not None:
                pages_by_key[key].add(page)
                keyed[key].append(position)
    n_pages = max(1, len(by_page))
    for key, positions in keyed.items():
        pages = len(pages_by_key[key])
        if pages < 2:
            continue
        evidence = [targets[position] for position in positions]
        for position in positions:
            center_y = _center(page_lines[position]["bbox"])[1]
            role = RUNNING_FOOTER if center_y >= 500.0 else RUNNING_HEADER
            tag(position, role, _EDGE_REPEAT, score=round(pages / n_pages, 3), evidence=evidence)
    # A lone number is a page number when another page carries one at the
    # same offset from its page number; a year or a volume that happens to
    # stand alone on an edge line does not.
    for members in numbered.values():
        pages = len({page for _, page in members})
        if pages < 2:
            continue
        evidence = [targets[position] for position, _ in members]
        for position, _ in members:
            tag(
                position,
                PAGE_NUMBER,
                _PAGE_NUMBER_RUN,
                score=round(pages / n_pages, 3),
                evidence=evidence,
            )
    return [tags[position] for position in sorted(tags)]


def _is_title_furniture(key: str) -> bool:
    """Text that is furniture even where a title would be: a copyright line, a masthead."""
    return (
        TextHandlersMixin._is_copyright_notice(key)
        or is_exact_front_matter_furniture(key)
        or bool(FRONT_MATTER_MASTHEAD_RE.match(key))
    )


def write_page_roles(
    regions: Sequence[Sequence[OcrRegionResult | dict]],
    *,
    first_page_index: int = 0,
    page_lines: Sequence[Mapping[str, Any]] | None = None,
) -> PageRoles:
    """Decide the page roles of one document's blocks and text-layer lines.

    *regions* is indexed by absolute page, as the OCR stage hands it over
    (pages before a ``start_page`` are empty), in the region IR or its wire
    format, as ``PDFParser`` takes it; *first_page_index* is the first page
    the run processed. *page_lines* are the text-layer lines
    (``PdfInspection.page_lines``), when the run read a text layer.
    """
    from bibr.structure.pdf_parser import LABEL_TREATMENT

    boxes: dict[Occurrence, Box | None] = {}
    labelled: dict[Occurrence, str] = {}
    label_boxes: dict[int, list[tuple[str, str, Box]]] = defaultdict(list)
    headings: dict[str, list[Occurrence]] = {}
    bodies: dict[str, list[Occurrence]] = {}
    doc_titles: list[tuple[int, int, str]] = []
    pages_with_text: set[int] = set()
    for page, page_regions in enumerate(regions):
        for index, region in enumerate(page_regions):
            if not isinstance(region, OcrRegionResult):
                region = OcrRegionResult.from_dict(region)
            native_label = region.native_label
            box = bbox_to_tuple(region.bbox_2d)
            # Rule 1 reads the label the region summaries carry.
            label_role = _LABEL_ROLES.get(native_label or region.label or "")
            if label_role is not None:
                labelled[(page, index)] = label_role
                if box is not None:
                    label_boxes[page].append((ids.block(page, index), label_role, box))
            label = native_label if native_label in LABEL_TREATMENT else region.label
            content = fix_ocr_artifacts(region.content).strip()
            if not content:
                continue
            pages_with_text.add(page)
            boxes[(page, index)] = box
            if label in _BODY_TEXT_LABELS:
                key = body_key(content)
                if key and (len(key) <= REPEAT_MAX_LEN or _inside_band(box)):
                    bodies.setdefault(key, []).append((page, index))
                continue
            if label not in _HEADING_LABELS:
                continue
            key = heading_key(content)
            if not key:
                continue
            headings.setdefault(key, []).append((page, index))
            if label == "doc_title":
                doc_titles.append((page, index, key))

    repeats: dict[Occurrence, TextRepeat] = {}
    for kind, groups in (("heading", headings), ("body", bodies)):
        for key, members in groups.items():
            n_pages = len({page for page, _ in members})
            if n_pages < 2:
                continue
            repeat = TextRepeat(kind=kind, key=key, members=tuple(members), pages=n_pages)
            for occurrence in members:
                repeats[occurrence] = repeat

    tags: dict[Occurrence, RoleTag] = {}

    def tag(occurrence: Occurrence, component: str, *, role=None, score=None, evidence=()) -> None:
        if occurrence in tags:
            return
        page, index = occurrence
        target = ids.block(page, index)
        tags[occurrence] = RoleTag(
            target=target,
            role=role or _edge_role(boxes[occurrence]),
            decided=Decided(
                component,
                PAGE_ROLES_RULE,
                score=score,
                evidence=tuple(ids.block(p, i) for p, i in evidence if (p, i) != occurrence),
            ),
        )

    # Rule 1: the layout label.
    for occurrence, label_role in labelled.items():
        tag(occurrence, _LABEL, role=label_role)

    # Rule 2: recurrence in a margin band. Repetition alone is not evidence:
    # multi-study papers legitimately repeat ``Method``/``Results`` per study,
    # and papers repeat a short body row mid-column ("where", "(TIF)").
    for occurrence, repeat in repeats.items():
        members = repeat.members
        # The first printed title occurrence stays the title even when the
        # same text is repeated as a running header on later pages.
        if (
            repeat.kind == "heading"
            and occurrence == members[0]
            and occurrence[0] == first_page_index
            and not _is_title_furniture(repeat.key)
        ):
            continue
        if not _touches_band(boxes[occurrence]):
            continue
        score = round(repeat.pages / max(1, len(pages_with_text)), 3)
        tag(occurrence, _REPEAT, score=score, evidence=members)

    # Rule 3: a doc_title in a margin band on a page after the title's. The
    # title is the first doc_title that is not a copyright or permission blurb
    # (publishers sometimes print one above it); a title split over several
    # doc_title rows of its own page is not a running head.
    anchor = next(
        (
            i
            for i, (_, _, key) in enumerate(doc_titles)
            if not TextHandlersMixin._is_copyright_notice(key)
        ),
        0,
    )
    if len(doc_titles) > anchor + 1:
        anchor_page, anchor_index, _ = doc_titles[anchor]
        for page, index, _ in doc_titles[anchor + 1 :]:
            if page != anchor_page and _touches_band(boxes[(page, index)]):
                tag((page, index), _LATER_TITLE, evidence=((anchor_page, anchor_index),))

    ordered = [tags[occurrence] for occurrence in sorted(tags)]
    ordered.extend(_line_roles(page_lines or (), label_boxes))
    return PageRoles(ordered, repeats)
