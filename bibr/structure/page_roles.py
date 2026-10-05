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
has. Its rules, in the order a block is tested:

1. Recurrence: a heading or a short body row whose normalised text is printed
   on two or more pages, in a margin band. The first printed occurrence of a
   heading on the first page is the title, not its running head, unless it
   reads as furniture itself (a copyright line, a masthead).
2. A later title: a ``doc_title`` in a margin band on a page after the
   title's, which the layout model gives to running heads and banners.

The margin bands (``_TOP_BAND`` / ``_BOTTOM_BAND``, 0..1000 layout space) are
the position evidence; a block without a box has none, and counts as in a
band, the behaviour from before regions carried coordinates.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

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
_REPEAT = "page_roles.repeat"
_LATER_TITLE = "page_roles.later_title"

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

Occurrence = tuple[int, int]


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
) -> PageRoles:
    """Decide the page roles of one document's blocks.

    *regions* is indexed by absolute page, as the OCR stage hands it over
    (pages before a ``start_page`` are empty), in the region IR or its wire
    format, as ``PDFParser`` takes it; *first_page_index* is the first page
    the run processed.
    """
    from bibr.structure.pdf_parser import LABEL_TREATMENT

    boxes: dict[Occurrence, tuple[float, float, float, float] | None] = {}
    headings: dict[str, list[Occurrence]] = {}
    bodies: dict[str, list[Occurrence]] = {}
    doc_titles: list[tuple[int, int, str]] = []
    pages_with_text: set[int] = set()
    for page, page_regions in enumerate(regions):
        for index, region in enumerate(page_regions):
            if not isinstance(region, OcrRegionResult):
                region = OcrRegionResult.from_dict(region)
            native_label = region.native_label
            label = native_label if native_label in LABEL_TREATMENT else region.label
            content = fix_ocr_artifacts(region.content).strip()
            if not content:
                continue
            pages_with_text.add(page)
            box = bbox_to_tuple(region.bbox_2d)
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

    def tag(occurrence: Occurrence, component: str, *, score=None, evidence=()) -> None:
        if occurrence in tags:
            return
        page, index = occurrence
        target = ids.block(page, index)
        tags[occurrence] = RoleTag(
            target=target,
            role=_edge_role(boxes[occurrence]),
            decided=Decided(
                component,
                PAGE_ROLES_RULE,
                score=score,
                evidence=tuple(ids.block(p, i) for p, i in evidence if (p, i) != occurrence),
            ),
        )

    # Rule 1: recurrence in a margin band. Repetition alone is not evidence:
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

    # Rule 2: a doc_title in a margin band on a page after the title's. The
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
    return PageRoles(ordered, repeats)
