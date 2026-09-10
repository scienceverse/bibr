"""Post-assembly normalization of figures and tables.

Runs at the tail of :meth:`PDFParser.parse`, before ``detect_xrefs`` and
``create_content_sections``, so merged ``figure_id``/``table_id`` numbering
stays consistent with xrefs and synthetic section headers.

Two pathologies are handled that the inline handlers in
:class:`~bibr.structure.parse_media.MediaHandlersMixin` cannot see:

* Multi-panel figures where each panel became its own ``PaperFigure`` with a
  bare marker caption ("A", "b", "(c)", "1") or no caption at all, next to
  one figure carrying the real "FIGURE N …" caption. The inline grouping
  only catches lowercase "(a)"-style title regions that follow their image
  region; uppercase markers and title-before-image orderings slip through.
* Multi-page tables emitted once per page with the caption repeated plus a
  trailing "(Continued)" marker.

This module only ever *merges*; deciding whether a float survives at all
belongs to the ownership layer in
:class:`~bibr.structure.parse_media.MediaHandlersMixin`, which reconciles
printed ids against caption receipts and would be contradicted by a second,
blinder survival heuristic here.

Both mergers renumber survivors, which moves the ids ``_finalize_media``
already froze into the caption-assignment receipt. The ``*_with_remap``
variants therefore hand back the old→new object-id map, and
:func:`remap_caption_receipt` replays it onto the receipt so its
``object_id`` values still name live floats. The renumbering keeps a printed
"Figure N"/"Table N" label as the id where a caption carries one — a bare
positional renumber from 1 broke the correspondence ``detect_xrefs`` resolves
body mentions by.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from bibr.paper_contents import (
    CaptionAssignment,
    CaptionAssignmentReceipt,
    PaperFigure,
    PaperTable,
)

logger = logging.getLogger(__name__)

# Bare panel markers only: a single letter (either case) or a 1-2 digit
# number, optionally parenthesized or dot-terminated. Anything with a
# descriptive tail ("A randomized trial …") is a real caption.
_BARE_PANEL_RE = re.compile(r"^\(?(?:[A-Za-z]|\d{1,2})[\).]?$")

# A caption that names its figure ("Figure 2:", "FIGURE 1 …", "Fig. 3").
_FIGURE_LABEL_RE = re.compile(r"^fig(?:ure)?\.?\s*\d+\b", re.IGNORECASE)
_FIGURE_LABEL_NUMBER_RE = re.compile(r"^fig(?:ure)?\.?\s*(\d+)\b", re.IGNORECASE)

_TABLE_LABEL_RE = re.compile(r"^table\s+(\d+)\b", re.IGNORECASE)

# Trailing continuation marker; parenthesized/bracketed only — a bare
# trailing word "continued" is too likely to be caption prose.
_CONTINUED_RE = re.compile(r"[([]\s*continued\s*[)\]]\s*[.:]?\s*$", re.IGNORECASE)

# Same threshold as parse_text._is_bbox_nearby.
_MAX_VERTICAL_GAP = 200


def _figure_bbox(fig: PaperFigure) -> list[float] | None:
    for prov in fig.provenance:
        if prov.bbox is not None:
            return list(prov.bbox)
    return None


def _union_bbox(a: list[float] | None, b: list[float] | None) -> list[float] | None:
    if a is None:
        return b
    if b is None:
        return a
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _vertically_near(a: list[float] | None, b: list[float] | None) -> bool:
    """Vertical-gap proximity, mirroring ``parse_text._is_bbox_nearby``:
    missing spatial info falls back to trusting sequential order."""
    if a is None or b is None:
        return True
    if a[3] <= b[1]:
        gap = b[1] - a[3]
    elif b[3] <= a[1]:
        gap = a[1] - b[3]
    else:
        gap = 0
    return gap <= _MAX_VERTICAL_GAP


def _renumber_honouring_printed_labels(objects, id_attribute: str, label_re) -> None:
    """Renumber survivors, keeping the printed label as the id where there is one.

    ``_reconcile_object_ids`` deliberately reserves the printed number as the
    object id — that is how ``detect_xrefs`` resolves a body mention of
    "Figure 2" — and it runs *before* the mergers here. Renumbering
    positionally from 1 therefore broke the correspondence whenever anything
    merged: with a caption-less panel absorbed into FIGURE 2, the survivors
    became 1 and 2, so a mention of "Figure 2" resolved to the figure
    captioned FIGURE 3. Unlabelled survivors take the lowest number the
    printed labels have not claimed.
    """
    printed: dict[int, int] = {}
    claimed: set[int] = set()
    for index, obj in enumerate(objects):
        match = label_re.match((obj.caption or "").strip())
        if match is None:
            continue
        number = int(match.group(1))
        if number > 0 and number not in claimed:
            printed[index] = number
            claimed.add(number)
    next_free = 1
    for index, obj in enumerate(objects):
        if index in printed:
            setattr(obj, id_attribute, printed[index])
            continue
        while next_free in claimed:
            next_free += 1
        setattr(obj, id_attribute, next_free)
        claimed.add(next_free)


def merge_figure_panels(figures: list[PaperFigure]) -> list[PaperFigure]:
    """Merged-list-only view of :func:`merge_figure_panels_with_remap`.

    Kept for callers that hold no caption receipt and so have nothing to
    repoint at the renumbered survivors.
    """
    return merge_figure_panels_with_remap(figures)[0]


def merge_figure_panels_with_remap(
    figures: list[PaperFigure],
) -> tuple[list[PaperFigure], dict[str, str]]:
    """Collapse per-panel figures into their labeled same-page figure.

    Figures whose caption is a bare panel marker are absorbed by the nearest
    labeled figure on the same page — following first (panels usually precede
    their caption in reading order), else preceding. Caption-less figures are
    absorbed only when spatially near the group being assembled (missing bbox
    info falls back to trusting sequential order, matching
    ``_is_bbox_nearby``). Absorbed figures donate their provenance, and their
    crop when the survivor has none. Figures with no target are always kept —
    survival is the ownership layer's call. Survivors are renumbered from 1.

    Returns the merged list plus the ``figure:<old>`` → ``figure:<new>`` map
    that renumbering implies, with every absorbed panel pointing at its
    survivor. The map is empty when no panel was absorbed, because then no id
    moved.
    """
    if not figures:
        return figures, {}
    is_target = [_FIGURE_LABEL_RE.match((f.caption or "").strip()) is not None for f in figures]
    absorbed: dict[int, int] = {}  # index -> target index
    group_bbox: dict[int, list[float] | None] = {}  # target index -> union bbox

    def nearest_target(i: int, fig: PaperFigure) -> int | None:
        following = next(
            (
                j
                for j in range(i + 1, len(figures))
                if is_target[j] and figures[j].page_number == fig.page_number
            ),
            None,
        )
        if following is not None:
            return following
        return next(
            (
                j
                for j in range(i - 1, -1, -1)
                if is_target[j] and figures[j].page_number == fig.page_number
            ),
            None,
        )

    def absorb(i: int, target: int) -> None:
        absorbed[i] = target
        current = group_bbox.get(target, _figure_bbox(figures[target]))
        group_bbox[target] = _union_bbox(current, _figure_bbox(figures[i]))

    # Pass 1 — bare panel markers: strong evidence, same-page target suffices.
    for i, fig in enumerate(figures):
        if is_target[i]:
            continue
        caption = (fig.caption or "").strip()
        if not caption or not _BARE_PANEL_RE.match(caption):
            continue
        target = nearest_target(i, fig)
        if target is not None:
            absorb(i, target)

    # Pass 2 — caption-less figures: weak evidence, require spatial proximity
    # to the group (which now includes pass-1 panels). Runs after pass 1 so a
    # grid of panels extends the reach to its caption-less members.
    for i, fig in enumerate(figures):
        if is_target[i] or i in absorbed or (fig.caption or "").strip():
            continue
        target = nearest_target(i, fig)
        bbox = _figure_bbox(fig)
        if target is not None and _vertically_near(
            bbox, group_bbox.get(target, _figure_bbox(figures[target]))
        ):
            absorb(i, target)
    if not absorbed:
        return figures, {}
    for i, target_idx in absorbed.items():
        target = figures[target_idx]
        panel = figures[i]
        target.provenance.extend(panel.provenance)
        if target.image_b64 is None and panel.image_b64 is not None:
            target.image_b64 = panel.image_b64
    # Snapshot the pre-merge ids before renumbering overwrites them; they are
    # what the caption receipt was frozen against.
    old_object_ids = [f"figure:{fig.figure_id}" for fig in figures]
    merged = [fig for i, fig in enumerate(figures) if i not in absorbed]
    logger.debug(
        "merge_figure_panels: %d -> %d figures (%d panels merged)",
        len(figures),
        len(merged),
        len(absorbed),
    )
    _renumber_honouring_printed_labels(merged, "figure_id", _FIGURE_LABEL_NUMBER_RE)
    remap = {
        old_object_ids[i]: f"figure:{fig.figure_id}"
        for i, fig in enumerate(figures)
        if i not in absorbed
    }
    # An absorbed panel no longer exists, so its receipt entry follows the
    # figure that swallowed it. Targets are never themselves absorbed (both
    # passes skip ``is_target``), so one hop always lands on a survivor.
    for i, target_idx in absorbed.items():
        remap[old_object_ids[i]] = f"figure:{figures[target_idx].figure_id}"
    return merged, remap


def merge_table_continuations(tables: list[PaperTable]) -> list[PaperTable]:
    """Merged-list-only view of :func:`merge_table_continuations_with_remap`.

    Kept for callers that hold no caption receipt and so have nothing to
    repoint at the renumbered survivors.
    """
    return merge_table_continuations_with_remap(tables)[0]


def merge_table_continuations_with_remap(
    tables: list[PaperTable],
) -> tuple[list[PaperTable], dict[str, str]]:
    """Collapse per-page continuation tables into the first page's table.

    A table continues an earlier one when its "Table N" label matches *and*
    its caption carries an explicit trailing "(Continued)" marker. A bare
    repeated caption is not enough: the inline ownership layer already weighs
    evidence this function cannot see (an intervening section heading, for
    one) and deliberately keeps such tables apart.
    Rows are concatenated; a continuation page whose
    header row was promoted to column names by the HTML parser (headerless
    page) has that row restored as data. Survivors are renumbered from 1.

    Returns the merged list plus the ``table:<old>`` → ``table:<new>`` map
    that renumbering implies, with every absorbed continuation page pointing
    at its survivor. The map is empty when nothing merged — unlike the figure
    pass, renumbering here is itself gated on ``merges``, so ids only move
    when a continuation was consumed.
    """
    if not tables:
        return tables, {}
    # Snapshot the pre-merge ids before renumbering overwrites them; they are
    # what the caption receipt was frozen against.
    old_object_ids = {id(table): f"table:{table.table_id}" for table in tables}
    result: list[PaperTable] = []
    by_label: dict[str, PaperTable] = {}
    absorbed: list[tuple[str, PaperTable]] = []  # old object id -> survivor
    merges = 0
    for table in tables:
        caption = (table.caption or "").strip()
        label_match = _TABLE_LABEL_RE.match(caption)
        if label_match is None:
            result.append(table)
            continue
        survivor = by_label.get(label_match.group(1))
        if (
            survivor is not None
            and _CONTINUED_RE.search(caption)
            and _concat_continuation(survivor, table)
        ):
            merges += 1
            absorbed.append((old_object_ids[id(table)], survivor))
            continue
        by_label[label_match.group(1)] = table
        result.append(table)
    if not merges:
        return result, {}
    logger.debug(
        "merge_table_continuations: %d -> %d tables (%d continuation pages merged)",
        len(tables),
        len(result),
        merges,
    )
    _renumber_honouring_printed_labels(result, "table_id", _TABLE_LABEL_RE)
    remap = {old_object_ids[id(table)]: f"table:{table.table_id}" for table in result}
    # A consumed continuation page no longer exists, so its receipt entry
    # follows the table that swallowed its rows. Survivors are never
    # themselves absorbed, so one hop always lands on a live table.
    for old_object_id, survivor in absorbed:
        remap[old_object_id] = f"table:{survivor.table_id}"
    return result, remap


def remap_caption_receipt(
    receipt: CaptionAssignmentReceipt, remap: dict[str, str]
) -> CaptionAssignmentReceipt:
    """Repoint a caption receipt at the ids the mergers just handed out.

    ``_finalize_media`` freezes the receipt against the printed-id
    reservation, and the two mergers above then renumber survivors from 1 —
    so an assignment naming ``figure:12`` dangles in a document whose highest
    ``figure_id`` is 10. An assignment whose object was absorbed is repointed
    at the *survivor* rather than dropped: the receipt exists to say where
    each printed caption ended up, and "absorbed into figure 1" is that
    answer.
    """
    if not remap:
        return receipt
    assignments: list[CaptionAssignment] = []
    for assignment in receipt.assignments:
        # Abstentions (object_id is None) and ids from a float class that did
        # not move are carried through untouched.
        new_object_id = remap.get(assignment.object_id or "")
        if new_object_id is None or new_object_id == assignment.object_id:
            assignments.append(assignment)
            continue
        assignments.append(
            CaptionAssignment(
                assignment.caption_id,
                new_object_id,
                assignment.score,
                assignment.reasons,
                assignment.ambiguous,
            )
        )
    return CaptionAssignmentReceipt(receipt.candidates, tuple(assignments))


def _concat_continuation(survivor: PaperTable, continuation: PaperTable) -> bool:
    """Append *continuation*'s rows to *survivor*; False when shapes differ."""
    s_df, c_df = survivor.df, continuation.df
    if len(s_df.columns) != len(c_df.columns):
        logger.debug(
            "table continuation on page %s not merged: %d vs %d columns",
            continuation.page_number,
            len(c_df.columns),
            len(s_df.columns),
        )
        return False
    s_cols = [str(c) for c in s_df.columns]
    if [str(c) for c in c_df.columns] == s_cols:
        merged = pd.concat([s_df, c_df], ignore_index=True)
    else:
        # Headerless continuation page: the HTML parser promoted its first
        # data row to column names — restore it and align positionally.
        header_row = pd.DataFrame([[str(c) for c in c_df.columns]], columns=s_df.columns)
        body = c_df.copy()
        body.columns = s_df.columns
        merged = pd.concat([s_df, header_row, body], ignore_index=True)
    survivor.df = merged
    survivor.tbl_html = f"{survivor.tbl_html}\n{continuation.tbl_html}"
    survivor.provenance.extend(continuation.provenance)
    return True
