"""Pure deterministic scoring and document-wide caption ownership."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from bibr.paper_contents import CaptionAssignment, CaptionCandidate

_NUMBERED_CAPTION_RE = re.compile(
    r"^(?:fig(?:ure)?\.?|table)\s+(?P<number>\d+|[ivxlcdm]+)\b", re.IGNORECASE
)


def _roman_value(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = previous = 0
    for character in reversed(value.casefold()):
        current = values.get(character)
        if current is None:
            return None
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total or None


@dataclass(frozen=True)
class CaptionTarget:
    """A physical/logical media object eligible to own one caption."""

    object_id: str
    object_type: str
    page_number: int | None
    bbox: tuple[float, float, float, float] | None
    source_index: int


@dataclass(frozen=True)
class _ScoredEdge:
    score: float
    reasons: tuple[str, ...]


def _horizontal_overlap(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    denominator = max(1.0, min(left[2] - left[0], right[2] - right[0]))
    return overlap / denominator


def _vertical_overlap(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    overlap = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    denominator = max(1.0, min(left[3] - left[1], right[3] - right[1]))
    return overlap / denominator


def _horizontal_gap(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    if left[2] <= right[0]:
        return right[0] - left[2]
    if right[2] <= left[0]:
        return left[0] - right[2]
    return 0.0


def _vertical_gap(
    caption: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    page_delta: int,
) -> tuple[float, str]:
    if page_delta == 1:
        return max(0.0, 1000.0 - caption[3]) + max(0.0, target[1]), "caption_before"
    if page_delta == -1:
        return max(0.0, 1000.0 - target[3]) + max(0.0, caption[1]), "caption_after"
    if caption[3] <= target[1]:
        return target[1] - caption[3], "caption_before"
    if target[3] <= caption[1]:
        return caption[1] - target[3], "caption_after"
    return 0.0, "vertical_overlap"


def _intervening_count(
    caption: CaptionCandidate, target: CaptionTarget, targets: tuple[CaptionTarget, ...]
) -> int:
    if caption.bbox is None or target.bbox is None or caption.page_number != target.page_number:
        return 0
    caption_y = (caption.bbox[1] + caption.bbox[3]) / 2
    target_y = (target.bbox[1] + target.bbox[3]) / 2
    low, high = sorted((caption_y, target_y))
    count = 0
    for other in targets:
        if (
            other.object_id == target.object_id
            or other.object_type != target.object_type
            or other.page_number != target.page_number
            or other.bbox is None
        ):
            continue
        other_y = (other.bbox[1] + other.bbox[3]) / 2
        if low < other_y < high and _horizontal_overlap(caption.bbox, other.bbox) >= 0.2:
            count += 1
    return count


def _score_edge(
    caption: CaptionCandidate,
    target: CaptionTarget,
    targets: tuple[CaptionTarget, ...],
    *,
    number_offset: int | None = 0,
) -> _ScoredEdge | None:
    """Score one caption-target edge.

    The printed number is compared with the target's id plus *number_offset*;
    with ``None`` it is not compared at all (geometry only).
    """
    if caption.object_type != target.object_type:
        return None
    if (
        caption.page_number is None
        or target.page_number is None
        or caption.bbox is None
        or target.bbox is None
    ):
        return None
    page_delta = target.page_number - caption.page_number
    if abs(page_delta) > 1:
        return None
    overlap = _horizontal_overlap(caption.bbox, target.bbox)
    caption_width = caption.bbox[2] - caption.bbox[0]
    caption_height = caption.bbox[3] - caption.bbox[1]
    rotated_overlap = _vertical_overlap(caption.bbox, target.bbox)
    horizontal_gap = _horizontal_gap(caption.bbox, target.bbox)
    rotated_side_caption = bool(
        page_delta == 0
        and overlap < 0.2
        and caption_width <= 80
        and caption_height >= 40
        and rotated_overlap >= 0.5
        and horizontal_gap <= 80
    )
    if overlap < 0.2 and not rotated_side_caption:
        return None
    if page_delta:
        for other in targets:
            if (
                other.object_id == target.object_id
                or other.object_type != target.object_type
                or other.bbox is None
            ):
                continue
            overlaps_caption = _horizontal_overlap(caption.bbox, other.bbox) >= 0.2
            overlaps_target = _horizontal_overlap(target.bbox, other.bbox) >= 0.2
            blocks_caption_page = (
                other.page_number == caption.page_number
                and overlaps_caption
                and (
                    (page_delta == 1 and other.bbox[1] >= caption.bbox[3])
                    or (page_delta == -1 and other.bbox[3] <= caption.bbox[1])
                )
            )
            blocks_target_page = (
                other.page_number == target.page_number
                and overlaps_target
                and (
                    (page_delta == 1 and other.bbox[3] <= target.bbox[1])
                    or (page_delta == -1 and other.bbox[1] >= target.bbox[3])
                )
            )
            if blocks_caption_page or blocks_target_page:
                return None
    if rotated_side_caption:
        gap, direction = horizontal_gap, "rotated_side_caption"
        overlap = rotated_overlap
    else:
        gap, direction = _vertical_gap(caption.bbox, target.bbox, page_delta)
    if gap > 500:
        return None
    intervening = _intervening_count(caption, target, targets)
    score = (
        (3.0 if page_delta == 0 else 0.8)
        + 3.0 * overlap
        + max(0.0, 2.0 - gap / 250.0)
        + 0.3
        - 0.75 * intervening
    )
    reasons = (
        "same_page" if page_delta == 0 else "adjacent_page",
        "vertical_overlap" if rotated_side_caption else "horizontal_overlap",
        direction,
        f"intervening_object:{intervening}",
    )
    if number_match := _NUMBERED_CAPTION_RE.match(caption.text.strip()):
        # A numbered caption is stronger ownership evidence than a nearby
        # unnumbered internal heading. Physical object IDs can include
        # unrelated decorative media, so number mismatch remains a modest
        # penalty rather than cancelling that evidence entirely.
        score += 1.5
        reasons += ("explicit_number",)
        target_suffix = target.object_id.rsplit(":", 1)[-1]
        caption_number = _roman_value(number_match.group("number"))
        target_number = _roman_value(target_suffix)
        if number_offset is not None and caption_number is not None and target_number is not None:
            if caption_number == target_number + number_offset:
                score += 2.5
                reasons += ("explicit_number_match",)
            else:
                score -= 1.0
                reasons += ("explicit_number_mismatch",)
    return _ScoredEdge(round(score, 6), reasons)


def _maximum_weight_assignment(rows: list[dict[int, float]], column_count: int) -> list[int]:
    """Return the selected column for each row using deterministic Hungarian matching.

    *rows* holds each row's weights by column; every other cell weighs zero.
    This is the former pure-Python column loop run on whole numpy rows: the
    same float operations in the same order, a strict ``<`` per relaxation
    and the lowest column on a tie (``argmin`` returns the first minimum),
    so it selects exactly what the loop did, without its per-cell Python
    cost or a dense rows x columns matrix.
    """

    row_count = len(rows)
    if row_count == 0:
        return []
    # The caller appends one zero-weight dummy column per row, so rows <= columns.
    # Column 0 is the algorithm's virtual start column.
    width = column_count + 1
    # Minimise the negative weight (-0.0 off the edges, as negating a zero
    # cell gave); stable row/column order makes equal-weight outcomes
    # deterministic.
    row_columns = [np.fromiter(row, dtype=np.intp, count=len(row)) + 1 for row in rows]
    row_costs = [-np.fromiter(row.values(), dtype=np.float64, count=len(row)) for row in rows]
    potentials_rows = np.zeros(row_count + 1)
    potentials_cols = np.zeros(width)
    matched_row = np.zeros(width, dtype=np.intp)
    predecessor = np.zeros(width, dtype=np.intp)
    cost = np.empty(width)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        min_value = np.full(width, np.inf)
        used = np.zeros(width, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            cost.fill(-0.0)
            cost[row_columns[current_row - 1]] = row_costs[current_row - 1]
            reduced = cost - potentials_rows[current_row] - potentials_cols
            improved = ~used & (reduced < min_value)
            np.copyto(min_value, reduced, where=improved)
            np.copyto(predecessor, column, where=improved)
            candidates = np.where(used, np.inf, min_value)
            next_column = int(candidates.argmin())
            delta = candidates[next_column]
            potentials_rows[matched_row[used]] += delta
            np.subtract(potentials_cols, delta, out=potentials_cols, where=used)
            np.subtract(min_value, delta, out=min_value, where=~used)
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = int(predecessor[column])
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    selected = [-1] * row_count
    for column in np.flatnonzero(matched_row[1:]) + 1:
        selected[int(matched_row[column]) - 1] = int(column) - 1
    return selected


def _targets_near_pages(
    captions: tuple[CaptionCandidate, ...],
    targets: tuple[CaptionTarget, ...],
    *,
    adjacent: bool,
) -> dict[int | None, tuple[tuple[int, ...], tuple[CaptionTarget, ...]]]:
    """Per caption page, the targets on it (and, if *adjacent*, the pages either side).

    ``_score_edge`` rejects a target more than one page away and only consults
    other targets on the caption's or the target's page, so scoring against
    this slice (indices and targets, in *targets* order) gives the same edges
    as scoring against every target, without a captions x targets scan.
    """
    by_page: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        if target.page_number is not None:
            by_page[target.page_number].append(index)
    near: dict[int | None, tuple[tuple[int, ...], tuple[CaptionTarget, ...]]] = {}
    for caption in captions:
        page = caption.page_number
        if page is None or page in near:
            continue
        pages = (page - 1, page, page + 1) if adjacent else (page,)
        indices = tuple(sorted(index for other in pages for index in by_page.get(other, ())))
        near[page] = (indices, tuple(targets[index] for index in indices))
    return near


def _number_offsets(
    captions: tuple[CaptionCandidate, ...], targets: tuple[CaptionTarget, ...]
) -> dict[tuple[int | None, str], int]:
    """Per page and kind, how far printed numbers run ahead of provisional ids.

    Provisional ids count every float, so a figure the parser keeps but the
    paper does not number shifts every later one. Each numbered caption whose
    geometrically best same-page target is clear (by 1.0) and its own (no other
    caption's best) votes with its number minus that target's id. A page's
    single most common vote is the offset the number bonus applies there; a
    tie, or no vote, leaves it at zero. A leftover offset of one would
    otherwise pull each caption onto its neighbour's figure.
    """
    votes: dict[tuple[int | None, str], list[tuple[int, str]]] = defaultdict(list)
    same_page = _targets_near_pages(captions, targets, adjacent=False)
    for caption in captions:
        number_match = _NUMBERED_CAPTION_RE.match(caption.text.strip())
        caption_number = _roman_value(number_match.group("number")) if number_match else None
        if caption_number is None:
            continue
        _indices, page_targets = same_page.get(caption.page_number, ((), ()))
        scored = sorted(
            (
                (edge.score, target.object_id)
                for target in page_targets
                if (edge := _score_edge(caption, target, page_targets, number_offset=None))
                is not None
            ),
            key=lambda item: -item[0],
        )
        if not scored or (len(scored) > 1 and scored[0][0] - scored[1][0] < 1.0):
            continue
        target_number = _roman_value(scored[0][1].rsplit(":", 1)[-1])
        if target_number is not None:
            votes[(caption.page_number, caption.object_type)].append(
                (caption_number - target_number, scored[0][1])
            )
    offsets: dict[tuple[int | None, str], int] = {}
    for key, page_votes in votes.items():
        best_of = Counter(target_id for _offset, target_id in page_votes)
        ranked = Counter(
            offset for offset, target_id in page_votes if best_of[target_id] == 1
        ).most_common(2)
        if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            offsets[key] = ranked[0][0]
    return offsets


def assign_captions(
    captions: list[CaptionCandidate] | tuple[CaptionCandidate, ...],
    targets: list[CaptionTarget] | tuple[CaptionTarget, ...],
    *,
    ambiguity_margin: float = 0.25,
    same_page_only: bool = False,
) -> tuple[CaptionAssignment, ...]:
    """Assign captions globally with strict type compatibility and geometry abstention.

    With *same_page_only*, adjacent-page edges are never built. A caller that
    refuses every cross-page assignment afterwards must pass it: an edge it
    would veto still wins targets in the solve and narrows the abstention
    margin, so the same-page caption it beat is lost too.
    """

    ordered_captions = tuple(
        sorted(captions, key=lambda item: (item.source_index, item.caption_id))
    )
    ordered_targets = tuple(sorted(targets, key=lambda item: (item.source_index, item.object_id)))
    offsets = _number_offsets(ordered_captions, ordered_targets)
    near = _targets_near_pages(ordered_captions, ordered_targets, adjacent=not same_page_only)
    types_missing_geometry = {
        target.object_type
        for target in ordered_targets
        if target.page_number is None or target.bbox is None
    }
    edges: list[dict[int, _ScoredEdge]] = []
    forced_ambiguous: set[int] = set()
    unmatched_reasons: dict[int, tuple[str, ...]] = {}
    for caption_index, caption in enumerate(ordered_captions):
        near_indices, near_targets = near.get(caption.page_number, ((), ()))
        row = {
            target_index: edge
            for target_index, target in zip(near_indices, near_targets, strict=True)
            if (
                edge := _score_edge(
                    caption,
                    target,
                    near_targets,
                    number_offset=offsets.get((target.page_number, target.object_type), 0),
                )
            )
            is not None
        }
        missing_geometry = (
            caption.page_number is None
            or caption.bbox is None
            or caption.object_type in types_missing_geometry
        )
        if not row and missing_geometry:
            unmatched_reasons[caption_index] = ("missing_geometry", "unmatched")
        elif not row:
            unmatched_reasons[caption_index] = ("unmatched",)
        ranked = sorted((edge.score for edge in row.values()), reverse=True)
        if len(ranked) >= 2 and ranked[0] - ranked[1] <= ambiguity_margin:
            forced_ambiguous.add(caption_index)
            row = {}
        edges.append(row)

    selected = _maximum_weight_assignment(
        [{column: edge.score for column, edge in row.items()} for row in edges],
        len(ordered_targets) + len(ordered_captions),
    )
    assignments: list[CaptionAssignment] = []
    for caption_index, caption in enumerate(ordered_captions):
        column = selected[caption_index]
        edge = edges[caption_index].get(column)
        if caption_index in forced_ambiguous:
            assignments.append(
                CaptionAssignment(caption.caption_id, None, 0.0, ("ambiguous",), ambiguous=True)
            )
        elif edge is None or edge.score <= 0:
            if edges[caption_index]:
                assignments.append(
                    CaptionAssignment(
                        caption.caption_id,
                        None,
                        0.0,
                        ("target_contention", "ambiguous"),
                        ambiguous=True,
                    )
                )
                continue
            assignments.append(
                CaptionAssignment(
                    caption.caption_id,
                    None,
                    0.0,
                    unmatched_reasons.get(caption_index, ("unmatched",)),
                )
            )
        else:
            assignments.append(
                CaptionAssignment(
                    caption.caption_id,
                    ordered_targets[column].object_id,
                    edge.score,
                    edge.reasons,
                )
            )
    return tuple(sorted(assignments, key=lambda item: item.caption_id))
