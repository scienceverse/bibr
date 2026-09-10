"""Pure deterministic scoring and document-wide caption ownership."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
) -> _ScoredEdge | None:
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
        if caption_number is not None and target_number is not None:
            if caption_number == target_number:
                score += 2.5
                reasons += ("explicit_number_match",)
            else:
                score -= 1.0
                reasons += ("explicit_number_mismatch",)
    return _ScoredEdge(round(score, 6), reasons)


def _maximum_weight_assignment(weights: list[list[float]]) -> list[int]:
    """Return the selected column for each row using deterministic Hungarian matching."""

    row_count = len(weights)
    if row_count == 0:
        return []
    column_count = len(weights[0])
    # The caller appends one zero-weight dummy column per row, so rows <= columns.
    potentials_rows = [0.0] * (row_count + 1)
    potentials_cols = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        min_value = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                # Minimise the negative weight; stable row/column iteration
                # makes equal-weight outcomes deterministic.
                cost = -weights[current_row - 1][candidate_column - 1]
                reduced = cost - potentials_rows[current_row] - potentials_cols[candidate_column]
                if reduced < min_value[candidate_column]:
                    min_value[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if min_value[candidate_column] < delta:
                    delta = min_value[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    potentials_rows[matched_row[candidate_column]] += delta
                    potentials_cols[candidate_column] -= delta
                else:
                    min_value[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    selected = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            selected[matched_row[column] - 1] = column - 1
    return selected


def assign_captions(
    captions: list[CaptionCandidate] | tuple[CaptionCandidate, ...],
    targets: list[CaptionTarget] | tuple[CaptionTarget, ...],
    *,
    ambiguity_margin: float = 0.25,
) -> tuple[CaptionAssignment, ...]:
    """Assign captions globally with strict type compatibility and geometry abstention."""

    ordered_captions = tuple(
        sorted(captions, key=lambda item: (item.source_index, item.caption_id))
    )
    ordered_targets = tuple(sorted(targets, key=lambda item: (item.source_index, item.object_id)))
    edges: list[dict[int, _ScoredEdge]] = []
    forced_ambiguous: set[int] = set()
    unmatched_reasons: dict[int, tuple[str, ...]] = {}
    for caption_index, caption in enumerate(ordered_captions):
        row = {
            target_index: edge
            for target_index, target in enumerate(ordered_targets)
            if (edge := _score_edge(caption, target, ordered_targets)) is not None
        }
        compatible = [
            target for target in ordered_targets if target.object_type == caption.object_type
        ]
        missing_geometry = (
            caption.page_number is None
            or caption.bbox is None
            or any(target.page_number is None or target.bbox is None for target in compatible)
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

    weights = [
        [
            edges[row].get(column, _ScoredEdge(0.0, ())).score
            for column in range(len(ordered_targets))
        ]
        + [0.0] * len(ordered_captions)
        for row in range(len(ordered_captions))
    ]
    selected = _maximum_weight_assignment(weights)
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
