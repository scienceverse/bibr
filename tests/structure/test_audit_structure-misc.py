"""Audit regressions: malformed region boxes and caption-assignment cost."""

from __future__ import annotations

import math
import random
import tracemalloc
from collections import Counter, defaultdict

import pytest

from bibr.paper_contents import CaptionAssignment, CaptionCandidate
from bibr.structure import caption_matcher
from bibr.structure.caption_matcher import (
    _NUMBERED_CAPTION_RE,
    CaptionTarget,
    _roman_value,
    _score_edge,
    _ScoredEdge,
    assign_captions,
)

_TABLE = "| Group | Mean |\n|---|---|\n| A | 1.0 |"


def _region(index, label, content, bbox):
    return {"index": index, "label": label, "content": content, "bbox_2d": bbox}


def _parse(pages):
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(pages)
    contents = parser.parse()
    return (
        [section.header for section in parser.sections],
        [table.caption for table in contents.tables],
        parser._detected_title,
    )


# --- malformed bbox_2d ------------------------------------------------------

_MALFORMED_LAYOUTS = {
    "fragment_after_bare_table_label": [
        _region(0, "figure_title", "Table 2", [100, 100, 900, 120]),
        _region(1, "text", "Descriptive statistics by group", "BOX"),
        _region(2, "table", _TABLE, [100, 200, 900, 500]),
    ],
    "bare_table_label": [
        _region(0, "figure_title", "Table 2", "BOX"),
        _region(1, "text", "Descriptive statistics by group", [100, 140, 900, 170]),
        _region(2, "table", _TABLE, [100, 200, 900, 500]),
    ],
    "table_confirming_the_fragment": [
        _region(0, "figure_title", "Table 2", [100, 100, 900, 120]),
        _region(1, "text", "Descriptive statistics by group", [100, 140, 900, 170]),
        _region(2, "table", _TABLE, "BOX"),
    ],
    "split_title_continuation": [
        _region(0, "doc_title", "A Study of", [100, 100, 900, 120]),
        _region(1, "doc_title", "Many Things", "BOX"),
        _region(2, "text", "Body text.", [100, 300, 900, 400]),
    ],
    "title_after_kicker": [
        _region(0, "doc_title", "Research Article", [100, 100, 900, 120]),
        _region(1, "doc_title", "Many Things", "BOX"),
        _region(2, "text", "Body text.", [100, 300, 900, 400]),
    ],
}


def _with_box(layout, box):
    return [
        [
            {**region, "bbox_2d": box if region["bbox_2d"] == "BOX" else region["bbox_2d"]}
            for region in layout
        ]
    ]


@pytest.mark.parametrize("box", [[], [100], [100, 140, 900]])
@pytest.mark.parametrize("layout", sorted(_MALFORMED_LAYOUTS))
def test_malformed_bbox_parses_like_a_missing_one(layout, box):
    """A short ``bbox_2d`` (bad OCR JSON, corrupted cache) used to raise
    IndexError in ``_is_bbox_nearby`` and abort the whole parse; it now
    counts as missing, as it already did for the region's provenance."""
    malformed = _parse(_with_box(_MALFORMED_LAYOUTS[layout], box))

    assert malformed == _parse(_with_box(_MALFORMED_LAYOUTS[layout], None))


def test_short_boxes_keep_the_composed_caption_and_title():
    _headers, captions, _title = _parse(_with_box(_MALFORMED_LAYOUTS["bare_table_label"], [1, 2]))
    assert captions == ["Table 2 Descriptive statistics by group"]

    layout = _MALFORMED_LAYOUTS["split_title_continuation"]
    _headers, _captions, title = _parse(_with_box(layout, [1, 2]))
    assert title == "A Study of Many Things"


def test_is_bbox_nearby_treats_malformed_boxes_as_missing():
    from bibr.structure.parse_text import TextHandlersMixin

    nearby = TextHandlersMixin._is_bbox_nearby
    far = [100, 900, 900, 950]
    for malformed in ([], [100], [100, 100, 900]):
        assert nearby(malformed, 1, far, 1) is True
        assert nearby(far, 1, malformed, 1) is True
        assert nearby(malformed, 1, far, 2) is False
    # Well-formed boxes still measure the vertical gap; extra values are ignored.
    assert nearby([100, 100, 900, 120], 1, far, 1) is False
    assert nearby([100, 100, 900, 120, 7], 1, [100, 130, 900, 150], 1) is True


def test_merge_bboxes_ignores_a_malformed_box():
    from bibr.structure.parse_text import TextHandlersMixin

    merge = TextHandlersMixin._merge_bboxes
    assert merge([100, 100, 900], [10, 20, 30, 40]) == [10.0, 20.0, 30.0, 40.0]
    assert merge([10, 20, 30, 40], []) == [10.0, 20.0, 30.0, 40.0]
    assert merge([], None) is None
    assert merge([0, 50, 10, 60], [5, 0, 20, 55]) == [0.0, 0.0, 20.0, 60.0]


# --- caption assignment: reference implementation ---------------------------
# The pre-audit global solve, kept verbatim (dense matrix, pure-Python
# Hungarian, every caption scored against every target) as the oracle the
# faster implementation must match exactly, ties included.


def _reference_hungarian(weights: list[list[float]]) -> list[int]:
    row_count = len(weights)
    if row_count == 0:
        return []
    column_count = len(weights[0])
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


def _reference_number_offsets(captions, targets):
    votes = defaultdict(list)
    for caption in captions:
        number_match = _NUMBERED_CAPTION_RE.match(caption.text.strip())
        caption_number = _roman_value(number_match.group("number")) if number_match else None
        if caption_number is None:
            continue
        scored = sorted(
            (
                (edge.score, target.object_id)
                for target in targets
                if target.page_number == caption.page_number
                and (edge := _score_edge(caption, target, targets, number_offset=None)) is not None
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
    offsets = {}
    for key, page_votes in votes.items():
        best_of = Counter(target_id for _offset, target_id in page_votes)
        ranked = Counter(
            offset for offset, target_id in page_votes if best_of[target_id] == 1
        ).most_common(2)
        if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            offsets[key] = ranked[0][0]
    return offsets


def _reference_assign_captions(captions, targets, *, ambiguity_margin=0.25, same_page_only=False):
    ordered_captions = tuple(
        sorted(captions, key=lambda item: (item.source_index, item.caption_id))
    )
    ordered_targets = tuple(sorted(targets, key=lambda item: (item.source_index, item.object_id)))
    offsets = _reference_number_offsets(ordered_captions, ordered_targets)
    edges = []
    forced_ambiguous = set()
    unmatched_reasons = {}
    for caption_index, caption in enumerate(ordered_captions):
        row = {
            target_index: edge
            for target_index, target in enumerate(ordered_targets)
            if not (same_page_only and caption.page_number != target.page_number)
            and (
                edge := _score_edge(
                    caption,
                    target,
                    ordered_targets,
                    number_offset=offsets.get((target.page_number, target.object_type), 0),
                )
            )
            is not None
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
    selected = _reference_hungarian(weights)
    assignments = []
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


# --- caption assignment: exact equivalence ----------------------------------


def _random_weight_rows(rng):
    """Sparse weight rows built to tie: additive scores, repeated values,
    zero/negative/NaN weights and empty rows that occupy low columns."""
    row_count, target_count = rng.randint(0, 12), rng.randint(0, 12)
    density = rng.choice([0.1, 0.3, 0.6, 1.0])
    additive = rng.random() < 0.4
    row_bias = [rng.choice([0.0, 0.5, 1.0, 2.0]) for _ in range(row_count)]
    column_bias = [rng.choice([0.0, 0.5, 1.0, 2.0]) for _ in range(target_count)]
    values = [0.0, -0.0, 1.0, 2.0, 3.0, 0.5, -1.0, 5.33, 5.83, 1e-7]
    rows = []
    for row in range(row_count):
        weights = {}
        if rng.random() >= 0.2:
            for column in range(target_count):
                if rng.random() >= density:
                    continue
                if additive:
                    weights[column] = row_bias[row] + column_bias[column]
                elif rng.random() < 0.05:
                    weights[column] = math.nan
                elif rng.random() < 0.7:
                    weights[column] = rng.choice(values)
                else:
                    weights[column] = round(rng.uniform(-2, 9), 6)
        rows.append(weights)
    return rows, target_count + row_count


@pytest.mark.parametrize("seed", range(4))
def test_solver_selects_exactly_what_the_reference_selected(seed):
    rng = random.Random(seed)  # noqa: S311 - deterministic test fixture, not crypto
    for _ in range(400):
        rows, width = _random_weight_rows(rng)
        dense = [[row.get(column, 0.0) for column in range(width)] for row in rows]

        assert caption_matcher._maximum_weight_assignment(rows, width) == _reference_hungarian(
            dense
        ), (rows, width)


def test_solver_keeps_tie_breaks_that_depend_on_unrelated_rows():
    """Equal-weight matchings exist here, and which one the solve picks
    depends on the edgeless rows parked on low columns: why the solve is
    not split per connected component."""
    rows = [
        {},
        {},
        {0: 5.0, 1: 6.0},
        {},
        {0: 3.0, 1: 5.0, 4: 4.0},
        {0: 5.0, 1: 6.0, 2: 4.0, 3: 6.0, 4: 5.0, 6: 7.0, 7: 3.0},
        {},
    ]
    width = 8 + len(rows)
    dense = [[row.get(column, 0.0) for column in range(width)] for row in rows]

    selected = caption_matcher._maximum_weight_assignment(rows, width)

    assert selected == _reference_hungarian(dense)
    assert [selected[2], selected[4], selected[5]] == [1, 4, 6]


def _random_layout(rng):
    captions, targets = [], []
    source = 0
    for page in range(1, rng.randint(1, 5) + 1):
        for _ in range(rng.randint(0, 8)):
            kind = rng.choice(["figure", "figure", "table"])
            x = rng.choice([0, 0, 250, 500, 940])
            y = rng.choice([0, 100, 200, 300, 400, 500, 600, 700, 900])
            box = (x, y, x + rng.choice([60, 500]), y + rng.choice([20, 50, 100, 300]))
            page_number = page if rng.random() > 0.05 else None
            bbox = box if rng.random() > 0.05 else None
            if rng.random() < 0.5:
                suffix = rng.choice([len(targets) + 1, "x", "iv"])
                targets.append((f"{kind}:{suffix}", kind, page_number, bbox, source))
            else:
                number = rng.randint(1, 6)
                text = rng.choice([f"Figure {number}. x", f"Table {number}. x", "Panel"])
                captions.append(
                    (f"caption:{len(captions) + 1}", text, kind, page_number, bbox, source)
                )
            source += rng.choice([0, 1, 1])
    return captions, targets


@pytest.mark.parametrize("seed", range(4))
def test_assign_captions_matches_the_reference_global_solve(seed):
    rng = random.Random(seed)  # noqa: S311 - deterministic test fixture, not crypto
    for _ in range(80):
        captions, targets = _random_layout(rng)
        options = {
            "same_page_only": rng.random() < 0.3,
            "ambiguity_margin": rng.choice([0.25, 0.0, 1.0]),
        }
        caption_items = [CaptionCandidate(*caption) for caption in captions]
        target_items = [CaptionTarget(*target) for target in targets]

        assert assign_captions(caption_items, target_items, **options) == (
            _reference_assign_captions(caption_items, target_items, **options)
        ), (captions, targets, options)


# --- caption assignment: bounded work ----------------------------------------


def _two_per_page(count):
    captions, targets = [], []
    for index in range(count):
        page = index // 2 + 1
        y = 100 if index % 2 == 0 else 550
        targets.append(
            CaptionTarget(f"figure:{index + 1}", "figure", page, (100, y, 900, y + 300), 2 * index)
        )
        captions.append(
            CaptionCandidate(
                f"caption:{index + 1:05d}",
                f"Figure {index + 1}. A caption",
                "figure",
                page,
                (100, y + 310, 900, y + 340),
                2 * index + 1,
            )
        )
    return captions, targets


def test_captions_are_scored_only_against_targets_on_nearby_pages(monkeypatch):
    """Every caption used to be scored against every target in the document
    (1,000 captions: 1M edge scores and 5.8 s); only the targets on its own
    and the adjacent pages can own it."""
    calls = 0

    def counting_score_edge(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _score_edge(*args, **kwargs)

    monkeypatch.setattr(caption_matcher, "_score_edge", counting_score_edge)
    captions, targets = _two_per_page(300)

    assignments = assign_captions(captions, targets)

    assert [item.object_id for item in assignments] == [f"figure:{i + 1}" for i in range(300)]
    # <= 6 nearby targets per caption, plus <= 2 for the number-offset votes.
    assert calls <= 8 * len(captions)


def test_solve_builds_no_dense_captions_by_columns_matrix():
    """The solve used to materialise a captions x (targets + captions) list
    matrix (800 captions: 1.28M cells, over 10 MB) before matching."""
    captions, targets = _two_per_page(800)

    tracemalloc.start()
    try:
        assignments = assign_captions(captions, targets)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert sum(item.object_id is not None for item in assignments) == 800
    assert peak < 5_000_000
