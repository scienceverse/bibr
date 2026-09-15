"""Deterministic document-wide caption ownership tests."""

from __future__ import annotations


def _caption(
    caption_id: str,
    text: str,
    object_type: str,
    page: int | None,
    bbox: tuple[float, float, float, float] | None,
    source_index: int,
):
    from bibr.paper_contents import CaptionCandidate

    return CaptionCandidate(caption_id, text, object_type, page, bbox, source_index)


def _target(
    object_id: str,
    object_type: str,
    page: int | None,
    bbox: tuple[float, float, float, float] | None,
    source_index: int,
):
    from bibr.structure.caption_matcher import CaptionTarget

    return CaptionTarget(object_id, object_type, page, bbox, source_index)


def _assigned(captions, targets, **kwargs):
    from bibr.structure.caption_matcher import assign_captions

    return {item.caption_id: item for item in assign_captions(captions, targets, **kwargs)}


def test_required_caption_models_are_frozen_and_trailing_receipt_is_typed():
    from dataclasses import fields, is_dataclass

    from bibr.paper_contents import (
        CaptionAssignment,
        CaptionAssignmentReceipt,
        CaptionCandidate,
        PaperContents,
    )

    assert is_dataclass(CaptionAssignment)
    assert [field.name for field in fields(CaptionAssignment)] == [
        "caption_id",
        "object_id",
        "score",
        "reasons",
        "ambiguous",
    ]
    assignment = CaptionAssignment("caption:1", None, 0.0, ("unmatched",))
    candidate = CaptionCandidate("caption:1", "Figure 1. Result", "figure", 1, None, 0)
    receipt = CaptionAssignmentReceipt((candidate,), (assignment,))

    assert assignment.__dataclass_params__.frozen is True
    assert candidate.__dataclass_params__.frozen is True
    assert receipt.__dataclass_params__.frozen is True
    content_fields = {field.name: field for field in fields(PaperContents)}
    assert content_fields["caption_assignment_receipt"].default is None


def test_crossing_reading_order_uses_global_geometry_not_fifo():
    captions = [
        _caption("c2", "Figure 2. Lower", "figure", 1, (500, 300, 900, 330), 0),
        _caption("c1", "Figure 1. Upper", "figure", 1, (100, 110, 480, 140), 1),
    ]
    targets = [
        _target("figure:1", "figure", 1, (100, 150, 480, 280), 3),
        _target("figure:2", "figure", 1, (500, 340, 900, 470), 2),
    ]

    assignments = _assigned(captions, targets)

    assert assignments["c1"].object_id == "figure:1"
    assert assignments["c2"].object_id == "figure:2"


def test_caption_type_is_strict_and_each_target_is_used_once():
    captions = [
        _caption("table", "Table 1. Values", "table", 2, (0, 100, 400, 130), 0),
        _caption("figure", "Figure 1. Plot", "figure", 2, (0, 105, 400, 135), 1),
    ]
    targets = [
        _target("figure:1", "figure", 2, (0, 150, 400, 300), 2),
        _target("table:1", "table", 2, (0, 155, 400, 305), 3),
    ]

    assignments = _assigned(captions, targets)

    assert assignments["table"].object_id == "table:1"
    assert assignments["figure"].object_id == "figure:1"
    assert len({item.object_id for item in assignments.values() if item.object_id}) == 2


def test_horizontal_overlap_and_intervening_object_affect_score():
    caption = _caption("c", "Figure 1. Plot", "figure", 1, (20, 100, 420, 130), 0)
    targets = [
        _target("figure:near", "figure", 1, (25, 145, 425, 260), 1),
        _target("figure:wrong-column", "figure", 1, (550, 135, 950, 250), 2),
        _target("figure:past-intervening", "figure", 1, (20, 270, 420, 390), 3),
    ]

    assignment = _assigned([caption], targets)["c"]

    assert assignment.object_id == "figure:near"
    assert "horizontal_overlap" in assignment.reasons
    assert "intervening_object:0" in assignment.reasons


def test_adjacent_page_is_bounded_and_missing_geometry_abstains():
    captions = [
        _caption("adjacent", "Figure 1. Plot", "figure", 1, (0, 900, 500, 940), 0),
        _caption("missing", "Figure 2. Plot", "figure", 4, None, 5),
    ]
    targets = [
        _target("figure:1", "figure", 2, (0, 20, 500, 300), 1),
        _target("figure:2", "figure", 4, None, 6),
        _target("figure:3", "figure", 4, None, 7),
        _target("figure:far", "figure", 4, (0, 20, 500, 300), 8),
    ]

    assignments = _assigned(captions, targets)

    assert assignments["adjacent"].object_id == "figure:1"
    assert assignments["missing"].object_id is None
    assert "missing_geometry" in assignments["missing"].reasons


def test_adjacent_page_edge_cannot_cross_an_unrelated_same_type_object():
    captions = [
        _caption("c1", "Figure 1. First", "figure", 1, (0, 840, 500, 870), 0),
        _caption("c2", "Figure 2. Next page", "figure", 1, (0, 800, 500, 830), 1),
    ]
    targets = [
        _target("figure:1", "figure", 1, (0, 880, 500, 990), 2),
        _target("figure:2", "figure", 2, (0, 10, 500, 400), 3),
    ]

    assignments = _assigned(captions, targets)

    assert assignments["c1"].object_id == "figure:1"
    assert assignments["c2"].object_id is None


def test_unique_explicit_number_with_missing_geometry_abstains():
    caption = _caption("c", "Figure 7. Plot", "figure", None, None, 0)
    target = _target("figure:physical", "figure", None, None, 1)

    assignment = _assigned([caption], [target])["c"]

    assert assignment.object_id is None
    assert "missing_geometry" in assignment.reasons


def test_unique_explicit_number_does_not_override_contradictory_complete_geometry():
    caption = _caption("c", "Figure 7. Plot", "figure", 1, (0, 20, 200, 60), 0)
    target = _target("figure:physical", "figure", 1, (800, 800, 990, 990), 1)

    assignment = _assigned([caption], [target])["c"]

    assert assignment.object_id is None
    assert "unique_explicit_number" not in assignment.reasons


def test_ambiguity_margin_preserves_candidate_as_unassigned():
    caption = _caption("c", "Figure 1. Plot", "figure", 1, (0, 100, 500, 130), 0)
    targets = [
        _target("figure:left", "figure", 1, (0, 150, 500, 300), 1),
        _target("figure:right", "figure", 1, (0, 150, 500, 300), 2),
    ]

    assignment = _assigned([caption], targets, ambiguity_margin=0.1)["c"]

    assert assignment.object_id is None
    assert assignment.ambiguous is True
    assert "ambiguous" in assignment.reasons


def test_losing_caption_retains_target_contention_evidence():
    captions = [
        _caption("winner", "Figure 1. Full caption", "figure", 1, (0, 310, 500, 340), 1),
        _caption("loser", "Figure 1. Alternate caption", "figure", 1, (0, 300, 500, 330), 2),
    ]
    target = _target("figure:1", "figure", 1, (0, 100, 500, 290), 0)

    assignments = _assigned(captions, [target])
    unassigned = next(item for item in assignments.values() if item.object_id is None)

    assert unassigned.ambiguous is True
    assert "target_contention" in unassigned.reasons
