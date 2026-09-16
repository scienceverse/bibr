"""A deliberate selected-record title abstention cannot trigger a guess."""

from types import SimpleNamespace

import pytest

from bibr.pipeline.stages.post_parse import (
    _resolve_detected_title_fallback,
    _resolve_selected_title,
)
from bibr.validation import ValidationIssue


@pytest.mark.parametrize("marker", ["title_recovery_ambiguous", "title_recovery_unverified"])
@pytest.mark.parametrize("fallback", [_resolve_selected_title, _resolve_detected_title_fallback])
def test_explicit_title_abstention_blocks_null_fallback(marker, fallback):
    contents = SimpleNamespace(detected_title="A different article title")
    metadata = SimpleNamespace(title="", journal=None, publisher=None)
    issues = [
        ValidationIssue(
            "VAL_TITLE_UNGROUNDED", "warning", "Title declined", evidence_ids=(f"reason:{marker}",)
        )
    ]

    assert not fallback(contents, metadata, validation_issue_sink=issues)
    assert metadata.title == ""
    assert len(issues) == 1


def test_printed_primary_variant_is_not_erased_by_an_earlier_abstention():
    contents = SimpleNamespace(detected_title="A different article title")
    metadata = SimpleNamespace(title="Selected printed original", journal=None, publisher=None)
    issues = [
        ValidationIssue(
            "VAL_TITLE_UNGROUNDED",
            "warning",
            "Title declined",
            evidence_ids=("reason:title_recovery_ambiguous",),
        )
    ]

    assert not _resolve_detected_title_fallback(contents, metadata, validation_issue_sink=issues)
    assert metadata.title == "Selected printed original"
