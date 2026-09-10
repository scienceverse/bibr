"""Task 3c fix round 1, Finding 2: bibr's ``augment_adjacent_features`` is now
the single implementation -- both the default (production) path and the
training-eval-only ``patterns=`` override run through this one function.
These tests pin that the override actually changes what gets computed, so a
future edit can't silently make ``patterns`` a dead parameter again."""

import re

from bibr.extract.geom_adjacent_features import (
    ADJACENT_FEATURE_KEYS,
    CURRENT_PATTERNS,
    AdjacentPatterns,
    augment_adjacent_features,
)
from bibr.ocr.ref_geometry import LineRecord


def _ln(text, x0, page=1, y=700.0):
    return LineRecord(text, page, x0, y, x0 + 400.0, y - 10.0, 10.0)


def test_augment_adjacent_features_defaults_to_current_patterns():
    lines = [_ln("References", 72.0), _ln("1. Doe J. First reference.", 72.0, y=688.0)]
    rows = augment_adjacent_features([{}, {}], lines)

    assert set(rows[0]) == set(ADJACENT_FEATURE_KEYS)
    assert rows[1]["looks_numbered_start"] == 1
    assert rows[0]["prev_looks_reference_header"] == 0
    assert rows[1]["prev_looks_reference_header"] == 1


def test_augment_adjacent_features_reject_line_row_length_mismatch():
    try:
        augment_adjacent_features([{}], [])
    except ValueError as exc:
        assert "feature/line length mismatch" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected a ValueError")


def test_patterns_override_actually_changes_the_computed_flags():
    # A pattern triple that never matches anything -- if `patterns` were a
    # dead parameter (e.g. the loop body still reading the module-level
    # regexes directly), this override would be silently ignored and the
    # assertions below would fail.
    never_matches = AdjacentPatterns(
        numbered_start=re.compile(r"(?!)"),
        author_year=re.compile(r"(?!)"),
        ref_header=re.compile(r"(?!)"),
    )
    lines = [_ln("References", 72.0), _ln("1. Doe J. First reference.", 72.0, y=688.0)]

    default_rows = augment_adjacent_features([{}, {}], lines)
    overridden_rows = augment_adjacent_features([{}, {}], lines, patterns=never_matches)

    assert default_rows[1]["looks_numbered_start"] == 1
    assert overridden_rows[1]["looks_numbered_start"] == 0
    assert default_rows[1]["prev_looks_reference_header"] == 1
    assert overridden_rows[1]["prev_looks_reference_header"] == 0
    # looks_doi_continuation isn't part of the AdjacentPatterns triple -- it
    # must be unaffected by the override.
    assert default_rows[1]["looks_doi_continuation"] == overridden_rows[1]["looks_doi_continuation"]


def test_current_patterns_wraps_the_modules_own_live_regexes():
    import bibr.extract.geom_adjacent_features as prod

    assert CURRENT_PATTERNS.numbered_start is prod.NUMBERED_START_RE
    assert CURRENT_PATTERNS.author_year is prod.REF_ONSET_RE
    assert CURRENT_PATTERNS.ref_header is prod.REF_HEADER_RE
