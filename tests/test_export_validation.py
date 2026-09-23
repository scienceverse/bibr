"""Unit + regression-fixture tests for the output validation gate.

Covers every check in ``bibr.export.validation``: a seeded defect emits its
code, a clean payload emits nothing, per-check exceptions are isolated, and two
frozen fixture dicts trip the expected codes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bibr.export import validation
from bibr.export.validation import ValidationIssue, validate_export
from bibr.validation import IssueSeverity

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "export_validation"


def _base() -> dict:
    """A minimal, well-formed export dict that trips no check."""
    return {
        "paper_id": "p1",
        "metadata": {"title": "A Study of Real Things", "abstract": "An abstract."},
        "author": [{"given": "Ann", "family": "Lee"}],
        "text": [
            {
                "text_id": 1,
                "section_id": 1,
                "text": "Body text with enough ordinary prose for a realistic abstract share.",
            }
        ],
        "section": [
            {
                "section_id": 1,
                "header": "Introduction",
                "section_type": "intro",
                "parent_section_id": None,
            }
        ],
        "url": [],
        "bib": [],
        "xref": [],
        "figure": [],
        "table": [],
        "eq": [],
        "funding": [],
        "affiliation": [],
    }


def _codes(issues: list[ValidationIssue]) -> set[str]:
    return {i.code for i in issues}


# ── clean payload ──────────────────────────────────────────────────────


def test_clean_payload_has_no_issues():
    assert validate_export(_base()) == []


def test_validation_issue_is_shared_typed_contract():
    import bibr.validation as shared_validation

    issue = shared_validation.ValidationIssue(
        code="VAL_TEST",
        severity=shared_validation.IssueSeverity.ERROR,
        message="typed",
        origin_stage="extract",
        evidence_ids=("ref-task",),
        count=2,
        blocking=True,
    )

    assert ValidationIssue is shared_validation.ValidationIssue
    assert issue.severity == "error"
    assert issue.evidence_ids == ("ref-task",)
    assert issue.count == 2
    assert issue.blocking is True


def test_raw_ocr_region_diagnostics_do_not_affect_canonical_validation():
    payload = _base()
    payload["_regions"] = [
        {
            "page": 1,
            "index": 0,
            "label": "text",
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "content": "Canonical text.",
            "raw_ocr_content": "Untruncated source diagnostic.",
        }
    ]

    assert validate_export(payload) == []


def test_private_use_in_canonical_identity_fields_is_nonblocking_warning():
    payload = _base()
    payload["metadata"].update(
        {
            "doi": "10.1000/927",
            "journal": "Journal of History",
            "volume": "29",
            "issue": "3",
            "pages": "319-345",
            "issn": "1234-5678",
            "publisher": "Birkhäuser",
            "published": "2021-09-07",
        }
    )
    payload["author"][0]["orcid"] = "0000-000-0002-0003"

    issues = [issue for issue in validate_export(payload) if issue.code == "VAL_UNICODE_CANONICAL"]

    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].blocking is False
    assert issues[0].count == 2


def test_private_use_diagnostics_body_refs_expected_identity_and_enrichment_are_ignored():
    payload = _base()
    payload["text"][0]["text"] = "Body native numeral  is diagnostic, not identity."
    payload["bib"] = [{"bib_id": 1, "title": "Reference from 927"}]
    payload["_regions"] = [
        {
            "content": "Canonical body.",
            "native_text_candidate": "Native body .",
            "raw_ocr_content": "Raw body .",
        }
    ]
    payload["expected_identity"] = {"expected_title": "Expected  title"}
    payload["enrichment"] = {"publisher": "Enriched  publisher"}

    assert "VAL_UNICODE_CANONICAL" not in _codes(validate_export(payload))


def test_clean_non_ascii_identity_unicode_is_allowed():
    payload = _base()
    payload["metadata"].update(
        {
            "title": "Bühler’s α-study — 1927",
            "journal": "Zeitschrift für Pädagogik",
            "publisher": "Birkhäuser",
        }
    )
    payload["author"][0].update({"given": "Zoë", "family": "Łukaszewicz"})

    assert "VAL_UNICODE_CANONICAL" not in _codes(validate_export(payload))


def test_non_dict_input_is_safe():
    assert validate_export(None) == []
    assert validate_export("not a dict") == []
    assert validate_export([1, 2, 3]) == []


# ── ERROR-severity checks ──────────────────────────────────────────────


def test_placeholder_in_metadata():
    p = _base()
    p["metadata"]["abstract"] = "verbatim-string"
    assert "VAL_PLACEHOLDER" in _codes(validate_export(p))


def test_placeholder_in_author():
    p = _base()
    p["author"][0]["family"] = "STRING"  # case-insensitive
    assert "VAL_PLACEHOLDER" in _codes(validate_export(p))


def test_placeholder_in_affiliation():
    """Regression: ``_check_placeholder`` reads the root ``affiliation`` key
    (singular, v11) — it silently stopped checking anything the moment the
    key was renamed from ``affiliations`` until the reader was updated."""
    p = _base()
    p["affiliation"] = [{"affiliation_id": 1, "text": "verbatim-string"}]
    assert "VAL_PLACEHOLDER" in _codes(validate_export(p))


def test_author_blank():
    p = _base()
    p["author"].append({"given": "", "family": "  "})
    issues = validate_export(p)
    assert "VAL_AUTHOR_BLANK" in _codes(issues)


def test_group_author_with_only_a_literal_name_is_not_blank():
    p = _base()
    p["author"].append({"given": None, "family": None, "literal": "The Consortium"})
    assert "VAL_AUTHOR_BLANK" not in _codes(validate_export(p))


def test_author_outlier_repeated_group_author():
    p = _base()
    p["author"] = [{"given": None, "family": None, "literal": "The Consortium"} for _ in range(3)]
    assert "VAL_AUTHOR_OUTLIER" in _codes(validate_export(p))


def test_author_outlier_count():
    p = _base()
    p["author"] = [{"given": f"A{i}", "family": f"B{i}"} for i in range(65)]
    assert "VAL_AUTHOR_OUTLIER" in _codes(validate_export(p))


def test_author_outlier_repeated_pair():
    p = _base()
    p["author"] = [{"given": "Jo", "family": "Ng"} for _ in range(3)]
    assert "VAL_AUTHOR_OUTLIER" in _codes(validate_export(p))


def test_author_repeated_twice_is_ok():
    p = _base()
    p["author"] = [{"given": "Jo", "family": "Ng"} for _ in range(2)]
    assert "VAL_AUTHOR_OUTLIER" not in _codes(validate_export(p))


def test_bbox_space_violation():
    p = _base()
    p["extraction"] = {
        "pages": [{"page_number": 1, "width": 600, "height": 800}],
        "text_regions": [
            {
                "text_id": i,
                "page_number": 1,
                "bbox": [10, 10, 9999, 20],  # x2 > the page width
            }
            for i in range(5)
        ],
    }
    assert "VAL_BBOX_SPACE" in _codes(validate_export(p))


def test_bbox_space_within_bounds_ok():
    p = _base()
    p["extraction"] = {
        "pages": [{"page_number": 1, "width": 600, "height": 800}],
        "text_regions": [
            {
                "text_id": i,
                "page_number": 1,
                "bbox": [10, 10, 100, 20],
            }
            for i in range(5)
        ],
    }
    assert "VAL_BBOX_SPACE" not in _codes(validate_export(p))


def test_dangling_xref_target():
    p = _base()
    p["xref"] = [{"target_id": 999, "xref_type": "bib", "contents": "[9]", "text_id": 1}]
    assert "VAL_DANGLING_REF" in _codes(validate_export(p))


def test_dangling_text_section():
    p = _base()
    p["text"] = [{"text_id": 1, "section_id": 42, "text": "x"}]
    assert "VAL_DANGLING_REF" in _codes(validate_export(p))


def test_dangling_bib_text_id():
    p = _base()
    p["bib"] = [{"bib_id": 1, "text_id": 777}]
    assert "VAL_DANGLING_REF" in _codes(validate_export(p))


def test_empty_eq():
    p = _base()
    p["eq"] = [{"text_id": 1, "grp_id": 1, "lhs": "", "df": "", "comp": "", "rhs": ""}]
    assert "VAL_EMPTY_EQ" in _codes(validate_export(p))


def test_nonempty_eq_ok():
    p = _base()
    p["eq"] = [{"text_id": 1, "grp_id": 1, "lhs": "x", "df": "", "comp": "=", "rhs": "1"}]
    assert "VAL_EMPTY_EQ" not in _codes(validate_export(p))


def test_url_malformed():
    p = _base()
    p["url"] = [{"href": "http://localhost", "link_text": None, "text_id": 1}]
    assert "VAL_URL_MALFORMED" in _codes(validate_export(p))


def test_url_wellformed_ok():
    p = _base()
    p["url"] = [{"href": "https://example.com/x", "link_text": None, "text_id": 1}]
    assert "VAL_URL_MALFORMED" not in _codes(validate_export(p))


# ── WARNING-severity checks ────────────────────────────────────────────


def test_title_generic():
    p = _base()
    p["metadata"]["title"] = "PhD Dissertation"
    assert "VAL_TITLE_GENERIC" in _codes(validate_export(p))


def test_title_empty():
    p = _base()
    p["metadata"]["title"] = ""
    assert "VAL_TITLE_GENERIC" in _codes(validate_export(p))


def test_abstract_missing():
    p = _base()
    p["section"] = [
        {
            "section_id": 1,
            "header": "Abstract",
            "section_type": "abstract",
            "parent_section_id": None,
        }
    ]
    p["text"] = [
        {"text_id": 1, "section_id": 1, "text": "one"},
        {"text_id": 2, "section_id": 1, "text": "two"},
    ]
    p["metadata"]["abstract"] = None
    assert "VAL_ABSTRACT_MISSING" in _codes(validate_export(p))


def test_abstract_suspect_when_ungrounded_against_abstract_section():
    p = _base()
    p["section"].append(
        {
            "section_id": 2,
            "header": "Abstract",
            "section_type": "abstract",
            "parent_section_id": None,
        }
    )
    p["text"].append({"text_id": 2, "section_id": 2, "text": "Grounded source abstract."})
    p["metadata"]["abstract"] = "Invented abstract."

    issue = next(i for i in validate_export(p) if i.code == "VAL_ABSTRACT_SUSPECT")
    assert issue.blocking is False
    assert "ungrounded" in issue.message


def test_abstract_suspect_when_value_crosses_into_body_section():
    p = _base()
    p["section"].append(
        {
            "section_id": 2,
            "header": "Abstract",
            "section_type": "abstract",
            "parent_section_id": None,
        }
    )
    p["text"] = [
        {"text_id": 1, "section_id": 2, "text": "Grounded source abstract."},
        {"text_id": 2, "section_id": 1, "text": "Body sentence outside the abstract."},
    ]
    p["metadata"]["abstract"] = "Grounded source abstract. Body sentence outside the abstract."

    issue = next(i for i in validate_export(p) if i.code == "VAL_ABSTRACT_SUSPECT")
    assert "cross_boundary" in issue.message


@pytest.mark.parametrize(
    ("abstract", "body", "reason"),
    [
        ("A" * 2501, "Body " * 4000, "length_gt_2500"),
        ("A" * 250, "Body " * 100, "non_reference_share_gt_20pct"),
    ],
)
def test_abstract_length_and_share_are_nonblocking_warnings(abstract, body, reason):
    p = _base()
    p["metadata"]["abstract"] = abstract
    p["text"] = [{"text_id": 1, "section_id": 1, "text": body}]

    issue = next(i for i in validate_export(p) if i.code == "VAL_ABSTRACT_SUSPECT")

    assert issue.blocking is False
    assert reason in issue.message
    assert p["metadata"]["abstract"] == abstract


def test_abstract_thresholds_are_strict_and_references_are_excluded():
    p = _base()
    p["section"].append(
        {
            "section_id": 2,
            "header": "References",
            "section_type": "references",
            "parent_section_id": None,
        }
    )
    p["metadata"]["abstract"] = "A" * 20
    p["text"] = [
        {"text_id": 1, "section_id": 1, "text": "B" * 100},
        {"text_id": 2, "section_id": 2, "text": "R" * 1000},
    ]
    assert "VAL_ABSTRACT_SUSPECT" not in _codes(validate_export(p))

    p["metadata"]["abstract"] = "A" * 21
    issues = [issue for issue in validate_export(p) if issue.code == "VAL_ABSTRACT_SUSPECT"]
    assert len(issues) == 1
    assert "non_reference_share_gt_20pct" in issues[0].message

    p["metadata"]["abstract"] = "A" * 2500
    p["text"][0]["text"] = "B" * 20000
    assert "VAL_ABSTRACT_SUSPECT" not in _codes(validate_export(p))

    p["metadata"]["abstract"] = "A" * 2501
    issue = next(i for i in validate_export(p) if i.code == "VAL_ABSTRACT_SUSPECT")
    assert "length_gt_2500" in issue.message


def test_abstract_share_zero_denominator_does_not_warn_or_mutate():
    p = _base()
    p["metadata"]["abstract"] = "No source denominator."
    p["text"] = []

    assert "VAL_ABSTRACT_SUSPECT" not in _codes(validate_export(p))
    assert p["metadata"]["abstract"] == "No source denominator."


def test_abstract_suspect_emits_once_with_bounded_source_ids():
    p = _base()
    p["section"].append(
        {
            "section_id": 2,
            "header": "Abstract",
            "section_type": "abstract",
            "parent_section_id": None,
        }
    )
    p["text"] = [
        {"text_id": index, "section_id": 2, "text": f"Source sentence {index}."}
        for index in range(1, 31)
    ]
    p["metadata"]["abstract"] = "A" * 2501

    issues = [issue for issue in validate_export(p) if issue.code == "VAL_ABSTRACT_SUSPECT"]

    assert len(issues) == 1
    assert len(issues[0].evidence_ids) == 20


def test_abstract_suspect_is_replay_fallback_when_payload_already_has_issue():
    p = _base()
    p["metadata"]["abstract"] = "A" * 2501
    p["validation"] = {
        "errors": 0,
        "warnings": 1,
        "blocking": 0,
        "promotable": True,
        "issues": [
            {
                "code": "VAL_ABSTRACT_SUSPECT",
                "severity": "warning",
                "message": "abstract suspicion: length_gt_2500",
                "origin_stage": "extract",
                "evidence_ids": ["text:1"],
                "count": 1,
                "blocking": False,
            }
        ],
    }

    assert "VAL_ABSTRACT_SUSPECT" not in _codes(validate_export(p))


def test_appendix_nested_under_references():
    p = _base()
    p["section"] = [
        {
            "section_id": 1,
            "header": "References",
            "section_type": "references",
            "parent_section_id": None,
        },
        {
            "section_id": 2,
            "header": "A. Extra Proofs",
            "section_type": None,
            "parent_section_id": 1,
        },
    ]
    assert "VAL_APPENDIX_NESTING" in _codes(validate_export(p))


def test_appendix_nested_under_appendix():
    p = _base()
    p["section"] = [
        {"section_id": 1, "header": "Appendix", "section_type": None, "parent_section_id": None},
        {"section_id": 2, "header": "B. Details", "section_type": None, "parent_section_id": 1},
    ]
    assert "VAL_APPENDIX_NESTING" in _codes(validate_export(p))


def test_caption_missing_majority():
    p = _base()
    p["table"] = [
        {"table_id": 1, "caption": None, "contents": []},
        {"table_id": 2, "caption": "", "contents": []},
        {"table_id": 3, "caption": "Table 3. Real caption", "contents": []},
    ]
    assert "VAL_CAPTION_MISSING" in _codes(validate_export(p))


def test_caption_present_ok():
    p = _base()
    p["table"] = [{"table_id": 1, "caption": "Table 1. Caption", "contents": []}]
    assert "VAL_CAPTION_MISSING" not in _codes(validate_export(p))


def test_panel_caption():
    p = _base()
    p["figure"] = [
        {"figure_id": 1, "caption": "(a)", "page_number": 1},
        {"figure_id": 2, "caption": "b)", "page_number": 1},
    ]
    assert "VAL_PANEL_CAPTION" in _codes(validate_export(p))


def test_xref_zero():
    p = _base()
    p["bib"] = [{"bib_id": i, "text_id": None} for i in range(1, 12)]
    p["xref"] = [{"target_id": 1, "xref_type": "figure", "contents": "Figure 1", "text_id": 1}]
    p["figure"] = [{"figure_id": 1, "caption": "Real", "page_number": 1}]
    assert "VAL_XREF_ZERO" in _codes(validate_export(p))


def test_xref_zero_ok_when_bib_target_present():
    p = _base()
    p["bib"] = [{"bib_id": i, "text_id": None} for i in range(1, 12)]
    p["xref"] = [{"target_id": 1, "xref_type": "bib", "contents": "[1]", "text_id": 1}]
    assert "VAL_XREF_ZERO" not in _codes(validate_export(p))


def test_xref_low_coverage_counts_unique_valid_bib_targets_only():
    p = _base()
    p["bib"] = [{"bib_id": i, "text_id": None} for i in range(1, 11)]
    p["xref"] = [
        {"target_id": 1, "xref_type": "bib", "contents": "[1]", "text_id": 1},
        {"target_id": 1, "xref_type": "bib", "contents": "[1]", "text_id": 2},
        {"target_id": 99, "xref_type": "bib", "contents": "[99]", "text_id": 3},
        {"target_id": 2, "xref_type": "figure", "contents": "Figure 2", "text_id": 4},
    ]

    issues = validate_export(p)

    low = [issue for issue in issues if issue.code == "VAL_XREF_LOW_COVERAGE"]
    assert len(low) == 1
    assert low[0].count == 1
    assert "1/10" in low[0].message
    assert "VAL_XREF_ZERO" not in _codes(issues)


def test_xref_low_coverage_uses_bibliography_row_count_denominator():
    p = _base()
    p["bib"] = [{"bib_id": i, "text_id": None} for i in range(1, 10)] + [
        {"bib_id": 9, "text_id": None}
    ]
    p["xref"] = [
        {"target_id": 1, "xref_type": "bib", "contents": "[1]", "text_id": 1},
    ]

    issues = validate_export(p)

    low = [issue for issue in issues if issue.code == "VAL_XREF_LOW_COVERAGE"]
    assert len(low) == 1
    assert low[0].count == 1
    assert "1/10" in low[0].message


def test_xref_low_coverage_threshold_is_strictly_below_twenty_percent():
    p = _base()
    p["bib"] = [{"bib_id": i, "text_id": None} for i in range(1, 11)]
    p["xref"] = [
        {"target_id": i, "xref_type": "bib", "contents": f"[{i}]", "text_id": i} for i in (1, 2)
    ]

    assert "VAL_XREF_LOW_COVERAGE" not in _codes(validate_export(p))


def test_xref_low_coverage_requires_at_least_ten_references():
    p = _base()
    p["bib"] = [{"bib_id": i, "text_id": None} for i in range(1, 10)]

    assert "VAL_XREF_LOW_COVERAGE" not in _codes(validate_export(p))


def test_post_parse_xref_low_coverage_issue_suppresses_export_replay_duplicate():
    from bibr.export.json_export import _apply_output_validation

    p = _base()
    p["bib"] = [{"bib_id": i, "text_id": None} for i in range(1, 11)]
    p["xref"] = [
        {"target_id": 1, "xref_type": "bib", "contents": "[1]", "text_id": 1},
    ]
    source_issue = ValidationIssue(
        "VAL_XREF_LOW_COVERAGE",
        IssueSeverity.WARNING,
        "1/10 bibliography entries have a resolved in-text citation xref (10.0%)",
        origin_stage="post_parse",
        evidence_ids=("bib:1",),
        count=1,
    )

    out = _apply_output_validation(p, [source_issue])
    survived = [
        issue
        for issue in out["extraction"]["validation"]["issues"]
        if issue["code"] == source_issue.code
    ]

    assert len(survived) == 1
    assert survived[0]["origin_stage"] == "post_parse"


def test_xref_low_coverage_export_replay_remains_available_without_source_issue():
    from bibr.export.json_export import _apply_output_validation

    p = _base()
    p["bib"] = [{"bib_id": i, "text_id": None} for i in range(1, 11)]
    p["xref"] = [
        {"target_id": 1, "xref_type": "bib", "contents": "[1]", "text_id": 1},
    ]

    out = _apply_output_validation(p)
    survived = [
        issue
        for issue in out["extraction"]["validation"]["issues"]
        if issue["code"] == "VAL_XREF_LOW_COVERAGE"
    ]

    assert len(survived) == 1
    assert survived[0]["origin_stage"] == "export"


def test_ref_count_mismatch():
    p = _base()
    p["section"] = [
        {
            "section_id": 1,
            "header": "References",
            "section_type": "references",
            "parent_section_id": None,
        }
    ]
    p["text"] = [{"text_id": i, "section_id": 1, "text": f"ref {i}"} for i in range(10)]
    p["bib"] = [{"bib_id": 1, "text_id": None}]
    assert "VAL_REF_COUNT_MISMATCH" in _codes(validate_export(p))


def test_ref_count_matched_ok():
    p = _base()
    p["section"] = [
        {
            "section_id": 1,
            "header": "References",
            "section_type": "references",
            "parent_section_id": None,
        }
    ]
    p["text"] = [{"text_id": i, "section_id": 1, "text": f"ref {i}"} for i in range(6)]
    p["bib"] = [{"bib_id": i, "text_id": None} for i in range(6)]
    assert "VAL_REF_COUNT_MISMATCH" not in _codes(validate_export(p))


def test_statement_orphan():
    p = _base()
    p["text"] = [
        {"text_id": 1, "section_id": 1, "text": "The authors declare no conflict of interest."}
    ]
    p["metadata"]["coi_statement"] = None
    assert "VAL_STATEMENT_ORPHAN" in _codes(validate_export(p))


def test_statement_present_ok():
    p = _base()
    p["text"] = [
        {"text_id": 1, "section_id": 1, "text": "The authors declare no conflict of interest."}
    ]
    p["metadata"]["coi_statement"] = "The authors declare no conflict of interest."
    assert "VAL_STATEMENT_ORPHAN" not in _codes(validate_export(p))


# ── exception isolation ────────────────────────────────────────────────


def test_broken_check_is_isolated(monkeypatch):
    def _boom(_payload):
        raise RuntimeError("kaboom")

    def _good(_payload):
        return [ValidationIssue("VAL_TEST", "warning", "ok")]

    monkeypatch.setattr(validation, "_CHECKS", (_boom, _good))
    issues = validate_export(_base())
    codes = _codes(issues)
    assert "VAL_INTERNAL" in codes  # the broken check degraded, did not crash
    assert "VAL_TEST" in codes  # the other check still ran


# ── integration on real fixture dicts (read-only) ──────────────────────


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("author_bbox_panel.json", {"VAL_AUTHOR_OUTLIER", "VAL_BBOX_SPACE", "VAL_PANEL_CAPTION"}),
        ("placeholder_caption.json", {"VAL_PLACEHOLDER", "VAL_CAPTION_MISSING"}),
    ],
)
def test_real_fixture_defects(fixture, expected):
    path = _FIXTURES / fixture
    if not path.exists():
        pytest.skip(f"fixture {fixture} not present")
    payload = json.loads(path.read_text())
    codes = _codes(validate_export(payload))
    assert expected <= codes, f"{fixture}: missing {expected - codes}"
