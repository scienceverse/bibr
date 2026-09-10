"""Shape assertions for the v11 export. Uses the shared paper fixture."""

import datetime as dt

import pytest

from tests.export.conftest import extraction_export


def test_engines_are_direct_children_no_models_wrapper():
    e = extraction_export(
        ocr={"backend": "glm-http", "model": "glm-ocr", "profile": "glm"},
        llm={"provider": "google", "model": "gemini-flash-lite", "backend": "api"},
    )
    dumped = e.model_dump()
    assert "models" not in dumped
    assert dumped["ocr"]["backend"] == "glm-http"
    assert dumped["llm"]["provider"] == "google"


def test_llm_null_means_the_run_had_no_llm():
    e = extraction_export(llm=None)
    assert e.model_dump()["llm"] is None


def test_ocr_null_on_docx_native_path():
    e = extraction_export(ocr=None)
    assert e.model_dump()["ocr"] is None


def test_completed_at_is_utc_iso8601():
    e = extraction_export()
    dt.datetime.fromisoformat(e.completed_at.replace("Z", "+00:00"))


def test_enrichment_is_omitted_not_zeroed_when_crossref_off():
    dumped = extraction_export().model_dump()
    assert "enrichment" not in dumped


def test_regions_and_trace_are_siblings_not_under_diagnostics():
    e = extraction_export(diagnostics={"text_quality": 0.9})
    dumped = e.model_dump()
    assert "regions" not in dumped["diagnostics"]
    assert "trace" not in dumped["diagnostics"]


def test_usage_totals_equal_breakdown_sums():
    e = extraction_export(
        usage={
            "totals": {
                "calls": 3,
                "input_tokens": 400,
                "cached_input_tokens": 40,
                "output_tokens": 70,
                "total_tokens": 470,
            },
            "breakdown": [
                {
                    "label": "a",
                    "provider": "google",
                    "model": "m",
                    "calls": 3,
                    "input_tokens": 400,
                    "cached_input_tokens": 40,
                    "output_tokens": 70,
                    "total_tokens": 470,
                }
            ],
        }
    )
    dumped = e.model_dump()
    totals = dumped["usage"]["totals"]
    rows = dumped["usage"]["breakdown"]
    for field in ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
        assert totals[field] == sum(r[field] for r in rows)


def test_gate_findings_are_not_mirrored_into_warnings():
    """Gate findings live only in validation.issues — never duplicated as prose."""
    from bibr.export.json_export import _apply_output_validation

    payload = {"extraction": {"warnings": ["ocr retried page 3"]}}
    out = _apply_output_validation(payload, stage_issues=[])

    assert "validation" in out
    assert out["extraction"]["warnings"] == ["ocr retried page 3"]
    assert not any(w.startswith("VALIDATION:") for w in out["extraction"]["warnings"])


# ── v11 root reshape ───────────────────────────────────────────────────


def test_schema_version_is_at_the_root(v11_payload):
    assert v11_payload["schema_version"] == "11.0"
    assert "schema_version" not in v11_payload.get("metadata", {})


def test_source_holds_file_identity(v11_payload):
    src = v11_payload["source"]
    assert set(src) == {"file_name", "file_hash", "input_format"}
    assert src["input_format"] == src["input_format"].lower()


def test_metadata_is_scalars_only(v11_payload):
    """The R uniform-tables contract: as.data.frame(metadata) must not blow up."""
    for key, value in v11_payload["metadata"].items():
        if key == "keywords":  # pre-existing, grandfathered
            continue
        assert not isinstance(value, (dict, list)), key


def test_info_key_is_gone(v11_payload):
    assert "info" not in v11_payload
    assert "info_match" not in v11_payload


def test_text_quality_moved_to_diagnostics(v11_payload):
    # The value, not just the key: ``DiagnosticsExport`` always emits
    # ``text_quality`` with a ``None`` default, so a presence-only assertion
    # cannot fail and would not notice the score being dropped on the way out.
    assert "text_quality" not in v11_payload["metadata"]
    assert v11_payload["extraction"]["diagnostics"]["text_quality"] == 0.87


def test_root_record_arrays_survive_refs_off(v11_payload_refs_off):
    """Uniform-tables contract: a table is never omitted, only empty."""
    for key in ("bib", "bib_match", "xref", "author", "text", "section", "url"):
        assert isinstance(v11_payload_refs_off[key], list), key
    assert v11_payload_refs_off["bib"] == []
    assert v11_payload_refs_off["bib_match"] == []
    assert v11_payload_refs_off["metadata_match"] != []


def test_core_replay_gate_tracks_the_live_schema_version():
    """``CORE_SCHEMA_VERSION`` is a hand-maintained copy of the exporter's
    version. Now that the gate accepts exactly one value, drift between the two
    would reject every freshly written core — so pin them together."""
    from bibr.export.json_export import _SCHEMA_VERSION
    from bibr.pipeline.artifacts import CORE_SCHEMA_VERSION, SUPPORTED_CORE_SCHEMA_VERSIONS

    assert CORE_SCHEMA_VERSION == _SCHEMA_VERSION
    assert set(SUPPORTED_CORE_SCHEMA_VERSIONS) == {_SCHEMA_VERSION}


# ── v11 convention fixes (Task 6) ──────────────────────────────────────


def test_xref_foreign_key_is_named_target_id(v11_payload):
    for row in v11_payload["xref"]:
        assert "xref_id" not in row
        assert "target_id" in row


def test_eq_has_a_primary_key_and_verbatim(v11_payload):
    for row in v11_payload["eq"]:
        assert isinstance(row["eq_id"], int)
        assert "verbatim" in row


def test_funding_has_a_primary_key(v11_payload):
    for index, row in enumerate(v11_payload["funding"], start=1):
        assert row["funding_id"] == index


def test_affiliation_table_is_singular(v11_payload):
    assert "affiliation" in v11_payload
    assert "affiliations" not in v11_payload


def test_root_record_arrays_are_always_present(v11_payload_refs_off):
    """Uniform-tables contract: every root record array exists (never omitted)
    under refs="off". Only ``bib``/``bib_match`` are actually forced empty by
    that flag in this fixture — ``xref`` keeps its non-bib (table) row and
    ``funding``/``affiliation``/``metadata_match`` are populated independently
    of reference extraction (parsed from the byline/funding statement, or the
    paper's own self-DOI match), matching the sibling
    ``test_root_record_arrays_survive_refs_off`` above, which already asserts
    ``metadata_match != []`` here. So: presence for all six, exact emptiness
    only for the two that are actually reference-shaped.
    """
    for table in ("bib", "bib_match", "xref", "affiliation", "funding", "metadata_match"):
        assert isinstance(v11_payload_refs_off[table], list), table
    assert v11_payload_refs_off["bib"] == []
    assert v11_payload_refs_off["bib_match"] == []


# ── v11 structured reference authors (Task 7) ──────────────────────────


def test_bib_keeps_verbatim_authors_and_gains_structured_author(v11_payload):
    for row in v11_payload["bib"]:
        if row.get("authors"):
            assert isinstance(row["authors"], str)
            assert isinstance(row["author"], list)


@pytest.mark.parametrize(
    ("structured_key", "verbatim_key"), [("author", "authors"), ("editor", "editors")]
)
def test_structured_name_parts_substring_match_the_verbatim(
    v11_payload, structured_key, verbatim_key
):
    """The ground-truth invariant, enforced on real output — for both the
    author and editor pairs (same splitter, same guarantee). Editor strings
    carry conventions author strings don't (e.g. a trailing "(Ed.)"/"(Eds.)"),
    so this must not be author-only."""
    checked = 0
    for row in v11_payload["bib"]:
        verbatim = row.get(verbatim_key)
        if not verbatim:
            continue
        for person in row[structured_key]:
            for value in person.values():
                checked += 1
                assert value in verbatim, (value, verbatim)
    # Guard against the invariant going quietly vacuous if the fixture ever
    # loses its non-empty editors row.
    assert checked > 0, f"fixture exercised no non-empty {verbatim_key!r} row"


def test_bib_and_match_tables_share_one_absence_encoding_for_names(demo_paper):
    """``bib[].author``/``editor`` are nullable, exactly like the identically
    named ``bib_match``/``metadata_match`` fields and like their own verbatim
    sibling. An empty split must yield ``null``, never ``[]`` — the two encode
    to different column types in R, and the versioning policy would make a
    later correction a 12.0-only change.
    """
    from bibr.export.json_export import _export_paper_payload

    ref = demo_paper.metadata.references[0]
    ref.authors = None
    ref.editors = "   "

    row = _export_paper_payload(demo_paper)["bib"][0]
    assert row["authors"] is None
    assert row["author"] is None
    assert row["editor"] is None


def test_bib_structured_names_are_populated_when_verbatim_exists(v11_payload):
    """The null encoding above must not be the only reachable state."""
    rows = [r for r in v11_payload["bib"] if r.get("authors")]
    assert rows, "fixture exercised no non-empty authors row"
    for row in rows:
        assert isinstance(row["author"], list) and row["author"]


def test_match_tables_use_singular_author(v11_payload):
    for table in ("bib_match", "metadata_match"):
        for row in v11_payload[table]:
            assert "authors" not in row
            assert "editors" not in row


def test_paper_authors_accept_a_suffix():
    from bibr.export.models import AuthorExport

    a = AuthorExport(author_id=1, given="M. L.", family="King", corresponding=False, suffix="Jr.")
    assert a.model_dump()["suffix"] == "Jr."


def test_person_name_export_rejects_an_empty_identity():
    """A person record identifying nobody (no family/given/literal) must not
    validate clean — that's a vacuous row, not a best-effort split. Reachable
    only via externally-sourced enrichment-sidecar replay rows, but the
    schema strictness this release is built on should still catch it."""
    from pydantic import ValidationError

    from bibr.export.models import PersonNameExport

    with pytest.raises(ValidationError):
        PersonNameExport()

    with pytest.raises(ValidationError):
        # A bare suffix with no name to attach it to is still empty-identity.
        PersonNameExport(suffix="Jr.")
