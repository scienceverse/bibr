"""Shape assertions for the v12 export. Uses the shared paper fixture."""

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
    """Gate findings live only in validation.issues — never duplicated as warnings."""
    from bibr.export.json_export import _apply_output_validation

    retried = {"code": "OCR_REGION_FAILED", "message": "ocr retried page 3"}
    payload = {"extraction": {"warnings": [retried]}}
    out = _apply_output_validation(payload, stage_issues=[])

    assert "validation" in out["extraction"]
    assert out["extraction"]["warnings"] == [retried]


# ── root reshape (v11) ───────────────────────────────────────────────────


def test_schema_version_is_at_the_root(export_payload):
    assert export_payload["schema_version"] == "12.0"
    assert "schema_version" not in export_payload.get("metadata", {})


def test_source_holds_file_identity(export_payload):
    src = export_payload["source"]
    assert set(src) == {"file_name", "sha256", "input_format"}
    assert len(src["sha256"]) == 64
    assert src["input_format"] == src["input_format"].lower()


def test_metadata_is_scalars_only(export_payload):
    """The R uniform-tables contract: as.data.frame(metadata) must not blow up."""
    for key, value in export_payload["metadata"].items():
        if key == "keywords":  # pre-existing, grandfathered
            continue
        assert not isinstance(value, (dict, list)), key


def test_info_key_is_gone(export_payload):
    assert "info" not in export_payload
    assert "info_match" not in export_payload


def test_text_quality_moved_to_diagnostics(export_payload):
    # The value, not just the key: ``DiagnosticsExport`` always emits
    # ``text_quality`` with a ``None`` default, so a presence-only assertion
    # cannot fail and would not notice the score being dropped on the way out.
    assert "text_quality" not in export_payload["metadata"]
    assert export_payload["extraction"]["diagnostics"]["text_quality"] == 0.87


def test_root_record_arrays_survive_refs_off(export_payload_refs_off):
    """Uniform-tables contract: a table is never omitted, only empty."""
    for key in ("bib", "bib_match", "xref", "author", "text", "section", "url"):
        assert isinstance(export_payload_refs_off[key], list), key
    assert export_payload_refs_off["bib"] == []
    assert export_payload_refs_off["bib_match"] == []
    assert export_payload_refs_off["metadata_match"] != []


def test_core_replay_gate_tracks_the_live_schema_version():
    """``CORE_SCHEMA_VERSION`` is a hand-maintained copy of the exporter's
    version. Now that the gate accepts exactly one value, drift between the two
    would reject every freshly written core — so pin them together."""
    from bibr.export.json_export import _SCHEMA_VERSION
    from bibr.pipeline.artifacts import CORE_SCHEMA_VERSION, SUPPORTED_CORE_SCHEMA_VERSIONS

    assert CORE_SCHEMA_VERSION == _SCHEMA_VERSION
    assert set(SUPPORTED_CORE_SCHEMA_VERSIONS) == {_SCHEMA_VERSION}


# ── record-table conventions ──────────────────────────────────────


def test_xref_foreign_key_is_named_target_id(export_payload):
    for row in export_payload["xref"]:
        assert "target_id" in row


def test_every_record_table_has_a_positional_primary_key(export_payload):
    """v12: xref and url were the last tables without a ``<table>_id``."""
    for table, key in (("xref", "xref_id"), ("url", "url_id"), ("eq", "eq_id")):
        rows = export_payload[table]
        assert rows, f"fixture exercised no {table} row"
        assert [row[key] for row in rows] == list(range(1, len(rows) + 1)), table


def test_eq_has_a_primary_key_and_verbatim(export_payload):
    for row in export_payload["eq"]:
        assert isinstance(row["eq_id"], int)
        assert "verbatim" in row


def test_funding_has_a_primary_key(export_payload):
    for index, row in enumerate(export_payload["funding"], start=1):
        assert row["funding_id"] == index


def test_affiliation_table_is_singular(export_payload):
    assert "affiliation" in export_payload
    assert "affiliations" not in export_payload


def test_root_record_arrays_are_always_present(export_payload_refs_off):
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
        assert isinstance(export_payload_refs_off[table], list), table
    assert export_payload_refs_off["bib"] == []
    assert export_payload_refs_off["bib_match"] == []


# ── v12: content rows carry no processing fields ──────────────────────


def test_root_keys_are_emitted_in_reading_order(export_payload):
    """The paper and its file, what it says, what registries returned, how it
    was produced — and every key is always present."""
    assert list(export_payload) == [
        "paper_id",
        "schema_version",
        "source",
        "metadata",
        "author",
        "affiliation",
        "funding",
        "text",
        "section",
        "url",
        "bib",
        "xref",
        "figure",
        "table",
        "eq",
        "metadata_match",
        "affiliation_match",
        "funding_match",
        "bib_match",
        "extraction",
    ]


def test_validation_lives_under_extraction(export_payload):
    from bibr.validation import payload_validation

    assert "validation" not in export_payload
    assert export_payload["extraction"]["validation"]["promotable"] in (True, False)
    assert payload_validation(export_payload) is export_payload["extraction"]["validation"]
    assert payload_validation({"validation": {"errors": 1}}) == {"errors": 1}  # v11 files


def test_extraction_is_always_present(demo_paper):
    """Outside the pipeline there is no stage-built skeleton, so the exporter
    builds a minimal one: package version and export time, no settings."""
    from bibr.export.json_export import _export_paper_payload

    demo_paper.extraction = None
    payload = _export_paper_payload(demo_paper)
    extraction = payload["extraction"]
    assert extraction["producer"]["name"] == "bibr"
    assert extraction["producer"]["version"]
    assert extraction["completed_at"].endswith("Z")
    assert "settings" not in extraction
    assert "validation" in extraction
    assert extraction["diagnostics"]["section_classification"]
    assert "classification_score" not in payload["section"][0]


@pytest.mark.parametrize(
    ("table", "moved"),
    [
        ("figure", {"parts"}),
        ("table", {"parts"}),
        ("section", {"classification_score", "classification_source"}),
        ("xref", {"tier"}),
        ("bib", {"consolidated_fields", "author", "editor"}),
        ("author", {"affiliation"}),
        (
            "text",
            {"_font_size", "_font_bold", "_is_italic", "_bbox_2d", "_region_type", "_page_w"},
        ),
    ],
)
def test_content_rows_carry_no_processing_or_duplicate_fields(export_payload, table, moved):
    rows = export_payload[table]
    assert rows, f"fixture exercised no {table} row"
    for row in rows:
        assert not moved & set(row), (table, moved & set(row))


def test_metadata_carries_no_confidences(export_payload):
    assert "paper_type_confidence" not in export_payload["metadata"]
    assert "oecd_confidence" not in export_payload["metadata"]
    assert "qualification_provenance" not in export_payload


def test_section_classification_moved_to_diagnostics(export_payload):
    rows = export_payload["extraction"]["diagnostics"]["section_classification"]
    assert [row["section_id"] for row in rows] == [
        s["section_id"] for s in export_payload["section"]
    ]
    # The fixture's sections are unclassified (PaperSection defaults: score 0.0,
    # no source), which is "not scored" and exports as null, not 0.0.
    assert all(row["score"] is None and row["source"] is None for row in rows)


def test_section_classification_keeps_a_recorded_score(demo_paper):
    from bibr.export.json_export import _export_paper_payload

    demo_paper.contents.sections[1].classification_score = 0.93
    demo_paper.contents.sections[1].classification_source = "model"
    rows = _export_paper_payload(demo_paper)["extraction"]["diagnostics"]["section_classification"]
    assert rows[0] == {"section_id": 1, "score": 0.93, "source": "model"}


def test_xref_tier_moved_to_diagnostics_keyed_by_xref_id(export_payload):
    tiers = export_payload["extraction"]["diagnostics"]["xref_tier"]
    bib_xrefs = [x for x in export_payload["xref"] if x["xref_type"] == "bib"]
    assert tiers == [{"xref_id": bib_xrefs[0]["xref_id"], "tier": "numeric"}]


def test_paper_classification_confidences_moved_to_diagnostics(export_payload):
    assert export_payload["extraction"]["diagnostics"]["paper_classification"] == {
        "paper_type_confidence": 0.91,
        "oecd_confidence": 0.77,
    }


def test_qualification_provenance_moved_under_extraction(demo_paper):
    from bibr.export.json_export import _export_paper_payload

    assert "qualification" not in _export_paper_payload(demo_paper)["extraction"]
    demo_paper.qualification_provenance = {"requests": 3}
    payload = _export_paper_payload(demo_paper)
    assert payload["extraction"]["qualification"] == {"requests": 3}
    assert "qualification_provenance" not in payload


def test_text_region_features_ride_extraction_when_opted_in(demo_paper):
    from bibr.export.json_export import _export_paper_payload

    demo_paper.contents.page_sizes = {1: (612.0, 792.0)}
    demo_paper.contents.sentences[0].region_meta = {
        "font_size": 9.5,
        "font_bold": False,
        "is_italic": True,
        "bbox": [100.0, 50.0, 500.0, 100.0],  # 0..1000 layout space
        "region_type": "text",
        "region_page": 1,
        "region_index": 0,
    }
    assert "text_regions" not in _export_paper_payload(demo_paper)["extraction"]
    payload = _export_paper_payload(demo_paper, include_region_meta=True)
    assert payload["extraction"]["text_regions"] == [
        {
            "text_id": 1,
            "font_size": 9.5,
            "font_bold": False,
            "is_italic": True,
            "page_number": 1,
            "bbox": [61.2, 39.6, 306.0, 79.2],  # points from the top-left
            "region_type": "text",
        }
    ]
    assert set(payload["text"][0]) == {
        "text",
        "text_id",
        "paragraph_id",
        "section_id",
        "page_number",
        "formatted",
    }


def test_float_part_locations_ride_extraction(demo_paper):
    from bibr.export.json_export import _export_paper_payload
    from bibr.paper_contents import PaperFigurePart, PaperTablePart

    demo_paper.contents.page_sizes = {2: (500.0, 800.0)}
    figure = demo_paper.contents.figures[0]
    figure.parts = [
        PaperFigurePart(page_number=2, bbox=(10, 20, 110, 90), image_b64=None),
        PaperFigurePart(page_number=2, bbox=(120, 20, 220, 90), image_b64=None),
    ]
    table = demo_paper.contents.tables[0]
    table.parts = [PaperTablePart(page_number=None, bbox=None, tbl_html="<table/>", df=None)]

    payload = _export_paper_payload(demo_paper)

    assert payload["extraction"]["float_parts"] == [
        {
            "object_type": "figure",
            "object_id": 1,
            "part_index": 1,
            "page_number": 2,
            "bbox": [5.0, 16.0, 55.0, 72.0],
        },
        {
            "object_type": "figure",
            "object_id": 1,
            "part_index": 2,
            "page_number": 2,
            "bbox": [60.0, 16.0, 110.0, 72.0],
        },
    ]  # the located-nowhere table part is left out
    assert payload["extraction"]["pages"] == [{"page_number": 2, "width": 500.0, "height": 800.0}]
    assert "parts" not in payload["figure"][0] and "parts" not in payload["table"][0]


def test_boxes_on_a_page_of_unknown_size_are_left_out(demo_paper):
    """Without the page size a 0..1000 layout box has no place in points."""
    from bibr.export.json_export import _export_paper_payload
    from bibr.paper_contents import PaperFigurePart

    demo_paper.contents.page_sizes = {}
    demo_paper.contents.figures[0].parts = [
        PaperFigurePart(page_number=3, bbox=(10, 20, 110, 90), image_b64=None)
    ]
    payload = _export_paper_payload(demo_paper)
    assert payload["extraction"]["float_parts"][0]["bbox"] is None
    assert "pages" not in payload["extraction"]


# ── v12: duplicates removed ────────────────────────────────────────────


def test_bib_keeps_only_the_printed_author_strings(export_payload):
    row = export_payload["bib"][0]
    assert row["authors"] == "Smith, J., & Doe, A."
    assert row["editors"] == "Roe, R."
    assert "author" not in row and "editor" not in row


def test_affiliation_table_links_authors_and_keeps_the_llm_parse(export_payload):
    assert export_payload["affiliation"] == [
        {
            "affiliation_id": 1,
            "text": "Department of Things, Example University",
            "institution": "Example University",
            "department": "Department of Things",
            "city": "Exampleville",
            "country": "Netherlands",
            "author_ids": [1],
        }
    ]


def test_affiliation_table_is_built_without_the_llm_parse(demo_paper):
    """The table is the only affiliation source now, so a run whose LLM parse
    never ran (``metadata.affiliations`` empty) must still get one row per
    distinct byline component, with the parsed fields null."""
    from bibr.export.json_export import _export_paper_payload
    from bibr.models import PaperAuthor

    demo_paper.metadata.affiliations = []
    demo_paper.metadata.authors.append(
        PaperAuthor(
            author_id=2,
            given="Ann",
            family="Lee",
            affiliation="Department of Things, Example University; Other Institute",
            corresponding=False,
        )
    )
    rows = _export_paper_payload(demo_paper)["affiliation"]
    assert [(r["affiliation_id"], r["text"], r["author_ids"]) for r in rows] == [
        (1, "Department of Things, Example University", [1, 2]),
        (2, "Other Institute", [2]),
    ]
    assert all(r["institution"] is None and r["country"] is None for r in rows)


# ── v12: absent values are null, never a sentinel ──────────────────────


def test_empty_strings_export_as_null(demo_paper):
    from bibr.export.json_export import _export_paper_payload

    demo_paper.metadata.authors[0].given = ""
    demo_paper.contents.sections[1].header = ""
    demo_paper.contents.equations[0].df = ""
    payload = _export_paper_payload(demo_paper)
    assert payload["author"][0]["given"] is None
    assert payload["section"][0]["header"] is None
    assert payload["eq"][0]["df"] is None


# ── v12: closed vocabularies ───────────────────────────────────────────


def test_foreign_bib_types_map_into_the_enum(demo_paper):
    from bibr.export.json_export import _export_paper_payload

    demo_paper.metadata.references[0].bib_type = "journal-article"
    assert _export_paper_payload(demo_paper)["bib"][0]["bib_type"] == "journal_article"
    demo_paper.metadata.references[0].bib_type = "misc"
    assert _export_paper_payload(demo_paper)["bib"][0]["bib_type"] == "other"


def test_off_vocabulary_classifier_labels_are_dropped_not_fatal(demo_paper, caplog):
    from bibr.export.json_export import _export_paper_payload

    demo_paper.metadata.oecd_l1 = "5. Social Sciences"
    payload = _export_paper_payload(demo_paper)
    assert payload["metadata"]["oecd_l1"] is None
    assert payload["extraction"]["diagnostics"]["paper_classification"] == {
        "paper_type_confidence": 0.91,
        "oecd_confidence": None,
    }
    assert "off-vocabulary oecd_l1" in caplog.text


def test_unknown_input_format_exports_as_unknown(demo_paper):
    from bibr.export.json_export import _export_paper_payload

    demo_paper.input_file.input_format.file_type = "TXT"
    assert _export_paper_payload(demo_paper)["source"]["input_format"] == "unknown"


def test_match_tables_use_singular_author(export_payload):
    for table in ("bib_match", "metadata_match"):
        for row in export_payload[table]:
            assert "authors" not in row
            assert "editors" not in row


def test_paper_authors_accept_a_suffix():
    from bibr.export.models import AuthorExport

    a = AuthorExport(author_id=1, given="M. L.", family="King", corresponding=False, suffix="Jr.")
    assert a.model_dump()["suffix"] == "Jr."


def test_person_name_export_omits_absent_parts():
    from bibr.export.models import PersonNameExport

    assert PersonNameExport(literal="ACME Inc").model_dump() == {"literal": "ACME Inc"}
    assert PersonNameExport(family="Wood", given="W.").model_dump() == {
        "family": "Wood",
        "given": "W.",
    }


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


# ── ROR matches of affiliations and funders ───────────────────────────


def test_the_export_turns_matches_into_rows(export_payload):
    assert export_payload["affiliation_match"] == [
        {
            "affiliation_id": 1,
            "service": "ror",
            "service_id": "https://ror.org/0abcde123",
            "score": 1.0,
            "name": "Example University",
            "country_code": "NL",
        }
    ]
    (funding,) = export_payload["funding_match"]
    assert funding["funding_id"] == 1 and funding["funder_doi"] == "10.13039/100000001"


# ── v12: adoption contract ─────────────────────────────────────────────


def test_xref_targets_name_a_row_or_nothing(demo_paper):
    """``target_id`` is a key of the row ``xref_type`` names, or null: the
    printed number of an equation, section or supplement names no row, and a
    footnote reference points at the footnote's text."""
    from bibr.export.json_export import _export_paper_payload
    from bibr.paper_contents import PaperXref

    demo_paper.contents.xrefs += [
        PaperXref(xref_id=5, xref_type="equation", contents="Eq. 5", text_id=2),
        PaperXref(xref_id=2, xref_type="section", contents="Section 2", text_id=2),
        PaperXref(xref_id=0, xref_type="supplementary", contents="Supplementary", text_id=2),
        PaperXref(xref_id=1, xref_type="foot", contents="1", text_id=1),
    ]
    rows = {x["xref_type"]: x for x in _export_paper_payload(demo_paper)["xref"]}
    assert rows["bib"]["target_id"] == 1
    assert rows["table"]["target_id"] == 1
    assert rows["equation"]["target_id"] is None
    assert rows["section"]["target_id"] is None
    assert rows["supplementary"]["target_id"] is None
    assert rows["foot"]["target_id"] == 1


def test_group_authors_are_literal(demo_paper):
    from bibr.export.json_export import _export_paper_payload
    from bibr.models import ORGANIZATION_ROLE, PaperAuthor

    demo_paper.metadata.authors.append(
        PaperAuthor(
            author_id=2,
            given="",
            family="The Consortium",
            affiliation="",
            role=[ORGANIZATION_ROLE, "Investigation"],
        )
    )
    group = _export_paper_payload(demo_paper)["author"][1]
    assert (group["given"], group["family"], group["literal"]) == (None, None, "The Consortium")
    # The marker is internal; the printed contribution role stays.
    assert group["role"] == ["Investigation"]
    assert group["credit_roles"] == ["https://credit.niso.org/contributor-roles/investigation/"]


@pytest.mark.parametrize(
    ("head", "media_type"),
    [
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\x01\x00\x00\x00" + b"\x00" * 36 + b" EMF", "image/emf"),
        (b"not an image", "application/octet-stream"),
    ],
)
def test_figure_images_are_data_uris_naming_their_type(demo_paper, head, media_type):
    import base64

    from bibr.export.json_export import _export_paper_payload

    encoded = base64.b64encode(head + b"\x00" * 32).decode()
    demo_paper.contents.figures[0].image_b64 = encoded
    image = _export_paper_payload(demo_paper)["figure"][0]["image"]
    assert image == f"data:{media_type};base64,{encoded}"


def test_dois_are_lowercase_everywhere(demo_paper):
    from bibr.export.json_export import _export_paper_payload

    demo_paper.metadata.doi = "10.1234/DEMO"
    demo_paper.metadata.references[0].doi = "https://doi.org/10.1234/PRIOR"
    payload = _export_paper_payload(demo_paper)
    assert payload["metadata"]["doi"] == "10.1234/demo"
    assert payload["bib"][0]["doi"] == "10.1234/prior"
    assert payload["bib_match"][0]["doi"] == "10.1234/prior"
    assert payload["metadata_match"][0]["funder"][0]["funder_doi"] == "10.13039/100000001"


def test_a_malformed_identifier_is_dropped_not_fatal(demo_paper, caplog):
    from bibr.export.json_export import _export_paper_payload

    demo_paper.metadata.references[0].doi = "10.1/too-short-a-prefix"
    payload = _export_paper_payload(demo_paper)
    assert payload["bib"][0]["doi"] is None
    assert "malformed bib.doi" in caplog.text


@pytest.mark.parametrize(
    ("date", "year", "published_date"),
    [
        ("2020, May 3", 2020, "2020-05-03"),
        (None, 2020, "2020"),
        ("in press", None, None),
        ("1890", 2020, "2020"),  # a date that contradicts the year yields the year
    ],
)
def test_bib_published_date_is_the_printed_date_or_year(demo_paper, date, year, published_date):
    from bibr.export.json_export import _export_paper_payload

    demo_paper.metadata.references[0].date = date
    demo_paper.metadata.references[0].year = year
    row = _export_paper_payload(demo_paper)["bib"][0]
    assert row["date"] == date
    assert row["published_date"] == published_date


def test_match_rows_carry_published_date_and_a_unit_score(export_payload):
    for table in ("bib_match", "metadata_match"):
        for row in export_payload[table]:
            assert "date" not in row
            assert "published_date" in row
            assert 0 <= row["score"] <= 1


def test_comparators_are_normalized_and_unknown_ones_dropped(demo_paper, caplog):
    from bibr.export.json_export import _export_paper_payload
    from bibr.paper_contents import PaperEquation

    demo_paper.contents.equations = [
        PaperEquation(text_id=2, grp_id=1, lhs="p", comp="<=", rhs=".05"),
        PaperEquation(text_id=2, grp_id=1, lhs="d", comp="≅", rhs="0.4"),
        PaperEquation(text_id=2, grp_id=1, lhs="r", comp="??", rhs="0.1"),
    ]
    eq = _export_paper_payload(demo_paper)["eq"]
    assert [(row["eq_id"], row["comp"]) for row in eq] == [(1, "≤"), (2, "≈")]
    assert "unknown comparator" in caplog.text


def test_vocabulary_tokens_are_snake_case(demo_paper):
    from bibr.export.json_export import _export_paper_payload
    from bibr.paper_contents import CanonicalSection

    demo_paper.metadata.paper_type = "meta-analysis"
    demo_paper.contents.sections[2].section_type = CanonicalSection.OPEN_DATA
    demo_paper.contents.xrefs[0].tier = "paren-numeric"
    payload = _export_paper_payload(demo_paper)
    assert payload["metadata"]["paper_type"] == "meta_analysis"
    assert payload["section"][1]["section_type"] == "data_availability"
    assert payload["extraction"]["diagnostics"]["xref_tier"][0]["tier"] == "paren_numeric"


def test_paper_id_does_not_follow_the_doi(demo_paper):
    from bibr.export.json_export import _export_paper_payload

    payload = _export_paper_payload(demo_paper)
    assert payload["metadata"]["doi"] == "10.1234/demo"
    assert payload["paper_id"] == "demo"


def test_producer_names_the_software(export_payload):
    producer = export_payload["extraction"]["producer"]
    assert producer == {"name": "bibr", "version": "0.0.0-test", "build_sha": None}
    assert "bibr_version" not in export_payload["extraction"]


def test_strict_schema_requires_every_emitted_key():
    """``required`` in the published strict schema means *present*: a nullable
    field is required too; only keys a model drops when absent are optional."""
    from bibr.export import models
    from bibr.export.schema_artifact import build_export_schema

    defs = build_export_schema()["$defs"]
    assert defs["BibExport"]["required"] == list(models.BibExport.model_fields)
    extraction = defs["ExtractionExport"]
    assert set(extraction["required"]) == set(models.ExtractionExport.model_fields) - set(
        models.ExtractionExport.OMITTED_WHEN_ABSENT
    )
    assert "required" not in defs["PersonNameExport"]
    # The reader keeps keys optional for a later minor's rows.
    reader = build_export_schema(reader=True)["$defs"]
    assert reader["BibExportReader"]["required"] == ["bib_id"]


def test_nullable_fields_are_type_lists():
    from bibr.export.schema_artifact import build_export_schema

    defs = build_export_schema()["$defs"]
    doi = defs["BibExport"]["properties"]["doi"]
    assert doi["type"] == ["string", "null"]
    assert "anyOf" not in doi
    bib_type = defs["BibExport"]["properties"]["bib_type"]
    assert bib_type["type"] == ["string", "null"] and bib_type["enum"][-1] is None
    target = defs["XrefExport"]["properties"]["target_id"]
    assert target["type"] == ["integer", "null"] and target["minimum"] == 1
    # A nullable model reference keeps its anyOf.
    diagnostics = defs["ExtractionExport"]["properties"]["diagnostics"]
    assert diagnostics["anyOf"][1] == {"type": "null"}


def test_a_nullable_one_value_literal_accepts_null():
    from bibr.export.schema_artifact import _collapse_nullable

    node = {"anyOf": [{"const": "x", "type": "string"}, {"type": "null"}], "default": None}
    assert _collapse_nullable(node) == {
        "type": ["string", "null"],
        "enum": ["x", None],
        "default": None,
    }


def test_an_orcid_in_another_scripts_digits_is_dropped_not_fatal(demo_paper):
    """``\\d`` matches Arabic-Indic or fullwidth digits; no ORCID has them, and one
    reaching the export (an LLM byline parse, OCR) must not fail the paper."""
    from bibr.export.json_export import _export_paper_payload

    demo_paper.metadata.authors[0].orcid = "٠٠٠٠-٠٠٠٢-١٨٢٥-٠٠٩٧"
    assert _export_paper_payload(demo_paper)["author"][0]["orcid"] is None
    demo_paper.metadata.authors[0].orcid = "００００-０００２-１８２５-００９７"
    assert _export_paper_payload(demo_paper)["author"][0]["orcid"] is None


def test_identity_receipt_spells_section_types_like_the_section_rows():
    from dataclasses import dataclass, field

    from bibr.pipeline.stages.export import _build_extraction

    @dataclass
    class Candidate:
        raw: str = "10.1234/x"
        section_type: str | None = "open_data"

    @dataclass
    class Selection:
        selected: Candidate | None = None
        candidates: list = field(default_factory=lambda: [Candidate()])
        issues: list = field(default_factory=list)

    from tests.test_export_units import TestExtractionProvenance, _minimal_paper

    paper = _minimal_paper()
    paper.doi_selection = Selection()
    ctx = TestExtractionProvenance()._ctx(timings={})
    receipt = _build_extraction(ctx, paper)["identity"]["receipt"]
    assert receipt["candidates"][0]["section_type"] == "data_availability"
