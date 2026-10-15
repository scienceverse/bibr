"""Example files any producer or reader of the export format can test against.

``tests/fixtures/schema_conformance/`` holds three sets:

- ``valid/``: exports the strict schema (what bibr writes) must accept;
- ``invalid/``: each breaks exactly one rule, and both schemas must reject it;
- ``reader_valid/``: exports from another 12.x writer (a 12.0 export without
  ``extraction.fields``, a later minor with keys this bibr does not know), which
  the strict schema rejects and the reader schema must accept.

Each file is checked against the published JSON Schema documents and the
Python models, so the two stay in agreement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bibr.export import PaperExport, PaperExportReader
from bibr.export.schema_artifact import schema_file_name

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "schema_conformance"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schema"


_STRICT_ONLY = {
    "unknown_root_key",
    "root_validation_block",
    "removed_author_affiliation",
    "section_type_off_vocabulary",
    # A 12.1 writer always writes every column; the reader leaves keys optional
    # so it can read a later minor's output, which may add columns.
    "dropped_column",
}

# Rules only the published strict schema states. "Required" there means the
# key is always *present*; the Python models fill a missing key's default, so
# code can build a row without spelling out every null.
_SCHEMA_ONLY = {"dropped_column"}


def _files(kind: str) -> list[Path]:
    return sorted((ROOT / kind).glob("*.json"))


def _validator(*, reader: bool):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMA_DIR / schema_file_name(reader=reader)).read_text())
    return jsonschema.Draft202012Validator(schema)


@pytest.mark.parametrize("path", _files("valid"), ids=lambda p: p.stem)
def test_valid_examples_pass_both_schemas(path):
    payload = json.loads(path.read_text())
    assert not list(_validator(reader=False).iter_errors(payload))
    assert not list(_validator(reader=True).iter_errors(payload))
    PaperExport.model_validate(payload)


@pytest.mark.parametrize("path", _files("invalid"), ids=lambda p: p.stem)
def test_invalid_examples_fail_both_schemas(path):
    payload = json.loads(path.read_text())
    assert list(_validator(reader=False).iter_errors(payload))
    if path.stem not in _SCHEMA_ONLY:
        with pytest.raises(ValidationError):
            PaperExport.model_validate(payload)
    # The reader tolerates unknown keys and enum values it does not know yet,
    # so those examples are failures for the strict schema only.
    if path.stem not in _STRICT_ONLY:
        assert list(_validator(reader=True).iter_errors(payload))
        with pytest.raises(ValidationError):
            PaperExportReader.model_validate(payload)


@pytest.mark.parametrize("path", _files("reader_valid"), ids=lambda p: p.stem)
def test_newer_minor_examples_pass_only_the_reader(path):
    payload = json.loads(path.read_text())
    assert list(_validator(reader=False).iter_errors(payload))
    assert not list(_validator(reader=True).iter_errors(payload))
    PaperExportReader.model_validate(payload)


def test_the_full_example_has_every_root_key_in_order():
    """``valid/full.json`` is a real export of the shared demo paper."""
    example = json.loads((ROOT / "valid" / "full.json").read_text())
    assert list(example) == list(PaperExport.model_fields)
    # carries every v12 feature
    assert example["xref"][0]["start"] is not None
    assert example["extraction"]["fields"]["title"]["state"] == "extracted"
    assert example["extraction"]["pages"] and example["affiliation_match"]
    assert example["funding_match"] and example["metadata_match"][0]["funder"]
    assert example["extraction"]["warnings"][0]["code"] == "STATEMENT_LEXICAL_FALLBACK"
    assert example["footnote"] and example["figure"][0]["text_id"] is not None


def test_the_enriched_example_carries_every_check_shape():
    """``valid/enriched_check.json`` is the demo paper enriched and checked:
    the 12.3 shapes (update notices, an organization author, lookups, a
    ``bib_check`` row with a note and a retraction) stated in the published
    schemas, as bibr writes them today."""
    path = ROOT / "valid" / "enriched_check.json"
    example = json.loads(path.read_text())
    assert example == _enriched_check_payload()

    (prior,) = (row for row in example["bib_match"] if row["bib_id"] == 1)
    assert prior["updated_by"][0]["type"] == "retraction"
    assert {"literal": "Open Science Collaboration"} in prior["author"]
    assert example["metadata_match"][0]["updated_by"][0]["type"] == "correction"
    lookups = {row["bib_id"]: row for row in example["extraction"]["enrichment"]["lookups"]}
    assert lookups[1]["outcome"] == "doi_hit"
    assert lookups[2] == {"bib_id": 2, "outcome": "doi_404", "registration_agency": "none"}
    assert lookups[3]["outcome"] == "doi_rejected" and lookups[3]["record_year"] == 2013
    assert lookups[4] == {"bib_id": 4, "outcome": "clean_miss"}
    checks = {row["bib_id"]: row for row in example["bib_check"]}
    assert checks[1]["status"] == "matched" and checks[1]["retraction"]["type"] == "retraction"
    assert checks[1]["notes"] == [
        {"field": "volume", "printed": "7", "record": "8", "severity": "minor"}
    ]
    assert checks[2]["status"] == "doi_unregistered"
    assert checks[3]["status"] == "fields_disagree"
    assert checks[3]["notes"][0]["record"] == "2013"
    assert checks[4]["status"] == "not_found"


def _enriched_check_payload() -> dict:
    """The demo paper as an enriched run exports it: its reference matched by
    DOI to a record that was retracted and lists an organization author, the
    paper's own record corrected, a DOI that does not exist, a book DOI of a
    later edition, and a reference Crossref has no record of."""
    from bibr.enrich.check import check_bibs
    from bibr.enrich.lookup import ReferenceLookup
    from bibr.enrich.verify import verify_bibs
    from bibr.export.json_export import _export_paper_payload
    from bibr.models import BibAuthor, MatchUpdate
    from bibr.paper import MatchSource, PaperReference
    from tests.export.conftest import _demo_paper

    paper = _demo_paper(with_refs=True)
    meta = paper.metadata
    prior = meta.references[0]
    record = prior.match[MatchSource.CROSSREF]
    record.score = 100.0
    record.volume = "8"
    record.authors = [
        *record.authors,
        BibAuthor(given="", family="", literal="Open Science Collaboration"),
    ]
    record.updated_by = [
        MatchUpdate(
            type="retraction",
            doi="10.1234/prior.retraction",
            date="2021-03-01",
            source="retraction-watch",
            label="Retraction",
        )
    ]
    prior._lookup = ReferenceLookup(queried=True, doi_status="found", hit="doi", finished=True)
    meta.match[MatchSource.CROSSREF].updated_by = [
        MatchUpdate(type="correction", doi="10.1234/demo.correction", date="2026-02-01")
    ]
    unregistered = PaperReference(
        bib_id=2,
        first_page=None,
        volume=None,
        title="A Study Nobody Registered",
        authors="Roe, R.",
        year=2019,
        container="Journal of Things",
        doi="10.1234/missing",
        bib_type="journal_article",
    )
    unregistered._lookup = ReferenceLookup(
        queried=True, doi_status="not_found", finished=True, registration_agency="none"
    )
    book = PaperReference(
        bib_id=3,
        container=None,
        first_page=None,
        volume=None,
        title="Statistical power analysis for the behavioral sciences",
        authors="Cohen, J.",
        year=1988,
        publisher="Lawrence Erlbaum Associates",
        doi="10.4324/9780203771587",
        bib_type="book",
    )
    book._lookup = ReferenceLookup(queried=True, finished=True)
    book._lookup.reject(2013, "Statistical Power Analysis for the Behavioral Sciences", "book")
    unknown = PaperReference(
        bib_id=4,
        container=None,
        first_page=None,
        volume=None,
        title="A Report Crossref Does Not Hold",
        authors="Poe, E.",
        year=2001,
        bib_type="report",
    )
    unknown._lookup = ReferenceLookup(queried=True, finished=True)
    meta.references = [prior, unregistered, book, unknown]
    payload = _export_paper_payload(paper)
    payload["extraction"]["completed_at"] = "2026-09-22T12:00:00Z"
    verify_bibs(payload)
    check_bibs(payload)
    return payload


def _regenerate_enriched_check_example() -> None:
    """Rewrite ``valid/enriched_check.json`` (run this module)."""
    (ROOT / "valid" / "enriched_check.json").write_text(
        json.dumps(_enriched_check_payload(), indent=2, ensure_ascii=False) + "\n"
    )


def _regenerate_full_example() -> None:
    """Rewrite ``valid/full.json`` from the shared demo paper (run this module)."""
    from bibr.export.json_export import _export_paper_payload
    from bibr.extract.field_decisions import FieldCandidate, FieldDecision, FieldDecisions
    from bibr.extract.statement_scan import lexical_fallback_warning
    from bibr.field_states import FieldScope
    from bibr.paper_contents import PaperFigurePart
    from tests.export.conftest import _demo_paper, as_parsed

    paper = _demo_paper(with_refs=True)
    as_parsed(paper)
    # A pipeline run's field states (12.1): what the run attempted, and the
    # decisions behind the demo values.
    paper.field_scope = FieldScope(references_source="llm")
    decisions = FieldDecisions()
    for field, source, rule in (
        ("title", "llm", "extracted"),
        ("abstract", "llm", "extracted"),
        ("keywords", "llm", "extracted"),
        ("author", "llm", "extracted"),
        ("published", "llm", "extracted"),
        ("journal", "llm", "extracted"),
        ("paper_type", "classifier", "classifier"),
        ("funding_statement", "integrity_statement", "integrity_resolution"),
        ("funding", "llm", "extracted"),
    ):
        candidate = FieldCandidate(field, source, None)
        decisions.record(FieldDecision(field, None, candidate, rule))
    paper.field_decisions = decisions
    paper.contents.sentences[
        1
    ].text = "It replicated prior work [1], t(28) = 3.42, see Table 1 and data."
    paper.contents.sections[1].classification_score = 0.97
    paper.contents.sections[1].classification_source = "exact_alias"
    paper.contents.page_sizes = {1: (612.0, 792.0), 2: (612.0, 792.0)}
    paper.contents.figures[0].parts = [
        PaperFigurePart(page_number=2, bbox=(96.0, 410.0, 512.0, 688.0), image_b64=None)
    ]
    paper.processing_warnings = [lexical_fallback_warning("data_availability")]
    payload = _export_paper_payload(paper)
    payload["extraction"]["completed_at"] = "2026-09-22T12:00:00Z"
    (ROOT / "valid" / "full.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    _regenerate_full_example()
    _regenerate_enriched_check_example()
