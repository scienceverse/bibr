"""Export schema 12.3: per-field registry outcomes and the record's subtitle."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bibr.export import PaperExport, PaperExportReader
from bibr.export.models import FieldRecordExport, MetadataMatchExport

FULL = (
    Path(__file__).resolve().parents[1] / "fixtures" / "schema_conformance" / "valid" / "full.json"
)


def _full() -> dict:
    return json.loads(FULL.read_text(encoding="utf-8"))


def test_a_field_record_without_registry_serializes_as_in_12_2():
    record = FieldRecordExport(state="extracted", source="llm", issues=[], rule="extracted")

    assert record.model_dump() == {
        "state": "extracted",
        "source": "llm",
        "issues": [],
        "rule": "extracted",
    }


def test_a_field_record_keeps_its_registry_outcome():
    record = FieldRecordExport.model_validate(
        {
            "state": "extracted",
            "source": "llm",
            "issues": [],
            "registry": {"service": "crossref", "outcome": "disagree"},
        }
    )

    assert record.model_dump()["registry"] == {"service": "crossref", "outcome": "disagree"}


@pytest.mark.parametrize(
    "registry",
    [
        {"service": "crossref", "outcome": "maybe"},
        {"service": "crossref"},
        {"service": "crossref", "outcome": "agree", "score": 1.0},
    ],
)
def test_the_writer_model_rejects_a_malformed_registry(registry):
    payload = _full()
    payload["extraction"]["fields"]["title"]["registry"] = registry

    with pytest.raises(ValidationError):
        PaperExport.model_validate(payload)


def test_a_12_2_export_is_valid_reader_input():
    payload = _full()
    payload["schema_version"] = "12.2"
    for record in payload["extraction"]["fields"].values():
        record.pop("registry", None)
    for row in payload["metadata_match"]:
        row.pop("subtitle", None)

    model = PaperExportReader.model_validate(payload)

    assert model.extraction.fields.title.registry is None


def test_metadata_match_writes_subtitle_only_when_present():
    row = {"service": "crossref", "score": 1.0, "title": "Main"}

    assert "subtitle" not in MetadataMatchExport.model_validate(row).model_dump()
    with_subtitle = MetadataMatchExport.model_validate({**row, "subtitle": "The rest"})
    assert with_subtitle.model_dump()["subtitle"] == "The rest"


def test_the_crossref_subtitle_reaches_metadata_match(demo_paper):
    from bibr.enrich.references import _build_match
    from bibr.enrich.schemas import CrossrefWorkItem
    from bibr.export.json_export import _export_paper_payload
    from bibr.models import MatchSource

    item = CrossrefWorkItem.from_raw(
        {
            "DOI": "10.1234/demo",
            "title": ["A Demonstration Paper"],
            "subtitle": ["Methods and <i>Results</i>"],
            "author": [{"given": "Jane", "family": "Smith"}],
        }
    )
    assert item.subtitle == "Methods and Results"
    demo_paper.metadata.match = {MatchSource.CROSSREF: _build_match(item, 1.0)}

    payload = _export_paper_payload(demo_paper)

    assert payload["metadata_match"][0]["subtitle"] == "Methods and Results"
    PaperExport.model_validate(payload)


def test_a_record_without_subtitle_exports_no_subtitle_key(demo_paper):
    from bibr.export.json_export import _export_paper_payload

    payload = _export_paper_payload(demo_paper)

    assert payload["metadata_match"]
    assert "subtitle" not in payload["metadata_match"][0]


def test_an_empty_crossref_subtitle_is_no_subtitle():
    from bibr.enrich.schemas import CrossrefWorkItem

    assert CrossrefWorkItem.from_raw({"title": ["Main"], "subtitle": []}).subtitle is None
    assert CrossrefWorkItem.from_raw({"title": ["Main"], "subtitle": [""]}).subtitle is None


def test_the_full_example_round_trips_with_registry():
    payload = _full()
    model = PaperExport.model_validate(copy.deepcopy(payload))

    dumped = model.model_dump(mode="json")

    assert dumped["extraction"]["fields"] == payload["extraction"]["fields"]
    assert dumped["metadata_match"] == payload["metadata_match"]
