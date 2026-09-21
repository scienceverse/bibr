import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bibr.export import PaperExport, PaperExportReader, export_paper_to_json
from bibr.export.models import OMITTABLE_ROOT_KEYS
from bibr.export.schema_artifact import build_export_schema

ARTIFACT = Path(__file__).resolve().parents[2] / "docs" / "schema" / "bibr-export-v11.schema.json"
READER_ARTIFACT = ARTIFACT.with_name("bibr-export-v11-reader.schema.json")


def test_artifact_matches_the_models():
    """Regenerate with: uv run python scripts/generate_schema.py"""
    assert ARTIFACT.exists(), f"missing {ARTIFACT}; run scripts/generate_schema.py"
    on_disk = json.loads(ARTIFACT.read_text())
    current = build_export_schema()
    assert on_disk == current, (
        "docs/schema/bibr-export-v11.schema.json is stale relative to "
        "bibr/export/models.py — regenerate with: "
        "uv run python scripts/generate_schema.py"
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["v11_payload", "v11_payload_refs_off"],
)
def test_required_covers_every_root_key_the_exporter_always_emits(fixture_name, request):
    """The artifact must not call a root key optional that bibr always emits.

    Pydantic derives ``required`` from default-absence, so five always-emitted
    root keys (``bib_match``, ``metadata_match``, ``funding``, ``affiliation``,
    ``qualification_provenance``) fall out of it unless corrected. An artifact
    missing them validates payloads bibr never produces and tells a generated
    reader that four root tables are optional — the exact failure the
    uniform-tables rule exists to prevent. Asserted against the *on-disk*
    artifact, since that is what scienceverse/schema publishes.
    """
    payload = request.getfixturevalue(fixture_name)
    required = set(json.loads(ARTIFACT.read_text())["required"])

    always_emitted = set(payload) - set(OMITTABLE_ROOT_KEYS)
    assert always_emitted <= required, (
        f"root keys emitted but not required: {sorted(always_emitted - required)}"
    )
    # And nothing is required that this payload does not carry — otherwise the
    # artifact would reject a real export.
    assert required <= set(payload), (
        f"root keys required but not emitted: {sorted(required - set(payload))}"
    )


def test_required_excludes_only_the_omittable_root_keys():
    """``required`` is the property list minus the two genuinely-optional keys."""
    on_disk = json.loads(ARTIFACT.read_text())
    assert set(on_disk["properties"]) - set(on_disk["required"]) == set(OMITTABLE_ROOT_KEYS)


def test_artifact_uses_defs_refs_metacheck_can_resolve():
    """metacheck's paper() constructor resolves "#/$defs/Name" refs."""
    on_disk = json.loads(ARTIFACT.read_text())
    assert "$defs" in on_disk
    for prop in on_disk["properties"].values():
        ref = prop.get("$ref") or (prop.get("items") or {}).get("$ref")
        if ref:
            assert ref.startswith("#/$defs/"), ref


def test_reader_artifact_matches_the_models():
    """Regenerate with: uv run python scripts/generate_schema.py"""
    assert READER_ARTIFACT.exists(), f"missing {READER_ARTIFACT}; run scripts/generate_schema.py"
    on_disk = json.loads(READER_ARTIFACT.read_text())
    assert on_disk == build_export_schema(reader=True), (
        "docs/schema/bibr-export-v11-reader.schema.json is stale relative to "
        "bibr/export/models.py — regenerate with: "
        "uv run python scripts/generate_schema.py"
    )


def test_reader_artifact_relaxes_only_unknown_keys_and_the_minor_version():
    strict = json.loads(ARTIFACT.read_text())
    reader = json.loads(READER_ARTIFACT.read_text())

    assert reader["additionalProperties"] is True
    assert {name: d["additionalProperties"] for name, d in reader["$defs"].items()} == {
        f"{name}Reader": True for name in strict["$defs"]
    }
    assert reader["properties"].keys() == strict["properties"].keys()
    assert reader["required"] == strict["required"]
    assert strict["properties"]["schema_version"]["const"] == "11.0"
    assert reader["properties"]["schema_version"]["pattern"] == r"^11\.[0-9]+$"


def _with_next_minor_fields(payload: dict) -> dict:
    """Mimic a later 11.x writer: next minor version, unknown keys at several depths."""
    payload["schema_version"] = "11.1"
    payload["future_block"] = {"enabled": True}
    payload["metadata"]["language"] = "en"
    payload["section"][0]["numbering"] = "1."
    payload["text"][0]["language"] = "en"
    payload["bib"][0]["raw"] = "Doe J. A title. 2020."
    payload["table"][0]["cell_count"] = 4
    payload["extraction"]["settings"]["future_knob"] = "on"
    return payload


def _validator(artifact: Path):
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft202012Validator(json.loads(artifact.read_text()))


def test_reader_artifact_and_model_accept_a_newer_11x_export(demo_paper):
    payload = _with_next_minor_fields(json.loads(json.dumps(export_paper_to_json(demo_paper))))

    assert not list(_validator(READER_ARTIFACT).iter_errors(payload))
    assert PaperExportReader.model_validate(payload).schema_version == "11.1"

    # The strict artifact and the producer model keep rejecting it.
    assert list(_validator(ARTIFACT).iter_errors(payload))
    with pytest.raises(ValidationError):
        PaperExport.model_validate(payload)


@pytest.mark.parametrize("version", ["12.0", "10.9"])
def test_reader_artifact_and_model_reject_another_major_version(demo_paper, version):
    payload = _with_next_minor_fields(json.loads(json.dumps(export_paper_to_json(demo_paper))))
    payload["schema_version"] = version

    assert list(_validator(READER_ARTIFACT).iter_errors(payload))
    with pytest.raises(ValidationError, match="schema_version"):
        PaperExportReader.model_validate(payload)
