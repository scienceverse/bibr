import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bibr.export import PaperExport, PaperExportReader, export_paper_to_json
from bibr.export.models import OMITTABLE_ROOT_KEYS
from bibr.export.schema_artifact import SCHEMA_BASE_URL, build_export_schema, schema_file_name

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schema"
ARTIFACT = SCHEMA_DIR / schema_file_name()
READER_ARTIFACT = SCHEMA_DIR / schema_file_name(reader=True)


def test_artifact_matches_the_models():
    """Regenerate with: uv run python scripts/generate_schema.py"""
    assert ARTIFACT.exists(), f"missing {ARTIFACT}; run scripts/generate_schema.py"
    on_disk = json.loads(ARTIFACT.read_text())
    current = build_export_schema()
    assert on_disk == current, (
        f"docs/schema/{ARTIFACT.name} is stale relative to "
        "bibr/export/models.py — regenerate with: "
        "uv run python scripts/generate_schema.py"
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["export_payload", "export_payload_refs_off"],
)
def test_required_covers_every_root_key_the_exporter_always_emits(fixture_name, request):
    """The artifact must not call a root key optional that bibr always emits.

    Pydantic derives ``required`` from default-absence, so four always-emitted
    root tables (``affiliation``, ``funding``, ``metadata_match``,
    ``bib_match``) fall out of it unless corrected. An artifact missing them
    validates payloads bibr never produces and tells a generated reader that
    those tables are optional — the exact failure the
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
    """``required`` is the property list minus the omittable keys (none in v12)."""
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
        f"docs/schema/{READER_ARTIFACT.name} is stale relative to "
        "bibr/export/models.py — regenerate with: "
        "uv run python scripts/generate_schema.py"
    )


def test_reader_artifact_relaxes_only_unknown_keys_enum_values_and_the_minor_version():
    strict = json.loads(ARTIFACT.read_text())
    reader = json.loads(READER_ARTIFACT.read_text())

    # Every closed vocabulary is open in the reader, with the known values kept
    # as examples; nothing else about the field changes.
    for name, definition in strict["$defs"].items():
        for field, prop in definition.get("properties", {}).items():
            text = json.dumps(prop)
            if '"enum"' not in text and '"const"' not in text:
                continue
            opened = reader["$defs"][f"{name}Reader"]["properties"][field]
            assert '"enum"' not in json.dumps(opened), (name, field)
            assert opened["examples"], (name, field)
            assert opened["description"] == prop["description"]

    assert reader["additionalProperties"] is True
    assert {name: d["additionalProperties"] for name, d in reader["$defs"].items()} == {
        f"{name}Reader": True for name in strict["$defs"]
    }
    assert reader["properties"].keys() == strict["properties"].keys()
    assert reader["required"] == strict["required"]
    assert strict["properties"]["schema_version"]["const"] == "12.1"
    assert reader["properties"]["schema_version"]["pattern"] == r"^12\.[0-9]+$"


def _with_next_minor_fields(payload: dict) -> dict:
    """Mimic a later 12.x writer: next minor version, unknown keys at several depths."""
    payload["schema_version"] = "12.2"
    payload["future_block"] = {"enabled": True}
    payload["metadata"]["subtitle"] = "A sequel"
    payload["section"][0]["numbering"] = "1."
    payload["text"][0]["language"] = "en"
    payload["bib"][0]["raw"] = "Doe J. A title. 2020."
    payload["table"][0]["cell_count"] = 4
    payload["extraction"]["settings"]["future_knob"] = "on"
    return payload


def _validator(artifact: Path):
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft202012Validator(json.loads(artifact.read_text()))


def test_reader_artifact_and_model_accept_a_newer_minor_export(demo_paper):
    payload = _with_next_minor_fields(json.loads(json.dumps(export_paper_to_json(demo_paper))))

    assert not list(_validator(READER_ARTIFACT).iter_errors(payload))
    assert PaperExportReader.model_validate(payload).schema_version == "12.2"

    # The strict artifact and the producer model keep rejecting it.
    assert list(_validator(ARTIFACT).iter_errors(payload))
    with pytest.raises(ValidationError):
        PaperExport.model_validate(payload)


def test_reader_accepts_enum_values_a_later_minor_adds(demo_paper):
    """A 12.2 writer may add a value to a closed vocabulary; the strict model
    and artifact reject it, the reader model and artifact keep it."""
    payload = json.loads(json.dumps(export_paper_to_json(demo_paper)))
    payload["schema_version"] = "12.2"
    payload["section"][0]["section_type"] = "preregistration"
    payload["bib"][0]["bib_type"] = "patent"
    payload["bib_match"][0]["service"] = "semantic-scholar"
    payload["xref"][0]["xref_type"] = "code"

    assert not list(_validator(READER_ARTIFACT).iter_errors(payload))
    model = PaperExportReader.model_validate(payload)
    assert model.section[0].section_type == "preregistration"
    assert model.bib_match[0].service == "semantic-scholar"

    assert list(_validator(ARTIFACT).iter_errors(payload))
    with pytest.raises(ValidationError):
        PaperExport.model_validate(payload)


@pytest.mark.parametrize("version", ["13.0", "11.0"])
def test_reader_artifact_and_model_reject_another_major_version(demo_paper, version):
    payload = _with_next_minor_fields(json.loads(json.dumps(export_paper_to_json(demo_paper))))
    payload["schema_version"] = version

    assert list(_validator(READER_ARTIFACT).iter_errors(payload))
    with pytest.raises(ValidationError, match="schema_version"):
        PaperExportReader.model_validate(payload)


def test_artifacts_have_a_stable_id_on_the_docs_site():
    """``$id`` names where the file is published, so readers can reference the
    contract instead of vendoring a copy."""
    for artifact in (ARTIFACT, READER_ARTIFACT):
        assert json.loads(artifact.read_text())["$id"] == SCHEMA_BASE_URL + artifact.name


def _properties(schema: dict):
    """Yield ``(owner, name, property)`` for every object property in *schema*."""
    yield from (("root", name, prop) for name, prop in schema["properties"].items())
    for owner, definition in schema["$defs"].items():
        for name, prop in (definition.get("properties") or {}).items():
            yield owner, name, prop


def test_every_property_and_definition_is_described():
    """The generated schema is the only contract metacheck and
    scienceverse/schema get, so a field without a description is a gap in it."""
    on_disk = json.loads(ARTIFACT.read_text())
    undescribed = [
        f"{owner}.{name}"
        for owner, name, prop in _properties(on_disk)
        if not prop.get("description")
    ]
    assert not undescribed, f"add a Field(description=...) for: {undescribed}"
    assert not [name for name, d in on_disk["$defs"].items() if not d.get("description")]


def test_superseded_major_artifacts_stay_published():
    """Readers of older exports still resolve the schema they were written to."""
    for name in ("bibr-export-v10.schema.json", "bibr-export-v11.schema.json"):
        assert (SCHEMA_DIR / name).exists(), name
