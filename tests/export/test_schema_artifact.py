import json
from pathlib import Path

import pytest

from bibr.export.models import OMITTABLE_ROOT_KEYS
from bibr.export.schema_artifact import build_export_schema

ARTIFACT = Path(__file__).resolve().parents[2] / "docs" / "schema" / "bibr-export-v11.schema.json"


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
