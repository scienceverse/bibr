"""Builds the JSON Schema document for the generated export artifact.

Single source of truth for ``docs/schema/bibr-export-v11.schema.json``:
``scripts/generate_schema.py`` (the writer) and
``tests/export/test_schema_artifact.py`` (the snapshot guard) both call
:func:`build_export_schema` instead of calling
``PaperExport.model_json_schema()`` directly and separately layering on
extra keys. That way the two can never diverge — any key this function adds
on top of pydantic's raw output is automatically part of what the test
compares the on-disk file against.
"""

from __future__ import annotations

from bibr.export.models import _SCHEMA_VERSION, OMITTABLE_ROOT_KEYS, PaperExport

# Pydantic v2's ``model_json_schema()`` emits JSON Schema draft 2020-12 but
# does not declare the dialect via ``$schema``. This document is a published
# deliverable (scienceverse/schema publishes it, metacheck vendors it), so
# generic JSON-Schema tooling downstream needs it declared explicitly.
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def build_export_schema() -> dict:
    """Return the full JSON Schema document for :class:`PaperExport`."""
    schema = PaperExport.model_json_schema(by_alias=True)
    # Publish the current writer contract. The runtime model also accepts older
    # v11 payloads for replay, which predate newly always-emitted root tables.
    version = schema["properties"]["schema_version"]
    version.pop("enum", None)
    version["const"] = _SCHEMA_VERSION
    schema["required"] = _required_root_keys(schema)
    return {"$schema": JSON_SCHEMA_DIALECT, **schema}


def _required_root_keys(schema: dict) -> list[str]:
    """Return every root key the exporter always emits, in declaration order.

    Pydantic derives ``required`` from *default-absence*, so fields with a
    default (``bib_match``, ``metadata_match``, ``funding``, ``affiliation``,
    ``qualification_provenance``) drop out of it — even though the exporter
    emits all of them on every payload. Left uncorrected, the published
    artifact would validate payloads bibr never produces and would tell a
    generated reader that four root tables are optional, which is exactly what
    the uniform-tables rule exists to prevent.

    The authority is :data:`bibr.export.models.OMITTABLE_ROOT_KEYS`: everything
    not in it is required. Deriving rather than hardcoding means a root key
    added later is required by default, which is the safe direction.
    """
    return [key for key in schema.get("properties", {}) if key not in OMITTABLE_ROOT_KEYS]
