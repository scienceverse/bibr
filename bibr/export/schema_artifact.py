"""Builds the JSON Schema documents for the generated export artifacts.

Single source of truth for ``docs/schema/bibr-export-v<major>.schema.json`` and
its lenient reader twin ``docs/schema/bibr-export-v<major>-reader.schema.json``:
``scripts/generate_schema.py`` (the writer) and
``tests/export/test_schema_artifact.py`` (the snapshot guard) both call
:func:`build_export_schema` instead of calling
``PaperExport.model_json_schema()`` directly and separately layering on
extra keys. That way the two can never diverge — any key this function adds
on top of pydantic's raw output is automatically part of what the test
compares the on-disk file against.

The strict artifact describes what this bibr writes: every object forbids
unknown keys and ``schema_version`` is exactly the current version. The reader
artifact describes what a reader of the current major must accept: every
object allows unknown keys and ``schema_version`` may be any
``<major>.<minor>``.

Both are published on the documentation site under :data:`SCHEMA_BASE_URL`,
which is what each document's ``$id`` names, so downstream readers (metacheck,
scienceverse/schema) can reference the contract instead of copying it.
Superseded majors' files stay beside them, frozen.
"""

from __future__ import annotations

from bibr.export.models import (
    _SCHEMA_VERSION,
    OMITTABLE_ROOT_KEYS,
    PaperExport,
    PaperExportReader,
)

# Pydantic v2's ``model_json_schema()`` emits JSON Schema draft 2020-12 but
# does not declare the dialect via ``$schema``. This document is a published
# deliverable (scienceverse/schema publishes it, metacheck vendors it), so
# generic JSON-Schema tooling downstream needs it declared explicitly.
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# ``docs/schema/`` is served at this URL by the documentation site.
SCHEMA_BASE_URL = "https://bibr.org/schema/"

# The schema documents are CC0 so other producers and readers can adopt the
# format; the bibr software itself stays AGPL (see README "License").
SCHEMA_LICENSE_COMMENT = (
    "This schema document is dedicated to the public domain under CC0 1.0 "
    "(https://creativecommons.org/publicdomain/zero/1.0/). The bibr software that "
    "produces it is AGPL-3.0-or-later."
)

SCHEMA_MAJOR = _SCHEMA_VERSION.split(".")[0]


def schema_file_name(*, reader: bool = False) -> str:
    """File name of the current major's strict or reader schema artifact."""
    return f"bibr-export-v{SCHEMA_MAJOR}{'-reader' if reader else ''}.schema.json"


def build_export_schema(*, reader: bool = False) -> dict:
    """Return the full JSON Schema document for :class:`PaperExport`.

    With ``reader=True``, return the document for the lenient
    :data:`~bibr.export.models.PaperExportReader` instead.
    """
    model = PaperExportReader if reader else PaperExport
    schema = model.model_json_schema(by_alias=True)
    schema["required"] = _required_root_keys(schema)
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": SCHEMA_BASE_URL + schema_file_name(reader=reader),
        "$comment": SCHEMA_LICENSE_COMMENT,
        **schema,
    }


def _required_root_keys(schema: dict) -> list[str]:
    """Return every root key the exporter always emits, in declaration order.

    Pydantic derives ``required`` from *default-absence*, so fields with a
    default (``affiliation``, ``funding``, ``metadata_match``, ``bib_match``)
    drop out of it — even though the exporter emits all of them on every
    payload. Left uncorrected, the published
    artifact would validate payloads bibr never produces and would tell a
    generated reader that four root tables are optional, which is exactly what
    the uniform-tables rule exists to prevent.

    The authority is :data:`bibr.export.models.OMITTABLE_ROOT_KEYS`: everything
    not in it is required. Deriving rather than hardcoding means a root key
    added later is required by default, which is the safe direction.
    """
    return [key for key in schema.get("properties", {}) if key not in OMITTABLE_ROOT_KEYS]
