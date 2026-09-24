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

from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

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

    In the strict document every key the exporter always writes is
    ``required``, nullable or not: ``required`` means *present*, so a payload
    that drops a column (another producer, a hand-edited file) fails
    validation instead of reaching an R data frame without it. The reader keeps
    pydantic's defaults-derived ``required``, because a key added by a later
    12.x minor must stay optional for it. Both documents spell a nullable
    scalar or array ``"type": [T, "null"]`` rather than pydantic's ``anyOf``,
    the form R and codegen consumers read (metacheck's ``.paper_coerce()``
    takes a column's type from ``type[[1]]``).
    """
    model = PaperExportReader if reader else PaperExport
    schema = model.model_json_schema(by_alias=True)
    schema["required"] = _required_root_keys(schema)
    if not reader:
        _require_emitted_keys(schema)
    schema = _collapse_nullable(schema)
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": SCHEMA_BASE_URL + schema_file_name(reader=reader),
        "$comment": SCHEMA_LICENSE_COMMENT,
        **schema,
    }


def _export_models(model: type[BaseModel], found: dict[str, type[BaseModel]]) -> None:
    """Collect *model* and every model its fields nest, by class name (the
    ``$defs`` key pydantic gives each)."""
    if model.__name__ in found:
        return
    found[model.__name__] = model

    def visit(annotation: Any) -> None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            _export_models(annotation, found)
            return
        if get_origin(annotation) in (Union, UnionType) or get_args(annotation):
            for arg in get_args(annotation):
                visit(arg)

    for field in model.model_fields.values():
        visit(field.annotation)


def _require_emitted_keys(schema: dict) -> None:
    """Mark every property ``required`` that the exporter always writes.

    Only the keys a model lists in ``OMITTED_WHEN_ABSENT`` (dropped by its
    serializer when absent) stay optional.
    """
    models: dict[str, type[BaseModel]] = {}
    _export_models(PaperExport, models)
    for name, definition in schema.get("$defs", {}).items():
        model = models.get(name)
        properties = definition.get("properties")
        if model is None or not properties:
            continue
        omitted = getattr(model, "OMITTED_WHEN_ABSENT", ())
        required = [key for key in properties if key not in omitted]
        if required:
            definition["required"] = required
        else:
            definition.pop("required", None)


def _collapse_nullable(node: Any) -> Any:
    """Rewrite ``{"anyOf": [S, {"type": "null"}]}`` as ``S`` with
    ``"type": [T, "null"]`` wherever S names a single ``type``.

    A nullable enum gets ``null`` among its values, which JSON Schema requires
    for ``null`` to validate. A nullable ``$ref`` keeps its ``anyOf``: a
    reference cannot carry a type list.
    """
    if isinstance(node, list):
        return [_collapse_nullable(item) for item in node]
    if not isinstance(node, dict):
        return node
    node = {key: _collapse_nullable(value) for key, value in node.items()}
    branches = node.get("anyOf")
    null = {"type": "null"}
    if not (isinstance(branches, list) and len(branches) == 2 and null in branches):
        return node
    (other,) = [branch for branch in branches if branch != null]
    if not isinstance(other.get("type"), str) or "$ref" in other:
        return node
    collapsed = {"type": [other["type"], "null"]}
    collapsed.update((key, value) for key, value in other.items() if key != "type")
    if "const" in collapsed:  # a one-value Literal: null must validate too
        collapsed["enum"] = [collapsed.pop("const")]
    if "enum" in collapsed:
        collapsed["enum"] = [*collapsed["enum"], None]
    collapsed.update((key, value) for key, value in node.items() if key != "anyOf")
    return collapsed


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
