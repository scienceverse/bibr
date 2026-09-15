"""Document envelopes containing independently owned paper exports.

The root dispatch key differs from ``PaperExport.schema_version``. A document
record ID is local to this extraction result; nested paper IDs may repeat.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bibr.export.models import PaperExport, SourceExport

DOCUMENT_SCHEMA_VERSION = "1.0"
_STRICT = ConfigDict(extra="forbid")


class DocumentDiagnosticsExport(BaseModel):
    """Detector inventory retained independently of per-record extraction success."""

    model_config = _STRICT

    detected_record_ids: list[str] = Field(
        description="Every detected candidate record ID, including failed and unresolved records."
    )
    reason_flags: list[str] = Field(default_factory=list)
    unassigned_source_text_ids: list[int] = Field(
        default_factory=list,
        description="Parsed source text not assigned to any record scope, including leading fragments or unresolved areas; not proof of another article.",
    )


class DocumentRecordExport(BaseModel):
    """One document-local candidate and its extraction outcome.

    Source identifiers refer to the original document before per-record
    normalization. Author, bibliography and cross-reference IDs inside ``paper``
    belong only to that record, even when another record reuses their numbers.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"status": {"const": "extracted"}}},
                    "then": {
                        "properties": {
                            "paper": {
                                "type": "object",
                                "properties": {
                                    "validation": {
                                        "anyOf": [
                                            {"type": "null"},
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "blocking": {"maximum": 0},
                                                    "promotable": {"const": True},
                                                    "issues": {
                                                        "items": {
                                                            "properties": {
                                                                "blocking": {"const": False}
                                                            }
                                                        }
                                                    },
                                                },
                                            },
                                        ]
                                    }
                                },
                            },
                            "partial_paper": {"type": "null"},
                            "error": {"type": "null"},
                        }
                    },
                    "else": {"properties": {"paper": {"type": "null"}}},
                },
                {
                    "if": {
                        "required": ["partial_paper"],
                        "properties": {"partial_paper": {"type": "object"}},
                    },
                    "then": {
                        "properties": {
                            "status": {"enum": ["unresolved", "failed"]},
                            "partial_paper": {
                                "required": ["validation"],
                                "properties": {
                                    "validation": {
                                        "type": "object",
                                        "required": ["blocking", "promotable", "issues"],
                                        "properties": {
                                            "blocking": {"minimum": 1},
                                            "promotable": {"const": False},
                                            "issues": {
                                                "contains": {
                                                    "required": ["blocking"],
                                                    "properties": {"blocking": {"const": True}},
                                                }
                                            },
                                        },
                                    }
                                },
                            },
                        }
                    },
                },
            ]
        },
    )

    record_id: str = Field(min_length=1, pattern=r"\S", description="Document-local record key.")
    status: Literal["extracted", "unresolved", "failed"]
    paper: PaperExport | None = Field(
        description="A scoped paper export exactly when status is extracted; otherwise null."
    )
    partial_paper: PaperExport | None = Field(
        default=None,
        description="Independently retained scoped fields for an unresolved or failed record, "
        "with explicit blocking validation. Never an extraction success or a promotable paper.",
    )
    source_text_ids: list[int] = Field(default_factory=list)
    source_section_ids: list[int] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    reason_flags: list[str] = Field(default_factory=list)
    error: str | None = Field(
        default=None,
        max_length=500,
        description="Optional caller-sanitized failure summary, excluding source text and secrets.",
    )

    @model_validator(mode="after")
    def _status_matches_paper(self) -> DocumentRecordExport:
        if self.status == "extracted":
            if self.partial_paper is not None:
                raise ValueError("extracted records must have partial_paper=null")
            if self.paper is None:
                raise ValueError("extracted records require a paper export")
            if self.error is not None:
                raise ValueError("extracted records cannot carry a failure error")
            validation = self.paper.validation
            if validation is not None and (
                validation.blocking > 0
                or not validation.promotable
                or any(issue.blocking for issue in validation.issues)
            ):
                raise ValueError("extracted records cannot carry blocking paper validation")
        elif self.paper is not None:
            raise ValueError("unresolved and failed records must have paper=null")
        if self.partial_paper is not None:
            validation = self.partial_paper.validation
            if (
                validation is None
                or validation.promotable
                or validation.blocking < 1
                or not any(issue.blocking for issue in validation.issues)
            ):
                raise ValueError("partial_paper requires explicit blocking validation")
        for payload in (self.paper, self.partial_paper):
            if payload is not None and any(
                variant.record_id != self.record_id for variant in payload.metadata_variant
            ):
                raise ValueError("printed metadata variants must belong to their enclosing record")
        return self


class DocumentExport(BaseModel):
    """An input document and every detected article candidate.

    Complete means all detected candidates were extracted, not that the detector
    is guaranteed to have found every article. Runtime validation additionally
    checks unique record IDs, exact detector-inventory coverage, and identical
    source identity on nested papers; JSON Schema cannot compare those values.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"status": {"const": "complete"}}},
                    "then": {
                        "properties": {
                            "records": {
                                "minItems": 1,
                                "items": {"properties": {"status": {"const": "extracted"}}},
                            }
                        }
                    },
                },
                {
                    "if": {"properties": {"status": {"const": "partial"}}},
                    "then": {
                        "properties": {
                            "records": {
                                "allOf": [
                                    {
                                        "contains": {
                                            "properties": {"status": {"const": "extracted"}}
                                        }
                                    },
                                    {
                                        "contains": {
                                            "properties": {
                                                "status": {"enum": ["unresolved", "failed"]}
                                            }
                                        }
                                    },
                                ]
                            }
                        }
                    },
                },
                {
                    "if": {"properties": {"status": {"const": "unresolved"}}},
                    "then": {
                        "properties": {
                            "records": {
                                "items": {
                                    "properties": {"status": {"enum": ["unresolved", "failed"]}}
                                }
                            }
                        }
                    },
                },
            ]
        },
    )

    document_schema_version: Literal["1.0"]
    document_id: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Content-based document identity supplied by the producer, independent of DOI.",
    )
    source: SourceExport
    records: list[DocumentRecordExport]
    status: Literal["complete", "partial", "unresolved"]
    diagnostics: DocumentDiagnosticsExport

    @model_validator(mode="after")
    def _consistent_inventory_and_status(self) -> DocumentExport:
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("record_id values must be unique within a document")
        detected_ids = self.diagnostics.detected_record_ids
        if len(detected_ids) != len(set(detected_ids)):
            raise ValueError("detected_record_ids must be unique within a document")
        if set(record_ids) != set(detected_ids):
            raise ValueError("records must retain every detected_record_id and no additional IDs")
        extracted = sum(record.status == "extracted" for record in self.records)
        expected = (
            "complete"
            if self.records and extracted == len(self.records)
            else "partial"
            if extracted
            else "unresolved"
        )
        if self.status != expected:
            raise ValueError(f"document status must be {expected!r} for these record outcomes")
        for record in self.records:
            for payload in (record.paper, record.partial_paper):
                if payload is not None and payload.source != self.source:
                    raise ValueError("nested paper source must match the document source")
        return self


def build_document_schema() -> dict:
    """Generate the document writer schema with the current nested paper contract."""
    from bibr.export.schema_artifact import JSON_SCHEMA_DIALECT, build_export_schema

    schema = DocumentExport.model_json_schema()
    paper_schema = build_export_schema()
    schema["$defs"].update(paper_schema["$defs"])
    schema["$defs"]["PaperExport"] = {
        key: value for key, value in paper_schema.items() if key not in {"$schema", "$defs"}
    }
    return {"$schema": JSON_SCHEMA_DIALECT, **schema}


__all__ = [
    "DOCUMENT_SCHEMA_VERSION",
    "DocumentDiagnosticsExport",
    "DocumentExport",
    "DocumentRecordExport",
    "build_document_schema",
]
