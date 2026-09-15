"""Typed export models and JSON serialization for Paper objects.

The models come from :mod:`bibr.export.models` — the canonical home for the
v11 schema types. :mod:`bibr.export.json_export` owns the *serialization*
(building a payload from a :class:`~bibr.paper.Paper`) and imports the models
from ``models`` like everyone else, so it is not in the import path for a
caller that only wants a type.
"""

from bibr.export.document_models import (
    DocumentDiagnosticsExport,
    DocumentExport,
    DocumentRecordExport,
)
from bibr.export.json_export import (
    build_paper_export,
    export_paper_to_json,
    validate_export,
)
from bibr.export.models import (
    AffiliationExport,
    AuthorExport,
    BibExport,
    BibMatchExport,
    CaptionAssignmentExport,
    CaptionAssignmentReceiptExport,
    CaptionCandidateExport,
    EqExport,
    FigureExport,
    FigurePartExport,
    MetadataExport,
    MetadataMatchExport,
    MetadataVariantExport,
    PaperExport,
    PersonNameExport,
    ProvenanceExport,
    ReferenceSegmentationAttemptExport,
    ReferenceYieldExport,
    SectionExport,
    SourceExport,
    TableExport,
    TablePartExport,
    TextExport,
    UrlExport,
    ValidationExport,
    ValidationIssueExport,
    XrefExport,
)
from bibr.export.qualification_provenance import (
    DeploymentIdentity,
    build_qualification_provenance,
)

__all__ = [
    "DocumentDiagnosticsExport",
    "DocumentExport",
    "DocumentRecordExport",
    "DeploymentIdentity",
    "AffiliationExport",
    "AuthorExport",
    "BibExport",
    "BibMatchExport",
    "EqExport",
    "CaptionAssignmentExport",
    "CaptionAssignmentReceiptExport",
    "CaptionCandidateExport",
    "FigureExport",
    "FigurePartExport",
    "MetadataExport",
    "MetadataMatchExport",
    "MetadataVariantExport",
    "PaperExport",
    "PersonNameExport",
    "ProvenanceExport",
    "ReferenceSegmentationAttemptExport",
    "ReferenceYieldExport",
    "SectionExport",
    "SourceExport",
    "TableExport",
    "TablePartExport",
    "TextExport",
    "UrlExport",
    "ValidationExport",
    "ValidationIssueExport",
    "XrefExport",
    "build_paper_export",
    "build_qualification_provenance",
    "export_paper_to_json",
    "validate_export",
]
