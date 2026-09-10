"""Durable extraction artifact contracts and replay validation."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext
    from bibr.pipeline.state import FileState

# Bumped to "2" for v11: the persisted paper-enrichment key was renamed
# ``info_match`` -> ``metadata_match``. A v1 sidecar now fails its own version
# check with an accurate message instead of the misleading core-version one.
ENRICHMENT_SIDECAR_SCHEMA_VERSION = "2"
CORE_SCHEMA_VERSION = "11.0"
# v11 is a clean break: a v10 core cannot be replayed into a v11 payload, so
# the gate accepts exactly one version.
SUPPORTED_CORE_SCHEMA_VERSIONS = frozenset({CORE_SCHEMA_VERSION})
CROSSREF_ENRICHMENT_SCHEMA_REVISION = "crossref-v1"


class RunState(StrEnum):
    STARTED = "started"
    CORE_WRITTEN = "core_written"
    ENRICHMENT_PARTIAL = "enrichment_partial"
    ENRICHMENT_COMPLETE = "enrichment_complete"
    FAILED = "failed"
    CUTOFF_INTERRUPTED = "cutoff_interrupted"


class ArtifactDisposition(StrEnum):
    PROMOTABLE = "promotable"
    REFERENCES_INCOMPLETE = "references_incomplete"
    IDENTITY_CONFLICT = "identity_conflict"
    SOURCE_INTEGRITY = "source_integrity"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class EnrichmentSidecar:
    """The minimum external-enrichment delta needed to replay a core payload."""

    schema_version: str
    core_sha256: str
    settings_digest: str
    completeness: Literal["partial", "complete"]
    bib_match: tuple[dict, ...] = ()
    metadata_match: tuple[dict, ...] = ()
    warnings: tuple[str, ...] = ()
    detail: str | None = None

    def to_dict(self) -> dict:
        value = asdict(self)
        # Keep the common complete sidecar compact and backward compatible.
        if not self.warnings:
            value.pop("warnings")
        if self.detail is None:
            value.pop("detail")
        return value

    @classmethod
    def from_dict(cls, value: dict) -> EnrichmentSidecar:
        return cls(
            schema_version=str(value.get("schema_version", "")),
            core_sha256=str(value.get("core_sha256", "")),
            settings_digest=str(value.get("settings_digest", "")),
            completeness=value.get("completeness"),
            bib_match=tuple(value.get("bib_match") or ()),
            metadata_match=tuple(value.get("metadata_match") or ()),
            warnings=tuple(str(item)[:512] for item in (value.get("warnings") or ())),
            detail=(str(value["detail"])[:512] if value.get("detail") is not None else None),
        )


class ArtifactSink(Protocol):
    def write_core(self, fs: FileState, payload: dict) -> str: ...

    def write_enrichment(self, fs: FileState, sidecar: EnrichmentSidecar) -> None: ...

    def read_enrichment(self, fs: FileState) -> EnrichmentSidecar: ...

    def materialize(self, fs: FileState, payload: dict) -> None: ...

    def record(self, fs: FileState, state: RunState, *, detail: str | None = None) -> None: ...


class ArtifactReplayError(ValueError):
    """An enrichment sidecar does not belong to the requested core/configuration."""


def canonical_json_bytes(value: object) -> bytes:
    """Serialize canonical JSON for stable content identity."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def enrichment_settings_digest(ctx: PipelineContext) -> str:
    """Hash result-affecting enrichment settings without persisting secrets."""

    crossref = ctx.settings.crossref
    resolver = ctx.settings.resolver
    settings = {
        "schema_revision": CROSSREF_ENRICHMENT_SCHEMA_REVISION,
        "enabled": ctx.config.enrichment_enabled(ctx.settings),
        "consolidate": ctx.config.consolidate or crossref.consolidate,
        "enrich_concurrency": crossref.enrich_concurrency,
        "enrich_timeout": crossref.enrich_timeout,
        "request_timeout": crossref.request_timeout,
        "rate_limit_rpm": crossref.rate_limit_rpm,
        "cache_size": crossref.cache_size,
        "redis_cache": crossref.redis_cache,
        "cache_ttl_seconds": crossref.cache_ttl_seconds,
        "resolver": {
            "url": resolver.url,
            "enabled": bool(resolver.enrich),
            "timeout": resolver.timeout,
            "limit": resolver.limit,
            "search_concurrency": resolver.search_concurrency,
            "authoritative": resolver.authoritative,
        },
    }
    return canonical_json_sha256(settings)


def disposition_for_issues(issues) -> ArtifactDisposition:
    """Derive one terminal artifact disposition from blocking issue codes.

    Source corruption outranks identity disagreement because the latter may be
    a consequence of processing the wrong/corrupt bytes. Identity disagreement
    then outranks incomplete references; both keep their diagnostic artifact.
    """

    def _field(issue, name, default=None):
        return (
            issue.get(name, default) if isinstance(issue, dict) else getattr(issue, name, default)
        )

    blocking_codes = {
        _field(issue, "code") for issue in issues if bool(_field(issue, "blocking", False))
    }
    if "VAL_SOURCE_INTEGRITY" in blocking_codes:
        return ArtifactDisposition.SOURCE_INTEGRITY
    if blocking_codes & {"VAL_EXPECTED_ID_MISMATCH", "VAL_EXPECTED_ID_MISSING"}:
        return ArtifactDisposition.IDENTITY_CONFLICT
    if "VAL_REFERENCES_INCOMPLETE" in blocking_codes:
        return ArtifactDisposition.REFERENCES_INCOMPLETE
    if blocking_codes:
        return ArtifactDisposition.BLOCKED
    return ArtifactDisposition.PROMOTABLE


def mark_enrichment_pending(payload: dict) -> dict:
    """Add a temporary fail-closed promotion gate to an enrichment-bound core."""

    validation = payload.setdefault(
        "validation",
        {"errors": 0, "warnings": 0, "blocking": 0, "promotable": True, "issues": []},
    )
    issues = validation.setdefault("issues", [])
    if not any(issue.get("code") == "VAL_ENRICHMENT_PENDING" for issue in issues):
        issues.append(
            {
                "code": "VAL_ENRICHMENT_PENDING",
                "severity": "error",
                "message": "Optional enrichment was requested but has not completed",
                "origin_stage": "core_checkpoint",
                "evidence_ids": [],
                "count": 1,
                "blocking": True,
            }
        )
        validation["errors"] = int(validation.get("errors") or 0) + 1
        validation["blocking"] = int(validation.get("blocking") or 0) + 1
    validation["promotable"] = False
    # The pending gate lives only in ``validation.issues``; it is deliberately
    # not mirrored into ``extraction.warnings`` (see _apply_output_validation).
    return payload


def _clear_enrichment_pending(payload: dict) -> None:
    validation = payload.get("validation") or {}
    issues = validation.get("issues") or []
    removed = [issue for issue in issues if issue.get("code") == "VAL_ENRICHMENT_PENDING"]
    if not removed:
        return
    validation["issues"] = [
        issue for issue in issues if issue.get("code") != "VAL_ENRICHMENT_PENDING"
    ]
    validation["errors"] = max(0, int(validation.get("errors") or 0) - len(removed))
    validation["blocking"] = max(0, int(validation.get("blocking") or 0) - len(removed))
    validation["promotable"] = validation["blocking"] == 0


def make_enrichment_sidecar(
    enriched_payload: dict,
    *,
    core_sha256: str,
    settings_digest: str,
    completeness: Literal["partial", "complete"],
    warnings: tuple[str, ...] = (),
    detail: str | None = None,
) -> EnrichmentSidecar:
    """Extract only replayable enrichment rows from a fully exported payload."""

    bib_match = sorted(
        copy.deepcopy(enriched_payload.get("bib_match") or ()),
        key=lambda row: (
            (str(row.get("bib_id", "")), str(row.get("service", "")))
            if isinstance(row, dict)
            else ("", "")
        ),
    )
    metadata_match = sorted(
        copy.deepcopy(enriched_payload.get("metadata_match") or ()),
        key=lambda row: str(row.get("service", "")) if isinstance(row, dict) else "",
    )

    return EnrichmentSidecar(
        schema_version=ENRICHMENT_SIDECAR_SCHEMA_VERSION,
        core_sha256=core_sha256,
        settings_digest=settings_digest,
        completeness=completeness,
        bib_match=tuple(bib_match),
        metadata_match=tuple(metadata_match),
        warnings=tuple(" ".join(str(item).split())[:512] for item in warnings),
        detail=" ".join(str(detail).split())[:512] if detail else None,
    )


def replay_enrichment_sidecar(
    core_payload: dict,
    sidecar: EnrichmentSidecar,
    *,
    expected_settings_digest: str,
) -> dict:
    """Merge a sidecar only when core, settings, and schema identities match."""

    if sidecar.schema_version != ENRICHMENT_SIDECAR_SCHEMA_VERSION:
        raise ArtifactReplayError("unsupported enrichment sidecar schema")
    if core_payload.get("schema_version") not in SUPPORTED_CORE_SCHEMA_VERSIONS:
        raise ArtifactReplayError("core payload schema does not match replay contract")
    if sidecar.core_sha256 != canonical_json_sha256(core_payload):
        raise ArtifactReplayError("enrichment sidecar core hash mismatch")
    if sidecar.settings_digest != expected_settings_digest:
        raise ArtifactReplayError("enrichment sidecar settings digest mismatch")
    if sidecar.completeness not in {"partial", "complete"}:
        raise ArtifactReplayError("invalid enrichment completeness")
    # v11 hangs the completeness receipt and the enrichment diagnostics off
    # ``extraction``. Without that block they would be dropped silently and a
    # partial enrichment would read as clean, so treat its absence as the
    # contract violation it is rather than replaying a lossy payload.
    if not isinstance(core_payload.get("extraction"), dict):
        raise ArtifactReplayError("core payload has no extraction block")

    core_bib_ids = {
        row.get("bib_id")
        for row in core_payload.get("bib") or []
        if isinstance(row, dict) and row.get("bib_id") is not None
    }
    seen_bib_services: set[tuple[object, object]] = set()
    for row in sidecar.bib_match:
        if not isinstance(row, dict) or not row.get("service"):
            raise ArtifactReplayError("invalid bibliography enrichment row")
        if row.get("bib_id") not in core_bib_ids:
            raise ArtifactReplayError("bibliography enrichment references unknown bib_id")
        key = (row.get("bib_id"), row.get("service"))
        if key in seen_bib_services:
            raise ArtifactReplayError("duplicate bibliography enrichment service row")
        seen_bib_services.add(key)
    seen_metadata_services: set[object] = set()
    for row in sidecar.metadata_match:
        if not isinstance(row, dict) or not row.get("service"):
            raise ArtifactReplayError("invalid paper enrichment row")
        if row["service"] in seen_metadata_services:
            raise ArtifactReplayError("duplicate paper enrichment service row")
        seen_metadata_services.add(row["service"])

    # Identity checks alone do not make external rows safe to publish. Apply
    # the same strict, extra-forbid models used by the public export so bad
    # scalar/container types and unknown fields fail before replay can clear
    # the pending gate or reach materialization.
    from pydantic import ValidationError

    from bibr.export.json_export import (
        BibMatchExport,
        MetadataMatchExport,
        append_payload_warning,
    )

    try:
        for row in sidecar.bib_match:
            BibMatchExport.model_validate(row, strict=True)
        for row in sidecar.metadata_match:
            MetadataMatchExport.model_validate(row, strict=True)
    except ValidationError as exc:
        raise ArtifactReplayError("invalid typed enrichment row") from exc

    replayed = copy.deepcopy(core_payload)
    replayed["bib_match"] = copy.deepcopy(list(sidecar.bib_match))
    replayed["metadata_match"] = copy.deepcopy(list(sidecar.metadata_match))
    bibliography = replayed.get("bib") or []
    # v11: enrichment completeness and warnings live under ``extraction``, whose
    # presence the contract checks above already guaranteed.
    extraction = replayed["extraction"]
    if bibliography:
        matched_ids = {
            row.get("bib_id")
            for row in sidecar.bib_match
            if isinstance(row, dict) and row.get("bib_id") is not None
        }
        extraction["enrichment"] = {
            "complete": sidecar.completeness == "complete",
            "refs_enriched": len(matched_ids),
            "refs_total": len(bibliography),
        }
    else:
        # Nothing to enrich — absent, not a zeroed row.
        extraction.pop("enrichment", None)
    if sidecar.completeness == "complete":
        _clear_enrichment_pending(replayed)
    diagnostics = [*sidecar.warnings]
    if sidecar.detail:
        diagnostics.append(f"enrichment: {sidecar.detail}")
    for diagnostic in diagnostics:
        append_payload_warning(replayed, diagnostic)
    return replayed
