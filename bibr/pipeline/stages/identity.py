"""Validate manifest-backed identity against source-visible DOI evidence."""

from __future__ import annotations

import logging

from bibr.extract.doi_identity import collect_doi_candidates, doi_sha256, select_doi_candidates
from bibr.field_states import set_field_source
from bibr.validation import IssueSeverity, ValidationIssue

logger = logging.getLogger(__name__)


def _source_integrity_issue(queue_record_id: str) -> ValidationIssue:
    return ValidationIssue(
        code="VAL_SOURCE_INTEGRITY",
        severity=IssueSeverity.ERROR,
        message="Processed source SHA-256 does not match the manifest identity",
        origin_stage="identity",
        evidence_ids=(queue_record_id,),
        blocking=True,
    )


def doi_field_source(source_kind: str) -> str:
    """``extraction.fields.doi.source`` for a selected candidate's source kind."""
    return "native" if source_kind == "structured_metadata" else source_kind


class IdentityValidationStage:
    name = "identity"
    requires = ("paper",)
    produces = ("doi_selection",)

    async def run(self, ctx) -> None:
        ctx.progress.stage_start(self.name)
        for fs in ctx.alive():
            try:
                self._validate(fs)
            except Exception as exc:  # noqa: BLE001 - per-file failure, never the chunk
                fs.set_error(
                    f"Identity validation failed: {exc}",
                    code="identity_failed",
                    stage=self.name,
                    exc=exc,
                )
                logger.warning("Identity validation failed for %s", fs.path.name, exc_info=True)
        ctx.progress.stage_end(self.name)

    @staticmethod
    def _validate(fs) -> None:
        """Select the paper's DOI from source evidence and check the manifest identity.

        This is the only step that writes ``metadata.doi``.
        """
        paper = fs.paper
        if paper is None or paper.contents is None:
            return
        expected = fs.expected_identity
        selection = select_doi_candidates(collect_doi_candidates(paper.contents), expected)
        fs.doi_selection = selection
        paper.expected_identity = expected
        paper.doi_selection = selection

        if paper.metadata is not None:
            if selection.selected is not None:
                paper.metadata.doi = selection.selected.normalized
                set_field_source(
                    paper.metadata, "doi", doi_field_source(selection.selected.source_kind)
                )
            else:
                # A scalar DOI without selected source evidence contradicts the receipt.
                paper.metadata.doi = ""

        issues = list(selection.issues)
        if expected is not None:
            source_matches = expected.source_sha256 is None or (
                fs.content_sha256 is not None
                and fs.content_sha256.casefold() == expected.source_sha256.casefold()
            )
            if not source_matches:
                issues.append(_source_integrity_issue(expected.queue_record_id))

            if (
                expected.expected_doi
                and expected.expected_doi_sha256
                and doi_sha256(expected.expected_doi) != expected.expected_doi_sha256.casefold()
            ):
                issues.append(
                    ValidationIssue(
                        code="VAL_EXPECTED_ID_MISMATCH",
                        severity=IssueSeverity.ERROR,
                        message="Expected DOI and expected DOI hash disagree",
                        origin_stage="identity",
                        evidence_ids=(expected.queue_record_id,),
                        blocking=True,
                    )
                )

        existing = {
            (
                issue.code,
                str(issue.severity),
                issue.origin_stage,
                issue.evidence_ids,
                issue.message,
            )
            for issue in paper.validation_issues
        }
        paper.validation_issues.extend(
            issue
            for issue in issues
            if (
                issue.code,
                str(issue.severity),
                issue.origin_stage,
                issue.evidence_ids,
                issue.message,
            )
            not in existing
        )
