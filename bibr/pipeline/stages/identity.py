"""Validate manifest-backed identity against source-visible DOI evidence."""

from __future__ import annotations

from bibr.extract.doi_identity import collect_doi_candidates, doi_sha256, select_doi_candidates
from bibr.validation import IssueSeverity, ValidationIssue


def _source_integrity_issue(queue_record_id: str) -> ValidationIssue:
    return ValidationIssue(
        code="VAL_SOURCE_INTEGRITY",
        severity=IssueSeverity.ERROR,
        message="Processed source SHA-256 does not match the manifest identity",
        origin_stage="identity",
        evidence_ids=(queue_record_id,),
        blocking=True,
    )


class IdentityValidationStage:
    name = "identity"
    requires = ("paper",)
    produces = ("doi_selection",)

    async def run(self, ctx) -> None:
        ctx.progress.stage_start(self.name)
        for fs in ctx.alive():
            paper = fs.paper
            if paper is None or paper.contents is None:
                continue
            expected = fs.expected_identity
            selection = select_doi_candidates(collect_doi_candidates(paper.contents), expected)
            fs.doi_selection = selection
            paper.expected_identity = expected
            paper.doi_selection = selection

            if paper.metadata is not None:
                if selection.selected is not None:
                    paper.metadata.doi = selection.selected.normalized
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
        ctx.progress.stage_end(self.name)
