"""Validate manifest-backed identity against source-visible DOI evidence."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

from bibr.extract.doi_identity import collect_doi_candidates, doi_sha256, select_doi_candidates
from bibr.extract.pdf_doi_evidence import is_pdf, read_pdf_doi_evidence
from bibr.field_states import set_field_source
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.extract.pdf_doi_evidence import PdfDoiEvidence

logger = logging.getLogger(__name__)

# The front pages whose text layer and links join the DOI candidate pool: the
# pages the front-matter tier covers.
_EVIDENCE_PAGES = (1, 2)


def _source_integrity_issue(queue_record_id: str) -> ValidationIssue:
    return ValidationIssue(
        code="VAL_SOURCE_INTEGRITY",
        severity=IssueSeverity.ERROR,
        message="Processed source SHA-256 does not match the manifest identity",
        origin_stage="identity",
        evidence_ids=(queue_record_id,),
        blocking=True,
    )


def _processed_bytes(fs) -> bytes | None:
    """The bytes this run processed, if they are still at hand.

    ``pdf_bytes`` are freed after OCR. A caller-supplied upload keeps them on
    ``caller_bytes``; an input file is reread only while it still hashes to
    the processed bytes.
    """
    if fs.pdf_bytes is not None:
        return fs.pdf_bytes
    if fs.caller_bytes is not None:
        return fs.caller_bytes
    if not fs.content_sha256:
        return None
    try:
        data = fs.path.read_bytes() if fs.path.is_file() else None
    except OSError:
        return None
    if data is None or hashlib.sha256(data).hexdigest() != fs.content_sha256.casefold():
        return None
    return data


def _pdf_doi_evidence(fs) -> PdfDoiEvidence | None:
    """The input PDF's own DOI evidence; None for other inputs or an unreadable PDF."""
    contents = getattr(fs.paper, "contents", None)
    if contents is None or getattr(contents, "preparsed_metadata", None) is not None:
        return None
    # The front pages this run processed: a page range may start later.
    processed = {region.page for region in getattr(contents, "region_summaries", None) or ()}
    processed.update(
        sentence.page_number for sentence in getattr(contents, "sentences", None) or ()
    )
    pages = [page for page in _EVIDENCE_PAGES if page in processed]
    data = _processed_bytes(fs)
    if not pages or not is_pdf(data):
        return None
    try:
        return read_pdf_doi_evidence(data, pages)
    except Exception:  # noqa: BLE001 - the parsed-text pool stands on its own
        logger.warning("Could not read DOI evidence from %s", fs.path.name, exc_info=True)
        return None


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
                # Reading the PDF blocks on pdfium; keep it off the shared loop.
                evidence = await asyncio.to_thread(_pdf_doi_evidence, fs)
                self._validate(fs, evidence)
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
    def _validate(fs, pdf_evidence: PdfDoiEvidence | None = None) -> None:
        """Select the paper's DOI from source evidence and check the manifest identity.

        This is the only step that writes ``metadata.doi``.
        """
        paper = fs.paper
        if paper is None or paper.contents is None:
            return
        expected = fs.expected_identity
        selection = select_doi_candidates(
            collect_doi_candidates(paper.contents, pdf_evidence), expected
        )
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
