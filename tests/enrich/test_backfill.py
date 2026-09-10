"""Saved exports can acquire enrichment without losing their extraction identity."""

import asyncio
import copy
from unittest.mock import AsyncMock

import pytest

from bibr.enrich.backfill import ExportEnricher, validate_backfill_paper
from bibr.enrich.references import EnrichmentReport
from bibr.models import BibAuthor, ExternalMatch, MatchSource
from bibr.pipeline.artifacts import (
    ArtifactReplayError,
    EnrichmentSidecar,
    canonical_json_sha256,
    mark_enrichment_pending,
    replay_enrichment_sidecar,
)


@pytest.fixture
def saved_paper():
    return {
        "paper_id": "paper-1",
        "info": {
            "title": "Printed paper title",
            "doi": "10.1234/self",
            "keywords": [],
            "file_hash": "abc",
            "file_name": "paper.pdf",
            "input_format": "pdf",
            "schema_version": "10.7",
            "bibr_version": "0.1.0",
        },
        "bib": [{"bib_id": 7, "title": "Printed reference", "year": 2024}],
        "bib_match": [],
        "info_match": [],
        "enrichment": None,
        "author": [],
        "text": [],
        "section": [],
        "url": [],
        "xref": [],
        "figure": [],
        "table": [],
        "eq": [],
        "processing_warnings": ["Original warning"],
    }


@pytest.fixture
def enricher(monkeypatch):
    from bibr.enrich import references

    worker = ExportEnricher()
    worker._client = AsyncMock()
    worker.settings.crossref.enrich = False  # Explicit backfill is independent of inline policy.
    worker.settings.crossref.consolidate = "replace"

    async def refs(values, **_kwargs):
        for ref in values:
            ref.match[MatchSource.CROSSREF] = ExternalMatch(
                id=f"10.1234/ref-{ref.bib_id}",
                title="External title",
                doi="10.1234/ref",
                authors=[BibAuthor(given="A", family="Author")],
            )
        return EnrichmentReport(attempted=len(values), matched=len(values))

    async def identity(meta, **_kwargs):
        meta.match[MatchSource.CROSSREF] = ExternalMatch(id=meta.doi, title="External self title")
        return EnrichmentReport(attempted=1, matched=1)

    monkeypatch.setattr(references, "enrich_references", AsyncMock(side_effect=refs))
    monkeypatch.setattr(references, "enrich_paper_identity", AsyncMock(side_effect=identity))
    return worker


async def test_backfill_preserves_core_and_produces_replayable_sidecar(saved_paper, enricher):
    original = copy.deepcopy(saved_paper)
    result = await enricher.enrich(saved_paper)
    assert saved_paper == original
    assert result["status"] == "complete"
    assert result["enrichment"]["core_sha256"] == canonical_json_sha256(original)
    sidecar = EnrichmentSidecar.from_dict(result["enrichment"])
    assert (
        replay_enrichment_sidecar(
            original, sidecar, expected_settings_digest=enricher.settings_digest
        )
        == result["paper"]
    )
    for field, value in original.items():
        if field not in {"bib_match", "info_match", "enrichment"}:
            assert result["paper"][field] == value
    assert result["paper"]["bib_match"][0]["authors"] == [{"given": "A", "family": "Author"}]
    assert result["paper"]["bib_match"][0]["bib_id"] == 7
    assert result["paper"]["enrichment"] == {"complete": True, "refs_total": 1, "refs_enriched": 1}
    changed = copy.deepcopy(original)
    changed["bib"][0]["title"] = "A different extraction"
    with pytest.raises(ArtifactReplayError, match="core hash"):
        replay_enrichment_sidecar(
            changed, sidecar, expected_settings_digest=enricher.settings_digest
        )


async def test_repeat_completed_export_needs_no_calls(saved_paper, enricher):
    from bibr.enrich.references import enrich_paper_identity, enrich_references

    first = await enricher.enrich(saved_paper)
    enrich_references.reset_mock()
    enrich_paper_identity.reset_mock()
    second = await enricher.enrich(first["paper"])
    assert second["status"] == "no_work"
    assert second["paper"] == first["paper"]
    enrich_references.assert_not_awaited()
    enrich_paper_identity.assert_not_awaited()


async def test_complete_miss_does_not_repeat_reference_lookup(saved_paper, enricher, monkeypatch):
    from bibr.enrich import references

    monkeypatch.setattr(
        references, "enrich_references", AsyncMock(return_value=EnrichmentReport(attempted=1))
    )
    first = await enricher.enrich(saved_paper)
    assert first["paper"]["bib_match"] == []
    assert first["paper"]["enrichment"]["complete"] is True
    references.enrich_references.reset_mock()
    second = await enricher.enrich(first["paper"])
    assert second["status"] == "no_work"
    references.enrich_references.assert_not_awaited()


async def test_partial_retry_retains_matches_and_only_requests_missing(
    saved_paper, enricher, monkeypatch
):
    from bibr.enrich import references

    saved_paper["bib"].append({"bib_id": 42, "title": "Another printed reference"})
    normal = references.enrich_references.side_effect

    async def fail_one(refs, **kwargs):
        await normal(refs[:1], **kwargs)
        return EnrichmentReport(attempted=2, matched=1, failed=1)

    monkeypatch.setattr(references, "enrich_references", AsyncMock(side_effect=fail_one))
    mark_enrichment_pending(saved_paper)
    first = await enricher.enrich(saved_paper)
    assert first["status"] == "partial"
    assert first["paper"]["enrichment"]["complete"] is False
    assert first["paper"]["validation"]["promotable"] is False
    original_match = first["paper"]["bib_match"][0]
    monkeypatch.setattr(references, "enrich_references", AsyncMock(side_effect=normal))
    second = await enricher.enrich(first["paper"])
    requested = references.enrich_references.await_args.args[0]
    assert [ref.bib_id for ref in requested] == [42]
    assert original_match in second["paper"]["bib_match"]
    assert second["paper"]["validation"]["promotable"] is True
    assert second["paper"]["bib"] == saved_paper["bib"]


async def test_timeout_keeps_completed_matches(saved_paper, enricher, monkeypatch):
    from bibr.enrich import references

    normal = references.enrich_references.side_effect

    async def stalls(refs, **kwargs):
        await normal(refs, **kwargs)
        await asyncio.Event().wait()

    monkeypatch.setattr(references, "enrich_references", stalls)
    enricher.settings.crossref.enrich_timeout = 0.02
    result = await enricher.enrich(saved_paper)
    assert result["status"] == "partial"
    assert len(result["paper"]["bib_match"]) == 1
    assert len(result["paper"]["info_match"]) == 1
    assert "timed out" in result["enrichment"]["warnings"][0]


async def test_cancellation_propagates(saved_paper, enricher, monkeypatch):
    from bibr.enrich import references

    monkeypatch.setattr(
        references, "enrich_references", AsyncMock(side_effect=asyncio.CancelledError)
    )
    with pytest.raises(asyncio.CancelledError):
        await enricher.enrich(saved_paper)


@pytest.mark.parametrize(
    "issue", ["duplicate_id", "dangling_match", "duplicate_service", "counts", "type", "schema"]
)
async def test_bad_artifact_rejected_before_http(saved_paper, enricher, issue):
    from bibr.enrich.references import enrich_paper_identity, enrich_references

    if issue == "duplicate_id":
        saved_paper["bib"] *= 2
    elif issue == "dangling_match":
        saved_paper["bib_match"] = [{"bib_id": 99, "service": "crossref"}]
    elif issue == "duplicate_service":
        saved_paper["info_match"] = [{"service": "crossref"}] * 2
    elif issue == "counts":
        saved_paper["enrichment"] = {"complete": True, "refs_total": 100, "refs_enriched": 0}
    elif issue == "type":
        saved_paper["bib"][0]["bib_id"] = "7"
    else:
        saved_paper["info"]["schema_version"] = "1.0"
    with pytest.raises(ValueError):
        await enricher.enrich(saved_paper)
    enrich_paper_identity.assert_not_awaited()
    enrich_references.assert_not_awaited()


async def test_empty_bibliography_is_not_reextracted(saved_paper, enricher):
    from bibr.enrich.references import enrich_references

    saved_paper["bib"] = []
    result = await enricher.enrich(saved_paper)
    assert result["paper"]["bib"] == []
    assert result["paper"]["bib_match"] == []
    enrich_references.assert_not_awaited()
    assert len(result["paper"]["info_match"]) == 1


async def test_completed_backfill_does_not_clear_other_validation_failures(saved_paper, enricher):
    issue = {
        "code": "VAL_REFERENCES_INCOMPLETE",
        "severity": "error",
        "message": "Extraction failed",
        "origin_stage": "extract",
        "evidence_ids": [],
        "count": 1,
        "blocking": True,
    }
    saved_paper["validation"] = {
        "errors": 1,
        "warnings": 0,
        "blocking": 1,
        "promotable": False,
        "issues": [issue],
    }
    mark_enrichment_pending(saved_paper)
    result = await enricher.enrich(saved_paper)
    assert result["status"] == "complete"
    assert result["paper"]["validation"]["issues"] == [issue]
    assert result["paper"]["validation"]["promotable"] is False


def test_snapshot_and_result_version_ignore_credentials(saved_paper):
    from bibr.config import snapshot_settings

    settings = snapshot_settings()
    one = ExportEnricher(settings)
    settings.crossref.api_key = "a-secret"
    two = ExportEnricher(settings)
    assert one.settings_digest == two.settings_digest
    settings.resolver.authoritative = not settings.resolver.authoritative
    assert ExportEnricher(settings).settings_digest != one.settings_digest
    assert one.settings.resolver.authoritative != settings.resolver.authoritative
    saved_paper["info"]["schema_version"] = "10.6"
    validate_backfill_paper(saved_paper)


async def test_close_is_idempotent(enricher):
    client = enricher._client
    await enricher.close()
    await enricher.close()
    client.close.assert_awaited_once()
