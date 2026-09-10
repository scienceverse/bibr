"""Enrich a saved export without reconstructing or rerunning its extraction."""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import TYPE_CHECKING

from bibr.pipeline.artifacts import (
    SUPPORTED_CORE_SCHEMA_VERSIONS,
    canonical_json_sha256,
    make_enrichment_sidecar,
    replay_enrichment_sidecar,
)

if TYPE_CHECKING:
    from bibr.clients.crossref import CrossrefClient
    from bibr.config import GlobalSettings
    from bibr.enrich.references import EnrichmentReport

logger = logging.getLogger(__name__)
BACKFILL_VERSION = "crossref-backfill-v1"


def validate_backfill_paper(paper: dict) -> None:
    """Reject invalid exports and ambiguous IDs before any external work."""
    from bibr.export.json_export import PaperExport

    parsed = PaperExport.model_validate(paper, strict=True)
    if parsed.info.schema_version not in SUPPORTED_CORE_SCHEMA_VERSIONS:
        raise ValueError("Backfill accepts bibr export schema 10.6 or 10.7")
    ids = [ref.bib_id for ref in parsed.bib]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate bibliography IDs")
    known = set(ids)
    seen = set()
    for match in parsed.bib_match:
        if match.bib_id not in known:
            raise ValueError("Enrichment references an unknown bibliography ID")
        key = (match.bib_id, match.service)
        if not match.service or key in seen:
            raise ValueError("Empty or duplicate bibliography match service")
        seen.add(key)
    services = [match.service for match in parsed.info_match]
    if not all(services) or len(set(services)) != len(services):
        raise ValueError("Empty or duplicate paper match service")
    if parsed.enrichment is not None:
        matched = len({row.bib_id for row in parsed.bib_match})
        if parsed.enrichment.refs_total != len(ids) or parsed.enrichment.refs_enriched != matched:
            raise ValueError("Enrichment counts disagree with bibliography and matches")


class ExportEnricher:
    """Own one lazy HTTP client and its shared rate limit/cache for backfill calls.

    An explicit backfill runs even when CROSSREF_ENRICH disabled inline enrichment.
    Consolidation is deliberately absent: printed fields must survive unchanged.
    """

    def __init__(self, settings: GlobalSettings | None = None) -> None:
        from bibr.config import snapshot_settings

        self.settings = snapshot_settings(settings)
        self._client: CrossrefClient | None = None
        self.settings_digest = canonical_json_sha256(
            {
                "version": BACKFILL_VERSION,
                "crossref": self.settings.crossref.model_dump(
                    exclude={"api_key", "api_email", "cache_redis_url", "enrich", "consolidate"}
                ),
                "resolver": self.settings.resolver.model_dump(),
            }
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def enrich(self, paper: dict) -> dict:
        from bibr.enrich.references import enrich_paper_identity, enrich_references
        from bibr.export.json_export import BibMatchExport, InfoMatchExport
        from bibr.models import PaperMetadata, PaperReference

        validate_backfill_paper(paper)
        # The sidecar names the exact submitted artifact, including prior matches.
        core_hash = canonical_json_sha256(paper)
        enriched = copy.deepcopy(paper)
        existing = {row["bib_id"] for row in paper.get("bib_match", [])}
        refs_complete = bool((paper.get("enrichment") or {}).get("complete"))
        refs = []
        if not refs_complete:
            for row in paper["bib"]:
                if row["bib_id"] in existing:
                    continue
                fields = {k: v for k, v in row.items() if k in PaperReference.model_fields}
                fields["title"] = fields.get("title") or ""
                for field in ("first_page", "volume", "authors", "year", "container"):
                    fields.setdefault(field, None)
                refs.append(PaperReference(**fields))
        info = paper["info"]
        meta = PaperMetadata(
            doi=(info.get("doi") or "") if not paper.get("info_match") else "",
            title=info.get("title") or "",
            published=info.get("published"),
        )
        reports: list[EnrichmentReport] = []
        warnings: list[str] = []
        interrupted = False
        attempted = len(refs) + bool(meta.doi)

        async def run():
            if self._client is None:
                from bibr.clients.crossref import CrossrefClient

                self._client = CrossrefClient(settings=self.settings)
            if meta.doi:
                reports.append(
                    await enrich_paper_identity(
                        meta, crossref_client=self._client, settings=self.settings
                    )
                )
            if refs:
                reports.append(
                    await enrich_references(
                        refs, crossref_client=self._client, settings=self.settings
                    )
                )

        if attempted:
            try:
                await asyncio.wait_for(run(), timeout=self.settings.crossref.enrich_timeout)
            except TimeoutError:
                interrupted = True
                warnings.append("Backfill enrichment timed out; successful matches were retained")
            except Exception:  # noqa: BLE001 - preserve partial results on upstream failures
                interrupted = True
                logger.warning("Backfill enrichment failed", exc_info=True)
                warnings.append("Backfill enrichment failed; successful matches were retained")
        failed = sum(r.failed for r in reports)
        partial = interrupted or failed > 0
        # Upstream diagnostics can contain URLs or credentials; expose a bounded,
        # service-independent summary, leaving detailed diagnostics in server logs.
        if failed:
            warnings.append(f"Backfill had {failed} terminal enrichment failures")
        for report in reports:
            for detail in report.details:
                logger.warning("Backfill upstream diagnostic: %s", detail)

        enriched.setdefault("bib_match", [])
        enriched.setdefault("info_match", [])
        for ref in refs:
            for service, match in ref.match.items():
                data = match.model_dump()
                data["service_id"] = data.pop("id")
                enriched["bib_match"].append(
                    BibMatchExport(bib_id=ref.bib_id, service=service.value, **data).model_dump(
                        mode="json"
                    )
                )
        for service, match in meta.match.items():
            data = match.model_dump()
            data["service_id"] = data.pop("id")
            enriched["info_match"].append(
                InfoMatchExport(service=service.value, **data).model_dump(mode="json")
            )
        sidecar = make_enrichment_sidecar(
            enriched,
            core_sha256=core_hash,
            settings_digest=self.settings_digest,
            completeness="partial" if partial else "complete",
            warnings=tuple(warnings),
        )
        result = replay_enrichment_sidecar(
            paper, sidecar, expected_settings_digest=self.settings_digest
        )
        return {
            "paper": result,
            "enrichment": sidecar.to_dict(),
            "enrichment_version": BACKFILL_VERSION,
            "enrichment_key": canonical_json_sha256(
                {
                    "core_sha256": core_hash,
                    "settings_digest": self.settings_digest,
                }
            ),
            "status": "partial" if partial else "complete" if attempted else "no_work",
        }
