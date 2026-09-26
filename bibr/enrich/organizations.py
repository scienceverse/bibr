"""Match a paper's affiliation strings and funder names to ROR organizations.

The printed strings stay as they are; matches go to
``metadata.affiliation_match`` / ``metadata.funder_match`` keyed by the exact
string, and the export turns them into ``affiliation_match[]`` and
``funding_match[]`` rows.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bibr.extract.research_integrity import collect_affiliations

if TYPE_CHECKING:
    from bibr.clients.ror import RorClient
    from bibr.models import PaperMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrganizationReport:
    attempted: int = 0
    matched: int = 0
    timed_out: bool = False
    # Lookups ROR did not answer usefully, and why (``RorClient.lookup``).
    failed: int = 0
    failure_reasons: tuple[str, ...] = ()


def affiliation_texts(metadata: PaperMetadata) -> list[str]:
    """The distinct affiliation strings, in ``affiliation[]`` table order."""
    texts, _ = collect_affiliations(metadata.authors)
    return list(dict.fromkeys([*texts, *(a.text for a in metadata.affiliations)]))


async def enrich_organizations(
    metadata: PaperMetadata, client: RorClient, *, timeout: float
) -> OrganizationReport:
    """Fill ``metadata.affiliation_match`` and ``metadata.funder_match``.

    Matches found before *timeout* are kept; the rest stay unmatched.
    """
    affiliations = affiliation_texts(metadata)
    funders = list(dict.fromkeys(f.funder for f in metadata.funding if f.funder))
    if not affiliations and not funders:
        return OrganizationReport()

    failures: list[str] = []

    async def run() -> None:
        # Stored as they arrive, so a timeout keeps what was already matched.
        async def one(text: str, into: dict) -> None:
            result, failure = await client.lookup(text)
            if failure is not None:
                failures.append(failure)
            if result is not None:
                into[text] = result

        semaphore = asyncio.Semaphore(4)

        async def bounded(text: str, into: dict) -> None:
            async with semaphore:
                await one(text, into)

        await asyncio.gather(
            *(bounded(text, metadata.affiliation_match) for text in affiliations),
            *(bounded(text, metadata.funder_match) for text in funders),
        )

    timed_out = False
    try:
        await asyncio.wait_for(run(), timeout=timeout)
    except TimeoutError:
        timed_out = True
        logger.info("ROR matching stopped after %.0fs; remaining strings left unmatched", timeout)
    return OrganizationReport(
        attempted=len(affiliations) + len(funders),
        matched=len(metadata.affiliation_match) + len(metadata.funder_match),
        timed_out=timed_out,
        failed=len(failures),
        failure_reasons=tuple(sorted(set(failures))),
    )
