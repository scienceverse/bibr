"""What happened when bibr looked a reference up.

Enrichment records, per reference, which questions it asked and how they
were answered. :meth:`ReferenceLookup.outcome` reduces that to one word:

- ``doi_hit``: the printed DOI resolved to a record that was accepted.
- ``doi_404``: the registry has no record of the printed DOI. For these,
  doi.org is asked which agency registered the DOI, if any
  (:func:`registration_agencies`): a DataCite DOI is unknown to Crossref
  but exists.
- ``doi_rejected``: the printed DOI resolved, but to a work published more
  than a year from the printed year, so it names another work.
- ``search_hit``: a title, fingerprint or reference-string search found the
  work.
- ``clean_miss``: every search answered, and none found the work.
- ``error``: a lookup failed (a network error, a server error) and nothing
  else found the work.
- ``timed_out``: the per-paper enrichment budget ran out before the
  reference's lookup finished.
- ``not_attempted``: the reference gave nothing to look up (no DOI, no
  usable title, no volume to fingerprint).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

LookupOutcome = Literal[
    "doi_hit",
    "doi_404",
    "doi_rejected",
    "search_hit",
    "clean_miss",
    "error",
    "timed_out",
    "not_attempted",
]
LOOKUP_OUTCOMES: tuple[LookupOutcome, ...] = (
    "doi_hit",
    "doi_404",
    "doi_rejected",
    "search_hit",
    "clean_miss",
    "error",
    "timed_out",
    "not_attempted",
)

# doi.org's answer for a DOI no agency registered.
UNREGISTERED = "none"

_DOI_RA_URL = "https://doi.org/ra/"
# DOIs per doi.org request; the list goes in the URL path.
_RA_CHUNK = 40
_RA_TIMEOUT_SECONDS = 10.0


@dataclass
class ReferenceLookup:
    """The questions enrichment asked about one reference, and the answers."""

    queried: bool = False
    errored: bool = False
    # The printed DOI's lookup: "found", "not_found" (404) or "rejected" (the
    # record is of another year). A later answer replaces an earlier one, so a
    # resolver miss followed by a Crossref hit reads as found.
    doi_status: Literal["found", "not_found", "rejected"] | None = None
    # How the accepted match was found: "doi" or "search".
    hit: Literal["doi", "search"] | None = None
    finished: bool = False
    timed_out: bool = False
    # For a DOI the registry has no record of: the agency doi.org names for
    # it, lowercase ("datacite", "medra"), or UNREGISTERED.
    registration_agency: str | None = None

    def outcome(self, matched: bool) -> LookupOutcome:
        """The lookup in one word; *matched* is whether the reference has a match."""
        if matched:
            return "doi_hit" if self.hit == "doi" else "search_hit"
        if self.timed_out:
            return "timed_out"
        if self.doi_status == "rejected":
            return "doi_rejected"
        if self.doi_status == "not_found" and not self.errored:
            return "doi_404"
        if self.errored:
            return "error"
        if self.queried:
            return "clean_miss"
        return "not_attempted"


def lookup_of(ref) -> ReferenceLookup:
    """*ref*'s lookup record, created on first use."""
    lookup = getattr(ref, "_lookup", None)
    if lookup is None:
        lookup = ReferenceLookup()
        ref._lookup = lookup
    return lookup


def recorded_lookup(ref) -> ReferenceLookup | None:
    """*ref*'s lookup record, or ``None`` when enrichment never touched it."""
    return getattr(ref, "_lookup", None)


def finish_unfinished(references, *, timed_out: bool) -> None:
    """Close the lookups a stopped enrichment left open: as timed out when the
    budget ran out, else as failed."""
    for ref in references:
        lookup = lookup_of(ref)
        if lookup.finished or ref.match:
            continue
        lookup.finished = True
        if timed_out:
            lookup.timed_out = True
        else:
            lookup.errored = True


def _ra_name(entry: dict) -> str | None:
    agency = entry.get("RA")
    if isinstance(agency, str) and agency.strip():
        return agency.strip().lower()
    status = entry.get("status")
    if isinstance(status, str) and status.strip().lower() in {"doi does not exist", "invalid doi"}:
        return UNREGISTERED
    return None


async def registration_agencies(
    dois: list[str], *, user_agent: str | None = None, http=None
) -> dict[str, str]:
    """Which agency registered each DOI, by doi.org: ``{doi.casefold(): agency}``.

    Agencies are lowercase (``crossref``, ``datacite``, ``medra``);
    :data:`UNREGISTERED` means doi.org says the DOI does not exist. DOIs
    doi.org did not answer for are left out. Never raises: a failed request
    leaves its DOIs out.
    """
    from urllib.parse import quote

    import httpx

    from bibr.utils.text import is_url_safe_doi

    wanted: list[str] = []
    seen: set[str] = set()
    for doi in dois:
        folded = doi.casefold()
        # The list is comma-separated, so a DOI with a comma cannot be asked.
        if not doi or folded in seen or "," in doi or not is_url_safe_doi(doi):
            continue
        seen.add(folded)
        wanted.append(doi)
    if not wanted:
        return {}

    owns_http = http is None
    if http is None:
        headers = {"User-Agent": user_agent} if user_agent else {}
        http = httpx.AsyncClient(timeout=_RA_TIMEOUT_SECONDS, headers=headers)
    agencies: dict[str, str] = {}
    try:
        for start in range(0, len(wanted), _RA_CHUNK):
            chunk = wanted[start : start + _RA_CHUNK]
            url = _DOI_RA_URL + quote(",".join(chunk), safe="/,();:")
            try:
                response = await http.get(url)
                response.raise_for_status()
                entries = response.json()
            except Exception as exc:  # noqa: BLE001 — the agency is a nicety, not a result
                logger.debug("doi.org registration-agency lookup failed: %s", exc)
                continue
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict) or not isinstance(entry.get("DOI"), str):
                    continue
                agency = _ra_name(entry)
                if agency is not None:
                    agencies[entry["DOI"].casefold()] = agency
    finally:
        if owns_http:
            await http.aclose()
    return agencies
