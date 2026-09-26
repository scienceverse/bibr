"""Crossref-based reference enrichment.

Enriches extracted PaperReference objects with Crossref metadata:
- DOI lookup for references that already have a DOI
- Bibliographic search for references without a DOI
- Writes matches directly into each reference's ``match`` dict
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import httpx
from rapidfuzz import fuzz

from bibr.config import GlobalSettings, snapshot_settings
from bibr.enrich.schemas import CrossrefAuthor, CrossrefWorkItem, plain_text
from bibr.models import BibAuthor, MatchFunder, MatchOrganization, canonicalize_orcid
from bibr.paper import ExternalMatch, MatchSource, PaperReference, migrate_bib_type
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.utils.text import normalize_doi

logger = logging.getLogger(__name__)

# Minimum fuzzy title similarity to accept a Crossref search match
_TITLE_MATCH_THRESHOLD = 80

# Minimum fuzzy container (journal/venue) similarity to accept a title-less
# fingerprint match (Strategy 3). Higher than the title gate: the container is one
# of only a few fingerprint fields, so it must align closely.
_CONTAINER_MATCH_THRESHOLD = 85

# Candidates requested for the title-less fingerprint query. See the call site.
_FINGERPRINT_ROWS = 20

# Map a resolver candidate's `source` field to the MatchSource it is stored
# under. The resolver is multi-source (OpenAlex, and Crossref when a Crossref
# index is configured); preserve that provenance rather than tagging everything
# OpenAlex. Unknown/absent source falls back to OpenAlex (no regression).
_RESOLVER_SOURCE_TO_MATCH = {
    "openalex": MatchSource.OPENALEX,
    "crossref": MatchSource.CROSSREF,
    "datacite": MatchSource.DATACITE,
}


def _resolver_match_source(cand: dict) -> MatchSource:
    """Resolve a candidate's ``source`` field to the MatchSource key for ``ref.match``."""
    return _RESOLVER_SOURCE_TO_MATCH.get(
        str(cand.get("source") or "").lower(), MatchSource.OPENALEX
    )


def _printed_fields(ref: PaperReference) -> dict[str, str | None]:
    """What the reference printed that a title match must not contradict."""
    return {
        "doi": ref.doi,
        "container": ref.container,
        "volume": ref.volume,
        "first_page": ref.first_page,
    }


@dataclass
class ResolutionStats:
    """Per-batch tally of how references were resolved. Counts are not mutually
    exclusive: a ref whose DOI lookup errors (non-404) also attempts a search."""

    doi_attempts: int = 0
    doi_matches: int = 0
    search_attempts: int = 0
    search_matches: int = 0
    resolver_attempts: int = 0
    resolver_matches: int = 0
    fallback_attempts: int = 0
    fallback_matches: int = 0
    fallback_misses: int = 0
    fallback_errors: int = 0
    fallback_timeouts: int = 0
    fingerprint_attempts: int = 0
    fingerprint_matches: int = 0
    failed_bib_ids: set[int] = field(default_factory=set)
    failure_details: list[ProcessingWarning] = field(default_factory=list)


@dataclass(frozen=True)
class EnrichmentReport:
    """Explicit terminal accounting for a best-effort enrichment operation."""

    attempted: int = 0
    matched: int = 0
    failed: int = 0
    details: tuple[ProcessingWarning, ...] = ()


def _record_terminal_failure(
    stats: ResolutionStats | None,
    ref: PaperReference,
    operation: str,
    exc: BaseException,
) -> None:
    if stats is None:
        return
    stats.failed_bib_ids.add(ref.bib_id)
    diagnostic = " ".join(str(exc).split())
    stats.failure_details.append(
        ProcessingWarning(
            WarningCode.ENRICHMENT_LOOKUP_FAILED,
            f"bib_id={ref.bib_id} {operation} failed" + (f": {diagnostic}" if diagnostic else ""),
        )
    )


def _record_fallback_warning(
    stats: ResolutionStats,
    code: WarningCode,
    detail: str,
) -> None:
    stats.failure_details.append(ProcessingWarning(code, detail))


@dataclass
class EnrichmentPrefetch:
    """The up-front network work for one reference batch, reusable by
    :func:`enrich_references`.

    Built by :func:`prefetch_enrichment`: the Crossref client whose response
    cache the bulk DOI lookup seeded, the resolver client (constructed here
    when settings enable it and none was supplied — then *owned*, and closed
    by :meth:`aclose`), its one-time health verdict, and the batch of
    title-search candidates keyed by ``id(ref)`` exactly as
    ``_prefetch_resolver_searches`` returns them. Timing is recorded so the
    pipeline can report how much of it overlapped extraction. The prefetch
    only *reads* the references — ``enrich_references`` remains the sole
    mutator of ``PaperReference`` objects.
    """

    crossref_client: Any
    resolver_client: Any | None = None
    owns_resolver: bool = False
    resolver_healthy: bool | None = None
    """``None`` when no resolver was configured, else the health-probe verdict."""
    resolver_prefetch: dict[int, list[dict] | Exception] = field(default_factory=dict)
    resolver_prefetch_error: Exception | None = None
    """A failed prefetch search, replayed as a terminal failure per eligible ref."""
    started_at: float = 0.0
    finished_at: float = 0.0
    _closed: bool = field(default=False, repr=False)

    @property
    def seconds(self) -> float:
        """Wall-clock seconds the prefetch took."""
        return max(0.0, self.finished_at - self.started_at)

    async def aclose(self) -> None:
        """Close the resolver client if this prefetch constructed it. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self.owns_resolver and self.resolver_client is not None:
            await self.resolver_client.close()


async def prefetch_enrichment(
    references: list[PaperReference],
    *,
    settings: GlobalSettings | None = None,
    crossref_client=None,
    resolver_client=None,
) -> EnrichmentPrefetch:
    """Run the network work that precedes the per-reference fan-out.

    1. Resolver health probe (one call; an unhealthy service drops the whole
       resolver tier to bare Crossref, with a warning — it was configured).
    2. Every title-searchable reference's resolver search, concurrently.
    3. Crossref's bulk DOI filter query for DOI-bearing references (skipped
       for an authoritative resolver, whose clean miss never reaches Crossref).

    Only *reads* the references, so it can start as soon as they are parsed
    and overlap the rest of extraction; ``enrich_references`` then consumes the
    result instead of repeating the round-trips. An owned resolver client is
    closed on every failure path, including cancellation.
    """
    effective = settings if settings is not None else snapshot_settings()
    started = time.monotonic()
    if crossref_client is None:
        from bibr.clients.crossref import get_client

        crossref_client = get_client(settings=effective)
    prefetch = EnrichmentPrefetch(crossref_client=crossref_client, started_at=started)
    if not references:
        prefetch.finished_at = time.monotonic()
        return prefetch

    try:
        owns_resolver = False
        if resolver_client is None and effective.resolver.url and effective.resolver.enrich:
            from bibr.clients.resolver import ResolverClient

            resolver_client = ResolverClient(
                effective.resolver.url,
                timeout=effective.resolver.timeout,
            )
            owns_resolver = True
        prefetch.resolver_client = resolver_client
        prefetch.owns_resolver = owns_resolver

        # One-time health probe: a down service or degraded index means every
        # reference would pay a failed round-trip before falling through. Skip
        # the whole resolver tier instead.
        if resolver_client is not None:
            healthy = await resolver_client.healthy()
            prefetch.resolver_healthy = healthy
            if not healthy:
                # A configured-but-unreachable resolver silently drops the whole
                # batch to bare Crossref. That is a real degradation (you asked
                # for the resolver and didn't get it), so warn rather than info —
                # the fallback is otherwise invisible. Only fires when the
                # resolver WAS configured, never when it is simply unset.
                logger.warning(
                    "Resolver unhealthy/unreachable — skipping resolver tier, "
                    "falling back to Crossref for %d refs",
                    len(references),
                )
                if owns_resolver:
                    await resolver_client.close()
                prefetch.resolver_client = None
                prefetch.owns_resolver = False
                resolver_client = None

        # Resolve every no-DOI title search up front, concurrently, rather than
        # inline inside the per-ref fan-out — off the CrossRef-sized semaphore.
        # DOI-bearing refs keep the per-ref lookup_doi fast path. A prefetch
        # failure degrades to empty so every ref simply falls through to
        # CrossRef; the resolver never fails enrichment.
        resolver_authoritative = effective.resolver.authoritative
        if resolver_client is not None:
            try:
                prefetch.resolver_prefetch = await _prefetch_resolver_searches(
                    references,
                    resolver_client,
                    settings=effective,
                    raise_on_error=resolver_authoritative,
                )
            except Exception as e:  # noqa: BLE001 — must fall through, not crash
                logger.warning("Resolver prefetch search failed, falling back to CrossRef: %s", e)
                prefetch.resolver_prefetch_error = e

        # Collapse every DOI-bearing reference into one filter query before the
        # fan-out. The per-ref lookup is unchanged — it just finds these already
        # cached, spending one rate-limited slot per chunk instead of one per
        # reference. An authoritative resolver answers from the same corpus and
        # its clean miss deliberately skips CrossRef, so a CrossRef prefetch
        # there would buy nothing — unless its search prefetch failed, which
        # sends the batch to CrossRef after all (see enrich_references).
        skip_bulk = (
            resolver_client is not None
            and resolver_authoritative
            and prefetch.resolver_prefetch_error is None
        )
        if effective.crossref.bulk_doi_lookup and not skip_bulk:
            try:
                await crossref_client.prefetch_works_by_doi([r.doi for r in references if r.doi])
            except Exception as e:  # noqa: BLE001 — must fall through to per-ref lookups
                logger.debug("Crossref bulk DOI prefetch skipped: %s", e)
    except BaseException:
        await prefetch.aclose()
        raise

    prefetch.finished_at = time.monotonic()
    return prefetch


async def enrich_references(
    references: list[PaperReference],
    crossref_client=None,
    resolver_client=None,
    *,
    settings: GlobalSettings | None = None,
    prefetch: EnrichmentPrefetch | None = None,
) -> EnrichmentReport:
    """Enrich references with Crossref metadata.

    For each reference:
    1. If it has a DOI, look it up directly via works(ids=doi)
    2. If no DOI, search bibliographically (title + author + year)
    3. Validate search matches by fuzzy title comparison
    4. Write an ExternalMatch into ``ref.match["crossref"]``

    Failures are handled per-reference (graceful degradation).
    Requests run concurrently (bounded by semaphore) for throughput.
    Mutates references in place.

    When a resolver_client is provided (or BIBR_RESOLVER_URL + BIBR_RESOLVER_ENRICH
    are set), the resolver is tried first (after a one-time health probe; an
    unhealthy service skips the whole tier). A hit is stored under the MatchSource
    matching the candidate's source (OpenAlex or Crossref) and CrossRef is skipped
    for that reference.

    Args:
        references: List of PaperReference to match against Crossref.
        crossref_client: Optional CrossrefClient instance.
            Created on demand if not provided.
        resolver_client: Optional ResolverClient instance. When provided (or
            auto-constructed from settings), the resolver is consulted first.
        prefetch: An :class:`EnrichmentPrefetch` built earlier for *these same
            reference objects* (the pipeline starts it while extraction is
            still running). Its clients replace ``crossref_client`` /
            ``resolver_client`` and its results replace the up-front round-trips;
            ``None`` performs that work inline, exactly as before.
    """
    if not references:
        return EnrichmentReport()
    effective = settings if settings is not None else snapshot_settings()
    stats = ResolutionStats()

    if prefetch is None:
        prefetch = await prefetch_enrichment(
            references,
            settings=effective,
            crossref_client=crossref_client,
            resolver_client=resolver_client,
        )
    crossref_client = prefetch.crossref_client
    resolver_client = prefetch.resolver_client
    resolver_authoritative = effective.resolver.authoritative
    resolver_prefetch = prefetch.resolver_prefetch
    if prefetch.resolver_prefetch_error is not None:
        # The resolver was never asked about these references, so its empty
        # answer is not a clean miss: an authoritative resolver would otherwise
        # skip CrossRef for every one of them.
        resolver_authoritative = False
        for ref in references:
            if _resolver_search_eligible(ref):
                _record_terminal_failure(
                    stats, ref, "resolver prefetch", prefetch.resolver_prefetch_error
                )

    semaphore = crossref_client.enrich_semaphore

    async def _enrich_one(ref: PaperReference) -> None:
        async with semaphore:
            outcome = await _resolve_reference(
                ref,
                crossref_client,
                resolver_client,
                stats,
                resolver_prefetch,
                resolver_authoritative=resolver_authoritative,
                settings=effective,
            )
            if outcome is not None:
                source, match = outcome
                ref.match[source] = match

    try:
        results = await asyncio.gather(
            *[_enrich_one(ref) for ref in references], return_exceptions=True
        )
        for ref, result in zip(references, results, strict=True):
            if isinstance(result, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise result
            if isinstance(result, BaseException):
                logger.warning("Reference enrichment raised for bib_id=%s: %s", ref.bib_id, result)
                _record_terminal_failure(stats, ref, "reference enrichment", result)

        if resolver_client is not None:
            try:
                await _enrich_resolver_fallback(
                    references,
                    resolver_client,
                    stats,
                    settings=effective,
                )
            except Exception as e:  # noqa: BLE001 — optional fallback cannot fail extraction
                eligible = [ref for ref in references if _resolver_fallback_eligible(ref)]
                stats.fallback_errors += len(eligible)
                logger.warning("Resolver fallback failed for %d refs: %s", len(eligible), e)
                _record_fallback_warning(
                    stats,
                    WarningCode.RESOLVER_FALLBACK_FAILED,
                    f"resolver fallback failed for {len(eligible)} refs: {' '.join(str(e).split())}",
                )
    finally:
        await prefetch.aclose()

    matched = sum(1 for ref in references if ref.match)
    logger.info(
        "Reference enrichment: %d/%d matched "
        "(resolver %d/%d, fallback matches=%d misses=%d errors=%d timeouts=%d attempts=%d, "
        "crossref doi %d/%d, search %d/%d, fingerprint %d/%d)",
        matched,
        len(references),
        stats.resolver_matches,
        stats.resolver_attempts,
        stats.fallback_matches,
        stats.fallback_misses,
        stats.fallback_errors,
        stats.fallback_timeouts,
        stats.fallback_attempts,
        stats.doi_matches,
        stats.doi_attempts,
        stats.search_matches,
        stats.search_attempts,
        stats.fingerprint_matches,
        stats.fingerprint_attempts,
    )
    # A reference is only "failed" if it ended UNMATCHED. A transient error in
    # an early strategy still lands in ``failed_bib_ids``, so counting that set
    # directly reported failures for references a later strategy went on to
    # match — exporting ``enrichment.complete=false`` and an ENRICHMENT_PARTIAL
    # checkpoint for papers where every reference actually matched. The
    # diagnostics still carry every attempt's failure detail.
    unresolved = {ref.bib_id for ref in references if not ref.match}
    return EnrichmentReport(
        attempted=len(references),
        matched=matched,
        failed=len(stats.failed_bib_ids & unresolved),
        details=tuple(dict.fromkeys(stats.failure_details)),
    )


def _year_from_published(published: str | None) -> int | None:
    """Parse a leading 4-digit year from a printed publication date string."""
    if not published:
        return None
    m = re.match(r"\s*(\d{4})", published)
    return int(m.group(1)) if m else None


async def enrich_paper_identity(
    metadata,
    crossref_client=None,
    resolver_client=None,
    *,
    settings: GlobalSettings | None = None,
) -> EnrichmentReport:
    """Look up the paper's OWN DOI once and store the hit in ``metadata.match``.

    Reuses the resolver-first / CrossRef-fallback routing that reference
    enrichment uses, but only the DOI-lookup path — the paper's DOI is always
    known when this runs. The printed self-identity fields are never touched;
    enrichment lives solely in ``metadata.match``. Any failure degrades
    silently to an empty match.
    """
    if not metadata.doi:
        return EnrichmentReport()
    effective = settings if settings is not None else snapshot_settings()

    if crossref_client is None:
        from bibr.clients.crossref import get_client

        crossref_client = get_client(settings=effective)

    owns_resolver = False
    if resolver_client is None and effective.resolver.url and effective.resolver.enrich:
        from bibr.clients.resolver import ResolverClient

        resolver_client = ResolverClient(
            effective.resolver.url,
            timeout=effective.resolver.timeout,
        )
        owns_resolver = True
    if resolver_client is not None and not await resolver_client.healthy():
        if owns_resolver:
            await resolver_client.close()
        resolver_client = None
        owns_resolver = False

    # An empty title confines _resolve_reference to its DOI-lookup branch (the
    # title-search / fingerprint tiers are gated on a >=10-char title), so this
    # is exactly one DOI lookup — no title search on the paper's own identity.
    probe = PaperReference(
        bib_id=0,
        title="",
        doi=metadata.doi,
        year=_year_from_published(metadata.published),
        first_page=None,
        volume=None,
        authors=None,
        container=None,
    )
    stats = ResolutionStats()
    try:
        outcome = await _resolve_reference(
            probe,
            crossref_client,
            resolver_client,
            stats,
            resolver_authoritative=effective.resolver.authoritative,
            settings=effective,
        )
    except Exception as e:  # noqa: BLE001 — self-DOI enrichment must never fail the pipeline
        logger.debug("Self-DOI enrichment failed for %s: %s", metadata.doi, e)
        _record_terminal_failure(stats, probe, "paper DOI enrichment", e)
        outcome = None
    finally:
        if owns_resolver and resolver_client is not None:
            await resolver_client.close()

    if outcome is not None:
        source, match = outcome
        metadata.match[source] = match
    return EnrichmentReport(
        attempted=1,
        matched=1 if outcome is not None else 0,
        # Same rule as the reference path: a strategy that errored before a
        # later one succeeded is not a failure of the enrichment.
        failed=0 if outcome is not None else len(stats.failed_bib_ids),
        details=tuple(dict.fromkeys(stats.failure_details)),
    )


async def _fetch_crossref_item(
    ref: PaperReference, client, stats: ResolutionStats | None = None
) -> tuple[CrossrefWorkItem, float] | None:
    """Fetch a Crossref work item for a reference.

    Strategy 1: Direct DOI lookup (most reliable) — score = 100.0
    Strategy 2: Bibliographic search + title validation — score = fuzzy ratio

    Returns:
        (CrossrefWorkItem, score) tuple or None if no match found.
    """
    # Strategy 1: DOI lookup
    if ref.doi:
        if stats is not None:
            stats.doi_attempts += 1
        try:
            result = await client.works(ids=ref.doi)
            if result and "message" in result:
                cr_item = CrossrefWorkItem.from_raw(result["message"])
                if ref.year and cr_item.year and abs(ref.year - cr_item.year) > 1:
                    logger.warning(
                        "DOI %s year mismatch: ref=%d, crossref=%d — discarding match",
                        ref.doi,
                        ref.year,
                        cr_item.year,
                    )
                    return None
                if stats is not None:
                    stats.doi_matches += 1
                return cr_item, 100.0
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("DOI not found in Crossref (404): %s", ref.doi)
                return None
            logger.debug("Crossref DOI lookup HTTP error for %s: %s", ref.doi, e)
            _record_terminal_failure(stats, ref, "DOI lookup", e)
        except Exception as e:
            logger.debug("Crossref DOI lookup failed for %s: %s", ref.doi, e)
            _record_terminal_failure(stats, ref, "DOI lookup", e)

    # Strategy 2: Bibliographic search (requires a meaningful title)
    if ref.title and len(ref.title) >= 10:
        if stats is not None:
            stats.search_attempts += 1
        query_parts = [ref.title]
        if ref.authors:
            # Use author string for better precision
            query_parts.append(ref.authors)
        if ref.year:
            query_parts.append(str(ref.year))

        query_str = " ".join(query_parts)
        try:
            result = await client.search(query=query_str, limit=3)
            if result and "message" in result and "items" in result["message"]:
                items = result["message"]["items"]
                match = _find_best_match(
                    ref.title,
                    items,
                    ref_year=ref.year,
                    ref_authors=ref.authors,
                    **_printed_fields(ref),
                )
                if match is not None and stats is not None:
                    stats.search_matches += 1
                return match
        except Exception as e:
            logger.debug(f"Crossref search failed for '{ref.title[:50]}...': {e}")
            _record_terminal_failure(stats, ref, "bibliographic search", e)

    # Strategy 3: title-less fingerprint match.
    # Science/Nature/physics citation styles omit the article title entirely
    # ("Author, Journal Vol, Page (Year)."), so Strategies 1-2 never fire. Match on
    # the printed fingerprint instead — author + container + volume + year — via
    # CrossRef's bibliographic query, validated on a strict field fingerprint rather
    # than a title. Gated to refs that carry a volume (the discriminating field for a
    # journal reference) so the path stays high-precision.
    title_usable = bool(ref.title) and len(ref.title) >= 10
    # A container OR a first page: either is enough to validate against once
    # author, volume and year agree, and parsers drop the container often enough
    # that requiring it discards recoverable references.
    fingerprint_usable = (
        ref.authors and ref.volume and ref.year and (ref.container or ref.first_page)
    )
    if not title_usable and fingerprint_usable:
        if stats is not None:
            stats.fingerprint_attempts += 1
        try:
            # Deeper than the title path's 3: a title-less fingerprint query is
            # bag-of-words over author+journal+volume+page, so the correct record
            # frequently ranks below 3 on free-text relevance. Precision is enforced
            # by _score_fingerprint's exact volume/page/year/container gates, not by
            # truncating the candidate list.
            result = await client.search(query=_fingerprint_query(ref), limit=_FINGERPRINT_ROWS)
            if result and "message" in result and "items" in result["message"]:
                match = _find_best_fingerprint_match(ref, result["message"]["items"])
                if match is not None and stats is not None:
                    stats.fingerprint_matches += 1
                return match
        except Exception as e:
            logger.debug(f"Crossref fingerprint search failed for bib_id={ref.bib_id}: {e}")
            _record_terminal_failure(stats, ref, "fingerprint search", e)

    return None


def _fingerprint_query(ref) -> str:
    """Build a CrossRef bibliographic query for a title-less reference from its
    printed fingerprint (authors, container, volume, first page, year)."""
    parts = [ref.authors, ref.container, str(ref.volume)]
    if ref.first_page:
        parts.append(str(ref.first_page))
    if ref.year:
        parts.append(str(ref.year))
    return " ".join(p for p in parts if p)


def _norm_token(value) -> str:
    """Normalize a volume/page token for equality: strip whitespace, lowercase, and
    drop leading zeros so ``"04"`` == ``"4"``."""
    return re.sub(r"\s+", "", str(value)).lower().lstrip("0") or "0"


# Crossref returns page ranges with whichever dash the publisher deposited —
# ASCII hyphen, en dash, em dash, or a double hyphen. Splitting on "-" alone
# left "123–130" whole, so ``first_page`` became the entire range and
# consolidation wrote that back into ``bib``.
_PAGE_RANGE_SEP_RE = re.compile(r"\s*(?:--|[-‐‑‒–—―])\s*")


def _split_page_range(page: str | None) -> tuple[str | None, str | None]:
    """Split a printed page range into ``(first_page, last_page)``."""
    if not page:
        return None, None
    parts = _PAGE_RANGE_SEP_RE.split(page.strip(), maxsplit=1)
    first = parts[0].strip() or None
    last = parts[1].strip() or None if len(parts) > 1 else None
    return first, last


def _fingerprint_field_match(ref_val, cand_val) -> bool:
    """True unless BOTH values are present and differ. A value missing on either side
    is uninformative, not disqualifying."""
    if not ref_val or not cand_val:
        return True
    return _norm_token(ref_val) == _norm_token(cand_val)


def _score_fingerprint(ref, cand: CrossrefWorkItem) -> float | None:
    """Score a CrossRef candidate for a *title-less* reference on its bibliographic
    fingerprint (no title to fuzz-match).

    Hard gates, all required: first-author surname overlap, year within ±1, container
    similarity >= threshold, and an exact volume match (volume is the discriminating
    field for a journal reference). A first page present on both sides must also agree.
    Returns the container similarity as the confidence, or None if any gate fails.
    """
    if not ref.authors:
        return None
    cand_families = [a.family for a in cand.authors if a.family]
    # First author only. Accepting ANY co-author (what this used to do, despite
    # the docstring) makes the gate nearly free on a large collaboration: a
    # 200-author paper matches almost any reference that shares one name.
    if not cand_families or not _families_overlap(ref.authors, cand_families[:1]):
        return None
    if not (ref.year and cand.year and abs(ref.year - cand.year) <= 1):
        return None
    if _initials_conflict(ref.authors, cand.authors):
        return None
    cand_first_page = _split_page_range(cand.page)[0]
    if ref.container and cand.container_title:
        # token_sort_ratio, not token_set_ratio: the "set" variant scores a subset
        # against its superset as a perfect 100, so "Nature" == "Nature
        # Communications" and every Nature-family journal collapsed into one.
        csim = _container_matches(ref.container, cand.container_title)
        if csim is None:
            return None
    elif ref.first_page and cand_first_page:
        # No container to compare — the parser dropped it, or the style never
        # printed one. First author + year + volume + first page still pin the
        # article, so stand in for the container score rather than discarding a
        # reference the remaining fields identify. Volume alone would not do:
        # it spans a journal-year.
        csim = float(_CONTAINER_MATCH_THRESHOLD)
    else:
        return None
    if not (ref.volume and cand.volume and _norm_token(ref.volume) == _norm_token(cand.volume)):
        return None
    if not _fingerprint_field_match(ref.first_page, cand_first_page):
        return None
    return float(csim)


def _parse_search_items(items: list[dict]) -> list[CrossrefWorkItem]:
    """Search rows as work items. A row that does not parse is skipped: every
    row is parsed before scoring, and one malformed record must not fail the
    reference's whole search."""
    parsed: list[CrossrefWorkItem] = []
    for item in items:
        try:
            parsed.append(CrossrefWorkItem.from_raw(item))
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("Skipping a Crossref search row that does not parse: %s", e)
    return parsed


def _find_best_fingerprint_match(ref, items: list[dict]) -> tuple[CrossrefWorkItem, float] | None:
    """Pick the highest-scoring candidate that clears every fingerprint gate."""
    best_item = None
    best_score = 0.0
    for cand in _parse_search_items(items):
        if not _doi_agrees(ref.doi, cand.doi):
            continue
        score = _score_fingerprint(ref, cand)
        if score is not None and score > best_score:
            best_score = score
            best_item = cand
    if best_item is not None:
        return best_item, best_score
    return None


# Crossref work types that are *about* a work rather than the work itself. A
# reference never cites these, but their titles embed the cited title verbatim
# ("Review of: X" scores 89.5 against "X"), so they clear the fuzzy title gate.
_VETOED_WORK_TYPES = frozenset({"peer-review", "component", "grant", "journal-issue"})

# Titles of works *about* another work. Crossref types these `journal-article`,
# so _VETOED_WORK_TYPES can't catch them, yet they embed the cited title verbatim
# and clear the fuzzy gate ("Erratum: X" scores 92.7 against "X").
_ABOUT_TITLE_PREFIX = re.compile(
    r"^\s*(review of|comment(?:ary)? on|reply to|response to|erratum|corrigendum"
    r"|correction|author response|editorial on|retraction)\b",
    re.IGNORECASE,
)


def _about_the_work(ref_title: str, cand_title: str) -> bool:
    """True when the candidate is a commentary on the reference rather than it.

    Only fires when the *candidate* carries the prefix and the *reference* does
    not, so a reference that genuinely cites an erratum still resolves to it.
    """
    return bool(_ABOUT_TITLE_PREFIX.match(cand_title)) and not _ABOUT_TITLE_PREFIX.match(ref_title)


# Dropped when matching a printed journal abbreviation against a full title:
# ISO4 omits them ("Q. J. Econ." -> "Quarterly Journal of Economics").
_CONTAINER_STOPWORDS = frozenset({"of", "the", "and", "for", "in", "on", "a", "an"})

# A parenthesised expansion appended to a printed journal abbreviation.
_CONTAINER_GLOSS_RE = re.compile(r"\([^)]*\)")


def _abbrev_container_match(ref_container: str, cand_container: str) -> bool:
    """True if ``ref_container`` reads as an ISO4/LTWA-style abbreviation of
    ``cand_container``.

    The citation styles that omit article titles are the same ones that abbreviate
    journal names, so the fingerprint path routinely compares "Clim. Change" against
    Crossref's "Climatic Change" — which ``token_sort_ratio`` scores at 81.5, under
    the 85 gate. Match token-by-token instead: an abbreviated token matches a
    candidate token it prefixes, and candidate stopwords may be skipped.
    """
    ref_tokens = [t for t in re.split(r"[^\w.]+", ref_container.lower()) if t.strip(".")]
    cand_tokens = [t for t in re.split(r"[^\w]+", cand_container.lower()) if t]
    if not ref_tokens or not cand_tokens:
        return False
    ci = 0
    for rt in ref_tokens:
        # Prefix-match whether or not the token carries ISO4's period: publishers
        # print "J. Risk Uncertain" for "Journal of Risk and Uncertainty", marking
        # only some of the truncations. Over-matching is held in check by the
        # consume-the-whole-candidate rule below, not by the period.
        stem = rt.rstrip(".")
        while ci < len(cand_tokens):
            ct = cand_tokens[ci]
            ci += 1
            if ct.startswith(stem):
                break
            if ct in _CONTAINER_STOPWORDS:
                continue
            return False
        else:
            return False
    # The abbreviation must account for the WHOLE candidate name. Without this,
    # "Nature" consumes the first token of "Nature Communications" and every
    # Nature-family journal collapses into one.
    return all(ct in _CONTAINER_STOPWORDS for ct in cand_tokens[ci:])


def _container_matches(ref_container: str, cand_container: str) -> float | None:
    """Container confidence for the fingerprint gate, or None if it fails."""
    # Science-style bibliographies gloss an acronym in parentheses -- "Commun. ACM
    # (Assoc. Comput. Machin.)" -- which no masthead carries, so the gloss can only
    # depress the score. Drop it before comparing.
    ref_container = _CONTAINER_GLOSS_RE.sub(" ", ref_container).strip() or ref_container
    csim = fuzz.token_sort_ratio(ref_container.lower(), cand_container.lower())
    if csim >= _CONTAINER_MATCH_THRESHOLD:
        return float(csim)
    if _abbrev_container_match(ref_container, cand_container):
        return float(_CONTAINER_MATCH_THRESHOLD)
    # Crossref stores the expanded masthead with its subtitle attached
    # ("JAMA: The Journal of the American Medical Association"); the printed
    # abbreviation only ever covers the head, so retry against that alone.
    head, sep, _ = cand_container.partition(":")
    if sep and head.strip() and _abbrev_container_match(ref_container, head.strip()):
        return float(_CONTAINER_MATCH_THRESHOLD)
    return None


# Letters NFKD leaves whole because their accent is not a separable mark.
_NAME_BASE_LETTERS = str.maketrans(
    {"ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ı": "i", "ß": "ss"}
    | {"æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "þ": "th", "Þ": "TH"}
)
# The German transliteration of an umlaut: "Müller" is deposited as "Mueller" as
# often as "Muller".
_UMLAUT_SPELLED_OUT = str.maketrans(
    {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}
)


# Small on purpose: the keys include whole reference-author strings, and one
# reference's string stays hot while its candidates' surnames are compared.
@functools.lru_cache(maxsize=256)
def _folded_name_text(text: str, umlauts_spelled_out: bool) -> tuple[str, tuple[int, ...]]:
    """``text`` accent-folded and casefolded, with each folded character's index
    in ``text`` so a match found in the folded form maps back to the original."""
    out: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        if umlauts_spelled_out:
            char = char.translate(_UMLAUT_SPELLED_OUT)
        piece = "".join(
            c
            for c in unicodedata.normalize("NFKD", char.translate(_NAME_BASE_LETTERS))
            if not unicodedata.combining(c)
        ).casefold()
        out.append(piece)
        offsets.extend([index] * len(piece))
    return "".join(out), tuple(offsets)


@functools.lru_cache(maxsize=256)
def _umlaut_spellings(text: str) -> tuple[bool, ...]:
    """The ``umlauts_spelled_out`` values worth folding ``text`` with: spelling
    out changes nothing in a text without an umlaut."""
    return (False, True) if text != text.translate(_UMLAUT_SPELLED_OUT) else (False,)


def _surname_spans(ref_authors: str, family: str) -> list[tuple[int, int]]:
    """Every ``(start, end)`` in ``ref_authors`` where ``family`` is printed as a
    word, ignoring case and diacritics ("Gonzalez" finds "González", "Mueller"
    finds "Müller" and the reverse). Registry deposits and PDF text layers drop
    or transliterate accents independently of each other."""
    spans: set[tuple[int, int]] = set()
    for spelled_out in _umlaut_spellings(ref_authors):
        text, offsets = _folded_name_text(ref_authors, spelled_out)
        for family_spelled_out in _umlaut_spellings(family):
            needle = _folded_name_text(family, family_spelled_out)[0]
            if not needle or needle not in text:
                continue
            for m in re.finditer(r"\b" + re.escape(needle) + r"\b", text):
                spans.add((offsets[m.start()], offsets[m.end() - 1] + 1))
    return sorted(spans)


def _initial(letter: str) -> str:
    """``letter`` as an initial: upper-cased, accent dropped ("é" and "E" agree)."""
    return (_folded_name_text(letter, False)[0][:1] or letter).upper()


# Where one printed name ends: punctuation, "&", the word for "and" in the
# languages reference lists are printed in, and anything else that cannot sit
# inside a personal name ("(Eds)", "2010", ":"). The conjunctions match
# lower-case only, so an initial ("E.", "Y.") is never taken for Italian "e"
# or Spanish "y".
_NAME_BOUNDARY_RE = re.compile(
    r"(?<![^\W\d_])(?:and|et|und|y|e|og|och|en)(?![^\W\d_]|\.)|[^\w\s.\-‐'’]|[\d_]"
)
_NAME_TOKEN_RE = re.compile(r"[^\W\d_]+\.?")
_DOTTED_INITIALS_RE = re.compile(r"\s+((?:[^\W\d_]\.\s*)+)")
# Characters that join two halves of one surname ("Karkhoff-Schweizer").
_SURNAME_JOINERS = frozenset("-‐'’")
# Lower-case surname particles printed between a given name and the surname
# ("Ludwig van Beethoven" when Crossref files the family as "Beethoven").
_SURNAME_PARTICLES = frozenset(
    {"van", "von", "der", "den", "de", "del", "della", "di", "da", "du", "la", "le"}
    | {"dos", "das", "ten", "ter", "zu", "zum"}
)


def _initials_before_surname(before: str) -> set[str]:
    """Initials of the given names printed directly before a surname, read back to
    the previous name boundary: "Thomas E." in "…, Thomas E. Brownlee" gives
    {T, E}.

    Empty when that text is not a given name: nothing, a separator ("and",
    ","), or a bare Vancouver-style initial without a period right before the
    surname ("Smith J Weber"), which is more likely the previous author's.
    """
    tokens = _NAME_TOKEN_RE.findall(_NAME_BOUNDARY_RE.split(before)[-1])
    while tokens and tokens[-1] in _SURNAME_PARTICLES:
        tokens.pop()
    initials: set[str] = set()
    for position, token in enumerate(tokens):
        letters = token.rstrip(".")
        if len(letters) == 1 and (token.endswith(".") or position < len(tokens) - 1):
            initials.add(_initial(letters))
        elif len(letters) > 1 and letters[0].isupper() and not letters.isupper():
            initials.add(_initial(letters[0]))
        else:
            return set()
    return initials


def _initials_after_surname(
    after: str, *, inverted: bool, look_ahead: bool = True
) -> tuple[set[str], bool]:
    """Initials printed after a surname and its comma ("Weber, E. U."), and
    whether they are bare initials only.

    Spelled given names count too: one before any initials ("Weber, Elke U."),
    or any number when the name is ``inverted`` — nothing printed before the
    surname, as for the first author of a Chicago list ("Lam, Hui Kwan
    Nicholas, John Sproule"). Otherwise a spelled word after the first is the
    next author's surname ("Brownlee, Liam D. Harper"), and nothing is read.
    Nothing is read either when the word after the comma is itself a surname
    followed by its initials ("Al-Muzaini, Beg, K. R.", where Al-Muzaini's were
    not printed): with ``look_ahead`` the next comma-separated segment is read,
    one segment only, to find out. Without a comma only dotted initials count
    ("Emons P.A.A.").
    """
    comma = re.match(r"\s*,", after)
    if comma is None:
        # A comma-less Vancouver name with dotted initials ("Emons P.A.A.").
        dotted = _DOTTED_INITIALS_RE.match(after)
        if dotted is None:
            return set(), False
        return {_initial(c) for c in dotted.group(1) if c.isalpha()}, True
    rest = after[comma.end() :]
    boundary = _NAME_BOUNDARY_RE.search(rest)
    cut = boundary.start() if boundary else len(rest)
    initials: set[str] = set()
    bare = True
    for position, token in enumerate(_NAME_TOKEN_RE.findall(rest[:cut])):
        letters = token.rstrip(".")
        if len(letters) == 1 or (letters.isupper() and len(letters) <= 3):
            initials.update(_initial(letter) for letter in letters)
        elif letters[0].isupper() and (inverted or position == 0):
            initials.add(_initial(letters[0]))
            bare = False
        else:
            return set(), False
    if (
        look_ahead
        and not bare
        and _initials_after_surname(rest[cut:], inverted=True, look_ahead=False)[1]
    ):
        return set(), False
    return initials, bare and bool(initials)


def _ref_initials_for_surname(ref_authors: str, family: str) -> set[str]:
    """Every given-name initial printed alongside ``family`` in the raw ref author
    string. Empty when the reference exposes none.

    Returns the whole run, not just the first: a reference printing "H. N.
    Rehnqvist" is the same person as Crossref's given "N.", and comparing only
    the leading initial rejects the correct record. A spelled given name counts
    with the initials after it ("Eric J. Johnson" gives {E, J}).

    Unions across *every* occurrence of the surname. Co-authors sharing one are
    ordinary ("A. C. Klassen, D. K. Klassen"); reading only the first vetoes the
    record whenever CrossRef happens to name the second. An occurrence inside a
    hyphenated surname ("Krolak-Salmon") is another person's name and is skipped.

    The given name is read from both sides of the surname: before it ("E. U.
    Weber") and, for an inverted name, after the comma ("Weber, E. U."). A
    conjunction is a separator, not a first name, so "Smith, J. and Weber, E.
    U." reads Weber as {E, U}. Both readings are kept because either can be the
    right one: in "Wansink, B. Sobal, J." the separator after "B." is missing,
    and in "Giraldo Peláez, Santiago" the word before the surname is part of it.
    """
    initials: set[str] = set()
    for start, end in _surname_spans(ref_authors, family):
        if ref_authors[start - 1 : start] in _SURNAME_JOINERS or (
            ref_authors[end : end + 1] in _SURNAME_JOINERS
        ):
            continue
        before = _initials_before_surname(ref_authors[:start])
        initials |= before | _initials_after_surname(ref_authors[end:], inverted=not before)[0]
    return initials


# Titles Crossref deposits inside a given name ("Prof Haiyue", "Dr Rui").
_HONORIFICS = frozenset({"prof", "professor", "dr", "mr", "mrs", "ms", "miss", "sir", "dame"})


def _given_initial(given: str) -> str | None:
    """The initial of a deposited given name, past any honorific."""
    words = given.split()
    while len(words) > 1 and words[0].rstrip(".").casefold() in _HONORIFICS:
        words.pop(0)
    letter = next((c for c in " ".join(words) if c.isalpha()), None)
    return _initial(letter) if letter else None


def _initials_conflict(ref_authors: str | None, cand_authors: list) -> bool:
    """True when the reference and candidate share a surname but disagree on that
    author's first initial.

    Surname-only validation accepts a book by *Max* Weber for a reference by
    *E. U.* Weber. Only vetoes when both sides actually expose an initial, so a
    reference that prints bare surnames is unaffected. Candidate authors who
    share a surname are pooled: a reference cut short by "et al." prints one
    Soltis, and the candidate's other Soltis is no evidence against it.
    """
    if not ref_authors or not cand_authors:
        return False
    by_family: dict[str, tuple[str, set[str]]] = {}
    for author in cand_authors:
        family = author.get("family") if isinstance(author, dict) else getattr(author, "family", "")
        given = author.get("given") if isinstance(author, dict) else getattr(author, "given", "")
        if not isinstance(family, str) or len(family) < 2 or not isinstance(given, str):
            continue
        cand_initial = _given_initial(given)
        if cand_initial:
            key = _folded_name_text(family, False)[0]
            by_family.setdefault(key, (family, set()))[1].add(cand_initial)
    for family, cand_initials in by_family.values():
        ref_initials = _ref_initials_for_surname(ref_authors, family)
        if ref_initials and not ref_initials & cand_initials:
            return True
    return False


def _families_overlap(ref_authors: str, families: list[str]) -> bool:
    """True if any family surname (>= 2 chars) appears as a word in the ref author
    string, ignoring case and diacritics."""
    return any(
        family and len(family) >= 2 and _surname_spans(ref_authors, family) for family in families
    )


def _score_candidate(
    ref_title: str,
    cand_title: str,
    ref_year: int | None,
    cand_year: int | None,
    ref_authors: str | None,
    cand_families: list[str],
) -> float:
    """Shared rapidfuzz scoring for one candidate, source-agnostic.

    token_sort_ratio on lowercased titles; year >2 apart -> x0.5; title >= threshold
    with author surnames present but no overlap -> x0.7. An organization author
    (a ``name`` with no family) carries no surname to compare, so it is no
    evidence against the match.
    """
    pre_score = fuzz.token_sort_ratio(ref_title.lower(), cand_title.lower())
    final_score = float(pre_score)
    if ref_year and ref_year > 0 and cand_year and abs(ref_year - cand_year) > 2:
        final_score *= 0.5
    cand_families = [f for f in cand_families if f]
    if (
        ref_authors
        and pre_score >= _TITLE_MATCH_THRESHOLD
        and cand_families
        and not _families_overlap(ref_authors, cand_families)
    ):
        final_score *= 0.7
    return final_score


@dataclass(frozen=True)
class _TitleCandidate:
    """A title-search candidate reduced to what the title matcher compares,
    whichever service returned it: a Crossref work item or a resolver dict.

    The Crossref search and the resolver (primary and fallback) used to run
    separate copies of the matching loop, and fixes landed in one copy and not
    the others. Both shapes now go through :func:`_best_title_candidate`.
    """

    title: str | None
    year: int | None
    work_type: str | None
    doi: str | None
    authors: tuple[CrossrefAuthor, ...]
    container: str | None
    volume: str | None
    first_page: str | None

    @classmethod
    def from_crossref(cls, item: CrossrefWorkItem) -> _TitleCandidate:
        return cls(
            title=item.title,
            year=item.year,
            work_type=item.work_type,
            doi=item.doi,
            authors=tuple(item.authors),
            container=item.container_title,
            volume=item.volume,
            first_page=_split_page_range(item.page)[0],
        )

    @classmethod
    def from_resolver(cls, cand: object) -> _TitleCandidate | None:
        """``None`` for a candidate that is not a JSON object. ``authors: null``
        and non-object author entries are treated as no authors."""
        if not isinstance(cand, dict):
            return None
        year = cand.get("year")
        return cls(
            title=plain_text(cand.get("title")),
            year=year if isinstance(year, int) else None,
            work_type=_str_or_none(cand.get("type")),
            doi=_str_or_none(cand.get("doi")),
            authors=tuple(
                CrossrefAuthor(
                    given=_str_or_none(a.get("given")) or "",
                    family=_str_or_none(a.get("family")) or "",
                )
                for a in cand.get("authors") or []
                if isinstance(a, dict)
            ),
            container=plain_text(cand.get("container")),
            volume=_str_or_none(cand.get("volume")),
            first_page=_str_or_none(cand.get("first_page")),
        )


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _doi_agrees(printed_doi: str | None, candidate_doi: str | None) -> bool:
    """A reference that prints a DOI names its work: a candidate is that work only
    if it carries the same DOI. A printed DOI that does not parse is no evidence
    either way — those are the references whose DOI came out of OCR mangled."""
    printed = normalize_doi(printed_doi) if printed_doi else None
    if printed is None:
        return True
    candidate = normalize_doi(candidate_doi) if candidate_doi else None
    return candidate is not None and candidate.casefold() == printed.casefold()


# At most this many words, a title does not identify a work on its own
# ("Introduction", "Emotion regulation", "Capital structure").
_GENERIC_TITLE_WORDS = 3


def _printed_fields_conflict(
    title: str,
    cand: _TitleCandidate,
    *,
    container: str | None,
    volume: str | None,
    first_page: str | None,
) -> bool:
    """True when the printed volume, first page or container rule the candidate out.

    A value missing on either side is no evidence. A volume *and* a first page
    that both disagree name a different article whatever the title says. A
    generic title needs more: any printed volume must agree too, and so must
    any printed container, unless the volume and first page are both printed
    and agree, which pins the article even when the container is an
    abbreviation :func:`_container_matches` cannot expand ("PNAS", "Br Med J").
    One disagreeing field alone does not veto a distinctive title, because
    parsers misread a volume ("19:262") or a page (an article number) often
    enough to cost correct matches.
    """
    volume_differs = not _fingerprint_field_match(volume, cand.volume)
    page_differs = not _fingerprint_field_match(first_page, cand.first_page)
    if volume_differs and page_differs:
        return True
    if len(title.split()) > _GENERIC_TITLE_WORDS:
        return False
    if volume_differs:
        return True
    if volume and cand.volume and first_page and cand.first_page and not page_differs:
        return False
    return bool(
        container and cand.container and _container_matches(container, cand.container) is None
    )


def _best_title_candidate(
    title: str,
    candidates: list[_TitleCandidate | None],
    ref_year: int | None = None,
    ref_authors: str | None = None,
    *,
    doi: str | None = None,
    container: str | None = None,
    volume: str | None = None,
    first_page: str | None = None,
) -> tuple[int, float] | None:
    """Index and final score of the best candidate at or above the title threshold.

    Selects the candidate with the highest *final* (post-penalty) score so
    that a strong-but-penalized candidate can't displace a weaker but
    unpenalized one below the threshold gate. Vetoes, before scoring: works
    *about* the cited work, a DOI other than the printed one, printed
    bibliographic fields that rule the candidate out, and a same-surname
    author with another initial.
    """
    best_index = None
    best_score = 0.0
    for index, cand in enumerate(candidates):
        if cand is None or not cand.title:
            continue
        if cand.work_type in _VETOED_WORK_TYPES:
            continue
        if _about_the_work(title, cand.title):
            continue
        if not _doi_agrees(doi, cand.doi):
            continue
        if _printed_fields_conflict(
            title, cand, container=container, volume=volume, first_page=first_page
        ):
            continue
        if _initials_conflict(ref_authors, list(cand.authors)):
            continue
        score = _score_candidate(
            title, cand.title, ref_year, cand.year, ref_authors, [a.family for a in cand.authors]
        )
        if score > best_score:
            best_score = score
            best_index = index
    if best_score >= _TITLE_MATCH_THRESHOLD and best_index is not None:
        return best_index, best_score
    return None


def _find_best_match(
    title: str,
    items: list[dict],
    ref_year: int | None = None,
    ref_authors: str | None = None,
    *,
    doi: str | None = None,
    container: str | None = None,
    volume: str | None = None,
    first_page: str | None = None,
) -> tuple[CrossrefWorkItem, float] | None:
    """Find the best-matching Crossref search item (see :func:`_best_title_candidate`).

    Returns:
        (CrossrefWorkItem, final_score) tuple or None if no match above threshold.
    """
    parsed = _parse_search_items(items)
    best = _best_title_candidate(
        title,
        [_TitleCandidate.from_crossref(item) for item in parsed],
        ref_year,
        ref_authors,
        doi=doi,
        container=container,
        volume=volume,
        first_page=first_page,
    )
    if best is None:
        return None
    index, score = best
    return parsed[index], score


def _build_match(cr_item: CrossrefWorkItem, score: float) -> ExternalMatch:
    """Build an ExternalMatch from a parsed CrossrefWorkItem."""
    bib_type = migrate_bib_type(cr_item.work_type) if cr_item.work_type else None

    authors = None
    if cr_item.authors:
        author_list = [
            BibAuthor(
                given=a.given,
                family=a.family,
                orcid=canonicalize_orcid(a.orcid),
                affiliation=[
                    MatchOrganization(name=org.name, ror=org.ror) for org in a.affiliations
                ]
                or None,
            )
            for a in cr_item.authors
            if a.family
        ]
        authors = author_list if author_list else None

    editors = None
    if cr_item.editors:
        editor_list = [
            BibAuthor(given=e.given, family=e.family) for e in cr_item.editors if e.family
        ]
        editors = editor_list if editor_list else None

    first_page, last_page = _split_page_range(cr_item.page)

    return ExternalMatch(
        id=cr_item.doi,
        score=score,
        title=cr_item.title,
        authors=authors,
        year=cr_item.year,
        container=cr_item.container_title,
        volume=cr_item.volume,
        issue=cr_item.issue,
        first_page=first_page,
        last_page=last_page,
        publisher=cr_item.publisher,
        editors=editors,
        doi=cr_item.doi,
        bib_type=bib_type,
        url=cr_item.url,
        date=cr_item.date,
        license_url=cr_item.license_url,
        funders=[
            MatchFunder(name=f.name, funder_doi=f.funder_doi, ror=f.ror, award_ids=f.award_ids)
            for f in cr_item.funders
        ]
        or None,
    )


def _build_match_from_candidate(cand: dict, score: float) -> ExternalMatch:
    """Build an ExternalMatch from a bibr-resolver Candidate dict.

    Mirrors _build_match (the CrossRef mapper). The resolver's `type` is
    Crossref-aligned, so it goes through the same migrate_bib_type. first_page/
    last_page arrive already split. edition/version are not supplied by the
    resolver (left None).
    """
    work_type = cand.get("type")
    bib_type = migrate_bib_type(work_type) if work_type else None

    def people(key: str) -> list[BibAuthor] | None:
        # ``"authors": null`` and non-object entries are no people, not a crash.
        listed = [
            BibAuthor(given=p.get("given") or "", family=p["family"])
            for p in cand.get(key) or []
            if isinstance(p, dict) and p.get("family")
        ]
        return listed or None

    return ExternalMatch(
        id=cand.get("doi") or cand.get("id"),
        score=score,
        title=plain_text(cand.get("title")),
        authors=people("authors"),
        year=cand.get("year"),
        container=plain_text(cand.get("container")),
        volume=cand.get("volume"),
        issue=cand.get("issue"),
        first_page=cand.get("first_page"),
        last_page=cand.get("last_page"),
        publisher=cand.get("publisher"),
        editors=people("editors"),
        doi=cand.get("doi"),
        bib_type=bib_type,
        url=cand.get("url"),
        date=cand.get("date"),
    )


def _best_resolver_match(
    title: str,
    candidates: list[dict],
    ref_year: int | None = None,
    ref_authors: str | None = None,
    *,
    doi: str | None = None,
    container: str | None = None,
    volume: str | None = None,
    first_page: str | None = None,
) -> tuple[dict, float] | None:
    """Pick the best resolver candidate (see :func:`_best_title_candidate`)."""
    best = _best_title_candidate(
        title,
        [_TitleCandidate.from_resolver(cand) for cand in candidates],
        ref_year,
        ref_authors,
        doi=doi,
        container=container,
        volume=volume,
        first_page=first_page,
    )
    if best is None:
        return None
    index, score = best
    return candidates[index], score


def _resolver_query_eligible(ref) -> bool:
    """True when _try_resolver has a branch it can actually take for this ref —
    a DOI lookup or a title search. False means no resolver query was issued, so a
    None outcome is 'not asked', not 'asked and missed'."""
    return bool(ref.doi) or (bool(ref.title) and len(ref.title) >= 10)


def _resolver_search_eligible(ref) -> bool:
    """True for refs the resolver resolves by *title search*: no DOI (DOI refs use the
    lookup_doi fast path) and a title long enough to query — matching the title branch of
    _try_resolver."""
    return not ref.doi and bool(ref.title) and len(ref.title) >= 10


def _effective_fallback_sources(settings: GlobalSettings) -> list[str]:
    """Return fallback sources not already queried by the primary pass."""
    primary = set(settings.resolver.sources)
    seen: set[str] = set()
    effective: list[str] = []
    for source in settings.resolver.fallback_sources:
        if source not in primary and source not in seen:
            effective.append(source)
            seen.add(source)
    return effective


def _resolver_fallback_eligible(ref: PaperReference) -> bool:
    """A second-pass search only considers still-unmatched, title-bearing refs."""
    return not ref.match and bool(ref.title) and len(ref.title) >= 10


async def _enrich_resolver_fallback(
    references: list[PaperReference],
    resolver_client,
    stats: ResolutionStats,
    *,
    settings: GlobalSettings,
) -> None:
    """Resolve primary misses through separately configured resolver sources.

    Each reference's search is applied as soon as it answers, so the
    whole-paper deadline cancels only the searches still outstanding. A single
    batched call returned nothing until its slowest query had, and a timeout
    threw away every search that had already come back.
    """
    sources = _effective_fallback_sources(settings)
    eligible = [ref for ref in references if _resolver_fallback_eligible(ref)]
    if not sources or not eligible:
        return

    stats.fallback_attempts += len(eligible)
    semaphore = asyncio.Semaphore(settings.resolver.fallback_search_concurrency)
    answered = 0

    async def resolve(ref: PaperReference) -> None:
        nonlocal answered
        try:
            async with semaphore:
                candidates = await resolver_client.search(
                    ref.title,
                    ref.year,
                    settings.resolver.limit,
                    sources=sources,
                    raise_on_error=True,
                )
            best = _best_resolver_match(
                ref.title,
                candidates,
                ref_year=ref.year,
                ref_authors=ref.authors,
                **_printed_fields(ref),
            )
        except Exception as e:  # noqa: BLE001 — one bad answer costs only its own reference
            answered += 1
            stats.fallback_errors += 1
            diagnostic = " ".join(str(e).split())
            _record_fallback_warning(
                stats,
                WarningCode.RESOLVER_FALLBACK_FAILED,
                f"bib_id={ref.bib_id} resolver fallback failed"
                + (f": {diagnostic}" if diagnostic else ""),
            )
            return
        answered += 1
        if best is None:
            stats.fallback_misses += 1
            return
        candidate, score = best
        if ref.match:
            return
        ref.match[_resolver_match_source(candidate)] = _build_match_from_candidate(candidate, score)
        stats.fallback_matches += 1

    try:
        async with asyncio.timeout(settings.resolver.fallback_timeout):
            await asyncio.gather(*(resolve(ref) for ref in eligible))
    except TimeoutError:
        stats.fallback_timeouts += 1
        detail = (
            f"resolver fallback timed out after {settings.resolver.fallback_timeout:g}s "
            f"with {len(eligible) - answered} of {len(eligible)} refs unanswered"
        )
        logger.warning(detail)
        _record_fallback_warning(stats, WarningCode.RESOLVER_FALLBACK_TIMEOUT, detail)


async def _prefetch_resolver_searches(
    references: list[PaperReference],
    resolver_client,
    *,
    settings: GlobalSettings | None = None,
    raise_on_error: bool = False,
) -> dict[int, list[dict] | Exception]:
    """Resolve every title-searchable reference concurrently, up front.

    Returns a map ``id(ref) -> candidate list``. Refs absent from the map (DOI-bearing or
    too-short titles) are handled on their own paths in _try_resolver. Per-query failures
    degrade to ``[]`` inside ResolverClient.search_many, so a slow or down resolver simply
    yields no candidates and the caller falls through to CrossRef.

    With ``raise_on_error=True`` (authoritative resolver) a per-query error is preserved as
    an ``Exception`` in that ref's slot rather than degrading to ``[]``, so _try_resolver can
    tell a genuine miss (empty list) apart from a transient error (which still falls through
    to CrossRef)."""
    eligible = [r for r in references if _resolver_search_eligible(r)]
    if not eligible:
        return {}
    effective = settings if settings is not None else snapshot_settings()
    results = await resolver_client.search_many(
        [{"title": r.title, "year": r.year, "limit": effective.resolver.limit} for r in eligible],
        concurrency=effective.resolver.search_concurrency,
        sources=effective.resolver.sources,
        raise_on_error=raise_on_error,
    )
    return {id(ref): cands for ref, cands in zip(eligible, results, strict=True)}


async def _try_resolver(
    ref,
    resolver_client,
    stats: ResolutionStats | None = None,
    prefetch: dict[int, list[dict] | Exception] | None = None,
    *,
    settings: GlobalSettings | None = None,
    raise_on_error: bool = False,
) -> tuple[MatchSource, ExternalMatch] | None:
    """Try to resolve a reference via the bibr-resolver service.

    Returns ``(MatchSource, ExternalMatch)`` — the MatchSource reflects the
    candidate's true ``source`` (OpenAlex or Crossref), the ExternalMatch scores
    100 for a DOI hit and the fuzzy ratio for a search hit — or None to fall
    through to CrossRef.

    Title-search candidates come from ``prefetch`` (the batch resolved up front in
    enrich_references) when supplied; ``prefetch=None`` falls back to a per-ref /search,
    preserving direct callers.

    With ``raise_on_error=True`` a transient resolver error propagates (a prefetch slot
    holding an ``Exception`` is re-raised, live calls pass the flag through) so the caller
    can distinguish it from a clean miss and still fall through to CrossRef on error."""
    if stats is not None:
        stats.resolver_attempts += 1
    effective = settings if settings is not None else snapshot_settings()

    if ref.doi:
        cand = await resolver_client.lookup_doi(ref.doi, raise_on_error=raise_on_error)
        if cand is None:
            return None
        cand_year = cand.get("year")
        if ref.year and cand_year and abs(ref.year - cand_year) > 1:
            logger.warning(
                "Resolver DOI %s year mismatch: ref=%d, resolver=%d — discarding match",
                ref.doi,
                ref.year,
                cand_year,
            )
            return None
        if stats is not None:
            stats.resolver_matches += 1
        return _resolver_match_source(cand), _build_match_from_candidate(cand, 100.0)

    if ref.title and len(ref.title) >= 10:
        if prefetch is not None:
            candidates = prefetch.get(id(ref), [])
            if isinstance(candidates, Exception):
                # Authoritative prefetch preserved a per-query error in this slot —
                # re-raise so _resolve_reference treats it as an error (→ CrossRef), not a miss.
                raise candidates
        else:
            candidates = await resolver_client.search(
                ref.title,
                ref.year,
                effective.resolver.limit,
                sources=effective.resolver.sources,
                raise_on_error=raise_on_error,
            )
        best = _best_resolver_match(
            ref.title,
            candidates,
            ref_year=ref.year,
            ref_authors=ref.authors,
            **_printed_fields(ref),
        )
        if best is not None:
            cand, score = best
            if stats is not None:
                stats.resolver_matches += 1
            return _resolver_match_source(cand), _build_match_from_candidate(cand, score)

    return None


async def _resolve_reference(
    ref,
    crossref_client,
    resolver_client,
    stats: ResolutionStats | None = None,
    resolver_prefetch: dict[int, list[dict] | Exception] | None = None,
    *,
    resolver_authoritative: bool = False,
    settings: GlobalSettings | None = None,
) -> tuple[MatchSource, ExternalMatch] | None:
    """Resolve one reference: resolver first (if available), else CrossRef.

    When ``resolver_authoritative`` is set the resolver is asserted to be backed by the
    same corpus as CrossRef, so a *clean* resolver miss skips the redundant CrossRef
    re-query. A resolver *error* (transport failure) still falls through to CrossRef —
    ``_try_resolver`` runs with ``raise_on_error`` so a miss and an error are distinguishable."""
    if resolver_client is not None:
        resolver_raised = False
        try:
            resolver_outcome = await _try_resolver(
                ref,
                resolver_client,
                stats,
                resolver_prefetch,
                settings=settings,
                raise_on_error=resolver_authoritative,
            )
        except Exception as e:
            logger.warning(
                "Resolver raised for bib_id=%s, falling back to CrossRef: %s",
                ref.bib_id,
                e,
            )
            resolver_outcome = None
            resolver_raised = True
            _record_terminal_failure(stats, ref, "resolver", e)
        if resolver_outcome is not None:
            return resolver_outcome
        # Clean miss (queried OK, no accepted match) against an authoritative resolver:
        # CrossRef would only re-query the same corpus, so skip it. Errors fall through.
        # A ref matching NEITHER resolver branch (no DOI, no usable title — every
        # Science/Nature-style citation) was never queried at all, so there is no miss
        # to honour; it must still reach CrossRef's title-less fingerprint path.
        if resolver_authoritative and not resolver_raised and _resolver_query_eligible(ref):
            return None

    cr = await _fetch_crossref_item(ref, crossref_client, stats)
    if cr is not None:
        cr_item, score = cr
        return MatchSource.CROSSREF, _build_match(cr_item, score)

    return None
