import asyncio
from unittest import mock

from bibr.clients.crossref import CrossrefClient
from bibr.models import PaperReference
from bibr.paper import MatchSource
from bibr.processing_warnings import WarningCode


def _mock_crossref():
    client = mock.MagicMock(spec_set=CrossrefClient)
    client.enrich_semaphore = asyncio.Semaphore(100)
    return client


def _make_ref(**kw):
    base = {
        "bib_id": 1,
        "title": "A Great Paper on Testing Methods",
        "first_page": None,
        "volume": None,
        "authors": None,
        "year": None,
        "container": None,
    }
    base.update(kw)
    return PaperReference(**base)


def _candidate(**kw):
    base = {"title": "A Great Paper on Testing Methods", "source": "openalex"}
    base.update(kw)
    return base


def _fallback_settings(**overrides):
    from bibr.config import GlobalSettings, ResolverOptions

    resolver_values = {
        "sources": ["crossref"],
        "fallback_sources": ["openalex"],
        "fallback_search_concurrency": 2,
        "fallback_timeout": 30.0,
        "authoritative": True,
    }
    resolver_values.update(overrides)
    return GlobalSettings(
        _env_file=None,
        resolver=ResolverOptions(_env_file=None, **resolver_values),
    )


async def test_resolver_doi_hit_stored_as_openalex_and_skips_crossref():
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi="10.1/abc", year=2020)
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.lookup_doi = mock.AsyncMock(return_value=_candidate(doi="10.1/abc", year=2020))

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.OPENALEX in ref.match
    assert MatchSource.CROSSREF not in ref.match
    assert ref.match[MatchSource.OPENALEX].score == 100.0
    crossref.works.assert_not_called()
    crossref.search.assert_not_called()


async def test_resolver_search_hit_stored_as_openalex():
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi=None, title="A Great Paper on Testing Methods")
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(
        return_value=[[_candidate(title="A Great Paper on Testing Methods", doi="10.1/x")]]
    )

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.OPENALEX in ref.match
    assert MatchSource.CROSSREF not in ref.match
    crossref.search.assert_not_called()


async def test_resolver_searches_are_prefetched_not_per_reference():
    """All no-DOI title searches resolve via ONE search_many prefetch call, not a /search
    call issued inline per reference. (DOI refs still use the per-ref lookup_doi fast path.)"""
    from bibr.enrich.references import enrich_references

    refs = [
        _make_ref(bib_id=1, doi=None, title="A Great Paper on Testing Methods"),
        _make_ref(bib_id=2, doi=None, title="Another Fine Study of Many Things"),
    ]
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(
        return_value=[
            [_candidate(title="A Great Paper on Testing Methods", doi="10.1/a")],
            [_candidate(title="Another Fine Study of Many Things", doi="10.1/b")],
        ]
    )

    await enrich_references(refs, crossref_client=crossref, resolver_client=resolver)

    resolver.search_many.assert_called_once()
    resolver.search.assert_not_called()  # prefetched, never called inline per-reference
    assert MatchSource.OPENALEX in refs[0].match
    assert MatchSource.OPENALEX in refs[1].match
    crossref.search.assert_not_called()


async def test_resolver_miss_falls_through_to_crossref():
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi=None, title="A Great Paper on Testing Methods")
    crossref = _mock_crossref()
    crossref.search = mock.AsyncMock(
        return_value={
            "message": {
                "items": [
                    {
                        "title": ["A Great Paper on Testing Methods"],
                        "DOI": "10.1/cr",
                        "issued": {"date-parts": [[2020]]},
                    }
                ]
            }
        }
    )
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])  # resolver miss

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.OPENALEX not in ref.match
    assert MatchSource.CROSSREF in ref.match
    crossref.search.assert_called_once()


async def test_no_resolver_is_pure_crossref():
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi="10.1/abc", year=2020)
    crossref = _mock_crossref()
    crossref.works = mock.AsyncMock(
        return_value={
            "message": {
                "DOI": "10.1/abc",
                "title": ["A Great Paper on Testing Methods"],
                "issued": {"date-parts": [[2020]]},
            }
        }
    )

    # resolver_client omitted entirely -> today's behavior
    await enrich_references([ref], crossref_client=crossref)

    assert MatchSource.CROSSREF in ref.match
    assert MatchSource.OPENALEX not in ref.match


async def test_resolver_doi_year_mismatch_discards_and_falls_through():
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi="10.1/abc", year=2000)
    crossref = _mock_crossref()
    crossref.works = mock.AsyncMock(return_value=None)  # crossref also misses
    resolver = mock.AsyncMock()
    resolver.lookup_doi = mock.AsyncMock(return_value=_candidate(doi="10.1/abc", year=2020))

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.OPENALEX not in ref.match  # year mismatch discarded
    crossref.works.assert_called_once()  # fell through to crossref DOI lookup


async def test_resolver_crossref_source_stored_as_crossref():
    """A resolver hit whose candidate is sourced from Crossref is tagged
    MatchSource.CROSSREF (provenance), not always OPENALEX — without ever
    calling the live Crossref client."""
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi="10.1/abc", year=2020)
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.lookup_doi = mock.AsyncMock(
        return_value=_candidate(source="crossref", doi="10.1/abc", year=2020)
    )

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.CROSSREF in ref.match
    assert MatchSource.OPENALEX not in ref.match
    assert ref.match[MatchSource.CROSSREF].score == 100.0
    crossref.works.assert_not_called()  # proves it came via the resolver, not live Crossref


async def test_resolver_missing_source_defaults_to_openalex():
    """A candidate with no `source` field falls back to OPENALEX (no regression)."""
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi="10.1/abc", year=2020)
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    cand = {"title": "A Great Paper on Testing Methods", "doi": "10.1/abc", "year": 2020}
    resolver.lookup_doi = mock.AsyncMock(return_value=cand)

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.OPENALEX in ref.match
    assert MatchSource.CROSSREF not in ref.match


async def test_unhealthy_resolver_skips_resolver_and_uses_crossref():
    """When the resolver reports unhealthy, the whole tier is skipped (no
    per-reference round-trip) and enrichment falls through to Crossref."""
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi="10.1/abc", year=2020)
    crossref = _mock_crossref()
    crossref.works = mock.AsyncMock(
        return_value={
            "message": {
                "DOI": "10.1/abc",
                "title": ["A Great Paper on Testing Methods"],
                "issued": {"date-parts": [[2020]]},
            }
        }
    )
    resolver = mock.AsyncMock()
    resolver.healthy = mock.AsyncMock(return_value=False)

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    resolver.lookup_doi.assert_not_called()
    resolver.search.assert_not_called()
    assert MatchSource.CROSSREF in ref.match
    assert MatchSource.OPENALEX not in ref.match


async def test_resolver_unexpected_error_falls_through_to_crossref():
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi=None, title="A Great Paper on Testing Methods")
    crossref = _mock_crossref()
    crossref.search = mock.AsyncMock(
        return_value={
            "message": {
                "items": [
                    {
                        "title": ["A Great Paper on Testing Methods"],
                        "DOI": "10.1/cr",
                        "issued": {"date-parts": [[2020]]},
                    }
                ]
            }
        }
    )
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(side_effect=RuntimeError("boom"))

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.OPENALEX not in ref.match
    assert MatchSource.CROSSREF in ref.match
    crossref.search.assert_called_once()


async def test_authoritative_resolver_search_miss_skips_crossref(monkeypatch):
    """In authoritative mode the resolver is backed by the same corpus as CrossRef,
    so a clean title-search miss must NOT re-query CrossRef."""
    from bibr.config import Settings
    from bibr.enrich.references import enrich_references

    monkeypatch.setattr(Settings.resolver, "authoritative", True)

    ref = _make_ref(doi=None, title="A Great Paper on Testing Methods")
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])  # clean resolver miss

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert not ref.match
    crossref.search.assert_not_called()
    crossref.works.assert_not_called()


async def test_authoritative_resolver_doi_miss_skips_crossref(monkeypatch):
    """A clean DOI miss (resolver 404 → None) is authoritative: CrossRef is skipped."""
    from bibr.config import Settings
    from bibr.enrich.references import enrich_references

    monkeypatch.setattr(Settings.resolver, "authoritative", True)

    ref = _make_ref(doi="10.1/abc", year=2020)
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.lookup_doi = mock.AsyncMock(return_value=None)  # clean miss

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert not ref.match
    crossref.works.assert_not_called()


async def test_authoritative_resolver_error_still_falls_through_to_crossref(monkeypatch):
    """Authoritative suppresses CrossRef on a clean MISS only — an actual resolver
    ERROR still falls through, preserving the safety net."""
    from bibr.config import Settings
    from bibr.enrich.references import enrich_references

    monkeypatch.setattr(Settings.resolver, "authoritative", True)

    ref = _make_ref(doi="10.1/abc", year=2020)
    crossref = _mock_crossref()
    crossref.works = mock.AsyncMock(
        return_value={
            "message": {
                "DOI": "10.1/abc",
                "title": ["A Great Paper on Testing Methods"],
                "issued": {"date-parts": [[2020]]},
            }
        }
    )
    resolver = mock.AsyncMock()
    resolver.lookup_doi = mock.AsyncMock(side_effect=RuntimeError("boom"))

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.CROSSREF in ref.match
    crossref.works.assert_called_once()


async def test_authoritative_resolver_search_error_slot_falls_through_to_crossref(monkeypatch):
    """A per-query search error surfaced as an exception in the prefetch slot is an
    ERROR, not a miss — even in authoritative mode it falls through to CrossRef."""
    from bibr.config import Settings
    from bibr.enrich.references import enrich_references

    monkeypatch.setattr(Settings.resolver, "authoritative", True)

    ref = _make_ref(doi=None, title="A Great Paper on Testing Methods")
    crossref = _mock_crossref()
    crossref.search = mock.AsyncMock(
        return_value={
            "message": {
                "items": [
                    {
                        "title": ["A Great Paper on Testing Methods"],
                        "DOI": "10.1/cr",
                        "issued": {"date-parts": [[2020]]},
                    }
                ]
            }
        }
    )
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[RuntimeError("boom")])  # slot error

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.CROSSREF in ref.match
    crossref.search.assert_called_once()


async def test_non_authoritative_resolver_miss_still_falls_through(monkeypatch):
    """Default (non-authoritative): a resolver miss falls through to CrossRef — the
    resolver is only a fast accelerator, not the sole source of truth."""
    from bibr.config import Settings
    from bibr.enrich.references import enrich_references

    monkeypatch.setattr(Settings.resolver, "authoritative", False)

    ref = _make_ref(doi=None, title="A Great Paper on Testing Methods")
    crossref = _mock_crossref()
    crossref.search = mock.AsyncMock(
        return_value={
            "message": {
                "items": [
                    {
                        "title": ["A Great Paper on Testing Methods"],
                        "DOI": "10.1/cr",
                        "issued": {"date-parts": [[2020]]},
                    }
                ]
            }
        }
    )
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])  # resolver miss

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.CROSSREF in ref.match
    crossref.search.assert_called_once()


async def test_resolver_per_reference_error_falls_through_to_crossref():
    """An unexpected error on the per-ref resolver path (here a DOI lookup) is caught and
    the reference falls through to CrossRef rather than failing enrichment."""
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi="10.1/abc", year=2020)
    crossref = _mock_crossref()
    crossref.works = mock.AsyncMock(
        return_value={
            "message": {
                "DOI": "10.1/abc",
                "title": ["A Great Paper on Testing Methods"],
                "issued": {"date-parts": [[2020]]},
            }
        }
    )
    resolver = mock.AsyncMock()
    resolver.lookup_doi = mock.AsyncMock(side_effect=RuntimeError("boom"))

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    assert MatchSource.OPENALEX not in ref.match
    assert MatchSource.CROSSREF in ref.match
    crossref.works.assert_called_once()


async def test_resolver_fallback_disabled_preserves_primary_call_count():
    from bibr.enrich.references import enrich_references

    ref = _make_ref()
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])

    await enrich_references(
        [ref],
        crossref_client=crossref,
        resolver_client=resolver,
        settings=_fallback_settings(fallback_sources=[]),
    )

    resolver.search_many.assert_awaited_once()
    assert not ref.match


async def test_primary_resolver_match_is_not_sent_to_fallback():
    from bibr.enrich.references import enrich_references

    ref = _make_ref()
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(
        return_value=[[_candidate(source="crossref", doi="10.1234/primary")]]
    )

    await enrich_references(
        [ref],
        crossref_client=crossref,
        resolver_client=resolver,
        settings=_fallback_settings(),
    )

    resolver.search_many.assert_awaited_once()
    assert MatchSource.CROSSREF in ref.match
    assert MatchSource.OPENALEX not in ref.match


async def test_primary_miss_uses_openalex_fallback_with_separate_concurrency():
    from bibr.enrich.references import enrich_references

    ref = _make_ref()
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])
    resolver.search = mock.AsyncMock(return_value=[_candidate(id="W123", doi="10.1234/fallback")])

    await enrich_references(
        [ref],
        crossref_client=crossref,
        resolver_client=resolver,
        settings=_fallback_settings(fallback_search_concurrency=3),
    )

    resolver.search_many.assert_awaited_once()
    resolver.search.assert_awaited_once()
    fallback_call = resolver.search.await_args
    assert fallback_call.kwargs["sources"] == ["openalex"]
    assert fallback_call.kwargs["raise_on_error"] is True
    assert MatchSource.OPENALEX in ref.match
    assert ref.match[MatchSource.OPENALEX].id == "10.1234/fallback"


async def test_fallback_searches_run_at_most_the_fallback_concurrency_at_once():
    from bibr.enrich.references import enrich_references

    refs = [_make_ref(bib_id=i, title=f"A distinct fallback title number {i}") for i in range(7)]
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(side_effect=lambda queries, **kw: [[] for _ in queries])
    in_flight = 0
    peak = 0

    async def search(title, year, limit, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return []

    resolver.search = mock.AsyncMock(side_effect=search)

    await enrich_references(
        refs,
        crossref_client=crossref,
        resolver_client=resolver,
        settings=_fallback_settings(fallback_search_concurrency=3),
    )

    assert resolver.search.await_count == 7
    assert peak == 3


async def test_fallback_removes_sources_already_used_by_primary():
    from bibr.enrich.references import enrich_references

    ref = _make_ref()
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])
    resolver.search = mock.AsyncMock(return_value=[_candidate(id="W123")])

    await enrich_references(
        [ref],
        crossref_client=crossref,
        resolver_client=resolver,
        settings=_fallback_settings(fallback_sources=["crossref", "openalex"]),
    )

    assert resolver.search.await_args.kwargs["sources"] == ["openalex"]


async def test_fallback_accepts_same_normalized_printed_doi():
    from bibr.enrich.references import enrich_references

    ref = _make_ref(doi="https://doi.org/10.1234/ABC.")
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.lookup_doi = mock.AsyncMock(return_value=None)
    resolver.search = mock.AsyncMock(return_value=[_candidate(id="W123", doi="10.1234/abc")])

    await enrich_references(
        [ref],
        crossref_client=crossref,
        resolver_client=resolver,
        settings=_fallback_settings(),
    )

    resolver.search.assert_awaited_once()
    assert resolver.search.await_args.kwargs["sources"] == ["openalex"]
    assert MatchSource.OPENALEX in ref.match


async def test_fallback_rejects_missing_or_conflicting_candidate_doi():
    from bibr.enrich.references import enrich_references

    refs = [
        _make_ref(bib_id=1, doi="10.1234/printed"),
        _make_ref(bib_id=2, doi="10.1234/printed"),
    ]
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.lookup_doi = mock.AsyncMock(return_value=None)
    resolver.search = mock.AsyncMock(
        side_effect=[
            [_candidate(id="W-missing")],
            [_candidate(id="W-conflict", doi="10.1234/different")],
        ]
    )

    await enrich_references(
        refs,
        crossref_client=crossref,
        resolver_client=resolver,
        settings=_fallback_settings(),
    )

    assert resolver.search.await_count == 2
    assert resolver.search.await_args.kwargs["sources"] == ["openalex"]
    assert all(not ref.match for ref in refs)


async def test_fallback_clean_miss_remains_unmatched_without_failure():
    from bibr.enrich.references import enrich_references

    ref = _make_ref()
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])
    resolver.search = mock.AsyncMock(return_value=[])

    report = await enrich_references(
        [ref],
        crossref_client=crossref,
        resolver_client=resolver,
        settings=_fallback_settings(),
    )

    resolver.search_many.assert_awaited_once()
    resolver.search.assert_awaited_once()
    assert not ref.match
    assert report.failed == 0
    assert report.details == ()


async def test_fallback_item_error_warns_without_marking_enrichment_partial():
    from bibr.enrich.references import enrich_references

    ref = _make_ref()
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])
    resolver.search = mock.AsyncMock(side_effect=RuntimeError("openalex down"))

    report = await enrich_references(
        [ref],
        crossref_client=crossref,
        resolver_client=resolver,
        settings=_fallback_settings(),
    )

    assert not ref.match
    assert report.failed == 0
    assert len(report.details) == 1
    assert report.details[0].code == WarningCode.RESOLVER_FALLBACK_FAILED
    assert "resolver fallback" in report.details[0].message
    assert "openalex down" in report.details[0].message


async def test_fallback_timeout_cancels_the_searches_still_outstanding():
    from bibr.enrich.references import enrich_references

    refs = [_make_ref(bib_id=1), _make_ref(bib_id=2, title="Another Excellent Testing Paper")]
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    fallback_started = asyncio.Event()
    fallback_cancelled = asyncio.Event()

    async def search(title, year, limit, **kwargs):
        fallback_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            fallback_cancelled.set()
            raise

    resolver.search_many = mock.AsyncMock(side_effect=lambda queries, **kw: [[] for _ in queries])
    resolver.search = mock.AsyncMock(side_effect=search)

    report = await enrich_references(
        refs,
        crossref_client=crossref,
        resolver_client=resolver,
        settings=_fallback_settings(fallback_timeout=0.01),
    )

    assert fallback_started.is_set()
    assert fallback_cancelled.is_set()
    assert all(not ref.match for ref in refs)
    assert report.failed == 0
    assert len(report.details) == 1
    assert report.details[0].code == WarningCode.RESOLVER_FALLBACK_TIMEOUT
    assert "fallback timed out" in report.details[0].message


async def test_authoritative_resolver_skips_the_crossref_doi_prefetch(monkeypatch):
    """An authoritative resolver answers from CrossRef's own corpus and its clean
    miss skips CrossRef, so bulk-prefetching CrossRef DOIs would buy nothing."""
    from bibr.config import Settings
    from bibr.enrich.references import enrich_references

    monkeypatch.setattr(Settings.resolver, "authoritative", True)

    ref = _make_ref(doi="10.1/abc", year=2020)
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.lookup_doi = mock.AsyncMock(return_value=_candidate(doi="10.1/abc", year=2020))

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    crossref.prefetch_works_by_doi.assert_not_called()


async def test_non_authoritative_resolver_keeps_the_crossref_doi_prefetch(monkeypatch):
    """A non-authoritative resolver falls through to CrossRef on every miss, so
    the batched DOI prefetch still pays for itself."""
    from bibr.config import Settings
    from bibr.enrich.references import enrich_references

    monkeypatch.setattr(Settings.resolver, "authoritative", False)

    ref = _make_ref(doi="10.1/abc", year=2020)
    crossref = _mock_crossref()
    resolver = mock.AsyncMock()
    resolver.lookup_doi = mock.AsyncMock(return_value=None)

    await enrich_references([ref], crossref_client=crossref, resolver_client=resolver)

    crossref.prefetch_works_by_doi.assert_awaited_once_with(["10.1/abc"])
