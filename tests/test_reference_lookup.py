"""How each reference's registry lookup ended (extraction.enrichment.lookups),
from recorded Crossref and doi.org responses."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from bibr.config import GlobalSettings
from bibr.enrich import lookup as lookup_mod
from bibr.enrich.lookup import ReferenceLookup, recorded_lookup, registration_agencies
from bibr.paper import PaperReference
from tests.crossref_recorded import (
    CARNEY_DOI,
    COHEN_DOI,
    DATACITE_DOI,
    OSC_DOI,
    UNREGISTERED_DOI,
    WAKEFIELD_DOI,
    RecordedCrossref,
    RecordedDoiOrg,
)


def _ref(bib_id, title, *, doi=None, year=None, authors=None, volume=None, container=None):
    return PaperReference(
        bib_id=bib_id,
        title=title,
        first_page=None,
        volume=volume,
        authors=authors,
        year=year,
        container=container,
        doi=doi,
    )


def _settings(**crossref) -> GlobalSettings:
    return GlobalSettings(crossref={"doi_agency_lookup": True, **crossref})


@pytest.fixture
def doi_org(monkeypatch):
    """doi.org answers from the recording."""
    fake = RecordedDoiOrg()
    real = lookup_mod.registration_agencies

    async def recorded(dois, *, http=None):  # noqa: ARG001
        return await real(dois, http=fake)

    monkeypatch.setattr(lookup_mod, "registration_agencies", recorded)
    return fake


async def _enrich(refs, client=None, settings=None):
    from bibr.enrich.references import enrich_references

    await enrich_references(
        refs, crossref_client=client or RecordedCrossref(), settings=settings or _settings()
    )
    return {ref.bib_id: recorded_lookup(ref).outcome(bool(ref.match)) for ref in refs}


async def test_outcomes_from_recorded_responses(doi_org):
    refs = [
        _ref(1, "Power posing", doi=CARNEY_DOI, year=2010),
        # Crossref deposits the subtitle apart, and a search is scored on the
        # title alone.
        _ref(2, "Power posing", year=2010, authors="Carney, D. R., Cuddy, A. J. C., & Yap, A. J."),
        _ref(3, "A title nobody ever wrote about sourdough", year=2021, authors="Nobody, A."),
        _ref(4, "Some dataset", doi=DATACITE_DOI, year=2019),
        _ref(5, "Power posing", doi=UNREGISTERED_DOI, year=2010),
        # The OSC DOI printed for Carney et al.: a 2015 record for a 2010 work.
        _ref(6, "Power posing", doi=OSC_DOI, year=2010),
        _ref(7, "Short"),
    ]
    outcomes = await _enrich(refs)

    assert outcomes == {
        1: "doi_hit",
        2: "search_hit",
        3: "clean_miss",
        4: "doi_404",
        5: "doi_404",
        6: "doi_rejected",
        7: "not_attempted",
    }
    assert recorded_lookup(refs[3]).registration_agency == "datacite"
    assert recorded_lookup(refs[4]).registration_agency == "none"
    # One batched doi.org request for both unknown DOIs.
    (url,) = doi_org.urls
    assert url.startswith("https://doi.org/ra/")
    # The rejected DOI's record is kept, so the check can say what differs.
    rejected = recorded_lookup(refs[5])
    assert (rejected.rejected_year, rejected.rejected_type) == (2015, "journal_article")
    assert rejected.rejected_title == "Estimating the reproducibility of psychological science"


async def test_a_book_doi_of_another_years_edition_is_rejected_and_described(doi_org):
    """Cohen's 1988 book is registered as its 2013 edition. Enrichment accepts
    a DOI's record only within a year of the printed year, whatever its type;
    the record stays on the lookup for the check, which calls the year
    difference of a book minor."""
    ref = _ref(
        1, "Statistical power analysis for the behavioral sciences", doi=COHEN_DOI, year=1988
    )
    assert await _enrich([ref]) == {1: "doi_rejected"}
    assert not ref.match
    lookup = recorded_lookup(ref)
    assert (lookup.rejected_year, lookup.rejected_type) == (2013, "book")


def test_doi_year_guard():
    """A DOI's record more than a year from the printed year is not accepted,
    whatever the gap's direction or the record's type."""
    from bibr.enrich.references import _doi_year_conflicts

    assert not _doi_year_conflicts(2019, 2018)
    assert not _doi_year_conflicts(2018, 2019)
    assert _doi_year_conflicts(2019, 2017)
    assert _doi_year_conflicts(2017, 2019)
    assert _doi_year_conflicts(1988, 2013)
    assert not _doi_year_conflicts(None, 2013)
    assert not _doi_year_conflicts(1988, None)


async def test_retracted_work_is_found_by_its_doi(doi_org):
    ref = _ref(1, "Ileal-lymphoid-nodular hyperplasia", doi=WAKEFIELD_DOI, year=1998)
    assert await _enrich([ref]) == {1: "doi_hit"}
    (match,) = ref.match.values()
    assert {u.type for u in match.updated_by} == {"retraction", "correction"}


async def test_a_failed_lookup_is_an_error_not_a_miss(doi_org):
    class Broken(RecordedCrossref):
        async def search(self, query, limit=3, **_kw):
            raise RuntimeError("connection reset")

    ref = _ref(1, "A title long enough to search for", year=2020)
    assert await _enrich([ref], Broken()) == {1: "error"}


async def test_agency_lookup_can_be_turned_off(doi_org):
    ref = _ref(1, "Some dataset", doi=DATACITE_DOI, year=2019)
    outcomes = await _enrich([ref], settings=_settings(doi_agency_lookup=False))
    assert outcomes == {1: "doi_404"}
    assert recorded_lookup(ref).registration_agency is None
    assert doi_org.urls == []


async def test_timed_out_references_are_marked_by_the_enricher(doi_org):
    from bibr.pipeline.enricher import CrossrefEnricher
    from bibr.pipeline.state import FileState
    from tests.export.conftest import _demo_paper

    paper = _demo_paper(with_refs=True)
    paper.metadata.doi = ""
    ref = paper.metadata.references[0]
    ref.doi = CARNEY_DOI
    ref.match = {}
    fs = FileState(path=Path("p.pdf"))
    fs.paper = paper
    client = RecordedCrossref(hang={CARNEY_DOI})
    with patch("bibr.clients.crossref.get_client", return_value=client):
        await CrossrefEnricher(timeout=0.05, settings=_settings()).enrich(fs)

    assert paper.metadata.enrichment_complete is False
    assert recorded_lookup(ref).outcome(bool(ref.match)) == "timed_out"


async def test_a_hanging_doi_org_leaves_the_enrichment_complete(doi_org, monkeypatch):
    """doi.org only refines a doi_404: when it does not answer, the paper's
    Crossref lookups that all finished still make a complete enrichment."""
    from bibr.pipeline.enricher import CrossrefEnricher, EnrichmentStatus
    from bibr.pipeline.state import FileState
    from tests.export.conftest import _demo_paper

    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(lookup_mod, "registration_agencies", hang)
    paper = _demo_paper(with_refs=True)
    paper.metadata.doi = ""
    first = paper.metadata.references[0]
    first.doi, first.year, first.match = CARNEY_DOI, 2010, {}
    second = _ref(2, "Some dataset", doi=DATACITE_DOI, year=2019)
    paper.metadata.references = [first, second]
    fs = FileState(path=Path("p.pdf"))
    fs.paper = paper
    with patch("bibr.clients.crossref.get_client", return_value=RecordedCrossref()):
        outcome = await CrossrefEnricher(timeout=0.8, settings=_settings()).enrich(fs)

    assert outcome.status is EnrichmentStatus.COMPLETE
    assert paper.metadata.enrichment_complete is True
    assert fs.warnings == []
    assert recorded_lookup(first).outcome(bool(first.match)) == "doi_hit"
    assert recorded_lookup(second).outcome(False) == "doi_404"
    assert recorded_lookup(second).registration_agency is None


async def test_the_agency_lookup_has_a_budget_of_its_own(doi_org, monkeypatch):
    """Called without a deadline, a doi.org that never answers costs its own
    few seconds, not more."""
    from bibr.enrich import references as references_mod

    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(lookup_mod, "registration_agencies", hang)
    monkeypatch.setattr(references_mod, "_AGENCY_BUDGET_SECONDS", 0.05)
    ref = _ref(1, "Some dataset", doi=DATACITE_DOI, year=2019)
    assert await asyncio.wait_for(_enrich([ref]), timeout=2) == {1: "doi_404"}
    assert recorded_lookup(ref).registration_agency is None


async def test_no_time_left_skips_the_agency_lookup(doi_org):
    from bibr.enrich.references import enrich_references

    ref = _ref(1, "Some dataset", doi=DATACITE_DOI, year=2019)
    deadline = asyncio.get_running_loop().time() + 0.1
    await enrich_references(
        [ref], crossref_client=RecordedCrossref(), settings=_settings(), deadline=deadline
    )
    assert doi_org.urls == []
    assert recorded_lookup(ref).registration_agency is None


async def test_doi_org_is_not_sent_the_contact_address(monkeypatch):
    """The Crossref contact email is for Crossref: doi.org gets a User-Agent
    that names bibr and nothing else."""
    import httpx

    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"DOI": CARNEY_DOI, "RA": "Crossref"}])

    real_client = httpx.AsyncClient

    def mocked(**kwargs):
        return real_client(transport=httpx.MockTransport(answer), **kwargs)

    monkeypatch.setenv("CROSSREF_API_EMAIL", "ops@example.org")
    monkeypatch.setattr(httpx, "AsyncClient", mocked)
    assert await registration_agencies([CARNEY_DOI]) == {CARNEY_DOI: "crossref"}
    (request,) = seen
    user_agent = request.headers["User-Agent"]
    assert user_agent.startswith("bibr/")
    assert "@" not in user_agent and "mailto" not in user_agent


def _resolver_settings(**resolver) -> GlobalSettings:
    from bibr.config import ResolverOptions

    values = {
        "sources": ["crossref"],
        "fallback_sources": ["openalex"],
        "raw_search": False,
        "authoritative": False,
        **resolver,
    }
    return GlobalSettings(
        _env_file=None,
        crossref={"doi_agency_lookup": False},
        resolver=ResolverOptions(_env_file=None, **values),
    )


def _resolver(search):
    from unittest import mock

    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])
    resolver.lookup_doi = mock.AsyncMock(return_value=None)
    resolver.search = search
    return resolver


async def _hang(*_args, **_kwargs):
    await asyncio.Event().wait()


async def test_a_budget_running_out_in_the_resolver_fallback_is_a_timeout(doi_org):
    """Crossref answered (no match), but the resolver fallback still owed an
    answer when the paper's budget ran out: not every lookup answered, so the
    reference timed out rather than missed cleanly."""
    from bibr.enrich.lookup import finish_unfinished
    from bibr.enrich.references import enrich_references

    ref = _ref(1, "A title nobody ever wrote about sourdough", year=2021, authors="Nobody, A.")
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            enrich_references(
                [ref],
                crossref_client=RecordedCrossref(),
                resolver_client=_resolver(_hang),
                settings=_resolver_settings(),
            ),
            timeout=0.2,
        )
    finish_unfinished([ref], timed_out=True)
    assert recorded_lookup(ref).outcome(False) == "timed_out"


async def test_a_reference_no_later_pass_searches_is_finished_after_crossref(doi_org):
    """A reference no resolver pass applies to (no usable title) is finished
    once Crossref answered, so a budget running out later leaves it alone."""
    from bibr.enrich.lookup import finish_unfinished
    from bibr.enrich.references import enrich_references

    searched = _ref(1, "A title nobody ever wrote about sourdough", year=2021)
    untitled = _ref(2, "Short", doi=DATACITE_DOI, year=2019)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            enrich_references(
                [searched, untitled],
                crossref_client=RecordedCrossref(),
                resolver_client=_resolver(_hang),
                settings=_resolver_settings(),
            ),
            timeout=0.2,
        )
    finish_unfinished([searched, untitled], timed_out=True)
    assert recorded_lookup(searched).outcome(False) == "timed_out"
    assert recorded_lookup(untitled).outcome(False) == "doi_404"


async def test_the_resolver_fallbacks_own_timeout_and_errors_are_not_clean_misses(doi_org):
    from bibr.enrich.references import enrich_references

    async def search(title, *_args, **_kwargs):
        if "error" in title:
            raise RuntimeError("resolver down")
        await asyncio.Event().wait()

    hung = _ref(1, "A title nobody ever wrote about sourdough", year=2021)
    failed = _ref(2, "A title that meets a resolver error", year=2021)
    await enrich_references(
        [hung, failed],
        crossref_client=RecordedCrossref(),
        resolver_client=_resolver(search),
        settings=_resolver_settings(fallback_timeout=0.1),
    )
    assert recorded_lookup(hung).outcome(False) == "timed_out"
    assert recorded_lookup(failed).outcome(False) == "error"


async def test_a_clean_resolver_fallback_miss_is_a_clean_miss(doi_org):
    from bibr.enrich.references import enrich_references

    async def search(*_args, **_kwargs):
        return []

    ref = _ref(1, "A title nobody ever wrote about sourdough", year=2021)
    await enrich_references(
        [ref],
        crossref_client=RecordedCrossref(),
        resolver_client=_resolver(search),
        settings=_resolver_settings(),
    )
    assert recorded_lookup(ref).outcome(False) == "clean_miss"
    assert recorded_lookup(ref).finished


def test_outcome_precedence():
    assert ReferenceLookup(hit="doi").outcome(True) == "doi_hit"
    assert ReferenceLookup(queried=True).outcome(True) == "search_hit"
    assert ReferenceLookup(doi_status="not_found", errored=True).outcome(False) == "error"
    assert ReferenceLookup(doi_status="rejected", errored=True).outcome(False) == "doi_rejected"
    assert ReferenceLookup(queried=True, timed_out=True).outcome(False) == "timed_out"
    assert ReferenceLookup().outcome(False) == "not_attempted"


async def test_registration_agencies_skips_unaskable_dois():
    fake = RecordedDoiOrg()
    agencies = await registration_agencies(
        [CARNEY_DOI, CARNEY_DOI.upper(), "10.1000/a,b", DATACITE_DOI], http=fake
    )
    assert agencies == {CARNEY_DOI: "crossref", DATACITE_DOI: "datacite"}
    (url,) = fake.urls
    assert "a,b" not in url


async def test_registration_agencies_never_raises():
    class Down:
        async def get(self, url):
            raise OSError("unreachable")

    assert await registration_agencies([CARNEY_DOI], http=Down()) == {}


def test_a_rejected_dois_record_is_exported():
    from bibr.export.json_export import _export_paper_payload
    from bibr.export.models import PaperExport
    from tests.export.conftest import _demo_paper

    paper = _demo_paper(with_refs=True)
    ref = paper.metadata.references[0]
    ref.match = {}
    ref._lookup = ReferenceLookup(queried=True, finished=True)
    ref._lookup.reject(2013, "Statistical Power Analysis for the Behavioral Sciences", "book")
    payload = _export_paper_payload(paper)
    PaperExport.model_validate(payload)
    assert payload["extraction"]["enrichment"]["lookups"] == [
        {
            "bib_id": 1,
            "outcome": "doi_rejected",
            "record_year": 2013,
            "record_title": "Statistical Power Analysis for the Behavioral Sciences",
            "record_bib_type": "book",
        }
    ]


def test_lookups_are_exported(demo_paper_for_lookups):
    from bibr.export.json_export import _export_paper_payload
    from bibr.export.models import PaperExport

    payload = _export_paper_payload(demo_paper_for_lookups)
    PaperExport.model_validate(payload)
    assert payload["extraction"]["enrichment"]["lookups"] == [
        {"bib_id": 1, "outcome": "doi_404", "registration_agency": "datacite"}
    ]


@pytest.fixture
def demo_paper_for_lookups():
    from tests.export.conftest import _demo_paper

    paper = _demo_paper(with_refs=True)
    ref = paper.metadata.references[0]
    ref.match = {}
    ref._lookup = ReferenceLookup(
        queried=True, doi_status="not_found", finished=True, registration_agency="datacite"
    )
    return paper


def test_exports_without_recorded_lookups_omit_them():
    from bibr.export.json_export import _export_paper_payload
    from tests.export.conftest import _demo_paper

    payload = _export_paper_payload(_demo_paper(with_refs=True))
    assert "lookups" not in payload["extraction"]["enrichment"]
