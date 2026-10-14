"""How each reference's registry lookup ended (extraction.enrichment.lookups),
from recorded Crossref and doi.org responses."""

from __future__ import annotations

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

    async def recorded(dois, *, user_agent=None, http=None):  # noqa: ARG001
        return await real(dois, user_agent=user_agent, http=fake)

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


async def test_book_doi_is_not_rejected_for_its_edition_year(doi_org):
    """Cohen's 1988 book is registered as its 2013 edition; a year apart is no
    evidence against a book's printed DOI."""
    ref = _ref(
        1, "Statistical power analysis for the behavioral sciences", doi=COHEN_DOI, year=1988
    )
    assert await _enrich([ref]) == {1: "doi_hit"}


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
