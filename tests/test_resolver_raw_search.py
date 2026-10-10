"""Resolver search with the whole reference string, for references the title passes missed."""

import asyncio
from unittest import mock

from bibr.clients.crossref import CrossrefClient
from bibr.models import PaperReference
from bibr.paper import MatchSource

RAW = (
    "12. Kopf M, Baumann H, Freer G, et al. Impaired immune and acute-phase responses in "
    "interleukin-6-deficient mice. Nature. 1994;368(6469):339-342. https://doi.org/x"
)


def _crossref():
    client = mock.MagicMock(spec_set=CrossrefClient)
    client.enrich_semaphore = asyncio.Semaphore(100)
    return client


def _ref(**kw):
    # A parse that lost the title: the title passes have nothing to query.
    base = {
        "bib_id": 1,
        "title": "Nature",
        "first_page": None,
        "volume": None,
        "authors": "Kopf M, Baumann H, Freer G, et al.",
        "year": 1994,
        "container": None,
    }
    base.update(kw)
    return PaperReference(**base)


def _candidate(**kw):
    base = {
        "title": "Impaired immune and acute-phase responses in interleukin-6-deficient mice",
        "source": "crossref",
        "doi": "10.1038/368339a0",
        "year": 1994,
        "authors": [{"given": "Manfred", "family": "Kopf"}],
        "volume": "368",
        "first_page": "339",
    }
    base.update(kw)
    return base


def _settings(**overrides):
    from bibr.config import GlobalSettings, ResolverOptions

    values = {"sources": ["crossref"], "raw_search": True, "authoritative": True}
    values.update(overrides)
    return GlobalSettings(_env_file=None, resolver=ResolverOptions(_env_file=None, **values))


async def _enrich(ref, candidates, **settings):
    from bibr.enrich.references import enrich_references

    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])
    resolver.search = mock.AsyncMock(return_value=candidates)
    await enrich_references(
        [ref],
        crossref_client=_crossref(),
        resolver_client=resolver,
        settings=_settings(**settings),
        raw_strings={ref.bib_id: RAW},
    )
    return resolver


async def test_a_lost_title_is_found_through_the_reference_string():
    ref = _ref()
    resolver = await _enrich(ref, [_candidate()])

    query = resolver.search.await_args.args[0]
    assert query.startswith("Kopf M")  # entry number dropped
    assert "doi.org" not in query
    match = ref.match[MatchSource.CROSSREF]
    assert match.doi == "10.1038/368339a0"
    assert match.score >= 90


async def test_off_by_default():
    ref = _ref()
    resolver = await _enrich(ref, [_candidate()], raw_search=False)
    resolver.search.assert_not_called()
    assert not ref.match


async def test_a_title_the_string_does_not_print_is_rejected():
    ref = _ref()
    await _enrich(ref, [_candidate(title="Interleukin-6 signalling in the healthy liver")])
    assert not ref.match


async def test_a_year_the_string_does_not_print_is_rejected():
    ref = _ref()
    await _enrich(ref, [_candidate(year=2004)])
    assert not ref.match


async def test_an_unpinned_candidate_is_rejected():
    # Neither the first author nor volume + first page is printed.
    ref = _ref()
    await _enrich(
        ref,
        [_candidate(authors=[{"family": "Smith"}], volume="12", first_page="1")],
    )
    assert not ref.match


async def test_two_works_printed_equally_well_take_neither():
    ref = _ref()
    await _enrich(ref, [_candidate(), _candidate(doi="10.9999/reprint")])
    assert not ref.match


async def test_the_same_work_twice_is_not_ambiguous():
    ref = _ref()
    await _enrich(ref, [_candidate(), _candidate(source="openalex")])
    assert ref.match[MatchSource.CROSSREF].doi == "10.1038/368339a0"


def test_a_generic_title_is_never_accepted_from_the_string():
    from bibr.enrich.references import _raw_candidate_score, _TitleCandidate

    cand = _TitleCandidate.from_resolver(_candidate(title="Introduction"))
    assert _raw_candidate_score("Kopf M. Introduction. Nature 1994;368:339.", cand) is None


async def test_the_enricher_hands_over_each_reference_string():
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from bibr.enrich.references import EnrichmentReport
    from bibr.pipeline.enricher import CrossrefEnricher
    from bibr.pipeline.state import FileState

    fs = FileState(path=Path("p.pdf"))
    fs.paper = MagicMock()
    fs.paper.metadata = MagicMock(references=[_ref(text_id=7), _ref(bib_id=2, text_id=99)], doi="")
    fs.paper.contents = SimpleNamespace(
        sentences=[SimpleNamespace(text_id=7, text=RAW), SimpleNamespace(text_id=8, text="x")]
    )
    seen = {}

    async def capture(_refs, **kwargs):
        seen.update(kwargs)
        return EnrichmentReport(attempted=2)

    with (
        patch("bibr.enrich.references.enrich_references", capture),
        patch("bibr.pipeline.enrich_prefetch.take_prefetch_handle", return_value=None),
    ):
        await CrossrefEnricher(settings=_settings()).enrich(fs)

    assert seen["raw_strings"] == {1: RAW}
