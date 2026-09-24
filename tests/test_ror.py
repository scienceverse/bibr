"""ROR matching of affiliation strings and funder names."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from bibr.clients.ror import RorClient, chosen_organization
from bibr.config import GlobalSettings
from bibr.enrich.organizations import enrich_organizations
from bibr.models import FundingEntry, PaperAuthor, PaperMetadata


def _item(ror_id: str, *, chosen: bool, score: float = 1.0, fundref: str | None = None) -> dict:
    org = {
        "id": f"https://ror.org/{ror_id}",
        "names": [
            {"value": "UofG", "types": ["acronym"]},
            {"value": f"Org {ror_id}", "types": ["ror_display", "label"]},
        ],
        "locations": [{"geonames_details": {"country_code": "gb", "name": "Glasgow"}}],
        "external_ids": (
            [{"type": "fundref", "all": [fundref, "999"], "preferred": fundref}] if fundref else []
        ),
    }
    return {"chosen": chosen, "score": score, "matching_type": "SINGLE SEARCH", "organization": org}


def test_only_the_chosen_result_is_taken():
    payload = {
        "items": [
            _item("0aaaaaa11", chosen=False, score=0.99),
            _item("00vtgdb53", chosen=True, fundref="501100000853"),
        ]
    }
    match = chosen_organization(payload)
    assert match is not None
    assert match.service_id == "https://ror.org/00vtgdb53"
    assert match.name == "Org 00vtgdb53"
    assert match.country_code == "GB"
    assert match.funder_doi == "10.13039/501100000853"
    assert chosen_organization({"items": [_item("0aaaaaa11", chosen=False)]}) is None
    assert chosen_organization({"items": []}) is None
    assert chosen_organization("garbage") is None


def _settings(**ror) -> GlobalSettings:
    settings = GlobalSettings()
    for key, value in ror.items():
        setattr(settings.ror, key, value)
    return settings


def _client(handler, **ror) -> RorClient:
    transport = httpx.MockTransport(handler)
    return RorClient(settings=_settings(**ror), client=httpx.AsyncClient(transport=transport))


def test_request_uses_single_search_and_client_id_and_caches():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"items": [_item("00vtgdb53", chosen=True)]})

    client = _client(handler, client_id="abc123")

    async def run():
        first = await client.match("University of  Glasgow")
        second = await client.match("university of glasgow")  # same string after normalizing
        return first, second

    first, second = asyncio.run(run())
    assert first == second and first is not None
    assert len(seen) == 1
    assert seen[0].url.params["affiliation"] == "University of Glasgow"
    assert "single_search" in seen[0].url.params
    assert seen[0].headers["Client-Id"] == "abc123"


def test_misses_are_cached_and_short_strings_skipped():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"items": [_item("0aaaaaa11", chosen=False)]})

    client = _client(handler)

    async def run():
        return [
            await client.match("Nowhere Institute"),
            await client.match("Nowhere Institute"),
            await client.match("ab"),
        ]

    assert asyncio.run(run()) == [None, None, None]
    assert calls == 1


def test_a_429_stops_lookups_for_the_retry_interval():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "120"})

    client = _client(handler)

    async def run():
        return [await client.match("First University"), await client.match("Second University")]

    assert asyncio.run(run()) == [None, None]
    assert calls == 1 and client.blocked


def test_errors_degrade_to_no_match():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert asyncio.run(_client(handler).match("Some University")) is None


def test_default_budget_follows_rors_published_limits():
    assert RorClient(settings=_settings())._limiter.max_requests == 50
    assert RorClient(settings=_settings(client_id="x"))._limiter.max_requests == 2000
    assert RorClient(settings=_settings(requests_per_5min=7))._limiter.max_requests == 7


def _metadata() -> PaperMetadata:
    return PaperMetadata(
        doi="",
        title="A paper",
        authors=[
            PaperAuthor(author_id=1, given="A", family="B", affiliation="Dept X, Uni One; Uni Two"),
            PaperAuthor(author_id=2, given="C", family="D", affiliation="Uni Two"),
        ],
        funding=[
            FundingEntry(funder="Wellcome Trust", award_ids=[]),
            FundingEntry(funder="Unknown Fund", award_ids=[]),
        ],
    )


def test_enrichment_stores_matches_keyed_by_the_printed_string():
    def handler(request: httpx.Request) -> httpx.Response:
        text = request.url.params["affiliation"]
        known = {
            "Dept X, Uni One": "0uni00011",
            "Uni Two": "0uni00022",
            "Wellcome Trust": "029chgv08",
        }
        if text in known:
            return httpx.Response(200, json={"items": [_item(known[text], chosen=True)]})
        return httpx.Response(200, json={"items": []})

    metadata = _metadata()
    report = asyncio.run(enrich_organizations(metadata, _client(handler), timeout=10))

    assert report.attempted == 4 and report.matched == 3 and not report.timed_out
    assert set(metadata.affiliation_match) == {"Dept X, Uni One", "Uni Two"}
    assert metadata.funder_match["Wellcome Trust"].service_id == "https://ror.org/029chgv08"
    assert "Unknown Fund" not in metadata.funder_match


def test_enrichment_keeps_what_it_matched_before_the_timeout():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["affiliation"] == "Uni Two":
            await asyncio.sleep(5)
        return httpx.Response(200, json={"items": [_item("0uni00011", chosen=True)]})

    metadata = _metadata()
    report = asyncio.run(enrich_organizations(metadata, _client(handler), timeout=0.5))
    assert report.timed_out
    assert "Dept X, Uni One" in metadata.affiliation_match
    assert "Uni Two" not in metadata.affiliation_match


def test_enricher_never_reports_partial(monkeypatch):
    """A timeout leaves strings unmatched with a warning; it must not hold the
    export behind the enrichment-pending gate."""
    from types import SimpleNamespace

    from bibr.enrich import organizations
    from bibr.enrich.organizations import OrganizationReport
    from bibr.pipeline.enricher import EnrichmentStatus, RorEnricher
    from bibr.processing_warnings import WarningCode

    async def fake(metadata, client, *, timeout):
        return OrganizationReport(attempted=5, matched=2, timed_out=True)

    monkeypatch.setattr(organizations, "enrich_organizations", fake)
    fs = SimpleNamespace(paper=SimpleNamespace(metadata=_metadata()))
    outcome = asyncio.run(RorEnricher(settings=_settings()).enrich(fs))
    assert outcome.status is EnrichmentStatus.COMPLETE
    assert [w.code for w in outcome.warnings] == [WarningCode.ROR_MATCHING_TIMEOUT]
    assert "2/5" in outcome.warnings[0].message


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://ror.org/00vtgdb53", "https://ror.org/00vtgdb53"),
        ("ror.org/00VTGDB53/", "https://ror.org/00vtgdb53"),
        ("00vtgdb53", None),
        ("https://example.org/00vtgdb53", None),
        (None, None),
    ],
)
def test_canonical_ror(value, expected):
    from bibr.enrich.schemas import canonical_ror

    assert canonical_ror(value) == expected
