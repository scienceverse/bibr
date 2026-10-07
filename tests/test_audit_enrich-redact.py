"""Audit regressions: enrichment warning redaction, ROR robustness."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest

from bibr.clients.crossref import CrossrefClient
from bibr.clients.ror import RorClient, chosen_organization
from bibr.config import GlobalSettings, ResolverOptions
from bibr.models import PaperAuthor, PaperMetadata, PaperReference
from bibr.processing_warnings import WarningCode

# --- enrichment warnings never quote the request URL (report 2.3) ------------

_SECRET_URL = "https://svc:S3cretPass@resolver.internal.corp:8080"
_LEAKS = ("S3cretPass", "svc:", "resolver.internal.corp")


def _status_error(status: int = 503) -> httpx.HTTPStatusError:
    response = httpx.Response(status, request=httpx.Request("POST", f"{_SECRET_URL}/search"))
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise AssertionError("not an error status")


def _ref() -> PaperReference:
    return PaperReference(
        bib_id=1,
        title="A sufficiently long reference title",
        first_page=None,
        volume=None,
        authors="Smith, J.",
        year=2020,
        container=None,
    )


def _crossref():
    client = mock.MagicMock(spec_set=CrossrefClient)
    client.enrich_semaphore = asyncio.Semaphore(4)
    client.works = mock.AsyncMock(side_effect=RuntimeError("no"))
    client.search = mock.AsyncMock(return_value={"message": {"items": []}})
    client.prefetch_works_by_doi = mock.AsyncMock(return_value=0)
    return client


def _settings(**resolver) -> GlobalSettings:
    settings = GlobalSettings(
        _env_file=None,
        resolver=ResolverOptions(_env_file=None, **resolver),
    )
    settings.crossref.bulk_doi_lookup = False
    return settings


def _assert_no_leak(messages: list[str]) -> None:
    assert messages
    for message in messages:
        for leak in _LEAKS:
            assert leak not in message, message


async def test_resolver_http_error_warning_names_the_status_not_the_url():
    """The reviewer's repro: a 503 from a resolver whose URL carries user-info
    exported the password in ``ENRICHMENT_LOOKUP_FAILED``."""
    from bibr.clients.resolver import ResolverClient
    from bibr.enrich.references import enrich_references

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(503, json={})

    settings = _settings(url=_SECRET_URL, enrich=True, authoritative=True)
    http = httpx.AsyncClient(base_url=_SECRET_URL, transport=httpx.MockTransport(handler))
    resolver = ResolverClient(_SECRET_URL, client=http)
    report = await enrich_references(
        [_ref()], crossref_client=_crossref(), resolver_client=resolver, settings=settings
    )
    await http.aclose()

    messages = [d.message for d in report.details]
    _assert_no_leak(messages)
    assert "bib_id=1 resolver failed: HTTPStatusError: HTTP 503 Service Unavailable" in messages


async def test_resolver_fallback_warning_names_the_status_not_the_url():
    from bibr.enrich.references import enrich_references

    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])
    resolver.search = mock.AsyncMock(side_effect=_status_error())
    settings = _settings(sources=["crossref"], fallback_sources=["openalex"], authoritative=True)

    report = await enrich_references(
        [_ref()], crossref_client=_crossref(), resolver_client=resolver, settings=settings
    )

    fallback = [d for d in report.details if d.code == WarningCode.RESOLVER_FALLBACK_FAILED]
    assert [d.message for d in fallback] == [
        "bib_id=1 resolver fallback failed: HTTPStatusError: HTTP 503 Service Unavailable"
    ]


async def test_whole_resolver_fallback_failure_warning_drops_urls(monkeypatch):
    from bibr.enrich import references

    async def boom(*args, **kwargs):
        raise httpx.ConnectError(f"connect to {_SECRET_URL}/search failed\nretrying")

    monkeypatch.setattr(references, "_enrich_resolver_fallback", boom)
    resolver = mock.AsyncMock()
    resolver.search_many = mock.AsyncMock(return_value=[[]])
    settings = _settings(sources=["crossref"], fallback_sources=["openalex"])

    report = await references.enrich_references(
        [_ref()], crossref_client=_crossref(), resolver_client=resolver, settings=settings
    )

    fallback = [d for d in report.details if d.code == WarningCode.RESOLVER_FALLBACK_FAILED]
    assert [d.message for d in fallback] == [
        "resolver fallback failed for 1 refs: ConnectError: connect to <url> failed retrying"
    ]


def test_terminal_failure_detail_is_redacted_and_single_line():
    from bibr.enrich.references import ResolutionStats, _record_terminal_failure

    stats = ResolutionStats()
    _record_terminal_failure(stats, _ref(), "DOI lookup", _status_error(500))
    _record_terminal_failure(
        stats, _ref(), "DOI lookup", RuntimeError(f"boom\n  at {_SECRET_URL}/works")
    )
    assert [d.message for d in stats.failure_details] == [
        "bib_id=1 DOI lookup failed: HTTPStatusError: HTTP 500 Internal Server Error",
        "bib_id=1 DOI lookup failed: RuntimeError: boom at <url>",
    ]


# --- a malformed ROR answer is no match, never a failed paper (report 5.4) ----

_ROR_ID = "https://ror.org/00vtgdb53"


def _ror_client(handler) -> RorClient:
    transport = httpx.MockTransport(handler)
    return RorClient(settings=GlobalSettings(), client=httpx.AsyncClient(transport=transport))


def _chosen(org) -> dict:
    return {"items": [{"chosen": True, "score": 1.0, "organization": org}]}


@pytest.mark.parametrize("org", ["not-an-object", ["list"], 7, True])
def test_a_non_object_organization_is_no_match(org):
    assert chosen_organization(_chosen(org)) is None


@pytest.mark.parametrize(
    "org",
    [
        {"id": _ROR_ID, "names": "Uni"},
        {"id": _ROR_ID, "names": [{"types": 3, "value": "Uni"}]},
        {"id": _ROR_ID, "names": [{"types": ["ror_display"], "value": 12}]},
        {"id": _ROR_ID, "name": ["v1 name"]},
        {"id": _ROR_ID, "locations": 5},
        {"id": _ROR_ID, "external_ids": 5},
        {"id": _ROR_ID, "external_ids": [{"type": "fundref", "all": 7}]},
    ],
)
def test_malformed_organization_fields_count_as_absent(org):
    match = chosen_organization(_chosen(org))
    assert match is not None
    assert match.service_id == _ROR_ID
    assert (match.name, match.country_code, match.funder_doi) == (None, None, None)


def test_lookup_of_a_malformed_record_is_an_answered_miss():
    client = _ror_client(lambda request: httpx.Response(200, json=_chosen("not-an-object")))
    assert asyncio.run(client.lookup("University of Glasgow")) == (None, None)


def test_an_unexpected_lookup_error_is_a_failed_lookup_not_an_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("transport bug")

    client = _ror_client(handler)
    assert asyncio.run(client.lookup("University of Glasgow")) == (None, "RuntimeError")


def test_a_non_ascii_retry_after_still_backs_off():
    # isdigit() accepts a superscript two, which float() rejects.
    client = _ror_client(lambda request: httpx.Response(429, headers=[(b"Retry-After", b"\xb2")]))
    assert asyncio.run(client.lookup("University of Glasgow")) == (None, "rate limited")
    assert client.blocked


def test_ror_enricher_stays_complete_on_a_malformed_answer(monkeypatch):
    """The AttributeError escaped ``RorEnricher.enrich`` and the enrich stage's
    generic failure path marked the paper's enrichment partial."""
    from bibr.clients import ror
    from bibr.pipeline.enricher import EnrichmentStatus, RorEnricher

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["affiliation"] == "Uni Two":
            return httpx.Response(200, json=_chosen({"id": _ROR_ID}))
        return httpx.Response(200, json=_chosen("not-an-object"))

    client = _ror_client(handler)
    monkeypatch.setattr(ror, "get_client", lambda settings: client)
    metadata = PaperMetadata(
        doi="",
        title="A paper",
        authors=[PaperAuthor(author_id=1, given="A", family="B", affiliation="Uni One; Uni Two")],
    )
    fs = SimpleNamespace(paper=SimpleNamespace(metadata=metadata))

    outcome = asyncio.run(RorEnricher(settings=GlobalSettings()).enrich(fs))

    assert outcome.status is EnrichmentStatus.COMPLETE
    assert outcome.warnings == ()
    assert set(metadata.affiliation_match) == {"Uni Two"}
