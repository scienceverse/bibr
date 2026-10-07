"""Audit regressions: enrichment warning redaction."""

from __future__ import annotations

import asyncio
from unittest import mock

import httpx

from bibr.clients.crossref import CrossrefClient
from bibr.config import GlobalSettings, ResolverOptions
from bibr.models import PaperReference
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
