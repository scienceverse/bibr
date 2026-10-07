"""Audit regressions: enrichment warning redaction, ROR robustness, secret patterns."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest

from bibr.clients.crossref import CrossrefClient
from bibr.clients.ror import RorClient, chosen_organization
from bibr.config import GlobalSettings, ResolverOptions
from bibr.models import PaperAuthor, PaperMetadata, PaperReference
from bibr.processing_warnings import WarningCode
from bibr.utils.redact import redact_url_secrets, redact_urls, scrub_secrets

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


def test_a_huge_retry_after_backs_off_for_one_window_not_forever():
    # float("9" * 400) is inf: the back-off is process-wide, so ROR matching
    # would have stayed off until the process restarted.
    from bibr.clients.ror import _WINDOW_SECONDS

    client = _ror_client(lambda request: httpx.Response(429, headers={"Retry-After": "9" * 400}))
    assert asyncio.run(client.lookup("University of Glasgow")) == (None, "rate limited")
    assert client.blocked
    assert client._blocked_until <= time.monotonic() + _WINDOW_SECONDS


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


# --- secret shapes the scrubbers missed (report 2.6) --------------------------

_GHP = "ghp_" + "a1B2" * 9
_HF = "hf_" + "QwErTy12" * 4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A ``:`` or ``@`` in the password; a token as the user name.
        ("redis://:pa:ss@redis:6379/0", "redis://***@redis:6379/0"),
        ("https://user:p@ss@host.internal/v1 down", "https://***@host.internal/v1 down"),
        (f"git clone https://{_GHP}@github.com/o/r", "git clone https://***@github.com/o/r"),
        # Vendor token shapes.
        (f"token {_HF} rejected", "token *** rejected"),
        ("as AKIAIOSFODNN7EXAMPLE denied", "as *** denied"),
        (f"{_GHP} gho_{'x' * 36} ghs_{'y' * 36}", "*** *** ***"),
        (f"pat github_pat_11ABCDEFG0_{'z' * 40} bad", "pat *** bad"),
        # JSON / dict entries and API-key headers.
        ('body {"api_key": "abc123", "model": "m"}', 'body {"api_key": "***", "model": "m"}'),
        ("{'token': 'tok-1', 'n': 'v'}", "{'token': '***', 'n': 'v'}"),
        ('{"client_secret": "a\\"b", "x": 1}', '{"client_secret": "***", "x": 1}'),
        ("sent x-api-key: k-123abc, accept: json", "sent x-api-key: ***, accept: json"),
        ("x-goog-api-key: AIzaShort api-key: azure1", "x-goog-api-key: *** api-key: ***"),
        # Query parameters.
        (
            "GET https://idp.example/token?client_id=a&client_secret=s3cr3t&x=1",
            "GET https://idp.example/token?client_id=a&client_secret=***&x=1",
        ),
        (
            "GET https://acct.blob.core.windows.net/c/b?sv=2020&sig=ab%2Bcd&se=1",
            "GET https://acct.blob.core.windows.net/c/b?sv=2020&sig=***&se=1",
        ),
    ],
)
def test_scrub_secrets_masks_the_missed_shapes(text, expected):
    assert scrub_secrets(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("redis://:pa:ss@redis:6379/0", "redis://:***@redis:6379/0"),
        ("https://user:p@ss@host/v1", "https://user:***@host/v1"),
        ("redis://default:a:b@c@redis:6379/0", "redis://default:***@redis:6379/0"),
        (f"https://{_GHP}@github.com/o/r", "https://***@github.com/o/r"),
        ("https://idp/x?client_secret=abc", "https://idp/x?client_secret=***"),
        # The user name stays readable; no password, nothing to mask.
        ("https://user@host:8080/path", "https://user@host:8080/path"),
    ],
)
def test_redact_url_secrets_handles_the_missed_shapes(text, expected):
    assert redact_url_secrets(text) == expected


# A password may hold any character but ``/`` and whitespace: httpx accepts a
# raw ``'`` and quotes it verbatim; urllib and redis-py accept ``"`` and ``<>``;
# a stray ``#`` or ``?`` breaks the URL but is still the secret.
_ODD_PASSWORDS = ("pa'ss", 'Xy"9k', "pa#ss", "pa?ss", "pa<ss>", "p'a@ss")


@pytest.mark.parametrize("password", _ODD_PASSWORDS)
def test_scrubbers_mask_a_password_with_quotes_or_url_delimiters(password):
    url = f"redis://user:{password}@redis.internal:6379/0"
    assert scrub_secrets(f"cannot connect to {url}") == (
        "cannot connect to redis://***@redis.internal:6379/0"
    )
    assert redact_url_secrets(url) == "redis://user:***@redis.internal:6379/0"
    assert redact_url_secrets(f"redis://:{password}@redis:6379/0") == "redis://:***@redis:6379/0"


def test_scrub_secrets_masks_a_quoted_password_in_httpx_status_text():
    request = httpx.Request("GET", "https://svc:pa'ss@resolver.internal.corp:8080/search")
    response = httpx.Response(503, request=request)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        response.raise_for_status()
    first_line = str(caught.value).splitlines()[0]
    assert "pa'ss@" in first_line  # httpx keeps the quote unencoded
    assert scrub_secrets(first_line) == (
        "Server error '503 Service Unavailable' for url "
        "'https://***@resolver.internal.corp:8080/search'"
    )


def test_describe_error_drops_a_url_whose_password_holds_a_quote():
    from bibr.enrich.references import ResolutionStats, _record_terminal_failure
    from bibr.utils.redact import describe_error

    exc = httpx.ConnectError(
        "connect to https://svc:pa'ss@resolver.internal.corp:8080/search failed"
    )
    assert describe_error(exc) == "ConnectError: connect to <url> failed"
    stats = ResolutionStats()
    _record_terminal_failure(stats, _ref(), "resolver", exc)
    assert [d.message for d in stats.failure_details] == [
        "bib_id=1 resolver failed: ConnectError: connect to <url> failed"
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"ocr_api_key": "k-1", "n": 1}', '{"ocr_api_key": "***", "n": 1}'),
        ("{'id_token': 'eyJ.x.y'}", "{'id_token': '***'}"),
        (
            '{"private_key": "-----BEGIN PRIVATE KEY-----\\nMIIE\\n", "x": 1}',
            '{"private_key": "***", "x": 1}',
        ),
        ('"X-Goog-Api-Key": "abc"', '"X-Goog-Api-Key": "***"'),
    ],
)
def test_scrub_secrets_masks_a_quoted_entry_by_its_key_ending(text, expected):
    assert scrub_secrets(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Contact john.doe@uni.edu or mailto:jane@uni.edu; doi 10.1000/xyz123",
        "https://doi.org/10.1002/(SICI)1097-4636(199606)31:2<213::AID-JBM9>3.0.CO;2-K",
        "GET https://api.crossref.org/works?mailto=you@example.com",
        "GET https://api.crossref.org?mailto=you@example.com",
        "connecting to http://gpu-box:2010/search/batch",
        "the token and the key: tokenizers split words; signature=1 in prose",
        "hf_hub_download(repo) failed; ghp_short is not a token",
        'def f(api_key: str | None) -> None:\n    if verdict == "password":',
        "{'title': 'A token-free paper', 'key': 'value'}",
        '{"max_tokens": 512, "tokenizer": "bpe", "sort_key": "year"}',
        "GET 'https://api.openalex.org/works?mailto=me@uni.edu' -> 'me@uni.edu'",
    ],
)
def test_scrubbers_leave_ordinary_text_alone(text):
    assert scrub_secrets(text) == text
    assert redact_url_secrets(text) == text


def test_describe_error_masks_a_quoted_key_in_an_error_body():
    from bibr.utils.redact import describe_error

    exc = RuntimeError('upstream 401: {"error": "bad", "api_key": "sk-short"}')
    assert describe_error(exc) == 'RuntimeError: upstream 401: {"error": "bad", "api_key": "***"}'


def test_new_patterns_stay_linear_on_pathological_input():
    """Every log record passes through these; unclosed quotes, user-info
    without a host and repeated headers must not rescan the rest of the line."""
    texts = [
        '"token": \'' * 6_000,
        '"token": "\\' * 6_000,
        "x-api-key:" * 6_000,
        "https://" + "a:" * 30_000,
        "://" + "@" * 60_000,
        "://:" + "x'" * 30_000,
        "://a:" * 15_000,
        '"' + "a_" * 30_000,
    ]
    started = time.perf_counter()
    for text in texts:
        scrub_secrets(text)
        redact_url_secrets(text)
        redact_urls(text)
    assert time.perf_counter() - started < 2.0
