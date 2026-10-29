"""ROR matching of affiliation strings and funder names."""

from __future__ import annotations

import asyncio
import logging
import time

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


def test_default_budget_is_half_rors_per_address_limit():
    # ROR allows 2000 requests per 5 minutes per IP address and does not
    # enforce its anonymous limit of 50; two processes at 1000 stay under 2000.
    assert RorClient(settings=_settings())._limiter.max_requests == 1000
    assert RorClient(settings=_settings(client_id="x"))._limiter.max_requests == 2000
    assert RorClient(settings=_settings(requests_per_5min=7))._limiter.max_requests == 7


def _answering(calls: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["affiliation"])
        return httpx.Response(200, json={"items": [_item("0uni00011", chosen=True)]})

    return handler


def test_a_lookup_does_not_wait_for_a_slot_past_its_deadline(monkeypatch):
    # A 4 s window instead of 300 s: if the deadline check stops working, the
    # lookup waits about 4 s for the slot and fails, rather than hanging.
    monkeypatch.setattr("bibr.clients.ror._WINDOW_SECONDS", 4.0)
    calls: list[str] = []
    client = _client(_answering(calls), requests_per_5min=1)

    async def run():
        first = await client.lookup("First University", deadline=time.monotonic() + 30)
        started = time.monotonic()
        # The next slot frees about 4 s after the first request, after this
        # deadline: no wait, no request.
        second = await client.lookup("Second University", deadline=started + 0.5)
        return first, second, time.monotonic() - started

    first, second, waited = asyncio.run(run())
    assert first[0] is not None and first[1] is None
    assert second == (None, "request budget spent")
    assert waited < 2.0
    assert calls == ["First University"]


def test_a_lookup_waits_for_the_oldest_slot_when_it_frees_before_the_deadline(monkeypatch):
    # Two slots in a 10 s window, taken 9.5 s and 1 s ago: the older one frees
    # in 0.5 s, before the 3 s deadline, so the lookup waits for it and is
    # sent. Timing from the newer slot (9 s) or from the whole window (10 s)
    # would turn it away.
    monkeypatch.setattr("bibr.clients.ror._WINDOW_SECONDS", 10.0)
    calls: list[str] = []
    client = _client(_answering(calls), requests_per_5min=2)
    now = time.monotonic()
    client._limiter._timestamps.extend([now - 9.5, now - 1.0])

    result, failure = asyncio.run(client.lookup("Late University", deadline=now + 3.0))
    assert result is not None and failure is None
    assert calls == ["Late University"]


def test_the_429_backoff_reads_rate_limited_even_with_the_budget_spent(monkeypatch):
    """The backoff after a 429 is checked before the request budget, so while
    both hold a lookup reports "rate limited", not "request budget spent"."""
    # A 5 s window caps the backoff at 5 s and frees the only slot after the
    # 2 s deadline, so the deadline check alone would turn the lookup away.
    monkeypatch.setattr("bibr.clients.ror._WINDOW_SECONDS", 5.0)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["affiliation"])
        return httpx.Response(429, headers={"Retry-After": "120"})

    client = _client(handler, requests_per_5min=1)

    async def run():
        deadline = time.monotonic() + 2.0
        first = await client.lookup("First University", deadline=deadline)
        second = await client.lookup("Second University", deadline=deadline)
        return first, second

    first, second = asyncio.run(run())
    assert first == (None, "rate limited")
    assert second == (None, "rate limited")
    assert client.blocked
    assert calls == ["First University"]


class _Clock:
    """The ROR client's ``time``: ``monotonic()`` moves only when a test moves it.

    The limiter keeps the real clock, so these tests stay well inside its window.
    """

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


def _fake_clock(monkeypatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr("bibr.clients.ror.time", clock)
    return clock


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _four_in_flight(status: int, text: str):
    """Four lookups, all sent before ROR answers any of them with *status*."""
    sent = 0

    async def run():
        nonlocal sent
        all_sent = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal sent
            sent += 1
            if sent == 4:
                all_sent.set()
            await asyncio.wait_for(all_sent.wait(), timeout=5)
            return httpx.Response(status, text=text)

        client = _client(handler)
        results = await asyncio.gather(*(client.lookup(f"University {i}") for i in range(4)))
        return client, results

    client, results = asyncio.run(run())
    assert sent == 4
    return client, results


def test_a_429_halves_the_budget_once_per_pause(caplog):
    # Four lookups are in flight when ROR starts answering 429, with no
    # Retry-After as api.ror.org sends it. The budget halves once, not once per
    # answer (1000 / 16 = 62), and one WARNING names the new ceiling.
    with caplog.at_level(logging.WARNING, logger="bibr.clients.ror"):
        client, results = _four_in_flight(429, "Rate Limit Exceeded.")
    assert results == [(None, "rate limited")] * 4
    assert client._limiter.max_requests == 500
    assert _warnings(caplog) == [
        "ROR rate limit reached; skipping ROR lookups for 300s, then at most 500 requests "
        "per 5 minutes"
    ]


def test_403s_in_flight_warn_once(caplog):
    with caplog.at_level(logging.WARNING, logger="bibr.clients.ror"):
        client, results = _four_in_flight(403, "You have been blocked due to bad behavior.")
    assert results == [(None, "blocked (HTTP 403)")] * 4
    assert client._limiter.max_requests == 1000
    assert len(_warnings(caplog)) == 1


@pytest.mark.parametrize(
    ("configured", "budgets"),
    [
        (None, [500, 250, 125, 62, 50, 50]),  # the default, down to the floor of 50
        (60, [50, 50]),
        (7, [7, 7]),  # a configured budget below 50 is its own floor
    ],
)
def test_each_new_pause_halves_the_budget_down_to_its_floor(monkeypatch, configured, budgets):
    clock = _fake_clock(monkeypatch)
    client = _client(lambda request: httpx.Response(429), requests_per_5min=configured)
    seen = []
    for i in range(len(budgets)):
        assert not client.blocked
        assert asyncio.run(client.lookup(f"University {i}")) == (None, "rate limited")
        seen.append(client._limiter.max_requests)
        clock.now += 300.0  # the next 429 arrives just as this pause ends
    assert seen == budgets


def test_each_hour_without_a_429_doubles_the_budget_back(monkeypatch, caplog):
    clock = _fake_clock(monkeypatch)
    start = clock.now
    calls: list[str] = []
    answer = _answering(calls)

    def handler(request: httpx.Request) -> httpx.Response:
        if not calls:
            calls.append(request.url.params["affiliation"])
            return httpx.Response(429)
        if request.url.params["affiliation"] == "Blocked University":
            calls.append(request.url.params["affiliation"])
            return httpx.Response(403)
        return answer(request)

    # An odd budget, so the second doubling would overshoot it.
    client = _client(handler, requests_per_5min=999)

    def budget_at(seconds: float, text: str) -> int:
        clock.now = start + seconds
        asyncio.run(client.lookup(text))  # a new string each time: cached answers skip all this
        return client._limiter.max_requests

    with caplog.at_level(logging.INFO, logger="bibr.clients.ror"):
        assert budget_at(0, "Uni 0") == 499  # the 429
        # A 403 is not a 429: it does not restart the quiet hour.
        assert budget_at(1800, "Blocked University") == 499
        assert budget_at(3599, "Uni 1") == 499
        assert budget_at(3600, "Uni 2") == 998
        assert budget_at(3600, "Uni 3") == 998  # one doubling per quiet hour
        assert budget_at(7199, "Uni 4") == 998
        assert budget_at(7200, "Uni 5") == 999  # never above the configured budget
        assert budget_at(20000, "Uni 6") == 999
    recoveries = [
        r.getMessage()
        for r in caplog.records
        if r.name == "bibr.clients.ror" and r.levelno == logging.INFO
    ]
    assert recoveries == [
        f"No ROR 429 for an hour; ROR budget back up to {n} requests per 5 minutes"
        for n in (998, 999)
    ]
    assert calls == ["Uni 0", "Blocked University"] + [f"Uni {i}" for i in range(1, 7)]


def test_an_idle_process_gains_one_doubling_at_a_time(monkeypatch):
    # The budget recovers on a lookup: hours spent idle count as one quiet
    # hour, and the next doubling needs another hour from then.
    clock = _fake_clock(monkeypatch)
    calls: list[str] = []
    answer = _answering(calls)

    def handler(request: httpx.Request) -> httpx.Response:
        if len(calls) < 2:
            calls.append(request.url.params["affiliation"])
            return httpx.Response(429)
        return answer(request)

    client = _client(handler)

    def budget_after(seconds: float, text: str) -> int:
        clock.now += seconds
        asyncio.run(client.lookup(text))
        return client._limiter.max_requests

    assert budget_after(0, "Uni 0") == 500
    assert budget_after(300, "Uni 1") == 250  # the next 429, as the first pause ends
    assert budget_after(5 * 3600, "Uni 2") == 500
    assert budget_after(0, "Uni 3") == 500
    assert budget_after(3600, "Uni 4") == 1000


def test_a_403_pauses_lookups_without_halving(caplog):
    # ROR answers 403 only for addresses it has blocked: every later request
    # would fail too.
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["affiliation"])
        return httpx.Response(403, text="You have been blocked due to bad behavior.")

    client = _client(handler)

    async def run():
        return [await client.lookup("First University"), await client.lookup("Second University")]

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="bibr.clients.ror"):
        results = asyncio.run(run())
    assert results == [(None, "blocked (HTTP 403)")] * 2
    assert calls == ["First University"]
    assert client.blocked
    assert 300.0 <= client._blocked_until - started < 310.0
    assert client._limiter.max_requests == 1000
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "HTTP 403" in warnings[0]
    assert "You have been blocked due to bad behavior." in warnings[0]


def test_a_429_after_a_403_pause_reads_rate_limited(monkeypatch):
    clock = _fake_clock(monkeypatch)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["affiliation"])
        if len(calls) == 1:
            return httpx.Response(403, text="You have been blocked due to bad behavior.")
        return httpx.Response(429)

    client = _client(handler)
    assert asyncio.run(client.lookup("First University")) == (None, "blocked (HTTP 403)")
    clock.now += 300.0  # the 403 pause ends; ROR now answers 429
    assert asyncio.run(client.lookup("Second University")) == (None, "rate limited")
    # During the 429 pause lookups read "rate limited", not the old 403 reason.
    assert asyncio.run(client.lookup("Third University")) == (None, "rate limited")
    assert client._limiter.max_requests == 500
    assert calls == ["First University", "Second University"]


def test_a_429_in_flight_does_not_shorten_a_403_pause():
    # ROR blocks the address while a second lookup is in flight; that lookup
    # comes back 429 with a short Retry-After after the 403 has opened its
    # pause. The 403 pause keeps its 5 minutes and its reason.
    sent: list[str] = []

    async def run():
        all_sent = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request.url.params["affiliation"])
            if len(sent) == 2:
                all_sent.set()
            await asyncio.wait_for(all_sent.wait(), timeout=5)
            if request.url.params["affiliation"] == "First University":
                return httpx.Response(403, text="You have been blocked due to bad behavior.")

            async def until_paused() -> None:
                while not client.blocked:
                    await asyncio.sleep(0)

            await asyncio.wait_for(until_paused(), timeout=5)
            return httpx.Response(429, headers={"Retry-After": "10"})

        client = _client(handler)
        started = time.monotonic()
        results = await asyncio.gather(
            client.lookup("First University"), client.lookup("Second University")
        )
        later = await client.lookup("Third University")
        return client, started, results, later

    client, started, results, later = asyncio.run(run())
    assert results == [(None, "blocked (HTTP 403)"), (None, "rate limited")]
    assert later == (None, "blocked (HTTP 403)")
    assert sent == ["First University", "Second University"]
    assert 300.0 <= client._blocked_until - started < 310.0
    assert client._limiter.max_requests == 1000


def test_a_cached_answer_is_returned_after_the_budget_is_spent(monkeypatch):
    # A 4 s window instead of 300 s, so a lookup that waits for a slot fails
    # in about 4 s rather than hanging.
    monkeypatch.setattr("bibr.clients.ror._WINDOW_SECONDS", 4.0)
    calls: list[str] = []
    client = _client(_answering(calls), requests_per_5min=1)

    async def run():
        started = time.monotonic()
        deadline = started + 1.0
        first = await client.lookup("First University", deadline=deadline)
        # Same cache key; the only slot is taken until after the deadline.
        again = await client.lookup("first  university", deadline=deadline)
        return first, again, time.monotonic() - started

    first, again, elapsed = asyncio.run(run())
    assert first[0] is not None and first[1] is None
    assert again == first
    assert calls == ["First University"]
    assert elapsed < 2.0  # the free slot was used at once


def test_a_lookup_with_a_free_slot_ignores_its_deadline():
    calls: list[str] = []
    client = _client(_answering(calls), requests_per_5min=5)

    async def run():
        deadline = time.monotonic() + 30
        return [await client.lookup(name, deadline=deadline) for name in ("Uni A", "Uni B")]

    assert [failure for _, failure in asyncio.run(run())] == [None, None]
    assert calls == ["Uni A", "Uni B"]


def test_an_exhausted_budget_ends_a_papers_matching_at_once():
    """With the budget spent, the paper's other strings fail ("request budget spent")
    straight away instead of holding the paper for its whole ROR budget."""
    calls: list[str] = []
    client = _client(_answering(calls), requests_per_5min=1)

    async def run():
        started = time.monotonic()
        report = await enrich_organizations(_metadata(), client, timeout=60)
        return report, time.monotonic() - started

    report, elapsed = asyncio.run(run())
    assert len(calls) == 1
    assert report.attempted == 4 and report.matched == 1
    assert not report.timed_out
    assert report.failed == 3 and report.failure_reasons == ("request budget spent",)
    assert elapsed < 5


def test_the_deadline_is_the_papers_ror_time_budget(monkeypatch):
    """The deadline is the start plus the paper's timeout (``ROR_ENRICH_TIMEOUT``),
    not later: a slot that frees after the timeout is not waited for."""
    # The next slot frees 2.5 s after the first request: after the paper's
    # 2 s budget, before 1.5 times that.
    monkeypatch.setattr("bibr.clients.ror._WINDOW_SECONDS", 2.5)
    calls: list[str] = []
    client = _client(_answering(calls), requests_per_5min=1)

    report = asyncio.run(enrich_organizations(_metadata(), client, timeout=2.0))
    assert len(calls) == 1
    assert not report.timed_out
    assert report.failed == 3 and report.failure_reasons == ("request budget spent",)


def test_a_slot_that_frees_within_the_papers_ror_time_budget_is_waited_for(monkeypatch):
    """The deadline is not earlier than the paper's timeout either: a slot that
    frees before the timeout is waited for and the string is matched."""
    # The next slot frees 1.2 s after the first request: before the paper's
    # 2 s budget, after half of it.
    monkeypatch.setattr("bibr.clients.ror._WINDOW_SECONDS", 1.2)
    calls: list[str] = []
    client = _client(_answering(calls), requests_per_5min=1)
    metadata = PaperMetadata(
        doi="",
        title="A paper",
        authors=[PaperAuthor(author_id=1, given="A", family="B", affiliation="Uni One")],
        funding=[FundingEntry(funder="Wellcome Trust", award_ids=[])],
    )

    report = asyncio.run(enrich_organizations(metadata, client, timeout=2.0))
    assert calls == ["Uni One", "Wellcome Trust"]
    assert not report.timed_out
    assert report.attempted == 2 and report.matched == 2 and report.failed == 0


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
