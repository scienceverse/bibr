import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI

from bibr.config import Settings
from bibr.serve import enrichment


@pytest.fixture
def app():
    instance = FastAPI()
    enrichment.register_enrichment_route(instance, Settings)
    return instance


async def test_invalid_saved_paper_and_body(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for value in ({"paper": {}}, {}, {"paper": "wrong"}, {"paper": {}, "refresh": True}):
            response = await client.post("/papers/enrich", json=value)
            assert response.status_code == 422
        assert (
            await client.post(
                "/papers/enrich", content="{}", headers={"content-type": "text/plain"}
            )
        ).status_code == 415
        assert (
            await client.post(
                "/papers/enrich", content="{", headers={"content-type": "application/json"}
            )
        ).status_code == 422


async def test_duplicate_keys_and_deep_json_are_rejected(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for body in ('{"paper": {}, "paper": {}}', "[" * 2000 + "]" * 2000):
            response = await client.post(
                "/papers/enrich", content=body, headers={"content-type": "application/json"}
            )
            assert response.status_code == 422


async def test_real_handler_returns_replayable_json_without_extraction(app, monkeypatch):
    from bibr.enrich import references
    from bibr.enrich.references import EnrichmentReport
    from bibr.models import ExternalMatch, MatchSource

    async def lookup(refs, **_kwargs):
        refs[0].match[MatchSource.CROSSREF] = ExternalMatch(id="10.1234/example")
        return EnrichmentReport(attempted=1, matched=1)

    monkeypatch.setattr(references, "enrich_references", lookup)
    app.state.export_enricher._client = AsyncMock()
    paper = {
        "paper_id": "original",
        "info": {
            "title": "Printed title",
            "doi": None,
            "keywords": [],
            "file_hash": "hash",
            "file_name": "paper.pdf",
            "input_format": "pdf",
            "schema_version": "10.7",
            "bibr_version": "0.1.0",
        },
        "bib": [{"bib_id": 3, "title": "Printed reference"}],
        **{
            key: [] for key in ("text", "author", "section", "url", "xref", "figure", "table", "eq")
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/papers/enrich", json={"paper": paper})
        assert response.status_code == 200
        result = response.json()
        assert result["paper"]["bib"] == paper["bib"]
        assert result["paper"]["info"] == paper["info"]
        assert result["paper"]["bib_match"][0]["bib_id"] == 3
        assert result["status"] == "complete"
        assert len(result["enrichment"]["core_sha256"]) == 64
        assert result["enrichment"]["completeness"] == "complete"


async def test_chunked_body_and_reference_limits(app, monkeypatch):
    monkeypatch.setattr(enrichment, "MAX_ENRICH_BODY_BYTES", 64)

    async def oversized():
        yield b" " * 32
        yield b" " * 33
        raise AssertionError("Must stop reading once the limit is exceeded")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/papers/enrich", content=oversized(), headers={"content-type": "application/json"}
        )
        assert response.status_code == 413
        monkeypatch.setattr(enrichment, "MAX_ENRICH_REFERENCES", 1)
        assert (
            await client.post("/papers/enrich", json={"paper": {"bib": [{}, {}]}})
        ).status_code == 413


async def test_admission_before_reading_and_release_after_cancellation(monkeypatch):
    monkeypatch.setattr(enrichment, "MAX_ACTIVE_ENRICHMENTS", 1)
    app = FastAPI()
    worker = enrichment.register_enrichment_route(app, Settings)
    entered = asyncio.Event()

    async def held(_paper):
        entered.set()
        await asyncio.Event().wait()

    worker.enrich = AsyncMock(side_effect=held)

    async def unread():
        raise AssertionError("Full endpoint must reject before reading body")
        yield b""  # pragma: no cover

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = asyncio.create_task(client.post("/papers/enrich", json={"paper": {}}))
        await entered.wait()
        response = await client.post("/papers/enrich", content=unread())
        assert response.status_code == 429
        assert response.headers["retry-after"] == "1"
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        worker.enrich = AsyncMock(return_value={"status": "no_work"})
        assert (await client.post("/papers/enrich", json={"paper": {}})).status_code == 200


async def test_upload_timeout_releases_slot(app, monkeypatch):
    monkeypatch.setattr(enrichment, "ENRICH_BODY_TIMEOUT_SECONDS", 0.02)

    async def stalled():
        await asyncio.Event().wait()
        yield b""  # pragma: no cover

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/papers/enrich", content=stalled(), headers={"content-type": "application/json"}
        )
        assert response.status_code == 408
        assert (await client.post("/papers/enrich", json={})).status_code == 422


async def test_registered_route_inherits_auth_and_never_dispatches_extraction(monkeypatch):
    pytest.importorskip("litserve")
    from bibr.serve.app import build_server

    monkeypatch.setattr(Settings.auth, "api_key", "test-backfill-key")
    server = build_server()
    worker = server.app.state.export_enricher
    worker.enrich = AsyncMock(return_value={"status": "no_work"})
    tracker = server.app.state.inference_tracker
    tracker.submit = AsyncMock(side_effect=AssertionError("Backfill must not dispatch extraction"))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            assert (await client.post("/papers/enrich", json={"paper": {}})).status_code == 401
            worker.enrich.assert_not_awaited()
            response = await client.post(
                "/papers/enrich",
                json={"paper": {}},
                headers={"Authorization": "Bearer test-backfill-key"},
            )
            assert response.status_code == 200
            tracker.submit.assert_not_awaited()
    finally:
        await tracker.close()
        await server.app.state.upload_store.close()
        await worker.close()
