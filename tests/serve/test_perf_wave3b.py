"""Regression and exact-equivalence tests for Wave 3B performance optimizations."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import time
import zlib

import httpx
import pytest

from bibr.batch.remote import MAX_POLL_INTERVAL, RemoteExecutor, RemoteOptions
from bibr.clients import llm as llm_mod
from bibr.clients import nuextract as nuextract_mod
from bibr.clients.crossref import _NOT_FOUND_UNTIL, CrossrefClient
from bibr.clients.nuextract_schema import native_contract_for_model
from bibr.clients.response_cache import RedisResponseCache
from bibr.config import Settings
from bibr.enrich import references as enrich_refs
from bibr.local.ocr_transport import BaseHttpOcrClient
from bibr.models import PaperReference
from bibr.schemas import PaperReferenceList, TitleKeywordsLLM


def test_llm_cached_schema_json_and_sha256_exact_and_mock_safe():
    for model in (TitleKeywordsLLM, PaperReferenceList):
        expected_json = json.dumps(model.model_json_schema(), sort_keys=True)
        expected_sha = llm_mod._sha256_text(
            json.dumps(model.model_json_schema(), sort_keys=True, ensure_ascii=False)
        )
        assert llm_mod._cached_schema_json(model) == expected_json
        assert llm_mod._cached_schema_sha256(model) == expected_sha
        # Repeated calls hit the LRU cache
        assert llm_mod._cached_schema_json(model) == expected_json
        assert llm_mod._cached_schema_sha256(model) == expected_sha

    # Non-type / custom mock object fallback works without failing lru_cache hashability.
    class _UnhashableSchemaMock:
        __hash__ = None  # type: ignore[assignment]

        def model_json_schema(self):
            return {"type": "object", "properties": {"a": {"type": "string"}}}

    mock = _UnhashableSchemaMock()
    assert llm_mod._cached_schema_json(mock) == json.dumps(mock.model_json_schema(), sort_keys=True)
    assert llm_mod._cached_schema_sha256(mock) == llm_mod._sha256_text(
        json.dumps(mock.model_json_schema(), sort_keys=True, ensure_ascii=False)
    )
    assert llm_mod._cached_schema_json(object()) == ""


def test_nuextract_readonly_contract_reuses_cache():
    c1 = nuextract_mod._readonly_native_contract(TitleKeywordsLLM)
    c2 = nuextract_mod._readonly_native_contract(TitleKeywordsLLM)
    assert c1 is c2

    # Public API still returns an isolated copy so caller mutations do not pollute cache.
    public = native_contract_for_model(TitleKeywordsLLM)
    assert public is not c1
    public.template["__mutated__"] = "x"
    assert "__mutated__" not in nuextract_mod._readonly_native_contract(TitleKeywordsLLM).template


async def test_crossref_prefetch_mget_and_pipeline_and_expired_404():
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    fake_r = fakeredis.FakeRedis(decode_responses=True)
    ops = {"mget": 0, "set": 0, "pipe_exec": 0}
    real_mget = fake_r.mget
    real_set = fake_r.set
    real_pipe = fake_r.pipeline

    async def spy_mget(*args, **kwargs):
        ops["mget"] += 1
        return await real_mget(*args, **kwargs)

    async def spy_set(*args, **kwargs):
        ops["set"] += 1
        return await real_set(*args, **kwargs)

    def spy_pipe(*args, **kwargs):
        pipe = real_pipe(*args, **kwargs)
        orig_exec = pipe.execute

        async def wrapped_exec(*ea, **ekw):
            ops["pipe_exec"] += 1
            return await orig_exec(*ea, **ekw)

        pipe.execute = wrapped_exec
        return pipe

    fake_r.mget = spy_mget
    fake_r.set = spy_set
    fake_r.pipeline = spy_pipe

    rcache = RedisResponseCache.__new__(RedisResponseCache)
    rcache._redis = fake_r
    rcache._ttl = 3600
    rcache._prefix = "crossref:cache:"
    rcache.hits = 0
    rcache.misses = 0
    rcache._redis_url = "redis://fake"
    rcache._loop_clients = {asyncio.get_running_loop(): fake_r}
    rcache._loop_clients_lock = __import__("threading").Lock()
    rcache._redis_claimed = True

    await real_set(
        "crossref:cache:works:10.1000/warm",
        json.dumps({"message": {"DOI": "10.1000/warm", "title": ["Warm"]}}),
    )
    # Expired 404 sentinel must be ignored and re-fetched.
    await real_set(
        "crossref:cache:works:10.1000/expired404",
        json.dumps({_NOT_FOUND_UNTIL: time.time() - 10}),
    )

    cr_settings = Settings.model_copy(deep=True)
    cr_settings.crossref.redis_cache = True
    cr = CrossrefClient(mailto="test@example.com", settings=cr_settings)
    cr._response_cache_backend = rcache
    cr._response_cache_init = True
    requested: list[str] = []

    async def fake_request(path, params=None):
        filt = (params or {}).get("filter", "")
        dois = [part[4:] for part in filt.split(",") if part.startswith("doi:")]
        requested.extend(dois)
        return {"message": {"items": [{"DOI": d, "title": [f"Fetched {d}"]} for d in dois]}}

    cr._request = fake_request
    seeded = await cr.prefetch_works_by_doi(["10.1000/warm", "10.1000/cold", "10.1000/expired404"])
    assert seeded == 2
    assert sorted(requested) == ["10.1000/cold", "10.1000/expired404"]
    assert ops == {"mget": 1, "set": 0, "pipe_exec": 1}
    # L1 was hydrated for the warm DOI too.
    assert cr._cache_peek("works:10.1000/warm")["message"]["title"] == ["Warm"]
    assert cr._cache_peek("works:10.1000/cold")["message"]["title"] == ["Fetched 10.1000/cold"]


async def test_prefetch_enrichment_overlaps_resolver_and_crossref():
    refs = [
        PaperReference(
            bib_id=1,
            title="Alpha",
            authors="Smith, J.",
            year=2020,
            doi="10.1000/a",
            container="J. Test",
            volume=None,
            first_page=None,
        ),
        PaperReference(
            bib_id=2,
            title="Beta Paper Title",
            authors="Jones, A.",
            year=2021,
            container="J. Test",
            volume=None,
            first_page=None,
        ),
    ]
    started = []
    both_running = asyncio.Event()

    class _Resolver:
        async def healthy(self):
            return True

        async def search_many(self, queries, **kw):
            started.append("resolver")
            if len(started) == 2:
                both_running.set()
            await asyncio.wait_for(both_running.wait(), timeout=1.0)
            return [[] for _ in queries]

        async def aclose(self):
            pass

    class _Crossref:
        async def prefetch_works_by_doi(self, dois):
            started.append("crossref")
            if len(started) == 2:
                both_running.set()
            await asyncio.wait_for(both_running.wait(), timeout=1.0)
            return len(dois)

        async def close(self):
            pass

    s = Settings.model_copy(deep=True)
    s.resolver.authoritative = False
    s.crossref.bulk_doi_lookup = True
    pf = await enrich_refs.prefetch_enrichment(
        refs, settings=s, crossref_client=_Crossref(), resolver_client=_Resolver()
    )
    await pf.aclose()
    assert set(started) == {"resolver", "crossref"}


def test_best_title_candidate_preserves_initials_conflict_and_scoring():
    title = "Deep Learning for Biomedical Citation Extraction"
    conflict = enrich_refs._TitleCandidate(
        title=title,
        year=2023,
        work_type="journal-article",
        doi="10.1000/conflict",
        authors=(enrich_refs.CrossrefAuthor(given="Alice", family="Smith"),),
        container="J. One",
        volume="1",
        first_page="1",
    )
    match = enrich_refs._TitleCandidate(
        title=title,
        year=2023,
        work_type="journal-article",
        doi="10.1000/match",
        authors=(enrich_refs.CrossrefAuthor(given="John", family="Smith"),),
        container="J. Two",
        volume="2",
        first_page="2",
    )
    best_idx, score = enrich_refs._best_title_candidate(
        title, [conflict, match], ref_year=2023, ref_authors="J. Smith"
    )
    assert best_idx == 1
    assert score > 0.9


async def test_remote_poll_progressive_ramp_and_ceiling():
    now = [0.0]
    sleeps: list[float] = []

    async def fake_sleep(dt: float) -> None:
        sleeps.append(dt)
        now[0] += dt

    def handler(_req: httpx.Request) -> httpx.Response:
        if now[0] >= 55.0:
            return httpx.Response(200, json={"job_id": "j1", "status": "succeeded"})
        return httpx.Response(200, json={"job_id": "j1", "status": "running"})

    ex = RemoteExecutor(
        RemoteOptions(serve_url="http://serve", poll_interval=4.0, poll_timeout=120.0),
        sleep=fake_sleep,
        clock=lambda: now[0],
    )
    async with httpx.AsyncClient(
        base_url="http://serve", transport=httpx.MockTransport(handler)
    ) as client:
        body = await ex._poll(client, "j1")

    assert body["status"] == "succeeded"
    assert sleeps[:5] == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert max(sleeps) == MAX_POLL_INTERVAL


async def test_redis_job_store_zero_recompress_gzip_and_legacy_fallback():
    pytest.importorskip("fastapi")
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    pytest.importorskip("lupa")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.jobs import _RESULT_GZIP_MIN_BYTES, encode_result, register_job_routes
    from bibr.serve.jobs_redis import RedisJobStore

    fake_r = fakeredis.FakeRedis(decode_responses=False)
    store = RedisJobStore("redis://fake", client_factory=lambda _: fake_r)
    try:
        payload = {"rows": [{"i": i, "t": f"text {i} " * 5} for i in range(2000)]}
        encoded = encode_result(payload)
        assert len(encoded) >= _RESULT_GZIP_MIN_BYTES

        job = await store.create(filename="big.pdf")
        await store.set_succeeded(job.job_id, payload)

        # Raw Redis storage remains valid RFC-1950 zlib.
        raw = await fake_r.get(store.result_key(job.job_id))
        assert zlib.decompress(raw) == encoded

        # Zero-recompress gzip path produces valid RFC-1952 gzip matching exact bytes.
        gz_job = await store.get_with_gzip(job.job_id, min_bytes=_RESULT_GZIP_MIN_BYTES)
        assert gz_job is not None
        assert gz_job.result_gzip is not None
        assert gz_job.result_json is None
        assert gzip.decompress(gz_job.result_gzip) == encoded

        # Direct helper works too.
        assert gzip.decompress(await store.get_result_gzip(job.job_id)) == encoded

        # Legacy record missing result_crc32 falls back transparently.
        await fake_r.hdel(store.job_key(job.job_id), b"result_crc32")
        legacy_job = await store.get_with_gzip(job.job_id, min_bytes=_RESULT_GZIP_MIN_BYTES)
        assert legacy_job is not None
        assert legacy_job.result_gzip is None
        assert legacy_job.result_json == encoded
        assert gzip.decompress(await store.get_result_gzip(job.job_id)) == encoded

        # End-to-end HTTP route test with Accept-Encoding: gzip (with crc32 restored).
        job2 = await store.create(filename="big2.pdf")
        await store.set_succeeded(job2.job_id, payload)
        app = FastAPI()
        register_job_routes(app, store=store, upload_store=object(), tracker=object())
        with TestClient(app) as client:
            resp = client.get(
                f"/papers/jobs/{job2.job_id}/result",
                headers={"Accept-Encoding": "gzip"},
            )
            assert resp.status_code == 200
            assert resp.headers.get("content-encoding") == "gzip"
            assert resp.content == encoded
    finally:
        await store.close()


async def test_memory_job_store_get_o1_ttl_and_dynamic_limit_change(monkeypatch):
    pytest.importorskip("fastapi")
    from bibr.serve.jobs import MemoryJobStore

    now = [100.0]
    store = MemoryJobStore(clock=lambda: now[0], wall_clock=lambda: now[0])
    monkeypatch.setattr(Settings.jobs, "ttl_seconds", 60)
    monkeypatch.setattr(Settings.jobs, "max_retained", 10)
    monkeypatch.setattr(Settings.jobs, "max_retained_bytes", 0)

    ids = []
    for i in range(4):
        j = await store.create(filename=f"{i}.pdf")
        await store.set_succeeded(j.job_id, {"i": i})
        ids.append(j.job_id)
        now[0] += 1.0

    # Steady-state get works without calling _purge_locked.
    assert (await store.get(ids[0])).result == {"i": 0}

    # Dynamic reduction in max_retained triggers purge on get().
    monkeypatch.setattr(Settings.jobs, "max_retained", 2)
    assert await store.get(ids[0]) is None
    assert (await store.get(ids[-1])).result == {"i": 3}

    # Single-job TTL expiry on get().
    now[0] += 100.0
    assert await store.get(ids[-1]) is None


async def test_upload_store_throttles_sweep_and_hashes_large_uploads(monkeypatch):
    pytest.importorskip("fastapi")
    from starlette.datastructures import UploadFile

    from bibr.serve.ingress import UploadStore

    store = UploadStore.create(
        max_size=4 * 1024 * 1024,
        spool_memory_bytes=64 * 1024,
        stale_after_seconds=3600,
    )
    sweeps = 0
    real_sweep = store.sweep_stale

    def counting_sweep(*args, **kwargs):
        nonlocal sweeps
        sweeps += 1
        return real_sweep(*args, **kwargs)

    monkeypatch.setattr(store, "sweep_stale", counting_sweep)
    try:
        payload = b"ABCD1234" * 80_000  # 640 KiB (> 256 KiB chunk size)
        s1 = await store.persist(UploadFile(file=io.BytesIO(payload), filename="a.pdf"))
        s2 = await store.persist(UploadFile(file=io.BytesIO(payload), filename="b.pdf"))
        assert sweeps == 1
        assert s1.sha256_hex == hashlib.sha256(payload).hexdigest()
        assert s2.sha256_hex == s1.sha256_hex
        assert (store.root / s1.upload_id).read_bytes() == payload
    finally:
        await store.close()

    # stale_after_seconds=0 still sweeps every call.
    store0 = UploadStore.create(max_size=1024, spool_memory_bytes=1024, stale_after_seconds=0)
    sweeps0 = 0
    real_sweep0 = store0.sweep_stale

    def counting_sweep0(*args, **kwargs):
        nonlocal sweeps0
        sweeps0 += 1
        return real_sweep0(*args, **kwargs)

    monkeypatch.setattr(store0, "sweep_stale", counting_sweep0)
    try:
        await store0.persist(UploadFile(file=io.BytesIO(b"x"), filename="a.pdf"))
        await store0.persist(UploadFile(file=io.BytesIO(b"y"), filename="b.pdf"))
        assert sweeps0 == 2
    finally:
        await store0.close()


async def test_ocr_http_client_pool_scales_with_max_concurrent():
    class _DummyOcrClient(BaseHttpOcrClient):
        name = "dummy"
        _DEFAULT_MODEL = "glm-ocr"
        _SERVICE_LABEL = "dummy"

    s_small = Settings.model_copy(deep=True)
    s_small.ocr.max_concurrent_regions = 4
    c_small = _DummyOcrClient(base_url="http://localhost:8000", settings=s_small, profile="glm")
    pool_small = c_small._client._transport._pool
    assert pool_small._max_connections == 20
    assert pool_small._max_keepalive_connections == 10
    await c_small._client.aclose()

    s_large = Settings.model_copy(deep=True)
    s_large.ocr.max_concurrent_regions = 32
    c_large = _DummyOcrClient(base_url="http://localhost:8000", settings=s_large, profile="glm")
    pool_large = c_large._client._transport._pool
    assert pool_large._max_connections == 64
    assert pool_large._max_keepalive_connections == 32
    await c_large._client.aclose()
